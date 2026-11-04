"""对完整数据快照分阶段对照回放搜索，逐任务留痕并限制实验进程时间。

snapshot 只读取原数据的状态、夹爪与元信息，使用 200 个完整 episode，不读取视频；
run 启动独立子进程，依次执行 input、first、geometry、sample、full 指定阶段之一。
first 的完整结果经 freeze-first 生成公共首帧 XY 文件，geometry 由该文件生成公共候选。
参照串行结果运行并行阶段时用 --reference-events 限定同一任务集合。
本工具不修改原始数据、原缓存或生产回放逻辑；超时仅终止本实验进程组。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
from benchmark_replay_placement import (
    code_provenance,
    read_snapshot,
    replay_args,
    snapshot_loader,
)


def append_event(path, event):
    """立即追加 UTF-8 任务事件，异常或超时时保留已经完成的任务。"""
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()


def load_events(path):
    """读取已有完整 JSON 行，忽略进程中断时留下的不完整末行。"""
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return events


def task_ids(events):
    """按原有任务序号返回成功完成项，拒绝重复或非连续结果。"""
    completed = {}
    for event in events:
        if event.get("event") == "task_done":
            index = int(event["index"])
            if index in completed:
                raise ValueError(f"任务编号重复：{index}")
            completed[index] = event
    return completed


def create_snapshot(dataset, output, calibration):
    """读取全部原始轨迹及 TCP 标定到独立快照，并记录每段长度与摘要。"""
    import shutil

    import pyarrow.parquet as pq

    from handumi.dataset.episode_solve_cache import array_digest, digest_file
    from handumi.dataset.selection import resolve_dataset_selection
    from handumi.scripts.replay import replay_in_sim as replay

    dataset, output, calibration = (Path(item).expanduser().resolve()
                                    for item in (dataset, output, calibration))
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError("快照和源数据目录必须相互独立")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("快照目录必须为空")
    if not calibration.is_file():
        raise ValueError("机器人 TCP 标定文件不存在")
    rows = []
    for path in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "length"])
        rows.extend(zip(table["episode_index"].to_pylist(), table["length"].to_pylist()))
    rows = sorted((int(index), int(length)) for index, length in rows)
    if [index for index, _ in rows] != list(range(len(rows))) or not rows:
        raise ValueError("episode 索引必须从 0 起连续且非空")
    selection = resolve_dataset_selection(str(dataset))
    args = replay_args(dataset)
    args.repo_id, args.dataset_root = selection.repo_id, selection.root
    output.mkdir(parents=True, exist_ok=True)
    entries, info = [], None
    started = time.monotonic()
    for index, length in rows:
        args.episode = index
        episode_start = time.monotonic()
        states, fps, info, openings = replay.load_episode_states(args)
        if len(states) != length:
            raise ValueError(f"episode {index} 的帧数与元信息不一致")
        arrays = {"states": states}
        if openings is not None:
            arrays["openings"] = openings
        filename = f"episode-{index}.npz"
        np.savez_compressed(output / filename, **arrays)
        entries.append({"index": index, "local_index": index, "length": length,
                        "fps": fps, "file": filename, "states": array_digest(states),
                        "openings": array_digest(openings)})
        append_event(output / "snapshot-events.jsonl", {"event": "episode", "episode": index,
                     "frames": length, "seconds": time.monotonic() - episode_start,
                     "elapsed_seconds": time.monotonic() - started})
    target = output / "robot_tcp.yaml"
    shutil.copy2(calibration, target)
    manifest = {"version": 1, "selection": {"episodes": "all"}, "episodes": entries,
                "info": info, "robot_tcp_calibration": {
                    "file": target.name, "sha256": digest_file(target), "source": str(calibration)}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"episodes": len(entries), "frames": sum(item["length"] for item in entries),
            "elapsed_seconds": time.monotonic() - started}


def prepare_contexts(snapshot, event_path):
    """用生产输入准备逻辑从快照构造全部上下文，记录每段准备耗时。"""
    from handumi.scripts.replay import replay_dataset_placement as placement

    entries, loaded = read_snapshot(snapshot)
    args = replay_args(snapshot)
    reference = placement._reference_calibration(args)
    last_count = 0
    previous = time.monotonic()

    class Progress:
        """将真实输入准备完成事件写入持久日志。"""
        def update(self, stage, current, total, **details):
            """只有当前段完整准备完成后记录一次。"""
            nonlocal last_count, previous
            if stage == "准备输入" and total == len(entries) and current > last_count:
                now = time.monotonic()
                for index in range(last_count, current):
                    append_event(event_path, {"event": "input_done", "episode": index,
                                 "frames": entries[index]["length"], "seconds": now - previous})
                last_count = current
                previous = now

    started = time.monotonic()
    with snapshot_loader(loaded):
        contexts, geometry = placement._episode_contexts(args, reference, Progress())
    return args, reference, contexts, geometry, entries, time.monotonic() - started


def build_tasks(phase, contexts, reference, candidates):
    """按生产顺序构造首帧或候选乘 episode 的确定性任务。"""
    from scipy.spatial.transform import Rotation

    from handumi.robots.utils import pose_inv

    n = len(contexts)
    if phase == "first":
        reference_pose = pose_inv(reference.pose7).astype(float)
        reference_rotation = Rotation.from_quat(reference_pose[3:])
        result = []
        for yaw in range(8):
            pose = reference_pose.copy()
            pose[3:] = (Rotation.from_euler("z", yaw * np.pi / 4) * reference_rotation).as_quat()
            result.extend([("first", index, pose) for index in range(n)])
        return result
    return [(phase, index, candidate) for candidate in candidates for index in range(n)]


def timed_worker(task):
    """在优化版 worker 进程中调用原生产任务并返回实际执行耗时。"""
    from handumi.scripts.replay.replay_placement_pool import _task

    started = time.monotonic()
    result = _task(task)
    return result, time.monotonic() - started


def json_value(value):
    """将 NumPy 数据转换为标准 JSON 值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def serial_tasks(args, reference, contexts, tasks, ids, events, started, deadline):
    """调用生产串行求解，逐项记录结果及墙钟完成时刻。"""
    from handumi.retargeting.base_placement import first_frame_candidates
    from handumi.scripts.replay import replay_dataset_placement as placement

    view = [contexts[0]]
    evaluate = placement._sample_evaluator(args, view, reference) if tasks and tasks[0][0] == "sample" else None
    for index in ids:
        if time.monotonic() >= deadline:
            break
        kind, episode, candidate = tasks[index]
        context = contexts[episode]
        task_start = time.monotonic()
        with open(os.devnull, "w", encoding="utf-8") as discard, contextlib.redirect_stdout(discard):
            if kind == "first":
                result = [item.xy for item in first_frame_candidates(
                    context["runtime"], context["tool"], context["targets"], candidate,
                    context["locked"], max_seeds=1)]
            elif kind == "sample":
                view[0] = context
                result = evaluate(candidate)
            else:
                result = placement._full_episode(
                    args, context, episode, placement._candidate_calibration(reference, candidate.pose))
        now = time.monotonic()
        append_event(events, {"event": "task_done", "index": index, "episode": episode,
                     "candidate": index // len(contexts), "result": json_value(result),
                     "worker_seconds": now - task_start, "elapsed_seconds": now - started})


def parallel_tasks(args, reference, contexts, tasks, ids, events, workers, started, deadline):
    """复用生产 spawn 池，有界提交并按完成时刻写入任务结果。"""
    from handumi.scripts.replay.replay_placement_pool import PlacementPool
    from handumi.scripts.replay.replay_scheduler import _spawn_environment

    source = iter(ids)
    pending = {}
    with PlacementPool(args, contexts, reference, workers=workers) as pool:
        with _spawn_environment():
            while True:
                while len(pending) < workers * 2 and time.monotonic() < deadline:
                    try:
                        index = next(source)
                    except StopIteration:
                        break
                    pending[pool.executor.submit(timed_worker, tasks[index])] = index
                if not pending:
                    break
                if time.monotonic() >= deadline + 30:
                    raise TimeoutError("在途任务超出阶段宽限")
                ready, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in ready:
                    index = pending.pop(future)
                    result, worker_seconds = future.result()
                    append_event(events, {"event": "task_done", "index": index,
                                 "episode": tasks[index][1], "candidate": index // len(contexts),
                                 "result": json_value(result), "worker_seconds": worker_seconds,
                                 "elapsed_seconds": time.monotonic() - started})


def freeze_first(events, output, episode_count):
    """仅完整获得 8×episode 首帧结果时冻结共同输入，禁止截断冒充完整。"""
    done = task_ids(load_events(events))
    total = 8 * episode_count
    if set(done) != set(range(total)):
        raise ValueError(f"首帧任务未完整：{len(done)}/{total}")
    values = [[done[yaw * episode_count + ep]["result"] for ep in range(episode_count)]
              for yaw in range(8)]
    target = Path(output)
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"episode_count": episode_count, "xy": values}, stream, ensure_ascii=False)


def load_candidates(path):
    """从冻结文件恢复原有 Candidate 类型和稳定顺序。"""
    from handumi.retargeting.dataset_placement import Candidate
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Candidate(np.asarray(item["pose7"], dtype=float), int(item["yaw_index"])) for item in rows]


def select_finalists(snapshot, candidates_file, events, output):
    """仅使用全部候选和全部段的完整代表帧评分，重现正式决选。"""
    from handumi.retargeting.dataset_placement import choose_finalists
    from handumi.robots.utils import pose_inv
    from handumi.scripts.replay import replay_dataset_placement as placement

    entries, _loaded = read_snapshot(snapshot)
    candidates = load_candidates(candidates_file)
    done = task_ids(load_events(events))
    count = len(entries)
    if set(done) != set(range(count * len(candidates))):
        raise ValueError("代表帧任务不完整，不能声明正式决选")
    scores = {}
    for index, candidate in enumerate(candidates):
        results = [done[index * count + episode]["result"] for episode in range(count)]
        scores[id(candidate)] = (sum(row[0] for row in results), sum(row[1] for row in results))
    reference = placement._reference_calibration(replay_args(snapshot))
    finalists, _records = choose_finalists(candidates, pose_inv(reference.pose7).astype(float),
                                           8, lambda candidate: scores[id(candidate)])
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump([{"pose7": item.pose.tolist(), "yaw_index": item.yaw_index}
                   for item in finalists], stream, ensure_ascii=False, indent=2)


def geometry_shortlist(snapshot, candidates_file, output):
    """代表帧未完整时选参考位姿和每个其他朝向首项，明确不是正式决选。"""
    from handumi.robots.utils import pose_inv
    from handumi.scripts.replay import replay_dataset_placement as placement

    reference = placement._reference_calibration(replay_args(snapshot))
    reference_pose = pose_inv(reference.pose7).astype(float)
    candidates = load_candidates(candidates_file)
    first = next((item for item in candidates if item.yaw_index == 0
                  and np.array_equal(item.pose, reference_pose)), None)
    if first is None:
        raise ValueError("冻结候选中缺少参考摆放")
    chosen = [first]
    for yaw in range(1, 8):
        item = next((candidate for candidate in candidates if candidate.yaw_index == yaw), None)
        if item is None:
            raise ValueError(f"冻结候选缺少朝向 {yaw}")
        chosen.append(item)
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump([{"pose7": item.pose.tolist(), "yaw_index": item.yaw_index}
                   for item in chosen], stream, ensure_ascii=False, indent=2)


def compare_events(baseline_path, optimized_path):
    """计算两版相同连续任务前缀的墙钟与逐项判定，避免不同难度偏差。"""
    a, b = task_ids(load_events(baseline_path)), task_ids(load_events(optimized_path))
    common = set(a) & set(b)
    count = 0
    while count in common:
        count += 1
    if not count:
        return {"共同连续任务数": 0, "可比较": False}
    baseline_seconds = max(a[i]["elapsed_seconds"] for i in range(count))
    optimized_seconds = max(b[i]["elapsed_seconds"] for i in range(count))
    differences = [i for i in range(count) if json_value(a[i]["result"]) != json_value(b[i]["result"])]
    return {"共同连续任务数": count, "基线已完成任务数": len(a), "优化已完成任务数": len(b),
            "基线墙钟秒": baseline_seconds, "优化墙钟秒": optimized_seconds,
            "墙钟加速比": baseline_seconds / optimized_seconds if optimized_seconds else None,
            "结果完全一致": not differences, "差异任务编号": differences,
            "说明": "优化版在前缀完成前可能并行执行额外任务，墙钟包含其开销；结果差异需要数值复核。"}


def child(args):
    """在独立进程执行一个阶段，阶段输出和完成事件持续落盘。"""
    from handumi.retargeting.dataset_placement import generate_candidates
    from handumi.robots.utils import pose_inv

    events = Path(args.out) / "events.jsonl"
    start = time.monotonic()
    provenance = code_provenance()
    provenance["environment"].update({key: os.environ.get(key) for key in
                                      ("JAX_PLATFORMS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")})
    append_event(events, {"event": "provenance", "value": provenance})
    replay, reference, contexts, geometry, entries, setup = prepare_contexts(args.snapshot, events)
    append_event(events, {"event": "setup_done", "seconds": setup, "episodes": len(entries),
                          "frames": sum(e["length"] for e in entries)})
    stage_start = time.monotonic()
    if args.phase == "input":
        pass
    elif args.phase == "geometry":
        first = json.loads(Path(args.first_file).read_text(encoding="utf-8"))
        if first["episode_count"] != len(contexts):
            raise ValueError("冻结首帧结果的段数不一致")
        reference_pose = pose_inv(reference.pose7).astype(float)
        from scipy.spatial.transform import Rotation
        ref_rot = Rotation.from_quat(reference_pose[3:])
        def lookup(base, index):
            """根据水平角度和段索引恢复冻结首帧点。"""
            angle = (Rotation.from_quat(base[3:]) * ref_rot.inv()).as_euler("zyx")[0]
            yaw = int(round(angle / (np.pi / 4))) % 8
            return first["xy"][yaw][index]
        candidates = generate_candidates(reference_pose, [c["targets_for_search"] for c in contexts],
                                         geometry[0], lookup)
        rows = [{"pose7": c.pose.tolist(), "yaw_index": c.yaw_index} for c in candidates]
        with (Path(args.out) / "candidates.json").open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, ensure_ascii=False, indent=2)
        append_event(events, {"event": "geometry_done", "seconds": time.monotonic() - stage_start,
                              "candidate_count": len(rows)})
    else:
        candidates = load_candidates(args.candidates_file) if args.phase != "first" else []
        if args.phase == "full":
            candidates = candidates[:args.full_candidates]
        tasks = build_tasks(args.phase, contexts, reference, candidates)
        ids = list(range(len(tasks)))
        if args.reference_events:
            done = task_ids(load_events(args.reference_events))
            ids = sorted(index for index in done if 0 <= index < len(tasks))
        append_event(events, {"event": "stage_start", "phase": args.phase, "tasks": len(ids),
                              "workers": args.workers, "seconds_since_start": stage_start - start})
        deadline = stage_start + args.time_limit
        if args.workers == 1:
            serial_tasks(replay, reference, contexts, tasks, ids, events, stage_start, deadline)
        else:
            parallel_tasks(replay, reference, contexts, tasks, ids, events, args.workers, stage_start, deadline)
    append_event(events, {"event": "stage_end", "phase": args.phase,
                          "stage_seconds": time.monotonic() - stage_start,
                          "total_seconds": time.monotonic() - start})


def main():
    """解析全量快照、隔离阶段运行和共同任务比较命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("dataset")
    snap.add_argument("output")
    snap.add_argument("--robot-tcp-calibration", required=True)
    freeze = commands.add_parser("freeze-first")
    freeze.add_argument("events")
    freeze.add_argument("output")
    freeze.add_argument("--episodes", type=int, required=True)
    finalists = commands.add_parser("select-finalists")
    finalists.add_argument("snapshot")
    finalists.add_argument("candidates")
    finalists.add_argument("events")
    finalists.add_argument("output")
    shortlist = commands.add_parser("geometry-shortlist")
    shortlist.add_argument("snapshot")
    shortlist.add_argument("candidates")
    shortlist.add_argument("output")
    compare = commands.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("optimized")
    compare.add_argument("output")
    run = commands.add_parser("run")
    run.add_argument("snapshot")
    run.add_argument("out")
    run.add_argument("--phase", choices=["input", "first", "geometry", "sample", "full"], required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--time-limit", type=float, default=300)
    run.add_argument("--first-file")
    run.add_argument("--candidates-file")
    run.add_argument("--reference-events")
    run.add_argument("--full-candidates", type=int, default=8)
    run.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(create_snapshot(args.dataset, args.output, args.robot_tcp_calibration), ensure_ascii=False))
    elif args.command == "freeze-first":
        freeze_first(args.events, args.output, args.episodes)
    elif args.command == "select-finalists":
        select_finalists(args.snapshot, args.candidates, args.events, args.output)
    elif args.command == "geometry-shortlist":
        geometry_shortlist(args.snapshot, args.candidates, args.output)
    elif args.command == "compare":
        report = compare_events(args.baseline, args.optimized)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False))
    elif args._child:
        child(args)
    else:
        if args.workers < 1 or args.time_limit <= 0:
            raise ValueError("worker 数量和时间上限必须为正")
        output = Path(args.out).resolve()
        snapshot = Path(args.snapshot).resolve()
        if output == snapshot or output in snapshot.parents or snapshot in output.parents:
            raise ValueError("阶段输出和快照目录必须相互独立")
        if output.exists() and any(output.iterdir()):
            raise ValueError("阶段输出目录必须为空")
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_child"]
        # 独立会话确保硬超时仅终止此次实验的进程树。
        started = time.monotonic()
        with (output / "stage.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=args.time_limit + 30)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status, timed_out = process.returncode, True
        event_file = output / "events.jsonl"
        events = load_events(event_file) if event_file.exists() else []
        provenance = next((e["value"] for e in events if e.get("event") == "provenance"), None)
        setup = next((e["seconds"] for e in events if e.get("event") == "setup_done"), None)
        stage_end = next((e for e in reversed(events) if e.get("event") == "stage_end"), None)
        summary = {"status": status, "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started,
                   "setup_seconds": setup, "stage_seconds": stage_end.get("stage_seconds") if stage_end else None,
                   "completed_tasks": len(task_ids(events)), "code_provenance": provenance,
                   "phase": args.phase, "workers": args.workers, "time_limit": args.time_limit,
                   "command": command, "pythonpath": os.environ.get("PYTHONPATH"),
                   "process_group": process.pid}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        if status:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
