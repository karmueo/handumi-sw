"""隔离数据与缓存，对数据筛选命令执行可复现的冷、热性能对比。

先使用 snapshot 创建包含数据列、视频、元信息及标定的快照，再分别以基线和
待验证源码运行 run。全部轨迹帧均保留；生成的缓存、报告和日志仅写入
独立输出目录。计时从启动 Python 子进程开始，包含导入和报告写入。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def select_episodes(rows):
    """按帧数和编号排序，选择八个均匀分位的完整片段。"""
    ordered = sorted(((int(index), int(length)) for index, length in rows),
                     key=lambda item: (item[1], item[0]))
    if len(ordered) < 8:
        raise ValueError("至少需要八个 episode")
    if len({index for index, _ in ordered}) != len(ordered):
        raise ValueError("episode 编号重复")
    return [ordered[round((len(ordered) - 1) * i / 7)] for i in range(8)]


def file_digest(path):
    """返回文件内容的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _separate(first, second):
    """确认两个路径完全分离，避免误改数据集或已有结果。"""
    a, b = Path(first).resolve(), Path(second).resolve()
    return a != b and a not in b.parents and b not in a.parents


def make_snapshot(dataset, output, robot_tcp):
    """复制生产读取所需文件与标定，并冻结选段和每个文件的摘要。"""
    import pyarrow.parquet as pq

    dataset, output, robot_tcp = map(lambda p: Path(p).resolve(),
                                    (dataset, output, robot_tcp))
    if not _separate(dataset, output):
        raise ValueError("快照目录必须与源数据集完全分离")
    if output.exists():
        raise ValueError("快照目录已存在，禁止覆盖")
    if not robot_tcp.is_file():
        raise ValueError("缺少机器人 TCP 标定")
    deployment = dataset / ".handumi" / "deployment" / "rm75b.yaml"
    if not deployment.is_file():
        raise ValueError("缺少已估计的仿真部署标定")
    rows = []
    for file in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", "length"])
        rows.extend(zip(table["episode_index"].to_pylist(), table["length"].to_pylist()))
    selected = select_episodes(rows)
    output.mkdir(parents=True)
    target = output / "dataset"
    shutil.copytree(dataset / "data", target / "data")
    # LeRobot 即使设置 download_videos=False 也要求本地视频文件存在。
    # 完整复制视频也保留正式筛选报告计算数据清单时的 I/O 工作量。
    if (dataset / "videos").is_dir():
        shutil.copytree(dataset / "videos", target / "videos")
    shutil.copytree(dataset / "meta" / "episodes", target / "meta" / "episodes")
    for name in ("info.json", "stats.json", "tasks.parquet"):
        shutil.copy2(dataset / "meta" / name, target / "meta" / name)
    destination = target / ".handumi" / "deployment" / "rm75b.yaml"
    destination.parent.mkdir(parents=True)
    shutil.copy2(deployment, destination)
    shutil.copy2(robot_tcp, output / "robot_tcp.yaml")
    files = {str(path.relative_to(output)): file_digest(path)
             for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema_version": 1, "source": str(dataset),
                "episodes": [{"episode": i, "frames": n} for i, n in selected],
                "selection": "sort by (length, episode), round((N-1)*i/7), i=0..7",
                "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_snapshot(snapshot):
    """逐文件核对快照与标定，发现变化则拒绝运行。"""
    snapshot = Path(snapshot).resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = snapshot / relative
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"快照摘要不匹配: {relative}")
    if len(manifest["episodes"]) != 8:
        raise ValueError("快照必须包含八个完整 episode")
    return manifest


_INSTRUMENTED = False
_LAST_STAGE = None
_LAST_STAGE_TIME = 0.0


def _event(kind, **values):
    """以追加方式记录事件，多个 worker 可安全写入同一日志。"""
    path = os.environ.get("HANDUMI_SCREEN_BENCH_EVENTS")
    if path is None:
        return
    data = json.dumps({"event": kind, "pid": os.getpid(),
                       "time": time.monotonic(), **values}) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data.encode())
    finally:
        os.close(fd)


def _install_instrumentation():
    """统计真实模型构造、逐段缓存命中及阶段，不修改求解决策。"""
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return
    from handumi.dataset.episode_solve_cache import EpisodeSolveCache
    from handumi.robots import registry
    from handumi.scripts.replay import replay_in_sim, replay_prepare

    original_load = EpisodeSolveCache.load

    def load(cache, key):
        """记录缓存加载的真实结果。"""
        rollout, failure = original_load(cache, key)
        _event("cache", result="failure" if failure is not None else
               "hit" if rollout is not None else "miss")
        return rollout, failure

    EpisodeSolveCache.load = load
    for module in (registry, replay_in_sim):
        original = module.load_embodiment

        def model(*args, _original=original, **kwargs):
            """记录一次机器人运行时构造。"""
            _event("model", robot=str(args[0]) if args else str(kwargs.get("name")))
            return _original(*args, **kwargs)

        module.load_embodiment = model
    for module in (replay_in_sim, replay_prepare):
        original_report = module.report

        def report(args, stage, _original=original_report, **details):
            """限频记录当前阶段和进度，并继续调用原有回调。"""
            global _LAST_STAGE, _LAST_STAGE_TIME
            now = time.monotonic()
            if stage != _LAST_STAGE or now - _LAST_STAGE_TIME >= 5:
                _event("stage", episode=getattr(args, "episode", None),
                       stage=stage, **details)
                _LAST_STAGE, _LAST_STAGE_TIME = stage, now
            return _original(args, stage, **details)

        module.report = report
    original_finish = replay_prepare._finish_rollout

    def finish(args, rollout, **kwargs):
        """记录完整验收通过及碰撞证明身份，不改变实际检查。"""
        result = original_finish(args, rollout, **kwargs)
        _event("strict_validation", episode=getattr(args, "episode", None),
               valid_collision_summary=replay_prepare._valid_collision_summary(args, result))
        return result

    replay_prepare._finish_rollout = finish
    _INSTRUMENTED = True


def _bench_setup(**kwargs):
    """建立原始筛选状态，worker 内独立安装轻量计数器。"""
    _event("worker_environment", variables={name: os.environ.get(name) for name in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")})
    _install_instrumentation()
    from handumi.dataset.screening import _build_screen_state

    return _build_screen_state(**kwargs)


def _bench_task(episode, state):
    """执行完整原始筛选任务并记录开始和结束时间。"""
    _install_instrumentation()
    from handumi.dataset.screening import _screen_one_episode

    _event("episode_start", episode=episode)
    try:
        result = _screen_one_episode(episode, state)
    except BaseException as exc:
        _event("episode_error", episode=episode, error=str(exc))
        raise
    _event("episode_end", episode=episode, frames=result["frame_count"],
           unreachable=result["unreachable"] is not None)
    return result


def _child(arguments):
    """运行生产 CLI；仅替换调用包装器以采集统计事件。"""
    _install_instrumentation()
    from handumi.dataset import parallel, screening
    from handumi.scripts import screen_dataset

    # 旧版 None 保留父进程的线程环境；新版 None 使用一个线程。
    # 常量在父进程计算线程参数时读取，不需要跨进程传递模块状态。
    parallel.THREADS_PER_WORKER = None
    original_map = screening.map_episodes

    def instrumented_map(episodes, **kwargs):
        """保持任务和状态语义，仅包装生产入口。"""
        kwargs["setup"] = _bench_setup
        kwargs["task"] = _bench_task
        return original_map(episodes, **kwargs)

    screening.map_episodes = instrumented_map
    sys.argv = ["handumi dataset screen", *arguments]
    screen_dataset.main()


def _tree_rss(pid):
    """读取 Linux 进程树瞬时 RSS 字节数；进程退出时忽略缺失项。"""
    total, pending, seen = 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            statm = Path(f"/proc/{current}/statm").read_text().split()
            total += int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
            children = Path(f"/proc/{current}/task/{current}/children").read_text()
            pending.extend(int(item) for item in children.split())
        except (OSError, ValueError, IndexError):
            continue
    return total


def _events(path):
    """读取完整事件行，忽略进程中断导致的末尾不完整记录。"""
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def source_fingerprint(source):
    """摘要生产 Python 源文件，阻止不同源码混入同一组测试。"""
    source = Path(source).resolve()
    digest = hashlib.sha256()
    for file in sorted((source / "handumi").rglob("*.py")):
        digest.update(str(file.relative_to(source)).encode() + b"\0")
        digest.update(bytes.fromhex(file_digest(file)))
    return digest.hexdigest()


def verify_output(report_path, manifest, records):
    """核对每个已求解片段的完整帧、误差和实际碰撞终验证据。"""
    import numpy as np

    report_path = Path(report_path)
    payload = json.loads(report_path.read_text())
    expected = {int(e["episode"]): int(e["frames"]) for e in manifest["episodes"]}
    episodes = {int(e["episode_index"]): e for e in payload["episodes"]}
    errors = []
    if set(episodes) != set(expected):
        errors.append("报告中的 episode 集合与快照不同")
    solved = {index: item for index, item in episodes.items() if int(item["frame_count"]) > 0}
    if not solved:
        errors.append("没有完整求解成功的片段")
    validated = {int(event["episode"]) for event in records
                 if event["event"] == "strict_validation"
                 and event.get("valid_collision_summary") and event.get("episode") is not None}
    cache_path = report_path.with_name(report_path.stem + "_solves.npz")
    if solved and not cache_path.is_file():
        errors.append("缺少筛选轨迹缓存")
    elif solved:
        with np.load(cache_path, allow_pickle=False) as archive:
            for index, item in solved.items():
                count = expected.get(index)
                if count is None:
                    continue
                prefix = f"{index}/"
                required = ("qpos", "frame_indices", "left_pos_error_m", "right_pos_error_m",
                            "left_rot_error_deg", "right_rot_error_deg")
                if any(prefix + field not in archive for field in required):
                    errors.append(f"episode {index} 缺少轨迹字段")
                    continue
                frames = archive[prefix + "frame_indices"]
                qpos = archive[prefix + "qpos"]
                if (int(item["frame_count"]) != count or len(qpos) != count
                        or not np.array_equal(frames, np.arange(count))):
                    errors.append(f"episode {index} 帧数或原始帧索引不完整")
                if not np.isfinite(qpos).all():
                    errors.append(f"episode {index} 关节值无效")
                for side in ("left", "right"):
                    for field, limit in (("pos_error_m", 0.005), ("rot_error_deg", 5.0)):
                        values = archive[prefix + side + "_" + field]
                        if (len(values) != count or not np.isfinite(values).all()
                                or np.any(values > limit + 1e-6)):
                            errors.append(f"episode {index} {side}_{field} 超限或无效")
                if index not in validated:
                    errors.append(f"episode {index} 缺少严格碰撞终验证据")
    return {"ok": not errors, "errors": errors,
            "accepted": sorted(i for i, item in episodes.items() if item["status"] == "accepted"),
            "rejected": sorted(i for i, item in episodes.items() if item["status"] != "accepted"),
            "solved": sorted(solved), "reported_frames":
            {i: int(item["frame_count"]) for i, item in episodes.items()}}


def _one_run(snapshot, source, output, cache, jobs, timeout):
    """在独立子进程中执行一次完整筛选并采样进程树内存。"""
    manifest = validate_snapshot(snapshot)
    output.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "_child",
               str(Path(snapshot).resolve() / "dataset"), "--robot", "rm75b",
               "--deployment-profile", "sim", "--retarget-mode", "absolute-tcp",
               "--base-placement", "episode-xy", "--jobs", str(jobs),
               "--episodes", ",".join(str(e["episode"]) for e in manifest["episodes"]),
               "--robot-tcp-calibration", str(Path(snapshot).resolve() / "robot_tcp.yaml"),
               "--cache-dir", str(cache), "--report", str(output / "report.json"),
               "--markdown", str(output / "report.md")]
    events = output / "events.jsonl"
    env = dict(os.environ, PYTHONPATH=str(Path(source).resolve()),
               HANDUMI_SCREEN_BENCH_EVENTS=str(events), JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
               PYTHONHASHSEED="0", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    peak = [0]
    stop = threading.Event()
    started = time.monotonic()
    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   start_new_session=True)

        def sample():
            """定期采样本次任务整个进程树的常驻内存。"""
            while not stop.wait(0.2):
                peak[0] = max(peak[0], _tree_rss(process.pid))

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        finally:
            stop.set()
            thread.join()
    elapsed = time.monotonic() - started
    records = _events(events)
    starts = {e["episode"]: e["time"] for e in records if e["event"] == "episode_start"}
    episode_seconds = {str(e["episode"]): e["time"] - starts[e["episode"]]
                       for e in records if e["event"] == "episode_end" and e["episode"] in starts}
    cache_counts = {kind: sum(e["event"] == "cache" and e["result"] == kind for e in records)
                    for kind in ("hit", "miss", "failure")}
    report = output / "report.json"
    correctness = None
    if report.exists():
        correctness = verify_output(report, manifest, records)
    result = {"returncode": code, "timed_out": timed_out, "wall_seconds": elapsed,
              "peak_tree_rss_bytes_sampled": peak[0], "model_initializations":
              sum(e["event"] == "model" for e in records), "cache": cache_counts,
              "episode_seconds": episode_seconds, "correctness": correctness,
              "command": command, "pythonpath": str(Path(source).resolve())}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(snapshot, source, output, jobs, timeout, rounds):
    """按轮执行冷缓存和该轮缓存的热命中，完整记录失败或超时。"""
    snapshot, source, output = map(lambda p: Path(p).resolve(), (snapshot, source, output))
    if not _separate(snapshot, output) or not _separate(source, output):
        raise ValueError("输出目录必须与快照和源码完全分离")
    if output.exists():
        raise ValueError("输出目录已存在，禁止覆盖")
    if jobs < 1 or rounds < 1 or timeout <= 0:
        raise ValueError("jobs、rounds 和 timeout 必须为正数")
    manifest = validate_snapshot(snapshot)
    before = source_fingerprint(source)
    output.mkdir(parents=True)
    runs = []
    for index in range(rounds):
        cache = output / f"cache-{index + 1}"
        for mode in ("cold", "warm"):
            if source_fingerprint(source) != before:
                raise RuntimeError("源码在基准期间发生变化，终止比较")
            name = f"{index + 1:02d}-{mode}"
            print(f"[benchmark] starting {name}, jobs={jobs}", flush=True)
            item = _one_run(snapshot, source, output / name, cache, jobs, timeout)
            item.update(mode=mode, round=index + 1)
            runs.append(item)
            print(f"[benchmark] completed {name}: {item['wall_seconds']:.1f}s "
                  f"status={item['returncode']} timeout={item['timed_out']}", flush=True)
            if item["returncode"] != 0:
                break
        if runs[-1]["returncode"] != 0:
            break
    summary = {"source": str(source), "source_fingerprint": before,
               "snapshot_manifest": manifest, "jobs": jobs, "runs": runs,
               "median_seconds": {mode: statistics.median([r["wall_seconds"] for r in runs
                   if r["mode"] == mode and r["returncode"] == 0])
                   if any(r["mode"] == mode and r["returncode"] == 0 for r in runs) else None
                   for mode in ("cold", "warm")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def compare_reports(before_path, after_path):
    """比较接受集合、帧索引、关节误差和碰撞评分，保留允许的搜索差异。"""
    import numpy as np

    before_path, after_path = Path(before_path), Path(after_path)
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    left = {int(item["episode_index"]): item for item in before["episodes"]}
    right = {int(item["episode_index"]): item for item in after["episodes"]}
    result = {"same_episode_set": set(left) == set(right), "status_changes": {},
              "frame_count_changes": {}, "collision_metric_changes": {}, "trajectory": {}}
    for episode in sorted(set(left) & set(right)):
        a, b = left[episode], right[episode]
        if a["status"] != b["status"]:
            result["status_changes"][episode] = [a["status"], b["status"]]
        if a["frame_count"] != b["frame_count"]:
            result["frame_count_changes"][episode] = [a["frame_count"], b["frame_count"]]
        keys = ("self_collision_min_clearance_m", "self_collision_frames",
                "table_min_clearance_m", "table_penetration_frames")
        for key in keys:
            x, y = a.get("metrics", {}).get(key), b.get("metrics", {}).get(key)
            if x is not None and y is not None and not np.isclose(x, y, atol=1e-7, rtol=1e-6):
                result["collision_metric_changes"][f"{episode}/{key}"] = [x, y]
    left_cache = before_path.with_name(before_path.stem + "_solves.npz")
    right_cache = after_path.with_name(after_path.stem + "_solves.npz")
    if left_cache.exists() and right_cache.exists():
        with np.load(left_cache, allow_pickle=False) as old, np.load(right_cache, allow_pickle=False) as new:
            for episode in sorted(set(left) & set(right)):
                entry = {}
                for field in ("frame_indices", "qpos", "left_pos_error_m", "right_pos_error_m",
                              "left_rot_error_deg", "right_rot_error_deg"):
                    key = f"{episode}/{field}"
                    if key not in old or key not in new:
                        continue
                    a, b = old[key], new[key]
                    same_shape = a.shape == b.shape
                    entry[field] = {"same_shape": same_shape,
                                    "equal": bool(np.array_equal(a, b)) if same_shape else False,
                                    "max_abs_difference": float(np.max(np.abs(a - b)))
                                    if same_shape and a.size else None}
                if entry:
                    result["trajectory"][episode] = entry
    return result


def main():
    """解析快照、基准执行与内部子进程命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("dataset")
    snapshot.add_argument("output")
    snapshot.add_argument("--robot-tcp", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("snapshot")
    execute.add_argument("source")
    execute.add_argument("output")
    execute.add_argument("--jobs", type=int, required=True)
    execute.add_argument("--timeout", type=int, default=1200)
    execute.add_argument("--rounds", type=int, default=3)
    child = sub.add_parser("_child")
    child.add_argument("arguments", nargs=argparse.REMAINDER)
    compare = sub.add_parser("compare")
    compare.add_argument("before_report")
    compare.add_argument("after_report")
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(make_snapshot(args.dataset, args.output, args.robot_tcp), indent=2))
    elif args.command == "run":
        print(json.dumps(run(args.snapshot, args.source, args.output, args.jobs,
                             args.timeout, args.rounds), indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_reports(args.before_report, args.after_report), indent=2))
    else:
        _child(args.arguments)


if __name__ == "__main__":
    main()
