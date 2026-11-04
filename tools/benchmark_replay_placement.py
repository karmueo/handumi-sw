"""对固定的小样本执行回放摆放搜索阶段基准，隔离原数据集与缓存。

快照示例：python tools/benchmark_replay_placement.py snapshot DATASET SNAPSHOT
    --robot-tcp-calibration PATH
测试示例：PYTHONPATH=/path/to/worktree/src python tools/benchmark_replay_placement.py
    run SNAPSHOT FROZEN_CANDIDATES OUTPUT --workers 2 --warm-runs 3
首次运行会冻结候选，后续各版本必须复用相同的快照和候选文件。
使用 --capture-dir 单独验证轨迹数组，使用 --reuse-dir 的空目录验证缓存复用；
捕获会增加 I/O，不能混入纯性能比较。compare 和 compare-captures 用于验收。
禁止在原始数据集目录写入快照或缓存，也禁止运行全部 episode 的基准。
冷轮计时从调用 benchmark 起包含生产代码导入、输入准备和进程启动，
不包含 Python 启动和本脚本先前导入；进程树 RSS 为低频采样，可能漏掉瞬时峰值。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


def select_episodes(rows):
    """按长度及编号稳定排序，返回四个长度分位的 episode。"""
    ordered = sorted(((int(i), int(n)) for i, n in rows), key=lambda r: (r[1], r[0]))
    if len(ordered) < 4:
        raise ValueError("数据集至少需要 4 个 episode")
    indexes = [round((len(ordered) - 1) * q) for q in (0.1, 0.35, 0.65, 0.9)]
    return [ordered[i] for i in indexes]


def replay_args(root):
    """创建与目标回放命令一致的参数，所有写入都指向隔离目录。"""
    from handumi.scripts.replay import replay_in_sim as replay

    args = replay.build_parser().parse_args([
        str(root), "--robot", "rm75b", "--deployment-profile", "sim",
        "--retarget-mode", "absolute-tcp", "--base-placement", "dataset-xy-yaw",
        "--strict-ik", "--headless", "--prepare-mode", "all", "--progress", "off",
    ])
    args.repo_id = "benchmark/local"
    args.dataset_root = Path(root)
    calibration = Path(root) / "robot_tcp.yaml"
    if calibration.is_file():
        args.robot_tcp_calibration = calibration
    replay.resolve_replay_options(args)
    return args


def make_snapshot(dataset, output, robot_tcp_calibration):
    """读取四个完整 episode 和元信息，输出独立、无视频的数据快照。"""
    import pyarrow.parquet as pq

    from handumi.dataset.episode_solve_cache import array_digest, digest_file
    from handumi.dataset.selection import resolve_dataset_selection
    from handumi.scripts.replay import replay_in_sim as replay

    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("快照目录必须位于原数据集之外")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("快照目录必须为空，禁止覆盖已有内容")
    calibration = Path(robot_tcp_calibration).expanduser().resolve()
    if not calibration.is_file():
        raise ValueError("机器人 TCP 标定文件不存在")
    rows = []
    for file in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(file, columns=["episode_index", "length"])
        rows.extend(zip(table["episode_index"].to_pylist(), table["length"].to_pylist()))
    chosen = select_episodes(rows)
    selection = resolve_dataset_selection(str(dataset))
    args = replay_args(dataset)
    args.repo_id, args.dataset_root = selection.repo_id, selection.root
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    info = None
    for local_index, (episode, length) in enumerate(chosen):
        args.episode = episode
        states, fps, info, openings = replay.load_episode_states(args)
        if len(states) != length:
            raise ValueError(f"episode {episode} 的帧数与元数据不一致")
        arrays = {"states": states}
        if openings is not None:
            arrays["openings"] = openings
        name = f"episode-{local_index}.npz"
        np.savez_compressed(output / name, **arrays)
        entries.append({"index": episode, "local_index": local_index, "length": length,
                        "fps": fps, "file": name, "states": array_digest(states),
                        "openings": array_digest(openings)})
    # 保存完整的元数据，后续只替换数量，避免重复接触原数据集。
    target_calibration = output / "robot_tcp.yaml"
    shutil.copy2(calibration, target_calibration)
    manifest = {"version": 1, "episodes": entries, "info": info,
                "selection": {"quantiles": [0.1, 0.35, 0.65, 0.9],
                              "ordering": ["length", "episode_index"],
                              "rounding": "python round"},
                "robot_tcp_calibration": {"file": target_calibration.name,
                                          "sha256": digest_file(target_calibration),
                                          "source": str(calibration)}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def read_snapshot(directory, shortest=False):
    """读取并校验快照摘要，按需选取其中最短的两个完整 episode。"""
    from handumi.dataset.episode_solve_cache import array_digest, digest_file

    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    calibration = manifest.get("robot_tcp_calibration")
    if calibration is not None and digest_file(directory / calibration["file"]) != calibration["sha256"]:
        raise ValueError("快照机器人 TCP 标定摘要校验失败")
    entries = manifest["episodes"]
    if shortest:
        entries = sorted(entries, key=lambda item: (item["length"], item["index"]))[:2]
    info = dict(manifest["info"])
    info["total_episodes"] = len(entries)
    loaded = []
    for entry in entries:
        with np.load(directory / entry["file"], allow_pickle=False) as archive:
            states = np.asarray(archive["states"])
            openings = np.asarray(archive["openings"]) if "openings" in archive else None
        if array_digest(states) != entry["states"] or array_digest(openings) != entry["openings"]:
            raise ValueError(f"快照校验失败：{entry['file']}")
        loaded.append((states, entry["fps"], info, openings))
    return entries, loaded


@contextlib.contextmanager
def snapshot_loader(loaded):
    """让正式输入准备逻辑读取快照，仅在当前进程临时替换读取函数。"""
    from handumi.scripts.replay import replay_dataset_placement as placement
    from handumi.scripts.replay import replay_in_sim as replay

    original_metadata, original_loader = placement.ensure_metadata, replay.load_episode_states
    placement.ensure_metadata = lambda *_args, **_kwargs: loaded[0][2]
    replay.load_episode_states = lambda args: loaded[args.episode]
    try:
        yield
    finally:
        placement.ensure_metadata, replay.load_episode_states = original_metadata, original_loader


def normalized(value):
    """将 NumPy 数值与数组递归转换为 JSON 兼容值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def candidate_subset(candidates, reference):
    """冻结每个朝向第一个候选，完整验证使用参考位姿和另一朝向。"""
    selected = []
    for yaw in range(8):
        candidate = next((item for item in candidates if item.yaw_index == yaw), None)
        if candidate is not None:
            selected.append(candidate)
    if len(selected) != 8:
        raise ValueError("候选集合没有覆盖八个朝向")
    reference_candidate = next((item for item in candidates
                                if item.yaw_index == 0 and np.array_equal(item.pose, reference)), None)
    if reference_candidate is None:
        raise ValueError("候选集合缺少参考位姿")
    return selected, [reference_candidate, selected[4]]


def _benchmark(snapshot, candidate_file, workers=1, warm_runs=0, shortest=False, capture_dir=None, reuse_dir=None, benchmark_started=None):
    """运行各搜索阶段并报告原始耗时、评分和进程最大内存。"""
    from handumi.retargeting.base_placement import first_frame_candidates
    from handumi.retargeting.dataset_placement import (
        Candidate,
        generate_candidates,
        rank_result,
    )
    from handumi.robots.utils import pose_inv
    from handumi.scripts.replay import replay_dataset_placement as placement

    if capture_dir:
        os.environ["HANDUMI_BENCH_CAPTURE"] = str(Path(capture_dir).resolve())
        install_capture()
    args = replay_args(snapshot)
    if reuse_dir:
        root = Path(reuse_dir).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError("复用基准目录必须为空，避免污染冷缓存结果")
        root.mkdir(parents=True, exist_ok=True)
        args.cache_dir = root / "cache"
        args._placement_scratch_dir = root / "scratch"
    args.placement_workers = workers
    args.prepare_workers = workers
    entries, loaded = read_snapshot(snapshot, shortest)
    reference = placement._reference_calibration(args)
    start = time.monotonic()
    with snapshot_loader(loaded):
        contexts, geometry = placement._episode_contexts(args, reference)
    setup = time.monotonic() - start
    reference_pose = pose_inv(reference.pose7).astype(float)
    episodes = [context["targets_for_search"] for context in contexts]
    pool = None
    if workers > 1:
        from handumi.scripts.replay.replay_placement_pool import PlacementPool
        pool = PlacementPool(args, contexts, reference, workers=workers)
    with pool if pool is not None else contextlib.nullcontext() as worker_pool:
        def iteration():
            """逐阶段计时，使用独立冻结候选评估速度和输出一致性。"""
            iteration_started = time.monotonic()
            times = {}
            started = time.monotonic()
            if worker_pool is None:
                def first_xy(base, index):
                    """调用串行生产首帧搜索。"""
                    ctx = contexts[index]
                    return [c.xy for c in first_frame_candidates(ctx["runtime"], ctx["tool"],
                            ctx["targets"], base, ctx["locked"], max_seeds=1)]
            else:
                all_xy = worker_pool.first_frames(reference_pose)
                def first_xy(base, index):
                    """按生产 yaw 序读取已经完成的并行首帧搜索。"""
                    from scipy.spatial.transform import Rotation
                    ref_rot = Rotation.from_quat(reference_pose[3:])
                    angle = (Rotation.from_quat(base[3:]) * ref_rot.inv()).as_euler("zyx")[0]
                    yaw = int(round(angle / (np.pi / 4))) % 8
                    return all_xy[yaw][index]
            generated = generate_candidates(reference_pose, episodes, geometry[0], first_xy)
            times["candidate_generation"] = time.monotonic() - started
            candidate_path = Path(candidate_file)
            if not candidate_path.exists():
                candidate_path.write_text(json.dumps([
                    {"pose7": c.pose.tolist(), "yaw_index": c.yaw_index}
                    for c in generated
                ], indent=2), encoding="utf-8")
            frozen = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidates = [Candidate(np.asarray(row["pose7"]), row["yaw_index"]) for row in frozen]
            sample, full = candidate_subset(candidates, reference_pose)
            started = time.monotonic()
            if worker_pool is None:
                evaluate = placement._sample_evaluator(args, contexts, reference)
                sample_results = [evaluate(candidate) for candidate in sample]
            else:
                sample_results = worker_pool.sample_scores(sample)
            times["sample_evaluation"] = time.monotonic() - started
            # 候选由第一个串行基准冻结，后续所有版本读取同一文件。
            full_path = Path(str(candidate_file) + ".full.json")
            if not full_path.exists():
                best_index = max(range(len(sample)), key=lambda i: (tuple(sample_results[i]), -i))
                full_path.write_text(json.dumps([
                    {"pose7": reference_pose.tolist(), "yaw_index": 0},
                    {"pose7": sample[best_index].pose.tolist(), "yaw_index": sample[best_index].yaw_index},
                ], indent=2), encoding="utf-8")
            full = [Candidate(np.asarray(row["pose7"]), row["yaw_index"])
                    for row in json.loads(full_path.read_text(encoding="utf-8"))]
            started = time.monotonic()
            if worker_pool is None:
                full_results = [[placement._full_episode(args, ctx, i,
                                placement._candidate_calibration(reference, candidate.pose))
                                 for i, ctx in enumerate(contexts)] for candidate in full]
            else:
                full_results = worker_pool.full_scores(full)
            times["full_evaluation"] = time.monotonic() - started
            reuse_metrics = None
            if reuse_dir:
                reuse_metrics = check_promotion(args, reference, full[1], contexts, loaded)
            return {"seconds": times, "total_seconds": time.monotonic() - iteration_started,
                    "reuse": reuse_metrics,
                    "generated": [{"pose7": c.pose.tolist(), "yaw_index": c.yaw_index} for c in generated],
                    "sample_results": sample_results, "full_results": full_results,
                    "rank": [rank_result(c, reference_pose, results) for c, results in zip(full, full_results)]}
        runs = []
        cold_total_seconds = None
        for index in range(1 + warm_runs):
            runs.append(iteration())
            if index == 0:
                cold_total_seconds = time.monotonic() - benchmark_started
    return normalized({"setup_seconds": setup, "cold_total_seconds": cold_total_seconds, "workers": workers, "episodes": entries,
                       "workload": {"first_frame_tasks": 8 * len(entries),
                                    "sample_candidate_episode_tasks": 8 * len(entries),
                                    "full_candidate_episode_tasks": 2 * len(entries),
                                    "sample_frame_checks": 8 * sum(len(c["samples"]) for c in contexts)},
                       "runs": runs,
                       "warm_total_median_seconds": statistics.median(
                           run["total_seconds"] for run in runs[1:]) if warm_runs else None,
                       "warm_median_seconds": {
                           name: statistics.median(run["seconds"][name] for run in runs[1:])
                           for name in runs[0]["seconds"]} if warm_runs else None,
                       "rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       "rss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})


class ProcessTreeRssSampler:
    """低频采集父子进程同时 RSS；不把各进程历史峰值相加。"""

    def __init__(self, interval=0.5):
        """初始化可选 psutil 采样器，记录不可用和采样失败原因。"""
        self.interval = interval
        self.peak_bytes = None
        self.samples = 0
        self.errors = 0
        self.unavailable = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process()
        except (ImportError, OSError) as exc:
            self.unavailable = str(exc)

    def _sample(self):
        """采集活跃父子进程的 RSS 之和，进程退出或权限变化时跳过本次。"""
        try:
            processes = [self._process, *self._process.children(recursive=True)]
            total = sum(process.memory_info().rss for process in processes)
            self.peak_bytes = max(self.peak_bytes or 0, total)
            self.samples += 1
        except (self._psutil.Error, OSError):
            self.errors += 1

    def start(self):
        """开始后台采样；缺少 psutil 时保留明确的不可用说明。"""
        if self.unavailable is not None:
            return
        self._sample()
        def sample_loop():
            """等待固定间隔后采样，直到基准结束。"""
            while not self._stop.wait(self.interval):
                self._sample()
        self._thread = threading.Thread(target=sample_loop, daemon=True)
        self._thread.start()

    def finish(self):
        """停止采样并返回独立字段，历史峰值由 resource 单独提供。"""
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        return {"peak_kib": self.peak_bytes / 1024 if self.peak_bytes is not None else None,
                "interval_seconds": self.interval, "samples": self.samples,
                "sample_errors": self.errors, "unavailable": self.unavailable,
                "semantics": "simultaneous parent and live children; sampled lower bound"}


def code_provenance():
    """记录实际导入模块路径、git HEAD 与工作树状态摘要。"""
    import handumi
    from handumi.scripts.replay import replay_dataset_placement, replay_in_sim

    package = Path(handumi.__file__).resolve()
    repository = package.parents[2]

    def git(*arguments):
        """只读运行 git 获取工作区信息，失败时返回空值。"""
        try:
            result = subprocess.run(["git", "-C", str(repository), *arguments],
                                    capture_output=True, text=True, timeout=5, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {"handumi_module": str(package), "replay_module": str(Path(replay_in_sim.__file__).resolve()),
            "placement_module": str(Path(replay_dataset_placement.__file__).resolve()),
            "python": sys.executable, "worktree": git("rev-parse", "--show-toplevel"),
            "git_head": git("rev-parse", "HEAD"),
            "git_dirty": status != "" if status is not None else None,
            "git_status_short": status,
            "environment": {key: os.environ.get(key) for key in
                            ("PYTHONHASHSEED", "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def benchmark(snapshot, candidate_file, workers=1, warm_runs=0, shortest=False,
              capture_dir=None, reuse_dir=None):
    """记录冷启动与预热总耗时、导入版本及进程树 RSS，并执行原有基准。"""
    started = time.monotonic()
    sampler = ProcessTreeRssSampler()
    sampler.start()
    try:
        result = _benchmark(snapshot, candidate_file, workers, warm_runs, shortest,
                            capture_dir, reuse_dir, benchmark_started=started)
    finally:
        rss = sampler.finish()
    result["rss_process_tree_sampled"] = rss
    result["code_provenance"] = code_provenance()
    return result


def check_promotion(args, reference, candidate, contexts, loaded):
    """在隔离目录写入固定标定，测量胜出轨迹提升与后续缓存命中。"""
    import io

    import yaml

    from handumi.calibration.deployment import load_deployment_calibration
    from handumi.robots.utils import pose_inv
    from handumi.scripts.replay.replay_placement_cache import promote_winner
    from handumi.scripts.replay.replay_prepare import prepare_episode

    pose = pose_inv(candidate.pose)
    calibration_path = Path(args.cache_dir).parent / "calibration.yaml"
    calibration_path.write_text(yaml.safe_dump({
        "schema_version": 1, "kind": "handumi_robot_table_calibration",
        "robot": args.robot, "scope": "simulation", "verified": False,
        "source": "dataset_xy_yaw_search", "generation": "benchmark",
        "calibration": {"robot_from_table": {
            "position": pose[:3].tolist(), "quaternion": pose[3:].tolist(),
        }},
    }), encoding="utf-8")
    selection = load_deployment_calibration(calibration_path, expected_robot=args.robot,
                                             profile="dataset-xy-yaw")
    report = {"reference_episode_identities": [ctx["input_identity"] for ctx in contexts]}
    args.deployment_selection = selection
    start = time.monotonic()
    with snapshot_loader(loaded):
        promoted = promote_winner(args, reference, selection, candidate, report,
                                  Path(args._placement_scratch_dir))
    promotion_seconds = time.monotonic() - start
    # 只重放成功提升的段，避免失败段的冷求解混入缓存命中测试。
    from handumi.dataset.episode_solve_cache import EpisodeSolveCache
    from handumi.scripts.replay.replay_prepare import episode_identity
    hit_count = 0
    elapsed = 0.0
    for index, episode in enumerate(loaded):
        args.episode = index
        key, _ = episode_identity(args, episode)
        rollout, _ = EpisodeSolveCache(Path(args.cache_dir)).load(key)
        if rollout is None:
            continue
        output = io.StringIO()
        start = time.monotonic()
        with snapshot_loader(loaded), contextlib.redirect_stdout(output):
            prepare_episode(args)
        elapsed += time.monotonic() - start
        hit_count += "[replay] cache hit" in output.getvalue()
    return {"promoted": promoted, "cache_hits": hit_count,
            "promotion_seconds": promotion_seconds, "cache_hit_seconds": elapsed}


def install_capture():
    """在测试进程中临时捕获求解输出，不改变求解或严格验收。"""
    directory = os.environ.get("HANDUMI_BENCH_CAPTURE")
    if not directory:
        return
    from handumi.scripts.replay import replay_dataset_placement as placement
    from handumi.scripts.replay import replay_in_sim as replay

    original = placement._full_episode
    if getattr(original, "_benchmark_capture", False):
        return

    def capture(args, context, episode, selection, *positional, **keywords):
        """包装一次候选求解，在严格验收前保存完整数组以供独立比较。"""
        old_solve = replay.solve_episode

        def solve(trial):
            """记录原求解器返回的完整轨迹，原异常和结果照常向上传递。"""
            rollout = old_solve(trial)
            pose = np.asarray(selection.pose7, dtype=np.float64)
            key = hashlib.sha256(pose.tobytes()).hexdigest()[:16]
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target / f"{key}-{episode}.npz", **rollout)
            return rollout

        replay.solve_episode = solve
        try:
            return original(args, context, episode, selection, *positional, **keywords)
        finally:
            replay.solve_episode = old_solve

    capture._benchmark_capture = True
    placement._full_episode = capture


def compare_captures(left_dir, right_dir):
    """比较可复现的完整关节、目标、实际位姿与精度误差数组。"""
    left_files = {item.name: item for item in Path(left_dir).glob("*.npz")}
    right_files = {item.name: item for item in Path(right_dir).glob("*.npz")}
    differences = []
    if set(left_files) != set(right_files):
        differences.append(f"捕获文件不一致: {sorted(left_files)} vs {sorted(right_files)}")
    fields = ["qpos", "target_left_pose7_robot_world", "target_right_pose7_robot_world",
              "achieved_left_pose7_robot_world", "achieved_right_pose7_robot_world",
              "left_pos_error_m", "right_pos_error_m", "left_rot_error_deg", "right_rot_error_deg"]
    for name in sorted(left_files.keys() & right_files.keys()):
        with np.load(left_files[name], allow_pickle=False) as left, np.load(right_files[name], allow_pickle=False) as right:
            for field in fields:
                if field not in left or field not in right:
                    differences.append(f"{name}:{field} missing")
                elif left[field].shape != right[field].shape or not np.allclose(left[field], right[field], atol=1e-5, rtol=1e-5):
                    differences.append(f"{name}:{field} mismatch")
    if not left_files or not right_files:
        differences.append("没有可比较的完整轨迹")
    return differences


def compare_results(baseline, candidate):
    """比较候选、评分、通过状态及数值误差，返回可读差异列表。"""
    differences = []

    def compare_value(left, right, path):
        """递归比较结构；浮点按固定容差比较，状态和整型必须完全一致。"""
        if isinstance(left, dict) and isinstance(right, dict):
            for key in left.keys() | right.keys():
                if key == "scratch_key":
                    continue
                if key not in left or key not in right:
                    differences.append(f"{path}.{key}: missing")
                else:
                    compare_value(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append(f"{path}: length mismatch")
            else:
                for index, (a, b) in enumerate(zip(left, right, strict=True)):
                    compare_value(a, b, f"{path}[{index}]")
        elif isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                differences.append(f"{path}: boolean mismatch")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(left, int) and isinstance(right, int):
                equal = left == right
            else:
                equal = bool(np.isclose(left, right, atol=1e-5, rtol=1e-5))
            if not equal:
                differences.append(f"{path}: {left!r} != {right!r}")
        elif left != right:
            differences.append(f"{path}: {left!r} != {right!r}")

    left_run, right_run = baseline["runs"][0], candidate["runs"][0]
    for field in ("generated", "sample_results", "full_results", "rank"):
        compare_value(left_run[field], right_run[field], field)
    return differences


def main():
    """解析快照和分阶段基准命令，结果以 JSON 保存。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("dataset")
    snapshot.add_argument("output")
    snapshot.add_argument("--robot-tcp-calibration", required=True)
    run = commands.add_parser("run")
    run.add_argument("snapshot")
    run.add_argument("candidate_file")
    run.add_argument("output")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--warm-runs", type=int, default=0)
    run.add_argument("--shortest", action="store_true")
    run.add_argument("--capture-dir")
    run.add_argument("--reuse-dir")
    compare = commands.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    captures = commands.add_parser("compare-captures")
    captures.add_argument("baseline")
    captures.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "compare-captures":
        differences = compare_captures(args.baseline, args.candidate)
        print(json.dumps({"equivalent": not differences, "differences": differences}, indent=2))
        if differences:
            raise SystemExit(1)
    elif args.command == "compare":
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        differences = compare_results(baseline, candidate)
        print(json.dumps({"equivalent": not differences, "differences": differences}, indent=2))
        if differences:
            raise SystemExit(1)
    elif args.command == "snapshot":
        result = make_snapshot(args.dataset, args.output, args.robot_tcp_calibration)
        print(json.dumps(result["episodes"], ensure_ascii=False, indent=2))
    else:
        result = benchmark(args.snapshot, args.candidate_file, args.workers, args.warm_runs, args.shortest, args.capture_dir, args.reuse_dir)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


# spawn 子进程导入主脚本时也安装捕获钩子；普通性能测试不设置该变量。
if os.environ.get("HANDUMI_BENCH_CAPTURE"):
    install_capture()

if __name__ == "__main__":
    main()
