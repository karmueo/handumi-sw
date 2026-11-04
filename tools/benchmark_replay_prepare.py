"""测量固定摆放下逐段正式回放准备，提供隔离缓存与可选调用计数。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import resource
import shutil
import time
from pathlib import Path

from benchmark_replay_placement import read_snapshot, replay_args, snapshot_loader


@contextlib.contextmanager
def count_replay_calls():
    """统计 IK 与机器人模型初始化，退出时恢复全部原始调用入口。"""
    from handumi.robots import kinematics, registry
    from handumi.scripts.replay import replay_in_sim as replay

    counts = {"ik_calls": 0, "model_initializations": 0}
    solver = kinematics.BimanualKinematicsSolver
    original_ik = solver.ik
    original_registry = registry.load_embodiment
    original_replay = replay.load_embodiment

    def ik(*args, **kwargs):
        """调用原始 IK 并记录调用次数。"""
        counts["ik_calls"] += 1
        return original_ik(*args, **kwargs)

    def instrument_model(original):
        """包装一个已有别名，不通过另一个别名转发以防重复计数。"""
        def load(*args, **kwargs):
            """调用原始模型构造入口并计数。"""
            counts["model_initializations"] += 1
            return original(*args, **kwargs)
        return load

    solver.ik = ik
    registry.load_embodiment = instrument_model(original_registry)
    replay.load_embodiment = instrument_model(original_replay)
    try:
        yield counts
    finally:
        solver.ik = original_ik
        registry.load_embodiment = original_registry
        replay.load_embodiment = original_replay


def benchmark_prepare(snapshot, calibration, output_dir, episodes, *,
                      count_calls=False, existing_cache=None):
    """逐段准备真实完整轨迹，返回耗时、缓存命中及可选实际调用数。"""
    from handumi.calibration.deployment import load_deployment_calibration
    from handumi.scripts.replay.replay_prepare import prepare_episode
    from handumi.scripts.replay.replay_scheduler import _runtime_for

    root = Path(output_dir).resolve()
    snapshot_root = Path(snapshot).resolve()
    if root == snapshot_root or root in snapshot_root.parents or snapshot_root in root.parents:
        raise ValueError("基准输出和快照目录必须相互独立")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("基准输出目录必须为空")
    cache_source = None
    if existing_cache is not None:
        cache_source = Path(existing_cache).resolve()
        if (cache_source == snapshot_root or cache_source in snapshot_root.parents
                or snapshot_root in cache_source.parents):
            raise ValueError("已有缓存和快照目录必须相互独立")
        if not cache_source.is_dir():
            raise ValueError("已有缓存目录不存在")
        if cache_source == root or cache_source in root.parents or root in cache_source.parents:
            raise ValueError("已有缓存与基准输出目录必须相互独立")
    root.mkdir(parents=True, exist_ok=True)
    entries, loaded = read_snapshot(snapshot)
    args = replay_args(snapshot)
    args.cache_dir = root / "cache"
    if cache_source is not None:
        # 命中和意外未命中均只访问副本，绝不改写传入的缓存。
        shutil.copytree(cache_source, args.cache_dir)
    args.deployment_selection = load_deployment_calibration(
        Path(calibration).resolve(), expected_robot=args.robot, profile="dataset-xy-yaw"
    )
    result = []
    counter = count_replay_calls() if count_calls else contextlib.nullcontext(None)
    with counter as counts:
        started = time.monotonic()
        for episode in episodes:
            if not 0 <= episode < len(loaded):
                raise ValueError(f"快照局部 episode 索引无效：{episode}")
            args.episode = episode
            log = io.StringIO()
            episode_started = time.monotonic()
            try:
                with snapshot_loader(loaded), contextlib.redirect_stdout(log):
                    rollout = prepare_episode(args, runtime_provider=_runtime_for)
                success = True
                error = None
                count = len(rollout["qpos"])
            except (Exception, SystemExit) as exc:
                success, error, count = False, str(exc), None
            result.append({"local_index": episode, "source_index": entries[episode]["index"],
                           "frame_count": entries[episode]["length"], "solved_frames": count,
                           "passed": success, "error": error,
                           "cache_hit": "[replay] cache hit" in log.getvalue(),
                           "wall_seconds": time.monotonic() - episode_started})
        elapsed = time.monotonic() - started
    return {"episodes": result, "wall_seconds": elapsed, "counts": counts,
            "cache_mode": "existing_copy" if cache_source is not None else "cold",
            "rss_self_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_children_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}


def main():
    """解析独立快照、固定标定与需要测试的局部段号。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("calibration")
    parser.add_argument("output_dir")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--count-calls", action="store_true",
                        help="记录实际 IK 和模型初始化次数（默认关闭）")
    parser.add_argument("--existing-cache", type=Path,
                        help="将已有隔离缓存复制到输出目录后测试命中")
    args = parser.parse_args()
    result = benchmark_prepare(args.snapshot, args.calibration, args.output_dir, args.episodes,
                               count_calls=args.count_calls, existing_cache=args.existing_cache)
    target = Path(args.output_dir) / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
