"""暂存摆放搜索的完整轨迹，并在身份复核后提升至正式缓存。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from handumi.dataset.episode_solve_cache import EpisodeSolveCache, fingerprint


def scratch_key(episode: int, input_identity: str, pose7: np.ndarray) -> str:
    """按段号、原始输入身份和求解时的桌面变换标识临时结果。"""
    pose = np.asarray(pose7, dtype=np.float32).copy()
    pose[pose == 0] = 0.0  # JSON 中正负零不同，数值含义完全相同。
    return fingerprint({"episode": episode, "input_identity": input_identity,
                        "pose7": pose.tolist()})


def store_scratch(directory: Path, episode: int, input_identity: str,
                  pose7: np.ndarray, rollout: dict[str, np.ndarray]) -> None:
    """原子保存通过严格验收的完整轨迹，不产生正式缓存身份。"""
    EpisodeSolveCache(directory).store(scratch_key(episode, input_identity, pose7), rollout)


def promote_winner(args, reference, selection, candidate, report: dict,
                   directory: Path) -> int:
    """复核胜出轨迹和实际输入后写入正式缓存，返回成功提升段数。"""
    from handumi.calibration.deployment import load_deployment_calibration
    from handumi.dataset.episode_solve_cache import digest_file
    from handumi.robots.utils import pose_inv
    from handumi.scripts.replay import replay_in_sim as replay
    from handumi.scripts.replay.replay_prepare import (
        _finish_rollout,
        _valid_collision_summary,
        _valid_rollout,
        cache_directory,
        episode_identity,
    )

    def selection_matches() -> bool:
        """重新加载标定，确保生效的位姿和来源仍等于胜出后读取的值。"""
        if selection.path is None:
            return False
        try:
            current = load_deployment_calibration(
                selection.path, expected_robot=args.robot, profile="dataset-xy-yaw"
            )
        except (Exception, SystemExit):
            return False
        fields = ("robot", "scope", "source", "verified", "profile", "lab")
        return (all(getattr(current, field) == getattr(selection, field) for field in fields)
                and np.array_equal(current.pose7, selection.pose7))

    if not selection_matches():
        return 0
    selection_digest = digest_file(selection.path)
    # 标定加载会归一化四元数；只有位级相同才允许沿用已算目标。
    search_pose = pose_inv(candidate.pose)
    if not np.array_equal(search_pose, np.asarray(selection.pose7, dtype=np.float32)):
        print("[replay] placement cache skipped: saved calibration changed the pose")
        return 0
    scratch = EpisodeSolveCache(directory)
    target = EpisodeSolveCache(cache_directory(args))
    identities = report["reference_episode_identities"]
    promoted = 0
    runtime = checker = None
    for episode, source_identity in enumerate(identities):
        key = scratch_key(episode, source_identity, search_pose)
        rollout, failure = scratch.load(key)
        if failure is not None or rollout is None:
            continue
        trial = argparse.Namespace(**vars(args))
        trial.episode = episode
        trial.base_placement = "fixed"
        trial.deployment_selection = reference
        trial.refresh_base_placement = False
        try:
            loaded = replay.load_episode_states(trial)
            if episode_identity(trial, loaded)[0] != source_identity:
                continue
            if not _valid_rollout(rollout) or not _valid_collision_summary(trial, rollout):
                continue
            recorded_pose = np.asarray(rollout["robot_from_table_pose7"])
            if recorded_pose.shape != (1, 7) or not np.array_equal(recorded_pose[0], search_pose):
                continue
            final = argparse.Namespace(**vars(args))
            final.episode = episode
            final.base_placement = "dataset-xy-yaw"
            final.deployment_selection = selection
            final.strict_ik = True
            # 正式回放的显示与来源必须指向最终固定标定。
            rollout.update(replay._deployment_npz_entries(selection))
            rollout["base_placement_policy"] = np.asarray(["dataset-xy-yaw"])
            rollout["base_placement_scope"] = np.asarray([selection.scope])
            rollout["base_placement_verified"] = np.asarray([selection.verified])
            rollout["base_placement_diagnostics_json"] = np.asarray([""])
            final_key, _ = episode_identity(final, loaded)
            validation_context = {}
            _finish_rollout(final, rollout, runtime=runtime, checker=checker,
                            validation_context=validation_context)
            runtime = validation_context.get("runtime")
            checker = validation_context.get("checker")
            # 防止读入与验收期间源数据、配置或最终标定发生变化。
            fresh = replay.load_episode_states(final)
            if (not selection_matches() or digest_file(selection.path) != selection_digest
                    or episode_identity(trial, fresh)[0] != source_identity
                    or episode_identity(final, fresh)[0] != final_key):
                continue
            target.store(final_key, rollout)
            promoted += 1
        except (Exception, SystemExit) as exc:
            print(f"[replay] placement cache promotion skipped episode={episode}: {exc}")
    print(f"[replay] promoted {promoted} validated placement rollout(s) to cache")
    return promoted
