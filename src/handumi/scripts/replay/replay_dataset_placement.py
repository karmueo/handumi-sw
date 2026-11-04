"""协调数据集共享底座搜索、固定标定持久化和批次评估报告。"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import os
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from handumi.calibration.deployment import (
    load_deployment_calibration,
    resolve_deployment_calibration,
    simulation_calibration_path,
)
from handumi.dataset import DatasetRef, ensure_metadata
from handumi.dataset.episode_solve_cache import array_digest
from handumi.retargeting.base_placement import first_frame_candidates
from handumi.retargeting.dataset_placement import (
    EpisodeTargets,
    choose_finalists,
    generate_candidates,
    rank_result,
    representative_frames,
)
from handumi.robots.utils import pose7_to_mat, pose_inv, pose_mul

ALGORITHM_VERSION = "dataset-xy-yaw-v1"
_EXPECTED_FAILURES = (
    "Unable to prepare replay start pose:",
    "Self-collision validation failed:",
    "Joint constraint exceeded:",
    "IK fidelity threshold exceeded",
    "Invalid non-finite ",
    "Invalid zero ",
)


class PrefixObserver:
    """接收每帧最终精度与碰撞认证，记录严格连续前缀。"""

    def __init__(self):
        """初始化已通过帧数和第一个失败帧。"""
        self.prefix = 0
        self.failed_frame = None
        self.current_frame = None
        self.total_frames = None
        self.identity_key = None

    def configure_frames(self, count: int, first_frame: int = 0) -> None:
        """在首帧初始化及缓存查询前记录完整帧数和待求首帧。"""
        self.total_frames = count
        if self.current_frame is None:
            self.current_frame = first_frame

    def start_frame(self, frame: int, total: int) -> None:
        """记录将要验证的帧，以便硬失败时指出对应位置。"""
        self.current_frame = frame
        self.total_frames = total

    def frame_checked(self, frame: int, within: bool) -> None:
        """首个精度超差后冻结前缀，后续帧不增加计数。"""
        if self.failed_frame is not None:
            return
        if within:
            self.prefix += 1
        else:
            self.failed_frame = frame

    def rollout_checked(self, rollout: dict, args) -> None:
        """从完整结果重建前缀，支持缓存命中及完成后的严格门禁。"""
        pos = np.maximum(rollout["left_pos_error_m"], rollout["right_pos_error_m"])
        rot = np.maximum(rollout["left_rot_error_deg"], rollout["right_rot_error_deg"])
        bad = np.flatnonzero((pos > args.max_ik_position_error_m)
                             | (rot > args.max_ik_rotation_error_deg))
        self.prefix = int(bad[0]) if len(bad) else len(pos)
        self.total_frames = len(pos)
        self.failed_frame = int(rollout["frame_indices"][bad[0]]) if len(bad) else None


def _write_atomic(path: Path, content: str) -> None:
    """将 UTF-8 内容写入同目录临时文件并原子替换目标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _json_write(path: Path, value: dict) -> None:
    """以可读、无非有限数值的 JSON 原子保存报告。"""
    _write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _directory(args) -> Path:
    """返回数据集私有的共享摆放目录。"""
    return Path(args.dataset_root) / ".handumi" / "base_placement"


def _reference_calibration(args):
    """解析参考摆放并拒绝修改实体标定；RM75B 缺省仿真摆放为单位阵。"""
    if args.deployment_profile == "sim" and not simulation_calibration_path(args.robot).is_file():
        from handumi.calibration.deployment import DeploymentCalibration

        return DeploymentCalibration(
            robot=args.robot, pose7=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            path=None, scope="assumed", source="identity_assumed_table_equals_base",
            verified=False, profile="identity", lab=None,
        )
    result = resolve_deployment_calibration(
        args.robot, profile=args.deployment_profile, rig_config=args.rig_config,
        identity_fallback=True,
    )
    if result.scope == "physical":
        raise SystemExit("dataset-xy-yaw cannot modify a physical table calibration; select --deployment-profile sim.")
    return result


def _episode_contexts(args, reference, progress=None):
    """复用正式回放初始化与标定解析，读取全部段的桌面 TCP 目标。"""
    from handumi.scripts.replay import replay_in_sim as replay
    from handumi.scripts.replay.diagnose_placement import arm_geometry, wrist_envelope
    from handumi.scripts.replay.replay_prepare import episode_identity

    if progress is not None:
        progress.update("准备输入", 0, 1, detail="读取数据集元信息")
    info = ensure_metadata(DatasetRef.from_repo_id(args.repo_id, root=args.dataset_root, revision=args.revision))
    count = int(info.get("total_episodes", 0))
    if count <= 0:
        raise SystemExit("Dataset contains no episodes.")
    if not 0 <= args.episode < count:
        raise SystemExit(f"Episode {args.episode} is out of range 0..{count - 1}.")
    contexts = []
    runtime = checker = None
    geometry = None
    for episode in range(count):
        if progress is not None:
            progress.update("准备输入", episode, count, detail=f"episode {episode + 1}/{count}")
        trial = argparse.Namespace(**vars(args))
        trial.episode = episode
        trial.base_placement = "fixed"
        trial.refresh_base_placement = False
        trial.deployment_selection = reference
        trial._placement_context = {}
        trial._dataset_placement_context_only = True
        if progress is not None:
            def input_progress(stage, **_details):
                """将当前 episode 模型初始化阶段转发给独立显示器。"""
                progress.update("准备输入", episode, count,
                                detail=f"episode {episode + 1}/{count} · {stage}")

            trial._replay_progress = input_progress
        trial._episode_loaded = replay.load_episode_states(trial)
        if runtime is not None:
            trial._episode_runtime = runtime
            trial._episode_collision_checker = checker
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                replay._solve_episode_fixed(trial)
        except replay._PlacementContextReady:
            pass
        else:
            raise RuntimeError("Dataset placement context initialization did not stop.")
        context = trial._placement_context
        context["loaded"] = trial._episode_loaded
        context["input_identity"], _ = episode_identity(trial, trial._episode_loaded)
        runtime = context["runtime"]
        checker = context["collision_checker"]
        if geometry is None:
            geometry = arm_geometry(runtime.urdf_path)
        poses = np.asarray([
            replay.raw_state_pose7_pair(state)[1] for state in context["states_for_retarget"]
        ], dtype=np.float32)
        wrist, _, outer, inner, _ = wrist_envelope(
            pose7_to_mat(poses), context["tool"].matrices["right"], geometry,
            args.max_ik_position_error_m, args.max_ik_rotation_error_deg,
        )
        context["targets_for_search"] = EpisodeTargets(poses, wrist, inner, outer)
        context["samples"] = representative_frames(
            poses, args.max_ik_position_error_m, args.max_ik_rotation_error_deg,
        )
        contexts.append(context)
        if progress is not None:
            progress.update("准备输入", episode + 1, count,
                            detail=f"episode {episode + 1}/{count}")
    return contexts, geometry


def _candidate_calibration(reference, pose):
    """构造仅用于内存试算的桌面到机器人标定。"""
    return replace(reference, pose7=pose_inv(pose), path=None, scope="simulation",
                   source="dataset_xy_yaw_search", verified=False,
                   profile="dataset-xy-yaw", lab=None)


def _sample_evaluator(args, contexts, reference, *, frame_progress=None):
    """建立共享模型，返回不受帧间限速约束的代表帧评分器。"""
    runtime = contexts[0]["runtime"]
    checker = contexts[0]["collision_checker"]
    cfg = runtime.config.ik_weights
    if runtime.config.replay_ik_weights:
        cfg = replace(cfg, **{f"{key}_weight": value for key, value in runtime.config.replay_ik_weights.items()})
    cfg = replace(cfg, max_joint_delta=None, self_collision_weight=0.0)
    solver = runtime.solver_cls(config=cfg, locked_joint_indices=contexts[0]["locked"])
    if checker is not None:
        solver.robot_collision = checker.model
    lower = np.asarray(runtime.robot.joints.lower_limits)
    upper = np.asarray(runtime.robot.joints.upper_limits)

    def evaluate(candidate):
        """逐代表帧从 HOME 求解并检查精度、限位及自碰撞。"""
        robot_from_table = pose_inv(candidate.pose)
        full = passed_count = 0
        for episode_index, context in enumerate(contexts):
            passed_episode = True
            tool = context["tool"]
            for sample_index, frame in enumerate(context["samples"]):
                if frame_progress is not None:
                    frame_progress(episode_index, sample_index, len(context["samples"]))
                target = pose_mul(robot_from_table, context["targets_for_search"].poses[frame])
                target_ee = tool.target_ee("right", target)
                q = runtime.config.home_q.astype(np.float32).copy()
                passed = False
                for _ in range(4):
                    q = solver.ik(q, right_pose=(target_ee[:3], target_ee[3:]))
                    openings = context["gripper_openings"]
                    if openings is not None:
                        opening = openings[frame]
                        runtime.set_finger_positions(q, {"left": float(opening[0]), "right": float(opening[1])})
                    _, actual = solver.fk_pose7(q)
                    actual = tool.achieved_tcp("right", actual)
                    position = np.linalg.norm(actual[:3] - target[:3])
                    rotation = np.rad2deg((Rotation.from_quat(target[3:]).inv() * Rotation.from_quat(actual[3:])).magnitude())
                    if (np.isfinite(q).all() and np.isfinite([position, rotation]).all()
                            and np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)
                            and position <= args.max_ik_position_error_m
                            and rotation <= args.max_ik_rotation_error_deg
                            and (checker is None or checker.check_pose(q).safe)):
                        passed = True
                        break
                passed_episode &= passed
                passed_count += passed
                if frame_progress is not None:
                    frame_progress(episode_index, sample_index + 1, len(context["samples"]))
            full += passed_episode
        return int(full), int(passed_count)

    return evaluate


def _full_episode(args, context, episode: int, selection, *, progress_callback=None) -> dict:
    """用正式非严格求解和严格终验评分一段，保留硬失败前缀。"""
    from handumi.scripts.replay import replay_in_sim as replay
    from handumi.scripts.replay.replay_prepare import _finish_rollout

    observer = PrefixObserver()
    trial = argparse.Namespace(**vars(args))
    trial.episode = episode
    trial.base_placement = "fixed"
    trial.refresh_base_placement = False
    trial.deployment_selection = selection
    trial._episode_loaded = context["loaded"]
    trial._episode_runtime = context["runtime"]
    trial._episode_collision_checker = context["collision_checker"]
    trial._placement_score = observer
    if progress_callback is not None:
        trial._replay_progress = progress_callback
    trial.strict_ik = False
    total = len(context["targets_for_search"].poses)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rollout = replay.solve_episode(trial)
            trial.strict_ik = True
            _finish_rollout(trial, rollout, runtime=context["runtime"],
                            checker=context["collision_checker"])
    except SystemExit as exc:
        message = str(exc)
        if not message.startswith(_EXPECTED_FAILURES):
            raise
        return {"episode": episode, "passed": False, "prefix_frames": observer.prefix,
                "total_frames": total, "first_failed_frame": (observer.failed_frame if observer.failed_frame is not None else observer.current_frame),
                "error": message[:4000]}
    scratch_dir = getattr(args, "_placement_scratch_dir", None)
    input_identity = context.get("input_identity")
    if scratch_dir is not None and input_identity is not None:
        from handumi.scripts.replay.replay_placement_cache import store_scratch

        try:
            store_scratch(Path(scratch_dir), episode, input_identity, selection.pose7, rollout)
        except (OSError, ValueError) as exc:
            print(f"[replay] placement scratch write skipped episode={episode}: {exc}")
    return {"episode": episode, "passed": True, "prefix_frames": observer.prefix,
            "total_frames": total, "first_failed_frame": None,
            "max_position_error_m": float(max(np.max(rollout["left_pos_error_m"]), np.max(rollout["right_pos_error_m"]))),
            "max_rotation_error_deg": float(max(np.max(rollout["left_rot_error_deg"]), np.max(rollout["right_rot_error_deg"]))) }


def _estimate(args, reference):
    """执行有界两阶段搜索，返回赢家、可序列化诊断及数据摘要。"""
    from handumi.scripts.replay.replay_progress import DatasetPlacementProgress

    with DatasetPlacementProgress(getattr(args, "progress", "auto")) as progress:
        return _estimate_with_progress(args, reference, progress)


def _estimate_parallel_with_progress(args, reference, progress, contexts, geometry,
                                     started, stages, workers):
    """并行评分独立任务，保持候选顺序及原有排序与严格验收语义。"""
    from handumi.scripts.replay.replay_placement_pool import PlacementPool

    reference_pose = pose_inv(reference.pose7).astype(float)
    episodes = [context["targets_for_search"] for context in contexts]
    count = len(contexts)
    stage_started = time.monotonic()
    with PlacementPool(args, contexts, reference, workers, progress) as pool:
        first_results = pool.first_frames(reference_pose)
        call_index = 0

        def first_xy(_base, episode):
            """按生成器的固定朝向顺序读取已完成的首帧结果。"""
            nonlocal call_index
            yaw = call_index // count
            call_index += 1
            return first_results[yaw][episode]

        candidates = generate_candidates(reference_pose, episodes, geometry[0], first_xy)
        stages["candidate_generation"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        sample_results = iter(pool.sample_scores(candidates))
        finalists, sample_records = choose_finalists(
            candidates, reference_pose, args.placement_search_candidates,
            lambda _candidate: next(sample_results),
        )
        stages["sample_evaluation"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        full_results = pool.full_scores(finalists)
        evaluations = []
        winner = None
        best_key = None
        for candidate, results in zip(finalists, full_results, strict=True):
            key = rank_result(candidate, reference_pose, results)
            evaluations.append({"pose7": candidate.pose.tolist(),
                                "yaw_index": candidate.yaw_index,
                                "passed_episodes": -key[0], "prefix_frames": -key[1],
                                "episodes": results})
            if best_key is None or key < best_key:
                best_key, winner = key, candidate
        stages["full_evaluation"] = time.monotonic() - stage_started
    total_frames = sum(len(episode.poses) for episode in episodes)
    report = {
        "algorithm": ALGORITHM_VERSION, "robot": args.robot,
        "episode_count": count, "total_frames": total_frames,
        "episode_state_digests": [array_digest(context["loaded"][0]) for context in contexts],
        "reference_episode_identities": [context["input_identity"] for context in contexts],
        "max_position_error_m": args.max_ik_position_error_m,
        "max_rotation_error_deg": args.max_ik_rotation_error_deg,
        "initial_position_tolerance_m": args.initial_position_tolerance_m,
        "reference_table_from_robot_pose7": reference_pose.tolist(),
        "sample_candidates": sample_records, "candidates": evaluations,
        "candidate_budget": args.placement_search_candidates,
        "elapsed_seconds": time.monotonic() - started, "stage_seconds": stages,
        "passed_episodes": -best_key[0] if best_key is not None else 0,
        "prefix_frames": -best_key[1] if best_key is not None else 0,
        "verified_prefix_rate": (-best_key[1] / total_frames) if best_key is not None and total_frames else 0.0,
        "table_from_robot_pose7": winner.pose.tolist() if winner is not None else None,
    }
    return winner, report


def _estimate_with_progress(args, reference, progress):
    """在独立进度显示生命周期内运行搜索，不修改原始参数对象。"""
    from handumi.robots.utils import pose_inv

    started = time.monotonic()
    contexts, geometry = _episode_contexts(args, reference, progress)
    stages = {"input_preparation": time.monotonic() - started}
    workers = getattr(args, "placement_workers", None)
    if workers is None:
        workers = getattr(args, "prepare_workers", 8)
    if workers > 1:
        return _estimate_parallel_with_progress(
            args, reference, progress, contexts, geometry, started, stages, workers,
        )
    reference_pose = pose_inv(reference.pose7).astype(float)
    episodes = [context["targets_for_search"] for context in contexts]

    generated = 0
    generation_total = 8 * len(contexts)
    progress.update("生成候选", 0, generation_total, detail="准备水平角度与首帧候选")

    def first_xy(base, index):
        """联合首帧 IK 与 XY 生成候选，舍弃仅用于生成的关节种子。"""
        nonlocal generated
        progress.update("生成候选", generated, generation_total,
                        detail=f"朝向 {generated // len(contexts) + 1}/8 · episode {index + 1}/{len(contexts)}")
        context = contexts[index]
        candidates = [candidate.xy for candidate in first_frame_candidates(
            context["runtime"], context["tool"], context["targets"], base,
            context["locked"], max_seeds=1,
        )]
        generated += 1
        progress.update("生成候选", generated, generation_total,
                        detail=f"已处理 {generated}/{generation_total} 组首帧")
        return candidates

    stage_started = time.monotonic()
    candidates = generate_candidates(reference_pose, episodes, geometry[0], first_xy)
    stages["candidate_generation"] = time.monotonic() - stage_started
    sample_index = 0

    def sample_frame(episode, current, total):
        """报告当前候选、episode 和已评估代表帧。"""
        progress.update("代表帧预筛", sample_index - 1, len(candidates),
                        detail=f"候选 {sample_index}/{len(candidates)} · episode {episode + 1}/{len(contexts)}",
                        frame_current=current, frame_total=total)

    evaluate_sample = _sample_evaluator(args, contexts, reference, frame_progress=sample_frame)

    def sampled(candidate):
        """显示代表帧预筛进度并返回评分。"""
        nonlocal sample_index
        sample_index += 1
        progress.update("代表帧预筛", sample_index - 1, len(candidates),
                        detail=f"候选 {sample_index}/{len(candidates)}")
        result = evaluate_sample(candidate)
        progress.update("代表帧预筛", sample_index, len(candidates),
                        detail=f"候选 {sample_index}/{len(candidates)}")
        return result

    stage_started = time.monotonic()
    finalists, sample_records = choose_finalists(
        candidates, reference_pose, args.placement_search_candidates, sampled,
    )
    stages["sample_evaluation"] = time.monotonic() - stage_started
    evaluations = []
    winner = None
    best_key = None
    stage_started = time.monotonic()
    for index, candidate in enumerate(finalists):
        selection = _candidate_calibration(reference, candidate.pose)
        results = []
        for episode, context in enumerate(contexts):
            completed = index * len(contexts) + episode
            total = len(finalists) * len(contexts)
            detail = f"候选 {index + 1}/{len(finalists)} · episode {episode + 1}/{len(contexts)}"
            progress.update("完整轨迹验证", completed, total, detail=detail)

            def solve_progress(stage, **details):
                """显示正式求解当前阶段及真实已完成帧数。"""
                progress.update("完整轨迹验证", completed, total,
                                detail=f"{detail} · {stage}",
                                frame_current=details.get("current"),
                                frame_total=details.get("total"))

            results.append(_full_episode(args, context, episode, selection,
                                         progress_callback=solve_progress))
            progress.update("完整轨迹验证", completed + 1, total, detail=detail)
        key = rank_result(candidate, reference_pose, results)
        evaluations.append({"pose7": candidate.pose.tolist(), "yaw_index": candidate.yaw_index,
                            "passed_episodes": -key[0], "prefix_frames": -key[1],
                            "episodes": results})
        if best_key is None or key < best_key:
            best_key = key
            winner = candidate
    stages["full_evaluation"] = time.monotonic() - stage_started
    total_frames = sum(len(episode.poses) for episode in episodes)
    report = {
        "algorithm": ALGORITHM_VERSION, "robot": args.robot,
        "episode_count": len(episodes), "total_frames": total_frames,
        "episode_state_digests": [array_digest(context["loaded"][0]) for context in contexts],
        "reference_episode_identities": [context["input_identity"] for context in contexts],
        "max_position_error_m": args.max_ik_position_error_m,
        "max_rotation_error_deg": args.max_ik_rotation_error_deg,
        "initial_position_tolerance_m": args.initial_position_tolerance_m,
        "reference_table_from_robot_pose7": reference_pose.tolist(),
        "sample_candidates": sample_records, "candidates": evaluations,
        "candidate_budget": args.placement_search_candidates,
        "elapsed_seconds": time.monotonic() - started,
        "stage_seconds": stages,
        "passed_episodes": -best_key[0] if best_key is not None else 0,
        "prefix_frames": -best_key[1] if best_key is not None else 0,
        "verified_prefix_rate": (-best_key[1] / total_frames) if best_key is not None and total_frames else 0.0,
        "table_from_robot_pose7": winner.pose.tolist() if winner is not None else None,
    }
    return winner, report


def resolve_dataset_placement(args):
    """在排他锁中加载已固定标定，或完整搜索后原子固化新结果。"""
    directory = _directory(args)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.robot}.yaml"
    with (directory / f"{args.robot}.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        reference = _reference_calibration(args)
        if path.exists() and not args.refresh_base_placement:
            selection = load_deployment_calibration(path, expected_robot=args.robot, profile="dataset-xy-yaw")
            if selection.scope != "simulation":
                raise SystemExit(f"Dataset placement {path} must have simulation scope.")
            print(f"[replay] reusing fixed dataset placement: {path}", flush=True)
            return selection
        # 结果只在本次搜索期间存在，退出时连同未选中的候选一起清理。
        cache_enabled = not (getattr(args, "no_cache", False) or getattr(args, "refresh_cache", False))
        scratch_context = (tempfile.TemporaryDirectory(prefix=".search-", dir=directory)
                           if cache_enabled else contextlib.nullcontext(None))
        with scratch_context as scratch_dir:
            search_args = argparse.Namespace(**vars(args))
            if scratch_dir is not None:
                search_args._placement_scratch_dir = scratch_dir
            try:
                candidate, report = _estimate(search_args, reference)
            except (Exception, SystemExit) as exc:
                failure_path = directory / f"{args.robot}.latest.json"
                _json_write(failure_path, {"status": "search_error", "algorithm": ALGORITHM_VERSION,
                                          "robot": args.robot, "error": str(exc)})
                raise
            generation = uuid.uuid4().hex
            report["generation"] = generation
            report_path = directory / f"{args.robot}.{generation}.json"
            _json_write(report_path, report)
            if candidate is None or report["passed_episodes"] < 1:
                _json_write(directory / f"{args.robot}.latest.json", {"status": "search_failed", "search_report": report_path.name, **report})
                raise SystemExit(f"Dataset placement search found no fully passing episode; see {report_path}.")
            robot_from_table = pose_inv(candidate.pose)
            data = {
                "schema_version": 1, "kind": "handumi_robot_table_calibration",
                "robot": args.robot, "scope": "simulation", "verified": False,
                "source": "dataset_xy_yaw_search", "generation": generation,
                "search_report": report_path.name,
                "calibration": {"robot_from_table": {
                    "position": robot_from_table[:3].tolist(),
                    "quaternion": robot_from_table[3:].tolist(),
                }},
            }
            _write_atomic(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
            print(f"[replay] fixed dataset placement: {path} ({report['passed_episodes']}/{report['episode_count']} episodes passed)", flush=True)
            selection = load_deployment_calibration(path, expected_robot=args.robot, profile="dataset-xy-yaw")
            if scratch_dir is not None:
                from handumi.scripts.replay.replay_placement_cache import promote_winner

                try:
                    promote_winner(args, reference, selection, candidate, report, Path(scratch_dir))
                except (Exception, SystemExit) as exc:
                    print(f"[replay] placement cache promotion skipped: {exc}")
            return selection


def write_evaluation_report(args, failures: dict[int, str], *, completed=None) -> Path:
    """记录固定摆放下当前批次结果；此报告不改变标定文件和缓存身份。"""
    directory = _directory(args)
    info = ensure_metadata(DatasetRef.from_repo_id(args.repo_id, root=args.dataset_root, revision=args.revision))
    count = int(info.get("total_episodes", 0))
    metrics = getattr(args, "_dataset_evaluation_results", {})
    done = set(range(count)) if completed is None else set(completed)
    episodes = []
    for episode in sorted(done):
        if episode in failures:
            episodes.append({"episode": episode, "passed": False, "error": str(failures[episode]), **metrics.get(episode, {})})
        else:
            episodes.append({"episode": episode, "passed": True, **metrics.get(episode, {})})
    known_frames = all(entry.get("total_frames") is not None for entry in episodes)
    total_frames = sum(entry["total_frames"] for entry in episodes) if known_frames else None
    prefix_frames = sum(entry.get("prefix_frames", 0) for entry in episodes)
    result = {"algorithm": ALGORITHM_VERSION, "robot": args.robot,
              "placement": str(_directory(args) / f"{args.robot}.yaml"),
              "robot_from_table_pose7": np.asarray(args.deployment_selection.pose7).tolist(),
              "dataset_episode_count": count, "evaluated_episodes": len(episodes),
              "passed_episodes": sum(entry["passed"] for entry in episodes),
              "failed_episodes": len(failures), "episodes": episodes,
              "total_frames": total_frames, "prefix_frames": prefix_frames,
              "verified_prefix_rate": (prefix_frames / total_frames if total_frames else None),
              "stage_seconds": getattr(args, "_dataset_stage_seconds", {}),
              "wall_seconds": getattr(args, "_dataset_wall_seconds", None),
              "max_position_error_m": args.max_ik_position_error_m,
              "max_rotation_error_deg": args.max_ik_rotation_error_deg}
    path = directory / f"{args.robot}.latest.json"
    _json_write(path, result)
    print(f"[replay] dataset placement evaluation: {path}", flush=True)
    return path
