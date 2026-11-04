"""解析逐段求解依赖，复用完整回放缓存并在命中后执行严格验收。"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from handumi.dataset.episode_solve_cache import (
    EpisodeSolveCache,
    array_digest,
    digest_file,
    fingerprint,
)
from handumi.scripts.replay.replay_progress import report

# 变更求解语义或缓存身份规则时提高版本；桌面和机器人几何仍按内容摘要。
SOLVE_CACHE_ALGORITHM = "replay-complete-v3-mesh-collision"
_SOURCE_META_KEYS = (
    "tracking_schema",
    "capture_schema",
    "state_semantics",
    "tracking_workspace",
    "recording_device",
    "active_sides",
    "target_robot",
    "controller_tcp_calibration",
    "sources",
)
_SETTING_KEYS = (
    "source",
    "base_placement",
    "compose_source",
    "translation_scale",
    "absolute_orientation",
    "initial_solve_iterations",
    "initial_position_tolerance_m",
    "max_ik_position_error_m",
    "max_ik_rotation_error_deg",
    "only_manipulation",
    "start_frame",
    "max_frames",
    "stride",
    "raw_controller_debug",
)
_REQUIRED_ARRAYS = (
    "qpos",
    "frame_indices",
    "fps",
    "gripper_normalized",
    "retarget_mode",
    "target_left_pose7_robot_world",
    "target_right_pose7_robot_world",
    "achieved_left_pose7_robot_world",
    "achieved_right_pose7_robot_world",
    "left_pos_error_m",
    "right_pos_error_m",
    "left_rot_error_deg",
    "right_rot_error_deg",
    "raw_left_pose7_ground_truth",
    "raw_right_pose7_ground_truth",
    "raw_left_controller_pose7",
    "raw_right_controller_pose7",
    "calibrated_left_tcp_pose7",
    "calibrated_right_tcp_pose7",
    "robot_from_table_pose7",
    "table_from_robot_pose7",
    "base_placement_policy",
    "base_placement_scope",
    "base_placement_verified",
    "base_placement_diagnostics_json",
    "controller_device",
    "controller_tcp_calibration",
    "robot_tcp_calibration_json",
    "initial_solve_iterations",
    "initial_max_position_error_m",
    "initial_max_rotation_error_deg",
    "gripper_source",
    "tracking_workspace",
    "retarget_mode_requested",
)
_FAILURE_PREFIXES = (
    "Unable to prepare replay start pose:",
    "Unable to find strict episode XY placement:",
)


def cache_directory(args: argparse.Namespace) -> Path:
    """选择显式缓存目录或数据集私有的逐段缓存目录。"""
    explicit = getattr(args, "cache_dir", None)
    return (
        Path(explicit).expanduser()
        if explicit is not None
        else Path(args.dataset_root) / ".handumi" / "solve_cache"
    )


def _source_calibration(args, info, device):
    """按回放同一优先级解析实际控制器到 TCP 标定。"""
    from handumi.robots.registry import load_robot_config
    from handumi.scripts.replay import replay_in_sim as replay

    if args.raw_controller_debug:
        return None
    source_robot = replay._source_robot_from_metadata(info, fallback=args.robot)
    try:
        source_config = load_robot_config(source_robot)
    except (TypeError, ValueError):
        source_config = None
    selection = replay._resolved_tcp_calibration(
        args,
        info,
        robot=source_robot,
        controller_device=device,
        configured_path=(
            source_config.controller_tcp_calibrations.get(device)
            if source_config is not None
            else None
        ),
        configured_gripper=source_config.handumi_gripper
        if source_config is not None
        else None,
        configured_mount=source_config.handumi_controller_mount
        if source_config is not None
        else None,
    )
    return {
        "source": selection.source,
        "source_robot": selection.source_robot,
        "source_gripper": selection.source_gripper,
        "tracking_device": selection.tracking_device,
        "controller_mount": selection.controller_mount,
        "trusted": selection.trusted_dataset_snapshot,
        "left": selection.calibration.left.tolist(),
        "right": selection.calibration.right.tolist(),
    }


def episode_identity(
    args: argparse.Namespace, loaded: tuple
) -> tuple[str, dict[str, Any]]:
    """根据单段实际输入和生效标定生成缓存键，不依赖其他段或总数。"""
    from handumi.calibration.deployment import resolve_deployment_calibration
    from handumi.calibration.robot_tcp import load_robot_tcp_calibration
    from handumi.dataset import handumi_metadata
    from handumi.retargeting.base_placement import ALGORITHM_VERSION
    from handumi.robots.registry import CONFIG_DIR, load_robot_config
    from handumi.scripts.replay import replay_in_sim as replay

    replay.resolve_replay_options(args)
    states, fps, info, openings = loaded
    metadata = handumi_metadata(info)
    config = load_robot_config(args.robot)
    mode = replay._resolved_retarget_mode(args, info)
    device = replay._resolved_controller_device(args, info)
    source = _source_calibration(args, info, device)
    deployment = None
    tool = None
    if mode in {"absolute-table", "absolute-tcp"}:
        from handumi.scripts.replay.replay_deployment import estimate_dataset_deployment

        selection = getattr(
            args, "deployment_selection", None
        ) or resolve_deployment_calibration(
            args.robot,
            explicit_path=args.deployment_calibration,
            profile=args.deployment_profile,
            rig_config=args.rig_config,
            identity_fallback=mode == "absolute-tcp",
            simulation_fallback=lambda: estimate_dataset_deployment(args),
        )
        deployment = {
            "pose7": np.asarray(selection.pose7).tolist(),
            "scope": selection.scope,
            "source": selection.source,
            "verified": selection.verified,
            "profile": selection.profile,
            "lab": selection.lab,
            "sha256": digest_file(selection.path),
            "path": str(selection.path.resolve())
            if selection.path is not None
            else None,
        }
    if mode == "absolute-tcp":
        robot_tool = load_robot_tcp_calibration(
            args.robot,
            config.arms,
            explicit_path=getattr(args, "robot_tcp_calibration", None),
        )
        tool = {
            "sha256": robot_tool.sha256,
            "path": str(robot_tool.path.resolve()),
            "arms": {
                side: {
                    "ee_link": robot_tool.ee_links[side],
                    "ee_to_tcp": matrix.tolist(),
                }
                for side, matrix in robot_tool.matrices.items()
            },
        }
    code_root = Path(__file__).resolve().parents[2]
    code_files = (
        "robots/kinematics.py",
        "robots/registry.py",
        "robots/mesh_surface_audit.py",
        "robots/self_collision.py",
        "robots/utils.py",
        "retargeting/handumi_to_robot.py",
        "retargeting/base_placement.py",
        "retargeting/dataset_placement.py",
        "scripts/replay/replay_in_sim.py",
        "scripts/replay/replay_prepare.py",
        "scripts/replay/replay_dataset_placement.py",
        "scripts/replay/replay_placement_cache.py",
        "scripts/replay/replay_placement_pool.py",
    )
    dependencies = {}
    for name in ("numpy", "scipy", "jax", "jaxlib", "pyroki", "jaxls", "yourdfpy", "trimesh", "rtree"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    collision_meshes = None
    if args.robot == "rm75b" and config.ik_weights.collision_enabled:
        import yourdfpy

        from handumi.robots.registry import yourdfpy_handler

        resolver = yourdfpy_handler(config.pkg_root)
        urdf = yourdfpy.URDF.load(str(config.urdf), filename_handler=resolver, load_meshes=False)
        names = sorted({
            collision.geometry.mesh.filename
            for link in urdf.robot.links for collision in link.collisions
            if collision.geometry is not None and collision.geometry.mesh is not None
        })
        collision_meshes = {name: digest_file(resolver(name)) for name in names}
        if not collision_meshes or any(value is None for value in collision_meshes.values()):
            raise SystemExit("RM75B collision mesh missing; cannot validate replay cache.")
    settings = {name: getattr(args, name, None) for name in _SETTING_KEYS}
    if args.robot == "rm75b" and config.ik_weights.collision_enabled:
        # 接近段碰撞会改变摆放搜索结果及失败原因，必须区分缓存身份。
        settings["approach_seconds"] = args.approach_seconds
    payload = {
        "algorithm": SOLVE_CACHE_ALGORITHM,
        "collision_meshes": collision_meshes,
        "base_algorithm": ALGORITHM_VERSION,
        "code": {name: digest_file(code_root / name) for name in code_files},
        "dependencies": dependencies,
        "robot": args.robot,
        "robot_config_sha256": digest_file(CONFIG_DIR / f"{args.robot}.yaml"),
        "urdf_sha256": digest_file(config.urdf),
        "active_arms": {
            side: {"ee_link": arm.ee_link, "joint_names": list(arm.joint_names)}
            for side, arm in config.arms.items()
        },
        "gripper_max_width_m": float(
            config.gripper_max_width_m
            if args.gripper_max_width_m is None
            else args.gripper_max_width_m
        ),
        "mode": mode,
        "device": device,
        "settings": settings,
        "metadata": {name: metadata.get(name) for name in _SOURCE_META_KEYS},
        "source_calibration": source,
        "deployment": deployment,
        "robot_tcp": tool,
        "fps": float(fps),
        "states": array_digest(states),
        "openings": array_digest(openings),
    }
    return fingerprint(payload), payload


def _valid_rollout(rollout: dict[str, np.ndarray]) -> bool:
    """检查完整可视化轨迹的一致性，拒绝缺字段、错帧或非有限数值。"""
    if any(name not in rollout for name in _REQUIRED_ARRAYS):
        return False
    qpos = np.asarray(rollout["qpos"])
    if qpos.ndim != 2 or not len(qpos) or not np.isfinite(qpos).all():
        return False
    n = len(qpos)
    frame_fields = (
        "frame_indices",
        "gripper_normalized",
        "target_left_pose7_robot_world",
        "target_right_pose7_robot_world",
        "achieved_left_pose7_robot_world",
        "achieved_right_pose7_robot_world",
        "left_pos_error_m",
        "right_pos_error_m",
        "left_rot_error_deg",
        "right_rot_error_deg",
    )
    for name in frame_fields:
        value = np.asarray(rollout[name])
        if value.ndim == 0 or len(value) != n or not np.isfinite(value).all():
            return False
    for side in ("left", "right"):
        for kind in ("target", "achieved"):
            if np.asarray(rollout[f"{kind}_{side}_pose7_robot_world"]).shape != (n, 7):
                return False
        for kind in ("pos_error_m", "rot_error_deg"):
            if np.asarray(rollout[f"{side}_{kind}"]).shape != (n,):
                return False
        for name in (
            f"raw_{side}_pose7_ground_truth",
            f"raw_{side}_controller_pose7",
            f"calibrated_{side}_tcp_pose7",
        ):
            value = np.asarray(rollout[name])
            if value.shape != (n, 7):
                return False
    fps = np.asarray(rollout["fps"])
    return fps.size == 1 and np.isfinite(fps).all() and float(fps.reshape(-1)[0]) > 0


def _strict_gate(
    args: argparse.Namespace, rollout: dict[str, np.ndarray], *, warn: bool = False
) -> None:
    """缓存命中或新解均按本次 strict 参数重新检查全部帧。"""
    if not getattr(args, "strict_ik", False) and not warn:
        return
    pos = np.maximum(rollout["left_pos_error_m"], rollout["right_pos_error_m"])
    rot = np.maximum(rollout["left_rot_error_deg"], rollout["right_rot_error_deg"])
    bad = np.flatnonzero(
        (pos > args.max_ik_position_error_m) | (rot > args.max_ik_rotation_error_deg)
    )
    if len(bad):
        frames = rollout["frame_indices"]
        detail = ", ".join(
            f"{int(frames[i])}: {float(pos[i]) * 1000:.2f}mm/{float(rot[i]):.2f}deg"
            for i in bad
        )
        message = (
            "IK fidelity threshold exceeded; frames (position/rotation): " + detail
        )
        if getattr(args, "strict_ik", False):
            raise SystemExit(message)
        print(f"[replay] warning: {message}")


def _valid_collision_summary(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> bool:
    """验证 RM75B 缓存的碰撞认证版本及关节序列摘要。"""
    if args.robot != "rm75b":
        return True
    from handumi.robots.registry import load_robot_config
    from handumi.robots.self_collision import (
        COLLISION_VALIDATION_VERSION,
        GRIPPER_BOX_VALIDATION_VERSION,
    )

    if not load_robot_config(args.robot).ik_weights.collision_enabled:
        return True
    try:
        version = (GRIPPER_BOX_VALIDATION_VERSION if args.retarget_mode == "absolute-tcp"
                   else COLLISION_VALIDATION_VERSION)
        return (str(rollout["self_collision_validation"].item()) == version
                and str(rollout["self_collision_qpos_sha256"].item())
                == array_digest(rollout["qpos"])["sha256"]
                and np.asarray(rollout["self_collision_min_links"]).shape == (2,)
                and np.isfinite(rollout["self_collision_min_sampled_m"]).all()
                and float(rollout["self_collision_min_sampled_m"].item())
                >= load_robot_config(args.robot).ik_weights.self_collision_margin)
    except (KeyError, ValueError, TypeError):
        return False


def _finish_rollout(
    args: argparse.Namespace, rollout: dict[str, np.ndarray], *, cached: bool = False,
    runtime=None, checker=None, validation_context: dict | None = None,
) -> dict[str, np.ndarray]:
    """按当前显示配置重建 approach，并对本次调用重新执行 strict 检查。"""
    from handumi.robots.registry import load_robot_config
    from handumi.scripts.replay import replay_in_sim as replay

    report(args, "严格验收")
    config = load_robot_config(args.robot)
    rollout["approach_qpos"] = replay.joint_approach_ramp(
        config.home_q,
        rollout["qpos"][0],
        frames=int(round(float(rollout["fps"][0]) * args.approach_seconds)),
    )
    rollout["retarget_mode_requested"] = np.asarray([str(args.retarget_mode)])
    if args.robot == "rm75b" and config.ik_weights.collision_enabled:
        from handumi.robots.registry import load_embodiment
        from handumi.robots.self_collision import (
            CollisionChecker,
            GripperBoxCollisionChecker,
            require_clear,
        )

        if not _valid_collision_summary(args, rollout):
            raise SystemExit("Missing or invalid RM75B self-collision validation summary.")
        if checker is None:
            if runtime is None:
                if str(rollout["retarget_mode"].item()) == "absolute-tcp":
                    runtime = load_embodiment(args.robot, collision_kind="gripper-box")
                else:
                    runtime = load_embodiment(args.robot)
            solver = runtime.solver_cls()
            urdf = runtime.load_urdf()
            if str(rollout["retarget_mode"].item()) == "absolute-tcp":
                checker = GripperBoxCollisionChecker(
                    runtime.robot, solver.robot_collision, urdf,
                    config.ik_weights.self_collision_margin,
                )
            else:
                checker = CollisionChecker(runtime.robot, solver.robot_collision,
                                           urdf, config.ik_weights.self_collision_margin)
        if validation_context is not None:
            validation_context.update(runtime=runtime, checker=checker)
        if args.approach_seconds > 0 or str(rollout["retarget_mode"].item()) != "absolute-tcp":
            require_clear(checker.check_pose(config.home_q), phase="home", frame="home")
        qpos = rollout["qpos"]
        require_clear(checker.check_pose(qpos[0]), phase="cached pose", frame=int(rollout["frame_indices"][0]))
        for index in range(1, len(qpos)):
            require_clear(checker.check_segment(qpos[index - 1], qpos[index]),
                          phase="cached motion", frame=int(rollout["frame_indices"][index]))
        if len(rollout["approach_qpos"]):
            path = [config.home_q, *rollout["approach_qpos"], qpos[0]]
            for index in range(1, len(path)):
                require_clear(checker.check_segment(path[index - 1], path[index]),
                              phase="approach", frame=index)
    observer = getattr(args, "_placement_score", None)
    if observer is not None:
        observer.rollout_checked(rollout, args)
    _strict_gate(args, rollout, warn=cached)
    return rollout


def _require_dataset_placement(args: argparse.Namespace) -> None:
    """拒绝缺失或非仿真的共享标定，避免缓存入口隐式选用其他摆放。"""
    if getattr(args, "base_placement", None) != "dataset-xy-yaw":
        return
    selection = getattr(args, "deployment_selection", None)
    if selection is None:
        raise SystemExit("Dataset-wide base placement has not been resolved.")
    if selection.scope != "simulation":
        raise SystemExit("Dataset-wide base placement requires a simulation calibration.")


def load_prepared_episode(args: argparse.Namespace) -> dict[str, np.ndarray] | None:
    """只读尝试逐段缓存；未命中时返回 None，供其他入口复用。"""
    from handumi.scripts.replay import replay_in_sim as replay

    replay.resolve_replay_options(args)
    _require_dataset_placement(args)
    if getattr(args, "no_cache", False) or getattr(args, "refresh_cache", False):
        return None
    loaded = replay.load_episode_states(args)
    key, _ = episode_identity(args, loaded)
    observer = getattr(args, "_placement_score", None)
    if observer is not None:
        observer.configure_frames(len(loaded[0]))
        observer.identity_key = key
    cached, failure = EpisodeSolveCache(cache_directory(args)).load(key)
    if failure is not None:
        report(args, "cache_hit")
        print(
            f"[replay] cached deterministic failure episode={args.episode} key={key[:12]}"
        )
        raise SystemExit(failure)
    if cached is None or not _valid_rollout(cached) or not _valid_collision_summary(args, cached):
        return None
    print(f"[replay] cache hit episode={args.episode} key={key[:12]}")
    return _finish_rollout(args, cached, cached=True)


def prepare_episode(
    args: argparse.Namespace, *, runtime_provider=None
) -> dict[str, np.ndarray]:
    """载入或求解单段；非严格结果也可共享，严格调用始终重新检查。"""
    from handumi.scripts.replay import replay_in_sim as replay

    report(args, "读取数据与缓存查找")
    replay.resolve_replay_options(args)
    _require_dataset_placement(args)
    if getattr(args, "no_cache", False):
        if getattr(args, "base_placement", None) == "dataset-xy-yaw":
            run_args = argparse.Namespace(**vars(args))
            run_args.strict_ik = False
            loaded = replay.load_episode_states(args)
            key, identity = episode_identity(args, loaded)
            observer = getattr(args, "_placement_score", None)
            if observer is not None:
                observer.configure_frames(len(loaded[0]))
                observer.identity_key = key
            run_args._episode_loaded = loaded
            if runtime_provider is not None:
                run_args._episode_runtime = runtime_provider(args, identity)
            rollout = replay.solve_episode(run_args)
            run_args.strict_ik = args.strict_ik
            return _finish_rollout(
                run_args, rollout, runtime=getattr(run_args, "_episode_runtime", None),
                checker=getattr(run_args, "_episode_collision_checker", None),
            )
        if runtime_provider is not None:
            loaded = replay.load_episode_states(args)
            _, identity = episode_identity(args, loaded)
            run_args = argparse.Namespace(**vars(args))
            run_args._episode_loaded = loaded
            run_args._episode_runtime = runtime_provider(args, identity)
            return replay.solve_episode(run_args)
        return replay.solve_episode(args)
    loaded = replay.load_episode_states(args)
    observer = getattr(args, "_placement_score", None)
    if observer is not None:
        observer.configure_frames(len(loaded[0]))
    key, identity = episode_identity(args, loaded)
    if observer is not None:
        observer.identity_key = key
    cache = EpisodeSolveCache(cache_directory(args))
    cached, failure = (
        (None, None) if getattr(args, "refresh_cache", False) else cache.load(key)
    )
    if failure is not None:
        report(args, "cache_hit")
        print(
            f"[replay] cached deterministic failure episode={args.episode} key={key[:12]}"
        )
        raise SystemExit(failure)
    was_hit = (cached is not None and _valid_rollout(cached)
               and _valid_collision_summary(args, cached))
    finish_runtime = finish_checker = None
    if was_hit:
        if runtime_provider is not None:
            finish_runtime = runtime_provider(args, identity)
        report(args, "cache_hit")
        print(f"[replay] cache hit episode={args.episode} key={key[:12]}")
        rollout = cached
    else:
        print(f"[replay] cache miss episode={args.episode} key={key[:12]}")
        run_args = argparse.Namespace(**vars(args))
        run_args._episode_loaded = loaded
        if runtime_provider is not None:
            report(args, "模型初始化")
            run_args._episode_runtime = runtime_provider(args, identity)
        run_args.strict_ik = False
        try:
            rollout = replay.solve_episode(run_args)
            finish_runtime = getattr(run_args, "_episode_runtime", None)
            finish_checker = getattr(run_args, "_episode_collision_checker", None)
        except SystemExit as exc:
            message = str(exc)
            if message.startswith(_FAILURE_PREFIXES):
                try:
                    fresh_key, _ = episode_identity(
                        args, replay.load_episode_states(args)
                    )
                    if fresh_key == key:
                        cache.store(key, failure=message)
                except (Exception, SystemExit) as error:
                    print(f"[replay] cache write skipped: {error}")
            raise
        if not _valid_rollout(rollout) or not _valid_collision_summary(args, rollout):
            raise SystemExit("Solver returned an invalid or incomplete replay rollout or collision validation summary.")
        # 求解结束后再次检查实际输入和标定，避免并发修改污染旧内容键。
        report(args, "缓存复验与写入")
        try:
            fresh_loaded = replay.load_episode_states(args)
            fresh_key, _ = episode_identity(args, fresh_loaded)
        except (Exception, SystemExit) as error:
            print(f"[replay] cache revalidation skipped: {error}")
            fresh_key = None
        if fresh_key == key:
            try:
                cache.store(key, rollout)
            except OSError as error:
                print(f"[replay] cache write skipped: {error}")
        else:
            print(
                "[replay] source or calibration changed during solve; result not cached"
            )
    return _finish_rollout(args, rollout, cached=was_hit,
                           runtime=finish_runtime, checker=finish_checker)
