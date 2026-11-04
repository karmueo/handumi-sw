"""按机器人筛选录制数据，校验实际标定并缓存已验收的关节轨迹。

``handumi validate`` and ``handumi dataset analyze`` grade the *recording*:
tracking dropouts, gripper freezes, duration outliers. None of that knows
whether a given robot can actually follow the demonstration. This module runs
every episode through the exact solver ``handumi convert`` uses and grades the
result, so low-quality episodes are removed before they become joint targets.

Findings use the ``handumi validate`` quality-report schema, so the output
drops straight into ``handumi dataset analyze --quality-report`` and from there
into ``handumi dataset curate --exclude``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk

from handumi.dataset.analysis import dataset_payload_manifest
from handumi.dataset.parallel import describe_jobs, map_episodes, resolve_jobs
from handumi.dataset.paths import portable_path, repo_path
from handumi.dataset.quality import EpisodeQualityReport, QualityFinding

SCREENING_SCHEMA_VERSION = 1
SCREENING_KIND = "handumi_retarget_screening"


def screening_report_path(root: str | Path, robot: str) -> Path:
    """Canonical per-embodiment screening report location."""
    return Path(root) / "meta" / f"handumi_screening_{robot}.json"


def _file_digest(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robot_fingerprint(
    robot: str, *, deployment_path: str | Path | None = None,
    robot_tcp_path: str | Path | None = None,
) -> dict[str, str]:
    """Hash everything that changes where the solver puts the robot.

    Screening numbers are only valid for one geometry: moving the arm bases or
    the table invalidates every joint solution in the report while leaving the
    dataset untouched, so the dataset manifest alone cannot detect it.

    Meshes are deliberately excluded -- they are large, and swapping one
    without editing the URDF is not a workflow this repo has.
    """
    from handumi.robots.registry import CONFIG_DIR, load_robot_config

    entries = {"robot_config": _file_digest(CONFIG_DIR / f"{robot}.yaml")}
    try:
        entries["urdf"] = _file_digest(Path(load_robot_config(robot).urdf))
    except (ValueError, TypeError, KeyError):
        entries["urdf"] = "unresolved"
    if deployment_path is not None:
        entries["deployment_calibration"] = _file_digest(Path(deployment_path))
    if robot_tcp_path is not None:
        entries["robot_tcp_calibration"] = _file_digest(Path(robot_tcp_path))
    return entries


def solve_cache_path(root: str | Path, robot: str) -> Path:
    """Sidecar holding the trajectories screening already solved."""
    return Path(root) / "meta" / f"handumi_screening_{robot}_solves.npz"


# Everything the replay solver reads that can change the joints it produces.
# A cached trajectory is only reusable when all of it matches, so conversion
# falls back to solving rather than emitting joints from different settings.
_SOLVER_ARGS = (
    "source",
    "retarget_mode",
    "base_placement",
    "compose_source",
    "translation_scale",
    "controller_device",
    "controller_tcp_calibration",
    "use_dataset_tcp_calibration",
    "raw_controller_debug",
    "absolute_orientation",
    "initial_solve_iterations",
    "initial_position_tolerance_m",
    "max_ik_position_error_m",
    "max_ik_rotation_error_deg",
    "robot_tcp_calibration",
    "gripper_max_width_m",
    "only_manipulation",
    "start_frame",
    "max_frames",
    "stride",
)

# Cached solves are keyed by these fields plus the robot fingerprint and the
# dataset manifest, both already recorded in the screening report.
CACHED_ROLLOUT_FIELDS = (
    "qpos",
    "gripper_normalized",
    "left_pos_error_m",
    "right_pos_error_m",
    "left_rot_error_deg",
    "right_rot_error_deg",
    "initial_solve_iterations",
    "retarget_mode",
    "gripper_source",
    "frame_indices",
    "table_from_robot_pose7",
    "robot_from_table_pose7",
    "base_placement_policy",
    "base_placement_scope",
    "base_placement_verified",
    "base_placement_diagnostics_json",
)


def solver_signature(
    args,
    *,
    deployment_path: str | Path | None = None,
    source_info: dict[str, Any] | None = None,
) -> str:
    """摘要求解参数及实际标定；未提供源元数据时读取本地数据集信息。"""
    payload = {name: str(getattr(args, name, None)) for name in _SOLVER_ARGS}
    payload["deployment_calibration"] = str(
        deployment_path
        if deployment_path is not None
        else getattr(args, "deployment_calibration", None)
    )
    if getattr(args, "retarget_mode", None) == "absolute-tcp":
        from handumi.calibration.robot_tcp import LOCAL_ROBOT_TCP_DIR
        from handumi.retargeting.base_placement import ALGORITHM_VERSION
        from handumi.scripts.replay.replay_in_sim import _resolved_controller_device
        from handumi.scripts.replay.replay_prepare import _source_calibration

        payload["base_placement_algorithm"] = ALGORITHM_VERSION

        path = getattr(args, "robot_tcp_calibration", None) or (
            LOCAL_ROBOT_TCP_DIR / f"{args.robot}.yaml"
        )
        payload["robot_tcp_calibration"] = str(Path(path).resolve())
        payload["robot_tcp_sha256"] = _file_digest(Path(path))
        if source_info is None:
            dataset_root = getattr(args, "dataset_root", None)
            source_info = (
                json.loads(
                    (Path(dataset_root) / "meta" / "info.json").read_text(
                        encoding="utf-8"
                    )
                )
                if dataset_root is not None
                else {}
            )
        device = _resolved_controller_device(args, source_info)
        # 与完整回放缓存共享解析优先级，默认文件变更也必须使筛选结果失效。
        try:
            payload["controller_tcp_selection"] = _source_calibration(
                args, source_info, device
            )
        except SystemExit as exc:
            # 标定缺失的全拒绝报告仍需签名；修复标定后自然生成不同签名。
            payload["controller_tcp_selection"] = {"error": str(exc)}
        payload["deployment_sha256"] = (
            _file_digest(Path(deployment_path)) if deployment_path is not None else "identity"
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RetargetScreeningConfig:
    """Thresholds for grading a retargeted episode.

    The position and rotation ceilings mirror ``handumi replay --strict-ik``.
    The rotation *outlier* rule is relative to the dataset's own distribution:
    a session recorded with a different wrist pose can stay far below the
    absolute ceiling while still being inconsistent with the rest of the
    dataset, which is exactly what would teach a policy two behaviors for one
    task. ``rotation_outlier_floor_deg`` keeps that rule quiet on datasets
    whose spread is uniformly small.

    The threshold is a multiple of the median rather than an IQR fence. A
    second recording session is not a sprinkle of outliers but a whole cluster,
    and once such a cluster is a sizeable minority it sits inside Q3, inflating
    the interquartile range until the fence catches nothing. The median stays
    inside the dominant cluster until contamination passes half the dataset,
    which is the point where no automatic rule can help anyway.
    """

    max_position_error_m: float = 0.03
    max_rotation_error_deg: float = 45.0
    rotation_median_multiplier: float = 3.0
    rotation_outlier_floor_deg: float = 10.0
    self_collision_margin_m: float = 0.01
    self_collision_min_frames: int = 1
    # Largest swing of each arm's first joint (the base yaw on Piper and
    # OpenArm) away from its home value, in degrees. Off by default: what the
    # base may do is a property of the mount and the task, not of the solver.
    # When a target sits at the inner edge of the arm's reach with the wrist
    # pitched down, the folded arm runs out of wrist range and the solver
    # trades orientation for a base swing of 60-110 deg that nothing in the
    # cost pulls back. Setting the limit makes that a rejection rather than a
    # pose a policy is shown.
    max_base_rotation_deg: float | None = None


def _replay_args(
    *,
    root: Path,
    repo_id: str,
    revision: str,
    robot: str,
    episode: int,
    deployment_profile: str,
    rig_config: Path,
    controller_tcp_calibration: Path | None = None,
    retarget_mode: str = "auto",
    base_placement: str = "auto",
    robot_tcp_calibration: Path | None = None,
    deployment_calibration: Path | None = None,
    deployment_selection: Any = None,
    initial_position_tolerance_m: float | None = None,
    max_position_error_m: float | None = None,
    max_rotation_error_deg: float | None = None,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
):
    from handumi.scripts.replay.replay_in_sim import (
        build_parser,
        resolve_replay_options,
    )

    args = build_parser().parse_args([str(root)])
    args.repo_id = repo_id
    args.dataset_root = root
    args.revision = revision
    args.episode = episode
    # An explicit controller->TCP file overrides the capture's snapshot, the
    # same precedence convert applies; it is part of the solver signature.
    args.controller_tcp_calibration = controller_tcp_calibration
    args.robot = robot
    args.retarget_mode = retarget_mode
    args.base_placement = base_placement
    args.robot_tcp_calibration = robot_tcp_calibration
    args.deployment_calibration = deployment_calibration
    args.deployment_selection = deployment_selection
    args.initial_position_tolerance_m = initial_position_tolerance_m
    args.max_ik_position_error_m = max_position_error_m
    args.max_ik_rotation_error_deg = max_rotation_error_deg
    args.cache_dir = cache_dir
    args.no_cache = no_cache
    args.refresh_cache = refresh_cache
    args.deployment_profile = deployment_profile
    args.rig_config = rig_config
    from handumi.scripts.replay.replay_deployment import select_saved_dataset_placement

    if deployment_selection is None:
        select_saved_dataset_placement(args, allow_explicit=True)
    resolve_replay_options(args)
    # Grade every episode instead of aborting on the first violation, and skip
    # the playback lead-in: it is render-only and not part of the trajectory.
    args.strict_ik = args.retarget_mode == "absolute-tcp"
    args.approach_seconds = 0.0
    return args


def _pedestal_columns(coll, robot, home_q: np.ndarray, plane) -> set[int]:
    """Capsules already below the tabletop at rest (mounts, pedestals)."""
    distances = np.asarray(
        coll.compute_world_collision_distance(robot, jnp.asarray(home_q), plane)
    ).reshape(-1)
    return set(np.flatnonzero(distances < 0.0).tolist())


def _table_plane_geometry(
    robot_from_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a point and upward normal for the calibrated tabletop plane."""
    pose = np.asarray(robot_from_table, dtype=np.float32).reshape(7)
    qx, qy, qz, qw = pose[3:]
    # Third column of the quaternion rotation matrix: table +Z in robot world.
    normal = np.array(
        [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dtype=np.float32,
    )
    return pose[:3], normal


def collision_meshes(runtime) -> dict[str, Any]:
    """Load each link's collision geometry as a convex hull in its link frame.

    Capsules enclose these meshes generously: a capsule fitted to a long or
    curved link bulges several centimeters past the geometry, so it can report
    deep interpenetration between parts that are comfortably apart. Capsule
    overlap alone therefore cannot decide whether a trajectory is executable,
    and is used only to select candidate frames for this check.

    Hulls rather than raw meshes because collision geometry is often not
    watertight, which makes signed distance -- the only way to measure
    penetration rather than surface distance -- unreliable on it. A hull is
    watertight by construction and contains its mesh, so it can over-report a
    contact but never miss one, which is the safe direction for a filter that
    discards data.
    """
    import trimesh

    from handumi.robots.registry import yourdfpy_handler

    urdf = runtime.load_urdf(load_meshes=True)
    # The same resolver the URDF loader uses: some asset trees keep the
    # package directory under pkg_root (OpenArm), others drop it (Piper).
    resolve = yourdfpy_handler(runtime.config.pkg_root)
    meshes: dict[str, Any] = {}
    for link in urdf.robot.links:
        candidates = [
            c
            for c in (link.collisions or [])
            if c.geometry is not None and c.geometry.mesh is not None
        ]
        if not candidates:
            continue
        collision = candidates[0]
        try:
            mesh = trimesh.load(resolve(collision.geometry.mesh.filename), force="mesh")
        except (OSError, ValueError):
            continue
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh = mesh.copy()
        scale = collision.geometry.mesh.scale
        if scale is not None:
            # Millimeter meshes (OpenArm) and mirrored arms (negative axis).
            mesh.apply_transform(np.diag([*np.broadcast_to(scale, 3), 1.0]))
        if collision.origin is not None:
            mesh.apply_transform(np.asarray(collision.origin))
        meshes[link.name] = mesh.convex_hull
    return meshes


def _link_transforms(runtime, q: np.ndarray) -> dict[str, np.ndarray]:
    import trimesh

    poses = np.asarray(runtime.robot.forward_kinematics(jnp.asarray(q)))
    transforms = {}
    for index, name in enumerate(runtime.robot.links.names):
        matrix = trimesh.transformations.quaternion_matrix(poses[index][:4])
        matrix[:3, 3] = poses[index][4:7]
        transforms[name] = matrix
    return transforms


def _mesh_separation(mesh_a, transform_a, mesh_b, transform_b, *, stride: int = 1) -> float:
    """Signed separation in meters: positive is clear, negative is penetration.

    ``signed_distance`` is positive inside a watertight mesh, so negating the
    deepest reading over both directions yields one number that is the surface
    gap when apart and the penetration depth when overlapping.
    """
    import trimesh

    a = mesh_a.copy()
    a.apply_transform(transform_a)
    b = mesh_b.copy()
    b.apply_transform(transform_b)
    deepest = max(
        float(trimesh.proximity.signed_distance(a, b.vertices[::stride]).max()),
        float(trimesh.proximity.signed_distance(b, a.vertices[::stride]).max()),
    )
    return -deepest


COLLISION_CHUNK_FRAMES = 128


def _chunked(fn, qpos: np.ndarray, chunk: int = COLLISION_CHUNK_FRAMES) -> np.ndarray:
    """Evaluate ``fn`` over frames in fixed-size blocks.

    Episodes have different lengths, so passing a whole episode makes XLA
    recompile per episode -- measured at ~3 s each against 5 ms once the shape
    is cached, which dominated the screening run. Padding to a constant block
    means one compilation for the entire dataset.
    """
    count = len(qpos)
    padding = (-count) % chunk
    if padding:
        qpos = np.concatenate([qpos, np.repeat(qpos[-1:], padding, axis=0)])
    blocks = [
        np.asarray(fn(jnp.asarray(qpos[start : start + chunk])))
        for start in range(0, len(qpos), chunk)
    ]
    return np.concatenate(blocks, axis=0)[:count]


def _collision_metrics(
    qpos: np.ndarray,
    *,
    runtime,
    coll,
    plane,
    pedestal: set[int],
    world_fn,
    self_fn,
    meshes: dict[str, Any],
    narrow_phase_frames: int = 8,
    vertex_stride: int = 1,
) -> dict[str, float | int]:
    """Capsule broad phase, then a convex-hull check on the frames it flags."""
    self_distances = _chunked(self_fn, qpos)
    world = _chunked(world_fn, qpos)
    columns = [c for c in range(world.shape[1]) if c not in pedestal]
    world = world[:, columns]

    link_names = list(coll.link_names)
    idx_i = np.asarray(coll.active_idx_i)
    idx_j = np.asarray(coll.active_idx_j)
    confirmed_frames: set[int] = set()
    mesh_min = float("inf")
    unverifiable = 0
    for pair in np.flatnonzero((self_distances < 0.0).any(axis=0)):
        a, b = link_names[idx_i[pair]], link_names[idx_j[pair]]
        candidates = np.flatnonzero(self_distances[:, pair] < 0.0)
        if a not in meshes or b not in meshes:
            # No collision geometry to confirm against: keep the capsule verdict
            # rather than silently clearing the pair.
            confirmed_frames.update(candidates.tolist())
            unverifiable += 1
            continue
        # Deepest capsule overlaps first: if those clear, the pair is clear.
        ordered = candidates[np.argsort(self_distances[candidates, pair])]
        probe = ordered[:narrow_phase_frames]
        separations = {}
        for frame in probe:
            transforms = _link_transforms(runtime, qpos[frame])
            separations[int(frame)] = _mesh_separation(
                meshes[a], transforms[a], meshes[b], transforms[b], stride=vertex_stride
            )
        mesh_min = min(mesh_min, min(separations.values()))
        touching = [f for f, d in separations.items() if d <= 0.0]
        if not touching:
            continue
        # Real contact exists, so price the whole pair exactly.
        for frame in ordered:
            if int(frame) in separations:
                if separations[int(frame)] <= 0.0:
                    confirmed_frames.add(int(frame))
                continue
            transforms = _link_transforms(runtime, qpos[frame])
            if (
                _mesh_separation(
                    meshes[a],
                    transforms[a],
                    meshes[b],
                    transforms[b],
                    stride=vertex_stride,
                )
                <= 0.0
            ):
                confirmed_frames.add(int(frame))

    return {
        "self_collision_min_clearance_m": (
            float(mesh_min) if np.isfinite(mesh_min) else float(self_distances.min())
        ),
        "self_collision_frames": len(confirmed_frames),
        "self_collision_capsule_min_m": float(self_distances.min()),
        "self_collision_capsule_frames": int((self_distances.min(axis=1) < 0.0).sum()),
        "self_collision_unverifiable_pairs": unverifiable,
        "table_min_clearance_m": float(world.min()),
        "table_penetration_frames": int((world.min(axis=1) < 0.0).sum()),
    }


def _build_screen_state(
    *,
    dataset_root: Path,
    repo_id: str,
    revision: str,
    robot: str,
    deployment_profile: str,
    rig_config: Path,
    self_collision_margin_m: float,
    controller_tcp_calibration: Path | None = None,
    retarget_mode: str = "auto",
    base_placement: str = "auto",
    robot_tcp_calibration: Path | None = None,
    deployment_calibration: Path | None = None,
    deployment_selection: Any = None,
    initial_position_tolerance_m: float | None = None,
    max_position_error_m: float | None = None,
    max_rotation_error_deg: float | None = None,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Build everything an episode solve reuses: robot model, compiled fns, meshes.

    One call per process. In a worker this is the JIT compilation the pool pays
    for once and then amortises over that worker's share of the episodes.
    """
    from handumi.calibration.deployment import resolve_deployment_calibration
    from handumi.robots.registry import build_pruned_collision_model, load_embodiment

    runtime = load_embodiment(robot)
    deployment = deployment_selection or resolve_deployment_calibration(
        robot,
        explicit_path=deployment_calibration,
        profile=deployment_profile,
        rig_config=rig_config,
        identity_fallback=retarget_mode == "absolute-tcp",
    )
    coll = build_pruned_collision_model(
        urdf=runtime.load_urdf(load_meshes=True),
        robot=runtime.robot,
        home_q=runtime.config.home_q.astype(np.float32),
        arms=runtime.arms,
        gripper_joints={
            side: tuple(g.name for g in runtime.config.arms[side].gripper_joints)
            for side in runtime.arms
        },
        margin=self_collision_margin_m,
    )
    # 逐段调整只沿桌面平面平移基座，不改变无限桌面半空间，故可复用碰撞平面。
    table_point, table_normal = _table_plane_geometry(deployment.pose7)
    plane = pk.collision.HalfSpace.from_point_and_normal(
        jnp.asarray(table_point), jnp.asarray(table_normal)
    )
    # vmap over configurations: pyroki's world distance does not accept a
    # batch axis directly, and calling it per frame paid JAX dispatch 300+
    # times per episode.
    world_fn = jax.jit(
        jax.vmap(
            lambda q: coll.compute_world_collision_distance(
                runtime.robot, q, plane
            ).reshape(-1)
        )
    )
    self_fn = jax.jit(
        lambda q: coll.compute_self_collision_distance(runtime.robot, q)
    )
    return {
        "runtime": runtime,
        "coll": coll,
        "plane": plane,
        "world_fn": world_fn,
        "self_fn": self_fn,
        "pedestal": _pedestal_columns(
            coll, runtime.robot, runtime.config.home_q.astype(np.float32), plane
        ),
        "meshes": collision_meshes(runtime),
        "args_template": {
            "root": dataset_root,
            "repo_id": repo_id,
            "revision": revision,
            "robot": robot,
            "deployment_profile": deployment_profile,
            "rig_config": rig_config,
            "controller_tcp_calibration": controller_tcp_calibration,
            "retarget_mode": retarget_mode,
            "base_placement": base_placement,
            "robot_tcp_calibration": robot_tcp_calibration,
            "deployment_calibration": deployment_calibration,
            "deployment_selection": deployment_selection,
            "initial_position_tolerance_m": initial_position_tolerance_m,
            "max_position_error_m": max_position_error_m,
            "max_rotation_error_deg": max_rotation_error_deg,
            "cache_dir": cache_dir,
            "no_cache": no_cache,
            "refresh_cache": refresh_cache,
        },
    }


def base_rotation_max_deg(qpos: np.ndarray, runtime) -> float:
    """Largest swing of any arm's first joint away from home, in degrees.

    The first joint of each chain is the base yaw on every supported arm. Its
    excursion is measured from ``home_q`` rather than from zero so the number
    means the same thing on a robot whose rest pose is not the URDF zero.
    """
    home = np.asarray(runtime.config.home_q, dtype=np.float32)
    indices = [runtime.arm_joint_indices(side)[0] for side in runtime.arms]
    swing = np.abs(np.asarray(qpos, dtype=np.float32)[:, indices] - home[indices])
    return float(np.degrees(swing.max())) if swing.size else 0.0


def joint_limit_frames_pct(
    qpos: np.ndarray, runtime, *, within_rad: float = 0.0523599
) -> float:
    """Share of frames with any arm joint within 3 deg of a joint stop.

    Gripper fingers are excluded: their whole range is a few centimeters and
    they legitimately sit on a stop when closed.
    """
    q = np.asarray(qpos, dtype=np.float32)
    if q.size == 0:
        return 0.0
    lower = np.asarray(runtime.robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(runtime.robot.joints.upper_limits, dtype=np.float32)
    fingers = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    indices = [
        index
        for side in runtime.arms
        for index in runtime.arm_joint_indices(side)
        if index not in fingers and (upper[index] - lower[index]) > 3.0 * within_rad
    ]
    if not indices:
        return 0.0
    near = (q[:, indices] - lower[indices] < within_rad) | (
        upper[indices] - q[:, indices] < within_rad
    )
    return float(100.0 * near.any(axis=1).mean())


def arm_activity_metrics(rollout: dict[str, Any], runtime) -> dict[str, float | int]:
    """Per-arm travel and idleness of the demonstration itself.

    Read from the raw tool path, not the solved joints, so the number says
    what the demonstrator did: a hand that never moved is a single-arm episode
    whose idle pose the robot still has to hold.
    """
    from handumi.dataset.idle_arm import arm_activity, idle_sides

    out: dict[str, float | int] = {}
    activity = {}
    openings = rollout.get("gripper_normalized")
    widths = None
    if any(runtime.finger_joints.values()) and openings is not None and np.ndim(openings) == 2 and np.shape(openings)[1] == 2:
        widths = np.asarray(openings, dtype=np.float32) * float(
            runtime.config.gripper_max_width_m
        )
    for column, side in enumerate(("left", "right")):
        if side not in runtime.arms:
            continue
        poses = rollout.get(f"raw_{side}_pose7_ground_truth")
        if poses is None or len(poses) == 0:
            continue
        activity[side] = arm_activity(
            np.asarray(poses, dtype=np.float32)[:, :3],
            None if widths is None else widths[:, column],
            side=side,
        )
        out[f"{side}_travel_m"] = activity[side].travel_m
        out[f"{side}_idle"] = int(activity[side].idle)
    out["idle_arm_count"] = len(idle_sides(activity))
    return out


def _screen_one_episode(episode: int, state: dict[str, Any]) -> dict[str, Any]:
    """Solve and grade one episode. Runs in the calling process or in a worker.

    Returns plain arrays and floats: a compiled JAX function cannot cross a
    process boundary, and nothing here needs to.
    """
    from handumi.scripts.replay.replay_prepare import prepare_episode

    args = _replay_args(episode=episode, **state["args_template"])
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            rollout = prepare_episode(args)
    except SystemExit as exc:
        # The solver could not reach the demonstrated start pose at all.
        return {
            "episode_index": episode,
            "unreachable": str(exc),
            "frame_count": 0,
            "metrics": {},
            "solves": {},
            "resolved": {},
        }

    # Sign what the solve settled on, not what was asked for: screening
    # requests "auto" and lets the device come from metadata, while conversion
    # names both outright. Keying on the request would never match an identical
    # solve.
    resolved = {
        field: str(rollout[field][0])
        for field in ("retarget_mode", "controller_device")
        if rollout.get(field) is not None and len(rollout[field])
    }
    solves = {
        field: np.asarray(rollout[field])
        for field in CACHED_ROLLOUT_FIELDS
        if rollout.get(field) is not None
    }
    qpos = np.asarray(rollout["qpos"], dtype=np.float32)
    sides = tuple(state["runtime"].arms)
    pos = np.concatenate([rollout[f"{side}_pos_error_m"] for side in sides])
    rot = np.concatenate([rollout[f"{side}_rot_error_deg"] for side in sides])
    per_frame = np.maximum.reduce([rollout[f"{side}_pos_error_m"] for side in sides])
    metrics: dict[str, float | int] = {
        "position_error_mean_m": float(pos.mean()),
        "position_error_max_m": float(pos.max()),
        # Sustained miss, robust to the settling spike right after the
        # start solve: a single frame at 3 cm while the rate limit catches
        # up is not the same failure as an arm that cannot reach at all.
        "position_error_p99_m": float(np.percentile(per_frame, 99)),
        "position_error_frames_over_1cm": int((per_frame > 0.01).sum()),
        "rotation_error_mean_deg": float(rot.mean()),
        "rotation_error_max_deg": float(rot.max()),
        "initial_position_error_m": float(rollout["initial_max_position_error_m"][0]),
        "initial_solve_iterations": int(rollout["initial_solve_iterations"][0]),
        "base_rotation_max_deg": base_rotation_max_deg(qpos, state["runtime"]),
        "joint_limit_frames_pct": joint_limit_frames_pct(qpos, state["runtime"]),
    }
    metrics.update(arm_activity_metrics(rollout, state["runtime"]))
    metrics.update(
        _collision_metrics(
            qpos,
            runtime=state["runtime"],
            coll=state["coll"],
            plane=state["plane"],
            pedestal=state["pedestal"],
            world_fn=state["world_fn"],
            self_fn=state["self_fn"],
            meshes=state["meshes"],
        )
    )
    return {
        "episode_index": episode,
        "unreachable": None,
        "frame_count": int(len(qpos)),
        "metrics": metrics,
        "solves": solves,
        "resolved": resolved,
        "base_placement": ({
            "policy": str(rollout["base_placement_policy"][0]),
            "scope": str(rollout["base_placement_scope"][0]),
            "verified": bool(rollout["base_placement_verified"][0]),
            "robot_from_table_pose7": rollout["robot_from_table_pose7"][0].tolist(),
            "table_from_robot_pose7": rollout["table_from_robot_pose7"][0].tolist(),
            "diagnostics": json.loads(str(rollout["base_placement_diagnostics_json"][0]) or "{}"),
        } if args.retarget_mode == "absolute-tcp" else None),
    }


def screen_dataset(
    root: str | Path,
    *,
    robot: str,
    repo_id: str | None = None,
    revision: str = "main",
    episodes: list[int] | tuple[int, ...] | None = None,
    deployment_profile: str = "auto",
    rig_config: Path | None = None,
    config: RetargetScreeningConfig | None = None,
    progress: bool = True,
    jobs: int | None = 1,
    controller_tcp_calibration: Path | None = None,
    retarget_mode: str = "auto",
    base_placement: str = "auto",
    robot_tcp_calibration: Path | None = None,
    deployment_calibration: Path | None = None,
    deployment_selection: Any = None,
    initial_position_tolerance_m: float | None = None,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """逐段求解和评分，必要时保存数据集专用的仿真布局，保留原始采集数据。

    ``controller_tcp_calibration`` replaces the capture's controller->TCP
    snapshot with an explicit YAML, exactly as ``handumi convert
    --controller-tcp-calibration`` does, so the graded trajectories are the
    ones that conversion will reuse.

    ``jobs`` spreads the episodes across processes. The numbers do not depend on
    it: each worker runs the same per-episode solve, and episodes never share
    solver state.
    """
    from handumi.calibration.deployment import resolve_deployment_calibration
    from handumi.config import DEFAULT_RIG_CONFIG
    from handumi.scripts.replay.replay_deployment import estimate_dataset_deployment

    dataset_root = Path(root).resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Dataset is missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = int(info.get("total_episodes", 0))
    fps = float(info.get("fps", 0))
    if total_episodes <= 0 or fps <= 0:
        raise ValueError("Dataset info contains invalid episode or FPS totals")
    selected = (
        list(range(total_episodes)) if episodes is None else [int(e) for e in episodes]
    )
    out_of_range = [e for e in selected if e < 0 or e >= total_episodes]
    if out_of_range:
        raise ValueError(f"Episode indices out of range: {sorted(out_of_range)}")

    resolved_repo_id = repo_id or f"local/{dataset_root.name}"
    resolved_args = _replay_args(
        root=dataset_root, repo_id=resolved_repo_id, revision=revision,
        robot=robot, episode=selected[0] if selected else 0,
        deployment_profile=deployment_profile, rig_config=rig_config or DEFAULT_RIG_CONFIG,
        controller_tcp_calibration=controller_tcp_calibration,
        retarget_mode=retarget_mode, base_placement=base_placement,
        robot_tcp_calibration=robot_tcp_calibration,
        deployment_calibration=deployment_calibration,
        deployment_selection=deployment_selection,
        initial_position_tolerance_m=initial_position_tolerance_m,
    )
    retarget_mode = resolved_args.retarget_mode
    base_placement = resolved_args.base_placement
    deployment_selection = getattr(resolved_args, "deployment_selection", None)
    cfg = config or (RetargetScreeningConfig(max_position_error_m=0.005, max_rotation_error_deg=5.0)
                     if retarget_mode == "absolute-tcp" else RetargetScreeningConfig())

    def estimate_deployment():
        """在父进程估计一次共享布局，交给全部 episode 和工作进程使用。"""
        args = _replay_args(
            root=dataset_root, repo_id=resolved_repo_id, revision=revision,
            robot=robot, episode=selected[0] if selected else 0,
            deployment_profile=deployment_profile, rig_config=rig_config or DEFAULT_RIG_CONFIG,
            controller_tcp_calibration=controller_tcp_calibration,
            retarget_mode=retarget_mode, base_placement=base_placement,
            robot_tcp_calibration=robot_tcp_calibration,
        )
        return estimate_dataset_deployment(args)

    deployment = deployment_selection or resolve_deployment_calibration(
            robot,
            explicit_path=deployment_calibration,
            profile=deployment_profile,
            rig_config=rig_config or DEFAULT_RIG_CONFIG,
            identity_fallback=retarget_mode == "absolute-tcp",
            simulation_fallback=estimate_deployment,
    )
    deployment_path = deployment.path
    if deployment.profile == "estimated":
        deployment_calibration = deployment_path
    robot_tcp = None
    if retarget_mode == "absolute-tcp":
        from handumi.calibration.robot_tcp import load_robot_tcp_calibration
        from handumi.robots.registry import load_robot_config

        robot_tcp = load_robot_tcp_calibration(
            robot, load_robot_config(robot).arms, explicit_path=robot_tcp_calibration
        )
        robot_tcp_calibration = robot_tcp.path
    workers = resolve_jobs(jobs, len(selected))
    if progress:
        print(f"[screen] resolved mode={retarget_mode} placement={base_placement} "
              f"calibration={deployment.path or deployment.source} "
              f"max_position={cfg.max_position_error_m:g}m "
              f"max_rotation={cfg.max_rotation_error_deg:g}deg")
        print(f"[screen] {describe_jobs(workers, len(selected))}")

    def _report(episode: int, item: dict[str, Any]) -> None:
        if not progress:
            return
        if item["unreachable"] is not None:
            print(f"  ep {episode:3d}  IK REJECTED: {item['unreachable']}")
            return
        metrics = item["metrics"]
        print(
            f"  ep {episode:3d}  pos {metrics['position_error_mean_m'] * 100:5.2f}"
            f"/{metrics['position_error_p99_m'] * 100:5.2f}"
            f"/{metrics['position_error_max_m'] * 100:5.2f}cm  "
            f"rot {metrics['rotation_error_mean_deg']:5.1f}"
            f"/{metrics['rotation_error_max_deg']:5.1f}deg  "
            f"self {metrics['self_collision_min_clearance_m'] * 100:5.2f}cm"
            f"({metrics['self_collision_frames']})"
        )

    solved = map_episodes(
        selected,
        setup=_build_screen_state,
        setup_kwargs={
            "dataset_root": dataset_root,
            "repo_id": resolved_repo_id,
            "revision": revision,
            "robot": robot,
            "deployment_profile": deployment_profile,
            "rig_config": rig_config or DEFAULT_RIG_CONFIG,
            "self_collision_margin_m": cfg.self_collision_margin_m,
            "controller_tcp_calibration": controller_tcp_calibration,
            "retarget_mode": retarget_mode,
            "base_placement": base_placement,
            "robot_tcp_calibration": robot_tcp_calibration,
            "deployment_calibration": deployment_calibration,
            "deployment_selection": deployment_selection,
            "initial_position_tolerance_m": initial_position_tolerance_m,
            "max_position_error_m": cfg.max_position_error_m if retarget_mode == "absolute-tcp" else None,
            "max_rotation_error_deg": cfg.max_rotation_error_deg if retarget_mode == "absolute-tcp" else None,
            "cache_dir": cache_dir,
            "no_cache": no_cache,
            "refresh_cache": refresh_cache,
        },
        task=_screen_one_episode,
        jobs=workers,
        on_result=_report,
    )

    raw: list[dict[str, Any]] = []
    solves: dict[str, np.ndarray] = {}
    signature: str | None = None
    if retarget_mode == "absolute-tcp":
        # 即使全部轨迹未通过，也要签署本次参数，门禁才能显示真实拒绝原因。
        from handumi.scripts.replay.replay_in_sim import _resolved_controller_device

        resolved_args = _replay_args(
            root=dataset_root,
            repo_id=resolved_repo_id,
            revision=revision,
            robot=robot,
            episode=selected[0] if selected else 0,
            deployment_profile=deployment_profile,
            rig_config=rig_config or DEFAULT_RIG_CONFIG,
            controller_tcp_calibration=controller_tcp_calibration,
            retarget_mode=retarget_mode,
            base_placement=base_placement,
            robot_tcp_calibration=robot_tcp_calibration,
            deployment_calibration=deployment_calibration,
            deployment_selection=deployment_selection,
            initial_position_tolerance_m=initial_position_tolerance_m,
            max_position_error_m=cfg.max_position_error_m,
            max_rotation_error_deg=cfg.max_rotation_error_deg,
        )
        resolved_args.controller_device = _resolved_controller_device(resolved_args, info)
        signature = solver_signature(
            resolved_args, deployment_path=deployment_path, source_info=info
        )
    for item in solved:
        episode = item["episode_index"]
        raw.append(
            {
                "episode_index": episode,
                "unreachable": item["unreachable"],
                "frame_count": item["frame_count"],
                "metrics": item["metrics"],
                "base_placement": item.get("base_placement"),
            }
        )
        if item["unreachable"] is not None:
            continue
        if signature is None and item["resolved"]:
            resolved_args = _replay_args(
                root=dataset_root,
                repo_id=resolved_repo_id,
                revision=revision,
                robot=robot,
                episode=episode,
                deployment_profile=deployment_profile,
                rig_config=rig_config or DEFAULT_RIG_CONFIG,
                controller_tcp_calibration=controller_tcp_calibration,
                retarget_mode=retarget_mode,
                base_placement=base_placement,
                robot_tcp_calibration=robot_tcp_calibration,
                deployment_calibration=deployment_calibration,
                deployment_selection=deployment_selection,
                initial_position_tolerance_m=initial_position_tolerance_m,
                max_position_error_m=cfg.max_position_error_m if retarget_mode == "absolute-tcp" else None,
                max_rotation_error_deg=cfg.max_rotation_error_deg if retarget_mode == "absolute-tcp" else None,
            )
            for field, value in item["resolved"].items():
                setattr(resolved_args, field, value)
            signature = solver_signature(
                resolved_args, deployment_path=deployment_path, source_info=info
            )
        for field, value in item["solves"].items():
            solves[f"{episode}/{field}"] = value

    fences = _rotation_fences(raw, cfg)
    reports = [_grade(item, cfg=cfg, fps=fps, fences=fences) for item in raw]
    accepted = sum(report.accepted for report in reports)
    flagged = sorted(
        report.episode_index
        for report in reports
        if any(f.severity == "reject" for f in report.findings)
    )
    review = sorted(
        report.episode_index
        for report in reports
        if report.accepted and report.findings
    )
    return {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "kind": SCREENING_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": resolved_repo_id,
        "root": portable_path(dataset_root),
        "robot": robot,
        "retarget_mode": retarget_mode,
        "base_placement_policy": base_placement,
        "config": asdict(cfg),
        "rotation_fences_deg": fences,
        "payload_manifest": dataset_payload_manifest(dataset_root),
        "deployment_calibration_path": portable_path(deployment_path) if deployment_path is not None else None,
        "deployment_calibration": {
            "source": deployment.source, "pose7": deployment.pose7.tolist(),
            "scope": deployment.scope, "verified": deployment.verified,
            "profile": deployment.profile,
        },
        "robot_tcp_calibration_path": portable_path(robot_tcp.path) if robot_tcp is not None else None,
        "robot_tcp_calibration": robot_tcp.metadata() if robot_tcp is not None else None,
        "controller_tcp_calibration_path": (
            None
            if controller_tcp_calibration is None
            else portable_path(str(controller_tcp_calibration))
        ),
        "solver_signature": signature,
        "solve_cache": portable_path(solve_cache_path(dataset_root, robot))
        if solves
        else None,
        "robot_fingerprint": robot_fingerprint(
            robot, deployment_path=deployment_path,
            robot_tcp_path=robot_tcp.path if robot_tcp is not None else None,
        ),
        "summary": {
            "total": len(reports),
            "accepted": accepted,
            "rejected": len(reports) - accepted,
            "reject_episode_indices": flagged,
            "review_episode_indices": review,
        },
        "episodes": [{**report.to_dict(), "base_placement": item.get("base_placement")}
                     for report, item in zip(reports, raw, strict=True)],
        "_solves": solves,
    }


def _rotation_fences(
    raw: list[dict[str, Any]], cfg: RetargetScreeningConfig
) -> dict[str, float]:
    means = [
        float(item["metrics"]["rotation_error_mean_deg"])
        for item in raw
        if item["unreachable"] is None
    ]
    if len(means) < 4:
        # Too few episodes for a meaningful distribution; absolute ceiling only.
        return {}
    values = np.asarray(means)
    median = float(np.median(values))
    q1, q3 = (float(v) for v in np.quantile(values, [0.25, 0.75]))
    return {
        "median_deg": median,
        "q1_deg": q1,
        "q3_deg": q3,
        "upper_fence_deg": max(
            cfg.rotation_median_multiplier * median,
            cfg.rotation_outlier_floor_deg,
        ),
    }


def _grade(
    item: dict[str, Any],
    *,
    cfg: RetargetScreeningConfig,
    fps: float,
    fences: dict[str, float],
) -> EpisodeQualityReport:
    findings: list[QualityFinding] = []
    metrics = dict(item["metrics"])

    if item["unreachable"] is not None:
        findings.append(
            QualityFinding(
                code=("retarget_ik_fidelity" if str(item["unreachable"]).startswith(("IK fidelity", "Invalid non-finite"))
                      else "retarget_start_pose_unreachable"),
                severity="reject",
                message=(
                    "The demonstrated trajectory did not meet the IK fidelity requirements."
                ),
                metrics={"solver_message": str(item["unreachable"])},
            )
        )
        return EpisodeQualityReport(
            episode_index=int(item["episode_index"]),
            frame_count=int(item["frame_count"]),
            duration_s=0.0,
            findings=tuple(findings),
            metrics=metrics,
        )

    if metrics.get("position_error_p99_m", 0.0) > cfg.max_position_error_m:
        findings.append(
            QualityFinding(
                code="retarget_position_error",
                severity="reject",
                message=(
                    "The robot does not reach the demonstrated trajectory: the "
                    "position error stays above the ceiling, not just spikes."
                ),
                metrics={
                    "position_error_p99_m": metrics["position_error_p99_m"],
                    "position_error_frames_over_1cm": metrics.get(
                        "position_error_frames_over_1cm", 0
                    ),
                    "limit_m": cfg.max_position_error_m,
                },
            )
        )
    elif metrics["position_error_max_m"] > cfg.max_position_error_m:
        findings.append(
            QualityFinding(
                code="retarget_position_spike",
                severity="warning",
                message=(
                    "A brief position excursion above the ceiling, typically the "
                    "solver settling after the start pose rather than a "
                    "reachability failure."
                ),
                metrics={
                    "position_error_max_m": metrics["position_error_max_m"],
                    "position_error_p99_m": metrics.get("position_error_p99_m", 0.0),
                    "limit_m": cfg.max_position_error_m,
                },
            )
        )
    if metrics["rotation_error_max_deg"] > cfg.max_rotation_error_deg:
        findings.append(
            QualityFinding(
                code="retarget_rotation_error",
                severity="warning",
                message="Retargeted TCP orientation error exceeds the ceiling.",
                metrics={
                    "rotation_error_max_deg": metrics["rotation_error_max_deg"],
                    "limit_deg": cfg.max_rotation_error_deg,
                },
            )
        )
    upper = fences.get("upper_fence_deg")
    if upper is not None and metrics["rotation_error_mean_deg"] > upper:
        findings.append(
            QualityFinding(
                code="retarget_rotation_outlier",
                severity="warning",
                message=(
                    "Orientation tracking is inconsistent with the rest of the "
                    "dataset, which usually means a different recording session "
                    "or tool grip."
                ),
                metrics={
                    "rotation_error_mean_deg": metrics["rotation_error_mean_deg"],
                    "upper_fence_deg": upper,
                },
            )
        )
    swing = metrics.get("base_rotation_max_deg")
    if (
        cfg.max_base_rotation_deg is not None
        and swing is not None
        and float(swing) > cfg.max_base_rotation_deg
    ):
        findings.append(
            QualityFinding(
                code="retarget_base_rotation",
                severity="reject",
                # A rejection, not a warning: the rule is off unless the
                # reviewer sets the limit, so setting it is the decision.
                message=(
                    "The retargeted trajectory swings an arm base "
                    f"{float(swing):.1f} deg from home, past the "
                    f"{cfg.max_base_rotation_deg:.0f} deg limit. The target sits "
                    "where the folded arm has no wrist range left and the "
                    "solver traded orientation for base rotation."
                ),
                metrics={
                    "base_rotation_max_deg": float(swing),
                    "limit_deg": cfg.max_base_rotation_deg,
                },
            )
        )
    if metrics["self_collision_frames"] >= cfg.self_collision_min_frames:
        frame_count = max(int(item["frame_count"]), 1)
        share = 100.0 * metrics["self_collision_frames"] / frame_count
        findings.append(
            QualityFinding(
                code="retarget_self_collision",
                severity="warning",
                # The share, not the count, is what a reviewer decides on: a
                # touch while retracting and a pose held through the approach
                # produce the same code and the same message otherwise.
                message=(
                    "The retargeted trajectory intersects the robot itself on "
                    f"{metrics['self_collision_frames']} of {frame_count} frames "
                    f"({share:.1f}% of the episode)."
                ),
                metrics={
                    "self_collision_frames": metrics["self_collision_frames"],
                    "self_collision_episode_share_pct": round(share, 2),
                    "self_collision_min_clearance_m": metrics[
                        "self_collision_min_clearance_m"
                    ],
                },
            )
        )
    return EpisodeQualityReport(
        episode_index=int(item["episode_index"]),
        frame_count=int(item["frame_count"]),
        duration_s=float(item["frame_count"]) / fps if fps else 0.0,
        findings=tuple(findings),
        metrics=metrics,
    )


@dataclass(frozen=True)
class ScreeningGate:
    """Whether a dataset is cleared for joint conversion on one embodiment."""

    report_path: Path
    status: str  # "clear" | "review" | "missing" | "stale" | "flagged"
    flagged: dict[int, list[dict[str, Any]]]
    detail: str

    @property
    def blocks(self) -> bool:
        """Refuse only what a reviewer cannot already have settled.

        Warnings do not block. Curation is where a reviewer acts on them, so
        after removing the rejections a dataset still carries warnings by
        definition; blocking there would make the override reflexive, and that
        override also switches off the missing, stale and rejected checks --
        the ones that catch a report written against different data or a
        different robot.
        """
        return self.status in {"missing", "stale", "flagged"}


def evaluate_screening_gate(
    root: str | Path,
    *,
    robot: str,
    episodes: list[int] | tuple[int, ...] | None = None,
    expected_solver_signature: str | None = None,
    expected_deployment_path: str | Path | None = None,
    expected_robot_tcp_path: str | Path | None = None,
) -> ScreeningGate:
    """Check the screening report before a dataset becomes joint targets.

    Blocks on any finding, not only rejections: a warning means a human has
    not yet decided, and silently training on it is the failure this gate
    exists to prevent.
    """
    dataset_root = Path(root)
    path = screening_report_path(dataset_root, robot)
    if not path.is_file():
        return ScreeningGate(
            report_path=path,
            status="missing",
            flagged={},
            detail=f"No retargeting screening report for {robot!r} at {path}.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=f"Cannot read screening report {path}: {exc}",
        )
    if (expected_solver_signature is not None
            and payload.get("solver_signature") != expected_solver_signature):
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(f"Screening report {path} used different retargeting settings or "
                    "calibration; re-run handumi dataset screen with --retarget-mode absolute-tcp "
                    "and matching TCP, table and error thresholds."),
        )
    if payload.get("payload_manifest") != dataset_payload_manifest(dataset_root):
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(
                f"Screening report {path} is stale: the dataset payload changed "
                "since it was written."
            ),
        )
    recorded = payload.get("robot_fingerprint")
    current = robot_fingerprint(
        robot,
        deployment_path=(expected_deployment_path if expected_solver_signature is not None
                         else repo_path(payload.get("deployment_calibration_path"))),
        robot_tcp_path=(expected_robot_tcp_path if expected_solver_signature is not None
                        else repo_path(payload.get("robot_tcp_calibration_path"))),
    )
    if recorded != current:
        changed = (
            sorted(
                key
                for key in set(recorded) | set(current)
                if recorded.get(key) != current.get(key)
            )
            if isinstance(recorded, dict)
            else ["the whole fingerprint"]
        )
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(
                f"Screening report {path} is stale: {', '.join(changed)} changed "
                "since it was written, so every joint solution in it is invalid."
            ),
        )

    screened = {int(item["episode_index"]) for item in payload.get("episodes", ())}
    if episodes is None:
        # Coverage is judged against the dataset, not against the report: a
        # report written with --episodes must not look complete.
        info_path = dataset_root / "meta" / "info.json"
        try:
            total = int(
                json.loads(info_path.read_text(encoding="utf-8")).get(
                    "total_episodes", 0
                )
            )
        except (OSError, json.JSONDecodeError, ValueError):
            total = 0
        wanted = set(range(total)) if total > 0 else set(screened)
    else:
        wanted = {int(e) for e in episodes}
    missing = sorted(wanted - screened)
    if missing:
        return ScreeningGate(
            report_path=path,
            status="stale",
            flagged={},
            detail=(
                f"Screening report {path} does not cover episodes {missing}; "
                "re-run handumi dataset screen without --episodes."
            ),
        )

    flagged = {
        int(item["episode_index"]): list(item.get("findings", ()))
        for item in payload.get("episodes", ())
        if item.get("findings")
        and (wanted is None or int(item["episode_index"]) in wanted)
    }
    if flagged:
        rejected = {
            index: findings
            for index, findings in flagged.items()
            if any(f.get("severity") == "reject" for f in findings)
        }
        if rejected:
            return ScreeningGate(
                report_path=path,
                status="flagged",
                flagged=flagged,
                detail=(
                    f"{len(rejected)} episode(s) are unusable on {robot!r} and "
                    "are still included."
                ),
            )
        return ScreeningGate(
            report_path=path,
            status="review",
            flagged=flagged,
            detail=(
                f"{len(flagged)} episode(s) carry warnings for {robot!r} that a "
                "reviewer chose to keep."
            ),
        )
    return ScreeningGate(
        report_path=path, status="clear", flagged={}, detail="Screening is clear."
    )


def format_gate_guidance(gate: ScreeningGate, *, root: Path, robot: str) -> str:
    """Explain how to satisfy the gate, with runnable commands."""
    lines = [gate.detail]
    if gate.status in {"missing", "stale"}:
        lines += [
            "",
            "Run the retargeting screening first:",
            f"  handumi dataset screen {root} --robot {robot}",
        ]
    else:
        rejects = sorted(
            index
            for index, findings in gate.flagged.items()
            if any(f.get("severity") == "reject" for f in findings)
        )
        warnings = sorted(set(gate.flagged) - set(rejects))
        for index in sorted(gate.flagged):
            codes = ", ".join(
                f"{f.get('severity')}:{f.get('code')}" for f in gate.flagged[index]
            )
            lines.append(f"  episode {index}: {codes}")
        if rejects:
            lines.append(f"Unusable on this robot: {rejects}")
        if warnings:
            lines.append(f"Needs your judgement: {warnings}")
        excluded = ",".join(str(i) for i in sorted(gate.flagged))
        lines += [
            "",
            "Review them, then remove the ones you reject:",
            f"  handumi dataset analyze {root} --quality-report {gate.report_path}",
            f"  handumi dataset curate {root} --output <new_root> "
            f"--exclude {excluded}",
        ]
    lines += [
        "",
        "To convert anyway, pass --allow-flagged-episodes.",
    ]
    return "\n".join(lines)


def write_screening_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write the report, and the solved trajectories beside it.

    The trajectories live in a separate npz because they are arrays, not
    findings: the report stays a readable audit document, and conversion can
    reuse the solve instead of repeating it.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    solves = payload.pop("_solves", None)
    if solves:
        # Derived from where the report is being written rather than from the
        # recorded path, which is provenance and may have come from elsewhere.
        np.savez_compressed(
            output.with_name(output.stem + "_solves.npz"), **solves
        )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def load_cached_solve(
    root: str | Path,
    *,
    robot: str,
    episode: int,
    signature: str,
) -> dict[str, np.ndarray] | None:
    """Return a previously solved rollout, or None when it cannot be trusted.

    Freshness of the dataset and of the robot geometry is already enforced by
    the screening gate, so this only has to confirm the solver settings match
    and that this episode is present.
    """
    report_path = screening_report_path(root, robot)
    cache_path = solve_cache_path(root, robot)
    if not report_path.is_file() or not cache_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("solver_signature") != signature:
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            prefix = f"{episode}/"
            keys = [name for name in archive.files if name.startswith(prefix)]
            if not any(name == f"{prefix}qpos" for name in keys):
                return None
            return {name[len(prefix) :]: archive[name] for name in keys}
    except (OSError, ValueError):
        return None


def _fmt_deg(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def _idle_label(metrics: dict[str, Any]) -> str:
    idle = [side for side in ("left", "right") if metrics.get(f"{side}_idle")]
    if not idle:
        return "-"
    return "+".join(idle) if len(idle) > 1 else idle[0]


def render_screening_markdown(payload: dict[str, Any]) -> str:
    """生成筛选报告，并明确标注数据估计的仿真布局及其假设。"""
    summary = payload["summary"]
    lines = [
        f"# Retargeting screening: {payload['robot']}",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Episodes: {summary['total']} "
        f"({summary['accepted']} accepted, {summary['rejected']} rejected)",
    ]
    deployment = payload.get("deployment_calibration") or {}  # 报告对应的实际摆放来源。
    if deployment.get("source") == "dataset_home_alignment_v1":
        lines += [
            "- Deployment: data-derived simulation estimate; not physically verified.",
            "- Assumptions: upright robot base at tabletop height; XY and yaw estimated from TCP trajectories and HOME poses.",
            f"- Estimated calibration: `{payload.get('deployment_calibration_path')}`",
        ]
    fences = payload.get("rotation_fences_deg") or {}
    if fences:
        lines.append(
            f"- Rotation outlier threshold: {fences['upper_fence_deg']:.1f} deg "
            f"(median {fences['median_deg']:.1f}, "
            f"Q1 {fences['q1_deg']:.1f}, Q3 {fences['q3_deg']:.1f})"
        )
    lines += [
        "",
        "| ep | frames | pos mean/max (cm) | rot mean/max (deg) | self (cm/frames) "
        "| table (cm/frames) | base (deg) | limit (%) | idle | status |",
        "|---:|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for episode in payload["episodes"]:
        m = episode.get("metrics") or {}
        if not m:
            lines.append(
                f"| {episode['episode_index']} | - | - | - | - | - | - | - | - | "
                f"**{episode['status']}** |"
            )
            continue
        lines.append(
            f"| {episode['episode_index']} | {episode['frame_count']} "
            f"| {m['position_error_mean_m'] * 100:.2f} / "
            f"{m['position_error_max_m'] * 100:.2f} "
            f"| {m['rotation_error_mean_deg']:.1f} / "
            f"{m['rotation_error_max_deg']:.1f} "
            f"| {m['self_collision_min_clearance_m'] * 100:.2f} / "
            f"{m['self_collision_frames']} "
            f"| {m['table_min_clearance_m'] * 100:.2f} / "
            f"{m['table_penetration_frames']} "
            f"| {_fmt_deg(m.get('base_rotation_max_deg'))} "
            f"| {_fmt_deg(m.get('joint_limit_frames_pct'))} "
            f"| {_idle_label(m)} "
            f"| {episode['status']} |"
        )
    findings = [
        (episode["episode_index"], finding)
        for episode in payload["episodes"]
        for finding in episode.get("findings", ())
    ]
    if findings:
        lines += ["", "## Findings", ""]
        for index, finding in findings:
            lines.append(
                f"- episode {index}: **{finding['severity']}** "
                f"`{finding['code']}` -- {finding['message']}"
            )
    return "\n".join(lines) + "\n"
