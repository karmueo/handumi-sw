#!/usr/bin/env python3
"""根据机器人 YAML 配置与录制设备，将 HandUMI 数据集映射为关节轨迹并回放。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any

import numpy as np

from handumi.calibration.control_tcp import (
    DEFAULT_DEVICE as DEFAULT_CONTROLLER_DEVICE,
)
from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    apply_controller_tcp_calibration,
    calibration_path_for_device,
    controller_tcp_calibration_from_metadata,
    controller_tcp_calibration_sha256,
    is_identity_bound_controller_tcp_metadata,
    load_controller_tcp_calibration,
)
from handumi.calibration.deployment import (
    deployment_calibration_metadata,
    print_deployment_calibration,
    resolve_deployment_calibration,
)
from handumi.calibration.robot_tcp import load_robot_tcp_calibration
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset import (
    DatasetRef,
    ensure_metadata,
    handumi_metadata,
    open_dataset,
    validate_raw_state_metadata,
)
from handumi.dataset.raw import (
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)
from handumi.dataset.selection import resolve_dataset_selection
from handumi.retargeting.base_placement import (
    ALGORITHM_VERSION,
    first_frame_candidates,
    search_feasible_placement,
)
from handumi.retargeting.handumi_to_robot import (
    VR_TO_ROBOT,
    absolute_table_robot_target_pose7,
    local_frame_adapter,
    local_relative_robot_target_pose7,
    orientation_only_pose_adapter,
    raw_state_pose7_pair,
    raw_state_robot_target_pose7,
    retarget_anchors_from_raw_state,
)
from handumi.robots.kinematics import (
    optimization_score_from_errors,
    pose_error_arrays,
    posture_seed,
)
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment, load_robot_config
from handumi.robots.utils import pose7_to_mat, pose_inv, pose_mul
from handumi.scripts.replay.replay_progress import report

DEFAULT_OUT_DIR = Path("outputs/replay_in_sim")
GRIPPER_NORMALIZED_KEYS = (
    "observation.feetech.left_normalized",
    "observation.feetech.right_normalized",
)


@dataclass(frozen=True)
class TcpCalibrationSelection:
    calibration: ControllerTcpCalibration
    source: str
    source_robot: str
    source_gripper: str
    tracking_device: str
    controller_mount: str
    trusted_dataset_snapshot: bool


def build_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        description="Replay a raw HandUMI LeRobot episode through bimanual IK."
    )
    parser.add_argument(
        "dataset",
        help="Local dataset path or Hugging Face repo id.",
    )
    parser.add_argument("--help-advanced", action="store_true", help="Show expert IK and calibration options.")
    parser.add_argument("--revision", default="main", help=advanced("Hub dataset revision."))
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help=advanced("Machine-local rig YAML used to select lab deployment calibration."),
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Initial episode index (0-based); selectable in the viewer.",
    )
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, default="piper")
    parser.add_argument(
        "--source",
        choices=("observation.state", "action"),
        default="observation.state",
        help=advanced("Raw 16D LeRobot feature to replay."),
    )
    parser.add_argument("--start-frame", type=int, default=0, help=advanced("First source frame."))
    parser.add_argument("--max-frames", type=int, default=None, help=advanced("Maximum frames to solve."))
    parser.add_argument("--stride", type=int, default=1, help=advanced("Source frame stride."))
    parser.add_argument(
        "--retarget-mode",
        choices=("auto", "local-relative", "anchored", "absolute-table", "absolute-tcp"),
        default="auto",
        help=advanced(
            "auto selects absolute-table for datasets recorded in a calibrated table "
            "workspace, otherwise local-relative. local-relative replays frame-to-frame "
            "TCP SE(3) deltas in robot EE space. anchored preserves the older home + "
            "position-delta mode. absolute-table applies one shared table->robot "
            "transform to both TCPs."
        ),
    )
    parser.add_argument(
        "--compose-source",
        choices=("commanded", "achieved"),
        default="commanded",
        help=advanced(
            "For --retarget-mode local-relative, compose each adapted delta on "
            "the previous commanded target or previous achieved FK pose."
        ),
    )
    parser.add_argument(
        "--base-placement", choices=("auto", "fixed", "episode-xy", "dataset-xy-yaw"), default="auto",
        help="Select fixed placement or per-episode virtual XY placement for absolute-tcp.",
    )
    parser.add_argument(
        "--refresh-base-placement", action="store_true",
        help="Search for a new dataset-wide base placement even when one is saved.",
    )
    parser.add_argument(
        "--placement-search-candidates", type=int, default=8,
        help=advanced("Maximum number of dataset-wide candidates fully validated."),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help=advanced("Scale local-relative translation deltas after frame adaptation."),
    )
    parser.add_argument(
        "--controller-device",
        choices=("pico", "meta", "vive"),
        default=None,
        help=advanced(
            "Controller device. Defaults to handumi.recording_device in dataset "
            f"metadata, then {DEFAULT_CONTROLLER_DEVICE!r} for legacy datasets."
        ),
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=advanced(
            "Explicit YAML with controller->HandUMI TCP transforms. Overrides both "
            "the robot/device configured calibration and dataset metadata."
        ),
    )
    parser.add_argument(
        "--use-dataset-tcp-calibration",
        action="store_true",
        help=advanced(
            "Force the controller->TCP snapshot embedded in the dataset, including "
            "unidentified legacy snapshots, instead of normal precedence."
        ),
    )
    parser.add_argument(
        "--raw-controller-debug",
        action="store_true",
        help=advanced("Replay raw PICO controller poses without controller->TCP calibration."),
    )
    parser.add_argument(
        "--robot-tcp-calibration", type=Path, default=None,
        help=advanced("Robot end-link to TCP calibration snapshot for absolute-tcp."),
    )
    parser.add_argument(
        "--deployment-calibration",
        type=Path,
        default=None,
        help=advanced(
            "YAML containing robot_from_table for --retarget-mode absolute-table. "
            "Overrides both the lab-local and canonical simulation profiles."
        ),
    )
    parser.add_argument(
        "--deployment-profile",
        choices=("auto", "local", "sim"),
        default="auto",
        help=(
            "Table placement source: auto uses the lab-local calibration from "
            "rig.yaml when configured, otherwise the canonical simulation profile."
        ),
    )
    parser.add_argument(
        "--absolute-orientation",
        choices=("relative-start", "table-absolute"),
        default=None,
        help=advanced(
            "Orientation policy for absolute-table. relative-start aligns each "
            "HandUMI tool frame to the robot home TCP at frame 0 while preserving "
            "all subsequent wrist rotations. table-absolute requires matching, "
            "externally calibrated HandUMI and robot TCP frame conventions."
        ),
    )
    parser.add_argument(
        "--initial-solve-iterations",
        type=int,
        default=12,
        help=advanced(
            "Unrestricted IK iterations used to prepare the absolute-table start "
            "configuration before frame 0."
        ),
    )
    parser.add_argument(
        "--initial-position-tolerance-m",
        type=float,
        default=None,
        help=advanced("Required maximum TCP position error before starting replay."),
    )
    parser.add_argument(
        "--approach-seconds",
        type=float,
        default=None,
        help=advanced(
            "Playback lead-in that ramps the robot from its home pose to the "
            "episode start pose, mirroring how sim teleop parks at home. "
            "0 starts playback directly at frame 0."
        ),
    )
    parser.add_argument(
        "--gripper-max-width-m",
        type=float,
        default=None,
        help=advanced(
            "Fallback physical opening mapped to fully open. Defaults to the robot "
            "configuration and is used only when recorded Feetech "
            "normalized fields are unavailable."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--prepare-mode", choices=("background", "all", "off"), default="background",
        help="Prepare other episodes in background, all before viewing, or only on selection.",
    )
    parser.add_argument("--prepare-workers", type=int, default=8,
                        help="Maximum concurrent episode preparation workers (default: 8).")
    parser.add_argument("--placement-workers", type=int, default=None,
                        help="Concurrent dataset placement workers (default: --prepare-workers; 1 disables parallel search).")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto",
                        help="Show batch progress (auto/on) or disable it (off).")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory for per-episode replay cache.")
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--no-cache", action="store_true", help="Do not read or write replay caches.")
    cache_mode.add_argument("--refresh-cache", action="store_true", help="Recompute and replace each requested cache.")
    parser.add_argument(
        "--clean-view",
        action="store_true",
        help=(
            "Hide the table grid, axes and placement labels in the Viser viewer "
            "for clean screenshots on a white background."
        ),
    )
    parser.add_argument(
        "--hide-trajectories",
        action="store_true",
        help=(
            "Hide target and achieved TCP paths and markers in the Viser viewer, "
            "showing only the robot and scene."
        ),
    )
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "Render assets/scenes/<name>/scene.xml in the calibrated table frame. "
            "This provides initial task context; recorded object motion is unavailable."
        ),
    )
    parser.add_argument(
        "--max-ik-position-error-m",
        type=float,
        default=None,
        help=advanced("Maximum acceptable per-frame TCP position error."),
    )
    parser.add_argument(
        "--table-clearance-warning-m",
        type=float,
        default=0.10,
        help=advanced(
            "Warn when a table-calibrated episode never places either HandUMI TCP "
            "closer than this height to table z=0."
        ),
    )
    parser.add_argument(
        "--max-ik-rotation-error-deg",
        type=float,
        default=None,
        help=advanced("Maximum acceptable per-frame TCP rotation error."),
    )
    parser.add_argument(
        "--strict-ik",
        action="store_true",
        help="Fail instead of warning when an IK error threshold is exceeded.",
    )
    parser.add_argument(
        "--only-manipulation",
        action="store_true",
        help=advanced(
            "Lock manipulation.lock_joints from the robot yaml (e.g. torso). "
            "Default: all actuated joints remain free during replay."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help=advanced("Viser port."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the replay plan without loading the dataset.",
    )
    return parser


def resolve_replay_options(args: argparse.Namespace) -> None:
    """补全按模式区分的默认值，并拒绝与绝对 TCP 朝向冲突的设置。"""
    absolute_tcp = args.retarget_mode == "absolute-tcp"
    placement_workers = getattr(args, "placement_workers", None)
    if placement_workers is not None and placement_workers < 1:
        raise ValueError("--placement-workers must be >= 1.")
    robot = getattr(args, "robot", None) or getattr(args, "embodiment", None)  # 回放与转换共用的目标机器人。
    if getattr(args, "approach_seconds", None) is None:
        # RM75B 绝对 TCP 回放直接从首帧摆放，其他模式保留原接近轨迹。
        args.approach_seconds = 0.0 if absolute_tcp and robot == "rm75b" else 1.0
    policy = getattr(args, "base_placement", "auto")
    if policy in {"episode-xy", "dataset-xy-yaw"} and not absolute_tcp:
        raise SystemExit(f"--base-placement {policy} requires --retarget-mode absolute-tcp.")
    if policy == "dataset-xy-yaw":
        if robot != "rm75b":
            raise SystemExit("--base-placement dataset-xy-yaw currently requires --robot rm75b.")
        if (getattr(args, "start_frame", 0) != 0 or getattr(args, "max_frames", None) is not None
                or getattr(args, "stride", 1) != 1 or getattr(args, "only_manipulation", False)):
            raise SystemExit("dataset-xy-yaw requires all original frames; frame selection and --only-manipulation are not supported.")
        if getattr(args, "deployment_calibration", None) is not None:
            raise SystemExit("dataset-xy-yaw cannot be combined with --deployment-calibration; use --base-placement fixed to load a calibration.")
        if getattr(args, "placement_search_candidates", 8) < 1:
            raise SystemExit("--placement-search-candidates must be >= 1.")
    elif getattr(args, "refresh_base_placement", False):
        raise SystemExit("--refresh-base-placement requires --base-placement dataset-xy-yaw.")
    orientation = getattr(args, "absolute_orientation", None)
    if absolute_tcp and orientation == "relative-start":
        raise SystemExit("absolute-tcp requires table-absolute orientation; relative-start conflicts.")
    if orientation is None:
        args.absolute_orientation = "table-absolute" if absolute_tcp else "relative-start"
    for name, new_default, old_default in (
        ("initial_position_tolerance_m", 0.005, 0.01),
        ("max_ik_position_error_m", 0.005, 0.03),
        ("max_ik_rotation_error_deg", 5.0, 45.0),
    ):
        if getattr(args, name, None) is None:
            value = new_default if absolute_tcp else old_default
            setattr(args, name, value)
        if absolute_tcp and (not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            raise SystemExit(f"{name} must be finite and greater than zero.")


def _require_finite_frame(frame: int, label: str, *values: np.ndarray) -> None:
    """拒绝带非有限值的目标、关节或 FK，给出原始帧号。"""
    if any(not np.isfinite(np.asarray(value)).all() for value in values):
        raise SystemExit(f"Invalid non-finite {label} at frame {frame}.")


def joint_approach_ramp(
    home_q: np.ndarray,
    start_q: np.ndarray,
    *,
    frames: int,
) -> np.ndarray:
    """Smoothstep joint-space lead-in from the home pose to the episode start.

    Playback then begins where sim teleop parks the robot instead of popping
    straight into the first solved frame (absolute-table prepares that pose
    with the joint-delta limit disabled, so the jump can exceed a radian).
    The ramp includes ``home_q`` and excludes ``start_q``: the caller plays
    the solved trajectory right after it.
    """
    home = np.asarray(home_q, dtype=np.float32)
    start = np.asarray(start_q, dtype=np.float32)
    if frames <= 0 or float(np.abs(start - home).max()) <= 1e-3:
        return np.empty((0, home.shape[0]), dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, frames + 1, dtype=np.float32)[:-1]
    eased = alpha * alpha * (3.0 - 2.0 * alpha)
    return (home + eased[:, None] * (start - home)).astype(np.float32)


def _replay_locked_joint_indices(
    runtime,
    args: argparse.Namespace,
) -> tuple[int, ...]:
    if not args.only_manipulation:
        return ()
    return runtime.manipulation_lock_indices


def load_episode_states(
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, object], np.ndarray | None]:
    """Load one episode from the current 16D HandUMI raw layout."""
    ref = DatasetRef.from_repo_id(
        args.repo_id,
        root=args.dataset_root,
        revision=args.revision,
    )
    source_info = ensure_metadata(ref)
    validate_raw_state_metadata(source_info)
    if getattr(args, "gripper_retarget", None) == "physical-width":
        from handumi.dataset.gripper import require_physical_width
        require_physical_width(handumi_metadata(source_info))
    _reject_joint_level_dataset(source_info)
    dataset = open_dataset(ref, episode=args.episode, download_videos=False)
    fps = float(getattr(dataset, "fps", 30) or 30)
    table = dataset.hf_dataset
    if args.source not in table.column_names:
        raise ValueError(f"Dataset has no {args.source!r} feature.")
    states = _column_float32(table, args.source)
    if len(states) == 0:
        raise ValueError(f"Episode {args.episode} is empty.")
    if states.ndim != 2 or states.shape[1] != HANDUMI_RAW_STATE_SIZE:
        width = states.shape[1] if states.ndim == 2 else states.shape
        raise ValueError(
            f"Expected HandUMI state width {HANDUMI_RAW_STATE_SIZE} "
            f"(poses + grippers) in {args.source!r}, got {width}."
        )
    from handumi.dataset.gripper import read_normalized
    normalized = read_normalized(table, handumi_metadata(source_info))
    return states, fps, source_info, normalized


def _reject_joint_level_dataset(source_info: dict[str, object]) -> None:
    """Send an already-converted dataset to the command that can play it.

    A converted dataset inherits the capture's raw-layout tags, so the
    metadata check above passes and the only symptom would be a confusing
    state-width error further down.
    """
    from handumi.dataset.canonical import is_canonical_state_layout
    from handumi.dataset.external_layouts import detect_external_layout

    if is_canonical_state_layout(source_info) or detect_external_layout(source_info):
        raise SystemExit(
            "This dataset already holds joint angles "
            f"(robot_type={source_info.get('robot_type')!r}). `handumi replay` solves "
            "IK from the robot-agnostic capture; use `handumi replay-joints` to play a "
            "converted dataset."
        )


def _column_float32(table: Any, key: str) -> np.ndarray:
    """Read one column as float32 without the per-row torch formatter."""
    arrow = getattr(table, "data", None)
    if arrow is not None and hasattr(arrow, "column"):
        values = arrow.column(key).to_numpy(zero_copy_only=False)
        if len(values) and isinstance(values[0], np.ndarray):
            return np.stack(values).astype(np.float32)
        return np.asarray(values, dtype=np.float32)
    return np.asarray(table[key], dtype=np.float32)


def _resolve_gripper_openings(
    states: np.ndarray,
    recorded_normalized: np.ndarray | None,
    *,
    max_width_m: float,
    mode: str = "normalized",
) -> tuple[np.ndarray | None, str]:
    """Return per-frame 0-1 gripper openings and their source description."""
    if mode not in {"normalized", "physical-width"}:
        raise ValueError(f"Unsupported gripper retarget mode {mode!r}.")
    if mode == "normalized" and recorded_normalized is not None:
        return np.clip(recorded_normalized, 0.0, 1.0), "recorded normalized opening"
    if states.ndim != 2 or states.shape[1] != HANDUMI_RAW_STATE_SIZE:
        raise ValueError(
            f"Expected HandUMI states shape (T, {HANDUMI_RAW_STATE_SIZE}), "
            f"got {states.shape}."
        )
    if max_width_m <= 0:
        raise SystemExit("--gripper-max-width-m must be greater than zero.")
    widths_m = states[:, [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]]
    return np.clip(widths_m / float(max_width_m), 0.0, 1.0), "state widths in meters"


def _resolved_controller_device(
    args: argparse.Namespace,
    source_info: dict[str, object],
) -> str:
    """优先使用显式设备参数，其次读取录制设备，旧数据沿用默认设备。"""
    if args.controller_device is not None:
        snapshot = _metadata_tcp_snapshot(source_info)
        if snapshot is not None and _is_trusted_tcp_snapshot(snapshot):
            recorded_device = str(snapshot["tracking_device"])
            if str(args.controller_device) != recorded_device:
                raise SystemExit(
                    f"--controller-device {args.controller_device!r} conflicts with "
                    f"the identity-bound dataset snapshot ({recorded_device!r})."
                )
        return str(args.controller_device)
    recorded = handumi_metadata(source_info).get("recording_device")
    if recorded in ("pico", "meta", "vive"):
        return str(recorded)
    return DEFAULT_CONTROLLER_DEVICE


def _resolved_retarget_mode(
    args: argparse.Namespace,
    source_info: dict[str, object],
) -> str:
    """Choose geometry-preserving replay when the dataset has a table frame."""
    requested = str(args.retarget_mode)
    if requested != "auto":
        return requested
    workspace = handumi_metadata(source_info).get("tracking_workspace")
    return "absolute-table" if workspace == "table" else "local-relative"


def _metadata_tcp_calibration(
    source_info: dict[str, object],
) -> tuple[ControllerTcpCalibration, str] | None:
    snapshot = _metadata_tcp_snapshot(source_info)
    if not isinstance(snapshot, dict) or snapshot.get("applied_to_state") is True:
        return None
    calibration = controller_tcp_calibration_from_metadata(snapshot)
    sha256 = str(snapshot.get("sha256", "unknown"))
    identity = _metadata_tcp_identity(source_info)
    if _is_trusted_tcp_snapshot(snapshot):
        source = (
            "dataset robot-tool snapshot "
            f"{identity['source_robot']}/{identity['tracking_device']} "
            f"gripper={identity['source_gripper']} "
            f"mount={identity['controller_mount']} sha256={sha256}"
        )
    else:
        source = f"legacy dataset metadata sha256={sha256}"
    return calibration, source


def _metadata_tcp_snapshot(source_info: dict[str, object]) -> dict[str, object] | None:
    snapshot = handumi_metadata(source_info).get("controller_tcp_calibration")
    return snapshot if isinstance(snapshot, dict) else None


def _metadata_tcp_identity(source_info: dict[str, object]) -> dict[str, str]:
    metadata = handumi_metadata(source_info)
    snapshot = _metadata_tcp_snapshot(source_info) or {}
    target_robot = metadata.get("target_robot")
    target_robot_name = (
        str(target_robot.get("name"))
        if isinstance(target_robot, dict) and target_robot.get("name")
        else ""
    )
    return {
        "source_robot": str(snapshot.get("source_robot") or target_robot_name),
        "source_gripper": str(snapshot.get("source_gripper") or "unknown"),
        "tracking_device": str(
            snapshot.get("tracking_device")
            or metadata.get("recording_device")
            or "unknown"
        ),
        "controller_mount": str(snapshot.get("controller_mount") or "unknown"),
    }


def _is_trusted_tcp_snapshot(snapshot: dict[str, object]) -> bool:
    return is_identity_bound_controller_tcp_metadata(snapshot)


def _source_robot_from_metadata(
    source_info: dict[str, object],
    *,
    fallback: str,
) -> str:
    identity = _metadata_tcp_identity(source_info)
    return identity["source_robot"] or fallback


def _resolved_tcp_calibration(
    args: argparse.Namespace,
    source_info: dict[str, object],
    *,
    robot: str,
    controller_device: str,
    configured_path: Path | None,
    configured_gripper: str | None = None,
    configured_mount: str | None = None,
) -> TcpCalibrationSelection:
    """Resolve explicit, trusted dataset, robot setup, then legacy calibration."""
    explicit_path = args.controller_tcp_calibration
    use_dataset = bool(getattr(args, "use_dataset_tcp_calibration", False))
    if explicit_path is not None and use_dataset:
        raise SystemExit(
            "--controller-tcp-calibration and --use-dataset-tcp-calibration "
            "are mutually exclusive."
        )
    if explicit_path is not None:
        calibration = load_controller_tcp_calibration(explicit_path)
        sha256 = controller_tcp_calibration_sha256(explicit_path)
        return TcpCalibrationSelection(
            calibration=calibration,
            source=f"explicit {explicit_path} sha256={sha256}",
            source_robot=robot,
            source_gripper=configured_gripper or "unknown",
            tracking_device=controller_device,
            controller_mount=configured_mount or "unknown",
            trusted_dataset_snapshot=False,
        )

    snapshot = _metadata_tcp_snapshot(source_info)
    metadata_calibration = _metadata_tcp_calibration(source_info)
    if use_dataset:
        if metadata_calibration is None:
            raise SystemExit("Dataset has no unapplied controller->TCP calibration snapshot.")
        calibration, source = metadata_calibration
        identity = _metadata_tcp_identity(source_info)
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            trusted_dataset_snapshot=bool(
                snapshot is not None and _is_trusted_tcp_snapshot(snapshot)
            ),
            **identity,
        )

    if (
        snapshot is not None
        and _is_trusted_tcp_snapshot(snapshot)
        and metadata_calibration is not None
    ):
        calibration, source = metadata_calibration
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            trusted_dataset_snapshot=True,
            **_metadata_tcp_identity(source_info),
        )

    if configured_path is not None:
        calibration = load_controller_tcp_calibration(configured_path)
        sha256 = controller_tcp_calibration_sha256(configured_path)
        source = (
            f"configured {robot}/{controller_device}: {configured_path} "
            f"sha256={sha256}"
        )
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            source_robot=robot,
            source_gripper=configured_gripper or "unknown",
            tracking_device=controller_device,
            controller_mount=configured_mount or "unknown",
            trusted_dataset_snapshot=False,
        )

    if metadata_calibration is not None:
        calibration, source = metadata_calibration
        identity = _metadata_tcp_identity(source_info)
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            trusted_dataset_snapshot=False,
            **identity,
        )

    fallback_path = calibration_path_for_device(controller_device)
    calibration = load_controller_tcp_calibration(fallback_path)
    sha256 = controller_tcp_calibration_sha256(fallback_path)
    return TcpCalibrationSelection(
        calibration=calibration,
        source=f"legacy device fallback {fallback_path} sha256={sha256}",
        source_robot=robot,
        source_gripper=configured_gripper or "unknown",
        tracking_device=controller_device,
        controller_mount=configured_mount or "unknown",
        trusted_dataset_snapshot=False,
    )


def _deployment_npz_entries(calibration) -> dict[str, np.ndarray]:
    """Serialize placement provenance from the one canonical metadata dict."""
    if calibration is not None:
        metadata = deployment_calibration_metadata(calibration)
    else:
        metadata = {
            "robot": "",
            "path": "",
            "profile": "",
            "scope": "",
            "source": "",
            "verified": False,
            "lab": None,
        }
    entries: dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            entries[f"deployment_calibration_{key}"] = np.asarray([value], dtype=np.bool_)
        else:
            entries[f"deployment_calibration_{key}"] = np.asarray(
                [str(value) if value is not None else ""]
            )
    return entries


def apply_tcp_calibration_to_states(
    states: np.ndarray,
    calibration: ControllerTcpCalibration,
    calibration_source: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Return states whose left/right poses are calibrated gripper TCP poses."""
    raw_left: list[np.ndarray] = []
    raw_right: list[np.ndarray] = []
    for state in states:
        left, right = raw_state_pose7_pair(state)
        raw_left.append(left)
        raw_right.append(right)

    raw_left_arr = np.asarray(raw_left, dtype=np.float32)
    raw_right_arr = np.asarray(raw_right, dtype=np.float32)
    left_tcp, right_tcp = apply_controller_tcp_calibration(
        raw_left_arr,
        raw_right_arr,
        calibration,
    )

    calibrated = np.asarray(states, dtype=np.float32).copy()
    calibrated[:, LEFT_POSE_SLICE] = left_tcp
    calibrated[:, RIGHT_POSE_SLICE] = right_tcp
    return (
        calibrated,
        raw_left_arr,
        raw_right_arr,
        left_tcp,
        right_tcp,
        calibration_source,
    )


def _tcp_geometry_diagnostics(
    calibration: ControllerTcpCalibration,
    left_tcp_pose7: np.ndarray,
    right_tcp_pose7: np.ndarray,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> dict[str, np.ndarray]:
    """Summarize source geometry before IK or deployment transforms are applied."""
    left = np.asarray(left_tcp_pose7, dtype=np.float32)
    right = np.asarray(right_tcp_pose7, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left/right calibrated TCP trajectories must have equal 2-D shapes")
    if left.shape[1] < 3 or len(left) == 0:
        raise ValueError("calibrated TCP trajectories must contain at least one position")
    separation = np.linalg.norm(left[:, :3] - right[:, :3], axis=1)
    poses = {"left": left, "right": right}
    combined_positions = np.concatenate([poses[side][:, :3] for side in active_sides], axis=0)
    return {
        "offset_position_norm_m": np.asarray(
            [
                np.linalg.norm(calibration.left[:3]),
                np.linalg.norm(calibration.right[:3]),
            ],
            dtype=np.float32,
        ),
        "workspace_min_z_m": np.asarray(
            [np.min(left[:, 2]), np.min(right[:, 2])],
            dtype=np.float32,
        ),
        "same_frame_min_separation_m": np.asarray(
            [np.min(separation)],
            dtype=np.float32,
        ),
        "same_frame_max_separation_m": np.asarray(
            [np.max(separation)],
            dtype=np.float32,
        ),
        "workspace_position_min_m": np.min(combined_positions, axis=0).astype(
            np.float32
        ),
        "workspace_position_max_m": np.max(combined_positions, axis=0).astype(
            np.float32
        ),
    }


def _print_tcp_geometry_diagnostics(
    selection: TcpCalibrationSelection,
    diagnostics: dict[str, np.ndarray],
    *,
    workspace: str,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> None:
    offsets = diagnostics["offset_position_norm_m"]
    min_z = diagnostics["workspace_min_z_m"]
    min_separation = float(diagnostics["same_frame_min_separation_m"][0])
    max_separation = float(diagnostics["same_frame_max_separation_m"][0])
    workspace_min = diagnostics["workspace_position_min_m"]
    workspace_max = diagnostics["workspace_position_max_m"]
    print(
        "[replay] source tool: "
        f"robot={selection.source_robot} gripper={selection.source_gripper} "
        f"device={selection.tracking_device} mount={selection.controller_mount}"
    )
    print(f"[replay] TCP calibration: {selection.source}")
    print(
        "[replay] Controller->TCP position distance: "
        + " ".join(f"{side}={float(offsets[i]) * 100:.1f}cm"
                   for i, side in enumerate(("left", "right")) if side in active_sides)
    )
    separation_text = (
        f"; same-frame separation={min_separation * 100:.1f}..{max_separation * 100:.1f}cm"
        if len(active_sides) == 2 else ""
    )
    print(
        f"[replay] calibrated TCP geometry in {workspace or 'tracking'} frame: "
        + "z_min " + " ".join(f"{side}={float(min_z[i]) * 100:.1f}cm"
                              for i, side in enumerate(("left", "right")) if side in active_sides)
        + separation_text
    )
    print(
        "[replay] source TCP workspace bounds: "
        f"min=[{workspace_min[0]:.3f}, {workspace_min[1]:.3f}, {workspace_min[2]:.3f}]m "
        f"max=[{workspace_max[0]:.3f}, {workspace_max[1]:.3f}, {workspace_max[2]:.3f}]m"
    )


class PlacementSearchError(SystemExit):
    """保留候选明细，同时让普通回放错误保持简短。"""

    def __init__(self, message, attempts):
        """记录面向用户的摘要及供诊断工具使用的结构化候选。"""
        super().__init__(message)
        self.attempts = attempts


class _PlacementContextReady(Exception):
    """首帧上下文已就绪，可以直接搜索仿真摆放。"""


def solve_episode(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """按配置选择固定摆放或逐段搜索，再执行完整轨迹求解。"""
    import contextlib
    import io

    resolve_replay_options(args)
    policy = getattr(args, "base_placement", "auto")
    if policy == "dataset-xy-yaw" and getattr(args, "deployment_selection", None) is None:
        raise SystemExit("Dataset-wide base placement has not been resolved.")
    if args.retarget_mode != "absolute-tcp" or policy in {"fixed", "dataset-xy-yaw"}:
        return _solve_episode_fixed(argparse.Namespace(**vars(args)))

    attempt = argparse.Namespace(**vars(args))
    attempt._placement_context = {}
    attempt._placement_context_only = args.robot == "rm75b"
    attempt.strict_ik = False
    placement_started = time.perf_counter()
    baseline = None
    failure = None
    baseline_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(baseline_output):
            baseline = _solve_episode_fixed(attempt)
    except _PlacementContextReady:
        pass
    except SystemExit as exc:
        failure = exc
    attempt._placement_context_only = False
    context = attempt._placement_context
    if not context:
        if failure is not None:
            raise failure
        return baseline
    calibration = context["deployment"]
    if policy == "auto" and calibration.profile != "identity":
        # 明确或已登记的摆放保持固定，并保留原来的 strict 行为。
        if baseline is None or (args.strict_ik and not _rollout_within_thresholds(baseline, args)):
            return _solve_episode_fixed(argparse.Namespace(**vars(args)))
        print(baseline_output.getvalue(), end="")
        return baseline
    if policy == "episode-xy" and calibration.scope == "physical":
        raise SystemExit("--base-placement episode-xy cannot modify a physical table calibration; use fixed or a simulation placement.")

    table_from_base = pose_inv(calibration.pose7)
    if baseline is not None and _rollout_within_thresholds(baseline, args) and np.linalg.norm(table_from_base[:2]) < 1e-8:
        print(baseline_output.getvalue(), end="")
        return _record_base_placement(baseline, table_from_base, "episode-xy", {
            "algorithm": ALGORITHM_VERSION, "oracle_calls": 1, "zero_preserved": True,
            "elapsed_seconds": time.perf_counter() - placement_started,
        })
    print(f"[replay] episode {args.episode}: searching a fixed XY base placement")
    report(args, "生成首帧摆放候选")
    fast_search = args.robot == "rm75b"
    candidate_args = (context["runtime"], context["tool"], context["targets"],
                      table_from_base, context["locked"])
    candidates = (first_frame_candidates(*candidate_args, max_seeds=1) if fast_search
                  else first_frame_candidates(*candidate_args))
    if baseline is not None and _rollout_within_thresholds(baseline, args):
        from handumi.retargeting.base_placement import PlacementCandidate
        candidates.append(PlacementCandidate(table_from_base[:2].copy(), baseline["qpos"][0].copy()))
        candidates.sort(key=lambda item: float(np.linalg.norm(item.xy)))

    last_failure = None
    attempts = []
    best_trajectory_failure = None
    candidate_count = 0

    def evaluate(candidate):
        """以候选固定 base 和相应首帧关节分支验证完整轨迹。"""
        nonlocal last_failure, best_trajectory_failure, candidate_count
        candidate_count += 1
        report(args, "基座搜索", candidate=candidate_count)
        base = table_from_base.copy()
        base[:2] = candidate.xy
        trial = argparse.Namespace(**vars(attempt))
        trial._placement_candidate = candidate_count
        trial._placement_override = replace(
            calibration, pose7=pose_inv(base), path=None, scope="simulation",
            source="episode_xy_search", verified=False, profile="episode-xy",
        )
        # 优先使用优化出的关节分支；该分支整段失败时再尝试 HOME 分支。
        for seed in (candidate.q0, None):
            trial._initial_q_seed = seed
            record = {"xy_m": candidate.xy.tolist(),
                      "seed_qpos": None if seed is None else seed.tolist()}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rollout = _solve_episode_fixed(trial)
            except SystemExit as exc:
                last_failure = str(exc)
                record.update({"passed": False, "error": last_failure})
                attempts.append(record)
                continue
            pos = np.maximum(rollout["left_pos_error_m"], rollout["right_pos_error_m"])
            rot = np.maximum(rollout["left_rot_error_deg"], rollout["right_rot_error_deg"])
            worst = int(np.argmax(np.maximum(pos / args.max_ik_position_error_m,
                                             rot / args.max_ik_rotation_error_deg)))
            last_failure = (f"frame {int(rollout['frame_indices'][worst])}: "
                            f"{float(pos[worst]) * 1000:.2f}mm/{float(rot[worst]):.2f}deg")
            score = float(max(pos[worst] / args.max_ik_position_error_m,
                              rot[worst] / args.max_ik_rotation_error_deg))
            passed = _rollout_within_thresholds(rollout, args)
            record.update({"passed": passed, "worst_frame": int(rollout["frame_indices"][worst]),
                           "position_error_m": float(pos[worst]),
                           "rotation_error_deg": float(rot[worst]), "score": score})
            attempts.append(record)
            if passed:
                return rollout
            if best_trajectory_failure is None or score < best_trajectory_failure["score"]:
                best_trajectory_failure = record.copy()
        return None

    try:
        try:
            candidate, result, diagnostics = search_feasible_placement(
                candidates, evaluate, first_feasible=fast_search
            )
        except ValueError:
            if not fast_search:
                raise
            # 快速种子无法覆盖全部 IK 分支，失败时恢复完整种子搜索。
            candidates = first_frame_candidates(*candidate_args)
            candidate, result, diagnostics = search_feasible_placement(
                candidates, evaluate, first_feasible=True
            )
    except ValueError as exc:
        raise PlacementSearchError(
            f"Unable to find strict episode XY placement: {exc}; best trajectory failure: {best_trajectory_failure}; last failure: {last_failure}",
            attempts,
        ) from exc
    base = table_from_base.copy()
    base[:2] = candidate.xy
    diagnostics.update({"algorithm": ALGORITHM_VERSION, "zero_preserved": False,
                        "attempts": attempts, "best_trajectory_failure": best_trajectory_failure,
                        "elapsed_seconds": time.perf_counter() - placement_started})
    print(f"[replay] episode {args.episode}: base in table xy={base[:2].tolist()} m; "
          f"distance={np.linalg.norm(base[:2]):.4f} m; candidates={diagnostics['oracle_calls']}")
    return _record_base_placement(result, base, "episode-xy", diagnostics)


def _rollout_within_thresholds(rollout: dict[str, np.ndarray], args: argparse.Namespace) -> bool:
    """判断全段 TCP 误差是否满足用户设置的严格门槛。"""
    for side in ("left", "right"):
        pos = np.asarray(rollout[f"{side}_pos_error_m"])
        rot = np.asarray(rollout[f"{side}_rot_error_deg"])
        if (not np.isfinite(pos).all() or not np.isfinite(rot).all()
                or np.any(pos > args.max_ik_position_error_m)
                or np.any(rot > args.max_ik_rotation_error_deg)):
            return False
    return True


def _record_base_placement(rollout, table_from_base, policy, diagnostics):
    """将固定的每段 base 摆放和数值搜索说明写入回放结果。"""
    rollout["table_from_robot_pose7"] = np.asarray([table_from_base], dtype=np.float32)
    rollout["base_placement_policy"] = np.asarray([policy])
    rollout["base_placement_scope"] = np.asarray(["simulation"])
    rollout["base_placement_verified"] = np.asarray([False])
    rollout["base_placement_diagnostics_json"] = np.asarray([json.dumps(diagnostics, sort_keys=True)])
    return rollout


def _solve_episode_fixed(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """使用指定的固定桌面到机器人变换求解整条轨迹。"""
    report(args, "模型与轨迹初始化", candidate=getattr(args, "_placement_candidate", None))
    resolve_replay_options(args)
    runtime = getattr(args, "_episode_runtime", None)
    if runtime is None:
        if args.robot == "rm75b" and args.retarget_mode == "absolute-tcp":
            runtime = args._episode_runtime = load_embodiment(args.robot, collision_kind="gripper-box")
        else:
            runtime = args._episode_runtime = load_embodiment(args.robot)
    loaded = getattr(args, "_episode_loaded", None)
    if loaded is None:
        loaded = args._episode_loaded = load_episode_states(args)
    states, fps, source_info, recorded_gripper_openings = loaded
    source_metadata = handumi_metadata(source_info)
    from handumi.config import dataset_active_sides

    missing = set(runtime.arms) - set(dataset_active_sides(source_metadata))
    if missing:
        raise ValueError(f"Dataset does not record the sides required by {args.robot}: {sorted(missing)}")
    controller_device = _resolved_controller_device(args, source_info)
    requested_retarget_mode = str(args.retarget_mode)
    retarget_mode = _resolved_retarget_mode(args, source_info)
    if retarget_mode == "absolute-tcp" and args.raw_controller_debug:
        raise SystemExit("absolute-tcp requires calibrated TCP poses, not raw controller debug.")
    if (retarget_mode == "absolute-tcp" and
            (_metadata_tcp_snapshot(source_info) or {}).get("applied_to_state") is True):
        raise SystemExit("absolute-tcp requires the supported raw controller state layout; the dataset reports an already-applied TCP calibration.")
    if retarget_mode == "absolute-tcp":
        source_indices = list(range(args.start_frame, len(states), args.stride))
        if args.max_frames is not None:
            source_indices = source_indices[:args.max_frames]
        for frame in source_indices:
            state = states[frame]
            for side, pose_slice in (("left", LEFT_POSE_SLICE), ("right", RIGHT_POSE_SLICE)):
                if side in runtime.arms:
                    pose = state[pose_slice]
                    _require_finite_frame(frame, f"{side} source pose", pose)
                    if float(np.linalg.norm(pose[3:7])) < 1e-8:
                        raise SystemExit(f"Invalid zero {side} source quaternion at frame {frame}.")
    if runtime.config.replay_gripper_mode == "physical-width":
        from handumi.dataset.gripper import require_physical_width
        require_physical_width(source_metadata)
    gripper_openings, gripper_source = _resolve_gripper_openings(
        states,
        recorded_gripper_openings,
        max_width_m=(
            runtime.config.gripper_max_width_m
            if args.gripper_max_width_m is None
            else args.gripper_max_width_m
        ),
        mode=runtime.config.replay_gripper_mode,
    )
    if args.raw_controller_debug:
        states_for_retarget = states
        raw_left_controller: list[np.ndarray] = []
        raw_right_controller: list[np.ndarray] = []
        for state in states:
            left, right = raw_state_pose7_pair(state)
            raw_left_controller.append(left)
            raw_right_controller.append(right)
        raw_left_controller_arr = np.asarray(raw_left_controller, dtype=np.float32)
        raw_right_controller_arr = np.asarray(raw_right_controller, dtype=np.float32)
        left_tcp_arr = raw_left_controller_arr
        right_tcp_arr = raw_right_controller_arr
        calibration_source = None
        tcp_selection = None
        tcp_diagnostics: dict[str, np.ndarray] = {}
    else:
        source_robot = _source_robot_from_metadata(
            source_info,
            fallback=args.robot,
        )
        source_config = None
        try:
            source_config = load_robot_config(source_robot)
        except (TypeError, ValueError) as exc:
            snapshot = _metadata_tcp_snapshot(source_info)
            can_replay_without_profile = (
                args.controller_tcp_calibration is not None
                or bool(getattr(args, "use_dataset_tcp_calibration", False))
                or (snapshot is not None and _is_trusted_tcp_snapshot(snapshot))
            )
            if not can_replay_without_profile:
                raise SystemExit(
                    f"Cannot resolve source robot-tool calibration: {exc}"
                ) from exc

        tcp_selection = _resolved_tcp_calibration(
            args,
            source_info,
            robot=source_robot,
            controller_device=controller_device,
            configured_path=(
                source_config.controller_tcp_calibrations.get(controller_device)
                if source_config is not None
                else None
            ),
            configured_gripper=(
                source_config.handumi_gripper if source_config is not None else None
            ),
            configured_mount=(
                source_config.handumi_controller_mount
                if source_config is not None
                else None
            ),
        )
        if retarget_mode == "absolute-tcp" and tcp_selection.source.startswith("legacy device fallback"):
            raise SystemExit("absolute-tcp requires a dataset, configured, or explicit controller-to-TCP calibration; a legacy device fallback cannot establish the source tool frame.")
        (
            states_for_retarget,
            raw_left_controller_arr,
            raw_right_controller_arr,
            left_tcp_arr,
            right_tcp_arr,
            calibration_source,
        ) = apply_tcp_calibration_to_states(
            states,
            tcp_selection.calibration,
            tcp_selection.source,
        )
        tcp_diagnostics = _tcp_geometry_diagnostics(
            tcp_selection.calibration,
            left_tcp_arr,
            right_tcp_arr,
            active_sides=runtime.active_sides,
        )
        _print_tcp_geometry_diagnostics(
            tcp_selection,
            tcp_diagnostics,
            workspace=str(source_metadata.get("tracking_workspace", "")),
            active_sides=runtime.active_sides,
        )

    frame_indices = list(range(args.start_frame, len(states), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected for replay.")
    placement_score = getattr(args, "_placement_score", None)
    if placement_score is not None:
        placement_score.configure_frames(len(frame_indices), int(frame_indices[0]))
    if source_metadata.get("tracking_workspace") == "table" and tcp_diagnostics:
        minimum_tcp_z = min(float(tcp_diagnostics["workspace_min_z_m"][i])
                           for i, side in enumerate(("left", "right")) if side in runtime.arms)
        if minimum_tcp_z > args.table_clearance_warning_m:
            print(
                "[replay] warning: calibrated TCP never approaches table z=0 "
                f"(minimum={minimum_tcp_z * 100:.1f}cm). Re-run controller->TCP "
                "pivot calibration; do not compensate this with robot_from_table."
            )

    cfg = runtime.config.ik_weights
    if runtime.config.replay_max_joint_delta is not None:
        cfg = replace(
            cfg,
            max_joint_delta=runtime.config.replay_max_joint_delta,
        )
    if runtime.config.replay_ik_weights:
        cfg = replace(
            cfg,
            **{
                f"{key}_weight": value
                for key, value in runtime.config.replay_ik_weights.items()
            },
        )
        overrides = ", ".join(
            f"{key}={value:g}"
            for key, value in sorted(runtime.config.replay_ik_weights.items())
        )
        print(f"[replay] ik_weights overridden for replay: {overrides}")
    q = runtime.config.home_q.astype(np.float32).copy()
    locked_joint_indices = _replay_locked_joint_indices(runtime, args)
    solver = runtime.solver_cls(
        config=cfg,
        locked_joint_indices=locked_joint_indices,
    )
    collision_checker = None
    if args.robot == "rm75b" and cfg.collision_enabled:
        from handumi.robots.self_collision import CollisionChecker, require_clear

        collision_checker = getattr(args, "_episode_collision_checker", None)
        if collision_checker is None:
            urdf = runtime.load_urdf()
            if retarget_mode == "absolute-tcp":
                from handumi.robots.self_collision import GripperBoxCollisionChecker
                collision_checker = GripperBoxCollisionChecker(
                    runtime.robot, solver.robot_collision, urdf,
                    cfg.self_collision_margin,
                )
            else:
                collision_checker = CollisionChecker(
                    runtime.robot, solver.robot_collision, urdf, cfg.self_collision_margin,
                )
            # 单段基座搜索共用同一模型与配置，不重复建立网格索引。
            args._episode_collision_checker = collision_checker
        if args.approach_seconds > 0 or retarget_mode != "absolute-tcp":
            require_clear(collision_checker.check_pose(q), phase="home", frame="home")
        # 粗包络只引导重试；优先让无碰撞代价的候选经真实网格严格验收。
        solver.config = replace(cfg, self_collision_weight=0.0)
        if retarget_mode == "absolute-tcp":
            solver.robot_collision = collision_checker.model
    qs: list[np.ndarray] = []
    raw_left_gt: list[np.ndarray] = []
    raw_right_gt: list[np.ndarray] = []
    left_targets: list[np.ndarray] = []
    right_targets: list[np.ndarray] = []
    left_achieved: list[np.ndarray] = []
    right_achieved: list[np.ndarray] = []
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    first_left_pose7, first_right_pose7 = raw_state_pose7_pair(
        states_for_retarget[frame_indices[0]]
    )
    anchors = None
    left_adapter = None
    right_adapter = None
    robot_from_table = None
    deployment_calibration = None
    robot_tcp = None
    left_tool_adapter = None
    right_tool_adapter = None
    initial_solve_count = 0
    initial_max_position_error_m = 0.0
    initial_max_rotation_error_deg = 0.0
    if retarget_mode == "anchored":
        anchors = retarget_anchors_from_raw_state(
            states_for_retarget[frame_indices[0]],
            left_robot_pose7=home_left_pose7,
            right_robot_pose7=home_right_pose7,
            max_reach=cfg.max_reach,
        )
    elif retarget_mode == "local-relative":
        world_map = (
            VR_TO_ROBOT
            if controller_device == "pico"
            else np.eye(3, dtype=np.float32)
        )
        left_adapter = local_frame_adapter(
            first_left_pose7,
            home_left_pose7,
            source_world_to_robot_world=world_map,
        )
        right_adapter = local_frame_adapter(
            first_right_pose7,
            home_right_pose7,
            source_world_to_robot_world=world_map,
        )
    else:
        if source_metadata.get("tracking_workspace") != "table":
            raise SystemExit(
                f"--retarget-mode {retarget_mode} requires a dataset recorded in the "
                "calibrated table workspace."
            )
        deployment_calibration = (getattr(args, "_placement_override", None)
                                  or getattr(args, "deployment_selection", None))
        if deployment_calibration is None:
            from handumi.scripts.replay.replay_deployment import (
                estimate_dataset_deployment,
            )

            deployment_calibration = resolve_deployment_calibration(
                args.robot,
                explicit_path=args.deployment_calibration,
                profile=args.deployment_profile,
                rig_config=args.rig_config,
                identity_fallback=retarget_mode == "absolute-tcp",
                simulation_fallback=lambda: estimate_dataset_deployment(args),
            )
        robot_from_table = deployment_calibration.pose7
        print_deployment_calibration(deployment_calibration)
        print(
            "[replay] robot_from_table: "
            f"translation=[{robot_from_table[0]:.4f}, {robot_from_table[1]:.4f}, "
            f"{robot_from_table[2]:.4f}]m source={deployment_calibration.path}"
        )
        mapped_left, mapped_right = absolute_table_robot_target_pose7(
            states_for_retarget[frame_indices[0]],
            robot_from_table,
        )
        if retarget_mode == "absolute-tcp":
            robot_tcp = load_robot_tcp_calibration(
                args.robot,
                {side: runtime.config.arms[side] for side in runtime.arms},
                explicit_path=getattr(args, "robot_tcp_calibration", None),
            )
            print(f"[replay] robot TCP: {robot_tcp.path} sha256={robot_tcp.sha256} verified={robot_tcp.verified}")
            if hasattr(args, "_placement_context"):
                source_left, source_right = raw_state_pose7_pair(states_for_retarget[frame_indices[0]])
                args._placement_context.update({
                    "runtime": runtime, "tool": robot_tcp, "deployment": deployment_calibration,
                    "targets": {"left": source_left, "right": source_right},
                    "locked": locked_joint_indices,
                    "states_for_retarget": states_for_retarget,
                    "gripper_openings": gripper_openings,
                    "collision_checker": collision_checker,
                })
                if getattr(args, "_dataset_placement_context_only", False):
                    raise _PlacementContextReady
                if (getattr(args, "_placement_context_only", False)
                        and getattr(args, "deployment_selection", None) is None
                        and getattr(args, "deployment_calibration", None) is None
                        and (getattr(args, "base_placement", "auto") == "episode-xy"
                             or deployment_calibration.profile == "identity")
                        and deployment_calibration.scope != "physical"):
                    raise _PlacementContextReady
            if "left" in runtime.arms:
                home_left_pose7 = robot_tcp.achieved_tcp("left", home_left_pose7)
            if "right" in runtime.arms:
                home_right_pose7 = robot_tcp.achieved_tcp("right", home_right_pose7)
        if args.absolute_orientation == "relative-start":
            left_tool_adapter = orientation_only_pose_adapter(
                mapped_left, home_left_pose7
            )
            right_tool_adapter = orientation_only_pose_adapter(
                mapped_right, home_right_pose7
            )

        first_left_target, first_right_target = absolute_table_robot_target_pose7(
            states_for_retarget[frame_indices[0]],
            robot_from_table,
            left_tool_adapter_pose7=left_tool_adapter,
            right_tool_adapter_pose7=right_tool_adapter,
        )
        if "left" not in runtime.arms:
            first_left_target = home_left_pose7.copy()
        if "right" not in runtime.arms:
            first_right_target = home_right_pose7.copy()
        first_left_ee = (
            robot_tcp.target_ee("left", first_left_target)
            if robot_tcp is not None and "left" in runtime.arms else first_left_target
        )
        first_right_ee = (
            robot_tcp.target_ee("right", first_right_target)
            if robot_tcp is not None and "right" in runtime.arms else first_right_target
        )
        initial_solver = runtime.solver_cls(
            config=replace(cfg, max_joint_delta=None),
            locked_joint_indices=locked_joint_indices,
        )
        if collision_checker is not None:
            initial_solver.config = replace(initial_solver.config, self_collision_weight=0.0)
            if retarget_mode == "absolute-tcp":
                initial_solver.robot_collision = collision_checker.model
        # 绝对 TCP 模式从 HOME 求首帧；旧模式保留原有工作姿态种子。
        initial_seed = getattr(args, "_initial_q_seed", None)
        q = (np.asarray(initial_seed, dtype=np.float32).copy() if initial_seed is not None
             else runtime.config.home_q.astype(np.float32).copy() if robot_tcp is not None
             else posture_seed(runtime.config.posture_q, runtime.config.home_q))
        report(args, "首帧求解（可能含编译）", candidate=getattr(args, "_placement_candidate", None))
        for initial_solve_count in range(1, args.initial_solve_iterations + 1):
            initial_previous_q = q.copy()
            if initial_seed is None or initial_solve_count > 1:
                q = initial_solver.ik(
                    q,
                    left_pose=(first_left_ee[:3], first_left_ee[3:7]),
                    right_pose=(first_right_ee[:3], first_right_ee[3:7]),
                )
            if collision_checker is not None:
                from handumi.robots.self_collision import retry_checked_candidate

                q = retry_checked_candidate(
                    solver=initial_solver, checker=collision_checker,
                    previous=initial_previous_q, candidate=q,
                    left_pose=(first_left_ee[:3], first_left_ee[3:7]),
                    right_pose=(first_right_ee[:3], first_right_ee[3:7]),
                    assess=lambda _q: None, frame=int(frame_indices[0]),
                    check_motion=False,
                )
            initial_left, initial_right = initial_solver.fk_pose7(q)
            if robot_tcp is not None:
                _require_finite_frame(frame_indices[0], "initial IK joint or FK", q, initial_left, initial_right)
                if "left" in runtime.arms:
                    initial_left = robot_tcp.achieved_tcp("left", initial_left)
                if "right" in runtime.arms:
                    initial_right = robot_tcp.achieved_tcp("right", initial_right)
            initial_errors = pose_error_arrays(
                first_left_target[None],
                first_right_target[None],
                initial_left[None],
                initial_right[None],
            )
            if robot_tcp is not None:
                _require_finite_frame(frame_indices[0], "initial TCP error", *initial_errors.values())
            error_sides = runtime.arms if robot_tcp is not None else ("left", "right")
            initial_max_position_error_m = max(
                float(initial_errors[f"{side}_pos_error_m"][0]) for side in error_sides
            )
            initial_max_rotation_error_deg = max(
                float(initial_errors[f"{side}_rot_error_deg"][0]) for side in error_sides
            )
            if (initial_max_position_error_m <= args.initial_position_tolerance_m
                    and (robot_tcp is None or initial_max_rotation_error_deg <= args.max_ik_rotation_error_deg)):
                break
        if (initial_max_position_error_m > args.initial_position_tolerance_m
                or (robot_tcp is not None and initial_max_rotation_error_deg > args.max_ik_rotation_error_deg)):
            raise SystemExit(
                "Unable to prepare replay start pose: maximum TCP position and rotation errors "
                f"are {initial_max_position_error_m * 100:.2f} cm and {initial_max_rotation_error_deg:.2f} deg after "
                f"{initial_solve_count} IK iterations. Check robot_from_table, TCP "
                "calibration, or robot reachability. Pass a dataset-specific "
                "--deployment-calibration or raise --initial-position-tolerance-m "
                "to visualize; use --retarget-mode local-relative when morphology "
                "adaptation is acceptable."
            )

    start = time.perf_counter()
    # 关键帧之间保留每个原始帧，误差或碰撞不达标时重算局部轨迹。
    keyframe_end = -1
    keyframe_q = None
    for selected_index, frame_index in enumerate(frame_indices):
        placement_score = getattr(args, "_placement_score", None)
        if placement_score is not None:
            placement_score.start_frame(int(frame_index), len(frame_indices))
        report(args, "逐帧求解", current=selected_index, total=len(frame_indices),
               candidate=getattr(args, "_placement_candidate", None))
        previous_q = q.copy()
        state = states_for_retarget[frame_index]
        raw_left, raw_right = raw_state_pose7_pair(state)
        if retarget_mode == "anchored":
            if anchors is None:
                raise RuntimeError("Anchored retarget mode was not initialized.")
            left_pose7, right_pose7 = raw_state_robot_target_pose7(state, anchors)
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        elif retarget_mode in {"absolute-table", "absolute-tcp"}:
            if robot_from_table is None:
                raise RuntimeError("Absolute table retarget mode was not initialized.")
            left_pose7, right_pose7 = absolute_table_robot_target_pose7(
                state,
                robot_from_table,
                left_tool_adapter_pose7=left_tool_adapter,
                right_tool_adapter_pose7=right_tool_adapter,
            )
            left_ee = (robot_tcp.target_ee("left", left_pose7)
                       if robot_tcp is not None and "left" in runtime.arms else left_pose7)
            right_ee = (robot_tcp.target_ee("right", right_pose7)
                        if robot_tcp is not None and "right" in runtime.arms else right_pose7)
            if robot_tcp is not None:
                _require_finite_frame(frame_index, "TCP target", left_pose7, right_pose7, left_ee, right_ee)
            if robot_tcp is None or selected_index != 0:
                if args.robot == "rm75b" and retarget_mode == "absolute-tcp":
                    if selected_index > keyframe_end:
                        keyframe_end = min(selected_index + 4, len(frame_indices) - 1)
                        future_state = states_for_retarget[frame_indices[keyframe_end]]
                        future_left, future_right = absolute_table_robot_target_pose7(
                            future_state, robot_from_table,
                            left_tool_adapter_pose7=left_tool_adapter,
                            right_tool_adapter_pose7=right_tool_adapter,
                        )
                        if "left" in runtime.arms:
                            future_left = robot_tcp.target_ee("left", future_left)
                        if "right" in runtime.arms:
                            future_right = robot_tcp.target_ee("right", future_right)
                        keyframe_q = solver.ik(
                            q, left_pose=(future_left[:3], future_left[3:7]),
                            right_pose=(future_right[:3], future_right[3:7]),
                        )
                    q = q + (keyframe_q - q) / (keyframe_end - selected_index + 1)
                else:
                    q = solver.ik(
                        q,
                        left_pose=(left_ee[:3], left_ee[3:7]),
                        right_pose=(right_ee[:3], right_ee[3:7]),
                    )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
            if robot_tcp is not None:
                _require_finite_frame(frame_index, "IK joint or FK", q, fk_left_pose7, fk_right_pose7)
                if "left" in runtime.arms:
                    fk_left_pose7 = robot_tcp.achieved_tcp("left", fk_left_pose7)
                if "right" in runtime.arms:
                    fk_right_pose7 = robot_tcp.achieved_tcp("right", fk_right_pose7)
        elif selected_index == 0:
            left_pose7 = home_left_pose7.copy()
            right_pose7 = home_right_pose7.copy()
            fk_left_pose7 = home_left_pose7.copy()
            fk_right_pose7 = home_right_pose7.copy()
        else:
            if left_adapter is None or right_adapter is None:
                raise RuntimeError("Local-relative retarget mode was not initialized.")
            prev_state = states_for_retarget[frame_indices[selected_index - 1]]
            prev_left, prev_right = raw_state_pose7_pair(prev_state)
            base_left = (
                left_targets[-1]
                if args.compose_source == "commanded"
                else left_achieved[-1]
            )
            base_right = (
                right_targets[-1]
                if args.compose_source == "commanded"
                else right_achieved[-1]
            )
            left_pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=prev_left,
                current_source_pose7=raw_left,
                base_robot_pose7=base_left,
                adapter_rot=left_adapter,
                home_robot_pose7=home_left_pose7,
                translation_scale=args.translation_scale,
                max_reach=cfg.max_reach,
            )
            right_pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=prev_right,
                current_source_pose7=raw_right,
                base_robot_pose7=base_right,
                adapter_rot=right_adapter,
                home_robot_pose7=home_right_pose7,
                translation_scale=args.translation_scale,
                max_reach=cfg.max_reach,
            )
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        # Keep the legacy rollout pose pair neutral for a nonexistent arm;
        # only declared arms participate in IK, errors, and dataset columns.
        if "left" not in runtime.arms:
            left_pose7 = fk_left_pose7.copy()
        if "right" not in runtime.arms:
            right_pose7 = fk_right_pose7.copy()
        opening = None
        if gripper_openings is not None:
            opening = gripper_openings[frame_index]
            runtime.set_finger_positions(
                q,
                {"left": float(opening[0]), "right": float(opening[1])},
            )
        if collision_checker is not None:
            from handumi.robots.self_collision import retry_checked_candidate

            retry_left = (left_ee if retarget_mode in {"absolute-table", "absolute-tcp"}
                          else left_pose7)
            retry_right = (right_ee if retarget_mode in {"absolute-table", "absolute-tcp"}
                           else right_pose7)

            def assess_candidate(candidate):
                """对最终关节角执行当前 TCP 精度阈值检查。"""
                actual_left, actual_right = solver.fk_pose7(candidate)
                if robot_tcp is not None:
                    if "left" in runtime.arms:
                        actual_left = robot_tcp.achieved_tcp("left", actual_left)
                    if "right" in runtime.arms:
                        actual_right = robot_tcp.achieved_tcp("right", actual_right)
                values = pose_error_arrays(
                    left_pose7[None], right_pose7[None],
                    actual_left[None], actual_right[None],
                )
                _require_finite_frame(frame_index, "IK error", *values.values())
                pos = max(float(values[f"{side}_pos_error_m"][0]) for side in runtime.arms)
                rot = max(float(values[f"{side}_rot_error_deg"][0]) for side in runtime.arms)
                if (not np.isfinite([pos, rot]).all()
                        or pos > args.max_ik_position_error_m
                        or rot > args.max_ik_rotation_error_deg):
                    return f"position={pos * 1000:.2f}mm rotation={rot:.2f}deg"
                return None

            def preserve_gripper_opening(candidate):
                """碰撞重试后恢复本帧录制开度，避免求解器改写夹指。"""
                if opening is None:
                    return candidate
                runtime.set_finger_positions(
                    candidate,
                    {"left": float(opening[0]), "right": float(opening[1])},
                )
                return candidate

            interpolated_q = q.copy()
            q = retry_checked_candidate(
                solver=solver, checker=collision_checker, previous=previous_q,
                candidate=q, left_pose=(retry_left[:3], retry_left[3:7]),
                right_pose=(retry_right[:3], retry_right[3:7]),
                assess=assess_candidate, frame=int(frame_index),
                check_motion=selected_index > 0,
                strict_fidelity=args.strict_ik,
                candidate_transform=(
                    preserve_gripper_opening if gripper_openings is not None else None
                ),
            )
            if not np.array_equal(interpolated_q, q):
                keyframe_end = -1
            if args.robot == "rm75b" and retarget_mode == "absolute-tcp":
                lower = np.asarray(runtime.robot.joints.lower_limits)
                upper = np.asarray(runtime.robot.joints.upper_limits)
                if (np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6)
                        or (selected_index and cfg.max_joint_delta is not None
                            and np.any(np.abs(q - previous_q) > cfg.max_joint_delta + 1e-6))):
                    raise SystemExit(f"Joint constraint exceeded: frame {int(frame_index)}")
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
            if robot_tcp is not None:
                if "left" in runtime.arms:
                    fk_left_pose7 = robot_tcp.achieved_tcp("left", fk_left_pose7)
                if "right" in runtime.arms:
                    fk_right_pose7 = robot_tcp.achieved_tcp("right", fk_right_pose7)
            if "left" not in runtime.arms:
                left_pose7 = fk_left_pose7.copy()
            if "right" not in runtime.arms:
                right_pose7 = fk_right_pose7.copy()
            if selected_index == 0:
                # 首帧已经最终确定，提前检查接近段以免对无效摆放求解完整 episode。
                from handumi.robots.self_collision import require_clear

                early_approach = joint_approach_ramp(
                    runtime.config.home_q, q,
                    frames=int(round(fps * args.approach_seconds)),
                )
                if len(early_approach):
                    early_path = [runtime.config.home_q, *early_approach, q]
                    for approach_index in range(1, len(early_path)):
                        require_clear(
                            collision_checker.check_segment(
                                early_path[approach_index - 1], early_path[approach_index]
                            ),
                            phase="approach", frame=approach_index,
                        )
        if robot_tcp is not None:
            _require_finite_frame(frame_index, "output joint", q)
        placement_score = getattr(args, "_placement_score", None)
        if placement_score is not None:
            frame_errors = pose_error_arrays(
                left_pose7[None], right_pose7[None],
                fk_left_pose7[None], fk_right_pose7[None],
            )
            within = all(
                float(frame_errors[f"{side}_pos_error_m"][0]) <= args.max_ik_position_error_m
                and float(frame_errors[f"{side}_rot_error_deg"][0]) <= args.max_ik_rotation_error_deg
                for side in runtime.arms
            )
            placement_score.frame_checked(int(frame_index), within)
        qs.append(q.copy())
        raw_left_gt.append(raw_left)
        raw_right_gt.append(raw_right)
        left_targets.append(left_pose7)
        right_targets.append(right_pose7)
        left_achieved.append(fk_left_pose7)
        right_achieved.append(fk_right_pose7)
    report(args, "逐帧求解", current=len(frame_indices), total=len(frame_indices),
           candidate=getattr(args, "_placement_candidate", None))
    elapsed = time.perf_counter() - start
    approach_qs = joint_approach_ramp(
        runtime.config.home_q.astype(np.float32),
        qs[0],
        frames=int(round(fps * args.approach_seconds)),
    )
    collision_minimum = None
    if collision_checker is not None:
        from handumi.robots.self_collision import require_clear

        collision_minimum = collision_checker.check_pose(qs[0])
        require_clear(collision_minimum, phase="output", frame=int(frame_indices[0]))
        paths = [("output", qs)]
        if len(approach_qs):
            paths.append(("approach", [runtime.config.home_q, *approach_qs, qs[0]]))
        for phase, path in paths:
            for index in range(1, len(path)):
                result = collision_checker.check_segment(path[index - 1], path[index])
                frame = int(frame_indices[index]) if phase == "output" else index
                require_clear(result, phase=phase, frame=frame)
                if result.distance < collision_minimum.distance:
                    collision_minimum = result
        print(f"[replay] self-collision minimum certified pose lower bound={collision_minimum.distance * 1000:.2f}mm "
              f"links={collision_minimum.links[0]}/{collision_minimum.links[1]}")
    target_left = np.asarray(left_targets, dtype=np.float32)
    target_right = np.asarray(right_targets, dtype=np.float32)
    achieved_left = np.asarray(left_achieved, dtype=np.float32)
    achieved_right = np.asarray(right_achieved, dtype=np.float32)
    errors = pose_error_arrays(target_left, target_right, achieved_left, achieved_right)
    if robot_tcp is not None:
        for i, frame in enumerate(frame_indices):
            _require_finite_frame(frame, "TCP error", *(value[i] for value in errors.values()))
    all_pos_err = np.concatenate(
        [errors[f"{side}_pos_error_m"] for side in runtime.arms]
    )
    all_rot_err = np.concatenate(
        [errors[f"{side}_rot_error_deg"] for side in runtime.arms]
    )
    score = optimization_score_from_errors(
        float(all_pos_err.mean() * 100.0),
        float(all_pos_err.max() * 100.0),
        float(all_rot_err.mean()),
        float(all_rot_err.max()),
    )

    print(
        f"[replay] robot={args.robot} episode={args.episode} frames={len(qs)} "
        f"fps={fps:g} solved={elapsed:.2f}s ({elapsed / len(qs) * 1000:.1f} ms/frame)"
    )
    print(
        f"[replay] retarget={retarget_mode} "
        f"controller_device={controller_device} "
        f"compose={args.compose_source} translation_scale={args.translation_scale:g} "
        f"max_joint_delta={cfg.max_joint_delta}"
    )
    if requested_retarget_mode == "auto":
        print(
            "[replay] retarget auto: "
            f"tracking_workspace={source_metadata.get('tracking_workspace')!r} "
            f"-> {retarget_mode}"
        )
    if calibration_source is None:
        print("[replay] input pose: raw controller DEBUG mode")
    else:
        print(f"[replay] input pose: calibrated HandUMI TCP via {calibration_source}")
    print(f"[replay] grippers: {gripper_source}")
    if retarget_mode in {"absolute-table", "absolute-tcp"}:
        print(
            "[replay] start prepared: "
            f"iterations={initial_solve_count} "
            f"pos_max={initial_max_position_error_m * 100:.2f}cm "
            f"rot_max={initial_max_rotation_error_deg:.2f}deg "
            f"orientation={args.absolute_orientation}"
        )
    if len(approach_qs):
        print(
            "[replay] approach: "
            f"{len(approach_qs)} frames ({len(approach_qs) / fps:.2f}s) from home, "
            f"max joint step={float(np.abs(np.diff(np.concatenate([approach_qs, qs[0][None]]), axis=0)).max()):.3f}rad"
        )
    print(
        "[replay] IK EE error: "
        f"pos mean={all_pos_err.mean() * 100:.2f}cm "
        f"max={all_pos_err.max() * 100:.2f}cm; "
        f"rot mean={all_rot_err.mean():.2f}deg max={all_rot_err.max():.2f}deg; "
        f"score={score:.4f}"
    )
    violations = []
    if float(all_pos_err.max()) > args.max_ik_position_error_m:
        violations.append(
            f"position {all_pos_err.max() * 100:.2f}cm > "
            f"{args.max_ik_position_error_m * 100:.2f}cm"
        )
    if float(all_rot_err.max()) > args.max_ik_rotation_error_deg:
        violations.append(
            f"rotation {all_rot_err.max():.2f}deg > "
            f"{args.max_ik_rotation_error_deg:.2f}deg"
        )
    if violations:
        message = "IK fidelity threshold exceeded: " + "; ".join(violations)
        if robot_tcp is not None:
            frame_errors = []
            for i, frame in enumerate(frame_indices):
                pos = max(float(errors[f"{side}_pos_error_m"][i]) for side in runtime.arms)
                rot = max(float(errors[f"{side}_rot_error_deg"][i]) for side in runtime.arms)
                if pos > args.max_ik_position_error_m or rot > args.max_ik_rotation_error_deg:
                    frame_errors.append(f"{frame}: {pos * 1000:.2f}mm/{rot:.2f}deg")
            message += "; frames (position/rotation): " + ", ".join(frame_errors)
        if args.strict_ik:
            raise SystemExit(message)
        print(f"[replay] warning: {message}")
    if collision_checker is not None:
        from handumi.dataset.episode_solve_cache import array_digest
        from handumi.robots.self_collision import (
            COLLISION_VALIDATION_VERSION,
            GRIPPER_BOX_VALIDATION_VERSION,
        )

        collision_summary = {
            "self_collision_validation": np.asarray([
                GRIPPER_BOX_VALIDATION_VERSION if retarget_mode == "absolute-tcp"
                else COLLISION_VALIDATION_VERSION
            ]),
            "self_collision_qpos_sha256": np.asarray([
                array_digest(np.asarray(qs, dtype=np.float32))["sha256"]
            ]),
            "self_collision_min_sampled_m": np.asarray([collision_minimum.distance]),
            "self_collision_min_links": np.asarray(collision_minimum.links),
        }
    else:
        collision_summary = {}
    return {
        "qpos": np.asarray(qs, dtype=np.float32),
        **collision_summary,
        # Render-only lead-in; kept out of qpos so every per-frame array stays
        # index-aligned with the solved episode.
        "approach_qpos": approach_qs,
        "raw_left_pose7_ground_truth": np.asarray(raw_left_gt, dtype=np.float32),
        "raw_right_pose7_ground_truth": np.asarray(raw_right_gt, dtype=np.float32),
        "raw_left_controller_pose7": raw_left_controller_arr[frame_indices],
        "raw_right_controller_pose7": raw_right_controller_arr[frame_indices],
        "calibrated_left_tcp_pose7": left_tcp_arr[frame_indices],
        "calibrated_right_tcp_pose7": right_tcp_arr[frame_indices],
        "target_left_pose7_robot_world": target_left,
        "target_right_pose7_robot_world": target_right,
        "achieved_left_pose7_robot_world": achieved_left,
        "achieved_right_pose7_robot_world": achieved_right,
        "left_pos_error_m": errors["left_pos_error_m"],
        "right_pos_error_m": errors["right_pos_error_m"],
        "left_rot_error_deg": errors["left_rot_error_deg"],
        "right_rot_error_deg": errors["right_rot_error_deg"],
        "optimization_score": np.asarray([score], dtype=np.float32),
        "ik_weights": np.asarray(
            [cfg.pos_weight, cfg.ori_weight, cfg.rest_weight], dtype=np.float32
        ),
        "home_left_pose7_robot_world": home_left_pose7[None],
        "home_right_pose7_robot_world": home_right_pose7[None],
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fps": np.asarray([fps], dtype=np.float32),
        "controller_tcp_calibration": np.asarray([str(calibration_source or "")]),
        "controller_tcp_source_robot": np.asarray(
            [tcp_selection.source_robot if tcp_selection is not None else ""]
        ),
        "controller_tcp_source_gripper": np.asarray(
            [tcp_selection.source_gripper if tcp_selection is not None else ""]
        ),
        "controller_tcp_controller_mount": np.asarray(
            [tcp_selection.controller_mount if tcp_selection is not None else ""]
        ),
        "controller_tcp_trusted_dataset_snapshot": np.asarray(
            [
                tcp_selection.trusted_dataset_snapshot
                if tcp_selection is not None
                else False
            ],
            dtype=np.bool_,
        ),
        "controller_tcp_offset_position_norm_m": tcp_diagnostics.get(
            "offset_position_norm_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_workspace_min_z_m": tcp_diagnostics.get(
            "workspace_min_z_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_same_frame_min_separation_m": tcp_diagnostics.get(
            "same_frame_min_separation_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_same_frame_max_separation_m": tcp_diagnostics.get(
            "same_frame_max_separation_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_workspace_position_min_m": tcp_diagnostics.get(
            "workspace_position_min_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_workspace_position_max_m": tcp_diagnostics.get(
            "workspace_position_max_m",
            np.asarray([], dtype=np.float32),
        ),
        "controller_device": np.asarray([controller_device]),
        "tracking_workspace": np.asarray(
            [str(source_metadata.get("tracking_workspace", ""))]
        ),
        "robot_from_table_pose7": np.asarray(
            [robot_from_table] if robot_from_table is not None else [],
            dtype=np.float32,
        ),
        "table_from_robot_pose7": np.asarray(
            [pose_inv(robot_from_table)] if robot_from_table is not None else [], dtype=np.float32,
        ),
        "base_placement_policy": np.asarray(["dataset-xy-yaw" if getattr(args, "base_placement", "auto") == "dataset-xy-yaw" else "fixed"]),
        "base_placement_scope": np.asarray([deployment_calibration.scope if deployment_calibration is not None else ""]),
        "base_placement_verified": np.asarray([deployment_calibration.verified if deployment_calibration is not None else False]),
        "base_placement_diagnostics_json": np.asarray([""]),
        **_deployment_npz_entries(deployment_calibration),
        "robot_tcp_calibration_json": np.asarray([json.dumps(robot_tcp.metadata(), sort_keys=True) if robot_tcp is not None else ""]),
        "max_ik_position_error_m": np.asarray([args.max_ik_position_error_m]),
        "max_ik_rotation_error_deg": np.asarray([args.max_ik_rotation_error_deg]),
        "initial_position_tolerance_m": np.asarray([args.initial_position_tolerance_m]),
        "left_tool_adapter_pose7": np.asarray(
            [left_tool_adapter] if left_tool_adapter is not None else [],
            dtype=np.float32,
        ),
        "right_tool_adapter_pose7": np.asarray(
            [right_tool_adapter] if right_tool_adapter is not None else [],
            dtype=np.float32,
        ),
        "absolute_orientation": np.asarray([args.absolute_orientation]),
        "initial_solve_iterations": np.asarray(
            [initial_solve_count], dtype=np.int64
        ),
        "initial_max_position_error_m": np.asarray(
            [initial_max_position_error_m], dtype=np.float32
        ),
        "initial_max_rotation_error_deg": np.asarray(
            [initial_max_rotation_error_deg], dtype=np.float32
        ),
        "gripper_normalized": (
            gripper_openings[frame_indices]
            if gripper_openings is not None
            else np.empty((len(frame_indices), 0), dtype=np.float32)
        ),
        "gripper_source": np.asarray([gripper_source]),
        "raw_controller_debug": np.asarray([args.raw_controller_debug], dtype=np.bool_),
        "retarget_mode": np.asarray([retarget_mode]),
        "retarget_mode_requested": np.asarray([requested_retarget_mode]),
        "compose_source": np.asarray([args.compose_source]),
        "translation_scale": np.asarray([args.translation_scale], dtype=np.float32),
    }


def save_rollout(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> Path:
    output = args.output
    if output is None:
        output = DEFAULT_OUT_DIR / f"episode_{args.episode:06d}_{args.robot}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(
        repo_id=np.asarray([args.repo_id]),
        robot=np.asarray([args.robot]),
        episode=np.asarray([args.episode], dtype=np.int64),
        **rollout,
    )
    np.savez_compressed(output, **payload)
    print(f"[replay] saved: {output}")
    return output


def _render_task_scene(server, args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    if args.scene is None:
        return
    transforms = rollout["robot_from_table_pose7"]
    if len(transforms) != 1:
        raise SystemExit("--scene requires an absolute table or TCP retarget mode.")

    from handumi.sim.scene import load_scene

    robot_from_table = transforms[0]
    for body in load_scene(args.scene, position=(0.0, 0.0, 0.0)):
        body_table = np.concatenate(
            [
                body.rest_position,
                body.rest_quaternion_wxyz[[1, 2, 3, 0]],
            ]
        ).astype(np.float32)
        body_robot = pose_mul(robot_from_table, body_table)
        frame = server.scene.add_frame(
            f"/scene/{body.name}",
            position=tuple(body_robot[:3]),
            wxyz=tuple(body_robot[[6, 3, 4, 5]]),
            show_axes=False,
        )
        del frame
        for index, geom in enumerate(body.geoms):
            if geom.kind != "box":
                continue
            color = tuple(int(round(channel * 255)) for channel in geom.rgba[:3])
            server.scene.add_box(
                f"/scene/{body.name}/geom_{index}",
                dimensions=tuple(2.0 * value for value in geom.size),
                position=tuple(geom.local_position),
                wxyz=tuple(geom.local_quaternion_wxyz),
                color=color,
            )


#: Square cross-section of each T-slot rail (robot X width; height comes
#: from the table offset).
MOUNT_RAIL_PROFILE_M = 0.03
#: Rail overhang past the outer side of each base (robot Y).
MOUNT_RAIL_MARGIN_Y_M = 0.08
#: Footprint half-size assumed when a base link has no loaded mesh.
FALLBACK_BASE_HALF_SIZE_M = 0.05


def _base_footprints(runtime, urdf) -> list[np.ndarray]:
    """Return ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]`` per arm base.

    Uses the base link's visual meshes (world-frame bounds) and falls back to a
    square around the link origin when the URDF was loaded without meshes.
    """
    import trimesh

    graph = urdf.scene.graph if urdf.scene is not None else None
    footprints = []
    for side in runtime.active_sides:
        names = runtime.arm_joint_names(side)
        if not names:
            continue
        link = urdf.joint_map[names[0]].parent
        bounds = []
        if graph is not None:
            for node in graph.nodes_geometry:
                if graph.transforms.parents.get(node) != link:
                    continue
                transform, geometry = graph.get(node)
                bounds.append(trimesh.transform_points(
                    urdf.scene.geometry[geometry].bounds, transform
                ))
        if bounds:
            points = np.concatenate(bounds, axis=0)
            footprints.append(np.stack([points.min(axis=0), points.max(axis=0)]))
        else:
            origin = urdf.get_transform(link)[:3, 3]
            half = np.array([FALLBACK_BASE_HALF_SIZE_M] * 2 + [0.0])
            footprints.append(np.stack([origin - half, origin + half]))
    return footprints


def _render_table(
    server, runtime, urdf, rollout: dict[str, np.ndarray], *, clean: bool = False,
    base_pose7: np.ndarray | None = None,
) -> list:
    """Draw the grid on the calibrated tabletop and the rail under the bases.

    The grid lies in the table plane rather than at robot-world z=0, so a
    table placed below the base plane (the Piper's 3 cm T-slot rail) is
    visible. When the table is below the bases, two rails fill the gap.
    ``clean`` keeps only the rails, leaving out grid, axes and labels.
    """
    handles = []
    base_pose7 = np.asarray(base_pose7 if base_pose7 is not None else [0, 0, 0, 0, 0, 0, 1])
    table_transforms = rollout.get("robot_from_table_pose7")
    if table_transforms is None or len(table_transforms) != 1:
        if not clean:
            handles.append(server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1))
        return handles
    table_pose7 = np.asarray(table_transforms[0], dtype=np.float64)
    if not clean:
        handles.append(server.scene.add_grid(
            "/grid",
            width=3.0,
            height=3.0,
            cell_size=0.1,
            position=tuple(table_pose7[:3]),
            wxyz=tuple(table_pose7[[6, 3, 4, 5]]),
        ))
        handles.append(server.scene.add_frame(
            "/table_origin",
            position=tuple(table_pose7[:3]),
            wxyz=tuple(table_pose7[[6, 3, 4, 5]]),
            axes_length=0.15,
            axes_radius=0.004,
        ))
        handles.extend(_render_placement_measurements(server, runtime, urdf, table_pose7, base_pose7))
    table_z = float(pose_mul(pose_inv(base_pose7), table_pose7)[2])
    footprints = _base_footprints(runtime, urdf)
    if not footprints or table_z > -1e-3:
        return handles
    lower = np.min([f[0] for f in footprints], axis=0)
    upper = np.max([f[1] for f in footprints], axis=0)
    # Two T-slot rails run under the front and back edges of the bases,
    # each centered on its edge, leaving the middle open so the reference
    # point on the tabletop stays visible.
    y_min = float(lower[1]) - MOUNT_RAIL_MARGIN_Y_M
    y_max = float(upper[1]) + MOUNT_RAIL_MARGIN_Y_M
    base_z = float(lower[2])
    for name, edge_x in (("front", float(upper[0])), ("back", float(lower[0]))):
        local_pose = np.array([edge_x, (y_min + y_max) / 2.0,
                               (base_z + table_z) / 2.0, 0, 0, 0, 1])
        world_pose = pose_mul(base_pose7, local_pose)
        handles.append(server.scene.add_box(
            f"/mount_rail_{name}",
            dimensions=(MOUNT_RAIL_PROFILE_M, y_max - y_min, base_z - table_z),
            position=tuple(world_pose[:3]),
            wxyz=tuple(world_pose[[6, 3, 4, 5]]),
            color=(150, 155, 160),
        ))
    return handles


def _render_placement_measurements(server, runtime, urdf, table_pose7: np.ndarray,
                                   base_pose7: np.ndarray | None = None) -> list:
    """在侧栏显示摆放尺寸，场景仅保留标记与可选的简短文字。"""
    magenta = (220, 40, 200)
    table_center = np.asarray(table_pose7[:3], dtype=np.float64)
    base_pose7 = np.asarray(base_pose7 if base_pose7 is not None else [0, 0, 0, 0, 0, 0, 1])
    # 基座参考点位于桌面高度，便于测量桌面原点的水平偏移。
    normal = pose7_to_mat(table_pose7)[:3, 2]
    reference = base_pose7[:3] - normal * np.dot(base_pose7[:3] - table_center, normal)
    offset = table_center - reference
    coincident = float(np.linalg.norm(offset)) < 1e-6
    labels = []
    segments = [] if coincident else [[reference, table_center]]
    handles = [server.scene.add_frame("/measure", show_axes=False)]
    server.scene.add_icosphere(
        "/measure/reference", radius=0.012, color=magenta, position=tuple(reference)
    )
    # 重合时只创建一条标签；默认隐藏，避免标签随观察角度相互遮挡。
    label_specs = (
        [("origin", "桌面原点 / 基座参考点", table_center)]
        if coincident else [
            ("reference", "基座参考点", reference),
            ("charuco", "桌面原点", table_center),
        ]
    )
    for name, text, position in label_specs:
        labels.append(server.scene.add_label(
            f"/measure/{name}_label",
            text,
            position=tuple(position + [0.0, 0.0, 0.06]),
            anchor="bottom-center",
            font_screen_scale=0.8,
            visible=False,
        ))
    table_in_base = pose_mul(pose_inv(base_pose7), table_pose7)
    base_in_table = pose_mul(pose_inv(table_pose7), base_pose7)
    details = [
        "紫色圆点：基座在桌面的参考点。坐标轴：桌面原点。",
        "桌面原点与基座参考点重合。" if coincident else "桌面原点与基座参考点分开显示。",
        f"**水平距离：** {np.linalg.norm(offset[:2]):.3f} m",
        f"**水平偏移：** X {offset[0]:+.3f} m，Y {offset[1]:+.3f} m",
        f"**桌面原点（base 坐标）：**\n"
        f"X {table_in_base[0]:+.3f} m，Y {table_in_base[1]:+.3f} m，Z {table_in_base[2]:+.3f} m",
        f"**基座原点（桌面坐标）：**\nX {base_in_table[0]:+.3f} m，Y {base_in_table[1]:+.3f} m，Z {base_in_table[2]:+.3f} m",
    ]
    bases = [
        pose_mul(base_pose7, np.r_[urdf.get_transform(urdf.joint_map[names[0]].parent)[:3, 3], [0, 0, 0, 1]])[:3]
        for side in runtime.active_sides
        if (names := runtime.arm_joint_names(side))
    ]
    if len(bases) == 2:
        left, right = (np.asarray(base, dtype=np.float64) for base in bases)
        segments.append([left, right])
        for side, center in (("left", left), ("right", right)):
            server.scene.add_icosphere(
                f"/measure/base_center_{side}",
                radius=0.01,
                color=magenta,
                position=tuple(center),
            )
        details.append(
            f"**双臂基座间距：** {np.linalg.norm(left - right):.3f} m"
        )
    if segments:
        server.scene.add_line_segments(
            "/measure/lines",
            points=np.asarray(segments, dtype=np.float32),
            colors=magenta,
            line_width=3.0,
        )
    folder = server.gui.add_folder("坐标与距离", order=100)
    handles.append(folder)
    with folder:
        server.gui.add_markdown("\n\n".join(details))
        show_labels = server.gui.add_checkbox("显示场景文字", False)

    @show_labels.on_update
    def _toggle_labels(_) -> None:
        """按需显示简短场景文字，完整尺寸始终保留在侧栏。"""
        with server.atomic():
            for label in labels:
                label.visible = show_labels.value
    return handles


def _prepare_all_episodes(args: argparse.Namespace, *, initial_failure: str | None = None) -> dict[int, str]:
    """在启动窗口前有界并行准备其他 episode，并报告失败。"""
    from handumi.scripts.replay.replay_scheduler import EpisodeScheduler

    if args.no_cache:
        raise SystemExit("--prepare-mode all requires caching; remove --no-cache.")
    info = ensure_metadata(DatasetRef.from_repo_id(args.repo_id, root=args.dataset_root, revision=args.revision))
    count = int(info.get("total_episodes", 0))
    scheduler = EpisodeScheduler(args, count, current=args.episode,
                                 workers=args.prepare_workers, background=True)
    if initial_failure is not None:
        scheduler.failures[args.episode] = initial_failure
    previous = -1
    try:
        while scheduler.queue or scheduler.running:
            scheduler.poll()
            done = len(scheduler.completed)
            if done != previous:
                print(f"[replay] preparing {done}/{count} episodes; failed={len(scheduler.failures)}")
                previous = done
            time.sleep(0.05)
        if scheduler.failures:
            for episode, message in sorted(scheduler.failures.items()):
                print(f"[replay] episode {episode} failed: {message}")
        failures = dict(scheduler.failures)
        if getattr(args, "base_placement", None) == "dataset-xy-yaw":
            args._dataset_evaluation_results.update(getattr(scheduler, "placement_metrics", {}))
            args._dataset_stage_seconds = dict(getattr(scheduler, "stage_seconds", {}))
    finally:
        scheduler.close()
    # 本轮已经刷新完成，后续切换按内容键读取刚写入的结果。
    args.refresh_cache = False
    return failures


def _prepare_all_headless(args: argparse.Namespace) -> tuple[dict | None, dict[int, str]]:
    """从首段开始并行准备全部段，仅保留选中段以供导出。"""
    from handumi.scripts.replay.replay_progress import BatchProgress
    from handumi.scripts.replay.replay_scheduler import EpisodeScheduler

    info = ensure_metadata(DatasetRef.from_repo_id(
        args.repo_id, root=args.dataset_root, revision=args.revision
    ))
    count = int(info.get("total_episodes", 0))
    if count <= 0:
        raise SystemExit("Dataset contains no episodes.")
    if not 0 <= args.episode < count:
        raise SystemExit(f"Episode {args.episode} is out of range 0..{count - 1}.")
    started = time.monotonic()
    scheduler = EpisodeScheduler(
        args, count, current=args.episode, workers=args.prepare_workers,
        background=True, batch=True, progress=args.progress != "off",
    )
    try:
        with BatchProgress(count, args.progress) as display:
            scheduler.warning_sink = lambda message: display.console.print(
                message, markup=False, highlight=False
            )
            while scheduler.queue or scheduler.running:
                scheduler.poll()
                display.update(scheduler)
                if scheduler.queue or scheduler.running:
                    time.sleep(0.05)
            display.update(scheduler, force=True)
        for episode, message in sorted(scheduler.failures.items()):
            print(f"[replay] episode {episode} failed: {message}")
        if scheduler.stage_seconds:
            totals = ", ".join(
                f"{stage}={seconds:.2f}s" for stage, seconds in sorted(scheduler.stage_seconds.items())
            )
            print(f"[replay] worker stage time (sum across workers): {totals}")
        print(f"[replay] total preparation wall time: {time.monotonic() - started:.2f}s")
        if getattr(args, "base_placement", None) == "dataset-xy-yaw":
            args._dataset_evaluation_results = dict(getattr(scheduler, "placement_metrics", {}))
            args._dataset_stage_seconds = dict(getattr(scheduler, "stage_seconds", {}))
            args._dataset_wall_seconds = time.monotonic() - started
        rollout = None if args.episode in scheduler.failures else scheduler.take(args.episode)
        return rollout, dict(scheduler.failures)
    finally:
        scheduler.close()


def show_viewer(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    """显示回放窗口，允许选择 episode 并在加载失败时保留当前回放。"""
    import viser
    from viser.extras import ViserUrdf

    from handumi.scripts.replay.replay_scheduler import EpisodeScheduler
    from handumi.visualize.gripper_curve import add_gripper_curve, gripper_curve_image

    runtime = load_embodiment(args.robot)
    source_info = ensure_metadata(
        DatasetRef.from_repo_id(
            args.repo_id, root=args.dataset_root, revision=args.revision,
        )
    )
    active_episode = int(args.episode)
    episode_count = int(source_info.get("total_episodes", active_episode + 1))
    server = viser.ViserServer(port=args.port)
    episode_picker = server.gui.add_dropdown(
        "Episode",
        options=tuple(str(i) for i in range(episode_count)),
        initial_value=str(active_episode),
        disabled=episode_count < 2,
        hint="编号从 0 开始；切换时沿用当前机器人、标定与 IK 设置。",
        order=-10,
    )
    status = server.gui.add_markdown(
        f"Episode **{active_episode}** · **{len(rollout['qpos'])}** 帧", order=-9,
    )
    curve = None
    if "gripper_normalized" in rollout:
        curve = add_gripper_curve(
            server, rollout["gripper_normalized"], float(rollout["fps"][0]),
        )
    urdf = runtime.load_urdf(load_meshes=True)
    def display_base(data):
        """绝对 TCP 轨迹以桌面为世界坐标，其余回放保持原显示坐标。"""
        poses = data.get("table_from_robot_pose7")
        modes = data.get("retarget_mode")
        if poses is not None and len(poses) == 1 and modes is not None and str(modes[0]) == "absolute-tcp":
            return np.asarray(poses[0], dtype=np.float32)
        return None

    def display_table(data, base):
        """构造桌面世界下的显示参数，不修改轨迹和标定记录。"""
        if base is None:
            return data
        return {**data, "robot_from_table_pose7": np.asarray([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)}

    base_pose = display_base(rollout)
    robot_root = server.scene.add_frame(
        "/robot", show_axes=False,
        position=tuple(base_pose[:3]) if base_pose is not None else (0, 0, 0),
        wxyz=tuple(base_pose[[6, 3, 4, 5]]) if base_pose is not None else (1, 0, 0, 0),
    )
    robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
    _render_task_scene(server, args, display_table(rollout, base_pose))
    table_handles = _render_table(server, runtime, urdf, display_table(rollout, base_pose),
                                  clean=args.clean_view, base_pose7=base_pose) or []
    markers = {}
    trajectories = {}

    def trajectory_points(
        data: dict[str, np.ndarray], kind: str, side: str,
    ) -> np.ndarray:
        """提取轨迹点；单帧数据使用重合端点，避免空曲线。"""
        poses = np.asarray(data[f"{kind}_{side}_pose7_robot_world"])
        base = display_base(data)
        points = np.asarray([pose_mul(base, pose)[:3] for pose in poses]) if base is not None else poses[:, :3]
        return np.repeat(points, 2, axis=0) if len(points) == 1 else points

    if not args.hide_trajectories:
        colors = {"left": ((255, 190, 50), (80, 160, 255)),
                  "right": ((80, 220, 130), (255, 90, 90))}
        for side in runtime.active_sides:
            for kind, color in zip(("target", "achieved"), colors[side], strict=True):
                trajectories[kind, side] = server.scene.add_spline_catmull_rom(
                    f"/traj/{kind}_{side}",
                    points=trajectory_points(rollout, kind, side),
                    color=color, line_width=2.0,
                )
                markers[kind, side] = server.scene.add_icosphere(
                    f"/{kind}/{side}", radius=0.018 if kind == "target" else 0.014,
                    color=color,
                )
    approach = rollout.get("approach_qpos")
    approach_frames = 0 if approach is None else len(approach)
    total_frames = approach_frames + len(rollout["qpos"])
    play = server.gui.add_checkbox("Play", True)
    frame = server.gui.add_slider("Frame", 0, total_frames - 1, 1, 0)
    err_text = server.gui.add_text("EE error (cm/deg)", "-", disabled=True)
    opening_text = server.gui.add_text("Gripper opening (Link7 flange replay)", "-", disabled=True)
    # 回调只投递请求，求解和绘制由主循环串行处理，避免切换与播放竞争。
    requests: SimpleQueue[int] = SimpleQueue()
    scheduler = EpisodeScheduler(
        args, episode_count, current=active_episode,
        workers=getattr(args, "prepare_workers", 8),
        background=getattr(args, "prepare_mode", "background") == "background",
    )
    pending_episode = None

    @episode_picker.on_update
    def request_episode(event) -> None:
        """记录用户选择；程序回退选择器时不触发重复加载。"""
        requested = int(event.target.value)
        if requested != active_episode or pending_episode is not None:
            requests.put(requested)

    def draw(index: int) -> None:
        """刷新当前轨迹的机器人、标记、误差与夹爪开度。"""
        # Frames before the episode replay the home -> start lead-in; markers
        # stay pinned on frame 0 so the approach visibly aims at it.
        if index < approach_frames:
            assert approach is not None
            robot_view.update_cfg(approach[index])
            i = 0
        else:
            i = index - approach_frames
            robot_view.update_cfg(rollout["qpos"][i])
        opening_text.value = " ".join(f"{side}={rollout['gripper_normalized'][i, column]:.1%}" for column, side in enumerate(("left", "right")) if side in runtime.active_sides)
        for (kind, side), marker in markers.items():
            pose = rollout[f"{kind}_{side}_pose7_robot_world"][i]
            base = display_base(rollout)
            marker.position = tuple(pose_mul(base, pose)[:3] if base is not None else pose[:3])
        if index < approach_frames:
            err_text.value = "home -> start approach"
        else:
            err_text.value = " ".join(
                f"{side[0].upper()}={rollout[f'{side}_pos_error_m'][i] * 100:.1f}cm/"
                f"{rollout[f'{side}_rot_error_deg'][i]:.1f}deg"
                for side in runtime.active_sides
            )

    draw(0)
    print(f"[replay] viewer: http://localhost:{server.get_port()}")
    current = 0
    failure_status = None
    try:
        while True:
            requested = None
            # 快速连续操作时，只加载最新选择。
            while True:
                try:
                    requested = requests.get_nowait()
                except Empty:
                    break
            if requested is not None:
                failure_status = None
                if requested == active_episode:
                    pending_episode = None
                else:
                    pending_episode = requested
                    scheduler.request(requested)
            scheduler.poll()
            next_rollout = None
            if pending_episode is not None:
                try:
                    next_rollout = scheduler.take(pending_episode)
                except RuntimeError as exc:
                    failed_episode = pending_episode
                    pending_episode = None
                    print(f"[replay] cannot load episode {failed_episode}: {exc}")
                    with server.atomic():
                        episode_picker.value = str(active_episode)
                    failure_status = (
                        f"Episode **{failed_episode}** 加载失败，继续显示 **{active_episode}**。\n\n"
                        + str(exc)[:500]
                    )
                if (pending_episode is not None and next_rollout is None
                        and pending_episode not in scheduler.running.values()
                        and pending_episode not in scheduler.queue):
                    scheduler.request(pending_episode)
            if next_rollout is not None:
                requested = pending_episode
                try:
                    # 先完成求解和图表准备，失败时不改动当前可视化。
                    next_approach = next_rollout.get("approach_qpos")
                    next_approach_frames = (
                        0 if next_approach is None else len(next_approach)
                    )
                    next_total_frames = next_approach_frames + len(next_rollout["qpos"])
                    next_points = {
                        key: trajectory_points(next_rollout, *key) for key in trajectories
                    }
                    next_base_pose = display_base(next_rollout)
                    next_display = display_table(next_rollout, next_base_pose)
                    next_curve = (
                        gripper_curve_image(
                            next_rollout["gripper_normalized"],
                            float(next_rollout["fps"][0]),
                        )
                        if "gripper_normalized" in next_rollout else None
                    )
                except (Exception, SystemExit) as exc:
                    print(f"[replay] cannot load episode {requested}: {exc}")
                    pending_episode = None
                    with server.atomic():
                        episode_picker.value = str(active_episode)
                        status.content = (
                            f"Episode **{requested}** 加载失败，继续显示 **{active_episode}**。\n\n"
                            + str(exc)[:500]
                        )
                    failure_status = status.content
                else:
                    with server.atomic():
                        rollout = next_rollout
                        approach, approach_frames = next_approach, next_approach_frames
                        total_frames = next_total_frames
                        active_episode = requested
                        pending_episode = None
                        if next_base_pose is not None:
                            robot_root.position = tuple(next_base_pose[:3])
                            robot_root.wxyz = tuple(next_base_pose[[6, 3, 4, 5]])
                            for handle in table_handles:
                                handle.remove()
                            table_handles = _render_table(
                                server, runtime, urdf, next_display, clean=args.clean_view,
                                base_pose7=next_base_pose,
                            ) or []
                        frame.value = current = 0
                        frame.max = total_frames - 1
                        for key, trajectory in trajectories.items():
                            trajectory.points = next_points[key]
                        if next_curve is not None:
                            if curve is None:
                                curve = server.gui.add_image(
                                    next_curve, label="Gripper opening", order=0,
                                )
                            else:
                                curve.image = next_curve
                            curve.visible = True
                        elif curve is not None:
                            curve.visible = False
                        episode_picker.value = str(active_episode)
                        failure_status = None
                        draw(0)
                pending_episode = None
            progress_label = "按需处理" if getattr(args, "prepare_mode", "background") == "off" else "后台处理"
            progress = f"{progress_label} **{len(scheduler.completed)}/{episode_count}** · 失败 **{len(scheduler.failures)}**"
            if failure_status is not None:
                status.content = failure_status + "\n\n" + progress
            elif pending_episode is not None:
                status.content = f"正在准备 Episode **{pending_episode}**，继续播放 **{active_episode}**。\n\n" + progress
            else:
                status.content = f"Episode **{active_episode}** · **{len(rollout['qpos'])}** 帧\n\n" + progress
            if play.value:
                current = (current + 1) % total_frames
                frame.value = current
            else:
                current = min(max(int(frame.value), 0), total_frames - 1)
                frame.value = current
            draw(current)
            time.sleep(1.0 / 30.0)
    finally:
        scheduler.close()
        server.stop()


def main() -> None:
    raw_argv = list(sys.argv[1:])
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = build_parser(show_advanced=show_advanced)
    if show_advanced:
        parser.print_help()
        return
    args = parser.parse_args(raw_argv)
    try:
        selection = resolve_dataset_selection(
            args.dataset,
            revision=args.revision,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.repo_id = selection.repo_id
    args.dataset_root = selection.root
    from handumi.scripts.replay.replay_deployment import select_saved_dataset_placement

    select_saved_dataset_placement(args)
    resolve_replay_options(args)
    print(
        "Replay plan\n"
        f"  Dataset: {selection.root}\n"
        f"  Repository: {selection.repo_id}\n"
        f"  Episode: {args.episode}\n"
        f"  Robot profile: {args.robot}\n"
        f"  Retargeting: {args.retarget_mode}\n"
        f"  Base placement: {args.base_placement}\n"
        f"  Placement file: {getattr(getattr(args, 'deployment_selection', None), 'path', None) or args.deployment_calibration or 'automatic'}\n"
        f"  Deployment profile: {args.deployment_profile}"
    )
    if args.dry_run:
        return
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.prepare_workers < 1:
        raise ValueError("--prepare-workers must be >= 1.")
    if args.prepare_mode == "all" and args.no_cache:
        raise ValueError("--prepare-mode all requires caching; remove --no-cache.")
    if args.initial_solve_iterations < 1:
        raise ValueError("--initial-solve-iterations must be >= 1.")
    if args.initial_position_tolerance_m <= 0.0:
        raise ValueError("--initial-position-tolerance-m must be > 0.")
    if args.approach_seconds < 0.0:
        raise ValueError("--approach-seconds must be >= 0.")
    if args.max_ik_position_error_m <= 0.0:
        raise ValueError("--max-ik-position-error-m must be > 0.")
    if args.max_ik_rotation_error_deg <= 0.0:
        raise ValueError("--max-ik-rotation-error-deg must be > 0.")
    if args.table_clearance_warning_m <= 0.0:
        raise ValueError("--table-clearance-warning-m must be > 0.")
    from handumi.scripts.replay.replay_prepare import prepare_episode

    dataset_placement = args.base_placement == "dataset-xy-yaw"
    if dataset_placement:
        from handumi.scripts.replay.replay_dataset_placement import (
            PrefixObserver,
            resolve_dataset_placement,
            write_evaluation_report,
        )

        args.strict_ik = True
        args.deployment_selection = resolve_dataset_placement(args)
        args._dataset_evaluation_results = {}

    if args.headless and args.prepare_mode == "all":
        rollout, failures = _prepare_all_headless(args)
        if dataset_placement:
            write_evaluation_report(args, failures)
        if failures:
            raise SystemExit(f"Preparation failed for {len(failures)} episode(s).")
        if rollout is None:
            raise SystemExit(f"Prepared episode {args.episode} did not return a rollout.")
        save_rollout(args, rollout)
        return

    initial_failure = None
    if dataset_placement:
        args._placement_score = PrefixObserver()
    try:
        rollout = prepare_episode(args)
    except (Exception, SystemExit) as exc:
        if dataset_placement:
            observer = args._placement_score
            args._dataset_evaluation_results[args.episode] = {
                "prefix_frames": observer.prefix, "total_frames": observer.total_frames,
                "solve_identity": observer.identity_key,
                "first_failed_frame": (observer.failed_frame if observer.failed_frame is not None else observer.current_frame),
            }
        if args.prepare_mode != "all":
            if dataset_placement:
                write_evaluation_report(args, {args.episode: str(exc)}, completed=[args.episode])
            raise
        initial_failure = str(exc)
        print(f"[replay] episode {args.episode} failed: {initial_failure}")
        rollout = None
    if dataset_placement and rollout is not None:
        observer = args._placement_score
        args._dataset_evaluation_results[args.episode] = {
            "prefix_frames": observer.prefix, "total_frames": observer.total_frames,
            "solve_identity": observer.identity_key,
            "first_failed_frame": observer.failed_frame,
            "max_position_error_m": float(max(np.max(rollout["left_pos_error_m"]), np.max(rollout["right_pos_error_m"]))),
            "max_rotation_error_deg": float(max(np.max(rollout["left_rot_error_deg"]), np.max(rollout["right_rot_error_deg"]))),
        }
    if args.prepare_mode == "all":
        failures = _prepare_all_episodes(args, initial_failure=initial_failure)
        if dataset_placement:
            write_evaluation_report(args, failures)
        if args.headless and failures:
            raise SystemExit(f"Preparation failed for {len(failures)} episode(s).")
        if rollout is None and not args.headless:
            info = ensure_metadata(DatasetRef.from_repo_id(args.repo_id, root=args.dataset_root, revision=args.revision))
            usable = [index for index in range(int(info.get("total_episodes", 0))) if index not in failures]
            if not usable:
                raise SystemExit("No episode could be prepared.")
            args.episode = usable[0]
            rollout = prepare_episode(args)
    elif dataset_placement:
        write_evaluation_report(args, {}, completed=[args.episode])
    if rollout is not None:
        save_rollout(args, rollout)
    if not args.headless:
        assert rollout is not None
        show_viewer(args, rollout)


if __name__ == "__main__":
    main()
