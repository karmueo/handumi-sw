#!/usr/bin/env python3
"""Replay a HandUMI LeRobot episode in simulation with YAML robot configs."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
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
from handumi.robots.utils import pose_mul

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
    parser.add_argument("--episode", type=int, default=0)
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
        choices=("auto", "local-relative", "anchored", "absolute-table"),
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
        default="relative-start",
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
        default=0.01,
        help=advanced("Required maximum TCP position error before starting replay."),
    )
    parser.add_argument(
        "--approach-seconds",
        type=float,
        default=1.0,
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
        default=0.03,
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
        default=45.0,
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
    if recorded in ("pico", "meta"):
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


def solve_episode(args: argparse.Namespace) -> dict[str, np.ndarray]:
    runtime = load_embodiment(args.robot)
    states, fps, source_info, recorded_gripper_openings = load_episode_states(args)
    source_metadata = handumi_metadata(source_info)
    from handumi.config import dataset_active_sides

    missing = set(runtime.arms) - set(dataset_active_sides(source_metadata))
    if missing:
        raise ValueError(f"Dataset does not record the sides required by {args.robot}: {sorted(missing)}")
    controller_device = _resolved_controller_device(args, source_info)
    requested_retarget_mode = str(args.retarget_mode)
    retarget_mode = _resolved_retarget_mode(args, source_info)
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
                "--retarget-mode absolute-table requires a dataset recorded in the "
                "calibrated table workspace."
            )
        deployment_calibration = getattr(args, "deployment_selection", None)
        if deployment_calibration is None:
            deployment_calibration = resolve_deployment_calibration(
                args.robot,
                explicit_path=args.deployment_calibration,
                profile=args.deployment_profile,
                rig_config=args.rig_config,
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
        initial_solver = runtime.solver_cls(
            config=replace(cfg, max_joint_delta=None),
            locked_joint_indices=locked_joint_indices,
        )
        # Seed the start-pose solve from the working posture rather than the
        # folded home: the local solver lands in whichever elbow branch it
        # starts near, and every later frame warm-starts from this one.
        q = posture_seed(runtime.config.posture_q, runtime.config.home_q)
        for initial_solve_count in range(1, args.initial_solve_iterations + 1):
            q = initial_solver.ik(
                q,
                left_pose=(first_left_target[:3], first_left_target[3:7]),
                right_pose=(first_right_target[:3], first_right_target[3:7]),
            )
            initial_left, initial_right = initial_solver.fk_pose7(q)
            initial_errors = pose_error_arrays(
                first_left_target[None],
                first_right_target[None],
                initial_left[None],
                initial_right[None],
            )
            initial_max_position_error_m = max(
                float(initial_errors["left_pos_error_m"][0]),
                float(initial_errors["right_pos_error_m"][0]),
            )
            initial_max_rotation_error_deg = max(
                float(initial_errors["left_rot_error_deg"][0]),
                float(initial_errors["right_rot_error_deg"][0]),
            )
            if initial_max_position_error_m <= args.initial_position_tolerance_m:
                break
        if initial_max_position_error_m > args.initial_position_tolerance_m:
            raise SystemExit(
                "Unable to prepare replay start pose: maximum TCP position error "
                f"is {initial_max_position_error_m * 100:.2f} cm after "
                f"{initial_solve_count} IK iterations. Check robot_from_table, TCP "
                "calibration, or robot reachability. Pass a dataset-specific "
                "--deployment-calibration or raise --initial-position-tolerance-m "
                "to visualize; use --retarget-mode local-relative when morphology "
                "adaptation is acceptable."
            )

    start = time.perf_counter()
    for selected_index, frame_index in enumerate(frame_indices):
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
        elif retarget_mode == "absolute-table":
            if robot_from_table is None:
                raise RuntimeError("Absolute table retarget mode was not initialized.")
            left_pose7, right_pose7 = absolute_table_robot_target_pose7(
                state,
                robot_from_table,
                left_tool_adapter_pose7=left_tool_adapter,
                right_tool_adapter_pose7=right_tool_adapter,
            )
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
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
        if gripper_openings is not None:
            opening = gripper_openings[frame_index]
            runtime.set_finger_positions(
                q,
                {"left": float(opening[0]), "right": float(opening[1])},
            )
        qs.append(q.copy())
        raw_left_gt.append(raw_left)
        raw_right_gt.append(raw_right)
        left_targets.append(left_pose7)
        right_targets.append(right_pose7)
        left_achieved.append(fk_left_pose7)
        right_achieved.append(fk_right_pose7)
    elapsed = time.perf_counter() - start
    approach_qs = joint_approach_ramp(
        runtime.config.home_q.astype(np.float32),
        qs[0],
        frames=int(round(fps * args.approach_seconds)),
    )
    target_left = np.asarray(left_targets, dtype=np.float32)
    target_right = np.asarray(right_targets, dtype=np.float32)
    achieved_left = np.asarray(left_achieved, dtype=np.float32)
    achieved_right = np.asarray(right_achieved, dtype=np.float32)
    errors = pose_error_arrays(target_left, target_right, achieved_left, achieved_right)
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
    if retarget_mode == "absolute-table":
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
        if args.strict_ik:
            raise SystemExit(message)
        print(f"[replay] warning: {message}")
    return {
        "qpos": np.asarray(qs, dtype=np.float32),
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
        **_deployment_npz_entries(deployment_calibration),
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
        raise SystemExit("--scene currently requires --retarget-mode absolute-table.")

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
    server, runtime, urdf, rollout: dict[str, np.ndarray], *, clean: bool = False
) -> None:
    """Draw the grid on the calibrated tabletop and the rail under the bases.

    The grid lies in the table plane rather than at robot-world z=0, so a
    table placed below the base plane (the Piper's 3 cm T-slot rail) is
    visible. When the table is below the bases, two rails fill the gap.
    ``clean`` keeps only the rails, leaving out grid, axes and labels.
    """
    table_transforms = rollout.get("robot_from_table_pose7")
    if table_transforms is None or len(table_transforms) != 1:
        if not clean:
            server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
        return
    table_pose7 = np.asarray(table_transforms[0], dtype=np.float64)
    if not clean:
        server.scene.add_grid(
            "/grid",
            width=3.0,
            height=3.0,
            cell_size=0.1,
            position=tuple(table_pose7[:3]),
            wxyz=tuple(table_pose7[[6, 3, 4, 5]]),
        )
        server.scene.add_frame(
            "/table_origin",
            position=tuple(table_pose7[:3]),
            wxyz=tuple(table_pose7[[6, 3, 4, 5]]),
            axes_length=0.15,
            axes_radius=0.004,
        )
        _render_placement_measurements(server, runtime, urdf, table_pose7)
    table_z = float(table_pose7[2])
    footprints = _base_footprints(runtime, urdf)
    if not footprints or table_z > -1e-3:
        return
    lower = np.min([f[0] for f in footprints], axis=0)
    upper = np.max([f[1] for f in footprints], axis=0)
    # Two T-slot rails run under the front and back edges of the bases,
    # each centered on its edge, leaving the middle open so the reference
    # point on the tabletop stays visible.
    y_min = float(lower[1]) - MOUNT_RAIL_MARGIN_Y_M
    y_max = float(upper[1]) + MOUNT_RAIL_MARGIN_Y_M
    base_z = float(lower[2])
    for name, edge_x in (("front", float(upper[0])), ("back", float(lower[0]))):
        server.scene.add_box(
            f"/mount_rail_{name}",
            dimensions=(MOUNT_RAIL_PROFILE_M, y_max - y_min, base_z - table_z),
            position=(edge_x, (y_min + y_max) / 2.0, (base_z + table_z) / 2.0),
            color=(150, 155, 160),
        )


def _render_placement_measurements(server, runtime, urdf, table_pose7: np.ndarray) -> None:
    """Dimension the table placement on the tabletop (table z=0).

    The reference point is the robot-world origin (for bimanual layouts,
    midway between the arm bases) dropped onto the tabletop, so the line to
    the ChArUco center is the horizontal distance measured on the table. The
    base-to-base line joins the centers of the arm bases (their joint-1
    axes) on the base plane.
    """
    magenta = (220, 40, 200)
    table_center = np.asarray(table_pose7[:3], dtype=np.float64)
    reference = np.array([0.0, 0.0, table_center[2]])
    server.scene.add_icosphere(
        "/measure/reference", radius=0.012, color=magenta, position=tuple(reference)
    )
    server.scene.add_label(
        "/measure/reference/label",
        "reference: robot origin on table z=0",
        position=(0.0, 0.0, 0.03),
        anchor="bottom-center",
    )
    server.scene.add_label(
        "/measure/charuco/label",
        "ChArUco center",
        position=tuple(table_center + [0.0, 0.0, 0.03]),
        anchor="bottom-center",
    )
    offset = table_center - reference
    segments = [[reference, table_center]]
    server.scene.add_label(
        "/measure/reference_to_charuco",
        f"{np.linalg.norm(offset[:2]):.3f} m (x={offset[0]:.3f}, y={offset[1]:.3f})",
        position=tuple((reference + table_center) / 2.0 + [0.0, 0.0, 0.02]),
        anchor="bottom-center",
    )
    bases = [
        urdf.get_transform(urdf.joint_map[names[0]].parent)[:3, 3]
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
        server.scene.add_label(
            "/measure/base_to_base",
            f"base center to base center {np.linalg.norm(left - right):.3f} m",
            position=tuple((left + right) / 2.0 + [-0.15, 0.0, 0.02]),
            anchor="bottom-center",
        )
    server.scene.add_line_segments(
        "/measure/lines",
        points=np.asarray(segments, dtype=np.float32),
        colors=magenta,
        line_width=3.0,
    )


def show_viewer(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    import viser
    from viser.extras import ViserUrdf

    runtime = load_embodiment(args.robot)
    server = viser.ViserServer(port=args.port)
    from handumi.visualize.gripper_curve import add_gripper_curve
    if "gripper_normalized" in rollout:
        add_gripper_curve(server, rollout["gripper_normalized"], float(np.asarray(rollout.get("fps", [30])).reshape(-1)[0]))
    urdf = runtime.load_urdf(load_meshes=True)
    robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
    _render_task_scene(server, args, rollout)
    _render_table(server, runtime, urdf, rollout, clean=args.clean_view)
    markers = {}
    if not args.hide_trajectories:
        colors = {"left": ((255, 190, 50), (80, 160, 255)),
                  "right": ((80, 220, 130), (255, 90, 90))}
        for side in runtime.active_sides:
            for kind, color in zip(("target", "achieved"), colors[side], strict=True):
                server.scene.add_spline_catmull_rom(
                    f"/traj/{kind}_{side}",
                    positions=tuple(
                        (float(row[0]), float(row[1]), float(row[2]))
                        for row in rollout[f"{kind}_{side}_pose7_robot_world"]
                    ),
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

    def draw(index: int) -> None:
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
            marker.position = tuple(rollout[f"{kind}_{side}_pose7_robot_world"][i, :3])
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
    while True:
        if play.value:
            current = (current + 1) % total_frames
            frame.value = current
        else:
            current = int(frame.value)
        draw(current)
        time.sleep(1.0 / 30.0)


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
    print(
        "Replay plan\n"
        f"  Dataset: {selection.root}\n"
        f"  Repository: {selection.repo_id}\n"
        f"  Episode: {args.episode}\n"
        f"  Robot profile: {args.robot}\n"
        f"  Retargeting: {args.retarget_mode}\n"
        f"  Deployment profile: {args.deployment_profile}"
    )
    if args.dry_run:
        return
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
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
    rollout = solve_episode(args)
    save_rollout(args, rollout)
    if not args.headless:
        show_viewer(args, rollout)


if __name__ == "__main__":
    main()
