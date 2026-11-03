#!/usr/bin/env python3
"""Convert a raw HandUMI LeRobot dataset to robot-specific joint angles.

The script loads raw HandUMI controller states (``observation.state``), applies
the same optional controller->TCP calibration and local-relative retargeting
used by ``replay_in_sim.py``, then writes a LeRobot v3.0 dataset whose
``observation.state`` and ``action`` columns contain physical robot joint
values.

All video streams are preserved unchanged (the last frame of each episode is
dropped from the parquet data to produce ``action = state[t+1]``, but the
video files themselves are copied as-is).

Usage
-----
::

    # Axol embodiment (output defaults to NONHUMAN-RESEARCH/handumi-dataset-v2-axol)
    handumi convert NONHUMAN-RESEARCH/handumi-dataset-v2 --robot axol

    # Piper embodiment, custom output repo-id, push to hub afterwards
    handumi convert NONHUMAN-RESEARCH/handumi-dataset-v2 \
        --robot piper \
        --output NONHUMAN-RESEARCH/my-piper-dataset \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from handumi.calibration.control_tcp import (
    DEFAULT_DEVICE as DEFAULT_CONTROLLER_DEVICE,
)
from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    apply_controller_tcp_calibration,
    calibration_path_for_device,
    controller_tcp_calibration_from_metadata,
    controller_tcp_calibration_metadata,
    is_identity_bound_controller_tcp_metadata,
    load_controller_tcp_calibration,
)
from handumi.calibration.deployment import (
    deployment_calibration_metadata,
    print_deployment_calibration,
    resolve_deployment_calibration,
)
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.canonical import (
    canonical_joint_layout,
    canonicalize_joint_trajectory,
    raw_state_gripper_widths_m,
)
from handumi.dataset.raw import (
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)
from handumi.dataset.reader import dataset_root_from_repo_id, handumi_metadata
from handumi.dataset.screening import (
    evaluate_screening_gate,
    format_gate_guidance,
    load_cached_solve,
    solver_signature,
)
from handumi.dataset.tabular import row_scalar
from handumi.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
    raw_state_pose7_pair,
    raw_state_robot_target_pose7,
    retarget_anchors_from_raw_state,
)
from handumi.robots.registry import (
    EMBODIMENT_NAMES,
    load_embodiment,
    load_robot_config,
    robot_config_metadata,
)

load_dotenv()


@dataclass(frozen=True)
class ConversionTcpCalibrationSelection:
    calibration: ControllerTcpCalibration
    metadata: dict[str, Any]
    source: str

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        description=(
            "Convert a PICO/HandUMI LeRobot dataset to an embodiment-specific "
            "joint-angle dataset via IK retargeting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Source / output datasets
    # ------------------------------------------------------------------
    ds = parser.add_argument_group("Dataset I/O")
    ds.add_argument(
        "dataset",
        help="Local source dataset path or Hugging Face repo id.",
    )
    ds.add_argument(
        "--output",
        default=None,
        help=(
            "Local output path or Hugging Face repository id. Defaults to a "
            "robot-suffixed dataset under outputs/datasets/."
        ),
    )
    ds.add_argument(
        "--revision",
        default="main",
        help=advanced("Git revision of the source dataset."),
    )
    ds.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help=advanced("Machine-local rig YAML used to select lab deployment calibration."),
    )
    ds.add_argument(
        "--source",
        default="observation.state",
        help=advanced("Raw 16D HandUMI feature column to convert."),
    )
    ds.add_argument(
        "--output-layout",
        default="canonical",
        help=(
            "Joint vector of the written dataset. 'canonical' is HandUMI's "
            "physical layout (radians + meters). A LeRobot robot_type "
            "(e.g. bi_piper_follower, bi_openarm_follower) writes that "
            "plugin's own vector directly, ready for its training and "
            "deployment stack, without keeping an intermediate dataset."
        ),
    )
    ds.add_argument(
        "--use-degrees",
        action="store_true",
        help=(
            "With --output-layout: record arm joints in degrees, as LeRobot's "
            "use_degrees config does for plugins that expose it."
        ),
    )
    ds.add_argument(
        "--camera-map",
        default=None,
        help=advanced(
            "With --output-layout: comma-separated old=new camera names; "
            "defaults to the layout's own map."
        ),
    )
    ds.add_argument(
        "--keep-canonical",
        action="store_true",
        help=advanced(
            "With --output-layout: also keep the canonical dataset under its "
            "default '<source>-<robot>-joints' name."
        ),
    )
    ds.add_argument(
        "--replace-output",
        action="store_true",
        help=advanced("With --output-layout: replace an existing output directory."),
    )
    ds.add_argument(
        "--episodes",
        default=None,
        help=(
            "Comma-separated list of episode indices to process "
            "(default: all episodes)."
        ),
    )
    ds.add_argument(
        "--task",
        default=None,
        help=(
            "Override the task description for all episodes.  "
            "When not set the script tries to read tasks from the source "
            "dataset's tasks.parquet."
        ),
    )
    ds.add_argument(
        "--push-to-hub",
        action="store_true",
        help=advanced("Push the resulting dataset to the Hugging Face Hub."),
    )
    ds.add_argument(
        "--hub-token",
        default=None,
        help=advanced("Hugging Face token; defaults to HF_TOKEN."),
    )
    ds.add_argument(
        "--quality-config",
        type=Path,
        default=Path("configs/quality.yaml"),
        help=advanced("Offline episode-acceptance thresholds."),
    )
    ds.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help=advanced("Convert rejected episodes too; debugging only."),
    )

    # ------------------------------------------------------------------
    # Embodiment selection
    # ------------------------------------------------------------------
    emb = parser.add_argument_group("Embodiment")
    emb.add_argument(
        "--robot",
        dest="embodiment",
        choices=EMBODIMENT_NAMES,
        default="axol",
        help="Target robot; loads its validated retargeting profile.",
    )

    # ------------------------------------------------------------------
    # Shared IK parameters
    # ------------------------------------------------------------------
    ik = parser.add_argument_group("Shared IK parameters")
    ik.add_argument("--scale", type=float, default=1.0, help=advanced("Legacy global scale."))
    ik.add_argument(
        "--axis-map",
        default=None,
        help=advanced(
            "PICO delta → robot delta mapping, e.g. z,x,y or z,y,-x.  "
            "Defaults to the selected embodiment's validated mapping."
        ),
    )
    ik.add_argument("--left-only", action="store_true", help=advanced("Solve only the left arm."))
    ik.add_argument("--right-only", action="store_true", help=advanced("Solve only the right arm."))
    ik.add_argument(
        "--gripper",
        type=float,
        default=1.0,
        help=advanced("Fallback gripper opening in [0, 1], used only when the recording "
        "carries no Feetech widths (e.g. --skip-feetech). Otherwise the "
        "recorded opening drives the finger joints frame by frame."),
    )
    ik.add_argument(
        "--gripper-max-width-m",
        type=float,
        default=None,
        help=advanced("HandUMI full opening (m) that maps to the robot gripper fully "
        "open; defaults to the selected robot configuration. Recorded normalized "
        "Feetech values take precedence in replay-parity conversion."),
    )
    ik.add_argument("--pos-weight", type=float, default=None, help=advanced("IK position weight."))
    ik.add_argument("--ori-weight", type=float, default=None, help=advanced("IK orientation weight."))
    ik.add_argument("--max-joint-delta", type=float, default=None, help=advanced("IK joint step limit."))
    ik.add_argument("--max-reach", type=float, default=None, help=advanced("Workspace reach limit."))
    ik.add_argument(
        "--retarget-mode",
        choices=("auto", "local-relative", "anchored", "absolute-table", "absolute-tcp"),
        default="auto",
        help=advanced(
            "Retargeting mode shared with replay_in_sim.py. auto selects "
            "absolute-table when the source dataset declares a calibrated table "
            "workspace, otherwise local-relative."
        ),
    )
    ik.add_argument(
        "--compose-source",
        choices=("commanded", "achieved"),
        default="commanded",
        help=advanced("For local-relative mode, compose on previous target or achieved FK."),
    )
    ik.add_argument("--base-placement", choices=("auto", "fixed", "episode-xy"), default="auto")
    ik.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help=advanced("Scale local-relative translation deltas after frame adaptation."),
    )
    ik.add_argument(
        "--controller-device",
        choices=("pico", "meta", "vive"),
        default=None,
        help=advanced(
            "Override the source tracking device. Defaults to dataset metadata, "
            f"then {DEFAULT_CONTROLLER_DEVICE!r} for legacy datasets."
        ),
    )
    ik.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=advanced("Override the source robot/device Controller->TCP calibration."),
    )
    ik.add_argument(
        "--raw-controller-debug",
        action="store_true",
        help=advanced("Use raw controller poses without Controller->TCP calibration."),
    )
    ik.add_argument(
        "--robot-tcp-calibration", type=Path, default=None,
        help=advanced("Robot end-link to TCP calibration snapshot."),
    )
    ik.add_argument(
        "--deployment-calibration",
        type=Path,
        default=None,
        help=advanced(
            "YAML containing robot_from_table for absolute-table conversion. "
            "Overrides both lab-local and canonical simulation profiles."
        ),
    )
    ik.add_argument(
        "--deployment-profile",
        choices=("auto", "local", "sim"),
        default="auto",
        help=(
            "Table placement source: auto uses rig.yaml when configured, "
            "otherwise the canonical simulation profile."
        ),
    )
    ik.add_argument(
        "--absolute-orientation",
        choices=("relative-start", "table-absolute"),
        default=None,
        help=advanced("Absolute-table tool orientation policy used by replay."),
    )
    ik.add_argument("--initial-solve-iterations", type=int, default=12, help=advanced("Initial unrestricted IK iterations."))
    ik.add_argument("--initial-position-tolerance-m", type=float, default=None, help=advanced("Initial TCP tolerance."))
    ik.add_argument("--max-ik-position-error-m", type=float, default=None, help=advanced("Maximum TCP position error."))
    ik.add_argument("--max-ik-rotation-error-deg", type=float, default=None, help=advanced("Maximum TCP rotation error."))
    ik.add_argument("--table-clearance-warning-m", type=float, default=0.10, help=advanced("Table-clearance warning threshold."))
    ik.add_argument(
        "--strict-ik",
        action="store_true",
        help=advanced("Reject an episode when IK thresholds are exceeded."),
    )

    parser.add_argument(
        "--help-advanced", action="store_true", help="Show expert IK and calibration options."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Processes solving episodes at once. Episodes are independent, so "
        "this changes wall-clock only. The default leaves room for the threads "
        "the solver already uses inside one episode; 1 solves them in order.",
    )
    parser.add_argument(
        "--no-solve-cache",
        action="store_true",
        help=(
            "Re-solve every episode instead of reusing the trajectories "
            "screening produced under identical solver settings."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory for complete per-episode replay caches.")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Recompute complete per-episode caches.")
    parser.add_argument(
        "--allow-flagged-episodes",
        action="store_true",
        help=(
            "Convert even when the retargeting screening is missing, stale, or "
            "still flags episodes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the conversion plan without loading episodes.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


JOINT_DATASET_SUFFIX = "joints"


def _default_output_repo_id(source_repo_id: str, embodiment: str) -> str:
    """Derive ``{namespace}/{source_name}-{embodiment}-joints``.

    The name states both halves of what makes the dataset specific: the
    embodiment whose kinematics produced the values, and that they are joint
    angles rather than the robot-agnostic capture. A raw dataset converted for
    two robots then yields two names that cannot be mistaken for each other or
    for their source.
    """
    repo_id = source_repo_id.rstrip("/")
    suffix = f"-{embodiment}-{JOINT_DATASET_SUFFIX}"
    if "/" in repo_id:
        namespace, name = repo_id.rsplit("/", 1)
        return f"{namespace}/{name}{suffix}"
    return f"{repo_id}{suffix}"


def _resolve_conversion_output(
    value: str | None,
    *,
    source_repo_id: str,
    embodiment: str,
    layout_suffix: str | None = None,
) -> tuple[str, Path]:
    if value is None:
        if layout_suffix is None:
            repo_id = _default_output_repo_id(source_repo_id, embodiment)
        else:
            # The plugin's robot_type says both which robot and which vector.
            repo_id = f"{source_repo_id.rstrip('/')}-{layout_suffix}"
        return repo_id, dataset_root_from_repo_id(repo_id)

    candidate = Path(value).expanduser()
    looks_local = (
        value.startswith((".", "/", "~", "outputs/"))
        or candidate.exists()
        or candidate.parent.exists()
    )
    if looks_local:
        return f"local/{candidate.name}", candidate
    return value, dataset_root_from_repo_id(value)


def _capture_target_robot(source_info: dict[str, Any]) -> str | None:
    """Name of the robot profile the capture itself recorded, if any."""
    handumi = source_info.get("handumi")
    target = handumi.get("target_robot") if isinstance(handumi, dict) else None
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    return None


def _resolve_output_layout(parser: argparse.ArgumentParser, args: argparse.Namespace):
    """The LeRobot layout to write, or None for the canonical vector."""
    if args.output_layout == "canonical":
        return None
    from handumi.dataset.external_layouts import (
        EXTERNAL_LAYOUTS,
        external_layout_for_name,
    )

    try:
        layout = external_layout_for_name(args.output_layout)
    except ValueError:
        known = ", ".join(sorted(EXTERNAL_LAYOUTS))
        parser.error(f"--output-layout must be 'canonical' or one of: {known}.")
    if layout.robot != args.embodiment:
        parser.error(
            f"--output-layout {layout.robot_type} describes {layout.robot!r}; "
            f"pass --robot {layout.robot} or choose a layout for {args.embodiment!r}."
        )
    if args.use_degrees:
        from handumi.scripts.export_dataset import apply_use_degrees

        try:
            layout = apply_use_degrees(layout, True)
        except SystemExit as exc:
            parser.error(str(exc))
    return layout


def _resolve_cli_profile(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Resolve conversion defaults that do not require source dataset metadata."""
    args.piper = args.embodiment == "piper"
    if getattr(args, "refresh_cache", False) and args.no_solve_cache:
        parser.error("--refresh-cache and --no-solve-cache are mutually exclusive.")
    from handumi.scripts.replay.replay_in_sim import resolve_replay_options

    resolve_replay_options(args)

    if args.initial_solve_iterations < 1:
        parser.error("--initial-solve-iterations must be >= 1.")
    if args.initial_position_tolerance_m <= 0.0:
        parser.error("--initial-position-tolerance-m must be > 0.")
    if args.max_ik_position_error_m <= 0.0:
        parser.error("--max-ik-position-error-m must be > 0.")
    if args.max_ik_rotation_error_deg <= 0.0:
        parser.error("--max-ik-rotation-error-deg must be > 0.")
    if args.table_clearance_warning_m <= 0.0:
        parser.error("--table-clearance-warning-m must be > 0.")


def _resolve_retarget_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    source_info: dict[str, Any],
) -> None:
    """Resolve ``--retarget-mode auto`` with the same rule replay uses."""
    from handumi.scripts.replay.replay_in_sim import (
        _resolved_retarget_mode as resolve_replay_retarget_mode,
    )

    requested = args.retarget_mode
    args.retarget_mode = resolve_replay_retarget_mode(args, source_info)
    if requested == "auto":
        workspace = handumi_metadata(source_info).get("tracking_workspace")
        print(
            f"Retarget mode: auto -> {args.retarget_mode} "
            f"(tracking_workspace={workspace!r})"
        )

    if args.retarget_mode in {"absolute-table", "absolute-tcp"}:
        try:
            selection = resolve_deployment_calibration(
                args.embodiment,
                explicit_path=args.deployment_calibration,
                profile=args.deployment_profile,
                rig_config=args.rig_config,
                identity_fallback=args.retarget_mode == "absolute-tcp",
            )
        except SystemExit as exc:
            parser.error(str(exc))
        args.deployment_selection = selection
        args.deployment_calibration = selection.path
        print_deployment_calibration(selection, prefix="[convert]")
        if args.retarget_mode == "absolute-tcp":
            from handumi.calibration.robot_tcp import load_robot_tcp_calibration
            from handumi.robots.registry import load_robot_config

            config = load_robot_config(args.embodiment)
            args.robot_tcp_selection = load_robot_tcp_calibration(
                args.embodiment, config.arms,
                explicit_path=args.robot_tcp_calibration,
            )


def _deployment_calibration_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    selection = getattr(args, "deployment_selection", None)
    if selection is None:
        return None
    metadata = deployment_calibration_metadata(selection)
    metadata["sha256"] = (
        hashlib.sha256(selection.path.read_bytes()).hexdigest()
        if selection.path is not None else None
    )
    metadata["robot_from_table_pose7"] = selection.pose7.tolist()
    if args.retarget_mode == "absolute-tcp":
        metadata["role"] = "reference"
        metadata["per_episode"] = True
    return metadata


def _write_calibration_catalog(output_root: Path, args: argparse.Namespace) -> None:
    if args.raw_controller_debug:
        return
    selection = getattr(args, "controller_tcp_selection", None)
    if selection is None:
        return
    calibration_dir = output_root / "meta" / "calibrations"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "000000.json").write_text(
        json.dumps(
            {
                "calibration_id": 0,
                "type": "controller_tcp",
                "metadata": selection.metadata,
            },
            indent=4,
        )
    )


def _parse_episode_list(value: str | None, max_episodes: int) -> list[int]:
    if value is None:
        return list(range(max_episodes))
    indices = [int(x.strip()) for x in value.split(",") if x.strip()]
    out_of_range = [idx for idx in indices if idx < 0 or idx >= max_episodes]
    if out_of_range:
        raise ValueError(
            f"Episode indices out of range [0, {max_episodes - 1}]: {out_of_range}"
        )
    return indices


def _load_source_tasks(source_root: Path) -> dict[int, str]:
    """Return a mapping ``{episode_index: task_string}`` from tasks.parquet."""
    import pandas as pd

    tasks_path = source_root / "meta" / "tasks.parquet"
    episodes_dir = source_root / "meta" / "episodes"
    task_map: dict[int, str] = {}

    if not tasks_path.exists():
        return task_map

    tasks_df = pd.read_parquet(tasks_path)
    task_idx_to_str: dict[int, str] = {
        int(row_scalar(row, "task_index")): str(task_str)
        for task_str, row in tasks_df.iterrows()
    }

    if not episodes_dir.exists():
        return task_map

    import glob

    for parquet_path in sorted(glob.glob(str(episodes_dir / "**/*.parquet"), recursive=True)):
        ep_df = pd.read_parquet(parquet_path)
        for _, row in ep_df.iterrows():
            ep_idx = int(row_scalar(row, "episode_index"))
            task_list = row.get("tasks", [])
            if task_list and len(task_list) > 0:
                first_task_idx = task_list[0] if isinstance(task_list[0], (int, np.integer)) else 0
                task_map[ep_idx] = task_idx_to_str.get(int(first_task_idx), "task")
    return task_map


def _source_tcp_snapshot(source_info: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = handumi_metadata(source_info).get("controller_tcp_calibration")
    if not isinstance(snapshot, dict) or snapshot.get("applied_to_state") is True:
        return None
    return dict(snapshot)


def _resolve_conversion_tcp_calibration(
    args: argparse.Namespace,
    source_info: dict[str, Any],
) -> ConversionTcpCalibrationSelection:
    """Resolve the source robot-tool transform before converting to target joints."""
    metadata = handumi_metadata(source_info)
    snapshot = _source_tcp_snapshot(source_info)
    target_robot = metadata.get("target_robot")
    legacy_source_robot = (
        str(target_robot.get("name"))
        if isinstance(target_robot, dict) and target_robot.get("name")
        else ""
    )
    source_robot = str(
        (snapshot or {}).get("source_robot")
        or legacy_source_robot
        or args.embodiment
    )
    snapshot_device = str((snapshot or {}).get("tracking_device") or "")
    if (
        args.controller_device is not None
        and snapshot is not None
        and is_identity_bound_controller_tcp_metadata(snapshot)
        and str(args.controller_device) != snapshot_device
    ):
        raise ValueError(
            f"--controller-device {args.controller_device!r} conflicts with the "
            f"identity-bound dataset snapshot ({snapshot_device!r})."
        )
    controller_device = str(
        args.controller_device
        or (snapshot or {}).get("tracking_device")
        or metadata.get("recording_device")
        or DEFAULT_CONTROLLER_DEVICE
    )
    args.controller_device = controller_device

    try:
        source_config = load_robot_config(source_robot)
    except (TypeError, ValueError):
        source_config = None
    source_gripper = (
        source_config.handumi_gripper if source_config is not None else None
    )
    controller_mount = (
        source_config.handumi_controller_mount if source_config is not None else None
    )

    if args.controller_tcp_calibration is not None:
        path = args.controller_tcp_calibration
        calibration = load_controller_tcp_calibration(path)
        output_metadata = controller_tcp_calibration_metadata(
            path,
            applied_to_state=True,
            source_robot=source_robot,
            source_gripper=source_gripper,
            tracking_device=controller_device,
            controller_mount=controller_mount,
        )
        return ConversionTcpCalibrationSelection(
            calibration=calibration,
            metadata=output_metadata,
            source=f"explicit {path} sha256={output_metadata['sha256']}",
        )

    if snapshot is not None and is_identity_bound_controller_tcp_metadata(snapshot):
        output_metadata = dict(snapshot)
        output_metadata["applied_to_state"] = True
        source_identity = snapshot.get("source_robot") or snapshot.get("source_tool")
        return ConversionTcpCalibrationSelection(
            calibration=controller_tcp_calibration_from_metadata(snapshot),
            metadata=output_metadata,
            source=(
                "dataset robot-tool snapshot "
                f"{source_identity}/{snapshot['tracking_device']} "
                f"sha256={snapshot.get('sha256', 'unknown')}"
            ),
        )

    configured_path = (
        source_config.controller_tcp_calibrations.get(controller_device)
        if source_config is not None
        else None
    )
    if configured_path is not None:
        output_metadata = controller_tcp_calibration_metadata(
            configured_path,
            applied_to_state=True,
            source_robot=source_robot,
            source_gripper=source_gripper,
            tracking_device=controller_device,
            controller_mount=controller_mount,
        )
        return ConversionTcpCalibrationSelection(
            calibration=load_controller_tcp_calibration(configured_path),
            metadata=output_metadata,
            source=(
                f"configured {source_robot}/{controller_device}: {configured_path} "
                f"sha256={output_metadata['sha256']}"
            ),
        )

    if snapshot is not None:
        output_metadata = dict(snapshot)
        output_metadata["applied_to_state"] = True
        return ConversionTcpCalibrationSelection(
            calibration=controller_tcp_calibration_from_metadata(snapshot),
            metadata=output_metadata,
            source=f"legacy dataset metadata sha256={snapshot.get('sha256', 'unknown')}",
        )

    fallback_path = calibration_path_for_device(controller_device)
    output_metadata = controller_tcp_calibration_metadata(
        fallback_path,
        applied_to_state=True,
        source_robot=source_robot,
        tracking_device=controller_device,
    )
    return ConversionTcpCalibrationSelection(
        calibration=load_controller_tcp_calibration(fallback_path),
        metadata=output_metadata,
        source=(
            f"legacy device fallback {controller_device}: {fallback_path} "
            f"sha256={output_metadata['sha256']}"
        ),
    )


def _apply_tcp_calibration_to_states(
    states: np.ndarray,
    calibration: ControllerTcpCalibration,
) -> np.ndarray:
    """Return raw states whose left/right pose slots are calibrated TCP poses."""
    raw_left: list[np.ndarray] = []
    raw_right: list[np.ndarray] = []
    for state in states:
        left, right = raw_state_pose7_pair(state)
        raw_left.append(left)
        raw_right.append(right)

    left_tcp, right_tcp = apply_controller_tcp_calibration(
        np.asarray(raw_left, dtype=np.float32),
        np.asarray(raw_right, dtype=np.float32),
        calibration,
    )
    calibrated = np.asarray(states, dtype=np.float32).copy()
    calibrated[:, LEFT_POSE_SLICE] = left_tcp
    calibrated[:, RIGHT_POSE_SLICE] = right_tcp
    return calibrated


def _solver_config(runtime, args: argparse.Namespace):
    base = runtime.config.ik_weights

    def override(name: str, fallback):
        value = getattr(args, name, None)
        return fallback if value is None else float(value)

    return runtime.config_cls(
        pos_weight=override("pos_weight", base.pos_weight),
        ori_weight=override("ori_weight", base.ori_weight),
        rest_weight=override("rest_weight", base.rest_weight),
        posture_weight=override("posture_weight", base.posture_weight),
        manipulability_weight=override(
            "manipulability_weight",
            base.manipulability_weight,
        ),
        max_joint_delta=override("max_joint_delta", base.max_joint_delta)
        if base.max_joint_delta is not None or args.max_joint_delta is not None
        else None,
        max_reach=override("max_reach", base.max_reach)
        if base.max_reach is not None or args.max_reach is not None
        else None,
    )


def solve_joint_trajectory_from_raw_states(
    *,
    args: argparse.Namespace,
    states: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Solve joints with the same retargeting loop used by replay_in_sim.py."""
    if args.raw_controller_debug:
        states_for_retarget = np.asarray(states, dtype=np.float32)
        calibration_source = None
    else:
        selection = getattr(args, "controller_tcp_selection", None)
        if selection is None:
            calibration = load_controller_tcp_calibration(
                args.controller_tcp_calibration
            )
            calibration_source = str(args.controller_tcp_calibration)
        else:
            calibration = selection.calibration
            calibration_source = selection.source
        states_for_retarget = _apply_tcp_calibration_to_states(
            states,
            calibration,
        )

    runtime = load_embodiment(args.embodiment)
    cfg = _solver_config(runtime, args)
    q = runtime.config.home_q.astype(np.float32).copy()
    solver = runtime.solver_cls(config=cfg)
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    first_left_pose7, first_right_pose7 = raw_state_pose7_pair(states_for_retarget[0])

    anchors = None
    left_adapter = None
    right_adapter = None
    if args.retarget_mode == "anchored":
        anchors = retarget_anchors_from_raw_state(
            states_for_retarget[0],
            left_robot_pose7=home_left_pose7,
            right_robot_pose7=home_right_pose7,
            max_reach=cfg.max_reach,
        )
    else:
        left_adapter = local_frame_adapter(first_left_pose7, home_left_pose7)
        right_adapter = local_frame_adapter(first_right_pose7, home_right_pose7)

    qs: list[np.ndarray] = []
    left_targets: list[np.ndarray] = []
    right_targets: list[np.ndarray] = []
    left_achieved: list[np.ndarray] = []
    right_achieved: list[np.ndarray] = []

    for i, state in enumerate(states_for_retarget):
        raw_left, raw_right = raw_state_pose7_pair(state)
        if args.retarget_mode == "anchored":
            if anchors is None:
                raise RuntimeError("Anchored retarget mode was not initialized.")
            left_pose7, right_pose7 = raw_state_robot_target_pose7(state, anchors)
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        elif i == 0:
            left_pose7 = home_left_pose7.copy()
            right_pose7 = home_right_pose7.copy()
            fk_left_pose7 = home_left_pose7.copy()
            fk_right_pose7 = home_right_pose7.copy()
        else:
            if left_adapter is None or right_adapter is None:
                raise RuntimeError("Local-relative retarget mode was not initialized.")
            prev_left, prev_right = raw_state_pose7_pair(states_for_retarget[i - 1])
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

        qs.append(q.copy())
        left_targets.append(left_pose7)
        right_targets.append(right_pose7)
        left_achieved.append(fk_left_pose7)
        right_achieved.append(fk_right_pose7)
        if (i + 1) % 100 == 0 or (i + 1) == len(states_for_retarget):
            print(f"    frame {i + 1}/{len(states_for_retarget)}", end="\r", flush=True)

    print()
    joints = np.asarray(qs, dtype=np.float32)
    _write_gripper_joints(joints, states=states, runtime=runtime, args=args)
    return joints, calibration_source


def _write_gripper_joints(
    joints: np.ndarray,
    *,
    states: np.ndarray,
    runtime,
    args: argparse.Namespace,
) -> None:
    """Drive the finger joints from the recorded HandUMI opening widths.

    ``state[14]``/``state[15]`` carry the left/right opening in meters;
    normalized by --gripper-max-width-m and scaled to each finger's URDF
    range (same mapping ``handumi teleop`` renders). Recordings without Feetech
    (widths all zero) fall back to the constant --gripper opening.
    """
    widths_m = np.asarray(states, dtype=np.float32)[
        :, [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]
    ]
    configured_max_width = (
        runtime.config.gripper_max_width_m
        if args.gripper_max_width_m is None
        else args.gripper_max_width_m
    )
    max_w = max(float(configured_max_width), 1e-6)
    normalized = np.clip(widths_m / max_w, 0.0, 1.0)
    if not np.any(widths_m > 0):
        normalized[:] = float(np.clip(args.gripper, 0.0, 1.0))
        print(f"    no recorded widths — constant gripper opening {args.gripper:.2f}")
    for column, side in enumerate(("left", "right")):
        for finger in runtime.finger_joints.get(side, ()):
            if hasattr(finger, "closed_value") and hasattr(finger, "open_value"):
                joints[:, finger.index] = finger.closed_value + (
                    normalized[:, column] * (finger.open_value - finger.closed_value)
                )
            else:
                # Compatibility with lightweight test/runtime doubles.
                joint_index, open_value = finger
                joints[:, joint_index] = normalized[:, column] * open_value


# ---------------------------------------------------------------------------
# Per-episode IK processing
# ---------------------------------------------------------------------------


def _presolve_worker_state(**_: object) -> dict[str, object]:
    """Nothing shared to build: solve_episode caches the embodiment per process."""
    return {}


def _presolve_one(episode: int, _state: dict[str, object]) -> dict[str, object]:
    import contextlib
    import io

    from handumi.scripts.replay.replay_prepare import prepare_episode as _solve

    args = _PRESOLVE_ARGS["args"]
    replay_args = _replay_args_for_episode(args, episode)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            rollout = _solve(replay_args)
    except SystemExit as exc:
        return {"episode": episode, "error": str(exc)}
    return {
        "episode": episode,
        "rollout": {
            key: np.asarray(value)
            for key, value in rollout.items()
            if isinstance(value, np.ndarray)
        },
    }


_PRESOLVE_ARGS: dict[str, Any] = {}


def _cached_solve_for(
    args: argparse.Namespace, episode: int
) -> dict[str, np.ndarray] | None:
    """The screening solve for this episode, when it is still valid."""
    if args.no_solve_cache or getattr(args, "refresh_cache", False):
        return None
    selection = getattr(args, "deployment_selection", None)
    return load_cached_solve(
        args.root,
        robot=args.embodiment,
        episode=episode,
        signature=solver_signature(
            _replay_args_for_episode(args, episode),
            deployment_path=selection.path if selection is not None else None,
        ),
    )


def presolve_episodes(
    args: argparse.Namespace,
    episodes: list[int],
    *,
    jobs: int | None,
) -> dict[int, dict[str, np.ndarray]]:
    """Solve the episodes the screening cache does not already hold.

    Conversion writes one dataset sequentially, but the IK it needs per episode
    is independent, so the solving is done up front and the writing stays in
    order. Episodes already cached by screening are skipped: the cached solve is
    the one the review graded, and re-solving it would only invite the float32
    drift the cache exists to avoid.
    """
    from handumi.dataset.parallel import describe_jobs, map_episodes, resolve_jobs

    missing = [
        episode
        for episode in episodes
        if _cached_solve_for(args, episode) is None
    ]
    if not missing:
        return {}
    workers = resolve_jobs(jobs, len(missing))
    if workers <= 1:
        return {}
    print(f"[convert] pre-solving {describe_jobs(workers, len(missing))}")
    _PRESOLVE_ARGS["args"] = args
    results = map_episodes(
        missing,
        setup=_presolve_worker_state,
        setup_kwargs={},
        task=_presolve_one,
        jobs=workers,
    )
    solved: dict[int, dict[str, np.ndarray]] = {}
    for item in results:
        if "rollout" in item:
            solved[int(item["episode"])] = item["rollout"]
    return solved


def _replay_args_for_episode(
    args: argparse.Namespace, source_episode_index: int
) -> argparse.Namespace:
    """The replay solver's arguments for one episode.

    Shared by the in-line solve and the parallel pre-solve so both ask the
    solver for exactly the same thing; a divergence here would show up as
    joints that disagree with the screening report that graded them.
    """
    from handumi.scripts.replay.replay_in_sim import (
        build_parser as build_replay_parser,
    )

    replay_args = build_replay_parser().parse_args([str(args.root)])
    replay_args.repo_id = args.repo_id
    replay_args.dataset_root = args.root
    replay_args.revision = args.revision
    replay_args.episode = source_episode_index
    replay_args.robot = args.embodiment
    replay_args.source = args.source
    replay_args.retarget_mode = args.retarget_mode
    replay_args.base_placement = getattr(args, "base_placement", "auto")
    replay_args.compose_source = args.compose_source
    replay_args.translation_scale = args.translation_scale
    replay_args.controller_device = args.controller_device
    replay_args.controller_tcp_calibration = args.controller_tcp_calibration
    replay_args.raw_controller_debug = args.raw_controller_debug
    replay_args.deployment_calibration = args.deployment_calibration
    replay_args.deployment_profile = args.deployment_profile
    replay_args.rig_config = args.rig_config
    replay_args.deployment_selection = getattr(args, "deployment_selection", None)
    replay_args.robot_tcp_calibration = getattr(args, "robot_tcp_calibration", None)
    replay_args.absolute_orientation = args.absolute_orientation
    replay_args.initial_solve_iterations = args.initial_solve_iterations
    replay_args.initial_position_tolerance_m = args.initial_position_tolerance_m
    replay_args.gripper_max_width_m = args.gripper_max_width_m
    replay_args.max_ik_position_error_m = args.max_ik_position_error_m
    replay_args.max_ik_rotation_error_deg = args.max_ik_rotation_error_deg
    replay_args.table_clearance_warning_m = args.table_clearance_warning_m
    replay_args.strict_ik = args.strict_ik or args.retarget_mode == "absolute-tcp"
    replay_args.cache_dir = getattr(args, "cache_dir", None)
    replay_args.no_cache = args.no_solve_cache
    replay_args.refresh_cache = getattr(args, "refresh_cache", False)
    return replay_args


def _solve_with_replay_pipeline(
    *,
    args: argparse.Namespace,
    source_episode_index: int,
) -> dict[str, np.ndarray]:
    """Run the replay solver so absolute-table conversion has exact qpos parity."""
    from handumi.scripts.replay.replay_prepare import (
        load_prepared_episode,
        prepare_episode,
    )

    replay_args = _replay_args_for_episode(args, source_episode_index)
    # Screening already solved every episode with this solver; repeating it here
    # doubles the IK cost of the pipeline for an identical result. The cached
    # trajectory is only used when the solver settings match exactly, and the
    # gate has already refused a stale dataset or robot geometry.
    if not args.no_solve_cache and not getattr(args, "refresh_cache", False):
        complete = load_prepared_episode(replay_args)
        if complete is not None:
            _require_absolute_tcp_fidelity(args, complete)
            print(f"[convert] reusing complete replay cache for episode {source_episode_index}")
            return complete
        selection = getattr(args, "deployment_selection", None)
        cached = load_cached_solve(
            args.root,
            robot=args.embodiment,
            episode=source_episode_index,
            signature=solver_signature(
                replay_args,
                deployment_path=selection.path if selection is not None else None,
            ),
        )
        if cached is not None:
            _require_absolute_tcp_fidelity(args, cached)
            print(
                f"[convert] reusing the screened solve for episode "
                f"{source_episode_index}"
            )
            return cached
    presolved = getattr(args, "presolved", {}).get(source_episode_index)
    if presolved is not None:
        _require_absolute_tcp_fidelity(args, presolved)
        return presolved
    return prepare_episode(replay_args)


def _require_absolute_tcp_fidelity(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    """复核缓存和预求解的 TCP 误差，拒绝输出任何超差帧。"""
    if args.retarget_mode != "absolute-tcp":
        return
    frames = rollout.get("frame_indices", np.arange(len(rollout["qpos"])))
    if (len(frames) != len(rollout["qpos"]) or any(
        len(rollout[f"{side}_{metric}"]) != len(frames)
        for side in ("left", "right") for metric in ("pos_error_m", "rot_error_deg")
    )):
        raise SystemExit("Invalid cached TCP rollout: frame, joint and error lengths differ.")
    failures = []
    for i, frame in enumerate(frames):
        values = [float(rollout[f"{side}_{metric}"][i])
                  for side in ("left", "right")
                  for metric in ("pos_error_m", "rot_error_deg")]
        if not np.isfinite(rollout["qpos"][i]).all() or not np.isfinite(values).all():
            raise SystemExit(f"Invalid non-finite cached joint or TCP error at frame {int(frame)}.")
        pos = max(values[0], values[2])
        rot = max(values[1], values[3])
        if pos > args.max_ik_position_error_m or rot > args.max_ik_rotation_error_deg:
            failures.append(f"{int(frame)}: {pos * 1000:.2f}mm/{rot:.2f}deg")
    if failures:
        raise SystemExit("IK fidelity threshold exceeded; frames (position/rotation): " + ", ".join(failures))


def _canonical_command_states_from_rollout(
    rollout: dict[str, np.ndarray],
    *,
    runtime,
) -> np.ndarray:
    """Return YAML-derived arm joints plus one logical gripper width per side."""
    qpos = np.asarray(rollout["qpos"], dtype=np.float32)
    openings = rollout.get("gripper_normalized")
    return canonicalize_joint_trajectory(
        qpos,
        runtime=runtime,
        gripper_normalized=None
        if openings is None
        else np.asarray(openings, dtype=np.float32),
    )


def _piper_command_states_from_rollout(
    rollout: dict[str, np.ndarray],
    *,
    actuated_names: list[str] | tuple[str, ...],
    gripper_max_width_m: float,
) -> np.ndarray:
    """Backward-compatible Piper helper kept for older callers/tests."""
    qpos = np.asarray(rollout["qpos"], dtype=np.float32)
    openings = np.asarray(rollout["gripper_normalized"], dtype=np.float32)
    if openings.shape != (len(qpos), 2):
        raise ValueError(
            "Piper conversion requires replay gripper_normalized with shape "
            f"({len(qpos)}, 2), got {openings.shape}."
        )
    names = list(actuated_names)
    arm_indices = {
        side: [names.index(f"{side}_joint{joint}") for joint in range(1, 7)]
        for side in ("left", "right")
    }
    width_m = np.clip(openings, 0.0, 1.0) * np.float32(gripper_max_width_m)
    return np.column_stack(
        [
            qpos[:, arm_indices["left"]],
            width_m[:, 0],
            qpos[:, arm_indices["right"]],
            width_m[:, 1],
        ]
    ).astype(np.float32)


def _output_joint_names(
    args: argparse.Namespace,
    runtime,
) -> list[str]:
    del args
    return canonical_joint_layout(runtime).names


def process_episode(
    *,
    args: argparse.Namespace,
    states: np.ndarray,
    episode_index: int,
    source_episode_index: int,
    task: str,
) -> Any:
    """Run IK retargeting on one episode and return an EpisodeResult.

    A fresh retargeter is built for each episode so the local-relative
    calibration is relative to that episode's first frame.

    Parameters
    ----------
    args:
        Parsed CLI args.
    states:
        Raw HandUMI states of shape ``(T, 16)``.
    episode_index:
        Index in the *output* dataset.
    source_episode_index:
        Index in the *source* dataset (for video copying).
    task:
        Task description string.

    Returns
    -------
    EpisodeResult
    """
    from handumi.dataset import EpisodeResult

    if len(states) < 2:
        raise ValueError(
            f"Episode {source_episode_index} has fewer than 2 frames; "
            "cannot construct (state, action) pairs."
        )

    if args.retarget_mode in {"absolute-table", "absolute-tcp"} or getattr(args, "umi_v2", False):
        try:
            rollout = _solve_with_replay_pipeline(
                args=args,
                source_episode_index=source_episode_index,
            )
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        qpos = np.asarray(rollout["qpos"], dtype=np.float32)
        if len(qpos) != len(states):
            raise RuntimeError(
                "Replay-parity solver returned "
                f"{len(qpos)} frames for {len(states)} source frames."
            )
        if hasattr(args, "embodiment"):
            runtime = load_embodiment(args.embodiment)
            joint_array = _canonical_command_states_from_rollout(
                rollout,
                runtime=runtime,
            )
        else:
            joint_array = qpos
        args.ik_reports.append(
            {
                "source_episode_index": source_episode_index,
                "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                "output_state_sha256": hashlib.sha256(
                    joint_array.tobytes()
                ).hexdigest(),
                "frames": len(qpos),
                "retarget_mode": str(rollout["retarget_mode"][0]),
                "base_placement": {
                    "policy": str(rollout["base_placement_policy"][0]),
                    "scope": str(rollout["base_placement_scope"][0]),
                    "verified": bool(rollout["base_placement_verified"][0]),
                    "table_from_robot_pose7": rollout["table_from_robot_pose7"][0].tolist(),
                    "robot_from_table_pose7": rollout["robot_from_table_pose7"][0].tolist(),
                    "diagnostics": json.loads(str(rollout["base_placement_diagnostics_json"][0]) or "{}"),
                } if args.retarget_mode == "absolute-tcp" else None,
                "max_position_error_m": float(
                    max(
                        np.max(rollout["left_pos_error_m"]),
                        np.max(rollout["right_pos_error_m"]),
                    )
                ),
                "max_rotation_error_deg": float(
                    max(
                        np.max(rollout["left_rot_error_deg"]),
                        np.max(rollout["right_rot_error_deg"]),
                    )
                ),
                "initial_solve_iterations": int(
                    rollout["initial_solve_iterations"][0]
                ),
                "gripper_source": str(rollout["gripper_source"][0]),
            }
        )
    else:
        qpos, _ = solve_joint_trajectory_from_raw_states(
            args=args,
            states=states,
        )
        if hasattr(args, "embodiment"):
            runtime = load_embodiment(args.embodiment)
            joint_array = canonicalize_joint_trajectory(
                qpos,
                runtime=runtime,
                gripper_widths_m=raw_state_gripper_widths_m(states),
                fallback_gripper=getattr(args, "gripper", 1.0),
            )
        else:
            joint_array = qpos
    states = joint_array[:-1]               # t = 0 … T-2
    actions = joint_array[1:]               # t = 1 … T-1

    return EpisodeResult(
        episode_index=episode_index,
        states=states,
        actions=actions,
        task=task,
        source_episode_index=source_episode_index,
        calibration_id=-1 if getattr(args, "raw_controller_debug", False) else 0,
        source_kind=0,
        gripper_normalized=np.asarray(rollout["gripper_normalized"][:-1], dtype=np.float32) if getattr(args, "umi_v2", False) else None,
        action_gripper_normalized=np.asarray(rollout["gripper_normalized"][1:], dtype=np.float32) if getattr(args, "umi_v2", False) else None,
    )


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------


def push_to_hub(
    output_root: Path,
    repo_id: str,
    *,
    token: str | None = None,
) -> None:
    """Push the dataset directory to the HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --push-to-hub.  "
            "Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_root),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")


def _write_converted_dataset_readme(
    output_root: Path,
    *,
    repo_id: str,
    source_repo_id: str,
    embodiment: str,
) -> None:
    """Write a Hub card that identifies the joint-level derivation."""
    from lerobot.datasets.utils import create_lerobot_dataset_card

    from handumi.dataset.lerobot_compat import DatasetCardInfo

    info = json.loads((output_root / "meta" / "info.json").read_text())
    handumi_info = info.get("handumi", {})
    layout = str(handumi_info.get("state_layout", "full_sim_qpos"))
    if layout in {
        "bipiper_6dof_plus_gripper_width_m_per_side",
        "yaml_arm_joints_plus_logical_gripper_width_m",
    }:
        representation = (
            "Each side stores YAML-declared arm joints in radians and, when "
            "present, one physical gripper opening in meters."
        )
    elif isinstance(handumi_info.get("export"), dict):
        export = handumi_info["export"]
        representation = (
            f"The state is the LeRobot {export.get('robot_type')} follower vector "
            f"({', '.join(export.get('names', [])[:3])}, ...) in that plugin's own "
            "units, so it trains and deploys through its stack unchanged."
        )
    else:
        representation = "The state uses the selected embodiment joint layout."
    card = create_lerobot_dataset_card(
        tags=["HandUMI", embodiment, "joint-level"],
        dataset_info=DatasetCardInfo(info),
        license="other",
        repo_id=repo_id,
        dataset_description=(
            f"Joint-level {embodiment} dataset derived from "
            f"{source_repo_id}. {representation} observation.state[t] contains "
            "the command at t and action[t] contains the command at t+1."
        ),
        url="https://github.com/murobotics-ai/handumi-sw",
    )
    card.save(output_root / "README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    raw_argv = list(sys.argv[1:])
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = build_parser(show_advanced=show_advanced)
    if show_advanced:
        parser.print_help()
        return
    args = parser.parse_args(raw_argv)
    _resolve_cli_profile(parser, args)
    args.ik_reports = []

    if args.left_only and args.right_only:
        parser.error("Use only one of --left-only or --right-only.")
    from handumi.dataset.selection import resolve_dataset_selection

    try:
        selection = resolve_dataset_selection(
            args.dataset,
            revision=args.revision,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.repo_id = selection.repo_id
    args.root = selection.root

    # ------------------------------------------------------------------
    # Resolve paths and names
    # ------------------------------------------------------------------
    source_repo_id = args.repo_id
    source_root = selection.root
    output_layout = _resolve_output_layout(parser, args)
    output_repo_id, output_root = _resolve_conversion_output(
        args.output,
        source_repo_id=source_repo_id,
        embodiment=args.embodiment,
        layout_suffix=None if output_layout is None else output_layout.robot_type,
    )
    final_root = output_root
    if output_layout is not None:
        if final_root.exists():
            if not args.replace_output:
                parser.error(f"{final_root} exists; pass --replace-output to overwrite it.")
            shutil.rmtree(final_root)
        # The canonical dataset is still solved and written first: it is the
        # physical record and what the export transforms at the file level.
        output_root = final_root.parent / f".{final_root.name}.canonical"
        if output_root.exists():
            shutil.rmtree(output_root)

    # ------------------------------------------------------------------
    # Ensure source metadata is available, then resolve metadata-driven defaults
    # ------------------------------------------------------------------
    from handumi.dataset import ensure_metadata, validate_raw_state_metadata

    source_info = ensure_metadata(
        repo_id=source_repo_id,
        root=source_root,
        revision=args.revision,
    )
    try:
        validate_raw_state_metadata(source_info)
    except ValueError as exc:
        parser.error(str(exc))
    args.umi_v2 = source_info.get("handumi", {}).get("capture_schema") == "synchronized_sources_v2"
    _resolve_retarget_mode(parser, args, source_info)

    print(
        "Conversion plan\n"
        f"  Source: {source_root} ({source_repo_id})\n"
        f"  Output: {output_root} ({output_repo_id})\n"
        f"  Robot profile: {args.embodiment}\n"
        f"  Retargeting: {args.retarget_mode}\n"
        f"  Episodes: {args.episodes or 'all'}\n"
        f"  Output layout: {'canonical (radians + meters)' if output_layout is None else output_layout.robot_type + (' (use_degrees)' if output_layout.use_degrees and output_layout.degrees_option == 'optional' else '')}"
    )
    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # Resolve calibration, then read episode count
    # ------------------------------------------------------------------
    if not args.raw_controller_debug:
        try:
            args.controller_tcp_selection = _resolve_conversion_tcp_calibration(
                args,
                source_info,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"Could not resolve source Controller->TCP calibration: {exc}")
        print(f"Controller->TCP: {args.controller_tcp_selection.source}")
    total_source_episodes = int(source_info.get("total_episodes", 0))
    dataset_fps = int(source_info.get("fps", 30))

    if total_source_episodes <= 0:
        parser.error(
            f"Could not determine total_episodes from "
            f"{source_root / 'meta' / 'info.json'}. "
            "Check DATASET and --revision."
        )

    try:
        episode_indices = _parse_episode_list(args.episodes, total_source_episodes)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"Source episodes: {total_source_episodes}  "
        f"Processing: {len(episode_indices)}  "
        f"{episode_indices[:5]}{'…' if len(episode_indices) > 5 else ''}"
    )

    # Retargeting screening gate: never turn episodes a human has not cleared
    # for this embodiment into joint targets.
    gate = evaluate_screening_gate(
        source_root,
        robot=args.embodiment,
        episodes=episode_indices,
        expected_solver_signature=(
            solver_signature(
                _replay_args_for_episode(args, episode_indices[0]),
                deployment_path=args.deployment_selection.path,
            ) if args.retarget_mode == "absolute-tcp" and episode_indices else None
        ),
        expected_deployment_path=(args.deployment_selection.path
                                  if args.retarget_mode == "absolute-tcp" else None),
        expected_robot_tcp_path=(args.robot_tcp_selection.path
                                 if args.retarget_mode == "absolute-tcp" else None),
    )
    if gate.blocks:
        guidance = format_gate_guidance(
            gate, root=source_root, robot=args.embodiment
        )
        if not args.allow_flagged_episodes:
            raise SystemExit(f"[convert] {guidance}")
        print(f"[convert] warning: screening gate overridden.\n{gate.detail}")
    elif gate.status == "review":
        # Warnings were a reviewer's call during curation, so they are recorded
        # here rather than blocking; overriding for them would make the override
        # habitual and cover the checks that must never be waived.
        print(f"[convert] {gate.detail}")
        for index in sorted(gate.flagged):
            codes = ", ".join(
                str(finding.get("code")) for finding in gate.flagged[index]
            )
            print(f"[convert]   episode {index}: {codes}")
    else:
        print(f"[convert] screening: clear ({gate.report_path})")

    # ------------------------------------------------------------------
    # Load task descriptions for each episode
    # ------------------------------------------------------------------
    source_task_map = _load_source_tasks(source_root)
    default_task = args.task or "task"

    def get_task(ep_idx: int) -> str:
        if args.task:
            return args.task
        return source_task_map.get(ep_idx, default_task)

    # ------------------------------------------------------------------
    # Process each episode
    # ------------------------------------------------------------------
    from handumi.dataset import load_raw_episode
    from handumi.dataset.quality import EpisodeQualityConfig, validate_episode

    results = []
    quality_reports = []
    quality_config = None
    if not args.skip_quality_filter:
        try:
            quality_config = EpisodeQualityConfig.from_yaml(args.quality_config)
        except (OSError, ValueError) as exc:
            parser.error(f"Could not load quality config: {exc}")

    args.presolved = {}
    if args.retarget_mode in {"absolute-table", "absolute-tcp"}:
        args.presolved = presolve_episodes(args, list(episode_indices), jobs=args.jobs)

    for position, src_idx in enumerate(episode_indices, start=1):
        print(f"\nEpisode {position}/{len(episode_indices)}  (source ep {src_idx})")
        try:
            raw_episode = load_raw_episode(
                repo_id=source_repo_id,
                root=source_root,
                episode=src_idx,
                source=args.source,
                revision=args.revision,
                download_videos=True,
            )
        except Exception as exc:
            print(f"  SKIP: failed to load — {exc}", file=sys.stderr)
            continue

        raw_states = raw_episode.states
        if quality_config is not None:
            report = validate_episode(
                raw_states,
                fps=raw_episode.fps,
                signals=raw_episode.signals,
                episode_index=src_idx,
                config=quality_config,
            )
            quality_reports.append(report)
            if not report.accepted:
                reasons = ", ".join(
                    finding.code
                    for finding in report.findings
                    if finding.severity == "reject"
                )
                # Refuse rather than skip. Dropping episodes here produces an
                # output that silently disagrees with the curated input, and it
                # decides a second time what curation already decided once.
                if not args.allow_flagged_episodes:
                    raise SystemExit(
                        f"[convert] episode {src_idx} fails recording quality "
                        f"({reasons}).\n"
                        "Recording quality is decided during curation, not "
                        "here. Refresh the review and remove it:\n"
                        f"  handumi validate {source_root}\n"
                        f"  handumi dataset analyze {source_root}\n"
                        f"  handumi dataset curate {source_root} "
                        "--output <new_root> --exclude <indices>\n"
                        "To convert it anyway, pass --allow-flagged-episodes."
                    )
                print(
                    f"  WARNING: converting despite quality filter — {reasons}",
                    file=sys.stderr,
                )

        task = get_task(src_idx)
        try:
            result = process_episode(
                args=args,
                states=raw_states,
                episode_index=len(results),
                source_episode_index=src_idx,
                task=task,
            )
        except Exception as exc:
            print(f"  SKIP: IK failed — {exc}", file=sys.stderr)
            continue

        results.append(result)
        print(
            f"  Done: {len(result.states)} frames, "
            f"task={result.task!r}"
        )

    if not results:
        print("No episodes processed successfully. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Determine joint names and robot_type from the embodiment spec
    # ------------------------------------------------------------------
    runtime = load_embodiment(args.embodiment)
    robot_type = runtime.config.kind
    joint_names = _output_joint_names(args, runtime)

    # ------------------------------------------------------------------
    # Write output dataset
    # ------------------------------------------------------------------
    from handumi.dataset import write_dataset

    write_dataset(
        output_root=output_root,
        source_root=source_root,
        source_info=source_info,
        episodes=results,
        robot_type=robot_type,
        joint_names=joint_names,
        fps=dataset_fps,
        handumi_metadata={
            # The capture's target_robot is the HandUMI tool used to record;
            # a converted dataset must name the embodiment it was solved for.
            "target_robot": robot_config_metadata(args.embodiment),
            "capture_target_robot": _capture_target_robot(source_info),
            "conversion_source": args.source,
            "retarget_mode": args.retarget_mode,
            "base_placement": getattr(args, "base_placement", "auto"),
            "compose_source": args.compose_source,
            "translation_scale": float(args.translation_scale),
            "replay_qpos_parity": False,
            "replay_arm_qpos_parity": args.retarget_mode in {"absolute-table", "absolute-tcp"},
            "state_layout": "yaml_arm_joints_plus_logical_gripper_width_m",
            "gripper_representation": {
                "type": "opening_width",
                "unit": "m",
                "max_width_m": float(runtime.config.gripper_max_width_m),
                "source": (
                    "recorded HandUMI width in meters"
                    if runtime.config.replay_gripper_mode == "physical-width"
                    else "recorded Feetech normalized"
                ),
            },
            "deployment_calibration": _deployment_calibration_metadata(args),
            "robot_tcp_calibration": (args.robot_tcp_selection.metadata()
                                      if getattr(args, "robot_tcp_selection", None) is not None else None),
            "absolute_orientation": args.absolute_orientation,
            "initial_solve_iterations": int(args.initial_solve_iterations),
            "initial_position_tolerance_m": float(
                args.initial_position_tolerance_m
            ),
            "max_ik_position_error_m": float(args.max_ik_position_error_m),
            "max_ik_rotation_error_deg": float(args.max_ik_rotation_error_deg),
            "ik_fidelity": args.ik_reports,
            "controller_device": args.controller_device,
            "raw_controller_debug": bool(args.raw_controller_debug),
            "controller_tcp_calibration": (
                args.controller_tcp_selection.metadata
                if not args.raw_controller_debug
                else None
            ),
            "quality_filter_enabled": not args.skip_quality_filter,
            "quality_source_accepted": sum(
                report.accepted for report in quality_reports
            ),
            "quality_source_rejected": sum(
                not report.accepted for report in quality_reports
            ),
            "converted_source_episodes": len(results),
        },
    )
    _write_calibration_catalog(output_root, args)
    if quality_config is not None:
        from handumi.dataset.quality import write_quality_report

        write_quality_report(
            output_root / "meta" / "source_quality.json",
            quality_reports,
            config=quality_config,
            dataset=source_repo_id,
        )

    if output_layout is not None:
        from handumi.scripts.export_dataset import export_dataset, parse_camera_map

        video_keys = [
            key
            for key, feature in source_info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
        camera_map = parse_camera_map(
            args.camera_map, layout=output_layout, video_keys=video_keys
        )
        keep_root: Path | None = None
        if args.keep_canonical:
            keep_repo_id = _default_output_repo_id(source_repo_id, args.embodiment)
            keep_root = dataset_root_from_repo_id(keep_repo_id)
            if keep_root.exists() and not args.replace_output:
                shutil.rmtree(output_root)
                raise SystemExit(
                    f"--keep-canonical would replace {keep_root}; pass "
                    "--replace-output to allow it."
                )
        # Strict on purpose: the point of this layout is deployment, and the
        # plugin's driver freezes the robot on an out-of-range command.
        try:
            export_dataset(
                output_root,
                final_root,
                layout=output_layout,
                camera_map=camera_map,
                source_repo_id=source_repo_id,
                strict=True,
            )
        except SystemExit:
            # Leave nothing half-written behind: the export already removed
            # its own output, and the staging canonical is only a by-product.
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        if keep_root is not None:
            if keep_root.exists():
                shutil.rmtree(keep_root)
            shutil.move(str(output_root), str(keep_root))
            _write_converted_dataset_readme(
                keep_root,
                repo_id=_default_output_repo_id(source_repo_id, args.embodiment),
                source_repo_id=source_repo_id,
                embodiment=args.embodiment,
            )
            print(f"Kept canonical dataset at {keep_root}")
        else:
            shutil.rmtree(output_root)
        output_root = final_root

    _write_converted_dataset_readme(
        output_root,
        repo_id=output_repo_id,
        source_repo_id=source_repo_id,
        embodiment=args.embodiment,
    )
    print(f"Wrote {output_repo_id} to {output_root}")

    # ------------------------------------------------------------------
    # Optional: push to Hub
    # ------------------------------------------------------------------
    if args.push_to_hub:
        push_to_hub(output_root, output_repo_id, token=args.hub_token)


if __name__ == "__main__":
    main()
