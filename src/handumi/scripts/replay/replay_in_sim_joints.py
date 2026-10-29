#!/usr/bin/env python3
"""Replay a converted joint-level HandUMI dataset in simulation.

``handumi replay`` starts from the robot-agnostic capture and solves IK to
reach each frame. This command starts from a dataset a previous conversion
already solved, so no IK runs at all: the viewer shows exactly the joint
values the dataset stores, and forward kinematics only reports where those
joints put the TCPs.

That makes it the natural way to inspect what a policy will actually be
trained on, and ``--verify`` closes the loop: it proves the stored joints are
byte-identical to what conversion wrote, then re-solves the same capture
through the TCP replay pipeline and compares the two motions in task space.
The second half is deliberately not a bit-exact check -- the IK solver is not
reproducible run to run -- which is exactly why conversion reuses replay's
trajectory instead of solving again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from handumi.calibration.deployment import load_deployment_calibration
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset import DatasetRef, handumi_metadata, open_dataset
from handumi.dataset.canonical import (
    canonical_joint_layout,
    expand_canonical_trajectory,
    is_canonical_state_layout,
)
from handumi.dataset.external_layouts import (
    EXTERNAL_LAYOUTS,
    ExternalJointLayout,
    detect_external_layout,
    external_layout_for_name,
    to_canonical,
)
from handumi.dataset.paths import repo_path
from handumi.dataset.selection import resolve_dataset_selection
from handumi.robots.kinematics import pose_error_arrays
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment
from handumi.scripts.replay.replay_in_sim import (
    _column_float32,
    _render_table,
    _render_task_scene,
    joint_approach_ramp,
)

DEFAULT_OUT_DIR = Path("outputs/replay_in_sim")
JOINT_DATASET_SUFFIX = "joints"


def build_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        description=(
            "Replay an already-retargeted joint-level HandUMI dataset. No IK "
            "runs; forward kinematics only reports the resulting TCP poses."
        )
    )
    parser.add_argument(
        "dataset",
        help="Local path or Hugging Face repo id of a converted joints dataset.",
    )
    parser.add_argument(
        "--help-advanced", action="store_true", help="Show expert options."
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--robot",
        choices=EMBODIMENT_NAMES,
        default=None,
        help="Embodiment whose URDF renders the joints. Defaults to the "
        "target robot recorded in the dataset.",
    )
    parser.add_argument("--revision", default="main", help=advanced("Hub revision."))
    parser.add_argument(
        "--source",
        choices=("observation.state", "action"),
        default="observation.state",
        help=advanced(
            "Joint column to play. observation.state[t] is the command at t; "
            "action[t] is the command at t+1."
        ),
    )
    parser.add_argument(
        "--state-layout",
        choices=("auto", "handumi", *sorted(EXTERNAL_LAYOUTS)),
        default="auto",
        help=(
            "How the joint columns are encoded. 'handumi' is the canonical "
            "converted layout (radians + meters); an external layout name "
            "denormalizes that stack's vector first. 'auto' reads the dataset "
            "metadata."
        ),
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help=(
            "With an explicit --state-layout: the dataset was recorded with the "
            "plugin's use_degrees option (arm joints in degrees)."
        ),
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help=advanced("First frame.")
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help=advanced("Maximum frames.")
    )
    parser.add_argument("--stride", type=int, default=1, help=advanced("Frame stride."))
    parser.add_argument(
        "--approach-seconds",
        type=float,
        default=1.0,
        help=advanced(
            "Render-only lead-in ramping the robot from home to the first "
            "recorded pose. 0 starts playback directly at the first frame."
        ),
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help=advanced("Machine-local rig YAML (unused unless a scene is drawn)."),
    )
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "Render assets/scenes/<name>/scene.xml in the table frame the "
            "dataset was converted with."
        ),
    )
    parser.add_argument(
        "--hide-trajectories",
        action="store_true",
        help="Hide the TCP paths and markers, showing only the robot and scene.",
    )
    parser.add_argument(
        "--clean-view",
        action="store_true",
        help=(
            "Hide the table grid, axes and placement labels in the Viser viewer "
            "for clean screenshots on a white background."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Check the stored joints against the conversion hash and, when the "
            "source capture is available, re-solve the episode through the TCP "
            "replay pipeline and report the difference."
        ),
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=None,
        help=(
            "Raw capture this dataset was converted from, used by --verify. "
            "Defaults to the sibling directory named without the "
            "'-<robot>-joints' suffix."
        ),
    )
    parser.add_argument(
        "--verify-position-tolerance-mm",
        type=float,
        default=1.0,
        help=advanced(
            "Maximum accepted TCP position difference against the re-solved "
            "replay. The IK solver is not bit-reproducible, so agreement is "
            "measured in task space rather than by comparing joint values."
        ),
    )
    parser.add_argument(
        "--verify-rotation-tolerance-deg",
        type=float,
        default=0.5,
        help=advanced(
            "Maximum accepted TCP orientation difference against the re-solved "
            "replay."
        ),
    )
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Exit non-zero when a --verify check fails.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--port", type=int, default=8080, help=advanced("Viser port."))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the replay plan without loading the dataset.",
    )
    return parser


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


def dataset_info(root: Path) -> dict[str, object]:
    """Read ``meta/info.json`` for a local dataset root."""
    path = Path(root) / "meta" / "info.json"
    if not path.is_file():
        raise SystemExit(f"Not a LeRobot dataset root: missing {path}.")
    return json.loads(path.read_text())


def resolve_state_layout(
    args: argparse.Namespace, info: dict[str, object]
) -> ExternalJointLayout | None:
    """Return the external layout to denormalize from, or None for canonical.

    Getting this wrong is silent: an external 14-column vector has the same
    shape as the canonical one, so without the check it would render with
    normalized units read as radians.
    """
    requested = str(args.state_layout)
    if requested == "handumi":
        if not is_canonical_state_layout(info):
            raise SystemExit(_not_canonical_message(info))
        return None
    if requested != "auto":
        layout = external_layout_for_name(requested)
        if args.use_degrees:
            try:
                layout = layout.with_use_degrees(True)
            except ValueError as exc:
                raise SystemExit(f"--use-degrees: {exc}") from exc
        return layout
    if is_canonical_state_layout(info):
        return None
    detected = detect_external_layout(info)
    if detected is not None:
        return detected
    raise SystemExit(_not_canonical_message(info))


def _not_canonical_message(info: dict[str, object]) -> str:
    meta = handumi_metadata(info)
    layout = str(meta.get("state_layout", ""))
    known = ", ".join(sorted(EXTERNAL_LAYOUTS))
    return (
        "This command replays a joint-level dataset, but handumi.state_layout is "
        f"{layout or 'absent'!r} and robot_type {info.get('robot_type')!r} matches "
        f"no known external layout ({known}). Convert the capture first "
        "(`handumi convert`), pass --state-layout explicitly, or use "
        "`handumi replay` for a robot-agnostic recording."
    )


def resolved_robot(args: argparse.Namespace, info: dict[str, object]) -> str:
    """Prefer the explicit flag, then the robot recorded by conversion."""
    if args.robot is not None:
        return str(args.robot)
    target = handumi_metadata(info).get("target_robot")
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    robot_type = info.get("robot_type")
    if isinstance(robot_type, str) and robot_type in EMBODIMENT_NAMES:
        return robot_type
    external = getattr(args, "external_layout", None) or detect_external_layout(info)
    if external is not None:
        return external.robot
    raise SystemExit(
        "Dataset records no target robot; pass --robot explicitly."
    )


def load_joint_episode(
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(states, actions, fps)`` for one converted episode."""
    ref = DatasetRef.from_repo_id(
        args.repo_id,
        root=args.dataset_root,
        revision=args.revision,
    )
    dataset = open_dataset(ref, episode=args.episode, download_videos=False)
    fps = float(getattr(dataset, "fps", 30) or 30)
    table = dataset.hf_dataset
    missing = [
        key for key in ("observation.state", "action") if key not in table.column_names
    ]
    if missing:
        raise SystemExit(f"Dataset has no {', '.join(missing)} feature(s).")
    states = _column_float32(table, "observation.state")
    actions = _column_float32(table, "action")
    if len(states) == 0:
        raise SystemExit(f"Episode {args.episode} is empty.")
    prefix = "action" if getattr(args, "source", "observation.state") == "action" else "observation"
    keys = [f"{prefix}.gripper.{side}.normalized" for side in ("left", "right")]
    args.recorded_openings = None
    info = getattr(getattr(dataset, "meta", None), "info", {})
    if info.get("handumi", {}).get("capture_schema") == "synchronized_sources_v2" and not all(key in table.column_names for key in keys):
        raise ValueError("转换后的 UMI 数据缺少独立开度列")
    if all(key in table.column_names for key in keys):
        openings = np.column_stack([_column_float32(table, key).reshape(-1) for key in keys])
        if not np.isfinite(openings).all() or np.any((openings < 0) | (openings > 1)):
            raise ValueError("转换数据开度无效")
        args.recorded_openings = openings
    return states, actions, fps


def source_episode_index(root: Path, episode: int) -> int | None:
    """Map an output episode back to the capture episode conversion read."""
    import pyarrow.parquet as pq

    files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "source_episode_index"])
        for row in table.to_pylist():
            if int(row["episode_index"]) == episode:
                value = row.get("source_episode_index")
                return None if value is None else int(value)
    return None


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def build_rollout(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Expand the recorded joints and run forward kinematics over them."""
    info = args.dataset_info
    external = args.external_layout
    meta = handumi_metadata(info)
    runtime = load_embodiment(args.robot)
    layout = canonical_joint_layout(runtime)

    states, actions, fps = load_joint_episode(args)
    if states.ndim != 2 or states.shape[1] != layout.size:
        width = states.shape[1] if states.ndim == 2 else states.shape
        raise SystemExit(
            f"Expected {layout.size} joint columns for {args.robot}, got {width}. "
            "The dataset was converted for a different embodiment."
        )

    selected = states if args.source == "observation.state" else actions
    frame_indices = list(range(args.start_frame, len(selected), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise SystemExit("No frames selected for replay.")

    canonical = selected[frame_indices]
    if external is not None:
        canonical = to_canonical(canonical, layout=external, runtime=runtime)
    qpos, gripper_normalized = expand_canonical_trajectory(canonical, runtime=runtime)

    if getattr(args, "recorded_openings", None) is not None:
        gripper_normalized = args.recorded_openings[frame_indices]
    solver = runtime.solver_cls()
    start = time.perf_counter()
    left_tcp = np.empty((len(qpos), 7), dtype=np.float32)
    right_tcp = np.empty((len(qpos), 7), dtype=np.float32)
    for index, q in enumerate(qpos):
        left_tcp[index], right_tcp[index] = solver.fk_pose7(q)
    elapsed = time.perf_counter() - start

    approach_qpos = joint_approach_ramp(
        np.asarray(runtime.config.home_q, dtype=np.float32),
        qpos[0],
        frames=int(round(fps * args.approach_seconds)),
    )

    deployment = _deployment_pose7(meta)
    print(
        f"[replay-joints] robot={args.robot} episode={args.episode} "
        f"frames={len(qpos)} fps={fps:g} source={args.source} "
        f"fk={elapsed:.2f}s ({elapsed / len(qpos) * 1000:.1f} ms/frame)"
    )
    print(
        f"[replay-joints] joints: {layout.size} recorded columns -> "
        f"{qpos.shape[1]} URDF actuated joints "
        f"(state_layout={external.name if external else meta.get('state_layout')})"
    )
    if external is not None:
        print(f"[replay-joints] decoded {external.robot_type}: {external.describe()}")
    print(
        "[replay-joints] source conversion: "
        f"retarget={meta.get('retarget_mode')} "
        f"arm_qpos_parity={str(meta.get('replay_arm_qpos_parity')).lower()} "
        f"gripper={_gripper_source(meta)}"
    )
    if len(approach_qpos):
        print(
            f"[replay-joints] approach: {len(approach_qpos)} frames "
            f"({len(approach_qpos) / fps:.2f}s) from home"
        )

    return {
        "qpos": qpos,
        # Render-only lead-in, kept out of qpos so every per-frame array stays
        # index-aligned with the recorded episode.
        "approach_qpos": approach_qpos,
        "canonical_state": canonical,
        "gripper_normalized": gripper_normalized,
        "fk_left_pose7_robot_world": left_tcp,
        "fk_right_pose7_robot_world": right_tcp,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_names": np.asarray(list(runtime.joint_names)),
        "canonical_names": np.asarray(layout.names),
        "source_column": np.asarray([args.source]),
        "state_layout": np.asarray(
            [external.name if external else str(meta.get("state_layout", ""))]
        ),
        "retarget_mode": np.asarray([str(meta.get("retarget_mode", ""))]),
        "robot_from_table_pose7": (
            np.asarray([deployment], dtype=np.float32)
            if deployment is not None
            else np.empty((0, 7), dtype=np.float32)
        ),
    }


def _gripper_source(meta: dict[str, object]) -> str:
    representation = meta.get("gripper_representation")
    if isinstance(representation, dict):
        return str(representation.get("source", "unknown"))
    return "unknown"


def _deployment_pose7(meta: dict[str, object]) -> np.ndarray | None:
    """Load the exact table placement conversion used, if it is still here.

    The transform is only needed to draw the table frame and an optional task
    scene. Resolving a *different* placement would silently move the scene
    away from the one the joints were solved against, so a missing or changed
    file drops the scene instead of substituting one.
    """
    recorded = meta.get("deployment_calibration")
    if not isinstance(recorded, dict) or not recorded.get("path"):
        return None
    path = repo_path(str(recorded["path"]))
    if path is None or not path.is_file():
        print(
            "[replay-joints] note: table placement "
            f"{recorded['path']} is not available here; skipping table frame."
        )
        return None
    expected = str(recorded.get("sha256", ""))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and actual != expected:
        print(
            f"[replay-joints] warning: {path} changed since conversion "
            f"(sha256 {actual[:12]} != {expected[:12]}); skipping table frame."
        )
        return None
    calibration = load_deployment_calibration(
        path,
        expected_robot=str(recorded.get("robot")) or None,
        profile=str(recorded.get("profile", "explicit")),
    )
    print(
        "[replay-joints] table placement: "
        f"profile={calibration.profile} scope={calibration.scope} path={path}"
    )
    return np.asarray(calibration.pose7, dtype=np.float32)


# ---------------------------------------------------------------------------
# Verification against the TCP replay pipeline
# ---------------------------------------------------------------------------


def default_source_dataset(root: Path, robot: str) -> Path | None:
    """The sibling capture directory conversion derives its name from."""
    suffix = f"-{robot}-{JOINT_DATASET_SUFFIX}"
    name = Path(root).name
    if not name.endswith(suffix):
        return None
    candidate = Path(root).parent / name[: -len(suffix)]
    return candidate if (candidate / "meta" / "info.json").is_file() else None


def _ik_fidelity_entry(
    meta: dict[str, Any], source_index: int | None
) -> dict[str, Any] | None:
    reports = meta.get("ik_fidelity")
    if not isinstance(reports, list) or source_index is None:
        return None
    for report in reports:
        if (
            isinstance(report, dict)
            and int(report.get("source_episode_index", -1)) == source_index
        ):
            return report
    return None


def verify_episode(args: argparse.Namespace) -> bool:
    """Report whether the stored joints still match what replay produces.

    Two checks with deliberately different standards.

    Integrity is exact and offline: rebuilding the solved array from the
    overlapping state/action views must reproduce the conversion hash byte for
    byte. Nothing about it depends on re-running a solver.

    Solver agreement re-solves the capture and compares in task space. It is
    NOT a bit-exact check, because the IK solver is not reproducible run to
    run: two consecutive solves of the same episode with identical arguments
    on one machine differ by ~2e-3 rad. That is why conversion reuses the
    trajectory replay already solved instead of solving again, and why the
    honest question here is whether both land on the same motion, not on the
    same bits.
    """
    info = args.dataset_info
    meta = handumi_metadata(info)
    runtime = load_embodiment(args.robot)
    states, actions, _ = load_joint_episode(args)

    source_index = source_episode_index(args.dataset_root, args.episode)
    print(
        f"\n[verify] episode {args.episode} <- source episode "
        f"{'unknown' if source_index is None else source_index}"
    )

    ok = True
    print("[verify] --- integrity: stored joints vs conversion output ---")
    # The dataset splits one solved trajectory into overlapping views:
    # state[t] is the command at t and action[t] the command at t+1, so the
    # original array is the states plus the final action.
    joint_array = np.concatenate([states, actions[-1:]], axis=0).astype(np.float32)
    report = _ik_fidelity_entry(meta, source_index)
    if report is None:
        print("[verify] conversion recorded no ik_fidelity entry; skipping hash check.")
    else:
        expected_frames = int(report.get("frames", -1))
        if expected_frames != len(joint_array):
            ok = False
            print(
                f"[verify] FAIL frames: rebuilt {len(joint_array)} from "
                f"{len(states)} states, conversion solved {expected_frames}."
            )
        digest = hashlib.sha256(joint_array.tobytes()).hexdigest()
        expected = str(report.get("output_state_sha256", ""))
        match = digest == expected
        ok = ok and match
        print(
            f"[verify] {'OK  ' if match else 'FAIL'} stored joints sha256="
            f"{digest[:16]} expected={expected[:16]}"
        )

    source_root = args.source_dataset or default_source_dataset(
        args.dataset_root, args.robot
    )
    if source_root is None:
        print(
            "[verify] no source capture found; pass --source-dataset to also "
            "re-solve the episode through the TCP replay pipeline."
        )
        return ok
    if source_index is None:
        print("[verify] cannot map this episode to a capture episode; skipping re-solve.")
        return ok

    print("\n[verify] --- solver agreement: re-solved capture vs stored joints ---")
    print(f"[verify] re-solving {source_root} episode {source_index} through replay...")
    rollout = _solve_source_episode(args, meta, source_root, source_index)
    replay_qpos = np.asarray(rollout["qpos"], dtype=np.float32)
    if len(replay_qpos) != len(joint_array):
        print(
            f"[verify] FAIL frames: replay solved {len(replay_qpos)}, dataset "
            f"holds {len(joint_array)}."
        )
        return False

    expanded, _ = expand_canonical_trajectory(joint_array, runtime=runtime)
    solver = runtime.solver_cls()
    left = np.empty((len(expanded), 7), dtype=np.float32)
    right = np.empty((len(expanded), 7), dtype=np.float32)
    for index, q in enumerate(expanded):
        left[index], right[index] = solver.fk_pose7(q)
    errors = pose_error_arrays(
        np.asarray(rollout["achieved_left_pose7_robot_world"], dtype=np.float32),
        np.asarray(rollout["achieved_right_pose7_robot_world"], dtype=np.float32),
        left,
        right,
    )
    position_mm = 1000.0 * float(
        max(errors["left_pos_error_m"].max(), errors["right_pos_error_m"].max())
    )
    rotation_deg = float(
        max(errors["left_rot_error_deg"].max(), errors["right_rot_error_deg"].max())
    )
    within = (
        position_mm <= args.verify_position_tolerance_mm
        and rotation_deg <= args.verify_rotation_tolerance_deg
    )
    ok = ok and within
    print(
        f"[verify] {'OK  ' if within else 'FAIL'} TCP agreement: "
        f"pos max={position_mm:.4f}mm (tol {args.verify_position_tolerance_mm:g}) "
        f"rot max={rotation_deg:.4f}deg (tol {args.verify_rotation_tolerance_deg:g})"
    )

    layout = canonical_joint_layout(runtime)
    arm_indices = [
        index
        for index, side in zip(layout.indices, layout.gripper_sides, strict=True)
        if side is None and index is not None
    ]
    finger_indices = sorted(
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    )
    arm_diff = float(
        np.abs(expanded[:, arm_indices] - replay_qpos[:, arm_indices]).max()
    )
    print(f"[verify] .... arm joints: max |diff| = {arm_diff:.3e} rad")
    if finger_indices:
        finger_diff = float(
            np.abs(expanded[:, finger_indices] - replay_qpos[:, finger_indices]).max()
        )
        # Conversion stores a physical width, so the opening makes a round trip
        # through normalized * max_width in float32 before coming back.
        print(
            f"[verify] .... gripper joints: max |diff| = {finger_diff:.3e} "
            "(width round-trip)"
        )
    digest = hashlib.sha256(replay_qpos.tobytes()).hexdigest()
    if report is not None:
        expected = str(report.get("qpos_sha256", ""))
        print(
            f"[verify] .... re-solved qpos sha256={digest[:16]} "
            f"conversion recorded={expected[:16]} "
            f"({'equal' if digest == expected else 'differs, as expected'})"
        )
    args.verify_reference = {
        "left": np.asarray(
            rollout["achieved_left_pose7_robot_world"], dtype=np.float32
        ),
        "right": np.asarray(
            rollout["achieved_right_pose7_robot_world"], dtype=np.float32
        ),
    }
    return ok


def _solve_source_episode(
    args: argparse.Namespace,
    meta: dict[str, Any],
    source_root: Path,
    source_index: int,
) -> dict[str, np.ndarray]:
    """Run the TCP replay solver with the settings conversion recorded.

    Reading them back rather than taking the replay defaults is what makes
    this a comparison: a differing tolerance or orientation policy would
    produce a different trajectory and blame the dataset for it.
    """
    from handumi.scripts.replay.replay_in_sim import build_parser as build_replay_parser
    from handumi.scripts.replay.replay_in_sim import solve_episode

    replay_args = build_replay_parser().parse_args([str(source_root)])
    selection = resolve_dataset_selection(str(source_root), revision=args.revision)
    replay_args.repo_id = selection.repo_id
    replay_args.dataset_root = selection.root
    replay_args.revision = args.revision
    replay_args.episode = source_index
    replay_args.robot = args.robot
    replay_args.rig_config = args.rig_config
    replay_args.approach_seconds = 0.0
    replay_args.headless = True

    replay_args.source = str(meta.get("conversion_source", "observation.state"))
    replay_args.retarget_mode = str(meta.get("retarget_mode", "absolute-table"))
    replay_args.compose_source = str(meta.get("compose_source", "commanded"))
    replay_args.translation_scale = float(meta.get("translation_scale", 1.0))
    replay_args.absolute_orientation = str(
        meta.get("absolute_orientation", "relative-start")
    )
    replay_args.initial_solve_iterations = int(
        meta.get("initial_solve_iterations", 12)
    )
    replay_args.initial_position_tolerance_m = float(
        meta.get("initial_position_tolerance_m", 0.01)
    )
    replay_args.raw_controller_debug = bool(meta.get("raw_controller_debug", False))
    device = meta.get("controller_device")
    replay_args.controller_device = str(device) if device in ("pico", "meta") else None
    representation = meta.get("gripper_representation")
    if isinstance(representation, dict) and representation.get("max_width_m"):
        replay_args.gripper_max_width_m = float(representation["max_width_m"])

    recorded = meta.get("deployment_calibration")
    if isinstance(recorded, dict) and recorded.get("path"):
        path = repo_path(str(recorded["path"]))
        if path is not None and path.is_file():
            replay_args.deployment_calibration = path
    return solve_episode(replay_args)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_rollout(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> Path:
    output = args.output
    if output is None:
        output = (
            DEFAULT_OUT_DIR
            / f"episode_{args.episode:06d}_{args.robot}_from_joints.npz"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "repo_id": np.asarray([args.repo_id]),
        "robot": np.asarray([args.robot]),
        "episode": np.asarray([args.episode], dtype=np.int64),
        **rollout,
    }
    np.savez_compressed(output, **payload)
    print(f"[replay-joints] saved: {output}")
    return output


def _points(pose7: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple((float(x), float(y), float(z)) for x, y, z in np.asarray(pose7)[:, :3])


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

    reference = getattr(args, "verify_reference", None)
    marker_left = marker_right = None
    if not args.hide_trajectories:
        if reference is not None:
            # Drawn first and thicker, so the joint-level path rides on top of
            # the re-solved replay path instead of hiding it.
            server.scene.add_spline_catmull_rom(
                "/traj/replay_left",
                positions=_points(reference["left"]),
                color=(255, 190, 50),
                line_width=4.0,
            )
            server.scene.add_spline_catmull_rom(
                "/traj/replay_right",
                positions=_points(reference["right"]),
                color=(80, 220, 130),
                line_width=4.0,
            )
        server.scene.add_spline_catmull_rom(
            "/traj/tcp_left",
            positions=_points(rollout["fk_left_pose7_robot_world"]),
            color=(80, 160, 255),
            line_width=2.0,
        )
        server.scene.add_spline_catmull_rom(
            "/traj/tcp_right",
            positions=_points(rollout["fk_right_pose7_robot_world"]),
            color=(255, 90, 90),
            line_width=2.0,
        )
        marker_left = server.scene.add_icosphere(
            "/tcp/left", radius=0.014, color=(80, 160, 255)
        )
        marker_right = server.scene.add_icosphere(
            "/tcp/right", radius=0.014, color=(255, 90, 90)
        )

    approach = rollout["approach_qpos"]
    approach_frames = len(approach)
    total_frames = approach_frames + len(rollout["qpos"])
    play = server.gui.add_checkbox("Play", True)
    frame = server.gui.add_slider("Frame", 0, total_frames - 1, 1, 0)
    gripper_text = server.gui.add_text("Gripper opening", "-", disabled=True)

    def draw(index: int) -> None:
        if index < approach_frames:
            robot_view.update_cfg(approach[index])
            i = 0
        else:
            i = index - approach_frames
            robot_view.update_cfg(rollout["qpos"][i])
        if marker_left is not None and marker_right is not None:
            marker_left.position = tuple(rollout["fk_left_pose7_robot_world"][i, :3])
            marker_right.position = tuple(rollout["fk_right_pose7_robot_world"][i, :3])
        if index < approach_frames:
            gripper_text.value = "home -> start approach"
        else:
            opening = rollout["gripper_normalized"][i]
            gripper_text.value = f"L={opening[0] * 100:.0f}% R={opening[1] * 100:.0f}%"

    draw(0)
    print(f"[replay-joints] viewer: http://localhost:{server.get_port()}")
    current = 0
    fps = float(rollout["fps"][0])
    while True:
        if play.value:
            current = (current + 1) % total_frames
            frame.value = current
        else:
            current = int(frame.value)
        draw(current)
        time.sleep(1.0 / fps)


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
        selection = resolve_dataset_selection(args.dataset, revision=args.revision)
    except ValueError as exc:
        parser.error(str(exc))
    args.repo_id = selection.repo_id
    args.dataset_root = selection.root
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1.")
    if args.approach_seconds < 0.0:
        raise SystemExit("--approach-seconds must be >= 0.")
    if args.verify_position_tolerance_mm < 0.0:
        raise SystemExit("--verify-position-tolerance-mm must be >= 0.")
    if args.verify_rotation_tolerance_deg < 0.0:
        raise SystemExit("--verify-rotation-tolerance-deg must be >= 0.")

    args.dataset_info = dataset_info(selection.root)
    args.external_layout = resolve_state_layout(args, args.dataset_info)
    if args.external_layout is not None and args.verify:
        raise SystemExit(
            "--verify compares against the conversion hashes and the sibling "
            "HandUMI capture, which an external dataset "
            f"({args.external_layout.name}) does not have."
        )
    args.robot = resolved_robot(args, args.dataset_info)
    print(
        "Joint replay plan\n"
        f"  Dataset: {selection.root}\n"
        f"  Repository: {selection.repo_id}\n"
        f"  Episode: {args.episode}\n"
        f"  Robot profile: {args.robot}\n"
        f"  Joint column: {args.source}\n"
        f"  State layout: {args.external_layout.name if args.external_layout else 'handumi canonical'}"
    )
    if args.dry_run:
        return

    rollout = build_rollout(args)
    save_rollout(args, rollout)

    verified = True
    if args.verify:
        verified = verify_episode(args)
        print(f"[verify] result: {'PASS' if verified else 'FAIL'}")
    if not args.headless:
        show_viewer(args, rollout)
    if args.verify and not verified and args.strict_verify:
        raise SystemExit("Joint-level dataset does not match the replay solver.")


if __name__ == "__main__":
    main()
