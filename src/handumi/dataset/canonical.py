"""Canonical joint-vector layout shared by conversion and real teleop recording."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handumi.dataset.raw import LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX

SIDES = ("left", "right")

# ``handumi.state_layout`` values whose columns are this module's canonical
# vector. Older conversions wrote the Piper-specific name; the meaning is
# identical.
CANONICAL_STATE_LAYOUT = "yaml_arm_joints_plus_logical_gripper_width_m"
CANONICAL_STATE_LAYOUTS = frozenset(
    {CANONICAL_STATE_LAYOUT, "bipiper_6dof_plus_gripper_width_m_per_side"}
)


def is_canonical_state_layout(info: dict) -> bool:
    """True when a dataset's ``handumi.state_layout`` is the canonical vector."""
    handumi = info.get("handumi")
    if not isinstance(handumi, dict):
        return False
    return str(handumi.get("state_layout", "")) in CANONICAL_STATE_LAYOUTS


@dataclass(frozen=True)
class CanonicalJointLayout:
    """Robot command layout for IL datasets.

    The layout stores each non-gripper arm joint from the robot YAML plus one
    logical gripper opening column per side when the embodiment declares
    gripper joints. A parallel gripper with two URDF joints therefore becomes
    one dataset dimension.
    """

    names: list[str]
    indices: list[int | None]
    gripper_sides: list[str | None]

    @property
    def size(self) -> int:
        return len(self.names)


def canonical_joint_layout(runtime) -> CanonicalJointLayout:
    actuated_names = list(runtime.robot.joints.actuated_names)
    finger_indices = {
        side: {finger.index for finger in runtime.finger_joints.get(side, ())}
        for side in SIDES
    }
    names: list[str] = []
    indices: list[int | None] = []
    gripper_sides: list[str | None] = []

    for side in SIDES:
        for joint_name in runtime.arm_joint_names(side):
            joint_index = actuated_names.index(joint_name)
            if joint_index in finger_indices[side]:
                continue
            names.append(f"{joint_name}.pos")
            indices.append(joint_index)
            gripper_sides.append(None)
        if finger_indices[side]:
            names.append(f"{side}_gripper.width_m")
            indices.append(None)
            gripper_sides.append(side)

    return CanonicalJointLayout(names=names, indices=indices, gripper_sides=gripper_sides)


def raw_state_gripper_widths_m(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    return states[:, [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]]


def canonicalize_joint_trajectory(
    qpos: np.ndarray,
    *,
    runtime,
    gripper_widths_m: np.ndarray | None = None,
    gripper_normalized: np.ndarray | None = None,
    fallback_gripper: float = 1.0,
) -> np.ndarray:
    """Project full URDF qpos to the canonical IL command vector."""

    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2-D, got shape {qpos.shape}.")
    layout = canonical_joint_layout(runtime)
    widths_m = _resolve_widths_m(
        qpos,
        runtime=runtime,
        gripper_widths_m=gripper_widths_m,
        gripper_normalized=gripper_normalized,
        fallback_gripper=fallback_gripper,
    )

    out = np.empty((len(qpos), layout.size), dtype=np.float32)
    for column, (joint_index, side) in enumerate(
        zip(layout.indices, layout.gripper_sides, strict=True)
    ):
        if side is None:
            if joint_index is None:
                raise RuntimeError("Canonical joint column is missing its qpos index.")
            out[:, column] = qpos[:, joint_index]
        else:
            side_column = 0 if side == "left" else 1
            out[:, column] = widths_m[:, side_column]
    return out


def canonicalize_command(
    q: np.ndarray,
    *,
    runtime,
    openings: dict[str, float] | None = None,
) -> np.ndarray:
    qpos = np.asarray(q, dtype=np.float32).reshape(1, -1)
    normalized = None
    if openings is not None:
        normalized = np.asarray(
            [[openings.get("left", 0.0), openings.get("right", 0.0)]],
            dtype=np.float32,
        )
    return canonicalize_joint_trajectory(
        qpos,
        runtime=runtime,
        gripper_normalized=normalized,
    )[0]


def _resolve_widths_m(
    qpos: np.ndarray,
    *,
    runtime,
    gripper_widths_m: np.ndarray | None,
    gripper_normalized: np.ndarray | None,
    fallback_gripper: float,
) -> np.ndarray:
    if not any(runtime.finger_joints.values()):
        return np.zeros((len(qpos), 2), dtype=np.float32)
    max_width = np.float32(max(float(runtime.config.gripper_max_width_m), 1e-6))
    if gripper_widths_m is not None:
        widths = np.asarray(gripper_widths_m, dtype=np.float32)
        if widths.shape != (len(qpos), 2):
            raise ValueError(
                "gripper_widths_m must have shape "
                f"({len(qpos)}, 2), got {widths.shape}."
            )
        return np.clip(widths, 0.0, max_width)

    if gripper_normalized is not None:
        normalized = np.asarray(gripper_normalized, dtype=np.float32)
        if normalized.shape != (len(qpos), 2):
            raise ValueError(
                "gripper_normalized must have shape "
                f"({len(qpos)}, 2), got {normalized.shape}."
            )
        return np.clip(normalized, 0.0, 1.0) * max_width

    normalized = np.full((len(qpos), 2), float(fallback_gripper), dtype=np.float32)
    for side_index, side in enumerate(SIDES):
        fingers = runtime.finger_joints.get(side, ())
        fractions: list[np.ndarray] = []
        for finger in fingers:
            span = float(finger.open_value - finger.closed_value)
            if abs(span) < 1e-9:
                continue
            fractions.append((qpos[:, finger.index] - finger.closed_value) / span)
        if fractions:
            normalized[:, side_index] = np.mean(np.stack(fractions, axis=1), axis=1)
    return np.clip(normalized, 0.0, 1.0) * max_width


def expand_canonical_trajectory(
    canonical: np.ndarray,
    *,
    runtime,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild full URDF ``qpos`` from the canonical IL command vector.

    Inverse of :func:`canonicalize_joint_trajectory`. Arm columns are copied
    back to their actuated index and each logical gripper width is converted
    to the 0-1 opening the finger joints interpolate, exactly as
    :meth:`RobotRuntime.set_finger_positions` does during replay.

    Conversion only stores the columns the layout declares, so any actuated
    joint outside it (a torso the embodiment locks, say) is unrecoverable and
    keeps its home value here. Returns ``(qpos, gripper_normalized)``.
    """
    canonical = np.asarray(canonical, dtype=np.float32)
    if canonical.ndim != 2:
        raise ValueError(f"canonical states must be 2-D, got shape {canonical.shape}.")
    layout = canonical_joint_layout(runtime)
    if canonical.shape[1] != layout.size:
        raise ValueError(
            f"Expected {layout.size} canonical columns for {runtime.name} "
            f"({', '.join(layout.names)}), got {canonical.shape[1]}."
        )

    max_width = np.float32(max(float(runtime.config.gripper_max_width_m), 1e-6))
    qpos = np.tile(
        np.asarray(runtime.config.home_q, dtype=np.float32), (len(canonical), 1)
    )
    normalized = np.zeros((len(canonical), len(SIDES)), dtype=np.float32)
    for column, (joint_index, side) in enumerate(
        zip(layout.indices, layout.gripper_sides, strict=True)
    ):
        if side is None:
            if joint_index is None:
                raise RuntimeError("Canonical joint column is missing its qpos index.")
            qpos[:, joint_index] = canonical[:, column]
        else:
            normalized[:, SIDES.index(side)] = np.clip(
                canonical[:, column] / max_width, 0.0, 1.0
            )

    for side_index, side in enumerate(SIDES):
        for finger in (runtime.finger_joints or {}).get(side, ()):
            span = np.float32(finger.open_value - finger.closed_value)
            qpos[:, finger.index] = (
                np.float32(finger.closed_value) + normalized[:, side_index] * span
            )
    return qpos, normalized
