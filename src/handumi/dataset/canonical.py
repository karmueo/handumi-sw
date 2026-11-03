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
    """IL 数据集使用的机器人命令布局。

    布局先写入机器人 YAML 声明的非夹指臂关节；当配置声明夹指且
    ``canonical_gripper_width_column`` 为真时，再为每侧追加一条逻辑夹爪
    宽度列。因此带两个 URDF 夹指关节的平行夹爪只占一个数据维度。
    RM75B 关闭该宽度列以保持既有七轴契约，实际开度改由独立的
    normalized 辅助列携带。
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
        if finger_indices[side] and getattr(
            runtime.config, "canonical_gripper_width_column", True
        ):
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
    gripper_normalized: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """由规范 IL 命令向量重建完整 URDF ``qpos``。

    这是 :func:`canonicalize_joint_trajectory` 的逆操作：臂关节按列写回
    实际关节索引；存在逻辑宽度列时，按与回放相同的插值写入夹指。
    ``gripper_normalized`` 用于覆盖规范列中的默认开度，使带独立开度列的
    数据集按录制值显示夹指。没有宽度列且没有独立开度时，保留
    ``home_q`` 中的实际夹指位置，并据此报告归一化开度。

    转换只保存布局声明的列，因此布局之外的关节（例如被锁定的躯干）无法
    恢复，保持 home 值。返回 ``(qpos, gripper_normalized)``。
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

    # 没有宽度列时保留 home 的夹指值，并用它报告该侧默认开度。
    preserved_sides: set[str] = set()
    for side_index, side in enumerate(SIDES):
        if not runtime.finger_joints.get(side):
            continue
        if any(layout_side == side for layout_side in layout.gripper_sides):
            continue
        fractions = []
        for finger in runtime.finger_joints[side]:
            span = float(finger.open_value - finger.closed_value)
            if abs(span) < 1e-9:
                continue
            fractions.append(
                (qpos[:, finger.index] - finger.closed_value) / span
            )
        if fractions:
            normalized[:, side_index] = np.clip(
                np.mean(np.stack(fractions, axis=1), axis=1), 0.0, 1.0
            )
        preserved_sides.add(side)

    if gripper_normalized is not None:
        override = np.asarray(gripper_normalized, dtype=np.float32)
        if override.shape != normalized.shape:
            raise ValueError(
                "gripper_normalized must have shape "
                f"{normalized.shape}, got {override.shape}."
            )
        normalized = np.clip(override, 0.0, 1.0)
        preserved_sides.clear()

    for side_index, side in enumerate(SIDES):
        if side in preserved_sides:
            continue
        for finger in (runtime.finger_joints or {}).get(side, ()):
            span = np.float32(finger.open_value - finger.closed_value)
            qpos[:, finger.index] = (
                np.float32(finger.closed_value) + normalized[:, side_index] * span
            )
    return qpos, normalized
