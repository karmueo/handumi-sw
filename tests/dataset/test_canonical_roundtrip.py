"""The canonical joint vector must survive a round trip back to URDF qpos.

``handumi replay-joints`` renders a converted dataset by inverting the
projection ``handumi convert`` applied, so anything the inverse loses shows up
as a robot that does not stand where the dataset says it does.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from handumi.dataset.canonical import (
    canonical_joint_layout,
    canonicalize_joint_trajectory,
    expand_canonical_trajectory,
)
from handumi.robots.registry import load_embodiment

EMBODIMENTS = ("piper", "openarmv1", "rm75b")


@pytest.fixture(scope="module", params=EMBODIMENTS)
def runtime(request):
    return load_embodiment(request.param)


def _arm_and_finger_indices(runtime) -> tuple[list[int], list[int]]:
    layout = canonical_joint_layout(runtime)
    arm = [
        index
        for index, side in zip(layout.indices, layout.gripper_sides, strict=True)
        if side is None and index is not None
    ]
    fingers = sorted(
        finger.index
        for side_fingers in (runtime.finger_joints or {}).values()
        for finger in side_fingers
    )
    return arm, fingers


def _sample(runtime, frames: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    nq = len(runtime.joint_names)
    qpos = np.tile(np.asarray(runtime.config.home_q, dtype=np.float32), (frames, 1))
    arm, _ = _arm_and_finger_indices(runtime)
    qpos[:, arm] += rng.uniform(-0.4, 0.4, size=(frames, len(arm))).astype(np.float32)
    assert qpos.shape[1] == nq
    normalized = rng.uniform(0.0, 1.0, size=(frames, 2)).astype(np.float32)
    # set_finger_positions is what replay writes, so start from its output.
    for frame in range(frames):
        runtime.set_finger_positions(
            qpos[frame], {"left": float(normalized[frame, 0]), "right": float(normalized[frame, 1])}
        )
    return qpos, normalized


def test_arm_joints_round_trip_exactly(runtime) -> None:
    qpos, normalized = _sample(runtime)
    canonical = canonicalize_joint_trajectory(
        qpos, runtime=runtime, gripper_normalized=normalized
    )
    restored, _ = expand_canonical_trajectory(canonical, runtime=runtime)
    arm, _ = _arm_and_finger_indices(runtime)
    # Arm columns are copied, never rescaled: any drift here is a layout bug.
    np.testing.assert_array_equal(restored[:, arm], qpos[:, arm])


def test_gripper_survives_the_width_round_trip(runtime) -> None:
    qpos, normalized = _sample(runtime)
    canonical = canonicalize_joint_trajectory(
        qpos, runtime=runtime, gripper_normalized=normalized
    )
    restored, restored_normalized = expand_canonical_trajectory(
        canonical, runtime=runtime
    )
    _, fingers = _arm_and_finger_indices(runtime)
    if not fingers:
        pytest.skip(f"{runtime.name} declares no gripper joints")
    if not any(side is not None for side in canonical_joint_layout(runtime).gripper_sides):
        pytest.skip(f"{runtime.name} keeps gripper opening outside the canonical vector")
    # Conversion stores a physical width, so the opening passes through
    # normalized * max_width and back in float32 rather than being copied.
    for side_index, side in enumerate(("left", "right")):
        if not runtime.finger_joints.get(side):
            continue
        np.testing.assert_allclose(
            restored_normalized[:, side_index], normalized[:, side_index], atol=1e-6
        )
    np.testing.assert_allclose(restored[:, fingers], qpos[:, fingers], atol=1e-7)


def test_independent_opening_override_restores_fingers(runtime) -> None:
    """独立开度列覆盖规范列默认值时，展开结果同步更新双指。"""
    qpos, normalized = _sample(runtime)
    canonical = canonicalize_joint_trajectory(
        qpos, runtime=runtime, gripper_normalized=normalized
    )
    override = np.clip(normalized[::-1] * 1.3 - 0.1, 0.0, 1.0)
    restored, restored_normalized = expand_canonical_trajectory(
        canonical, runtime=runtime, gripper_normalized=override
    )
    _, fingers = _arm_and_finger_indices(runtime)
    np.testing.assert_allclose(restored_normalized, override, atol=1e-7)
    if not fingers:
        pytest.skip(f"{runtime.name} declares no gripper joints")
    for side_index, side in enumerate(("left", "right")):
        for finger in runtime.finger_joints.get(side, ()):
            expected = finger.closed_value + override[:, side_index] * (
                finger.open_value - finger.closed_value
            )
            np.testing.assert_allclose(
                restored[:, finger.index], expected, atol=1e-7
            )
    default, default_normalized = expand_canonical_trajectory(
        canonical, runtime=runtime
    )
    layout = canonical_joint_layout(runtime)
    if any(side is not None for side in layout.gripper_sides):
        np.testing.assert_allclose(default[:, fingers], qpos[:, fingers], atol=1e-7)
    else:
        # 没有宽度列时，未提供独立开度就沿用 home 的真实夹指姿态。
        np.testing.assert_allclose(
            default[:, fingers],
            np.tile(runtime.home_q()[fingers], (len(default), 1)),
            atol=1e-7,
        )
        for side_index, side in enumerate(("left", "right")):
            if runtime.finger_joints.get(side):
                np.testing.assert_allclose(default_normalized[:, side_index], 1.0)


def test_rm75b_keeps_seven_axis_layout_and_uses_independent_openings() -> None:
    """RM75B 规范列保持七轴，独立 normalized 列负责驱动模型双指。"""
    runtime = load_embodiment("rm75b")
    layout = canonical_joint_layout(runtime)
    assert layout.size == 7
    assert layout.names == [f"joint{index}.pos" for index in range(1, 8)]
    assert all(side is None for side in layout.gripper_sides)
    old = np.tile(runtime.home_q()[:7], (3, 1)).astype(np.float32)
    qpos, normalized = expand_canonical_trajectory(old, runtime=runtime)
    np.testing.assert_array_equal(qpos[:, :7], old)
    np.testing.assert_allclose(
        qpos[:, 7:], np.tile(runtime.home_q()[7:], (3, 1)), atol=1e-7
    )
    np.testing.assert_allclose(normalized[:, 1], 1.0, atol=1e-7)

    openings = np.asarray([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
    restored, restored_normalized = expand_canonical_trajectory(
        old, runtime=runtime, gripper_normalized=openings
    )
    np.testing.assert_allclose(restored_normalized, openings, atol=1e-7)
    expected_travel = 0.018134 * (1.0 - openings[:, 1])
    np.testing.assert_allclose(
        restored[:, 7], expected_travel, atol=1e-7
    )
    np.testing.assert_allclose(
        restored[:, 8], expected_travel, atol=1e-7
    )
    canonical = canonicalize_joint_trajectory(
        restored, runtime=runtime, gripper_normalized=openings
    )
    assert canonical.shape == (3, 7)
    np.testing.assert_array_equal(canonical, old)


def test_no_width_layout_preserves_non_open_home() -> None:
    """无宽度列时保留 home 的真实夹指位置，并如实报告其开度。"""
    runtime = load_embodiment("rm75b")
    home = runtime.config.home_q.copy()
    home[7:] = [0.006, 0.003]
    runtime = replace(
        runtime,
        config=replace(
            runtime.config,
            home_q=home,
            home_poses={runtime.config.default_home_pose: home},
        ),
    )
    canonical = np.tile(home[:7], (2, 1)).astype(np.float32)
    qpos, normalized = expand_canonical_trajectory(canonical, runtime=runtime)
    np.testing.assert_allclose(
        qpos[:, 7:], np.tile(home[7:], (2, 1)), atol=1e-7
    )
    fractions = [
        (home[index] - 0.018134) / (0.0 - 0.018134) for index in (7, 8)
    ]
    np.testing.assert_allclose(normalized[:, 1], np.mean(fractions), atol=1e-7)


def test_expansion_rejects_a_foreign_layout(runtime) -> None:
    layout = canonical_joint_layout(runtime)
    wrong = np.zeros((4, layout.size + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="canonical columns"):
        expand_canonical_trajectory(wrong, runtime=runtime)
