"""验证关节级回放使用独立开度列恢复 RM75B 双指位置。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from handumi.dataset.canonical import expand_canonical_trajectory
from handumi.robots.registry import load_embodiment
from handumi.scripts.replay import replay_in_sim_joints as joints


def test_rm75b_conversion_output_keeps_seven_axis_columns():
    """转换输出仍为七轴，独立开度列不进入 observation.state。"""
    from handumi.scripts import conversion

    runtime = load_embodiment("rm75b")
    names = conversion._output_joint_names(SimpleNamespace(), runtime)
    assert names == [f"joint{index}.pos" for index in range(1, 8)]
    qpos = np.tile(runtime.home_q(), (2, 1))
    qpos[:, 7:] = [[0.0, 0.0], [0.018134, 0.018134]]
    canonical = conversion._canonical_command_states_from_rollout(
        {
            "qpos": qpos,
            "gripper_normalized": np.asarray([[1.0, 1.0], [0.0, 0.0]]),
        },
        runtime=runtime,
    )
    assert canonical.shape == (2, 7)
    np.testing.assert_array_equal(canonical, np.tile(runtime.home_q()[:7], (2, 1)))


def test_verification_openings_use_last_action_and_drive_fingers():
    """验证重建用 state+最后 action，且首尾开度差异体现在显示姿态。"""
    runtime = load_embodiment("rm75b")
    observation = np.asarray(
        [[0.1, 0.2], [0.3, 0.4]], dtype=np.float32
    )
    action = np.asarray([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    expected = np.asarray(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
    )
    canonical = np.tile(runtime.home_q()[:7], (3, 1)).astype(np.float32)
    for source in ("observation.state", "action"):
        args = SimpleNamespace(
            source=source,
            observation_openings=observation,
            action_openings=action,
        )
        full = joints._full_episode_openings(args)
        np.testing.assert_allclose(full, expected, atol=1e-7)
        qpos, normalized = expand_canonical_trajectory(
            canonical, runtime=runtime, gripper_normalized=full
        )
        np.testing.assert_allclose(normalized, expected, atol=1e-7)
        expected_travel = 0.018134 * (1.0 - expected[:, 1])
        np.testing.assert_allclose(qpos[:, 7], expected_travel, atol=1e-7)
        np.testing.assert_allclose(qpos[:, 8], expected_travel, atol=1e-7)
        assert not np.allclose(qpos[0, 7], qpos[-1, 7])

    assert joints._full_episode_openings(
        SimpleNamespace(
            source="observation.state",
            observation_openings=observation,
            action_openings=None,
        )
    ) is None


def test_recorded_openings_drive_rm75b_fingers(monkeypatch, tmp_path):
    """旧七列规范数据与独立开度列并存时，以独立开度列为准。"""
    runtime = load_embodiment("rm75b")
    arm = runtime.home_q()[:7]
    states = np.tile(arm, (3, 1)).astype(np.float32)
    actions = states.copy()
    openings = np.asarray([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(
        joints,
        "load_joint_episode",
        lambda args: (states, actions, 30.0),
    )
    args = SimpleNamespace(
        dataset_info={},
        external_layout=None,
        recorded_openings=openings,
        robot="rm75b",
        source="observation.state",
        start_frame=0,
        stride=1,
        max_frames=None,
        approach_seconds=0.0,
        dataset_root=tmp_path,
        episode=0,
        revision="main",
        repo_id="local/test",
    )
    rollout = joints.build_rollout(args)
    assert rollout["canonical_state"].shape == (3, 7)
    np.testing.assert_allclose(rollout["gripper_normalized"], openings)
    expected = np.asarray([0.0, 0.009067, 0.018134], dtype=np.float32)
    np.testing.assert_allclose(rollout["qpos"][:, 7], expected, atol=1e-7)
    np.testing.assert_allclose(rollout["qpos"][:, 8], expected, atol=1e-7)
    np.testing.assert_allclose(rollout["qpos"][:, :7], np.tile(arm, (3, 1)))

    # 旧七列数据没有独立开度列时仍可加载，夹指保持 home 的张开姿态。
    args.recorded_openings = None
    legacy = joints.build_rollout(args)
    assert legacy["canonical_state"].shape == (3, 7)
    np.testing.assert_allclose(legacy["qpos"][:, :7], np.tile(arm, (3, 1)))
    np.testing.assert_allclose(legacy["qpos"][:, 7:], 0.0)
    np.testing.assert_allclose(legacy["gripper_normalized"][:, 1], 1.0)
