"""验证回放设备识别及 Vive 相对轨迹的坐标轴和姿态映射。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.robots.utils import pose7_to_mat
from handumi.scripts.replay import replay_in_sim as replay


@pytest.mark.parametrize("device", ["pico", "meta", "vive"])
def test_replay_uses_recorded_device(device):
    """所有支持的设备均应保留录制身份，防止错误套用 PICO 轴映射。"""
    args = replay.build_parser().parse_args(["unused"])
    info = {"handumi": {"recording_device": device}}
    assert replay._resolved_controller_device(args, info) == device


def test_legacy_device_default_and_explicit_override():
    """旧数据保留默认设备，显式参数仍可覆盖未绑定身份的元数据。"""
    args = replay.build_parser().parse_args(["unused"])
    assert replay._resolved_controller_device(args, {}) == "pico"
    args.controller_device = "vive"
    assert (
        replay._resolved_controller_device(
            args, {"handumi": {"recording_device": "pico"}}
        )
        == "vive"
    )


def test_vive_local_relative_preserves_world_motion(monkeypatch):
    """真实 RM75B 求解应保持桌面位移和旋转，自动设备与显式 Vive 一致。"""
    # 合成右侧轨迹包含水平、竖直移动和多轴转动，不依赖私人数据与硬件。
    frames = 40
    progress = np.linspace(0.0, 1.0, frames)
    states = np.zeros((frames, 16), dtype=np.float32)
    states[:, 6] = 1.0
    states[:, 7:10] = np.array([-0.2, 0.1, 0.15]) + progress[:, None] * [
        -0.04,
        0.10,
        0.02,
    ]
    states[:, 10:14] = (
        Rotation.from_rotvec(progress[:, None] * [0.04, -0.03, 0.05])
        * Rotation.from_euler("xyz", [0.2, -0.1, 0.3])
    ).as_quat()
    info = {
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "tracking_workspace": "table",
        }
    }
    monkeypatch.setattr(
        replay, "load_episode_states", lambda args: (states, 30.0, info, None)
    )
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "rm75b",
            "--retarget-mode",
            "local-relative",
            "--raw-controller-debug",
            "--strict-ik",
        ]
    )
    automatic = replay.solve_episode(args)
    targets = automatic["target_right_pose7_robot_world"]
    np.testing.assert_allclose(
        targets[:, :3] - targets[0, :3],
        states[:, 7:10] - states[0, 7:10],
        atol=2e-6,
    )
    # 相对模式以机器人 home 为起点，世界系旋转增量保持一致。
    source_rotations = pose7_to_mat(states[:, 7:14])[:, :3, :3]
    target_rotations = pose7_to_mat(targets)[:, :3, :3]
    np.testing.assert_allclose(
        target_rotations @ target_rotations[0].T,
        source_rotations @ source_rotations[0].T,
        atol=2e-6,
    )
    args.controller_device = "vive"
    explicit = replay.solve_episode(args)
    # 设备解析应生成相同目标；独立 IK 求解允许数值漂移，分别验收末端误差。
    np.testing.assert_array_equal(
        automatic["target_right_pose7_robot_world"],
        explicit["target_right_pose7_robot_world"],
    )
    for rollout in (automatic, explicit):
        assert rollout["controller_device"][0] == "vive"
        assert np.isfinite(rollout["qpos"]).all()
        assert rollout["right_pos_error_m"].max() <= 0.001
        assert rollout["right_rot_error_deg"].max() <= 0.1
