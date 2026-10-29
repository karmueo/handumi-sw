"""使用已知物理延迟的合成观测验证时间标定及退化拒绝。"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from handumi.calibration.umi_time import solve_time


def fixture_data(offset=0.025, stationary=False, uniform=False):
    """生成固定板和变速运动的原始鱼眼投影。"""
    stamps = np.arange(0, 7, 0.01)
    xyz = np.column_stack(
        [
            0.12 * np.sin(stamps * 1.7),
            0.08 * np.cos(stamps * 2.1),
            0.03 * np.sin(stamps),
        ]
    )
    rotvec = np.column_stack(
        [0.1 * np.sin(stamps * 2.5), 0.12 * np.cos(stamps * 1.2), 0.1 * np.sin(stamps)]
    )
    if stationary:
        xyz[:] = 0
        rotvec[:] = 0
    if uniform:
        xyz[:] = 0
        xyz[:, 0] = stamps * 0.01
        rotvec[:] = 0
    rotations = Rotation.from_rotvec(rotvec)
    poses = np.column_stack([xyz, rotations.as_quat()])
    slerp = Slerp(stamps, rotations)
    camera_times = np.linspace(0.3, 6.5, 40)
    matrix = np.array([[600.0, 0, 960], [0, 600.0, 540], [0, 0, 1]])
    points = np.array(
        [
            [0, 0, 0],
            [0.2, 0, 0],
            [0.2, 0.2, 0],
            [0, 0.2, 0],
            [0.1, 0.1, 0],
            [0.15, 0.1, 0],
        ]
    )
    board = np.eye(4)
    board[2, 3] = 1.2
    observations = []
    for stamp in camera_times:
        tracker = np.eye(4)
        tracker[:3, :3] = slerp(stamp + offset).as_matrix()
        tracker[:3, 3] = [
            np.interp(stamp + offset, stamps, xyz[:, j]) for j in range(3)
        ]
        camera_board = np.linalg.inv(tracker) @ board
        pixels, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3),
            Rotation.from_matrix(camera_board[:3, :3]).as_rotvec(),
            camera_board[:3, 3],
            matrix,
            np.zeros(4),
        )
        observations.append((points, pixels.reshape(-1, 2), camera_board))
    snapshot = {
        "tracker_camera": np.eye(4).tolist(),
        "intrinsics": [600, 600, 960, 540],
        "distortion": [0] * 4,
        "geometry_digest": "test",
    }
    return (
        camera_times,
        observations,
        stamps,
        poses,
        np.ones(len(stamps), bool),
        snapshot,
    )


@pytest.mark.parametrize("offset", [0.025, -0.019])
def test_signed_offset_recovery(offset):
    """正负偏移均恢复且固定安装外参保持不变。"""
    data = fixture_data(offset)
    result = solve_time(*data)
    assert result["offset_s"] == pytest.approx(offset, abs=0.0002)
    assert result["accepted"]
    assert data[-1]["tracker_camera"] == np.eye(4).tolist()


def test_stationary_rejected():
    """静止观测不能作为通过验收的延迟标定。"""
    with pytest.raises(ValueError, match="辨识|边界"):
        solve_time(*fixture_data(stationary=True))


def test_stationary_tracker_noise_rejected():
    """毫米以下的跟踪噪声不能被误判为可辨识的采集运动。"""
    data = list(fixture_data(stationary=True))
    rng = np.random.default_rng(42)
    data[3][:, :3] = rng.normal(0, 0.00005, data[3][:, :3].shape)
    data[3][:, 3:] = Rotation.from_rotvec(
        rng.normal(0, 0.0001, data[3][:, :3].shape)
    ).as_quat()
    with pytest.raises(ValueError, match="运动幅度不足.*辨识"):
        solve_time(*data)


def test_missing_tracking_rejected():
    """失效跟踪状态不能跨缺口插值。"""
    data = list(fixture_data())
    data[4][:] = False
    with pytest.raises(ValueError, match="30 帧"):
        solve_time(*data)


def test_missing_board_observations_report_board_problem():
    """标定板帧数不足时提示检查标定板，而不是误报 Tracker 故障。"""
    data = list(fixture_data())
    data[0] = data[0][:0]
    data[1] = []
    with pytest.raises(ValueError, match="标定板观测仅 0 帧.*至少需要 30 帧"):
        solve_time(*data)


def test_uniform_motion_rejected():
    """恒速运动可被板平移吸收，必须拒绝没有约束的延迟。"""
    with pytest.raises(ValueError, match="辨识|边界"):
        solve_time(*fixture_data(uniform=True))


def test_search_boundary_rejected():
    """搜索边界上的最优解不能被标记为有效标定。"""
    with pytest.raises(ValueError, match="边界"):
        solve_time(*fixture_data(offset=0.1))


def test_wrong_mount_geometry_rejected():
    """固定外参存在明显误差时应拒绝，而非联合调整外参掩盖问题。"""
    data = list(fixture_data())
    data[-1]["tracker_camera"][0][3] = 0.1
    with pytest.raises(ValueError, match="失败|边界"):
        solve_time(*data)
