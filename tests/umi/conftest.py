"""提供完全合成的 UMI 安装、AprilGrid 多视角和桌面会话夹具。"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.calibration.umi import digest, pose7
from handumi.calibration.umi_table import solve_table, verify_table


@pytest.fixture
def table_setup():
    """生成已知桌面变换和非平凡安装关系，不读取任何本机测量数据。"""
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler("xyz", [2, -5, 3], degrees=True).as_matrix()
    mount[:3, 3] = [0.04, 0, 0.02]
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    tcp[:3, 3] = [0.08, -0.03, 0.15]
    profile = {
        "schema": "umi_calibration_v1", "serial": "synthetic",
        "camera_device": "synthetic", "resolution": [64, 48],
        "fps": 30, "fourcc": "MJPG", "intrinsics": [50, 50, 32, 24],
        "distortion": [0] * 4, "tracker_camera": mount.tolist(),
        "tracker_tcp": tcp.tolist(),
        "gripper": {
            "min_marker_dist_mm": 40, "max_marker_dist_mm": 120,
            "marker_size_mm": 16, "dictionary_name": "DICT_4X4_50",
            "left_finger_tag_id": 0, "right_finger_tag_id": 1,
        },
        "tracker_clock": "host_query_midpoint", "camera_clock": "opencv_host_capture",
        "time": None,
    }
    profile["geometry_digest"] = digest({k: v for k, v in profile.items() if k != "time"})
    profile["time"] = {"accepted": True, "offset_s": 0.02, "geometry_digest": profile["geometry_digest"]}
    device_board = np.eye(4)
    device_board[:3, :3] = Rotation.from_euler("xyz", [170, 7, 22], degrees=True).as_matrix()
    device_board[:3, 3] = [0.05, 0.12, 1.2]
    points = np.array([
        [column * 0.0715 + x, row * 0.0715 + y, 0]
        for row in range(6) for column in range(6)
        for x, y in [(0, 0), (0.055, 0), (0.055, 0.055), (0, 0.055)]
    ])
    data = {
        "camera_times": np.linspace(0, 8, 40).tolist(), "tracker_poses": [],
        "observations": [], "geometry_digest": profile["geometry_digest"],
        "time_digest": digest(profile["time"]),
    }
    for index in range(40):
        tracker = np.eye(4)
        tracker[:3, 3] = [index * 0.006, 0.025 * np.sin(index / 5), 0]
        tracker[:3, :3] = Rotation.from_euler("y", index * 0.5, degrees=True).as_matrix()
        camera_board = np.linalg.inv(tracker @ mount) @ device_board
        pixels, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3), Rotation.from_matrix(camera_board[:3, :3]).as_rotvec(),
            camera_board[:3, 3], np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1]], float), np.zeros(4),
        )
        data["observations"].append([points.tolist(), pixels.reshape(-1, 2).tolist(), camera_board.tolist()])
        data["tracker_poses"].append(pose7(tracker).tolist())
    session = solve_table(data, profile, 0.012)
    session["verification"] = verify_table(data, profile, session)
    return profile, data, session, device_board
