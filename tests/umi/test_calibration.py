"""验证安装标定导入、设备绑定和时间结果的失效规则。"""

import hashlib

import numpy as np
import pytest
import yaml

from handumi.calibration.umi import import_calibration, verify


def test_import_geometry_and_ignore_legacy_offset(tmp_path):
    """变换仅组合一次，旧 V4L2 偏移不进入新的原生标定。"""
    camera_tracker = np.eye(4)
    camera_tracker[0, 3] = 0.1
    camera_tcp = np.eye(4)
    camera_tcp[1, 3] = 0.2
    camera = tmp_path / "camera.yaml"
    camera.write_text(
        yaml.safe_dump(
            {
                "cam0": {
                    "T_cam_hand": camera_tracker.tolist(),
                    "intrinsics": [500, 500, 320, 240],
                    "distortion_coeffs": [0] * 4,
                    "resolution": [640, 480],
                    "distortion_model": "fisheye",
                },
                "time_offset": 0.081,
            }
        )
    )
    tcp = tmp_path / "tcp.yaml"
    tcp.write_text(yaml.safe_dump({"T_u_cam_tcp": camera_tcp.tolist()}))
    gripper = tmp_path / "gripper.yaml"
    gripper.write_text(
        yaml.safe_dump(
            {
                "gripper_calibration": {
                    "resolution": [640, 480],
                    "min_marker_dist_mm": 40,
                    "max_marker_dist_mm": 120,
                    "marker_size_mm": 16,
                    "dictionary_name": "DICT_4X4_50",
                    "left_finger_tag_id": 0,
                    "right_finger_tag_id": 1,
                }
            }
        )
    )
    binding = camera.with_suffix(".binding.yaml")
    binding.write_text(
        yaml.safe_dump(
            {
                "tracker_serial": "test",
                "calibration_sha256": hashlib.sha256(camera.read_bytes()).hexdigest(),
            }
        )
    )
    result = import_calibration(
        camera, tcp, gripper, serial="test", camera_device="camera"
    )
    np.testing.assert_allclose(np.array(result["tracker_tcp"])[:3, 3], [-0.1, 0.2, 0])
    assert result["time"] is None
    verify(result, require_time=False)
    with pytest.raises(ValueError, match="时间标定"):
        verify(result)
    with pytest.raises(ValueError, match="绑定"):
        import_calibration(camera, tcp, gripper, serial="wrong", camera_device="camera")
    result["time"] = {
        "accepted": True,
        "offset_s": -0.02,
        "geometry_digest": result["geometry_digest"],
    }
    verify(result)
    result["fps"] = 60
    with pytest.raises(ValueError, match="配置已变化"):
        verify(result)
