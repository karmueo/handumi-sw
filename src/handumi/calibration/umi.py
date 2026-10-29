"""导入 UMI 几何标定，验证变换方向并绑定原生采集链路。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from handumi.grippers.aruco import GripperOpennessEstimator


def transform(value):
    """检查米制 SE(3) 齐次矩阵，拒绝非有限和非刚体变换。"""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("外参必须为有限的 4×4 矩阵")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("外参齐次行无效")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("外参旋转必须属于 SO(3)")
    return matrix


def digest(value):
    """计算标定内容摘要，用于防止时间结果与几何配置混用。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def pose7(matrix):
    """将刚体变换转换为米制位置及 xyzw 四元数。"""
    matrix = transform(matrix)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def import_calibration(camera_path, tcp_path, gripper_path, *, serial, camera_device):
    """读取参考标定并形成独立快照；旧时间偏移不会进入新链路。"""
    binding_path = Path(camera_path).with_suffix(".binding.yaml")
    if binding_path.exists():
        binding = yaml.safe_load(binding_path.read_text())
        actual_hash = hashlib.sha256(Path(camera_path).read_bytes()).hexdigest()
        if (
            binding.get("tracker_serial") != serial
            or binding.get("calibration_sha256") != actual_hash
        ):
            raise ValueError("Tracker 序列号或外参摘要与现有绑定文件不一致")
    camera_doc = yaml.safe_load(Path(camera_path).read_text())
    camera = camera_doc.get("cam0", camera_doc.get("rgb"))
    if not camera or camera.get("distortion_model") not in ("fisheye", "equidistant"):
        raise ValueError("需要鱼眼相机标定")
    gripper = yaml.safe_load(Path(gripper_path).read_text())["gripper_calibration"]
    if list(camera["resolution"]) != list(gripper["resolution"]):
        raise ValueError("相机与夹爪标定分辨率不一致")
    camera_tracker = transform(camera["T_cam_hand"])
    camera_tcp = transform(yaml.safe_load(Path(tcp_path).read_text())["T_u_cam_tcp"])
    snapshot = {
        "schema": "umi_calibration_v1",
        "serial": str(serial),
        "camera_device": str(camera_device),
        "resolution": list(camera["resolution"]),
        "fps": 30,
        "fourcc": "MJPG",
        "intrinsics": camera["intrinsics"],
        "distortion": camera["distortion_coeffs"],
        "tracker_camera": np.linalg.inv(camera_tracker).tolist(),
        "tracker_tcp": (np.linalg.inv(camera_tracker) @ camera_tcp).tolist(),
        "gripper": gripper,
        "tracker_clock": "host_query_midpoint",
        "camera_clock": "opencv_host_capture",
        "time": None,
    }
    if not serial or not camera_device:
        raise ValueError("需要明确的 Tracker 序列号与相机路径")
    estimator(snapshot)
    snapshot["geometry_digest"] = digest(
        {k: v for k, v in snapshot.items() if k != "time"}
    )
    return snapshot


def estimator(snapshot):
    """由快照构造纯视觉估计器，ROI 按标定边界外扩 50 像素。"""
    fx, fy, cx, cy = snapshot["intrinsics"]
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    width, height = snapshot["resolution"]
    gripper = snapshot["gripper"]
    crop = gripper.get("crop_reference")
    roi = (
        (0, 0, 1, 1)
        if crop is None
        else (
            max(0, crop["left_x"] - 50) / width,
            max(0, crop["top_y"] - 50) / height,
            min(width, crop["right_x"] + 50) / width,
            min(height, crop["bottom_y"] + 50) / height,
        )
    )
    return GripperOpennessEstimator(
        matrix,
        snapshot["distortion"],
        gripper["min_marker_dist_mm"],
        gripper["max_marker_dist_mm"],
        marker_size_mm=gripper["marker_size_mm"],
        dictionary_name=gripper["dictionary_name"],
        left_marker_id=gripper["left_finger_tag_id"],
        right_marker_id=gripper["right_finger_tag_id"],
        roi_ratios=roi,
        image_resolution=(width, height),
    )


def verify(snapshot, *, require_time=True):
    """核对快照及时间结果的绑定；无时间标定时禁止正式采集。"""
    if snapshot.get("schema") != "umi_calibration_v1":
        raise ValueError("不支持的 UMI 标定版本")
    expected = digest(
        {k: v for k, v in snapshot.items() if k not in ("time", "geometry_digest")}
    )
    if snapshot.get("geometry_digest") != expected:
        raise ValueError("几何或设备配置已变化，需要重新导入和时间标定")
    transform(snapshot["tracker_camera"])
    transform(snapshot["tracker_tcp"])
    estimator(snapshot)
    if require_time:
        timing = snapshot.get("time") or {}
        if not timing.get("accepted") or timing.get("geometry_digest") != expected:
            raise ValueError("缺少与当前配置绑定且通过验收的时间标定")
        if not np.isfinite(timing.get("offset_s", float("nan"))):
            raise ValueError("时间偏移无效")
    return snapshot
