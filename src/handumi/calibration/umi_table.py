"""求解和验证 Vive UMI 的 AprilGrid 桌面坐标系及不可变会话身份。"""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handumi.calibration.spatial import mean_transform
from handumi.calibration.umi import digest, transform, verify
from handumi.robots.utils import pose7_to_mat

BOARD_SPEC = {  # 与现有时间标定共用的实体板规格。
    "family": "tag36h11", "columns": 6, "rows": 6,
    "tag_size_m": 0.055, "spacing_ratio": 0.3,
}
FRAME_CONVENTION = "center_on_table_x_right_y_away_z_up_ids_0_5_near"
LIMITS = {  # 与桌面独立验证共用的验收界限。
    "translation_rms_m": 0.008, "rotation_rms_deg": 1.0,
    "median_px": 3.0, "p95_px": 10.0,
}


class TableQualityError(ValueError):
    """携带全部验收指标，便于失败时保存诊断而不重新求解。"""

    def __init__(self, message, metrics):
        """保存错误说明和指标副本，不改变验收结论。"""
        super().__init__(message)
        self.metrics = dict(metrics)  # 本次求解或独立验证的完整指标。


def board_from_table(height_m):
    """将桌面原点置于标签区域中心下方；板高必须显式为非负有限米数。"""
    if not np.isfinite(height_m) or height_m < 0:
        raise ValueError("板面距真实桌面的高度必须为非负有限数，单位米")
    result = np.eye(4)
    result[:3, 3] = [0.20625, 0.20625, -float(height_m)]
    return result


def session_core(session):
    """返回稳定会话身份；重复验证的报告和来源文件路径不参与续录比较。"""
    return {
        key: value for key, value in session.items()
        if key not in {"verification", "core_digest", "path"}
    }


def validate_session(session, snapshot, *, require_verified=True):
    """校验桌面会话、几何/时间绑定和独立验证状态；失败抛出 ValueError。"""
    verify(snapshot)
    if not isinstance(session, dict) or session.get("schema") != "umi_table_session_v1":
        raise ValueError("缺少有效的 Vive 桌面标定，请运行 calibrate umi table")
    if (
        session.get("tracking_device") != "vive"
        or session.get("tracking_origin") != "TrackingUniverseRawAndUncalibrated"
        or session.get("serial") != snapshot["serial"]
        or session.get("geometry_digest") != snapshot["geometry_digest"]
        or session.get("time_digest") != digest(snapshot["time"])
    ):
        raise ValueError("桌面标定与当前设备、几何或时间标定不一致，需要重新标定")
    if (
        not session.get("session_id")
        or session.get("board") != BOARD_SPEC
        or session.get("frame_convention") != FRAME_CONVENTION
        or session.get("core_digest") != digest(session_core(session))
    ):
        raise ValueError("桌面标定身份、规格或内容摘要无效")
    board_from_table(session["board_height_m"])
    transform(session["table_from_device"])
    if require_verified:
        report = session.get("verification") or {}
        if report.get("accepted") is not True or report.get("core_digest") != session["core_digest"]:
            raise ValueError("桌面标定尚未通过独立验证，请运行 calibrate umi verify-table")
        check_metrics(report.get("metrics", {}))
    return session


def _observations(data, snapshot):
    """检查同步观测、板几何与多视角覆盖，返回原始设备相机和板位姿。"""
    if data.get("geometry_digest") != snapshot["geometry_digest"]:
        raise ValueError("桌面观测与几何标定不一致")
    if data.get("time_digest") != digest(snapshot["time"]):
        raise ValueError("桌面观测与时间标定不一致")
    poses = np.asarray(data["tracker_poses"], float)
    observations = data["observations"]
    times = np.asarray(data["camera_times"], float)
    if len(observations) < 30:
        raise ValueError("桌面标定至少需要 30 帧有效 AprilGrid 观测")
    if (
        poses.shape != (len(observations), 7)
        or times.shape != (len(observations),)
        or not np.isfinite(poses).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-4)
    ):
        raise ValueError("桌面观测的时间线或 Tracker 位姿无效")
    rotations = Rotation.from_quat(poses[:, 3:])
    selected = []  # 选中的姿态必须与此前每个姿态都具有足够基线。
    for index, pose in enumerate(poses):
        if all(
            np.linalg.norm(pose[:3] - poses[other, :3]) >= 0.05
            or (rotations[other].inv() * rotations[index]).magnitude() >= np.deg2rad(10)
            for other in selected
        ):
            selected.append(index)
    if len(selected) < 4:
        raise ValueError("至少需要 4 个不同视角，两两相距至少 5 cm 或旋转至少 10°")
    mount = transform(snapshot["tracker_camera"])
    cameras, boards, checked = [], [], []
    for pose, observation in zip(poses, observations, strict=True):
        points, pixels, camera_board = (np.asarray(part, float) for part in observation)
        if (
            points.ndim != 2 or points.shape[1] != 3 or len(points) < 24
            or pixels.shape != (len(points), 2)
            or not np.isfinite(points).all() or not np.isfinite(pixels).all()
        ):
            raise ValueError("每帧需要至少 6 个标签的有限角点")
        camera_board = transform(camera_board)
        camera = pose7_to_mat(pose) @ mount
        cameras.append(camera)
        boards.append(camera @ camera_board)
        checked.append((points, pixels, camera_board))
    return cameras, boards, checked, len(selected)


def _metrics(cameras, boards, observations, device_board, snapshot, views):
    """计算固定板闭环及原始鱼眼图像重投影误差，不重新拟合验证观测。"""
    errors = [np.linalg.inv(device_board) @ board for board in boards]
    fx, fy, cx, cy = snapshot["intrinsics"]
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    distortion = np.asarray(snapshot["distortion"], float).reshape(4, 1)
    pixel_errors = []
    for camera, (points, pixels, _) in zip(cameras, observations, strict=True):
        camera_board = np.linalg.inv(camera) @ device_board
        if np.any((points @ camera_board[:3, :3].T + camera_board[:3, 3])[:, 2] <= 0):
            raise ValueError("标定板投影位于相机背面")
        predicted, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3),
            Rotation.from_matrix(camera_board[:3, :3]).as_rotvec(),
            camera_board[:3, 3], matrix, distortion,
        )
        pixel_errors.extend(np.linalg.norm(predicted.reshape(-1, 2) - pixels, axis=1))
    median, p95 = np.percentile(pixel_errors, [50, 95])
    return {
        "frames": len(boards), "views": views,
        "translation_rms_m": float(np.sqrt(np.mean([np.dot(e[:3, 3], e[:3, 3]) for e in errors]))),
        "rotation_rms_deg": float(np.rad2deg(np.sqrt(np.mean([
            Rotation.from_matrix(e[:3, :3]).magnitude() ** 2 for e in errors
        ])))),
        "median_px": float(median), "p95_px": float(p95),
    }


def check_metrics(metrics):
    """拒绝观测不足、非有限数值或超过桌面验收限值的结果。"""
    if metrics.get("frames", 0) < 30 or metrics.get("views", 0) < 4:
        raise TableQualityError("桌面验证观测或独立视角不足", metrics)
    failures = []  # 一次报告所有超限项，避免只看到首个失败原因。
    for key, limit in LIMITS.items():
        value = metrics.get(key, float("nan"))
        if not np.isfinite(value) or not 0 <= value <= limit:
            failures.append(f"{key}={value}，要求 <= {limit}")
    if failures:
        raise TableQualityError("桌面验证失败：" + "；".join(failures), metrics)


def solve_table(data, snapshot, height_m):
    """求解一个新桌面会话；失败不产生结果，通过后仍需独立采集验证。"""
    verify(snapshot)
    board_table = board_from_table(height_m)
    cameras, boards, observations, views = _observations(data, snapshot)
    device_board = mean_transform(boards)
    metrics = _metrics(cameras, boards, observations, device_board, snapshot, views)
    check_metrics(metrics)
    session = {
        "schema": "umi_table_session_v1", "session_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "tracking_device": "vive",
        "tracking_origin": "TrackingUniverseRawAndUncalibrated",
        "serial": snapshot["serial"], "geometry_digest": snapshot["geometry_digest"],
        "time_digest": digest(snapshot["time"]), "board": deepcopy(BOARD_SPEC),
        "board_height_m": float(height_m), "frame_convention": FRAME_CONVENTION,
        "table_from_device": np.linalg.inv(device_board @ board_table).tolist(),
        "solve_metrics": metrics,
    }
    session["core_digest"] = digest(session_core(session))
    session["verification"] = {"accepted": False, "reason": "需要独立采集验证"}
    return session


def verify_table(data, snapshot, session):
    """验证已有会话，返回新报告；无论成败都保持会话身份与变换不变。"""
    validate_session(session, snapshot, require_verified=False)
    cameras, boards, observations, views = _observations(data, snapshot)
    device_board = np.linalg.inv(transform(session["table_from_device"])) @ np.linalg.inv(
        board_from_table(session["board_height_m"])
    )
    metrics = _metrics(cameras, boards, observations, device_board, snapshot, views)
    check_metrics(metrics)
    return {
        "accepted": True, "core_digest": session["core_digest"],
        "checked_at": datetime.now(UTC).isoformat(), "metrics": metrics,
    }
