"""固定安装外参，使用 AprilGrid 原图残差求解相机到 Tracker 的时间偏移。"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from handumi.calibration.umi import transform


def refine_tag_corners(gray, corners):
    """定位标签黑色方块的外边界；边界不清晰或不完整时返回 None。

    参数为单通道图像和检测器返回的四角，返回同顺序的图像坐标。
    在归一化标签的四边中部寻找黑白交界，避开角部可能相接的黑色方块。
    """
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(corners).all() or abs(cv2.contourArea(corners)) < 16:
        return None
    target = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.float32)
    mapping = cv2.getPerspectiveTransform(corners, target)
    if abs(np.linalg.det(mapping)) < 1e-12:
        return None
    patch = cv2.warpPerspective(gray, mapping, (300, 300), borderValue=255)
    black, white = np.percentile(patch[80:220, 80:220], [15, 85])
    if white - black < 35:
        return None
    threshold = (black + white) / 2
    profiles = (
        np.median(patch[135:165, :], axis=0),
        np.median(patch[:, 135:165], axis=1),
    )
    edges = []
    for axis, start, stop, step in (
        (0, 105, 74, -1),
        (1, 105, 74, -1),
        (0, 195, 226, 1),
        (1, 195, 226, 1),
    ):
        profile = profiles[axis]
        if profile[start] >= threshold:
            return None
        for position in range(start + step, stop, step):
            if profile[position] >= threshold:
                previous = position - step
                dark, light = float(profile[previous]), float(profile[position])
                fraction = (threshold - dark) / max(light - dark, 1e-6)
                edges.append(previous + step * fraction)
                break
        else:
            return None
    left, top, right, bottom = edges
    box = np.array(
        [[[left, top], [right, top], [right, bottom], [left, bottom]]], np.float32
    )
    refined = cv2.perspectiveTransform(box, np.linalg.inv(mapping)).reshape(4, 2)
    height, width = gray.shape
    if (
        not np.isfinite(refined).all()
        or np.any(refined[:, 0] < 0)
        or np.any(refined[:, 0] >= width)
        or np.any(refined[:, 1] < 0)
        or np.any(refined[:, 1] >= height)
    ):
        return None
    return refined


def detect_board(rgb, matrix, distortion):
    """检测 6×6、55 mm、间距比 0.3 的 tag36h11 AprilGrid，返回米制点及原图角点。"""
    observation, _ = detect_board_details(rgb, matrix, distortion)
    return observation


def detect_board_details(rgb, matrix, distortion):
    """返回标定观测和有效标签的图像角点，观测不足时仍保留已检测标签。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    dictionary.maxCorrectionBits = 3
    parameters = cv2.aruco.DetectorParameters()
    parameters.errorCorrectionRate = 1.0
    parameters.perspectiveRemovePixelPerCell = 16
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.25
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, []
    points, pixels, markers = [], [], []
    for corner, tag_id in zip(corners, ids.reshape(-1)):
        if not 0 <= tag_id < 36:
            continue
        marker = refine_tag_corners(gray, corner)
        if marker is None:
            continue
        markers.append((int(tag_id), marker))
        # 沿用参考项目 Kalibr 板的角点映射 [1,0,3,2]。
        x, y = (tag_id % 6) * 0.0715, (tag_id // 6) * 0.0715
        points.extend(
            [[x, y, 0], [x + 0.055, y, 0], [x + 0.055, y + 0.055, 0], [x, y + 0.055, 0]]
        )
        pixels.extend(marker[[1, 0, 3, 2]])
    if len(points) < 24 or len({tag_id for tag_id, _ in markers}) != len(markers):
        return None, markers
    points, pixels = np.asarray(points, float), np.asarray(pixels, float)
    undistorted = cv2.fisheye.undistortPoints(
        pixels.reshape(-1, 1, 2), matrix, distortion.reshape(4, 1), P=matrix
    )
    ok, rotation, translation = cv2.solvePnP(points, undistorted, matrix, np.zeros(4))
    if not ok or translation[2, 0] <= 0:
        return None, markers
    camera_board = np.eye(4)
    camera_board[:3, :3] = Rotation.from_rotvec(rotation.reshape(3)).as_matrix()
    camera_board[:3, 3] = translation.reshape(3)
    return (points, pixels, camera_board), markers


def solve_time(
    camera_times, observations, tracker_times, tracker_poses, tracker_valid, snapshot
):
    """按原始单调时间求解偏移；输入时间单位秒，返回质量报告或抛出 ValueError。"""
    camera_times = np.asarray(camera_times, float)
    tracker_times = np.asarray(tracker_times, float)
    tracker_poses = np.asarray(tracker_poses, float)
    tracker_valid = np.asarray(tracker_valid, bool)
    if len(tracker_times) < 2 or np.any(np.diff(tracker_times) <= 0):
        raise ValueError("Tracker 时间必须严格递增")
    if (
        tracker_poses.shape != (len(tracker_times), 7)
        or tracker_valid.shape != tracker_times.shape
    ):
        raise ValueError("Tracker 位姿或有效性长度不一致")
    if not np.isfinite(tracker_poses).all() or not np.isfinite(tracker_times).all():
        raise ValueError("Tracker 数据包含非有限数值")
    if len(camera_times) < 30:
        raise ValueError(
            f"有效 AprilGrid 标定板观测仅 {len(camera_times)} 帧，至少需要 30 帧；"
            "请将指定标定板放入相机画面，确保至少 6 个标签可见"
        )
    selected = []
    for i, stamp in enumerate(camera_times):
        lo = np.searchsorted(tracker_times, stamp - 0.1) - 1
        hi = np.searchsorted(tracker_times, stamp + 0.1)
        if (
            lo >= 0
            and hi < len(tracker_times)
            and tracker_valid[lo : hi + 1].all()
            and np.all(np.diff(tracker_times[lo : hi + 1]) <= 0.05)
        ):
            selected.append(i)
    if len(selected) < 30:
        raise ValueError(
            f"{len(camera_times)} 帧标定板观测中，只有 {len(selected)} 帧在整个"
            "偏移搜索区间内具有连续有效的 Tracker 数据，少于 30 帧"
        )
    times = camera_times[selected]
    obs = [observations[i] for i in selected]
    # 将运动幅度限制在实际参与求解的时间范围，避免静止采集被噪声误判为可辨识。
    motion_mask = (tracker_times >= times[0] - 0.1) & (
        tracker_times <= times[-1] + 0.1
    ) & tracker_valid
    motion_poses = tracker_poses[motion_mask]
    low, high = np.percentile(motion_poses[:, :3], [5, 95], axis=0)
    translation_span = np.linalg.norm(high - low)
    motion_rotations = Rotation.from_quat(motion_poses[:, 3:])
    rotation_span = np.percentile(
        (motion_rotations[0].inv() * motion_rotations).magnitude(), 95
    )
    if translation_span < 0.01 and rotation_span < np.deg2rad(3):
        raise ValueError("运动幅度不足，无法辨识时间偏移；请进行明显的变速平移和旋转")
    # 相对于起点计算时间块，避免绝对时钟影响训练/验证分割。
    blocks = ((times - times[0]) / 2).astype(int)
    unique = np.unique(blocks)
    if len(unique) < 2:
        raise ValueError("至少需要两个 2 秒时间块")
    validation_blocks = unique[:: max(1, round(1 / 0.2))]
    validation = np.isin(blocks, validation_blocks)
    train = ~validation
    if train.sum() < 4 or not validation.any():
        raise ValueError("训练或验证观测不足")
    rotations = Slerp(tracker_times, Rotation.from_quat(tracker_poses[:, 3:]))
    tracker_camera = transform(snapshot["tracker_camera"])
    fx, fy, cx, cy = snapshot["intrinsics"]
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    distortion = np.asarray(snapshot["distortion"], float).reshape(4, 1)

    def world_camera(stamp):
        """在有效时间线内部构造相机世界位姿。"""
        pose = np.eye(4)
        pose[:3, :3] = rotations(stamp).as_matrix()
        pose[:3, 3] = [
            np.interp(stamp, tracker_times, tracker_poses[:, j]) for j in range(3)
        ]
        return pose @ tracker_camera

    def unpack(parameters):
        """将板的六维优化变量转换为齐次变换。"""
        board = np.eye(4)
        board[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        board[:3, 3] = parameters[3:6]
        return board

    def residual(parameters, mask):
        """计算未去畸变原图上的角点残差；安装外参始终固定。"""
        board = unpack(parameters)
        values = []
        for stamp, observation, use in zip(times, obs, mask):
            if not use:
                continue
            points, pixels, _ = observation
            camera_board = np.linalg.inv(world_camera(stamp + parameters[6])) @ board
            rotation = Rotation.from_matrix(camera_board[:3, :3]).as_rotvec()
            predicted, _ = cv2.fisheye.projectPoints(
                points.reshape(-1, 1, 3),
                rotation,
                camera_board[:3, 3],
                matrix,
                distortion,
            )
            values.extend((predicted.reshape(-1, 2) - pixels).reshape(-1))
        return np.asarray(values)

    best = None
    for offset in np.linspace(-0.1, 0.1, 101):
        boards = [
            world_camera(stamp + offset) @ observation[2]
            for stamp, observation, use in zip(times, obs, train)
            if use
        ]
        rotation = (
            Rotation.from_matrix(np.array([b[:3, :3] for b in boards]))
            .mean()
            .as_rotvec()
        )
        initial = np.r_[rotation, np.mean([b[:3, 3] for b in boards], axis=0), offset]
        cost = np.median(np.abs(residual(initial, train)))
        if best is None or cost < best[0]:
            best = cost, initial
    result = least_squares(
        lambda p: residual(p, train),
        best[1],
        bounds=([-np.inf] * 6 + [-0.1], [np.inf] * 6 + [0.1]),
        loss="soft_l1",
        f_scale=1,
        x_scale="jac",
        max_nfev=2000,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("时间优化失败")
    if 0.1 - abs(result.x[6]) <= 0.0001:
        raise ValueError("时间偏移触及搜索边界")
    # 将时间导数投影到板位姿导数的正交补，拒绝可被板位姿完全吸收的延迟。
    jacobian = result.jac
    temporal = jacobian[:, 6]
    unexplained = (
        temporal
        - jacobian[:, :6] @ np.linalg.lstsq(jacobian[:, :6], temporal, rcond=None)[0]
    )
    if (
        np.linalg.norm(unexplained) < 1
        or np.linalg.norm(unexplained) / max(np.linalg.norm(temporal), 1) < 1e-3
    ):
        raise ValueError("运动无法辨识时间偏移；请进行变速和多轴运动")
    errors = np.linalg.norm(residual(result.x, validation).reshape(-1, 2), axis=1)
    board = unpack(result.x)
    closed = [
        np.linalg.inv(board) @ world_camera(stamp + result.x[6]) @ observation[2]
        for stamp, observation, use in zip(times, obs, validation)
        if use
    ]
    position_rmse = float(
        np.sqrt(np.mean([np.dot(p[:3, 3], p[:3, 3]) for p in closed]))
    )
    rotation_rmse = float(
        np.sqrt(
            np.mean([Rotation.from_matrix(p[:3, :3]).magnitude() ** 2 for p in closed])
        )
        * 180
        / np.pi
    )
    median, p95 = np.percentile(errors, [50, 95])
    if median > 1 or p95 > 2 or position_rmse > 0.005 or rotation_rmse > 1:
        raise ValueError(
            f"时间/几何验证失败：像素 {median:.3f}/{p95:.3f}，闭环 {position_rmse:.4f} m/{rotation_rmse:.3f} deg"
        )
    return {
        "accepted": True,
        "offset_s": float(result.x[6]),
        "geometry_digest": snapshot["geometry_digest"],
        "median_px": float(median),
        "p95_px": float(p95),
        "position_rmse_m": position_rmse,
        "rotation_rmse_deg": rotation_rmse,
        "frames": len(times),
    }
