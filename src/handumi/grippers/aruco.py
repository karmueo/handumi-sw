"""提供与 ROS 解耦的双 ArUco 标记三维夹爪开合度估计算法。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerPose:
    """保存标记坐标系相对于 RGB 光学相机的有限 PnP 位姿。

    ``rotation_vector`` 是 Rodrigues 旋转向量；
    ``translation_mm`` 是从 RGB 光学相机原点指向标记原点的毫米平移向量。
    两个字段使用元组，以保证结果对象及其向量值不可变。

    Attributes:
        rotation_vector: 形状为 ``(3,)`` 的有限 Rodrigues 旋转向量。
        translation_mm: 形状为 ``(3,)`` 的有限毫米平移向量。
    """

    rotation_vector: tuple[float, float, float]
    translation_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        """验证、规范化位姿向量，并保持 dataclass 的冻结语义。

        Raises:
            ValueError: 任一向量不是三个有限数值时抛出。
        """
        rotation = np.asarray(self.rotation_vector, dtype=np.float64)
        translation = np.asarray(self.translation_mm, dtype=np.float64)
        if (
            rotation.shape != (3,)
            or translation.shape != (3,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
        ):
            raise ValueError("标记位姿必须包含两个有限的三维向量")
        object.__setattr__(
            self, "rotation_vector", tuple(float(value) for value in rotation)
        )
        object.__setattr__(
            self, "translation_mm", tuple(float(value) for value in translation)
        )


@dataclass(frozen=True)
class OpennessEstimate:
    """保存单帧夹爪归一化距离估计结果。

    Attributes:
        openness: 平滑后的无量纲归一化距离，范围为 0 到 1。
        raw_openness: 当前帧未经平滑的无量纲归一化距离。
        distance_mm: 两枚 ArUco 标记中心的三维距离，单位为毫米。
        left_center: 左侧标记中心在原始图像中的像素坐标。
        right_center: 右侧标记中心在原始图像中的像素坐标。
        left_pose: 左侧标记相对于 RGB 光学相机的 PnP 位姿。
        right_pose: 右侧标记相对于 RGB 光学相机的 PnP 位姿。
        roi: 检测区域，顺序为 x_min、y_min、x_max、y_max。
    """

    openness: float
    raw_openness: float
    distance_mm: float
    left_center: tuple[float, float]
    right_center: tuple[float, float]
    left_pose: MarkerPose
    right_pose: MarkerPose
    roi: tuple[int, int, int, int]


class GripperOpennessEstimator:
    """通过两枚 ArUco 标记的三维毫米距离估计无量纲夹爪距离。

    输入角点来自原始鱼眼图像。估计器先按 equidistant 模型校正角点，
    再用方形标记的 IPPE PnP 计算三维位置。只有两枚目标标记的位姿都
    有效时才更新指数平滑状态。

    Args:
        camera_matrix: 形状为 ``(3, 3)`` 的相机内参矩阵。
        distortion_coefficients: 四个 equidistant 鱼眼畸变系数。
        closed_distance_mm: 夹爪完全闭合时的三维标记距离，单位为毫米。
        open_distance_mm: 夹爪完全张开时的三维标记距离，单位为毫米。
        marker_size_mm: 正方形 ArUco 标记边长，单位为毫米。
        smoothing_alpha: 当前帧在指数平滑中的权重，范围为 (0, 1]。
        dictionary_name: OpenCV ArUco 预定义字典名称。
        left_marker_id: 左侧标记 ID。
        right_marker_id: 右侧标记 ID。
        roi_ratios: ROI 比例，顺序为 x_min、y_min、x_max、y_max。
        image_resolution: 输入图像的预期宽、高；为空时不限制尺寸。

    Raises:
        ValueError: 相机参数、距离、标记、平滑系数或 ROI 参数无效时抛出。
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        closed_distance_mm: float,
        open_distance_mm: float,
        marker_size_mm: float = 16.0,
        smoothing_alpha: float = 0.35,
        dictionary_name: str = "DICT_4X4_50",
        left_marker_id: int = 0,
        right_marker_id: int = 1,
        roi_ratios: tuple[float, float, float, float] = (
            0.15,
            0.58,
            0.85,
            0.82,
        ),
        image_resolution: tuple[int, int] | None = None,
    ) -> None:
        """初始化鱼眼校正、ArUco 检测、PnP 和归一化参数。"""
        self.camera_matrix = self._validate_camera_matrix(camera_matrix)
        self.distortion_coefficients = self._validate_distortion(
            distortion_coefficients
        )
        if not np.isfinite(closed_distance_mm) or not np.isfinite(open_distance_mm):
            raise ValueError("闭合和张开距离必须为有限数值")
        if closed_distance_mm < 0.0 or closed_distance_mm >= open_distance_mm:
            raise ValueError("closed_distance_mm 必须非负且小于 open_distance_mm")
        if not np.isfinite(marker_size_mm) or marker_size_mm <= 0.0:
            raise ValueError("marker_size_mm 必须为正的有限数值")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha 必须在 (0, 1] 范围内")
        if left_marker_id == right_marker_id:
            raise ValueError("left_marker_id 与 right_marker_id 必须不同")
        self._validate_roi(roi_ratios)
        self.image_resolution = self._validate_image_resolution(image_resolution)

        # 闭合和张开的毫米标定距离，仅用于生成无量纲输出。
        self.closed_distance_mm = float(closed_distance_mm)
        self.open_distance_mm = float(open_distance_mm)
        # 正方形标记边长，决定 PnP 平移向量的毫米尺度。
        self.marker_size_mm = float(marker_size_mm)
        # 指数平滑权重，较大值带来更快响应。
        self.smoothing_alpha = float(smoothing_alpha)
        # 两侧目标标记 ID。
        self.left_marker_id = int(left_marker_id)
        self.right_marker_id = int(right_marker_id)
        # 相对于输入图像尺寸的检测区域。
        self.roi_ratios = tuple(float(value) for value in roi_ratios)
        # 上一帧有效的无量纲平滑结果。
        self._smoothed_openness: float | None = None
        # 最近一次估计中检测到的目标标签数量，供节点发布质量信息。
        self._last_detected_marker_count = 0
        # 校正后的角点已处于针孔像素坐标，PnP 使用零畸变系数。
        self._zero_distortion = np.zeros((4, 1), dtype=np.float64)

        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"未知 ArUco 字典: {dictionary_name}")
        # ArUco 字典在节点生命周期内复用，兼容 OpenCV 4.6 的过程式 API。
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        # OpenCV 4.7 起可复用面向对象检测器，旧版本回退到模块函数。
        detector_class = getattr(cv2.aruco, "ArucoDetector", None)
        self._detector = (
            detector_class(self._dictionary) if callable(detector_class) else None
        )
        # IPPE_SQUARE 要求物点从左上角开始顺时针排列。
        half_size = self.marker_size_mm * 0.5
        self._marker_object_points = np.asarray(
            [
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _validate_camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
        """验证并复制相机内参矩阵。

        Args:
            camera_matrix: 待验证的相机内参矩阵。

        Returns:
            形状为 ``(3, 3)`` 的双精度矩阵。

        Raises:
            ValueError: 形状、数值或焦距无效时抛出。
        """
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix 必须是包含有限数值的 3x3 矩阵")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera_matrix 的焦距必须为正数")
        return matrix.copy()

    @staticmethod
    def _validate_distortion(
        distortion_coefficients: np.ndarray,
    ) -> np.ndarray:
        """验证并整理 equidistant 鱼眼畸变系数。

        Args:
            distortion_coefficients: 待验证的鱼眼畸变系数。

        Returns:
            形状为 ``(4, 1)`` 的双精度系数矩阵。

        Raises:
            ValueError: 系数数量或数值无效时抛出。
        """
        distortion = np.asarray(distortion_coefficients, dtype=np.float64)
        if distortion.size != 4 or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion_coefficients 必须包含四个有限数值")
        return distortion.reshape(4, 1).copy()

    @staticmethod
    def _validate_roi(roi_ratios: tuple[float, float, float, float]) -> None:
        """验证 ROI 比例参数。

        Args:
            roi_ratios: 顺序为 x_min、y_min、x_max、y_max 的比例。

        Raises:
            ValueError: 比例数量、范围或边界顺序无效时抛出。
        """
        if len(roi_ratios) != 4:
            raise ValueError("roi_ratios 必须包含四个值")
        x_min, y_min, x_max, y_max = roi_ratios
        if not all(0.0 <= value <= 1.0 for value in roi_ratios):
            raise ValueError("ROI 比例必须在 [0, 1] 范围内")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("ROI 最小边界必须小于最大边界")

    @staticmethod
    def _validate_image_resolution(
        image_resolution: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        """验证输入图像的预期宽高。

        Args:
            image_resolution: 输入图像的预期宽、高；为空时不限制尺寸。

        Returns:
            经过验证的宽、高元组，或 ``None``。

        Raises:
            ValueError: 分辨率不含两个正整数时抛出。
        """
        if image_resolution is None:
            return None
        if len(image_resolution) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in image_resolution
        ):
            raise ValueError("image_resolution 必须包含两个正整数")
        return int(image_resolution[0]), int(image_resolution[1])

    def normalize_distance(self, distance_mm: float) -> float:
        """将内部毫米距离映射为无量纲 0 到 1 数值。

        Args:
            distance_mm: 两枚标记中心的三维距离，单位为毫米。

        Returns:
            无量纲归一化距离；0 表示闭合，1 表示完全张开。
        """
        calibrated_range = self.open_distance_mm - self.closed_distance_mm
        normalized = (float(distance_mm) - self.closed_distance_mm) / calibrated_range
        return float(np.clip(normalized, 0.0, 1.0))

    def reset(self) -> None:
        """清除平滑状态，使下一次有效估计直接采用当前帧结果。"""
        self._smoothed_openness = None

    @property
    def last_detected_marker_count(self) -> int:
        """返回最近一次估计中检测到的目标标签数量。

        Returns:
            目标标签数量，范围为 0 到 2。
        """
        return self._last_detected_marker_count

    def estimate_marker_pose(self, distorted_corners: np.ndarray) -> MarkerPose | None:
        """从原始鱼眼角点估计标记相对于 RGB 光学相机的位姿。

        Args:
            distorted_corners: 原始图像中的四个角点，顺序遵循 ArUco 输出。

        Returns:
            有效的 Rodrigues 旋转和毫米平移；PnP 无效或位于相机后方时
            返回 ``None``。

        Raises:
            ValueError: 角点数量或数值无效时抛出。
        """
        corners = np.asarray(distorted_corners, dtype=np.float64)
        if corners.size != 8 or not np.all(np.isfinite(corners)):
            raise ValueError("标记角点必须包含四个有限二维坐标")
        corners = corners.reshape(4, 1, 2)
        # 将 equidistant 鱼眼角点转换为使用原相机矩阵的针孔像素坐标。
        undistorted_corners = cv2.fisheye.undistortPoints(
            corners,
            self.camera_matrix,
            self.distortion_coefficients,
            P=self.camera_matrix,
        ).reshape(4, 2)
        success, rotation, translation = cv2.solvePnP(
            self._marker_object_points,
            undistorted_corners,
            self.camera_matrix,
            self._zero_distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None
        rotation_vector = np.asarray(rotation, dtype=np.float64).reshape(3)
        position_mm = np.asarray(translation, dtype=np.float64).reshape(3)
        if (
            not np.all(np.isfinite(rotation_vector))
            or not np.all(np.isfinite(position_mm))
            or position_mm[2] <= 0.0
        ):
            return None
        return MarkerPose(
            rotation_vector=tuple(float(value) for value in rotation_vector),
            translation_mm=tuple(float(value) for value in position_mm),
        )

    def estimate_marker_position(
        self, distorted_corners: np.ndarray
    ) -> np.ndarray | None:
        """兼容旧接口，仅返回从原始鱼眼角点恢复的毫米平移向量。

        Args:
            distorted_corners: 原始图像中的四个角点，顺序遵循 ArUco 输出。

        Returns:
            形状为 ``(3,)`` 的毫米平移向量；位姿无效时返回 ``None``。

        Raises:
            ValueError: 角点数量或数值无效时抛出。
        """
        pose = self.estimate_marker_pose(distorted_corners)
        if pose is None:
            return None
        return np.asarray(pose.translation_mm, dtype=np.float64)

    def estimate(self, image: np.ndarray) -> OpennessEstimate | None:
        """估计单帧原始鱼眼图像中的无量纲夹爪距离。

        Args:
            image: 灰度、BGR 或 BGRA 原始鱼眼图像数组。

        Returns:
            两枚目标标记及其 PnP 位姿均有效时返回结果，否则返回 ``None``。

        Raises:
            ValueError: 输入图像为空或通道格式不受支持时抛出。
        """
        self._last_detected_marker_count = 0
        if image is None or image.size == 0:
            raise ValueError("输入图像不能为空")
        gray_image = self._to_grayscale(image)
        # 输入图像尺寸，用于将比例 ROI 转换为像素边界。
        image_height, image_width = gray_image.shape
        actual_resolution = (image_width, image_height)
        if (
            self.image_resolution is not None
            and actual_resolution != self.image_resolution
        ):
            expected_width, expected_height = self.image_resolution
            raise ValueError(
                "输入图像分辨率 "
                f"{image_width}x{image_height} 与相机标定分辨率 "
                f"{expected_width}x{expected_height} 不一致"
            )
        x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio = self.roi_ratios
        roi = (
            int(round(image_width * x_min_ratio)),
            int(round(image_height * y_min_ratio)),
            int(round(image_width * x_max_ratio)),
            int(round(image_height * y_max_ratio)),
        )
        x_min, y_min, x_max, y_max = roi
        # 限定在夹爪区域内检测，以降低计算量和背景误检概率。
        roi_image = gray_image[y_min:y_max, x_min:x_max]
        if self._detector is not None:
            corners, marker_ids, _ = self._detector.detectMarkers(roi_image)
        else:
            corners, marker_ids, _ = cv2.aruco.detectMarkers(
                roi_image, self._dictionary
            )
        if marker_ids is None:
            self.reset()
            return None

        # 目标标记的四角点和中心均恢复到原始完整图像坐标。
        offset = np.asarray([x_min, y_min], dtype=np.float64)
        marker_corners = {
            int(marker_id): detected_corners.reshape(4, 2).astype(np.float64) + offset
            for detected_corners, marker_id in zip(corners, marker_ids.flatten())
            if int(marker_id) in (self.left_marker_id, self.right_marker_id)
        }
        self._last_detected_marker_count = min(len(marker_corners), 2)
        if (
            self.left_marker_id not in marker_corners
            or self.right_marker_id not in marker_corners
        ):
            self.reset()
            return None

        left_corners = marker_corners[self.left_marker_id]
        right_corners = marker_corners[self.right_marker_id]
        left_pose = self.estimate_marker_pose(left_corners)
        right_pose = self.estimate_marker_pose(right_corners)
        if left_pose is None or right_pose is None:
            self.reset()
            return None

        # PnP 平移向量和距离都使用毫米；对外输出只使用无量纲结果。
        left_position_mm = np.asarray(left_pose.translation_mm)
        right_position_mm = np.asarray(right_pose.translation_mm)
        distance_mm = float(np.linalg.norm(left_position_mm - right_position_mm))
        if not np.isfinite(distance_mm):
            self.reset()
            return None
        raw_openness = self.normalize_distance(distance_mm)
        if self._smoothed_openness is None:
            smoothed_openness = raw_openness
        else:
            smoothed_openness = (
                self.smoothing_alpha * raw_openness
                + (1.0 - self.smoothing_alpha) * self._smoothed_openness
            )
        self._smoothed_openness = float(np.clip(smoothed_openness, 0.0, 1.0))

        left_center = tuple(float(value) for value in left_corners.mean(axis=0))
        right_center = tuple(float(value) for value in right_corners.mean(axis=0))
        return OpennessEstimate(
            openness=self._smoothed_openness,
            raw_openness=raw_openness,
            distance_mm=distance_mm,
            left_center=left_center,
            right_center=right_center,
            left_pose=left_pose,
            right_pose=right_pose,
            roi=roi,
        )

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """把支持的输入图像格式转换为单通道灰度图。

        Args:
            image: 灰度、BGR 或 BGRA 图像数组。

        Returns:
            单通道灰度图。

        Raises:
            ValueError: 图像维度或通道数量不受支持时抛出。
        """
        if image.ndim == 2:
            return image
        if image.ndim != 3:
            raise ValueError("输入图像必须是二维灰度图或三维彩色图")
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError("仅支持单通道、BGR 或 BGRA 图像")
