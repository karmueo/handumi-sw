"""绘制 Vive 相机去畸变画面及其在追踪参考系中的空间位置。"""

import cv2
import numpy as np

# 图像继承默认相机坐标，定位失效时由父节点递归清除。
CAMERA_PATH = "tracking/camera"
IMAGE_PATH = f"{CAMERA_PATH}/image"


class SpatialCameraPreview:
    """缓存鱼眼去畸变映射，并记录与当前追踪样本对应的相机模型。"""

    def __init__(self, snapshot):
        """从已验证的几何快照建立显示模型，不修改原始标定。"""
        self.resolution = tuple(int(value) for value in snapshot["resolution"])
        fx, fy, cx, cy = snapshot["intrinsics"]
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        distortion = np.asarray(snapshot["distortion"], dtype=float).reshape(4, 1)
        self.image_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            matrix, distortion, self.resolution, np.eye(3), balance=0.7,
            new_size=self.resolution,
        )
        self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, distortion, np.eye(3), self.image_matrix, self.resolution,
            cv2.CV_32FC1,
        )

    def log(self, viewer, image, tracking, camera_age_s):
        """在相机父坐标下绘制视频；画面失效时只清除图像子树。"""
        # 父相机节点由共享预览负责清除，避免无效位姿下重建子节点。
        if not (tracking.right_tracked and tracking.right_pose_valid):
            return
        if not np.isfinite(camera_age_s) or not 0 <= camera_age_s <= 0.35:
            viewer.log(IMAGE_PATH, viewer.Clear(recursive=True))
            return

        rectified = cv2.remap(
            image.image, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        viewer.log(
            IMAGE_PATH,
            viewer.Pinhole(
                image_from_camera=self.image_matrix,
                resolution=self.resolution,
                camera_xyz=viewer.ViewCoordinates.RDF,
                image_plane_distance=0.12,
                color=[110, 180, 255],
                line_width=0.002,
            ),
        )
        viewer.log(IMAGE_PATH, viewer.Image(rectified).compress(jpeg_quality=80))
