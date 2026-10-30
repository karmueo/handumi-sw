"""验证倾斜 AprilGrid 的真实 OpenCV 检测和检测统计。"""

import cv2
import numpy as np
import pytest

from handumi.calibration import umi_time

PIXELS_PER_M = 2000  # 合成纹理分辨率：1 像素对应 0.5 mm。
MARGIN_M = 0.03  # 纹理四周白边宽度。


def aprilgrid_texture():
    """生成带角块的 6×6、55 mm、间距比 0.3 的 tag36h11 板纹理。"""
    size = round((0.4125 + 2 * MARGIN_M) * PIXELS_PER_M)
    texture = np.full((size, size), 235, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side = round(0.055 * PIXELS_PER_M)
    gap = round(0.0165 * PIXELS_PER_M)
    for tag_id in range(36):
        u = round(((tag_id % 6) * 0.0715 + MARGIN_M) * PIXELS_PER_M)
        v = round(((tag_id // 6) * 0.0715 + MARGIN_M) * PIXELS_PER_M)
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, side, borderBits=1)
        texture[v:v + side, u:u + side] = np.where(tag > 128, 235, 20)
    for column in range(7):
        for row in range(7):
            u = round((column * 0.0715 - 0.0165 + MARGIN_M) * PIXELS_PER_M)
            v = round((row * 0.0715 - 0.0165 + MARGIN_M) * PIXELS_PER_M)
            texture[max(v, 0):v + gap, max(u, 0):u + gap] = 20
    return texture


def tilted_board(tilt_deg, *, blur=1.5, distance_m=0.45, focal=600.0):
    """以针孔相机渲染绕 x 轴倾斜的板，加入模糊和 JPEG 压缩模拟 MJPG 画面。"""
    width, height = 960, 720
    matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    rotation = cv2.Rodrigues(np.array([np.deg2rad(tilt_deg), 0.0, 0.0]))[0]
    translation = rotation @ [-0.20625, -0.20625, 0.0] + [0.0, 0.0, distance_m]
    texture_board = np.array([
        [1 / PIXELS_PER_M, 0, -MARGIN_M], [0, 1 / PIXELS_PER_M, -MARGIN_M], [0, 0, 1],
    ])
    homography = matrix @ np.c_[rotation[:, :2], translation] @ texture_board
    image = cv2.warpPerspective(
        aprilgrid_texture(), homography, (width, height), borderValue=200
    )
    image = cv2.GaussianBlur(image, (0, 0), blur)
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB), matrix


@pytest.mark.parametrize("tilt_deg,minimum", [(0, 34), (30, 28), (50, 24)])
def test_tilted_board_keeps_most_tags(tilt_deg, minimum):
    """倾斜和轻度模糊时标签黑边与角块相邻，仍应检出大部分标签。"""
    image, matrix = tilted_board(tilt_deg)
    stats = {}
    _, markers = umi_time.detect_board_details(image, matrix, np.zeros(4), stats=stats)
    tag_ids = [tag_id for tag_id, _ in markers]
    assert len(tag_ids) >= minimum
    assert len(set(tag_ids)) == len(tag_ids)
    assert stats["aruco_tag_count"] >= len(tag_ids)
    assert stats["rejected_candidate_count"] >= 0


def test_detection_stats_report_empty_frame():
    """无标签画面也写入零计数，诊断中不会缺字段。"""
    stats = {}
    observation, markers = umi_time.detect_board_details(
        np.full((120, 160, 3), 200, np.uint8), np.eye(3), np.zeros(4), stats=stats
    )
    assert observation is None and markers == []
    assert stats["aruco_tag_count"] == 0
    assert stats["rejected_candidate_count"] >= 0


def test_noise_does_not_produce_board_tags():
    """放宽阈值后随机纹理仍不能产生标定板标签。"""
    image = np.random.default_rng(0).integers(0, 256, (720, 960, 3), np.uint8)
    _, markers = umi_time.detect_board_details(image, np.eye(3), np.zeros(4))
    assert markers == []
