"""验证 AprilGrid 标签外边界修正和不可靠观测的拒绝。"""

import cv2
import numpy as np
import pytest

from handumi.calibration.umi_time import refine_tag_corners


def tag_image():
    """生成具有明确黑白边界和内部白色图案的合成标签。"""
    image = np.full((240, 240), 230, np.uint8)
    cv2.rectangle(image, (60, 60), (179, 179), 20, -1)
    cv2.rectangle(image, (95, 95), (145, 145), 230, -1)
    return image


@pytest.mark.parametrize("inset", [0, 10])
def test_refine_finds_outer_border(inset):
    """已有边界和向内偏移的角点均应定位到同一黑白交界。"""
    image = tag_image()
    corners = np.array(
        [
            [60 + inset, 60 + inset],
            [179 - inset, 60 + inset],
            [179 - inset, 179 - inset],
            [60 + inset, 179 - inset],
        ],
        np.float32,
    )
    refined = refine_tag_corners(image, corners)
    expected = np.array([[60, 60], [179, 60], [179, 179], [60, 179]])
    np.testing.assert_allclose(refined, expected, atol=1)


def test_refine_keeps_corner_order_under_perspective():
    """透视和旋转后仍恢复外围边界，并保持检测器的角点顺序。"""
    image = tag_image()
    source = np.array([[60, 60], [179, 60], [179, 179], [60, 179]], np.float32)
    target = np.array([[74, 41], [191, 69], [165, 192], [42, 155]], np.float32)
    mapping = cv2.getPerspectiveTransform(source, target)
    transformed = cv2.warpPerspective(image, mapping, (240, 240), borderValue=230)
    inset = np.array([[[70, 70], [169, 70], [169, 169], [70, 169]]], np.float32)
    corners = cv2.perspectiveTransform(inset, mapping).reshape(4, 2)
    refined = refine_tag_corners(transformed, np.roll(corners, 2, axis=0))
    np.testing.assert_allclose(refined, np.roll(target, 2, axis=0), atol=1.5)


def test_refine_rejects_missing_boundary():
    """边界不可见或角点退化时不能伪造标定角点。"""
    corners = np.array([[60, 60], [179, 60], [179, 179], [60, 179]], np.float32)
    assert refine_tag_corners(np.zeros((240, 240), np.uint8), corners) is None
    assert refine_tag_corners(tag_image(), np.zeros((4, 2), np.float32)) is None
