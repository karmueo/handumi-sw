"""验证三角表面间距保守认证的边界和失败行为。"""

import numpy as np
import pytest

pytest.importorskip("rtree")

from handumi.robots.mesh_surface_audit import (
    TriangleSurfaceIndex,
    certify_surface_clearance,
)


def test_separated_triangles_are_certified():
    """分离三角面安全距离获得证明。"""
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    assert certify_surface_clearance(tri, tri + [0, 0, 0.02], 0.01).proven


def test_crossing_or_near_edges_are_never_certified():
    """相交面或边内部最近距离不足阈值时不可误判安全。"""
    a = np.array([[[-1, 0, 0], [1, 0, 0], [0, -1, 0]]], dtype=float)
    b = np.array([[[0, -1, 0.005], [0, 1, 0.005], [1, 0, 0.005]]], dtype=float)
    assert not certify_surface_clearance(a, b, 0.006, max_depth=32).proven
    assert certify_surface_clearance(a, b, 0.004, max_depth=32).proven
    b[:, :, 2] = 0
    assert not certify_surface_clearance(a, b, 0.001, max_depth=20).proven


def test_small_edges_and_degenerate_triangles_fail_closed():
    """小尺度交叉边与退化点均不得被漏检。"""
    a = np.array([[[-1e-4, 0, 0], [1e-4, 0, 0], [0, -1e-4, 0]]])
    b = np.array([[[0, -1e-4, 5e-6], [0, 1e-4, 5e-6], [1e-4, 0, 5e-6]]])
    assert not certify_surface_clearance(a, b, 6e-6, max_depth=24).proven
    point = np.zeros((1, 3, 3))
    assert not certify_surface_clearance(point, point, 0.001).proven


def test_invalid_input_and_bound_limit():
    """空数据、非有限数值报错，耗尽细分预算返回未证明。"""
    tri = np.zeros((1, 3, 3))
    with pytest.raises(ValueError):
        certify_surface_clearance(np.empty((0, 3, 3)), tri, 0.01)
    with pytest.raises(ValueError):
        certify_surface_clearance(tri * np.nan, tri, 0.01)
    assert not certify_surface_clearance(tri, tri, 0.01, max_subdivisions=0).proven


def test_rotating_surface_certifies_whole_interval_and_rejects_midpoint_collision():
    """轴旋转扫掠可证明安全运动，且不会漏掉端点安全而中途相交的情况。"""
    static = np.array([[[0.99, -0.01, 0], [1.01, -0.01, 0], [1, 0.01, 0]]])
    checker = TriangleSurfaceIndex(static)
    safe = static + [0, 0, 0.03]
    assert checker.certify_rotation(safe, np.zeros(3), np.array([0, 0, 1]), 0.1, 0.01)
    assert not checker.certify_rotation(static, np.zeros(3), np.array([0, 0, 1]),
                                        0.2, 0.01, max_subdivisions=100)
    assert not checker.certify_rotation(safe, np.zeros(3), np.array([0, 0, 1]),
                                        np.pi, 0.01)
    assert not checker.certify_rotation(safe, np.zeros(3), np.array([0, 0, 1]),
                                        0.1, float("nan"))
    assert not checker.certify_rotation(safe, np.zeros(2), np.array([0, 0, 1]),
                                        0.1, 0.01)
    assert not checker.certify_rotation(safe, np.zeros(3), np.array([0, 0, 1]),
                                        0.1, 0.01, max_subdivisions=-1)
