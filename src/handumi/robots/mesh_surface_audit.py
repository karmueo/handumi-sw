"""核验三角网格表面间距的保守下界，不将表面距离解释为实体间距。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 对输入坐标、空间索引和浮点二分预留的距离裕量，单位米。
_AUDIT_PADDING_M = 1e-6


@dataclass(frozen=True)
class SurfaceClearance:
    """三角表面间距认证结果；未证明不代表碰撞。"""

    proven: bool
    clearance_m: float
    subdivisions: int
    reason: str


class TriangleSurfaceIndex:
    """缓存一组局部三角面及其空间索引，批量证明表面间距下界。"""

    def __init__(self, triangles: np.ndarray):
        """构建局部三角面的 R 树；调用者负责网格坐标与单位。"""
        try:
            from rtree import index
        except ImportError as exc:
            raise RuntimeError("精细自碰撞复核需要安装 rtree") from exc
        self.triangles = np.asarray(triangles, dtype=np.float64)
        if (self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3)
                or not len(self.triangles) or not np.isfinite(self.triangles).all()):
            raise ValueError("triangle arrays must be non-empty, finite, and shaped (N, 3, 3)")
        properties = index.Property()
        properties.dimension = 3
        lower = self.triangles.min(axis=1)
        upper = self.triangles.max(axis=1)
        self.tree = index.Index(
            ((i, tuple(np.r_[lo, hi]), None) for i, (lo, hi) in enumerate(zip(lower, upper))),
            interleaved=True, properties=properties,
        )

    def _candidates(self, lower: np.ndarray, upper: np.ndarray):
        """分批查询相交包围盒并限制候选数量，避免高度重叠时占满内存。"""
        indices, matches = [], []
        total = 0
        for start in range(0, len(lower), 128):
            ids, counts = self.tree.intersection_v(lower[start:start + 128],
                                                    upper[start:start + 128])
            total += len(ids)
            if total > 200_000:
                return None
            if len(ids):
                indices.append(np.repeat(np.arange(start, start + len(counts)),
                                         counts.astype(np.int64)))
                matches.append(ids)
        if not indices:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(indices), np.concatenate(matches)

    def certify(self, triangles: np.ndarray, clearance_m: float, *,
                physical_margin: float | None = None,
                max_subdivisions: int = 2_000_000, max_depth: int = 60) -> SurfaceClearance:
        """在索引坐标系证明指定间距，失败时区分实测见证与预算不足。"""
        a = np.asarray(triangles, dtype=np.float64)
        if (a.ndim != 3 or a.shape[1:] != (3, 3) or not len(a)
                or not np.isfinite(a).all() or not np.isfinite(clearance_m)
                or clearance_m < 0 or max_depth < 0 or max_subdivisions < 0):
            raise ValueError("invalid triangles, clearance or subdivision limit")
        threshold = float(clearance_m) + _AUDIT_PADDING_M
        candidates = self._candidates(a.min(axis=1) - threshold,
                                       a.max(axis=1) + threshold)
        if candidates is None:
            return SurfaceClearance(False, clearance_m, 0, "candidate limit; separation unproven")
        indices, ids = candidates
        if not len(ids):
            return SurfaceClearance(True, clearance_m, 0, "all triangle AABBs separated")
        left = a[indices]
        right = self.triangles[ids]
        subdivisions = 0
        for _depth in range(max_depth + 1):
            separation = np.maximum(np.maximum(left.min(axis=1) - right.max(axis=1),
                                                right.min(axis=1) - left.max(axis=1)), 0)
            unresolved = np.einsum("ij,ij->i", separation, separation) < threshold ** 2
            left, right = left[unresolved], right[unresolved]
            if not len(left):
                return SurfaceClearance(True, clearance_m, subdivisions, "all triangle AABBs separated")
            if physical_margin is not None:
                # 两三角面的顶点均为真实表面点；见证小于门槛才可断定不满足。
                delta = left[:, :, None, :] - right[:, None, :, :]
                closest = np.min(np.einsum("...i,...i->...", delta, delta))
                if closest < max(0.0, physical_margin - _AUDIT_PADDING_M) ** 2:
                    return SurfaceClearance(False, float(np.sqrt(closest)), subdivisions,
                                            "surface witness below clearance")
                if closest < max(0.0, clearance_m - _AUDIT_PADDING_M) ** 2:
                    return SurfaceClearance(False, float(np.sqrt(closest)), subdivisions,
                                            "surface witness below requested bound")
            if _depth >= max_depth or subdivisions + len(left) > max_subdivisions:
                return SurfaceClearance(False, clearance_m, subdivisions,
                                        "subdivision limit; separation unproven")
            lengths_a = np.square(np.roll(left, -1, axis=1) - left).sum(axis=2)
            lengths_b = np.square(np.roll(right, -1, axis=1) - right).sum(axis=2)
            if np.any((lengths_a.max(axis=1) == 0) & (lengths_b.max(axis=1) == 0)):
                return SurfaceClearance(False, clearance_m, subdivisions,
                                        "degenerate points; separation unproven")
            use_left = lengths_a.max(axis=1) >= lengths_b.max(axis=1)
            selected = np.where(use_left[:, None, None], left, right)
            longest = np.where(use_left, lengths_a.argmax(axis=1), lengths_b.argmax(axis=1))
            row = np.arange(len(selected))
            p = selected[row, longest]
            q = selected[row, (longest + 1) % 3]
            v = selected[row, (longest + 2) % 3]
            midpoint = (p + q) / 2
            first = np.stack([p, midpoint, v], axis=1)
            second = np.stack([midpoint, q, v], axis=1)
            left = np.concatenate((np.where(use_left[:, None, None], first, left),
                                   np.where(use_left[:, None, None], second, left)))
            right = np.concatenate((np.where(use_left[:, None, None], right, first),
                                    np.where(use_left[:, None, None], right, second)))
            subdivisions += len(selected)
        raise AssertionError("unreachable subdivision loop")

    def certify_rotation(self, triangles: np.ndarray, pivot: np.ndarray,
                         axis: np.ndarray, half_angle: float, clearance_m: float,
                         *, max_subdivisions: int = 500_000) -> bool:
        """保守认证绕固定轴旋转区间；无法证明时返回 False。

        输入三角面处于旋转区间的中心位置，旋转轴与三角面同坐标系。
        分离方向固定在每对细分三角形的中心连线上，对正弦轨迹取区间上界。
        """
        moving = np.asarray(triangles, dtype=np.float64)
        pivot = np.asarray(pivot, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        if (moving.ndim != 3 or moving.shape[1:] != (3, 3) or not len(moving)
                or pivot.shape != (3,) or axis.shape != (3,)
                or not np.isfinite(moving).all() or not np.isfinite(pivot).all()
                or not np.isfinite(axis).all() or not np.isfinite(half_angle)
                or not np.isfinite(clearance_m) or clearance_m < 0
                or not isinstance(max_subdivisions, (int, np.integer)) or max_subdivisions < 0
                or half_angle < 0 or half_angle > np.pi / 2
                or abs(float(np.linalg.norm(axis)) - 1.0) > 1e-6):
            return False
        sin_angle = np.sin(half_angle)
        cos_delta = 1.0 - np.cos(half_angle)
        rel = moving - pivot
        axial = np.einsum("nvi,i->nv", rel, axis)[..., None] * axis
        radial = rel - axial
        tangent = np.cross(axis, rel)
        expansion = np.abs(tangent) * sin_angle + np.abs(radial) * cos_delta
        threshold = clearance_m + _AUDIT_PADDING_M
        candidates = self._candidates((moving - expansion).min(axis=1) - threshold,
                                       (moving + expansion).max(axis=1) + threshold)
        if candidates is None:
            return False
        indices, ids = candidates
        if not len(ids):
            return True
        left = self.triangles[ids]
        right = moving[indices]
        subdivisions = 0
        for _ in range(60):
            direction = right.mean(axis=1) - left.mean(axis=1)
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-16)
            left_proj = np.einsum("ni,nvi->nv", direction, left)
            right_proj = np.einsum("ni,nvi->nv", direction, right)
            rel = right - pivot
            axial = np.einsum("nvi,i->nv", rel, axis)[..., None] * axis
            radial = rel - axial
            tangent = np.cross(axis, rel)
            variation = (np.abs(np.einsum("ni,nvi->nv", direction, tangent)) * sin_angle
                         + np.abs(np.einsum("ni,nvi->nv", direction, radial)) * cos_delta)
            gap = np.min(right_proj - variation, axis=1) - np.max(left_proj, axis=1)
            unresolved = gap < threshold
            left, right = left[unresolved], right[unresolved]
            if not len(left):
                return True
            if len(left) > 200_000 or subdivisions + len(left) > max_subdivisions:
                return False
            lengths_a = np.square(np.roll(left, -1, axis=1) - left).sum(axis=2)
            lengths_b = np.square(np.roll(right, -1, axis=1) - right).sum(axis=2)
            if np.any((lengths_a.max(axis=1) == 0) & (lengths_b.max(axis=1) == 0)):
                return False
            use_left = lengths_a.max(axis=1) >= lengths_b.max(axis=1)
            selected = np.where(use_left[:, None, None], left, right)
            longest = np.where(use_left, lengths_a.argmax(axis=1), lengths_b.argmax(axis=1))
            row = np.arange(len(selected))
            p, q, v = (selected[row, (longest + offset) % 3] for offset in range(3))
            midpoint = (p + q) / 2
            first = np.stack([p, midpoint, v], axis=1)
            second = np.stack([midpoint, q, v], axis=1)
            left = np.concatenate((np.where(use_left[:, None, None], first, left),
                                   np.where(use_left[:, None, None], second, left)))
            right = np.concatenate((np.where(use_left[:, None, None], right, first),
                                    np.where(use_left[:, None, None], right, second)))
            subdivisions += len(selected)
        return False


def _triangle_bounds(triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回三角形的轴对齐外包盒。"""
    return triangle.min(axis=0), triangle.max(axis=0)


def _box_distance(a: np.ndarray, b: np.ndarray) -> float:
    """返回两三角形包围盒的欧氏间距下界。"""
    alo, ahi = _triangle_bounds(a)
    blo, bhi = _triangle_bounds(b)
    return float(np.linalg.norm(np.maximum(np.maximum(alo - bhi, blo - ahi), 0)))


def _split_triangle(triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """沿最长边二分，两个子三角形共同覆盖原三角形。"""
    lengths = np.sum((np.roll(triangle, -1, axis=0) - triangle) ** 2, axis=1)
    index_longest = int(np.argmax(lengths))
    a, b, c = (triangle[(index_longest + i) % 3] for i in range(3))
    midpoint = (a + b) / 2
    return np.asarray([a, midpoint, c]), np.asarray([midpoint, b, c]), float(lengths[index_longest])


def certify_surface_clearance(
    triangles_a: np.ndarray,
    triangles_b: np.ndarray,
    clearance_m: float,
    *,
    max_subdivisions: int = 2_000_000,
    max_depth: int = 60,
) -> SurfaceClearance:
    """证明两组三角表面至少相距指定米数；无法证明时返回失败。

    空或非有限输入抛出 ValueError。允许退化三角形，若无法有效二分则保守拒绝。
    使用 R 树枚举所有可能比阈值更近的包围盒，再沿最长边递归二分；
    浮点裕量使参与比较的阈值比调用者所需值高 1 微米。
    """
    a = np.asarray(triangles_a, dtype=np.float64)
    b = np.asarray(triangles_b, dtype=np.float64)
    if (a.ndim != 3 or b.ndim != 3 or a.shape[1:] != (3, 3)
            or b.shape[1:] != (3, 3) or not len(a) or not len(b)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("triangle arrays must be non-empty, finite, and shaped (N, 3, 3)")
    if not np.isfinite(clearance_m) or clearance_m < 0 or max_depth < 0 or max_subdivisions < 0:
        raise ValueError("invalid clearance or subdivision limit")
    threshold = float(clearance_m) + _AUDIT_PADDING_M
    try:
        from rtree import index
    except ImportError as exc:
        raise RuntimeError("网格表面审计需要安装可选依赖 rtree") from exc
    properties = index.Property()
    properties.dimension = 3
    tree = index.Index(interleaved=True, properties=properties)
    for i, triangle in enumerate(b):
        low, high = _triangle_bounds(triangle)
        tree.insert(i, tuple(np.r_[low, high]))
    stack = []
    for triangle in a:
        low, high = _triangle_bounds(triangle)
        for candidate in tree.intersection(tuple(np.r_[low - threshold, high + threshold])):
            other = b[candidate]
            if _box_distance(triangle, other) < threshold:
                stack.append((triangle, other, 0))
    subdivisions = 0
    while stack:
        left, right, depth = stack.pop()
        if depth >= max_depth or subdivisions >= max_subdivisions:
            return SurfaceClearance(False, clearance_m, subdivisions, "subdivision limit; separation unproven")
        left_first, left_second, left_length = _split_triangle(left)
        right_first, right_second, right_length = _split_triangle(right)
        if max(left_length, right_length) == 0:
            return SurfaceClearance(False, clearance_m, subdivisions, "degenerate points too close")
        children = ((left_first, right), (left_second, right)) if left_length >= right_length else ((left, right_first), (left, right_second))
        subdivisions += 1
        for first, second in children:
            if _box_distance(first, second) < threshold:
                stack.append((first, second, depth + 1))
    return SurfaceClearance(True, clearance_m, subdivisions, "all triangle AABBs separated")


def audit_rm75b_home(clearance_m: float = 0.01005) -> dict:
    """认证所有有效连杆对的 home 表面下界，并测量最近候选的表面上界。"""
    try:
        import pyvista as pv
        import vtk
    except ImportError as exc:
        raise RuntimeError("网格上界审计需要安装 sim 可选依赖 pyvista 和 vtk") from exc
    from pyroki.collision import RobotCollision

    from handumi.dataset.episode_solve_cache import digest_file
    from handumi.dataset.screening import _link_transforms
    from handumi.robots.registry import CONFIG_DIR, load_embodiment, yourdfpy_handler

    runtime = load_embodiment("rm75b")
    urdf = runtime.load_urdf()
    transforms = _link_transforms(runtime, runtime.config.home_q)
    model = runtime.solver_cls().robot_collision
    resolver = yourdfpy_handler(runtime.config.pkg_root)
    meshes = {}
    hashes = {}
    for name in model.link_names:
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
        if not len(mesh.faces):
            raise ValueError(f"缺少碰撞网格：{name}")
        mesh.apply_transform(transforms[name])
        meshes[name] = mesh
        hashes[name] = [digest_file(resolver(c.geometry.mesh.filename))
                        for c in urdf.link_map[name].collisions if c.geometry.mesh is not None]
    result = {"robot": "rm75b", "algorithm": "surface-aabb-subdivision-v1",
              "urdf_sha256": digest_file(runtime.config.urdf),
              "robot_config_sha256": digest_file(CONFIG_DIR / "rm75b.yaml"),
              "qpos_rad": runtime.config.home_q.tolist(),
              "required_clearance_m": clearance_m, "float_padding_m": _AUDIT_PADDING_M,
              "mesh_sha256": hashes, "pairs": [],
              "limitation": "三角表面间距；非闭合网格不能证明实体未包含或真实装配间距。"}
    candidates = []
    for i, j in zip(model.active_idx_i, model.active_idx_j):
        name_a, name_b = model.link_names[i], model.link_names[j]
        a, b = meshes[name_a], meshes[name_b]
        lower = float(np.linalg.norm(np.maximum(np.maximum(a.bounds[0] - b.bounds[1], b.bounds[0] - a.bounds[1]), 0)))
        entry = {"links": [name_a, name_b], "aabb_lower_m": lower}
        if lower >= clearance_m + _AUDIT_PADDING_M:
            entry.update(proven=True, method="link_aabb")
        else:
            verdict = certify_surface_clearance(a.triangles, b.triangles, clearance_m)
            entry.update(proven=verdict.proven, method="triangle_aabb_subdivision",
                         subdivisions=verdict.subdivisions, reason=verdict.reason)
        result["pairs"].append(entry)
        candidates.append((lower, name_a, name_b))
    # 最小整体包围盒下界对应的候选提供实际表面点对，作为全局距离上界。
    _, name_a, name_b = min(candidates)
    best = (float("inf"), None, None, None, None)
    for source_name, target_name in ((name_a, name_b), (name_b, name_a)):
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(pv.wrap(meshes[target_name]))
        locator.BuildLocator()
        closest = [0.0, 0.0, 0.0]
        cell, sub, squared = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
        for point in meshes[source_name].vertices:
            locator.FindClosestPoint(point, closest, cell, sub, squared)
            if float(squared) < best[0]:
                best = (float(squared), np.asarray(point).tolist(), closest.copy(), source_name, target_name)
    result["witness"] = {"links": [best[3], best[4]], "upper_m": float(np.sqrt(best[0])),
                         "points_world_m": [best[1], best[2]]}
    result["all_surface_pairs_proven"] = all(pair["proven"] for pair in result["pairs"])
    return result


def main() -> None:
    """以 JSON 输出 RM75B home 网格表面认证结果。"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance-mm", type=float, default=10.05,
                        help="要认证的三角表面间距，单位毫米")
    args = parser.parse_args()
    result = audit_rm75b_home(args.clearance_mm / 1000)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_surface_pairs_proven"]:
        raise SystemExit("至少一组表面间距未能认证；这不等同于发生碰撞。")


if __name__ == "__main__":
    main()
