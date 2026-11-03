"""构建保守自碰撞包络，并验证姿态及关节插值运动。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as np
import pyroki as pk
import trimesh

# 浮点运算及网格读取误差的保守余量，单位米。
_GEOMETRY_PADDING = 1e-5
_NUMERIC_PADDING = 1e-6
COLLISION_VALIDATION_VERSION = "rm75b-mesh-swept-v3"
GRIPPER_BOX_VALIDATION_VERSION = "rm75b-gripper-box-swept-v1"


@jdc.pytree_dataclass
class ColumnBoxCollision:
    """每个网格的柱状分片由局部轴对齐盒包围，用 SAT 投影计算间距下界。"""

    num_links: jdc.Static[int]
    link_names: jdc.Static[tuple[str, ...]]
    active_idx_i: jdc.Static[tuple[int, ...]]
    active_idx_j: jdc.Static[tuple[int, ...]]
    centers: jax.Array
    half_extents: jax.Array
    geom_links: jax.Array
    pair_i: jax.Array
    pair_j: jax.Array
    pair_link: jax.Array

    def compute_self_collision_distance(self, robot, q):
        """返回每个有效连杆对的保守间距下界，可供 IK 与运动认证共享。"""
        poses = robot.forward_kinematics(q)
        rotations = jaxlie.SO3(poses[..., :4]).as_matrix()
        translations = poses[..., 4:]
        box_rotations = rotations[self.geom_links]
        box_centers = jnp.einsum("nij,nj->ni", box_rotations, self.centers) + translations[self.geom_links]
        ia = jnp.asarray(self.active_idx_i)
        ib = jnp.asarray(self.active_idx_j)
        axes_a = jnp.swapaxes(rotations[ia], -1, -2)
        axes_b = jnp.swapaxes(rotations[ib], -1, -2)
        cross = jnp.cross(axes_a[:, :, None, :], axes_b[:, None, :, :]).reshape((-1, 9, 3))
        axes = jnp.concatenate([axes_a, axes_b, cross], axis=1)
        norms_squared = jnp.sum(jnp.square(axes), axis=-1)
        norms = jnp.sqrt(jnp.maximum(norms_squared, 1e-12))
        axes = axes / norms[..., None]
        per_pair_axes = axes[self.pair_link]
        delta = box_centers[self.pair_j] - box_centers[self.pair_i]
        projected = jnp.abs(jnp.einsum("pai,pi->pa", per_pair_axes, delta))
        basis_a = box_rotations[self.pair_i]
        basis_b = box_rotations[self.pair_j]
        dot_a = jnp.abs(jnp.einsum("pik,pai->pka", basis_a, per_pair_axes))
        dot_b = jnp.abs(jnp.einsum("pik,pai->pka", basis_b, per_pair_axes))
        radii_a = jnp.einsum("pk,pka->pa", self.half_extents[self.pair_i], dot_a)
        radii_b = jnp.einsum("pk,pka->pa", self.half_extents[self.pair_j], dot_b)
        gaps = projected - radii_a - radii_b
        gaps = jnp.where(norms_squared[self.pair_link] > 1e-12, gaps, -1e6)
        box_distances = jnp.max(gaps, axis=1)
        return jax.ops.segment_min(box_distances, self.pair_link, num_segments=len(self.active_idx_i))


def _column_boxes(mesh, divisions: int):
    """以局部 Y/Z 平面裁剪三角面，每一柱分片用局部轴盒完整覆盖。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0 or not np.isfinite(vertices).all():
        raise ValueError("RM75B collision mesh is empty or non-finite")
    # 两个切割方向及盒轴均使用局部坐标，第三轴贯穿整个几何。
    axes = np.eye(3)
    edges = [np.linspace(vertices[:, i].min(), vertices[:, i].max(), divisions + 1) for i in (1, 2)]
    boxes = []
    for lo, hi in zip(edges[0][:-1], edges[0][1:]):
        v1, f1, _ = trimesh.intersections.slice_faces_plane(vertices, faces, axes[1], axes[1] * (lo - 1e-8))
        v2, f2, _ = trimesh.intersections.slice_faces_plane(v1, f1, -axes[1], axes[1] * (hi + 1e-8))
        if not len(f2):
            continue
        for lo2, hi2 in zip(edges[1][:-1], edges[1][1:]):
            v3, f3, _ = trimesh.intersections.slice_faces_plane(v2, f2, axes[2], axes[2] * (lo2 - 1e-8))
            v4, f4, _ = trimesh.intersections.slice_faces_plane(v3, f3, -axes[2], axes[2] * (hi2 + 1e-8))
            if not len(f4):
                continue
            points = v4[np.unique(f4)]
            lower, upper = points.min(axis=0), points.max(axis=0)
            boxes.append(((lower + upper) / 2, (upper - lower) / 2 + _GEOMETRY_PADDING))
    if not boxes:
        raise ValueError("RM75B collision mesh produced no conservative boxes")
    return boxes


def build_rm75b_collision_model(urdf):
    """构建覆盖原始网格的柱状盒集合，保留所有非邻接危险连杆对。"""
    from pyroki._robot_urdf_parser import RobotURDFParser

    assembly = ("gripper_link", "gripper_left_finger_link", "gripper_right_finger_link")
    missing = [name for name in assembly if name not in urdf.link_map]
    if missing:
        raise ValueError(f"RM75B collision geometry missing links: {missing}")
    _, info = RobotURDFParser.parse(urdf)
    names = info.names
    boxes = []
    counts = []
    for name in names:
        mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(urdf, name)
        pieces = _column_boxes(mesh, 8 if name in {"Link6", "gripper_link"} else 4)
        boxes.extend(pieces)
        counts.append(len(pieces))
    ignore = tuple(itertools.combinations(assembly, 2))
    link_i, link_j = pk.collision.RobotCollision._compute_collision_pair_indices(names, urdf, ignore, True)
    geom_i, geom_j = pk.collision.RobotCollision._compute_collision_pair_indices(names, urdf, ignore, True, np.asarray(counts))
    geom_links = np.repeat(np.arange(len(names)), counts)
    link_index = {(a, b): i for i, (a, b) in enumerate(zip(link_i, link_j))}
    pairs = np.asarray([link_index[(geom_links[a], geom_links[b])] for a, b in zip(geom_i, geom_j)])
    return ColumnBoxCollision(
        num_links=len(names), link_names=names, active_idx_i=link_i, active_idx_j=link_j,
        centers=jnp.asarray(np.asarray([b[0] for b in boxes]), dtype=jnp.float32),
        half_extents=jnp.asarray(np.asarray([b[1] for b in boxes]), dtype=jnp.float32),
        geom_links=jnp.asarray(geom_links), pair_i=jnp.asarray(geom_i), pair_j=jnp.asarray(geom_j),
        pair_link=jnp.asarray(pairs),
    )


def build_rm75b_gripper_box_model(urdf):
    """合并固定夹爪三个网格为一个盒，并与非邻接机械臂盒配对。"""
    from pyroki._robot_urdf_parser import RobotURDFParser

    _, info = RobotURDFParser.parse(urdf)
    names = info.names
    assembly = ("gripper_link", "gripper_left_finger_link", "gripper_right_finger_link")
    vertices = []
    for name in assembly:
        mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(urdf, name)
        transform = urdf.get_transform(name, "gripper_link")
        vertices.append(np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3])
    points = np.concatenate(vertices)
    if not np.isfinite(points).all() or not len(points):
        raise ValueError("RM75B gripper collision geometry is missing or invalid")
    lower, upper = points.min(axis=0), points.max(axis=0)
    grip_index = names.index("gripper_link")
    parent = next(j.parent for j in urdf.robot.joints if j.child == "gripper_link")
    boxes = []
    geom_links = []
    for link, name in enumerate(names):
        if name in assembly or name == parent:
            continue
        mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(urdf, name)
        pieces = _column_boxes(mesh, 8 if name == "Link6" else 4)
        boxes.extend(pieces)
        geom_links.extend([link] * len(pieces))
    boxes.append(((lower + upper) / 2, (upper - lower) / 2 + _GEOMETRY_PADDING))
    centers = np.asarray([box[0] for box in boxes])
    extents = np.asarray([box[1] for box in boxes])
    geom_links = np.asarray([*geom_links, grip_index])
    # 忽略 gripper_link 的直接固定父连杆，其余机械臂均参与检测。
    arms = [i for i, name in enumerate(names) if name not in assembly and name != parent]
    li, lj, gi, gj, pair = [], [], [], [], []
    for arm in arms:
        indices = np.flatnonzero(geom_links == arm)
        if not len(indices):
            raise ValueError(f"Missing arm collision geometry: {names[arm]}")
        li.append(arm)
        lj.append(grip_index)
        gi.extend(indices.tolist())
        gj.extend([len(geom_links) - 1] * len(indices))
        pair.extend([len(li) - 1] * len(indices))
    return ColumnBoxCollision(
        num_links=len(names), link_names=names, active_idx_i=tuple(li), active_idx_j=tuple(lj),
        centers=jnp.asarray(centers, dtype=jnp.float32), half_extents=jnp.asarray(extents, dtype=jnp.float32),
        geom_links=jnp.asarray(geom_links), pair_i=jnp.asarray(gi), pair_j=jnp.asarray(gj),
        pair_link=jnp.asarray(pair),
    )


@dataclass(frozen=True)
class CollisionResult:
    """记录最小间距及对应连杆；无法证明安全时 safe 为假。"""

    safe: bool
    distance: float
    links: tuple[str, str]
    reason: str


class CollisionChecker:
    """先用盒体下界筛查，再用三角表面复核姿态及连续运动。"""

    def __init__(self, robot, model, urdf, margin: float, max_depth: int = 12):
        """准备姿态距离函数和全链长上界，长度单位为米。"""
        self.robot = robot
        self.model = model
        self.margin = float(margin)
        self.max_depth = max_depth
        # 仅复用此检查器生命周期内完全相同运动段的成功证明。
        self._segment_proofs = {}
        self.indices_i = np.asarray(model.active_idx_i, dtype=int)
        self.indices_j = np.asarray(model.active_idx_j, dtype=int)
        if not len(self.indices_i):
            raise ValueError("Self-collision model contains no active link pairs")
        self._distances = jax.jit(lambda q: model.compute_self_collision_distance(robot, q))
        # 任一盒体点到任一上游转轴的距离不超过全部关节平移长度之和
        # 加该盒体相对连杆原点的最大长度。两连杆各自运动界相加。
        chain_length = sum(
            float(np.linalg.norm(j.origin[:3, 3]))
            for j in urdf.robot.joints if j.origin is not None
        )
        origins = np.asarray(model.centers)
        extents = np.asarray(model.half_extents)
        max_extent = float(np.max(np.linalg.norm(origins, axis=1) + np.linalg.norm(extents, axis=1)))
        self.pair_motion_scale = 2 * (chain_length + max_extent)
        joint_by_name = {joint.name: joint for joint in urdf.robot.joints}
        # 全链长上界适用于 RM75B 转动关节；平移关节会改变下游臂长，
        # 不具备专门上界时禁止复用此认证器。
        unsupported = [name for name in robot.joints.actuated_names
                       if joint_by_name[name].type not in {"revolute", "continuous"}]
        if unsupported:
            raise ValueError(f"Unsupported joints in collision motion bound: {unsupported}")
        self.motion_scales = np.full(len(robot.joints.actuated_names), self.pair_motion_scale)
        # 连杆对在共同祖先处的整体运动不改变其相对距离。
        by_child = {joint.child: joint for joint in urdf.robot.joints}
        joint_indices = {name: i for i, name in enumerate(robot.joints.actuated_names)}
        self._meshes = {}
        self._triangles = {}
        self._trees = {}
        self._representatives = {}
        self._pair_cache = [[] for _ in self.indices_i]
        paths = {}
        for name in model.link_names:
            mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(urdf, name)
            if mesh.is_empty or not np.isfinite(mesh.vertices).all():
                raise ValueError(f"Missing or invalid collision mesh: {name}")
            self._meshes[name] = mesh
            self._triangles[name] = np.asarray(mesh.triangles, dtype=np.float64)
            path = []
            child = name
            while child in by_child:
                joint = by_child[child]
                path.append(joint)
                child = joint.parent
            paths[name] = path
        self._pair_motion_scales = []
        self._rotation_pairs = {}
        for pair_index, (i, j) in enumerate(zip(self.indices_i, self.indices_j)):
            names = (model.link_names[i], model.link_names[j])
            pair_paths = (paths[names[0]], paths[names[1]])
            shared = {joint.name for joint in pair_paths[0]} & {joint.name for joint in pair_paths[1]}
            scales = np.zeros(len(joint_indices), dtype=np.float64)
            for name, path in zip(names, pair_paths):
                radius = float(np.linalg.norm(self._meshes[name].vertices, axis=1).max())
                for joint in path:
                    if joint.name not in shared and joint.name in joint_indices:
                        scales[joint_indices[joint.name]] = radius
                    if joint.origin is not None:
                        radius += float(np.linalg.norm(joint.origin[:3, 3]))
            self._pair_motion_scales.append(scales)
            exclusive = [joint for joint in pair_paths[1]
                         if joint.name not in shared and joint.name in joint_indices]
            # 当第一个连杆是唯一活动关节的父连杆时，其余后代为固定连接，
            # 第二个连杆相对第一个连杆只绕该轴转动，可使用解析扫掠认证。
            if (len(exclusive) == 1 and not any(joint.name not in shared and joint.name in joint_indices
                                                 for joint in pair_paths[0])
                    and exclusive[0].parent == names[0]):
                joint = exclusive[0]
                origin = np.eye(4) if joint.origin is None else np.asarray(joint.origin)
                joint_axis = (np.asarray([1.0, 0.0, 0.0]) if joint.axis is None
                              else np.asarray(joint.axis, dtype=np.float64))
                axis = origin[:3, :3] @ joint_axis
                norm = np.linalg.norm(axis)
                if norm > 0:
                    self._rotation_pairs[pair_index] = (joint_indices[joint.name],
                                                        origin[:3, 3], axis / norm)
        self._pair_motion_scales = np.asarray(self._pair_motion_scales)

    def _check_rotation_segment(self, index: int, a: np.ndarray, b: np.ndarray) -> bool:
        """解析证明单轴相对旋转的完整运动区间。"""
        if index not in self._rotation_pairs:
            return False
        joint_index, pivot, axis = self._rotation_pairs[index]
        half_angle = abs(float(b[joint_index] - a[joint_index])) / 2
        if half_angle > np.pi / 2:
            return False
        names = (self.model.link_names[self.indices_i[index]],
                 self.model.link_names[self.indices_j[index]])
        from handumi.robots.mesh_surface_audit import TriangleSurfaceIndex

        if names[0] not in self._trees:
            self._trees[names[0]] = TriangleSurfaceIndex(self._triangles[names[0]])
        midpoint = np.asarray((a + b) / 2, dtype=np.float32)
        poses = np.asarray(jaxlie.SE3(self.robot.forward_kinematics(jnp.asarray(midpoint))).as_matrix(), dtype=np.float64)
        relative = np.linalg.inv(poses[self.indices_i[index]]) @ poses[self.indices_j[index]]
        triangles = self._triangles[names[1]] @ relative[:3, :3].T + relative[:3, 3]
        return self._trees[names[0]].certify_rotation(
            triangles, pivot, axis, half_angle, self.margin + _NUMERIC_PADDING,
        )

    def _check_pair(self, index: int, q: np.ndarray, coarse: float,
                    required: float) -> CollisionResult:
        """为单个连杆对证明下界；历史证明仅在扣除相对运动界后复用。"""
        links = (self.model.link_names[self.indices_i[index]],
                 self.model.link_names[self.indices_j[index]])
        if coarse >= required:
            return CollisionResult(True, float(coarse), links, "box clearance lower bound")
        scales = self._pair_motion_scales[index]
        cached_lower = None
        for previous, distance in reversed(self._pair_cache[index]):
            lower = distance - float(np.dot(scales, np.abs(q.astype(np.float64) - previous)))
            if lower >= required + max(0.0, distance - required) / 2:
                return CollisionResult(True, lower, links, "cached mesh lower bound")
            if lower >= required and (cached_lower is None or lower > cached_lower):
                cached_lower = lower
        from handumi.robots.mesh_surface_audit import TriangleSurfaceIndex

        if links[1] not in self._trees:
            self._trees[links[1]] = TriangleSurfaceIndex(self._triangles[links[1]])
        poses = np.asarray(jaxlie.SE3(self.robot.forward_kinematics(jnp.asarray(q))).as_matrix(), dtype=np.float64)
        relative = np.linalg.inv(poses[self.indices_j[index]]) @ poses[self.indices_i[index]]
        triangles = self._triangles[links[0]] @ relative[:3, :3].T + relative[:3, 3]
        # 额外留出 50 µm 的认证裕量，支持相邻帧之间的保守运动证明。
        target = max(required, self.margin + 0.00005)
        targets = [target]
        # 真实余量不足 50 µm 时逐次降低目标，保留可供连续运动证明的正裕量。
        while target - required > 1e-7:
            target = (target + required) / 2
            targets.append(target)
        targets.append(required)
        verdict = None
        for target in dict.fromkeys(targets):
            verdict = self._trees[links[1]].certify(
                triangles, target, physical_margin=self.margin, max_subdivisions=600_000,
            )
            if verdict.proven or verdict.reason == "surface witness below clearance":
                break
        assert verdict is not None
        if not verdict.proven:
            if cached_lower is not None and verdict.reason != "surface witness below clearance":
                return CollisionResult(True, cached_lower, links, "cached mesh lower bound")
            return CollisionResult(False, verdict.clearance_m, links, verdict.reason)
        # 对闭合网格检查对方每个连通分量的代表点，避免表面相离但实体嵌套。
        # 开放网格无法可靠定义内部，仍按 URDF 三角表面语义处理。
        try:
            for source, target, transform in (
                (links[0], links[1], relative),
                (links[1], links[0], np.linalg.inv(relative)),
            ):
                if not self._meshes[target].is_watertight:
                    continue
                if source not in self._representatives:
                    components = self._meshes[source].split(only_watertight=False)
                    self._representatives[source] = np.asarray([
                        part.vertices[part.faces[0, 0]] for part in components if len(part.faces)
                    ])
                points = self._representatives[source] @ transform[:3, :3].T + transform[:3, 3]
                if len(points) and np.any(self._meshes[target].contains(points)):
                    return CollisionResult(False, verdict.clearance_m, links,
                                           "mesh component contained in closed link")
        except (ValueError, RuntimeError, ImportError) as exc:
            return CollisionResult(False, verdict.clearance_m, links,
                                   f"containment check unproven: {exc}")
        cache = self._pair_cache[index]
        cache.append((q.astype(np.float64).copy(), verdict.clearance_m))
        if len(cache) > 256:
            del cache[0]
        return CollisionResult(True, verdict.clearance_m, links, "mesh surface clearance lower bound")

    def check_pose(self, q) -> CollisionResult:
        """计算姿态的最小保守包络间距，并返回可读诊断。"""
        q = np.asarray(q, dtype=np.float32)
        if not np.isfinite(q).all():
            return CollisionResult(False, float("nan"), ("?", "?"), "non-finite joint values")
        distances = np.asarray(self._distances(jnp.asarray(q)))
        if not np.isfinite(distances).all():
            return CollisionResult(False, float("nan"), ("?", "?"), "non-finite collision distance")
        minimum = None
        for idx in np.argsort(distances):
            result = self._check_pair(int(idx), q, float(distances[idx]),
                                      self.margin + _NUMERIC_PADDING)
            if not result.safe:
                return result
            if minimum is None or result.distance < minimum.distance:
                minimum = result
        assert minimum is not None
        return minimum

    def check_segment(self, start, end) -> CollisionResult:
        """保守证明线性关节插值的整个区间满足间距；无法证明即失败。"""
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        key = (a.shape, a.tobytes(), b.shape, b.tobytes(), self.margin, self.max_depth)
        cached = self._segment_proofs.get(key)
        if cached is not None:
            return cached
        result = self._check_segment_uncached(a, b)
        if result.safe:
            if len(self._segment_proofs) >= 4096:
                self._segment_proofs.clear()
            self._segment_proofs[key] = result
        return result

    def _check_segment_uncached(self, a, b) -> CollisionResult:
        """对尚未认证的运动段执行保守连续碰撞检查。"""
        first = self.check_pose(a)
        if not first.safe:
            return first
        last = self.check_pose(b)
        if not last.safe:
            return last
        minimum = first if first.distance <= last.distance else last
        # 分连杆对认证，避免腕部间隙被无关的共同祖先运动放大。
        coarse_cache = {}

        def all_coarse(point):
            """同一分段端点的粗筛结果在所有连杆对之间复用。"""
            q = np.asarray(point, dtype=np.float32)
            key = q.tobytes()
            if key not in coarse_cache:
                coarse_cache[key] = np.asarray(self._distances(jnp.asarray(q)))
            return coarse_cache[key]

        for index in range(len(self.indices_i)):
            def pair_at(point):
                """按单个连杆对检查分段端点。"""
                q = np.asarray(point, dtype=np.float32)
                coarse = float(all_coarse(q)[index])
                return self._check_pair(index, q, coarse, self.margin + _NUMERIC_PADDING)

            dl, dr = pair_at(a), pair_at(b)
            stack = [(a, b, dl, dr, 0)]
            while stack:
                left, right, dl, dr, depth = stack.pop()
                if not dl.safe:
                    return dl
                if not dr.safe:
                    return dr
                bound = float(np.dot(self._pair_motion_scales[index], np.abs(right - left)))
                if min(dl.distance, dr.distance) - bound / 2 >= self.margin + _NUMERIC_PADDING:
                    continue
                if self._check_rotation_segment(index, left, right):
                    continue
                if depth >= self.max_depth:
                    return CollisionResult(False, min(dl.distance, dr.distance), dl.links,
                                           "motion clearance unproven (depth limit)")
                mid = (left + right) / 2
                dm = pair_at(mid)
                if not dm.safe:
                    return dm
                if dm.distance < minimum.distance:
                    minimum = dm
                stack.append((left, mid, dl, dm, depth + 1))
                stack.append((mid, right, dm, dr, depth + 1))
        return minimum


class GripperBoxCollisionChecker(CollisionChecker):
    """仅认证完整夹爪盒与非邻接机械臂包络之间的连续间距。"""

    def __init__(self, robot, model, urdf, margin: float, max_depth: int = 12):
        """构建保守运动界，并扩大到覆盖合并盒全部角点。"""
        super().__init__(robot, model, urdf, margin, max_depth)
        # 盒角点可能超出对应网格；以最大半径差放大每个有效关节的界。
        centers = np.asarray(model.centers)
        extents = np.asarray(model.half_extents)
        geom_links = np.asarray(model.geom_links)
        extra = 0.0
        for link in set(geom_links.tolist()):
            mask = geom_links == link
            radius = np.max(np.linalg.norm(centers[mask], axis=1) +
                            np.linalg.norm(extents[mask], axis=1))
            mesh_radius = np.linalg.norm(self._meshes[model.link_names[link]].vertices, axis=1).max()
            extra = max(extra, float(radius - mesh_radius))
        self._pair_motion_scales += (self._pair_motion_scales > 0) * extra
        self._rotation_pairs.clear()

    def _check_rotation_segment(self, index, a, b):
        """禁止使用基于原网格的旋转证明。"""
        return False

    def _check_pair(self, index, q, coarse, required):
        """仅用保守盒间距离下界判断，不能证明时拒绝。"""
        links = (self.model.link_names[self.indices_i[index]],
                 self.model.link_names[self.indices_j[index]])
        return CollisionResult(coarse >= required, float(coarse), links,
                               "box clearance lower bound" if coarse >= required else "box clearance unproven")


def require_clear(result: CollisionResult, *, phase: str, frame: int | str) -> None:
    """拒绝碰撞或无法证明安全的姿态，并输出部件与间距。"""
    if not result.safe:
        raise SystemExit(
            f"Self-collision validation failed: {phase} frame={frame} "
            f"links={result.links[0]}/{result.links[1]} "
            f"distance={result.distance * 1000:.2f}mm: {result.reason}"
        )


def retry_checked_candidate(
    *, solver, checker: CollisionChecker, previous, candidate, left_pose,
    right_pose, assess, frame, check_motion: bool = True,
    strict_fidelity: bool = True,
):
    """重试碰撞和精度检查；非严格精度模式可退回已认证安全的候选。

    assess 返回精度超限说明或 None；无效数值应由 assess 抛出异常拒绝。
    strict_fidelity 为假时，重试耗尽仍可返回首个安全候选，由调用方报告误差。
    所有返回候选均须通过姿态或连续运动碰撞检查。
    """
    from dataclasses import replace

    original = solver.config
    result = None
    error = None
    safe_fallback = None  # 保存首个通过碰撞检查的候选，供非严格模式回退。
    try:
        for weight in (None, 100.0, 300.0, 1000.0):
            if weight is not None:
                solver.config = replace(original, self_collision_weight=weight)
                candidate = solver.ik(previous, left_pose=left_pose, right_pose=right_pose)
            result = (checker.check_segment(previous, candidate) if check_motion
                      else checker.check_pose(candidate))
            error = assess(candidate)
            if result.safe and error is None:
                return candidate
            if result.safe and not strict_fidelity and safe_fallback is None:
                safe_fallback = np.asarray(candidate).copy()
    finally:
        solver.config = original
    if safe_fallback is not None:
        return safe_fallback
    if result is not None and not result.safe:
        require_clear(result, phase="IK output or interpolated motion", frame=frame)
    raise SystemExit(f"IK fidelity threshold exceeded: frame {frame}: {error}")
