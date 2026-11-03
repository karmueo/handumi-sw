"""验证 RM75B 胶囊覆盖、危险连杆对及保守轨迹验收。"""

import numpy as np
import pytest

from handumi.robots.registry import load_embodiment
from handumi.robots.self_collision import (
    CollisionChecker,
    CollisionResult,
    require_clear,
    retry_checked_candidate,
)


@pytest.fixture(scope="module")
def rm75b_collision():
    """加载真实 RM75B 碰撞模型供几何验收使用。"""
    runtime = load_embodiment("rm75b")
    solver = runtime.solver_cls()
    checker = CollisionChecker(runtime.robot, solver.robot_collision,
                               runtime.load_urdf(), solver.config.self_collision_margin)
    return runtime, checker


def test_rm75b_model_covers_meshes_and_preserves_dangerous_pairs(rm75b_collision):
    """每个网格三角面的顶点、边和内部采样均落入柱状盒。"""
    from pyroki.collision import RobotCollision

    runtime, checker = rm75b_collision
    model = checker.model
    centers = np.asarray(model.centers)
    half = np.asarray(model.half_extents)
    links = np.asarray(model.geom_links)
    for i, name in enumerate(model.link_names):
        mesh = RobotCollision._get_trimesh_collision_geometries(runtime.load_urdf(), name)
        triangles = np.asarray(mesh.triangles)
        # 每个三角面检查三个顶点、三个边中点及重心，补充闭合盒体体积测试。
        samples = np.concatenate([triangles.reshape(-1, 3), triangles.mean(axis=1),
                                  ((triangles + np.roll(triangles, 1, axis=1)) / 2).reshape(-1, 3)])
        c, h = centers[links == i], half[links == i]
        for start in range(0, len(samples), 1000):
            inside = np.all(np.abs(samples[start:start+1000, None] - c[None]) <= h[None] + 1e-7, axis=2)
            assert np.all(np.any(inside, axis=1)), name
    pairs = {frozenset((model.link_names[i], model.link_names[j]))
             for i, j in zip(checker.indices_i, checker.indices_j)}
    for finger in ("gripper_link", "gripper_left_finger_link", "gripper_right_finger_link"):
        assert frozenset(("Link2", finger)) in pairs
    assert frozenset(("Link6", "gripper_link")) in pairs
    assert frozenset(("base_link", "Link2")) in pairs
    assert frozenset(("gripper_left_finger_link", "gripper_right_finger_link")) not in pairs


def test_home_is_certified_by_mesh_without_pruning(rm75b_collision):
    """home 粗包络误报时以实际网格认证，不自动删除碰撞对。"""
    runtime, checker = rm75b_collision
    checker = CollisionChecker(runtime.robot, checker.model, runtime.load_urdf(), 0.01)
    result = checker.check_pose(runtime.config.home_q)
    assert result.safe and result.links == ("Link6", "gripper_link")
    assert result.distance >= 0.01 and "mesh" in result.reason
    require_clear(result, phase="home", frame="home")


def test_column_boxes_cover_entire_closed_volume():
    """闭合实体中各柱分片的表面与体积均落入保守盒。"""
    import trimesh

    from handumi.robots.self_collision import _column_boxes

    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    boxes = _column_boxes(mesh, 4)
    centers = np.asarray([box[0] for box in boxes])
    half = np.asarray([box[1] for box in boxes])
    points = np.array(np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-1, 1, 17),
                                  np.linspace(-1.5, 1.5, 25))).reshape(3, -1).T
    inside = np.all(np.abs(points[:, None] - centers) <= half + 1e-10, axis=2)
    assert np.all(np.any(inside, axis=1))


def test_sat_jacobian_finite_at_parallel_axes(rm75b_collision):
    """平行盒轴及零姿态均不产生 NaN 梯度。"""
    import jax
    import jax.numpy as jnp

    runtime, checker = rm75b_collision
    jacobian = jax.jit(jax.jacfwd(lambda q: checker.model.compute_self_collision_distance(runtime.robot, q)))
    for q in (runtime.config.home_q, np.zeros_like(runtime.config.home_q)):
        assert np.isfinite(np.asarray(jacobian(jnp.asarray(q)))).all()


def _synthetic_checker(distance, max_depth=12):
    """以一维可验证距离函数构造连续检查器。"""
    checker = CollisionChecker.__new__(CollisionChecker)
    checker._segment_proofs = {}
    checker.margin = 0.01
    checker.max_depth = max_depth
    checker.motion_scales = np.asarray([1.0])
    checker.indices_i = np.asarray([0])
    checker._pair_motion_scales = np.asarray([[1.0]])
    checker._rotation_pairs = {}
    checker._distances = lambda q: np.asarray([distance(float(q[0]))])

    def pose(q):
        """模拟一组胶囊的距离。"""
        value = float(distance(float(q[0])))
        return CollisionResult(value >= 0.010001, value, ("arm", "gripper"), "clearance violation")

    checker.check_pose = pose
    checker._check_pair = lambda index, q, coarse, required: CollisionResult(
        coarse >= required, coarse, ("arm", "gripper"), "clearance violation"
    )
    return checker


def test_segment_detects_collision_between_safe_endpoints():
    """端点无碰撞但中途穿透时必须拒绝。"""
    checker = _synthetic_checker(lambda x: abs(x - 0.5) - 0.1)
    assert checker.check_pose([0]).safe and checker.check_pose([1]).safe
    assert not checker.check_segment([0], [1]).safe


def test_segment_proofs_require_identical_motion_and_parameters():
    """仅完全相同的运动与检查参数可以复用成功证明。"""
    checker = _synthetic_checker(lambda x: 0.1)
    calls = []
    original = checker._check_segment_uncached

    def track(a, b):
        """记录实际执行的连续证明次数。"""
        calls.append(1)
        return original(a, b)

    checker._check_segment_uncached = track
    assert checker.check_segment([0], [0.1]).safe
    assert checker.check_segment([0], [0.1]).safe
    assert len(calls) == 1
    assert checker.check_segment([0], [0.11]).safe
    checker.margin = 0.02
    assert checker.check_segment([0], [0.1]).safe
    assert len(calls) == 3


def test_segment_certifies_safe_motion_and_rejects_unproven():
    """安全区间可通过，达到细分上限仍无法证明时拒绝。"""
    checker = _synthetic_checker(lambda x: 0.1)
    assert checker.check_segment([0], [0.1]).safe
    checker.max_depth = 0
    result = checker.check_segment([0], [1])
    assert not result.safe and "unproven" in result.reason


def test_fine_certification_reduces_requested_margin_without_losing_slack():
    """实际余量介于 1 和 50 µm 时仍返回高于门槛的有效下界。"""
    from types import SimpleNamespace

    from handumi.robots.mesh_surface_audit import SurfaceClearance

    class Tree:
        """模拟真实表面间距 10.01 mm 的证明器。"""
        def certify(self, triangles, target, **kwargs):
            """仅对不高于真实间距的请求给出证明。"""
            return SurfaceClearance(target < 0.01001, target, 0, "bound unproven")

    checker = CollisionChecker.__new__(CollisionChecker)
    checker._segment_proofs = {}
    checker.model = SimpleNamespace(link_names=("a", "b"))
    checker.indices_i, checker.indices_j = np.asarray([0]), np.asarray([1])
    checker._pair_motion_scales = np.asarray([[1.0]])
    checker._pair_cache = [[]]
    checker._trees = {"b": Tree()}
    checker._triangles = {"a": np.zeros((1, 3, 3))}
    checker._meshes = {name: SimpleNamespace(is_watertight=False) for name in ("a", "b")}
    checker.robot = SimpleNamespace(forward_kinematics=lambda q: np.tile([1., 0, 0, 0, 0, 0, 0], (2, 1)))
    checker.margin = 0.01
    result = checker._check_pair(0, np.zeros(1, dtype=np.float32), 0.0, 0.010001)
    assert result.safe and 0.010001 < result.distance < 0.01001
    # 余量仅数微米时，非零运动段仍可用该余量和位移上界证明安全。
    checker.max_depth = 12
    checker._rotation_pairs = {}
    checker._distances = lambda q: np.asarray([0.0])
    assert checker.check_segment(np.asarray([0.0]), np.asarray([4e-6])).safe


def test_retry_enforces_both_collision_and_pose_error():
    """权重逐次提升，仅接收碰撞和精度均合格的限位后结果。"""
    checker = _synthetic_checker(lambda x: x)
    weights = []

    class Solver:
        """记录重试权重并返回不同候选姿态。"""
        def __init__(self):
            """提供可替换配置。"""
            self.config = replace_config(100.0)

        def ik(self, previous, **kwargs):
            """模拟限位后的候选。"""
            weights.append(self.config.self_collision_weight)
            return np.asarray([0.02 if len(weights) == 1 else 0.05])

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Config:
        """仅包含重试过程需要的权重。"""
        self_collision_weight: float

    def replace_config(weight):
        """构造测试配置。"""
        return Config(weight)

    solver = Solver()
    accepted = retry_checked_candidate(
        solver=solver, checker=checker, previous=np.asarray([0.05]), candidate=np.asarray([0.0]),
        left_pose=None, right_pose=None,
        assess=lambda q: "pose error" if q[0] < 0.04 else None,
        frame=3, check_motion=False,
    )
    assert weights == [100.0, 300.0]
    assert accepted[0] == 0.05 and solver.config.self_collision_weight == 100.0


@pytest.mark.parametrize("strict", [False, True])
def test_retry_fallback_only_returns_a_previously_certified_candidate(strict):
    """后续重试碰撞时，仅非严格模式可返回此前安全但超差的候选。"""
    from handumi.robots.kinematics import KinematicsConfig

    checker = _synthetic_checker(lambda x: x)

    class Solver:
        """模拟提高碰撞权重后仍得到不安全结果的求解器。"""

        def __init__(self):
            """保存原始配置和用于验证重试次数的记录。"""
            self.config = KinematicsConfig(self_collision_weight=0.0)
            self.weights = []

        def ik(self, previous, **kwargs):
            """每次重试都返回碰撞候选。"""
            self.weights.append(self.config.self_collision_weight)
            return np.asarray([-0.01])

    solver = Solver()
    original_config = solver.config

    def retry():
        """以安全但超差的初始候选执行真实重试流程。"""
        return retry_checked_candidate(
            solver=solver, checker=checker, previous=np.asarray([0.05]),
            candidate=np.asarray([0.02]), left_pose=None, right_pose=None,
            assess=lambda q: "pose error", frame=3, check_motion=False,
            strict_fidelity=strict,
        )

    if strict:
        with pytest.raises(SystemExit, match="Self-collision validation failed"):
            retry()
    else:
        np.testing.assert_array_equal(retry(), [0.02])
    assert solver.weights == [100.0, 300.0, 1000.0]
    assert solver.config is original_config


def test_retry_checks_candidate_after_joint_step_limit(rm75b_collision, monkeypatch):
    """碰撞检查接收限幅后的输出，而不是未限幅的求解器原始结果。"""
    from dataclasses import replace

    from handumi.robots import kinematics

    runtime, _ = rm75b_collision
    solver = runtime.solver_cls(config=replace(runtime.config.ik_weights, max_joint_delta=0.2))
    previous = runtime.config.home_q.copy()
    unbounded = previous.copy()
    unbounded[0] += 1.0
    monkeypatch.setattr(kinematics, "solve_bimanual", lambda *a, **kw: unbounded.copy())
    observed = []

    class Checker:
        """记录实际输入并模拟碰撞。"""
        def check_pose(self, q):
            """记录已限幅关节并拒绝候选。"""
            observed.append(np.asarray(q).copy())
            return CollisionResult(False, -0.01, ("Link2", "gripper_link"), "collision")

    _, target = solver.fk_pose7(previous)
    pose = (target[:3], target[3:7])
    candidate = solver.ik(previous, right_pose=pose)
    with pytest.raises(SystemExit, match="Link2/gripper_link"):
        retry_checked_candidate(solver=solver, checker=Checker(), previous=previous,
                                candidate=candidate, left_pose=None, right_pose=pose,
                                assess=lambda q: None, frame=7, check_motion=False)
    assert len(observed) == 4
    assert all(abs(float(q[0] - previous[0]) - 0.2) < 1e-6 for q in observed)


def test_rm75b_ik_always_uses_collision_cost(rm75b_collision, monkeypatch):
    """启用全程碰撞代价时不根据上一帧距离跳过检查。"""
    runtime, _ = rm75b_collision
    solver = runtime.solver_cls()
    monkeypatch.setattr(solver, "_collision_gate", lambda q: pytest.fail("距离门控不应执行"))
    assert solver.config.self_collision_always
    assert solver._collision_active(runtime.config.home_q)


def test_gripper_box_covers_entire_opening_range_and_checks_arm():
    """合并盒覆盖双指全行程，且保留腕部等危险碰撞对。"""
    from pyroki.collision import RobotCollision

    from handumi.robots.self_collision import (
        GripperBoxCollisionChecker,
        build_rm75b_gripper_box_model,
    )

    runtime = load_embodiment("rm75b")
    urdf = runtime.load_urdf()
    model = build_rm75b_gripper_box_model(urdf)
    index = model.link_names.index("gripper_link")
    geom = np.flatnonzero(np.asarray(model.geom_links) == index)
    assert len(geom) == 1
    center = np.asarray(model.centers)[geom[0]]
    extent = np.asarray(model.half_extents)[geom[0]]
    finger_joints = (
        "gripper_left_finger_joint",
        "gripper_right_finger_joint",
    )
    for travel in (0.0, 0.018134):
        cfg = np.zeros(len(urdf.actuated_joint_names))
        for name in finger_joints:
            cfg[urdf.actuated_joint_names.index(name)] = travel
        urdf.update_cfg(cfg)
        for name in (
            "gripper_link",
            "gripper_left_finger_link",
            "gripper_right_finger_link",
        ):
            mesh = RobotCollision._get_trimesh_collision_geometries(urdf, name)
            transform = urdf.get_transform(name, "gripper_link")
            points = np.asarray(mesh.vertices) @ transform[:3, :3].T + transform[:3, 3]
            assert np.all(np.abs(points - center) <= extent + 1e-7), name
    urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
    pairs = {(model.link_names[i], model.link_names[j]) for i, j in zip(model.active_idx_i, model.active_idx_j)}
    assert ("Link6", "gripper_link") in pairs
    assert ("Link7", "gripper_link") not in pairs
    checker = GripperBoxCollisionChecker(runtime.robot, model, urdf, 0.001)
    assert not checker._rotation_pairs
    assert not checker._check_pair(0, None, 0.0005, 0.001001).safe
    assert checker._check_pair(0, None, 0.002, 0.001001).safe


def test_prismatic_fingers_use_displacement_motion_bound(rm75b_collision):
    """平移关节按单位位移计，旋转专用扫掠认证不用于平移关节。"""
    runtime, checker = rm75b_collision
    finger_indices = [
        index
        for index, name in enumerate(runtime.joint_names)
        if name in {
            "gripper_left_finger_joint",
            "gripper_right_finger_joint",
        }
    ]
    assert finger_indices == [7, 8]
    left_index, right_index = finger_indices
    for pair_index, (i, j) in enumerate(zip(checker.indices_i, checker.indices_j)):
        names = (checker.model.link_names[i], checker.model.link_names[j])
        scales = checker._pair_motion_scales[pair_index]
        if "gripper_left_finger_link" in names:
            assert scales[left_index] == pytest.approx(1.0)
        if "gripper_right_finger_link" in names:
            assert scales[right_index] == pytest.approx(1.0)
    for joint_index, _, _ in checker._rotation_pairs.values():
        assert joint_index not in finger_indices
    closed = runtime.home_q()
    runtime.set_finger_positions(closed, {"right": 0.0})
    result = checker.check_segment(runtime.home_q(), closed)
    assert result.safe
    assert np.max(np.abs(closed - runtime.home_q())) == pytest.approx(0.018134)


def test_retry_preserves_explicit_finger_opening():
    """碰撞重试期间求解器不得改写调用方指定的夹指开度。"""
    from dataclasses import dataclass

    checker = _synthetic_checker(lambda x: 0.1)
    seen = []
    original_pose = checker.check_pose

    def record_pose(q):
        """记录碰撞检查实际看到的完整候选。"""
        seen.append(np.asarray(q, dtype=np.float32).copy())
        return original_pose(q)

    checker.check_pose = record_pose

    @dataclass(frozen=True)
    class Config:
        """仅包含重试过程需要的权重。"""

        self_collision_weight: float = 0.0

    class Solver:
        """模拟求解器把夹指重置为零，验证外部修正会被重新应用。"""

        config = Config()

        def ik(self, previous, **kwargs):
            """返回错误开度，迫使重试流程重新应用指定值。"""
            return np.asarray([previous[0], 0.0, 0.0], dtype=np.float32)

    def preserve(candidate):
        """恢复录制开度。"""
        candidate = np.asarray(candidate, dtype=np.float32)
        candidate[1:] = 0.018134
        return candidate

    attempts = []

    def assess(candidate):
        """首次超差以触发重试，之后接受。"""
        attempts.append(candidate.copy())
        return "pose error" if len(attempts) == 1 else None

    accepted = retry_checked_candidate(
        solver=Solver(), checker=checker, previous=np.asarray([0.05]),
        candidate=np.asarray([0.05, 0.0, 0.0]), left_pose=None, right_pose=None,
        assess=assess, frame=3, check_motion=False,
        candidate_transform=preserve,
    )
    assert len(seen) == 2
    for q in seen:
        assert q[1] == pytest.approx(0.018134) and q[2] == pytest.approx(0.018134)
    assert accepted[1] == pytest.approx(0.018134) and accepted[2] == pytest.approx(0.018134)
