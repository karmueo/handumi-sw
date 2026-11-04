"""根据机器人配置封装单臂与双臂 IK，约束关节输出并计算末端误差。"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk


@dataclass(frozen=True)
class KinematicsConfig:
    """Position-dominant IK weights."""

    pos_weight: float = 100.0
    ori_weight: float = 15.0
    rest_weight: float = 2.0
    posture_weight: float = 0.0
    # Soft margin inside the joint limits. The hard limit constraint lets a
    # joint sit exactly on its stop at no cost, so a wrist that reaches a
    # target by folding to its limit stays there; this penalizes the last
    # ``limit_margin_rad`` before each limit so the solver spends other joints
    # first. Joints whose range is too small for the margin (gripper fingers)
    # are exempt.
    limit_margin_weight: float = 0.0
    limit_margin_rad: float = 0.17453292
    manipulability_weight: float = 0.0
    max_joint_delta: float | None = None
    max_reach: float | None = None
    # Self-collision avoidance is opt-in per robot YAML. Residuals are
    # penetration depths in meters, so weights compare against pos_weight.
    self_collision_weight: float = 0.0
    self_collision_margin: float = 0.01
    # "all" keeps every non-adjacent capsule pair; "inter-arm" restricts the
    # cost to left_*-vs-right_* pairs (intra-arm safety is the vendor's joint
    # limits' job) -- a large solve-time saving for bimanual robots.
    self_collision_pairs: str = "all"
    # 启用时每次 IK 都纳入碰撞代价，不根据上一帧距离跳过。
    self_collision_always: bool = False
    # Frames whose capsules all clear the margins by this distance skip the
    # collision costs entirely and use the legacy fast solve. Soft bound: a
    # commanded step could theoretically close more than this in one frame,
    # but retargeted human motion at 30 fps moves ~1-2 cm per frame.
    collision_activation_distance: float = 0.10

    @property
    def collision_enabled(self) -> bool:
        return self.self_collision_weight > 0.0


def limit_joint_delta(
    q_current: np.ndarray,
    q_target: np.ndarray,
    max_delta: float | None,
) -> np.ndarray:
    """Limit each joint's change while preserving the solver's direction."""
    current = np.asarray(q_current, dtype=np.float32)
    target = np.asarray(q_target, dtype=np.float32)
    if max_delta is None:
        return target
    if max_delta <= 0.0:
        raise ValueError("max_joint_delta must be > 0")
    delta = np.clip(target - current, -float(max_delta), float(max_delta))
    return (current + delta).astype(np.float32)


def posture_terms(posture_q, posture_weight: float):
    """Split a posture vector into the rest pose and per-joint weights.

    A NaN entry means "no preference for this joint": its weight is zero and
    the rest value is irrelevant. This lets a robot bias only the joints that
    select an IK branch (base yaw, shoulder, elbow) while the wrist stays free
    to satisfy the orientation target.
    """
    posture = np.asarray(posture_q, dtype=np.float32)
    free = np.isnan(posture)
    weights = np.where(free, 0.0, float(posture_weight)).astype(np.float32)
    return jnp.array(np.where(free, 0.0, posture)), jnp.array(weights)


def posture_seed(posture_q, fallback_q) -> np.ndarray:
    """The posture with its free (NaN) joints filled from ``fallback_q``."""
    posture = np.asarray(posture_q, dtype=np.float32)
    fallback = np.asarray(fallback_q, dtype=np.float32)
    return np.where(np.isnan(posture), fallback, posture).astype(np.float32)


def _limit_margin_residual(vals, joint_var, lower, upper, weights):
    q = vals[joint_var]
    inside = jnp.maximum(0.0, lower - q) + jnp.maximum(0.0, q - upper)
    return (inside * weights).flatten()


_limit_margin_cost = jaxls.Cost.factory(_limit_margin_residual)


def limit_margin_terms(robot: pk.Robot, config: KinematicsConfig):
    """Shrunk limits and per-joint weights for the limit-margin cost.

    Returns ``None`` when the cost is off. A joint whose range cannot hold two
    margins plus some travel (the mirrored gripper fingers) gets weight 0.
    """
    if config.limit_margin_weight <= 0.0:
        return None
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float32)
    margin = float(config.limit_margin_rad)
    usable = (upper - lower) > 3.0 * margin
    weights = np.where(usable, config.limit_margin_weight, 0.0).astype(np.float32)
    return (
        jnp.array(np.where(usable, lower + margin, lower)),
        jnp.array(np.where(usable, upper - margin, upper)),
        jnp.array(weights),
    )


def _posture_cost(JointVar, posture_q, posture_weight):
    """Weak pull toward a fixed nominal posture.

    The rest cost anchors each solve to the previous frame, which keeps the
    trajectory smooth but expresses no preference between IK branches: once
    the solver folds an elbow the wrong way it stays there, and when that
    saturates the wrist it swings the base instead. This term breaks the tie
    toward the posture a teleoperator would hold (elbow forward, wrist
    pitched down), and its weight is kept far below the pose weights so it
    chooses between equivalent solutions rather than distorting the TCP.
    """
    return pk.costs.rest_cost(JointVar(0), rest_pose=posture_q, weight=posture_weight)


def _stabilize_cost_order_names(costs) -> None:
    """为本次工厂闭包指定稳定名称，避免代价组按内存地址排序。"""
    from types import FunctionType

    def function_name(func):
        """提取底层函数名称；非工厂函数保留其本身的标识。"""
        if not isinstance(func, FunctionType):
            return "none"
        for cell in func.__closure__ or ():
            value = cell.cell_contents
            if isinstance(value, FunctionType):
                return f"{value.__module__}.{value.__qualname__}"
        return f"{func.__module__}.{func.__qualname__}"

    prefix = "Cost.factory.<locals>.decorator.<locals>.inner.<locals>.<lambda>"
    for cost in costs:
        residual = cost.compute_residual
        if not isinstance(residual, FunctionType) or residual.__qualname__ != prefix:
            continue
        # 仅改每次创建的闭包显示名称，不修改底层函数、代价或参数。
        residual.__qualname__ = (
            f"handumi_cost.{function_name(residual)}.{cost.kind}.{cost.jac_mode}."
            f"{cost.jac_batch_size}.{function_name(cost.jac_custom_fn)}."
            f"{function_name(cost.jac_custom_with_cache_fn)}"
        )


@jdc.jit
def _solve(
    robot,
    ee_indices,
    tgt_pos,
    tgt_wxyz,
    q_prev,
    pos_weight,
    ori_weight,
    rest_weight,
    posture_q,
    posture_weight,
    margin_lower,
    margin_upper,
    margin_weights,
    include_posture: jdc.Static[bool] = False,
    include_limit_margin: jdc.Static[bool] = False,
):
    JointVar = robot.joint_var_cls
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(tgt_wxyz), tgt_pos
    )
    batch = target_pose.get_batch_axes()
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch, 0)),
            target_pose,
            ee_indices,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=q_prev,
            weight=rest_weight,
        ),
        pk.costs.limit_constraint(robot, JointVar(0)),
    ]
    if include_posture:
        costs.append(_posture_cost(JointVar, posture_q, posture_weight))
    if include_limit_margin:
        costs.append(
            _limit_margin_cost(JointVar(0), margin_lower, margin_upper, margin_weights)
        )
    _stabilize_cost_order_names(costs)
    # q_prev 必须作为优化初值传入。rest cost 只是软约束，不能代替初值；
    # 否则优化器会从 pyroki 的默认中间位形启动，在奇异 home 附近跳到另一
    # IK 分支，即使目标只移动了毫米级。
    initial_vals = jaxls.VarValues.make([JointVar(0).with_value(q_prev)])
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
        .analyze()
        .solve(
            initial_vals=initial_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


@jdc.jit
def _solve_self_collision(
    robot,
    robot_coll,
    ee_indices,
    tgt_pos,
    tgt_wxyz,
    q_prev,
    pos_weight,
    ori_weight,
    rest_weight,
    self_collision_weight,
    self_collision_margin,
    posture_q,
    posture_weight,
    margin_lower,
    margin_upper,
    margin_weights,
    include_posture: jdc.Static[bool] = False,
    include_limit_margin: jdc.Static[bool] = False,
):
    """The legacy ``_solve`` costs plus a capsule self-collision penalty.

    Kept separate from ``_solve`` so robots without a collision model keep
    the exact problem structure (and compiled cache) they had before.
    """
    JointVar = robot.joint_var_cls
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(tgt_wxyz), tgt_pos
    )
    batch = target_pose.get_batch_axes()
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch, 0)),
            target_pose,
            ee_indices,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=q_prev,
            weight=rest_weight,
        ),
        pk.costs.limit_constraint(robot, JointVar(0)),
    ]
    if include_posture:
        costs.append(_posture_cost(JointVar, posture_q, posture_weight))
    if include_limit_margin:
        costs.append(
            _limit_margin_cost(JointVar(0), margin_lower, margin_upper, margin_weights)
        )
    costs.append(
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            JointVar(0),
            margin=self_collision_margin,
            weight=self_collision_weight,
        )
    )
    _stabilize_cost_order_names(costs)
    # 与 _solve 相同：从上一关节角启动，保持分支连续。
    initial_vals = jaxls.VarValues.make([JointVar(0).with_value(q_prev)])
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
        .analyze()
        .solve(
            initial_vals=initial_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


def solve_bimanual(
    robot: pk.Robot,
    ee_indices,
    tgt_pos,
    tgt_wxyz,
    q_prev=None,
    pos_weight=100.0,
    ori_weight=15.0,
    rest_weight=2.0,
    robot_collision=None,
    config: KinematicsConfig | None = None,
    collision_active: bool = True,
    posture_q=None,
    posture_weight: float = 0.0,
    limit_margin=None,
) -> np.ndarray:
    """Solve two end-effector targets and return the full actuated config.

    ``limit_margin`` is the tuple from :func:`limit_margin_terms` or ``None``.
    """
    nq = robot.joints.num_actuated_joints
    if q_prev is None:
        q_prev = np.zeros(nq, dtype=np.float32)
    include_posture = posture_weight > 0.0 and posture_q is not None
    posture, posture_weights = posture_terms(
        posture_q if posture_q is not None else q_prev, posture_weight
    )
    include_limit_margin = limit_margin is not None
    if limit_margin is None:
        zeros = jnp.zeros(nq, dtype=jnp.float32)
        limit_margin = (zeros, zeros, zeros)
    margin_lower, margin_upper, margin_weights = limit_margin
    if (
        robot_collision is not None
        and config is not None
        and config.collision_enabled
        and collision_active
    ):
        cfg = _solve_self_collision(
            robot,
            robot_collision,
            jnp.array(ee_indices),
            jnp.array(tgt_pos),
            jnp.array(tgt_wxyz),
            jnp.array(q_prev),
            pos_weight,
            ori_weight,
            rest_weight,
            config.self_collision_weight,
            config.self_collision_margin,
            posture,
            posture_weights,
            margin_lower,
            margin_upper,
            margin_weights,
            include_posture=include_posture,
            include_limit_margin=include_limit_margin,
        )
    else:
        cfg = _solve(
            robot,
            jnp.array(ee_indices),
            jnp.array(tgt_pos),
            jnp.array(tgt_wxyz),
            jnp.array(q_prev),
            pos_weight,
            ori_weight,
            rest_weight,
            posture,
            posture_weights,
            margin_lower,
            margin_upper,
            margin_weights,
            include_posture=include_posture,
            include_limit_margin=include_limit_margin,
        )
    return np.array(cfg, dtype=np.float32)


class BimanualKinematicsSolver:
    """Compatibility wrapper around :func:`solve_bimanual`."""

    def __init__(
        self,
        *,
        robot: pk.Robot,
        ee_indices: tuple[int, int],
        arm_joint_indices: dict[str, list[int]] | None = None,
        home_q: np.ndarray,
        config: KinematicsConfig,
        locked_joint_indices: tuple[int, ...] = (),
        robot_collision=None,
        posture_q: np.ndarray | None = None,
    ) -> None:
        self.robot = robot
        self.ee_indices = ee_indices
        self.home_q = np.asarray(home_q, dtype=np.float32)
        # Nominal working posture for the posture cost; home unless the robot
        # YAML declares one (home is usually folded, which is the wrong tie
        # breaker for a working arm).
        self.posture_q = np.asarray(
            home_q if posture_q is None else posture_q, dtype=np.float32
        )
        self.limit_margin = limit_margin_terms(robot, config)
        self.config = config
        self.robot_collision = robot_collision
        self.locked_joint_indices = tuple(locked_joint_indices)
        self._collision_gate = None
        if robot_collision is not None and config.collision_enabled:
            @jax.jit
            def _gate(cfg):
                return robot_collision.compute_self_collision_distance(
                    robot, cfg
                ).min()

            self._collision_gate = _gate
        self.l_ee_idx, self.r_ee_idx = ee_indices
        arm_joint_indices = arm_joint_indices or {}
        self.left_indices = list(
            arm_joint_indices.get("left") or _side_indices(robot, "left")
        )
        self.right_indices = list(
            arm_joint_indices.get("right") or _side_indices(robot, "right")
        )
        self.active_sides = tuple(
            side for side, indices in (("left", self.left_indices), ("right", self.right_indices))
            if indices
        )
        self.left_joint_indices = self.left_indices
        self.right_joint_indices = self.right_indices
        self.l_elbow_idx = -1
        self.r_elbow_idx = -1

    @property
    def num_joints(self) -> int:
        return self.robot.joints.num_actuated_joints

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    def set_posture_pose(self, q: np.ndarray) -> None:
        self.posture_q = np.asarray(q, dtype=np.float32)

    def _collision_active(self, q_prev: np.ndarray) -> bool:
        """Skip the self-collision cost while every capsule pair is far apart.

        Far from every margin, the collision residuals are all zero and the
        costly term buys nothing; the cheap jitted gate picks the legacy solve.
        """
        if self.robot_collision is None or self._collision_gate is None:
            return False
        if self.config.self_collision_always:
            return True
        self_min = self._collision_gate(jnp.asarray(q_prev))
        return float(self_min) <= self.config.collision_activation_distance

    def _with_locked_joints(self, q: np.ndarray) -> np.ndarray:
        if not self.locked_joint_indices:
            return np.asarray(q, dtype=np.float32)
        out = np.asarray(q, dtype=np.float32).copy()
        for index in self.locked_joint_indices:
            out[index] = self.home_q[index]
        return out

    def fk(self, q: np.ndarray) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return jaxlie.SE3(fk[self.l_ee_idx]), jaxlie.SE3(fk[self.r_ee_idx])

    def fk_pose7(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return left/right FK as ``[x, y, z, qx, qy, qz, qw]`` poses."""
        left, right = self.fk(q)
        return se3_to_pose7(left), se3_to_pose7(right)

    def link_positions(self, q: np.ndarray, link_indices: list[int]) -> np.ndarray:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return np.asarray(
            [jaxlie.SE3(fk[index]).translation() for index in link_indices],
            dtype=np.float32,
        )

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """求解有效侧的末端目标，返回满足关节限位及每帧步长限制的关节角。"""
        del left_elbow_pos, right_elbow_pos
        if "left" not in self.active_sides:
            left_pose = None
        if "right" not in self.active_sides:
            right_pose = None
        if left_pose is None and right_pose is None:
            return self._with_locked_joints(q_current)

        q_prev = self._with_locked_joints(q_current)
        if any(
            side in self.active_sides and pose is None
            for side, pose in (("left", left_pose), ("right", right_pose))
        ):
            left_fk, right_fk = self.fk(q_prev)
        else:
            left_fk = right_fk = None
        tgt_pos = []
        tgt_wxyz = []
        target_indices = []
        for side, index, pose, fallback in (
            ("left", self.l_ee_idx, left_pose, left_fk),
            ("right", self.r_ee_idx, right_pose, right_fk),
        ):
            if side not in self.active_sides:
                continue
            target_indices.append(index)
            if pose is None:
                assert fallback is not None
                tgt_pos.append(np.asarray(fallback.translation(), dtype=np.float32))
                tgt_wxyz.append(np.asarray(fallback.rotation().wxyz, dtype=np.float32))
            else:
                pos, rot = pose
                tgt_pos.append(np.asarray(pos, dtype=np.float32))
                tgt_wxyz.append(
                    _pose_rotation_to_wxyz(np.asarray(rot, dtype=np.float32))
                )

        q_target = solve_bimanual(
            self.robot,
            tuple(target_indices),
            np.asarray(tgt_pos, dtype=np.float32),
            np.asarray(tgt_wxyz, dtype=np.float32),
            q_prev=q_prev,
            pos_weight=self.config.pos_weight,
            ori_weight=self.config.ori_weight,
            rest_weight=self.config.rest_weight,
            robot_collision=self.robot_collision,
            config=self.config,
            collision_active=self._collision_active(q_prev),
            posture_q=self.posture_q,
            posture_weight=self.config.posture_weight,
            limit_margin=self.limit_margin,
        )
        # 迭代限位约束可能尚未完全收敛；先投影到模型限位，再限制每帧步长。
        # 上一帧在限位内时，两步处理可同时保证限位与步长约束。
        q_target = np.clip(
            q_target,
            np.asarray(self.robot.joints.lower_limits),
            np.asarray(self.robot.joints.upper_limits),
        )
        q_limited = limit_joint_delta(q_prev, q_target, self.config.max_joint_delta)
        return self._with_locked_joints(q_limited)


def _side_indices(robot: pk.Robot, side: str) -> list[int]:
    return [
        i
        for i, name in enumerate(robot.joints.actuated_names)
        if name.startswith(f"{side}_")
    ]


def se3_to_pose7(transform: jaxlie.SE3) -> np.ndarray:
    """Convert a JAXLie SE3 to ``[x, y, z, qx, qy, qz, qw]``."""
    translation = np.asarray(transform.translation(), dtype=np.float32)
    wxyz = np.asarray(transform.rotation().wxyz, dtype=np.float32)
    quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quat = (quat / norm).astype(np.float32)
    return np.concatenate([translation, quat]).astype(np.float32)


def rotation_error_deg(
    target_pose7: np.ndarray, achieved_pose7: np.ndarray
) -> np.ndarray:
    """Shortest quaternion angular distance between pose7 arrays, in degrees."""
    target_quat = np.asarray(target_pose7, dtype=np.float32)[..., 3:7]
    achieved_quat = np.asarray(achieved_pose7, dtype=np.float32)[..., 3:7]
    target_quat = target_quat / np.maximum(
        np.linalg.norm(target_quat, axis=-1, keepdims=True),
        1e-8,
    )
    achieved_quat = achieved_quat / np.maximum(
        np.linalg.norm(achieved_quat, axis=-1, keepdims=True),
        1e-8,
    )
    dot = np.abs(np.sum(target_quat * achieved_quat, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))).astype(np.float32)


def pose_error_arrays(
    target_left_pose7: np.ndarray,
    target_right_pose7: np.ndarray,
    achieved_left_pose7: np.ndarray,
    achieved_right_pose7: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-frame EE position and orientation errors for both arms."""
    target_left_pose7 = np.asarray(target_left_pose7, dtype=np.float32)
    target_right_pose7 = np.asarray(target_right_pose7, dtype=np.float32)
    achieved_left_pose7 = np.asarray(achieved_left_pose7, dtype=np.float32)
    achieved_right_pose7 = np.asarray(achieved_right_pose7, dtype=np.float32)
    return {
        "left_pos_error_m": np.linalg.norm(
            target_left_pose7[:, :3] - achieved_left_pose7[:, :3], axis=1
        ).astype(np.float32),
        "right_pos_error_m": np.linalg.norm(
            target_right_pose7[:, :3] - achieved_right_pose7[:, :3], axis=1
        ).astype(np.float32),
        "left_rot_error_deg": rotation_error_deg(
            target_left_pose7, achieved_left_pose7
        ),
        "right_rot_error_deg": rotation_error_deg(
            target_right_pose7, achieved_right_pose7
        ),
    }


def optimization_score_from_errors(
    pos_mean_cm: float,
    pos_max_cm: float,
    rot_mean_deg: float,
    rot_max_deg: float,
) -> float:
    """Single scalar useful for comparing IK weight sweeps."""
    return float(
        pos_mean_cm
        + 0.35 * pos_max_cm
        + 0.25 * rot_mean_deg
        + 0.08 * rot_max_deg
    )


def _pose_rotation_to_wxyz(rot: np.ndarray) -> np.ndarray:
    if rot.shape == (3, 3):
        return _rot_3x3_to_wxyz(rot)
    if rot.shape == (4,):
        norm = float(np.linalg.norm(rot))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        xyzw = (rot / norm).astype(np.float32)
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    raise ValueError(
        f"Expected rotation as 3x3 matrix or xyzw quaternion, got {rot.shape}."
    )


def _rot_3x3_to_wxyz(rot: np.ndarray) -> np.ndarray:
    t = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if t > 0.0:
        r = np.sqrt(t + 1.0)
        s = 0.5 / r
        return np.array(
            [
                0.5 * r,
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
            ],
            dtype=np.float32,
        )
    if rot[0, 0] >= rot[1, 1] and rot[0, 0] >= rot[2, 2]:
        r = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) * s,
                0.5 * r,
                (rot[0, 1] + rot[1, 0]) * s,
                (rot[0, 2] + rot[2, 0]) * s,
            ],
            dtype=np.float32,
        )
    if rot[1, 1] >= rot[2, 2]:
        r = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[0, 1] + rot[1, 0]) * s,
                0.5 * r,
                (rot[1, 2] + rot[2, 1]) * s,
            ],
            dtype=np.float32,
        )
    r = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
    s = 0.5 / r
    return np.array(
        [
            (rot[1, 0] - rot[0, 1]) * s,
            (rot[0, 2] + rot[2, 0]) * s,
            (rot[1, 2] + rot[2, 1]) * s,
            0.5 * r,
        ],
        dtype=np.float32,
    )
