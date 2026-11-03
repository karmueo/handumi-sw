"""验证 IK 在迭代约束未完全收敛时仍输出限位内的关节角。"""

from dataclasses import replace

import numpy as np
import pytest

from handumi.robots import kinematics
from handumi.robots.registry import load_embodiment


@pytest.fixture(scope="module")
def rm75b():
    """加载单臂模型，复用其关节上下限与实际运动学接口。"""
    return load_embodiment("rm75b")


@pytest.mark.parametrize("bound", ["lower", "upper"])
@pytest.mark.parametrize("max_delta", [None, 0.02, 0.20])
def test_ik_enforces_limits_and_joint_step(rm75b, monkeypatch, bound, max_delta):
    """模拟求解结果越界，最终关节角应同时满足模型限位和每帧步长。"""
    solver = rm75b.solver_cls(
        config=replace(rm75b.config.ik_weights, max_joint_delta=max_delta)
    )
    lower = np.asarray(rm75b.robot.joints.lower_limits)
    upper = np.asarray(rm75b.robot.joints.upper_limits)
    # 从限位内 0.05 rad 出发，模拟迭代结束时残留 0.1 rad 的越界量。
    current = lower + 0.05 if bound == "lower" else upper - 0.05
    unconverged = lower - 0.1 if bound == "lower" else upper + 0.1
    monkeypatch.setattr(
        kinematics, "solve_bimanual", lambda *args, **kwargs: unconverged.copy()
    )
    _, target = solver.fk_pose7(current)
    actual = solver.ik(current, right_pose=(target[:3], target[3:7]))
    assert np.all(actual >= lower)
    assert np.all(actual <= upper)
    if max_delta is not None:
        assert np.max(np.abs(actual - current)) <= max_delta + 1e-6


def test_single_arm_target_skips_unused_fk(rm75b, monkeypatch):
    """单臂目标完整时不计算回退 FK，输出仍由原有求解路径决定。"""
    solver = rm75b.solver_cls(config=rm75b.config.ik_weights)
    q = rm75b.config.home_q.copy()
    monkeypatch.setattr(solver, "fk", lambda q: pytest.fail("unexpected FK"))
    monkeypatch.setattr(kinematics, "solve_bimanual", lambda *a, **k: q.copy())
    actual = solver.ik(q, right_pose=(np.zeros(3), np.array([0, 0, 0, 1])))
    np.testing.assert_allclose(actual, q)


def test_two_active_arms_missing_target_keeps_fk_fallback(rm75b, monkeypatch):
    """双臂缺少一侧目标时继续读取该侧当前 FK。"""
    solver = rm75b.solver_cls(config=rm75b.config.ik_weights)
    solver.active_sides = ("left", "right")
    q = rm75b.config.home_q.copy()
    original = solver.fk
    calls = []
    def capture_fk(value):
        """记录 FK 后复用真实机器人计算。"""
        calls.append(1)
        return original(value)
    monkeypatch.setattr(solver, "fk", capture_fk)
    captured = []
    def solve(*args, **kwargs):
        """记录送入求解器的双侧目标并保持关节角。"""
        captured.append(args[2])
        return q.copy()
    monkeypatch.setattr(kinematics, "solve_bimanual", solve)
    actual = solver.ik(q, right_pose=(np.zeros(3), np.array([0, 0, 0, 1])))
    assert len(calls) == 1 and captured[0].shape == (2, 3)
    np.testing.assert_allclose(actual, q)
