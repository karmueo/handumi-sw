"""验证 IK 代价组排序不受工厂闭包的进程内地址影响。"""

from __future__ import annotations

import os
import subprocess
import sys

import jaxls
import pyroki as pk

from handumi.robots.kinematics import _stabilize_cost_order_names


def _factory_costs():
    """构造姿态、休止位姿、关节约束与碰撞成本，但不求解。"""
    from pyroki._residuals._pose_residual_analytic_jac import _pose_cost_analytical_jac

    return [
        _pose_cost_analytical_jac(None),
        pk.costs.rest_cost(None),
        pk.costs.limit_constraint(None),
        pk.costs.self_collision_cost(None),
        pk.costs.rest_cost(None),
    ]


def test_cost_names_preserve_callable_and_dedup_signatures():
    """成本闭包、参数和 JAXLS 的去重签名均保持不变。"""
    from jaxls._problem import _get_function_signature

    costs = _factory_costs()
    residuals = [cost.compute_residual for cost in costs]
    signatures = [_get_function_signature(fn) for fn in residuals]
    args = [cost.args for cost in costs]
    _stabilize_cost_order_names(costs)
    assert all(cost.compute_residual is fn for cost, fn in zip(costs, residuals))
    assert all(cost.args is value for cost, value in zip(costs, args))
    assert [_get_function_signature(fn) for fn in residuals] == signatures
    assert signatures[1] == signatures[4]
    assert residuals[1].__qualname__ == residuals[4].__qualname__
    assert len({fn.__qualname__ for fn in residuals[:4]}) == 4


def test_non_factory_function_is_unchanged():
    """库全局函数不会被重命名。"""
    def residual(values):
        """提供一个与 Cost.factory 无关的残差函数。"""
        return values

    cost = jaxls.Cost(compute_residual=residual, args=())
    name = residual.__qualname__
    _stabilize_cost_order_names([cost])
    assert residual.__qualname__ == name and cost.compute_residual is residual
    assert residual(3) == 3


def test_factory_cost_order_stable_across_spawned_processes():
    """不同进程和哈希种子生成相同的代价组语义顺序。"""
    program = '''
import jax
import pyroki as pk
from pyroki._residuals._pose_residual_analytic_jac import _pose_cost_analytical_jac
from handumi.robots.kinematics import _stabilize_cost_order_names
costs = [(_pose_cost_analytical_jac(None), 'pose'), (pk.costs.rest_cost(None), 'rest'), (pk.costs.limit_constraint(None), 'limit'), (pk.costs.self_collision_cost(None), 'collision')]
_stabilize_cost_order_names([cost for cost, _ in costs])
print(','.join(name for cost, name in sorted(costs, key=lambda item: str(jax.tree.structure(item[0])))))
'''
    outputs = []
    for seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.check_output([sys.executable, "-c", program], env=env, text=True)
        outputs.append(result.strip())
    assert len(set(outputs)) == 1
