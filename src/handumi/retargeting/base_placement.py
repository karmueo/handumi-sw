"""为单条示教搜索固定的桌面平面底座位置，不依赖数据源和界面。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from handumi.robots.utils import pose7_to_mat

# 确定性的有界数值搜索；版本进入求解缓存签名。
ALGORITHM_VERSION = "episode-xy-v4"
START_SEEDS = 9
SLSQP_MAX_ITERATIONS = 200
MAX_ORACLE_CALLS = 112
PRIMARY_ORACLE_CALLS = 68
REFINEMENT_M = 0.002


@dataclass(frozen=True)
class PlacementCandidate:
    """一个固定底座平移及与首帧配套的关节分支。"""

    xy: np.ndarray
    q0: np.ndarray


def first_frame_candidates(runtime, tool, targets, table_from_base, locked=(), *, max_seeds=START_SEEDS):
    """联合求首帧关节与底座平移，按桌面原点距离排序。

    高度与完整姿态以等式约束保持；结果仅为局部候选，仍需全段 IK 验证。
    """
    model = runtime.load_urdf(load_meshes=False)
    names = runtime.joint_names
    home = np.asarray(runtime.config.home_q, dtype=float)
    lower = np.asarray(runtime.robot.joints.lower_limits, dtype=float).copy()
    upper = np.asarray(runtime.robot.joints.upper_limits, dtype=float).copy()
    for index in locked:
        lower[index] = upper[index] = home[index]
    bounds = list(zip(lower, upper, strict=True))
    reference = pose7_to_mat(table_from_base).astype(float)
    sides = tuple(runtime.arms)
    target_matrices = {
        side: pose7_to_mat(targets[side]).astype(float) for side in sides
    }

    def poses(q):
        """计算各活动侧在参考桌面坐标中的工具位姿。"""
        model.update_cfg(dict(zip(names, q, strict=True)))
        return {
            side: reference
            @ model.get_transform(runtime.config.arms[side].ee_link)
            @ tool.matrices[side]
            for side in sides
        }

    def displacement(computed):
        """由第一侧位置差消去未知平面平移。"""
        side = sides[0]
        return target_matrices[side][:2, 3] - computed[side][:2, 3]

    def objective(q):
        """最小化底座在桌面原点的水平距离。"""
        offset = displacement(poses(q))
        xy = reference[:2, 3] + offset
        return float(xy @ xy)

    def constraints(q):
        """约束高度、姿态及双臂共同的水平平移。"""
        computed = poses(q)
        offset = displacement(computed)
        values = []
        for index, side in enumerate(sides):
            actual, target = computed[side], target_matrices[side]
            values.append(actual[2, 3] - target[2, 3])
            values.extend(
                Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
            )
            if index:
                values.extend(actual[:2, 3] + offset - target[:2, 3])
        return np.asarray(values)

    rng = np.random.default_rng(2026)
    seeds = [home]
    sample_lower = np.where(np.isfinite(lower), lower, home - 2 * np.pi)
    sample_upper = np.where(np.isfinite(upper), upper, home + 2 * np.pi)
    for _ in range(max_seeds - 1):
        seeds.append(
            np.where(lower == upper, lower, rng.uniform(sample_lower, sample_upper))
        )
    candidates = []
    for seed in seeds:
        result = minimize(
            objective,
            seed,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": constraints},
            options={"maxiter": SLSQP_MAX_ITERATIONS, "ftol": 1e-10},
        )
        if np.isfinite(result.x).all() and np.max(np.abs(constraints(result.x))) < 1e-5:
            offset = displacement(poses(result.x))
            candidate = PlacementCandidate(
                reference[:2, 3] + offset, np.asarray(result.x)
            )
            if not any(
                np.linalg.norm(candidate.xy - previous.xy) < 1e-5
                and np.linalg.norm(candidate.q0 - previous.q0) < 1e-3
                for previous in candidates
            ):
                candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: float(np.linalg.norm(candidate.xy)))


def search_feasible_placement(
    candidates: list[PlacementCandidate],
    evaluate: Callable[[PlacementCandidate], object | None],
    *,
    first_feasible: bool = False,
) -> tuple[PlacementCandidate, object, dict[str, float | int]]:
    """全段验证候选并沿邻近方向细化，返回搜索内最小通过结果。"""
    calls = 0
    best = None
    seen = set()

    def probe(candidate):
        """在预算内检查候选，禁止重复且只保留全段通过者。"""
        nonlocal calls, best
        key = (tuple(np.round(candidate.xy, 5)), tuple(np.round(candidate.q0, 3)))
        if key in seen or calls >= MAX_ORACLE_CALLS:
            return False
        seen.add(key)
        calls += 1
        result = evaluate(candidate)
        if result is not None:
            if best is None or np.linalg.norm(candidate.xy) < np.linalg.norm(
                best[0].xy
            ):
                best = (candidate, result)
            return True
        return False

    for candidate in candidates:
        probe(candidate)
        if first_feasible and best is not None:
            return best[0], best[1], {"oracle_calls": calls, "oracle_budget": MAX_ORACLE_CALLS,
                                      "search_resolution_m": None, "start_candidates": len(candidates)}
    if not candidates:
        raise ValueError("首帧平面摆放优化未找到满足关节限制和完整姿态的候选。")

    # 从首帧下界向外按 2 cm 半径扫描，包含桌面坐标轴及首帧解方向。
    # 同一半径完整扫描后再停止，避免先遇到远处可行分支就忽略较近方向。
    min_radius = float(np.linalg.norm(candidates[0].xy))
    first_ring = max(0.02, np.ceil(min_radius / 0.02) * 0.02)
    seed_angles = [float(np.arctan2(seed.xy[1], seed.xy[0])) for seed in candidates[:2]]
    angles = seed_angles + [float(angle) for angle in np.arange(8) * np.pi / 4]
    for radius in np.arange(first_ring, first_ring + 0.321, 0.02):
        if best is not None and radius >= np.linalg.norm(best[0].xy):
            break
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            seed = min(
                candidates,
                key=lambda item: np.linalg.norm(item.xy - radius * direction),
            )
            probe(PlacementCandidate(radius * direction, seed.q0))
            if first_feasible and best is not None:
                return best[0], best[1], {"oracle_calls": calls, "oracle_budget": MAX_ORACLE_CALLS,
                                          "search_resolution_m": None, "start_candidates": len(candidates)}
            if calls >= PRIMARY_ORACLE_CALLS:
                break
        if best is not None or calls >= PRIMARY_ORACLE_CALLS:
            break

    # 原有近端细网格失败时，用更宽的粗环查找可行上界，并留预算细化。
    if best is None:
        for radius in np.arange(0.25, 0.651, 0.1):
            for angle in np.arange(8) * np.pi / 4:
                xy = radius * np.array([np.cos(angle), np.sin(angle)])
                seed = min(candidates, key=lambda item: np.linalg.norm(item.xy - xy))
                probe(PlacementCandidate(xy, seed.q0))
                if first_feasible and best is not None:
                    return best[0], best[1], {"oracle_calls": calls, "oracle_budget": MAX_ORACLE_CALLS,
                                              "search_resolution_m": None, "start_candidates": len(candidates)}
                if calls >= MAX_ORACLE_CALLS - 12:
                    break
            if best is not None or calls >= MAX_ORACLE_CALLS - 12:
                break

    if best is None:
        raise ValueError(f"在 {calls} 次全段验证预算内未找到严格达标的底座位置。")

    # 对已通过解进行向内和左右邻角坐标搜索，直至 2 mm 分辨率。
    step = 0.02
    last_tested_step = None
    while step >= REFINEMENT_M and calls < MAX_ORACLE_CALLS:
        if calls + 3 > MAX_ORACLE_CALLS:
            break
        last_tested_step = step
        current = best[0]
        radius = float(np.linalg.norm(current.xy))
        if radius < 1e-9:
            break
        radial = current.xy / radius
        tangent = np.array([-radial[1], radial[0]])
        improved = False
        for direction in (-radial, -radial + tangent * 0.5, -radial - tangent * 0.5):
            trial = PlacementCandidate(current.xy + step * direction, current.q0)
            previous = float(np.linalg.norm(best[0].xy))
            probe(trial)
            improved |= float(np.linalg.norm(best[0].xy)) < previous - 1e-8
        if not improved:
            if step <= REFINEMENT_M:
                break
            step = max(step * 0.5, REFINEMENT_M)
    return (
        best[0],
        best[1],
        {
            "oracle_calls": calls,
            "oracle_budget": MAX_ORACLE_CALLS,
            "search_resolution_m": last_tested_step,
            "start_candidates": len(candidates),
        },
    )
