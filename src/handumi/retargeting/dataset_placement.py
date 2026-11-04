"""为整个数据集确定共享底座：生成候选、粗筛并比较连续轨迹。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class EpisodeTargets:
    """一段桌面 TCP 目标和误差余量扩展后的腕心可达区域。"""

    poses: np.ndarray
    wrist: np.ndarray
    inner: float
    outer: float


@dataclass(frozen=True)
class Candidate:
    """一个桌面坐标下的底座位姿及相对参考姿态的水平转角。"""

    pose: np.ndarray
    yaw_index: int


def representative_frames(poses: np.ndarray, position_tolerance: float,
                          rotation_tolerance_deg: float, limit: int = 9) -> list[int]:
    """按归一化位姿距离选首尾及最远点，返回稳定排序的原始帧索引。"""
    count = len(poses)
    if count <= limit:
        return list(range(count))
    rotations = Rotation.from_quat(poses[:, 3:]).as_quat()
    selected = [0, count - 1]
    nearest = np.full(count, np.inf)
    while len(selected) < limit:
        index = selected[-1] if len(selected) > 2 else selected[0]
        distance = np.linalg.norm(poses[:, :3] - poses[index, :3], axis=1) / position_tolerance
        angle = np.rad2deg(2 * np.arccos(np.clip(np.abs(rotations @ rotations[index]), 0, 1)))
        nearest = np.minimum(nearest, np.hypot(distance, angle / rotation_tolerance_deg))
        if len(selected) == 2:
            index = selected[1]
            distance = np.linalg.norm(poses[:, :3] - poses[index, :3], axis=1) / position_tolerance
            angle = np.rad2deg(2 * np.arccos(np.clip(np.abs(rotations @ rotations[index]), 0, 1)))
            nearest = np.minimum(nearest, np.hypot(distance, angle / rotation_tolerance_deg))
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    return sorted(selected)


def _distance_key(candidate: Candidate, reference: np.ndarray) -> tuple:
    """产生稳定的参考位置、相对 yaw 和数值并列排序键。"""
    pose = candidate.pose
    yaw = min(candidate.yaw_index, 8 - candidate.yaw_index)
    return (float(np.linalg.norm(pose[:2] - reference[:2])), yaw,
            float(pose[0]), float(pose[1]), candidate.yaw_index)


def _shoulder(pose: np.ndarray, shoulder_z: float) -> np.ndarray:
    """计算该底座姿态下桌面坐标系中的肩关节点。"""
    return pose[:3] + Rotation.from_quat(pose[3:]).apply([0, 0, shoulder_z])


def _geometry_score(candidate: Candidate, episodes: list[EpisodeTargets],
                    shoulder_z: float, reference: np.ndarray) -> tuple:
    """按腕心环域必要条件评估可能完整通过的段及可行前缀。"""
    shoulder = _shoulder(candidate.pose, shoulder_z)
    full = prefix = 0
    for episode in episodes:
        distances = np.linalg.norm(episode.wrist - shoulder, axis=1)
        passed = (distances <= episode.outer) & (distances >= episode.inner)
        bad = np.flatnonzero(~passed)
        prefix += int(bad[0]) if len(bad) else len(passed)
        full += not len(bad)
    return (-full, -prefix, *_distance_key(candidate, reference))


def generate_candidates(
    reference: np.ndarray,
    episodes: list[EpisodeTargets],
    shoulder_z: float,
    first_frame_xy: Callable[[np.ndarray, int], list[np.ndarray]],
) -> list[Candidate]:
    """生成八个朝向的有界候选，并按几何上界筛至每个朝向四个。"""
    reference = np.asarray(reference, dtype=float)
    reference_rotation = Rotation.from_quat(reference[3:])
    result = []
    for yaw_index in range(8):
        pose = reference.copy()
        pose[3:] = (Rotation.from_euler("z", yaw_index * np.pi / 4) * reference_rotation).as_quat()
        offset = _shoulder(pose, shoulder_z) - pose[:3]
        disks = []
        for episode in episodes:
            first = episode.wrist[0]
            vertical = first[2] - reference[2] - offset[2]
            if abs(vertical) <= episode.outer:
                disks.append((first[:2] - offset[:2], np.sqrt(max(0.0, episode.outer**2 - vertical**2))))
        points = []
        if disks:
            lo = np.min([center - radius for center, radius in disks], axis=0)
            hi = np.max([center + radius for center, radius in disks], axis=0)
            for x in np.linspace(lo[0], hi[0], 9):
                for y in np.linspace(lo[1], hi[1], 9):
                    xy = np.array([x, y])
                    if any(np.linalg.norm(xy - center) <= radius + 1e-9 for center, radius in disks):
                        points.append(xy)
        for episode_index in range(len(episodes)):
            points.extend(first_frame_xy(pose, episode_index))
        unique = {}
        for point in points:
            candidate_pose = pose.copy()
            candidate_pose[:2] = point
            key = tuple(np.round(candidate_pose[:2], 5))
            unique.setdefault(key, Candidate(candidate_pose, yaw_index))
        if yaw_index == 0:
            unique[tuple(np.round(reference[:2], 5))] = Candidate(reference.copy(), 0)
        candidates = sorted(unique.values(), key=lambda c: _geometry_score(c, episodes, shoulder_z, reference))
        chosen = candidates[:4]
        if yaw_index == 0:
            ref = Candidate(reference.copy(), 0)
            if not any(np.array_equal(c.pose, ref.pose) for c in chosen):
                chosen = chosen[:3] + [ref]
        result.extend(chosen)
    return result


def choose_finalists(
    candidates: list[Candidate], reference: np.ndarray, budget: int,
    evaluate_samples: Callable[[Candidate], tuple[int, int]],
) -> tuple[list[Candidate], list[dict]]:
    """根据代表帧评分选候选，保留参考位姿并优先覆盖不同朝向。"""
    ranked = []
    for candidate in candidates:
        full, frames = evaluate_samples(candidate)
        ranked.append(((-full, -frames, *_distance_key(candidate, reference)), candidate))
    ranked.sort(key=lambda item: item[0])
    reference_candidate = next(c for _, c in ranked if c.yaw_index == 0 and np.array_equal(c.pose, reference))
    winners = [reference_candidate]
    per_yaw = []
    seen_yaw = set()
    for item in ranked:
        if item[1].yaw_index not in seen_yaw:
            seen_yaw.add(item[1].yaw_index)
            per_yaw.append(item)
    for _, candidate in per_yaw + ranked:
        if len(winners) >= budget:
            break
        if all(not (candidate.yaw_index == chosen.yaw_index and np.array_equal(candidate.pose, chosen.pose)) for chosen in winners):
            winners.append(candidate)
    records = [{"pose7": candidate.pose.tolist(), "yaw_index": candidate.yaw_index,
                "full_sample_episodes": -key[0], "passed_sample_frames": -key[1]}
               for key, candidate in ranked]
    return winners, records


def rank_result(candidate: Candidate, reference: np.ndarray, results: list[dict]) -> tuple:
    """按完整成功段、连续前缀、参考距离和稳定顺序排列完整评分。"""
    complete = sum(bool(result["passed"]) for result in results)
    prefix = sum(int(result["prefix_frames"]) for result in results)
    return (-complete, -prefix, *_distance_key(candidate, reference))
