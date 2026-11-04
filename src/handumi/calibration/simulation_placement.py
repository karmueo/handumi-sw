"""根据示教 TCP 和机器人 HOME 位姿估计可复现的水平仿真布局。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def fit_simulation_placement(
    trajectories: dict[str, np.ndarray], home_poses: dict[str, np.ndarray]
) -> np.ndarray:
    """估计桌面到机器人的 XY 平移及 yaw，保持 Z=0、桌面法向朝上。

    双臂优先对齐左右轨迹中心连线；单臂按工具姿态的水平分量对齐。
    将轨迹中心置于 HOME 工作区附近，仅提供初始布局，可达性由后续 IK 检查。
    输入为每侧 N×7 的 TCP 位姿和对应 HOME 位姿；无有效数据时抛出 ValueError。
    """
    centers = []  # 每侧录制轨迹的稳健中心。
    homes = []  # 对应机械臂的 HOME 位置。
    rotations = []  # 工具朝向到 HOME 朝向的旋转差。
    for side, home in home_poses.items():
        poses = np.asarray(trajectories[side], dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f"Invalid {side} TCP trajectory shape: {poses.shape}")
        valid = np.isfinite(poses).all(axis=1) & (np.linalg.norm(poses[:, 3:], axis=1) > 1e-8)
        poses = poses[valid]
        if not len(poses):
            raise ValueError(f"No valid {side} TCP poses for simulation placement.")
        home = np.asarray(home, dtype=float)
        if home.shape != (7,) or not np.isfinite(home).all() or np.linalg.norm(home[3:]) <= 1e-8:
            raise ValueError(f"Invalid {side} HOME pose for simulation placement.")
        centers.append(np.median(poses[:, :3], axis=0))
        homes.append(home[:3])
        rotations.extend((Rotation.from_quat(home[3:]) * Rotation.from_quat(poses[:, 3:]).inv()).as_matrix())
    if not centers:
        raise ValueError("No active arms for simulation placement.")
    centers = np.asarray(centers)
    homes = np.asarray(homes)
    source_xy = centers[:, :2] - centers[:, :2].mean(axis=0)  # 去中心的示教位置。
    target_xy = homes[:, :2] - homes[:, :2].mean(axis=0)  # 去中心的 HOME 位置。
    cross = float(np.sum(source_xy[:, 0] * target_xy[:, 1] - source_xy[:, 1] * target_xy[:, 0]))
    dot = float(np.sum(source_xy * target_xy))
    if np.hypot(cross, dot) < 1e-8:
        mean_rotation = np.mean(rotations, axis=0)  # 单臂或退化双臂的姿态参考。
        cross = float(mean_rotation[1, 0] - mean_rotation[0, 1])
        dot = float(mean_rotation[0, 0] + mean_rotation[1, 1])
    yaw = np.arctan2(cross, dot) if np.hypot(cross, dot) > 1e-8 else 0.0
    rotation = Rotation.from_euler("z", yaw)  # 只绕桌面法向旋转。
    translation = homes.mean(axis=0) - rotation.apply(centers).mean(axis=0)
    translation[2] = 0.0
    return np.r_[translation, rotation.as_quat()].astype(np.float32)
