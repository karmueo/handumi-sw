"""加载机器人末端链接到真实工具 TCP 的本地几何标定。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from handumi.robots.registry import RESOURCE_ROOT
from handumi.robots.utils import mat_to_pose7, pose_inv, pose_mul

LOCAL_ROBOT_TCP_DIR = RESOURCE_ROOT / "configs" / "calibration" / "robot_tcp" / "local"


def rigid_transform(value: Any) -> np.ndarray:
    """验证并返回米制齐次刚体矩阵，非法矩阵抛出 ValueError。"""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("ee_to_tcp 必须是有限的 4×4 矩阵")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("ee_to_tcp 必须是有效的 SE(3) 刚体变换")
    return matrix


@dataclass(frozen=True)
class RobotTcpCalibration:
    """一份已验证的末端到工具变换及其来源。"""

    robot: str
    path: Path
    sha256: str
    verified: bool
    matrices: dict[str, np.ndarray]
    ee_links: dict[str, str]
    sources: dict[str, Any]

    def ee_to_tcp_pose(self, side: str) -> np.ndarray:
        """返回指定机械臂的链接到 TCP 位姿。"""
        return mat_to_pose7(self.matrices[side])

    def target_ee(self, side: str, tcp_pose: np.ndarray) -> np.ndarray:
        """将 base 下 TCP 目标转换为链接目标。"""
        return pose_mul(tcp_pose, pose_inv(self.ee_to_tcp_pose(side)))

    def achieved_tcp(self, side: str, ee_pose: np.ndarray) -> np.ndarray:
        """将 base 下链接 FK 转换为实际 TCP 位姿。"""
        return pose_mul(ee_pose, self.ee_to_tcp_pose(side))

    def metadata(self) -> dict[str, Any]:
        """返回可序列化的快照来源和实际生效矩阵。"""
        from handumi.dataset.paths import portable_path

        sources = {
            name: (
                {**value, "path": portable_path(value.get("path"))}
                if isinstance(value, dict) and "path" in value
                else value
            )
            for name, value in self.sources.items()
        }
        return {
            "path": portable_path(self.path),
            "sha256": self.sha256,
            "robot": self.robot,
            "verified": self.verified,
            "arms": {
                side: {"ee_link": self.ee_links[side], "ee_to_tcp": matrix.tolist()}
                for side, matrix in self.matrices.items()
            },
            "sources": sources,
        }


def load_robot_tcp_calibration(
    robot: str, arms: dict[str, Any], *, explicit_path: Path | None = None
) -> RobotTcpCalibration:
    """按显式路径或本地约定路径加载并验证机器人 TCP 快照。"""
    path = (
        Path(explicit_path).expanduser()
        if explicit_path is not None
        else LOCAL_ROBOT_TCP_DIR / f"{robot}.yaml"
    )
    if not path.is_file():
        raise SystemExit(f"Missing robot TCP calibration: {path}")
    try:
        raw = path.read_bytes()
        data = yaml.safe_load(raw)
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or data.get("kind") != "handumi_robot_tcp_calibration"
        ):
            raise ValueError(
                "需要 schema_version: 1 和 kind: handumi_robot_tcp_calibration"
            )
        if data.get("robot") != robot:
            raise ValueError(f"机器人应为 {robot!r}，实际为 {data.get('robot')!r}")
        entries = data["calibration"]["arms"]
        matrices = {}
        links = {}
        for side, arm in arms.items():
            entry = entries[side]
            if entry["ee_link"] != arm.ee_link:
                raise ValueError(f"{side} 末端链接应为 {arm.ee_link!r}")
            matrices[side] = rigid_transform(entry["ee_to_tcp"])
            links[side] = str(entry["ee_link"])
        sources = data.get("sources") or {}
        if not isinstance(sources, dict):
            raise TypeError("sources 必须为映射")
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise SystemExit(f"Invalid robot TCP calibration {path}: {exc}") from exc
    return RobotTcpCalibration(
        robot,
        path,
        hashlib.sha256(raw).hexdigest(),
        data.get("verified") is True,
        matrices,
        links,
        sources,
    )
