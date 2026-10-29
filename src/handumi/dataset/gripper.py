"""集中读取新版视觉开度与旧 Feetech 数据，避免把未知净宽视为闭合。"""

from __future__ import annotations

import numpy as np

V2_CAPTURE_SCHEMA = "synchronized_sources_v2"


def read_normalized(table, metadata):
    """返回左右归一化开度；新版缺字段或无效测量直接失败，不回退占位零。"""
    modern = metadata.get("capture_schema") == V2_CAPTURE_SCHEMA
    keys = (
        [f"observation.gripper.{side}.normalized" for side in ("left", "right")]
        if modern
        else [f"observation.feetech.{side}_normalized" for side in ("left", "right")]
    )
    names = table.column_names if hasattr(table, "column_names") else table.keys()
    if not all(key in names for key in keys):
        if modern:
            raise ValueError("新版夹爪数据缺少开度列")
        return None
    values = np.column_stack(
        [np.asarray(table[key], dtype=np.float32).reshape(-1) for key in keys]
    )
    if modern:
        for side in ("left", "right"):
            for suffix in ("valid", "sample_time_ns", "sequence"):
                if f"observation.gripper.{side}.{suffix}" not in names:
                    raise ValueError(f"缺少 {side} 夹爪 {suffix}")
            if side in metadata.get("active_sides", []):
                valid = np.asarray(table[f"observation.gripper.{side}.valid"]).reshape(
                    -1
                )
                if not np.all(valid == 1):
                    raise ValueError("活动夹爪包含无效测量")
                camera = metadata["sources"]["gripper"][side].get("camera")
                for suffix in ("sample_time_ns", "sequence"):
                    measured = np.asarray(
                        table[f"observation.gripper.{side}.{suffix}"]
                    ).reshape(-1)
                    if (
                        not np.isfinite(measured).all()
                        or np.any(measured <= 0)
                        or np.any(np.diff(measured) <= 0)
                    ):
                        raise ValueError("视觉夹爪时间或序号重复/无效")
                    camera_key = f"observation.camera.{camera}.{suffix}"
                    if camera_key not in names or not np.array_equal(
                        measured, np.asarray(table[camera_key]).reshape(-1)
                    ):
                        raise ValueError("夹爪测量与来源相机不是同一帧")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("归一化开度必须为 [0,1] 有限数值")
    return np.clip(values, 0, 1)


def require_physical_width(metadata):
    """需要真实米制净宽的流程必须明确拒绝仅有归一化开度的数据。"""
    if metadata.get("capture_schema") == V2_CAPTURE_SCHEMA:
        for side in metadata["active_sides"]:
            if not metadata["sources"]["gripper"][side]["width_available"]:
                raise ValueError("此数据未测量物理净宽；请选择 normalized 夹爪映射")
