"""定义来源中立的夹爪观测，保留相机帧身份和物理净宽可用性。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class GripperObservation:
    """单侧夹爪观测；缺失净宽由 None 表示，不将零开度冒充毫米值。"""

    normalized: float
    valid: bool
    sample_time_ns: int
    sequence: int
    width_m: float | None = None
    reason: str = ""

    def frame(self, side="right"):
        """生成统一数据列；无效归一化值以有限占位值写入并由 valid 标记。"""
        prefix = f"observation.gripper.{side}"
        return {
            f"{prefix}.normalized": np.array(
                [self.normalized if self.valid else 0], dtype=np.float32
            ),
            f"{prefix}.valid": np.array([int(self.valid)], dtype=np.int64),
            f"{prefix}.sample_time_ns": np.array([self.sample_time_ns], dtype=np.int64),
            f"{prefix}.sequence": np.array([self.sequence], dtype=np.int64),
        }


class GripperProvider(Protocol):
    """由共享图像或独立传感器提供夹爪观测。"""

    def observe(self, sample, *, offset_ns=0) -> GripperObservation:
        """返回与输入样本对应的测量，不重新读取传感器。"""
        ...


class VisualGripperProvider:
    """缓存同一帧的估计结果，防止重复图像推进平滑状态。"""

    def __init__(self, estimator):
        """注入纯视觉估计器，输入图像采用 CameraSample 的 RGB 约定。"""
        self.estimator = estimator
        self._key = None
        self._result = None

    def observe(self, sample, *, offset_ns=0):
        """将共享 RGB 图像转换为 BGR 后估计；时间来自相机而非计算完成时刻。"""
        key = (sample.sequence, sample.capture_time_ns, offset_ns)
        if key != self._key:
            estimate = self.estimator.estimate(sample.image[:, :, ::-1])
            self._result = GripperObservation(
                normalized=0 if estimate is None else estimate.openness,
                valid=estimate is not None,
                sample_time_ns=sample.capture_time_ns + offset_ns,
                sequence=sample.sequence,
                reason="未检测到有效双标签" if estimate is None else "",
            )
            self._key = key
        return self._result


class FeetechGripperProvider:
    """将已有 Feetech 采样器适配为通用单侧观测，保持米制净宽。"""

    def __init__(self, sampler, *, side="right", max_skew_ns=60_000_000):
        """绑定采样器、活动侧及允许的采样时差。"""
        if side not in ("left", "right"):
            raise ValueError("无效夹爪侧")
        self.sampler = sampler
        self.side = side
        self.max_skew_ns = max_skew_ns

    def observe(self, sample, *, offset_ns=0):
        """按图像校正时间读取编码器样本，不修改旧采样器行为。"""
        target = sample.capture_time_ns + offset_ns
        result = self.sampler.sample_at(target)
        if result is None:
            return GripperObservation(0, False, 0, 0, reason="编码器暂无样本")
        valid = (
            result.enabled and abs(result.sample_time_ns - target) <= self.max_skew_ns
        )
        return GripperObservation(
            getattr(result.widths, f"{self.side}_normalized"),
            bool(valid),
            result.sample_time_ns,
            result.sequence,
            width_m=getattr(result.widths, self.side),
        )
