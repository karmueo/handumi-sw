"""定义来源中立的夹爪观测，保留相机帧身份和物理净宽可用性。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from handumi.feetech.diagnostics import log_diagnostic


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
    """缓存同一帧的估计结果，并在短暂漏检时保持最近成功开度。"""

    def __init__(self, estimator, *, hold_timeout_ns=500_000_000):
        """注入纯视觉估计器，默认在 500 毫秒漏检期内沿用上一开度。"""
        self.estimator = estimator
        self.hold_timeout_ns = hold_timeout_ns
        self._key = None
        self._result = None
        # 最近一次成功估计的开度及其相机采集时间；保持期间不刷新。
        self._last_openness = None
        self._last_success_ns = None
        # 上一输入帧的相机采集时间，用于识别时间倒退。
        self._last_frame_ns = None

    def observe(self, sample, *, offset_ns=0):
        """将共享 RGB 图像转换为 BGR 后估计；漏检时按相机时间保持旧值。"""
        key = (sample.sequence, sample.capture_time_ns, offset_ns)
        if key == self._key:
            return self._result
        capture_time_ns = sample.capture_time_ns
        # 时间倒退可能来自时钟重置，跨时间段复用旧开度会掩盖真实变化。
        if self._last_frame_ns is not None and capture_time_ns < self._last_frame_ns:
            self._last_openness = None
            self._last_success_ns = None
        estimate = self.estimator.estimate(sample.image[:, :, ::-1])
        detected = estimate is not None
        sample_time_ns = capture_time_ns + offset_ns
        age_ns = None
        if detected:
            self._last_openness = estimate.openness
            self._last_success_ns = capture_time_ns
            normalized = estimate.openness
            valid = True
            outcome = "measured"
            reason = ""
            age_ns = 0
        elif (
            self.hold_timeout_ns > 0
            and self._last_success_ns is not None
            and capture_time_ns - self._last_success_ns <= self.hold_timeout_ns
        ):
            # 保持值沿用上一次成功估计，但帧身份和校正时间属于当前帧。
            age_ns = capture_time_ns - self._last_success_ns
            normalized = self._last_openness
            valid = True
            outcome = "held"
            reason = "短暂漏检，保持最近成功开度"
        else:
            normalized = 0
            valid = False
            outcome = "invalid"
            if self._last_success_ns is None:
                reason = "未检测到有效双标签"
            else:
                age_ns = capture_time_ns - self._last_success_ns
                reason = "未检测到有效双标签且超过保持期限"
        self._result = GripperObservation(
            normalized=normalized,
            valid=valid,
            sample_time_ns=sample_time_ns,
            sequence=sample.sequence,
            reason=reason,
        )
        # 记录真实视觉来源及结果类别；无效帧的 0 是占位值，不能解释为闭合。
        # 漏检时位姿和距离保持为空，避免把保持值伪装成新测量。
        log_diagnostic(
            "visual_gripper", source="aruco", outcome=outcome,
            sequence=sample.sequence, capture_time_ns=capture_time_ns,
            sample_time_ns=sample_time_ns,
            valid=valid, reason=reason, normalized=normalized,
            detected=detected,
            last_success_time_ns=self._last_success_ns,
            last_success_age_ns=age_ns,
            hold_timeout_ns=self.hold_timeout_ns,
            detected_marker_count=getattr(
                self.estimator, "last_detected_marker_count", None
            ),
            distance_mm=getattr(estimate, "distance_mm", None),
            raw_openness=getattr(estimate, "raw_openness", None),
            left_position_mm=getattr(
                getattr(estimate, "left_pose", None), "translation_mm", None
            ),
            right_position_mm=getattr(
                getattr(estimate, "right_pose", None), "translation_mm", None
            ),
            roi=getattr(estimate, "roi", None),
            closed_distance_mm=getattr(self.estimator, "closed_distance_mm", None),
            open_distance_mm=getattr(self.estimator, "open_distance_mm", None),
        )
        self._key = key
        self._last_frame_ns = capture_time_ns
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
