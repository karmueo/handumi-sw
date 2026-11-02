"""验证 Vive 采集健康判定和易读的实时状态内容。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.cameras.base import CameraSample
from handumi.grippers.base import GripperObservation
from handumi.scripts.umi_capture import (
    CaptureHealth,
    NativeUmi,
    capture_health,
    capture_status_text,
    log_capture_status,
)
from handumi.tracking.base import ControllerPairSample


def _inputs():
    """构建正常且时间同步的一组样本。"""
    image = CameraSample(np.zeros((2, 2, 3), dtype=np.uint8), 1_000_000_000, 1)
    tracking = replace(
        ControllerPairSample.empty("vive"), right_tracked=True, right_pose_valid=True,
    )
    gripper = GripperObservation(0.5, True, image.capture_time_ns, image.sequence)
    return image, tracking, gripper


@pytest.mark.parametrize(
    ("age", "offset", "valid"),
    [(0, 60_000_000, True), (350_000_000, -60_000_000, True),
     (-1, 0, False), (350_000_001, 0, False), (150_000_000, 60_000_001, False)],
)
def test_health_uses_original_thresholds(age, offset, valid):
    """边界值沿用采集健康规则，位姿标志不改变既有判定。"""
    image, tracking, gripper = _inputs()
    tracking = replace(tracking, right_pose_valid=False)
    health = capture_health(
        image, tracking, gripper, image.capture_time_ns,
        image.capture_time_ns + offset, image.capture_time_ns + age,
    )
    assert health.healthy is valid


@pytest.mark.parametrize(
    ("elapsed", "offset", "tracked", "gripper_valid", "expected"),
    [(0, 0, True, True, True), (200_000_001, 0, True, True, False),
     (0, 60_000_001, True, True, False), (0, 0, False, True, False),
     (0, 0, True, False, False)],
)
def test_native_sample_reuses_health_result(
    monkeypatch, elapsed, offset, tracked, gripper_valid, expected,
):
    """NativeUmi 的返回值和面板读取的检查结果来自同一次计算。"""
    from handumi.scripts import umi_capture

    image, tracking, gripper = _inputs()
    tracking = replace(tracking, right_tracked=tracked, right_pose_valid=False)
    gripper = replace(gripper, valid=gripper_valid)
    device = NativeUmi.__new__(NativeUmi)
    device.offset_ns = offset
    device.camera = SimpleNamespace(sample_at=lambda _: image)
    device.tracker = SimpleNamespace(error=None, sample_at=lambda _: tracking)
    device.gripper = SimpleNamespace(observe=lambda *_args, **_kwargs: gripper)
    times = iter([0, elapsed])
    monkeypatch.setattr(umi_capture.time, "monotonic_ns", lambda: next(times))

    result = device.sample(1_150_000_000)

    assert result[-1] is expected
    assert device.last_health.healthy is result[-1]
    assert device.last_health.camera_age_ns == 150_000_000 + elapsed
    assert device.last_health.synchronization_error_ns == offset


def test_status_explains_capture_failures_and_held_gripper():
    """同时显示多项实际阻塞原因、重复帧和无效位姿。"""
    _, tracking, gripper = _inputs()
    tracking = replace(tracking, right_tracked=False, right_pose_valid=False)
    gripper = replace(gripper, valid=False, reason="未检测到有效双标签且超过保持期限")
    health = CaptureHealth(False, False, 351_000_000, 61_000_000)
    text = capture_status_text(
        mode="record", health=health, tracking=tracking, gripper=gripper,
        healthy=False, duplicate_frame=True, frame_count=0,
        last_result="当前段已丢弃：相机帧重复",
    )
    for content in (
        "等待设备就绪", "相机帧过旧", "相机帧重复", "Tracker 未追踪",
        "位姿无效", "超过保持期限", "夹爪检测无效", "相机同步超限", "当前段已丢弃",
    ):
        assert content in text

    gripper = replace(gripper, valid=True, reason="短暂漏检，保持最近成功开度")
    tracking = replace(tracking, right_tracked=True)
    health = CaptureHealth(True, True, 150_000_000, 0)
    text = capture_status_text(
        mode="record", health=health, tracking=tracking, gripper=gripper,
        healthy=True, recording=True, frame_count=45, fps=30, saved=3,
    )
    for content in ("录制中", "45 帧", "1.5 秒", "本次已保存：3 段", "短暂漏检", "位姿无效"):
        assert content in text
    assert "不可录制原因" not in text


@pytest.mark.parametrize("mode, label", [("doctor", "设备诊断"), ("visualize", "采集预览")])
def test_nonrecord_modes_hide_recording_controls(mode, label):
    """诊断和预览仍显示设备信息，隐藏录制进度及操作说明。"""
    _, tracking, gripper = _inputs()
    health = CaptureHealth(True, True, 150_000_000, 0)
    text = capture_status_text(
        mode=mode, health=health, tracking=tracking, gripper=gripper, healthy=True,
    )
    assert label in text and "设备就绪" in text and "同步：正常" in text
    assert "当前段" not in text and "空格" not in text


def test_saving_hides_recording_shortcuts():
    """保存期间提示等待，避免暗示此时按键立即生效。"""
    _, tracking, gripper = _inputs()
    text = capture_status_text(
        mode="record", health=CaptureHealth(True, True, 150_000_000, 0),
        tracking=tracking, gripper=gripper, healthy=True, recording=True, saving=True,
    )
    assert "保存中" in text and "正在保存，请稍候" in text
    assert "空格" not in text and "Esc" not in text


def test_status_document_is_static_and_serializable():
    """状态文档使用静态最新值，并由真实 Rerun 编码中文 Markdown。"""
    import rerun as rr

    recording = rr.RecordingStream("capture_status_test", make_default=False)
    memory = recording.memory_recording()
    viewer = SimpleNamespace(log=Mock(side_effect=recording.log), TextDocument=rr.TextDocument)
    _, tracking, gripper = _inputs()
    try:
        log_capture_status(
            viewer, mode="record", health=CaptureHealth(True, True, 150_000_000, 0),
            tracking=tracking, gripper=gripper, healthy=True,
        )
        assert viewer.log.call_args.kwargs == {"static": True}
        recording.flush()
        assert len(memory.drain_as_bytes()) > 0
    finally:
        recording.disconnect()
