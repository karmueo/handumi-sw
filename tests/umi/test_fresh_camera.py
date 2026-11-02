"""验证 VIVE 录制的新鲜相机帧选择及原有同步质量门。"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from handumi.cameras.base import CameraSample
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.grippers.base import GripperObservation
from handumi.scripts.umi_capture import NativeUmi, transition

TARGET = 2_770_186_000


def _camera(times):
    """构造带真实时间戳和序号的内存相机缓存。"""
    camera = OpenCVCameraDevice("synthetic", 30, 2, 2)
    camera._camera = object()
    for sequence, stamp in times:
        camera._samples.append(CameraSample(np.zeros((2, 2, 3), np.uint8), stamp, sequence))
    return camera


def test_newer_candidate_wins_over_slightly_closer_previous_frame():
    """两帧均同步合格时，选择未记录的后帧而非微近的旧帧。"""
    camera = _camera([(447, 2_754_281_000), (448, 2_786_229_000)])
    assert camera.sample_at(TARGET).sequence == 447
    chosen = camera.sample_at(TARGET, after_sequence=447)
    assert chosen.sequence == 448
    assert chosen.capture_time_ns == 2_786_229_000


def test_no_new_candidate_falls_back_to_original_frame():
    """无新候选时返回原最近帧，重复判断仍可拒绝该帧。"""
    camera = _camera([(446, 2_720_000_000), (447, 2_754_281_000)])
    assert camera.sample_at(TARGET, after_sequence=447).sequence == 447


def test_native_aligns_tracker_and_gripper_to_new_frame():
    """新帧决定 Tracker 插值与开度时间，所有时间戳保持实际值。"""
    device = NativeUmi.__new__(NativeUmi)
    device.offset_ns = 5_000_000
    # 相机目标时间为 TARGET；校正时间使用所选帧的真实时间加偏移。
    device.camera = _camera([(447, 2_754_281_000), (448, 2_786_229_000)])
    device.tracker = SimpleNamespace(error=None, sample_at=Mock(return_value=SimpleNamespace(right_tracked=True)))
    device.gripper = SimpleNamespace(observe=Mock(return_value=GripperObservation(0.5, True, TARGET, 448)))
    image, _, _, target, healthy = device.sample(TARGET + device.offset_ns + 150_000_000, after_sequence=447)
    assert image.sequence == 448 and healthy
    assert target == TARGET + device.offset_ns
    device.tracker.sample_at.assert_called_once_with(image.capture_time_ns + device.offset_ns)
    device.gripper.observe.assert_called_once_with(image, offset_ns=device.offset_ns)


def test_new_candidate_beyond_original_sync_limit_still_discards():
    """新帧超出原 ±60 ms 同步门时，即使按保存键也丢弃该段。"""
    device = NativeUmi.__new__(NativeUmi)
    device.offset_ns = 0
    device.camera = _camera([(447, TARGET), (448, TARGET + 60_000_001)])
    device.tracker = SimpleNamespace(error=None, sample_at=lambda _: SimpleNamespace(right_tracked=True))
    device.gripper = SimpleNamespace(observe=lambda *_args, **_kwargs: GripperObservation(0.5, True, TARGET, 448))
    image, _, _, _, healthy = device.sample(TARGET + 150_000_000, after_sequence=447)
    assert image.sequence == 448 and not healthy
    assert not device.last_health.synchronization_ok
    assert transition(True, " ", healthy) == (False, "discard")
