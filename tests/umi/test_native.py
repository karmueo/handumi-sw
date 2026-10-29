"""验证原生 UMI 的同帧感知、严格录制和数据语义。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.cameras.base import CameraSample
from handumi.dataset.gripper import read_normalized, require_physical_width
from handumi.grippers.aruco import GripperOpennessEstimator
from handumi.grippers.base import VisualGripperProvider
from handumi.scripts.umi_capture import NativeUmi, log_live_preview, transition
from handumi.tracking.base import ControllerPairSample
from handumi.tracking.vive import ViveTrackingProvider


def test_same_frame_is_estimated_once():
    """重复读取同一帧不得再次推进 EMA，输出沿用原相机序号和校正时间。"""

    class Estimator:
        """统计纯算法调用并检查 BGR 转换。"""

        calls = 0

        def estimate(self, image):
            """返回固定测试开度。"""
            self.calls += 1
            assert image[0, 0].tolist() == [3, 2, 1]
            return SimpleNamespace(openness=0.5)

    estimator = Estimator()
    provider = VisualGripperProvider(estimator)
    frame = CameraSample(np.array([[[1, 2, 3]]], dtype=np.uint8), 100, 7)
    first = provider.observe(frame, offset_ns=-5)
    assert provider.observe(frame, offset_ns=-5) is first
    assert (first.sample_time_ns, first.sequence, first.width_m) == (95, 7, None)
    assert estimator.calls == 1


def test_live_preview_is_independent_of_timeline():
    """相机脱离时间轴，原有曲线仍带采集时间，录制图像保持不变。"""
    import rerun as rr

    viewer = Mock(wraps=rr)
    viewer.log = Mock()
    viewer.set_time = Mock()
    pixels = np.full((48, 64, 3), 120, dtype=np.uint8)
    image = CameraSample(pixels, 100, 7)
    tracking = ControllerPairSample.empty("vive")
    gripper = SimpleNamespace(valid=False, normalized=0.5)

    log_live_preview(viewer, image, tracking, gripper, False, 0.15)

    assert viewer.log.call_count == 6
    assert viewer.log.call_args_list[0].kwargs == {"static": True}
    assert all(not call.kwargs for call in viewer.log.call_args_list[1:])
    assert [call.args[0] for call in viewer.log.call_args_list] == [
        "camera/right_wrist",
        "gripper/right",
        "health/valid",
        "health/camera_sequence",
        "health/camera_age_s",
        "tracking/tcp",
    ]
    viewer.set_time.assert_called_once_with("capture", duration=100 / 1e9)
    assert isinstance(viewer.log.call_args_list[0].args[1], rr.EncodedImage)
    assert np.all(pixels == 120)


def test_connection_error_survives_cleanup_failure():
    """相机连接失败后仍停止 Tracker，清理失败不会覆盖原始异常。"""
    devices = NativeUmi.__new__(NativeUmi)
    devices.tracker = Mock()
    devices.camera = Mock()
    devices.camera.connect.side_effect = ConnectionError("camera open failed")
    devices.camera.disconnect.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(ConnectionError, match="camera open failed") as error:
        devices.__enter__()

    devices.tracker.stop.assert_called_once()
    assert any("cleanup failed" in note for note in error.value.__notes__)


@pytest.mark.parametrize("key", [" ", "", "\x1b"])
def test_failure_precedes_save(key):
    """任意失效必须在保存和用户操作之前丢弃整段。"""
    assert transition(True, key, False) == (False, "discard")
    assert transition(False, " ", False) == (False, "wait")


def test_keyboard_restarts_only_explicitly():
    """丢弃后需再次空格，Esc 取消与退出有独立语义。"""
    assert transition(False, " ", True) == (True, "start")
    assert transition(True, " ", True) == (False, "save")
    assert transition(True, "\x1b", True) == (False, "discard")
    assert transition(False, "\x1b", True) == (False, "exit")


def test_tracker_interpolation_refuses_invalid_gap():
    """有效两端可以 SLERP，掉线或过大时间间隔不能补成有效位姿。"""
    identity = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    provider = ViveTrackingProvider(
        "serial", ControllerTcpCalibration(identity, identity)
    )
    first = replace(
        ControllerPairSample.empty("vive"),
        right_tracked=True,
        right_pose_valid=True,
        pc_monotonic_ns=10_000_000,
    )
    second = replace(
        first,
        pc_monotonic_ns=30_000_000,
        right_controller_pose=np.array([0.2, 0, 0, 0, 0, 0, 1]),
    )
    provider._history.extend([first, second])
    result = provider.sample_at(20_000_000)
    assert result.right_tracked
    assert result.right_controller_pose[0] == pytest.approx(0.1)
    assert not provider.sample_at(0).right_tracked
    provider._history[-1] = replace(second, right_tracked=False)
    assert not provider.sample_at(20_000_000).right_tracked


def test_detection_loss_resets_smoothing():
    """丢失双标时清除平滑状态，避免恢复后混入旧开度。"""
    estimator = GripperOpennessEstimator(
        np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float),
        np.zeros(4),
        40,
        120,
        image_resolution=(640, 480),
    )
    estimator._smoothed_openness = 0.8
    assert estimator.estimate(np.zeros((480, 640, 3), np.uint8)) is None
    assert estimator._smoothed_openness is None
    assert estimator.normalize_distance(40) == 0
    assert estimator.normalize_distance(120) == 1


def test_unknown_width_never_becomes_closed_measurement():
    """新版读取必须具有有效开度，不能从原始零占位推断闭合。"""
    metadata = {
        "capture_schema": "synchronized_sources_v2",
        "active_sides": ["right"],
        "sources": {"gripper": {"right": {"width_available": False}}},
    }
    with pytest.raises(ValueError, match="开度列"):
        read_normalized({}, metadata)
    with pytest.raises(ValueError, match="物理净宽"):
        require_physical_width(metadata)
