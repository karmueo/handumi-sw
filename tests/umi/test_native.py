"""验证原生 UMI 的同帧感知、严格录制和数据语义。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.calibration.umi import pose7
from handumi.cameras.base import CameraSample
from handumi.dataset.gripper import read_normalized, require_physical_width
from handumi.grippers.aruco import GripperOpennessEstimator
from handumi.grippers.base import VisualGripperProvider
from handumi.robots.utils import pose7_to_mat
from handumi.scripts.umi_capture import (
    NativeUmi,
    init_tracking_preview,
    log_live_preview,
    transition,
)
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

    assert viewer.log.call_count == 7
    assert viewer.log.call_args_list[0].kwargs == {"static": True}
    assert all(not call.kwargs for call in viewer.log.call_args_list[1:])
    assert [call.args[0] for call in viewer.log.call_args_list] == [
        "camera/right_wrist",
        "gripper/right",
        "health/valid",
        "health/camera_sequence",
        "health/camera_age_s",
        "tracking/tracker",
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
        right_device_controller_pose=np.array([0.2, 0, 0, 0, 0, 0, 1]),
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


def test_workspace_after_raw_interpolation_and_tcp_once(table_setup):
    """原始轨迹先插值再应用桌面和 TCP，掉线样本仍携带相同桌面变换。"""
    profile, data, session, _ = table_setup
    identity = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    provider = ViveTrackingProvider("synthetic", ControllerTcpCalibration(identity, pose7(profile["tracker_tcp"])))
    workspace = pose7(session["table_from_device"])
    provider.set_workspace_from_device_pose(workspace)
    first = replace(
        ControllerPairSample.empty("vive"), right_tracked=True, right_pose_valid=True,
        right_device_controller_pose=np.array(data["tracker_poses"][0]), pc_monotonic_ns=10_000_000,
    )
    second = replace(first, right_device_controller_pose=np.array(data["tracker_poses"][1]), pc_monotonic_ns=30_000_000)
    provider._history.extend([first, second])
    sample = provider.sample_at(20_000_000)
    np.testing.assert_allclose(sample.right_device_controller_pose[:3], (first.right_device_controller_pose[:3] + second.right_device_controller_pose[:3]) / 2)
    expected_anchor = np.array(session["table_from_device"]) @ pose7_to_mat(sample.right_device_controller_pose)
    np.testing.assert_allclose(pose7_to_mat(sample.right_controller_pose), expected_anchor, atol=1e-6)
    np.testing.assert_allclose(pose7_to_mat(sample.right_tcp_pose), expected_anchor @ profile["tracker_tcp"], atol=1e-6)
    np.testing.assert_allclose(sample.left_controller_pose, identity)
    for stamp in [0, 40_000_000]:
        invalid = provider.sample_at(stamp)
        assert not invalid.right_tracked
        np.testing.assert_allclose(invalid.workspace_from_device_pose, workspace)
    provider._history[-1] = replace(second, right_tracked=False)
    np.testing.assert_allclose(provider.sample_at(20_000_000).workspace_from_device_pose, workspace)
    provider._thread = object()
    with pytest.raises(RuntimeError, match="不能更换"):
        provider.set_workspace_from_device_pose(identity)


def _assert_axes(record, length, *, static):
    """验证三色轴箭头的局部原点、方向、长度和记录方式。"""
    import rerun as rr

    assert record.kwargs == ({"static": True} if static else {})
    axes = record.args[1]
    assert isinstance(axes, rr.Arrows3D)
    np.testing.assert_allclose(axes.origins.as_arrow_array().to_pylist(), np.zeros((3, 3)))
    np.testing.assert_allclose(axes.vectors.as_arrow_array().to_pylist(), np.eye(3) * length)
    assert axes.radii.as_arrow_array().to_pylist() == pytest.approx(
        [float(rr.Radius.ui_points(1.0))]
    )
    assert axes.colors.as_arrow_array().to_pylist() == [
        int.from_bytes(bytes(color), "big")
        for color in ([255, 0, 0, 255], [0, 128, 0, 255], [0, 0, 139, 255])
    ]
    assert axes.labels is None
    assert axes.show_labels.as_arrow_array().to_pylist() == [False]


def _assert_origin(record, label, color, *, static):
    """验证原点标记使用原生名称标签和原有颜色。"""
    import rerun as rr

    assert record.kwargs == ({"static": True} if static else {})
    point = record.args[1]
    assert isinstance(point, rr.Points3D)
    assert point.positions.as_arrow_array().to_pylist() == [[0, 0, 0]]
    assert point.radii.as_arrow_array().to_pylist() == pytest.approx([0.005])
    assert point.colors.as_arrow_array().to_pylist() == [
        int.from_bytes(bytes([*color, 255]), "big")
    ]
    assert point.labels.as_arrow_array().to_pylist() == [label]
    assert point.show_labels.as_arrow_array().to_pylist() == [True]


@pytest.mark.parametrize("session", [{"table_from_device": []}, None])
def test_preview_reference_axes_and_bounds(session):
    """桌面和 OpenVR 使用原生名称与彩色坐标轴，保留原有初始取景范围。"""
    import rerun as rr

    viewer = Mock(wraps=rr)
    viewer.log = Mock()
    viewer.ViewCoordinates = rr.ViewCoordinates
    init_tracking_preview(viewer, session)
    records = {call.args[0]: call for call in viewer.log.call_args_list}
    assert records["reference_frame"].kwargs == {"static": True}
    assert isinstance(records["reference_frame"].args[1], rr.TextDocument)
    assert "tracking/reference_grid" not in records
    assert not any("axis_labels" in path for path in records)
    assert not any(isinstance(call.args[1], rr.Mesh3D) for call in records.values())
    bounds = records["tracking/bounds"].args[1]
    assert bounds.labels is None
    assert bounds.show_labels.as_arrow_array().to_pylist() == [False]
    assert bounds.radii.as_arrow_array().to_pylist() == pytest.approx([0.004])
    assert bounds.colors.as_arrow_array().to_pylist() == [
        int.from_bytes(bytes([128, 100, 100, 90]), "big")
    ] * 8
    vertices = np.asarray(bounds.positions.as_arrow_array().to_pylist())
    assert vertices.shape == (8, 3)
    expected_max = [0.75, 0.75, 0.4] if session is not None else [0.75, 0.4, 0.75]
    np.testing.assert_allclose(vertices.max(axis=0), expected_max)
    np.testing.assert_allclose(vertices.min(axis=0), -np.asarray(expected_max))
    assert isinstance(records["tracking/reference"].args[1], rr.Transform3D)
    _assert_axes(records["tracking/reference/axes"], 0.12, static=True)
    _assert_origin(records["tracking/reference/origin"],
                   "Table" if session is not None else "OpenVR", [230, 230, 230], static=True)
    coordinates = (rr.ViewCoordinates.RIGHT_HAND_Z_UP if session is not None
                   else rr.ViewCoordinates.RIGHT_HAND_Y_UP)
    assert records["tracking"].args[1] is coordinates


@pytest.mark.parametrize("session", [{"table_from_device": []}, None])
def test_tracker_and_tcp_axes_follow_tracking_validity(session):
    """局部箭头跟随真实位姿，名称在失效时被递归清除并在恢复后重记。"""
    import rerun as rr
    from scipy.spatial.transform import Rotation

    viewer = Mock(wraps=rr)
    viewer.log = Mock()
    viewer.set_time = Mock()
    image = CameraSample(np.zeros((48, 64, 3), np.uint8), 100, 1)
    tracker = np.array([1, 2, 3, 0, np.sqrt(0.5), 0, np.sqrt(0.5)])
    tcp = np.array([2, 3, 4, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    sample = replace(
        ControllerPairSample.empty("vive"), right_tracked=True, right_pose_valid=True,
        right_controller_pose=tracker, right_tcp_pose=tcp,
    )
    for tracked in (True, False, True):
        viewer.log.reset_mock()
        log_live_preview(viewer, image, replace(sample, right_tracked=tracked),
                         SimpleNamespace(normalized=0.5, valid=False), False, 0.1, 0.11,
                         session=session)
        records = {call.args[0]: call for call in viewer.log.call_args_list}
        assert not any("axis_labels" in path for path in records)
        assert not any(isinstance(call.args[1], rr.Mesh3D) for call in records.values())
        for path, pose, length, label, color in (
            ("tracking/tracker", tracker, 0.06, "Tracker", [255, 190, 50]),
            ("tracking/tcp", tcp, 0.11, "TCP", [80, 220, 255]),
        ):
            value = records[path].args[1]
            if not tracked:
                assert isinstance(value, rr.Clear)
                assert value.is_recursive.as_arrow_array().to_pylist() == [True]
                assert not any(key.startswith(path + "/") for key in records)
                continue
            assert isinstance(value, rr.Transform3D)
            np.testing.assert_allclose(value.translation.as_arrow_array().to_pylist(), [pose[:3]])
            np.testing.assert_allclose(value.quaternion.as_arrow_array().to_pylist(), [pose[3:]])
            _assert_axes(records[path + "/axes"], length, static=False)
            _assert_origin(records[path + "/origin"], label, color, static=False)
            vectors = records[path + "/axes"].args[1].vectors.as_arrow_array().to_pylist()
            rotated = Rotation.from_quat(pose[3:]).apply(vectors)
            np.testing.assert_allclose(rotated, Rotation.from_quat(pose[3:]).as_matrix().T * length, atol=1e-7)


def test_native_axes_and_labels_serialize_with_real_recording():
    """真实 Rerun 记录流能够序列化静态和动态箭头及原生标签。"""
    import rerun as rr

    from handumi.scripts.umi_capture import _RerunCaptureViewer

    recording = rr.RecordingStream("native_axes_test", make_default=False)
    memory = recording.memory_recording()
    viewer = _RerunCaptureViewer(recording, rr)
    try:
        init_tracking_preview(viewer, {})
        sample = replace(
            ControllerPairSample.empty("vive"), right_tracked=True, right_pose_valid=True,
        )
        image = CameraSample(np.zeros((48, 64, 3), np.uint8), 100, 1)
        log_live_preview(
            viewer, image, sample, SimpleNamespace(normalized=0.5), True, 0.1,
            session={},
        )
        recording.flush()
        assert len(memory.drain_as_bytes()) > 0
    finally:
        recording.disconnect()
