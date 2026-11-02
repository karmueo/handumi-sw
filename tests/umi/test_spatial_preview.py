"""验证 Vive 可选空间相机视图的几何、时间语义和预览接入。"""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import rerun as rr
from scipy.spatial.transform import Rotation

from handumi.cameras.base import CameraSample
from handumi.grippers.base import GripperObservation
from handumi.scripts import umi_capture
from handumi.scripts._umi_spatial_preview import (
    CAMERA_PATH,
    IMAGE_PATH,
    SpatialCameraPreview,
)
from handumi.tracking.base import ControllerPairSample


@pytest.fixture
def snapshot():
    """构造包含旋转和平移的相机安装标定。"""
    mount = np.eye(4)
    mount[:3, :3] = Rotation.from_euler("xyz", [20, -15, 32], degrees=True).as_matrix()
    mount[:3, 3] = [0.08, -0.12, 0.04]
    return {
        "resolution": [64, 48], "intrinsics": [42, 43, 31, 23],
        "distortion": [0.04, -0.02, 0.001, -0.001],
        "tracker_camera": mount.tolist(),
    }


@pytest.fixture
def sample():
    """构造已落入参考系的非平凡 Tracker 位姿和图像。"""
    pose = np.r_[[0.4, -0.3, 0.7], Rotation.from_euler(
        "xyz", [12, 35, -25], degrees=True,
    ).as_quat()]
    tracking = replace(
        ControllerPairSample.empty("vive"), right_tracked=True, right_pose_valid=True,
        right_controller_pose=pose,
        # 原始姿态不同，确保实现使用已转换至参考系的 Tracker 位姿。
        right_device_controller_pose=np.array([9, 8, 7, 0, 0, 0, 1]),
    )
    pixels = np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
    return CameraSample(pixels, 1_000_000_000, 7), tracking


def test_spatial_camera_pose_rectification_and_reuse(snapshot, sample, monkeypatch):
    """旋转及平移按参考系乘安装外参，内参与去畸变映射只计算一次。"""
    estimate = Mock(wraps=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify)
    maps = Mock(wraps=cv2.fisheye.initUndistortRectifyMap)
    remap = Mock(wraps=cv2.remap)
    monkeypatch.setattr(cv2.fisheye, "estimateNewCameraMatrixForUndistortRectify", estimate)
    monkeypatch.setattr(cv2.fisheye, "initUndistortRectifyMap", maps)
    monkeypatch.setattr(cv2, "remap", remap)
    spatial = SpatialCameraPreview(snapshot)
    viewer = Mock(wraps=rr)
    viewer.TransformRelation = rr.TransformRelation
    viewer.ViewCoordinates = rr.ViewCoordinates
    viewer.log = Mock()
    image, tracking = sample
    original = image.image.copy()
    spatial.log(viewer, image, tracking, 0.15)
    spatial.log(viewer, image, tracking, 0.15)

    estimate.assert_called_once()
    assert estimate.call_args.kwargs["balance"] == 0.7
    maps.assert_called_once()
    assert remap.call_count == 2
    np.testing.assert_array_equal(image.image, original)
    assert [call.args[0] for call in viewer.log.call_args_list] == [IMAGE_PATH] * 4
    assert all(call.kwargs == {} for call in viewer.log.call_args_list)
    pinhole = viewer.log.call_args_list[0].args[1]
    expected_pinhole = rr.Pinhole(image_from_camera=spatial.image_matrix)
    np.testing.assert_allclose(
        pinhole.image_from_camera.as_arrow_array().to_pylist(),
        expected_pinhole.image_from_camera.as_arrow_array().to_pylist(), atol=1e-6,
    )
    assert pinhole.resolution.as_arrow_array().to_pylist() == [[64, 48]]
    assert pinhole.image_plane_distance.as_arrow_array().to_pylist() == pytest.approx([0.12])
    assert isinstance(viewer.log.call_args_list[1].args[1], rr.EncodedImage)


@pytest.mark.parametrize("age", [-0.01, 0.351, float("nan"), float("inf")])
def test_stale_image_clears_only_image_branch_and_recovers(snapshot, sample, age):
    """画面失效时只隐藏视锥与视频，恢复时重记针孔模型与图像。"""
    spatial = SpatialCameraPreview(snapshot)
    viewer = Mock(wraps=rr)
    viewer.ViewCoordinates = rr.ViewCoordinates
    viewer.log = Mock()
    image, tracking = sample
    spatial.log(viewer, image, tracking, age)
    path, clear = viewer.log.call_args.args
    assert path == IMAGE_PATH and isinstance(clear, rr.Clear)
    assert clear.is_recursive.as_arrow_array().to_pylist() == [True]
    assert viewer.log.call_count == 1
    viewer.log.reset_mock()
    spatial.log(viewer, image, tracking, 0.15)
    assert [call.args[0] for call in viewer.log.call_args_list] == [IMAGE_PATH] * 2


@pytest.mark.parametrize("tracked,valid", [(False, True), (True, False)])
def test_tracking_loss_clears_parent_and_rebuilds_image(snapshot, sample, tracked, valid):
    """定位失效递归清除相机父节点，空间子节点不被重建。"""
    spatial = SpatialCameraPreview(snapshot)
    viewer = Mock(wraps=rr)
    viewer.ViewCoordinates = rr.ViewCoordinates
    viewer.log = Mock()
    viewer.set_time = Mock()
    image, tracking = sample
    failed = replace(tracking, right_tracked=tracked, right_pose_valid=valid)
    gripper = GripperObservation(0.5, False, image.capture_time_ns, image.sequence)
    umi_capture.log_live_preview(
        viewer, image, failed, gripper, False, 0.15,
        tracker_from_camera=np.asarray(snapshot["tracker_camera"]),
    )
    spatial.log(viewer, image, failed, 0.15)
    records = {call.args[0]: call.args[1] for call in viewer.log.call_args_list}
    assert isinstance(records[CAMERA_PATH], rr.Clear)
    assert records[CAMERA_PATH].is_recursive.as_arrow_array().to_pylist() == [True]
    assert IMAGE_PATH not in records
    viewer.log.reset_mock()
    umi_capture.log_live_preview(
        viewer, image, tracking, gripper, False, 0.15,
        tracker_from_camera=np.asarray(snapshot["tracker_camera"]),
    )
    spatial.log(viewer, image, tracking, 0.15)
    paths = [call.args[0] for call in viewer.log.call_args_list]
    assert paths.index(CAMERA_PATH) < paths.index(IMAGE_PATH)
    assert paths.count(IMAGE_PATH) == 2


@pytest.mark.parametrize("mode", ["visualize", "record", "doctor"])
def test_conflicting_preview_options_fail_before_calibration(mode, monkeypatch):
    """所有入口均在加载配置、文件或设备前拒绝互斥参数。"""
    monkeypatch.setattr(umi_capture, "calibration_paths", Mock(side_effect=AssertionError))
    with pytest.raises(SystemExit) as error:
        umi_capture.main(["--spatial-view", "--no-preview"], mode=mode)
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["visualize", "record", "doctor"])
@pytest.mark.parametrize("spatial_enabled", [False, True])
def test_capture_entry_logs_spatial_only_when_selected(
    mode, spatial_enabled, snapshot, sample, tmp_path, monkeypatch,
):
    """三个入口复用当前帧；默认不初始化空间模型，丢失夹爪仍能显示。"""
    import handumi.scripts._umi_spatial_preview as spatial_module

    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    output = tmp_path / "dataset"
    image_template, tracking = sample
    viewer = Mock(wraps=rr)
    viewer.TransformRelation = rr.TransformRelation
    viewer.ViewCoordinates = rr.ViewCoordinates
    viewer.log = Mock()
    viewer.set_time = Mock()
    viewer.disconnect = Mock()
    spatial_factory = Mock(wraps=SpatialCameraPreview)
    monkeypatch.setattr(spatial_module, "SpatialCameraPreview", spatial_factory)
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: snapshot)
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {})
    # 无桌面标定时保留现有 OpenVR 回退。
    session = {} if mode == "record" else None
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: session)
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", lambda *a: viewer)
    monkeypatch.setattr(umi_capture, "_check_output_directory", lambda *a: None)

    def prepare(_args):
        """准备关闭模拟数据集时需要更新的元数据。"""
        (output / "meta").mkdir(parents=True)
        (output / "meta/info.json").write_text("{}")

    monkeypatch.setattr(umi_capture, "_prepare_output_directory", prepare)
    monkeypatch.setattr(umi_capture, "_open_recording_dataset", lambda *a: Mock())
    monkeypatch.setattr(umi_capture.time, "sleep", lambda *_: None)

    class Devices:
        """返回一次同步图像及 Tracker 样本，不打开硬件。"""

        def __init__(self, *args, **kwargs):
            """记录传入的桌面会话，验证 OpenVR 回退。"""
            assert kwargs["session"] is session
            self.last_health = umi_capture.CaptureHealth(True, False, 150_000_000, 0)

        def __enter__(self):
            """返回模拟设备。"""
            return self

        def __exit__(self, *args):
            """结束模拟设备。"""

        def sample(self, now, *, after_sequence=None):
            """返回有效追踪和无效夹爪的同帧数据。"""
            image = replace(image_template, capture_time_ns=now - 150_000_000)
            gripper = GripperObservation(0.5, False, image.capture_time_ns, image.sequence)
            return image, tracking, gripper, now, True

    @contextmanager
    def keyboard():
        """允许一帧预览后退出。"""
        keys = iter(["", "\x1b"])
        yield lambda: next(keys)

    monkeypatch.setattr(umi_capture, "NativeUmi", Devices)
    monkeypatch.setattr(umi_capture, "keyboard", keyboard)
    args = ["--calibration", str(calibration)]
    if mode == "record":
        args += ["--output-dir", str(output)]
    if spatial_enabled:
        args.append("--spatial-view")
    umi_capture.main(args, mode=mode)
    assert spatial_factory.call_count == int(spatial_enabled)
    paths = [call.args[0] for call in viewer.log.call_args_list]
    assert "camera/right_wrist" in paths
    assert CAMERA_PATH in paths
    assert "tracking/camera/axes" in paths
    assert "tracking/camera/origin" in paths
    assert (IMAGE_PATH in paths) is spatial_enabled
    viewer.set_time.assert_called_once()
    assert viewer.set_time.call_args.args == ("capture",)


def test_real_rerun_stream_serializes_spatial_frames_and_clear(snapshot, sample):
    """实际 Rerun 内存流可编码模型、动态帧、清除和恢复。"""
    import rerun.blueprint as rrb

    from handumi.scripts._umi_preview import capture_blueprint
    from handumi.scripts.umi_capture import _RerunCaptureViewer

    stream = rr.RecordingStream("spatial_camera_test", make_default=False)
    memory = stream.memory_recording()
    viewer = _RerunCaptureViewer(stream, rr)
    image, tracking = sample
    spatial = SpatialCameraPreview(snapshot)
    try:
        stream.send_blueprint(capture_blueprint(rrb, table_frame=False))
        gripper = GripperObservation(0.5, False, image.capture_time_ns, image.sequence)
        for index, tracked in enumerate((True, False, True), start=1):
            current_image = replace(image, capture_time_ns=index * 1_000_000_000)
            current_tracking = replace(tracking, right_tracked=tracked)
            umi_capture.log_live_preview(
                viewer, current_image, current_tracking, gripper, False, 0.15,
                tracker_from_camera=np.asarray(snapshot["tracker_camera"]),
            )
            spatial.log(viewer, current_image, current_tracking, 0.15)
        stream.flush()
        assert len(memory.drain_as_bytes()) > 0
    finally:
        stream.disconnect()


@pytest.mark.parametrize("session", [None, {"table_from_device": []}])
def test_camera_transform_is_independent_of_tcp_and_shared_by_frustum(snapshot, session):
    """相机使用 Tracker 外参，TCP 位姿及轴长变化不会移动相机或视锥。"""
    viewer = Mock(wraps=rr)
    viewer.ViewCoordinates = rr.ViewCoordinates
    viewer.log = Mock()
    viewer.set_time = Mock()
    image = CameraSample(np.zeros((48, 64, 3), np.uint8), 10, 1)
    tracker = np.array([1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    tcp = np.array([4, 5, 6, 0, 0, 0, 1])
    tracking = replace(
        ControllerPairSample.empty("vive"), right_tracked=True, right_pose_valid=True,
        right_controller_pose=tracker, right_tcp_pose=tcp,
    )
    mount = np.array([
        [1, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
    ], dtype=float)
    gripper = GripperObservation(0.5, False, image.capture_time_ns, image.sequence)
    spatial = SpatialCameraPreview(snapshot)

    def capture(current_tracking, current_mount, axis_length):
        """收集共享预览与相机图像在同一时间轴上的实体。"""
        viewer.log.reset_mock()
        umi_capture.log_live_preview(
            viewer, image, current_tracking, gripper, False, 0.15, axis_length,
            session=session, tracker_from_camera=current_mount,
        )
        spatial.log(viewer, image, current_tracking, 0.15)
        return {call.args[0]: call.args[1] for call in viewer.log.call_args_list}

    first = capture(tracking, mount, 0.08)
    assert CAMERA_PATH == "tracking/camera"
    assert IMAGE_PATH == "tracking/camera/image"
    camera = first[CAMERA_PATH]
    np.testing.assert_allclose(camera.translation.as_arrow_array().to_pylist(), [[1, 3, 3]], atol=1e-6)
    # Rz(90°) × Rx(90°) 的明确结果，避免按实现中的乘法重算预期。
    expected_rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    expected = rr.Transform3D(mat3x3=expected_rotation)
    np.testing.assert_allclose(
        camera.mat3x3.as_arrow_array().to_pylist(),
        expected.mat3x3.as_arrow_array().to_pylist(), atol=1e-6,
    )
    axes = first["tracking/camera/axes"]
    np.testing.assert_allclose(
        axes.vectors.as_arrow_array().to_pylist(), np.eye(3) * 0.06, atol=1e-7,
    )
    label = first["tracking/camera/origin"]
    assert label.labels.as_arrow_array().to_pylist() == ["Camera"]
    assert label.colors.as_arrow_array().to_pylist() == [
        int.from_bytes(bytes([110, 180, 255, 255]), "big")
    ]
    assert isinstance(first[IMAGE_PATH], rr.EncodedImage)
    first_model = next(
        call.args[1] for call in viewer.log.call_args_list
        if call.args[0] == IMAGE_PATH and isinstance(call.args[1], rr.Pinhole)
    )
    assert not any(
        isinstance(call.args[1], rr.Transform3D)
        for call in viewer.log.call_args_list if call.args[0] == IMAGE_PATH
    )
    # 改变 TCP 位姿、轴长和无关的 tracker_tcp 数据，Camera 与子视锥不变。
    tcp_mount = np.eye(4)
    tcp_mount[:3, :3] = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    tcp_mount[:3, 3] = [0.3, -0.1, 0.2]
    snapshot["tracker_tcp"] = tcp_mount.tolist()
    spatial = SpatialCameraPreview(snapshot)
    other_tcp = np.array([-9, 7, 5, 0, 0, 1, 0])
    second = capture(replace(tracking, right_tcp_pose=other_tcp), mount, 0.2)
    second_model = next(
        call.args[1] for call in viewer.log.call_args_list
        if call.args[0] == IMAGE_PATH and isinstance(call.args[1], rr.Pinhole)
    )
    np.testing.assert_array_equal(
        second_model.image_from_camera.as_arrow_array().to_pylist(),
        first_model.image_from_camera.as_arrow_array().to_pylist(),
    )
    np.testing.assert_array_equal(
        second[CAMERA_PATH].translation.as_arrow_array().to_pylist(),
        camera.translation.as_arrow_array().to_pylist(),
    )
    np.testing.assert_array_equal(
        second[CAMERA_PATH].mat3x3.as_arrow_array().to_pylist(),
        camera.mat3x3.as_arrow_array().to_pylist(),
    )
    np.testing.assert_allclose(second["tracking/tcp"].translation.as_arrow_array().to_pylist(), [other_tcp[:3]])
    assert isinstance(second[IMAGE_PATH], rr.EncodedImage)
    # 仅改变相机相对 Tracker 的位移；父节点与视锥一起移动，其他坐标不变。
    moved_mount = mount.copy()
    moved_mount[:3, 3] = [0, 1, 0]
    third = capture(tracking, moved_mount, 0.08)
    np.testing.assert_allclose(third[CAMERA_PATH].translation.as_arrow_array().to_pylist(), [[0, 2, 3]], atol=1e-6)
    np.testing.assert_allclose(third["tracking/tracker"].translation.as_arrow_array().to_pylist(), [tracker[:3]])
    np.testing.assert_allclose(third["tracking/tcp"].translation.as_arrow_array().to_pylist(), [tcp[:3]])
    assert isinstance(third[IMAGE_PATH], rr.EncodedImage)


def test_stale_image_keeps_camera_axes(snapshot, sample):
    """定位有效而相机帧过旧时，完整相机坐标依然保持。"""
    image, tracking = sample
    viewer = Mock(wraps=rr)
    viewer.ViewCoordinates = rr.ViewCoordinates
    viewer.log = Mock()
    viewer.set_time = Mock()
    gripper = GripperObservation(0.5, False, image.capture_time_ns, image.sequence)
    umi_capture.log_live_preview(
        viewer, image, tracking, gripper, False, 0.4,
        tracker_from_camera=np.asarray(snapshot["tracker_camera"]),
    )
    SpatialCameraPreview(snapshot).log(viewer, image, tracking, 0.4)
    records = {call.args[0]: call.args[1] for call in viewer.log.call_args_list}
    assert isinstance(records[CAMERA_PATH], rr.Transform3D)
    assert isinstance(records["tracking/camera/axes"], rr.Arrows3D)
    assert isinstance(records["tracking/camera/origin"], rr.Points3D)
    assert isinstance(records[IMAGE_PATH], rr.Clear)
