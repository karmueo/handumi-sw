"""验证时间标定预览、标签诊断及无图形界面采集路径。"""

import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from handumi.calibration import umi_time
from handumi.scripts import calibrate_umi


def test_board_detection_keeps_partial_markers(monkeypatch):
    """标签不足时保留可视化角点，同时继续拒绝不足的标定观测。"""
    corners = [np.array([[[10, 60], [30, 60], [30, 80], [10, 80]]], np.float32)]

    class Detector:
        """提供固定的标签检测结果。"""

        def __init__(self, *args):
            """接收与 OpenCV 检测器相同的构造参数。"""

        def detectMarkers(self, image):
            """返回一个有效 ID 和一个不属于标定板的 ID。"""
            return corners * 2, np.array([[3], [40]]), []

    monkeypatch.setattr(umi_time.cv2.aruco, "ArucoDetector", Detector)
    monkeypatch.setattr(
        umi_time, "refine_tag_corners", lambda gray, corner: corner.reshape(4, 2)
    )
    image = np.zeros((100, 100, 3), np.uint8)
    observation, markers = umi_time.detect_board_details(image, np.eye(3), np.zeros(4))
    assert observation is None
    assert [tag_id for tag_id, _ in markers] == [3]
    np.testing.assert_array_equal(markers[0][1], corners[0].reshape(4, 2))
    assert umi_time.detect_board(image, np.eye(3), np.zeros(4)) is None


def test_board_detection_preserves_observation_mapping(monkeypatch):
    """预览附加的标签数据不改变求解使用的角点顺序。"""
    corner = np.array([[[10, 60], [30, 60], [30, 80], [10, 80]]], np.float32)

    class Detector:
        """提供六个可见的标定板标签。"""

        def __init__(self, *args):
            """接收与 OpenCV 检测器相同的构造参数。"""

        def detectMarkers(self, image):
            """返回六个有效标签。"""
            return [corner] * 6, np.arange(6, dtype=np.int32).reshape(-1, 1), []

    monkeypatch.setattr(umi_time.cv2.aruco, "ArucoDetector", Detector)
    monkeypatch.setattr(
        umi_time, "refine_tag_corners", lambda gray, corner: corner.reshape(4, 2)
    )
    monkeypatch.setattr(
        umi_time.cv2.fisheye, "undistortPoints", lambda pixels, *args, **kwargs: pixels
    )
    monkeypatch.setattr(
        umi_time.cv2,
        "solvePnP",
        lambda *args: (True, np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]])),
    )
    image = np.zeros((100, 100, 3), np.uint8)
    observation, markers = umi_time.detect_board_details(image, np.eye(3), np.zeros(4))
    points, pixels, pose = observation
    assert len(markers) == 6
    assert points.shape == (24, 3)
    np.testing.assert_array_equal(pixels[:4], corner.reshape(4, 2)[[1, 0, 3, 2]])
    assert pose[2, 3] == 1


def test_collect_preview_matches_sampling_and_closes(monkeypatch):
    """预览与检测使用同帧数据，失败帧不保留旧结果，且不改变采集结果。"""
    clock = [0.0]
    monkeypatch.setattr(
        calibrate_umi,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    source_images = []
    closed = []

    class Camera:
        """依次提供四个具有独立像素标识的相机帧。"""

        def __init__(self):
            """重置帧序列。"""
            self.sequences = iter([2, 3, 4, 6])

        def sample_at(self):
            """提供帧序号、时间戳和图像。"""
            sequence = next(self.sequences)
            image = np.full((120, 640, 3), sequence, np.uint8)
            source_images.append(image)
            return SimpleNamespace(
                image=image, sequence=sequence, capture_time_ns=sequence * 10**9
            )

    class Devices:
        """模拟与预览共享的相机和 Tracker。"""

        def __init__(self, snapshot):
            """构造单个相机以及一条定位记录。"""
            self.camera = Camera()
            self.tracker = SimpleNamespace(
                _lock=threading.Lock(),
                _history=[
                    SimpleNamespace(
                        pc_monotonic_ns=100,
                        right_controller_pose=np.array([0, 0, 0, 0, 0, 0, 1]),
                        right_tracked=True,
                    )
                ],
            )

        def __enter__(self):
            """返回已连接设备。"""
            return self

        def __exit__(self, *args):
            """记录设备释放。"""
            closed.append(True)

    marker = np.array([[20, 75], [40, 75], [40, 95], [20, 95]], np.float32)
    observation = (np.zeros((24, 3)), np.zeros((24, 2)), np.eye(4))

    def detect(image, matrix, distortion):
        """只在序号 4 的帧返回可用于标定的观测。"""
        sequence = int(image[0, 0, 0])
        markers = [(sequence, marker)] if sequence != 6 else []
        return (observation if sequence == 4 else None), markers

    events = []
    original_preview = calibrate_umi.board_preview

    def render(image, markers, accepted, frame_count, remaining):
        """记录帧与检测结果的对应关系，再绘制真实预览。"""
        events.append((int(image[0, 0, 0]), len(markers), accepted, frame_count))
        return original_preview(image, markers, accepted, frame_count, remaining)

    class EncodedImage:
        """记录 Rerun 接收的图像和压缩参数。"""

        def __init__(self, image):
            """保留经过标注的帧。"""
            self.image = image

        def compress(self, jpeg_quality):
            """模拟 JPEG 压缩。"""
            assert jpeg_quality == 75
            return self

    logs, timestamps, initialized, disconnected = [], [], [], []
    viewer = SimpleNamespace(
        init=lambda *args, **kwargs: initialized.append((args, kwargs)),
        set_time=lambda name, duration: timestamps.append(duration),
        log=lambda path, image: logs.append((path, image.image)),
        Image=EncodedImage,
        disconnect=lambda: disconnected.append(True),
    )
    monkeypatch.setitem(sys.modules, "rerun", viewer)
    monkeypatch.setattr(calibrate_umi, "NativeUmi", Devices)
    monkeypatch.setattr(calibrate_umi, "detect_board_details", detect)
    monkeypatch.setattr(calibrate_umi, "board_preview", render)
    snapshot = {
        "intrinsics": [1, 1, 0, 0],
        "distortion": [0] * 4,
        "geometry_digest": "g",
    }

    preview_data = calibrate_umi.collect(snapshot, 0.016)
    assert events == [(2, 1, False, 0), (4, 1, True, 1), (6, 0, False, 1)]
    assert timestamps == [2, 4, 6]
    assert len(logs) == 3 and all(path == "calibration/camera" for path, _ in logs)
    assert initialized and disconnected and closed
    assert all(
        np.all(image == sequence)
        for image, sequence in zip(source_images, [2, 3, 4, 6])
    )
    assert np.any(logs[0][1] != 2)

    clock[0] = 0.0
    events.clear()
    plain_data = calibrate_umi.collect(snapshot, 0.016, preview=False)
    assert plain_data == preview_data
    assert preview_data["camera_times"] == [4.0]
    assert not events
    assert len(initialized) == len(disconnected) == 1
    assert len(closed) == 2


def test_preview_disconnects_when_device_open_fails(monkeypatch):
    """设备连接失败时也释放已启动的预览连接。"""
    disconnected = []
    viewer = SimpleNamespace(
        init=lambda *args, **kwargs: None,
        disconnect=lambda: disconnected.append(True),
    )
    monkeypatch.setitem(sys.modules, "rerun", viewer)

    class BrokenDevices:
        """模拟打开设备失败。"""

        def __init__(self, snapshot):
            """接收设备快照。"""

        def __enter__(self):
            """报告设备不可用。"""
            raise RuntimeError("camera unavailable")

        def __exit__(self, *args):
            """提供上下文管理接口。"""

    monkeypatch.setattr(calibrate_umi, "NativeUmi", BrokenDevices)
    with pytest.raises(RuntimeError, match="camera unavailable"):
        calibrate_umi.collect(
            {"intrinsics": [1, 1, 0, 0], "distortion": [0] * 4}, 1
        )
    assert disconnected == [True]


@pytest.mark.parametrize("offline", [False, True])
def test_time_command_no_preview_and_offline(monkeypatch, tmp_path, offline):
    """无预览采集和离线重算均不会打开预览或额外设备。"""
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(yaml.safe_dump({"geometry_digest": "g"}))
    data = {
        "geometry_digest": "g",
        "camera_times": [],
        "observations": [],
        "tracker_times": [],
        "tracker_poses": [],
        "tracker_valid": [],
    }
    observations_file = tmp_path / "observations.json"
    observations_file.write_text(json.dumps(data))
    calls = []
    monkeypatch.setattr(calibrate_umi, "verify", lambda snapshot, **kwargs: snapshot)
    monkeypatch.setattr(
        calibrate_umi,
        "collect",
        lambda snapshot, duration, *, preview: calls.append(preview) or data,
    )
    monkeypatch.setattr(calibrate_umi, "solve_time", lambda *args: {"accepted": True})
    args = ["time", "--calibration", str(calibration)]
    if offline:
        args.extend(["--input", str(observations_file)])
    else:
        args.append("--no-preview")
    calibrate_umi.main(args)
    assert calls == ([] if offline else [False])
    assert yaml.safe_load(calibration.read_text())["time"]["accepted"]
