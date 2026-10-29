from pathlib import Path

import pytest

from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.cameras.usb import build_camera_specs, make_camera_device


def _write_rig(path: Path, fourcc: str) -> None:
    path.write_text(
        "\n".join(
            (
                "cameras:",
                "  left_wrist:",
                "    type: opencv",
                "    index_or_path: /dev/video0",
                "    width: 1280",
                "    height: 960",
                "    fps: 30",
                f"    fourcc: {fourcc}",
            )
        )
    )


def test_opencv_fourcc_flows_from_rig_to_device(tmp_path: Path) -> None:
    rig = tmp_path / "rig.yaml"
    _write_rig(rig, "mjpg")

    specs, _ = build_camera_specs(
        ["/dev/video0"],
        camera_names=["left_wrist"],
        laptop_camera=False,
        laptop_cam_id=0,
        laptop_cam_name="laptop",
        rig_config=rig,
    )
    camera = make_camera_device(specs[0])

    assert specs[0]["fourcc"] == "MJPG"
    assert isinstance(camera, OpenCVCameraDevice)
    assert camera.fourcc == "MJPG"


def test_invalid_opencv_fourcc_is_rejected(tmp_path: Path) -> None:
    rig = tmp_path / "rig.yaml"
    _write_rig(rig, "jpegx")

    with pytest.raises(SystemExit, match="four-character code"):
        build_camera_specs(
            ["/dev/video0"],
            camera_names=["left_wrist"],
            laptop_camera=False,
            laptop_cam_id=0,
            laptop_cam_name="laptop",
            rig_config=rig,
        )


def test_failed_camera_connect_can_be_cleaned_up(monkeypatch) -> None:
    """底层打开失败后，清理不会用未连接异常掩盖原始连接错误。"""
    import lerobot.cameras.opencv as opencv
    from lerobot.utils.errors import DeviceNotConnectedError

    class BrokenCamera:
        """模拟无法打开的 LeRobot 相机。"""

        def __init__(self, config):
            """接收相机配置。"""

        def connect(self):
            """报告原始打开错误。"""
            raise ConnectionError("camera open failed")

        def disconnect(self):
            """模拟底层已释放未打开设备的情况。"""
            raise DeviceNotConnectedError("camera not connected")

    monkeypatch.setattr(opencv, "OpenCVCamera", BrokenCamera)
    camera = OpenCVCameraDevice(0, 30, 640, 480)
    with pytest.raises(ConnectionError, match="camera open failed"):
        camera.connect()
    camera.disconnect()
    camera.disconnect()
    assert camera._camera is None
