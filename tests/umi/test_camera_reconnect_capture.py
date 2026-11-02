"""验证 VIVE 录制主循环的相机断线、等待、恢复和重新录制。"""

from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from handumi.cameras.base import CameraSample
from handumi.cameras.opencv import CameraFrameUnavailable
from handumi.grippers.base import GripperObservation
from handumi.scripts import umi_capture
from handumi.tracking.base import ControllerPairSample


def test_record_discard_wait_recover_and_preserve_saved_episode(tmp_path, monkeypatch):
    """旧段保持已保存，断线段先于保存丢弃，恢复后必须重新按键。"""
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    output = tmp_path / "dataset"
    dataset = Mock()
    frames = []
    saved = [["已有已保存段"]]
    dataset.add_frame.side_effect = frames.append
    dataset.clear_episode_buffer.side_effect = frames.clear

    def save():
        """捕获完整片段并清空当前缓存。"""
        saved.append(list(frames))
        frames.clear()

    dataset.save_episode.side_effect = save

    class ImmediateExecutor:
        """让主循环在下一次迭代处理保存结果。"""

        def __init__(self, **kwargs):
            """兼容单线程执行器的构造。"""

        def submit(self, function, *args):
            """同步完成模拟保存。"""
            future = Future()
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)
            return future

        def shutdown(self, wait=True):
            """结束模拟执行器。"""

    class FakeDevice:
        """按顺序模拟首帧缺失、正常录制、断线停帧和恢复。"""

        def __init__(self, *args, **kwargs):
            """初始化观测次序。"""
            self.step = 0

        def __enter__(self):
            """启动模拟会话。"""
            return self

        def __exit__(self, *args):
            """结束模拟会话。"""

        def sample(self, now, *, after_sequence=None):
            """返回陈旧或真实新帧，重连期使用同一个旧序号。"""
            self.step += 1
            if self.step == 1:
                raise CameraFrameUnavailable("尚无首帧")
            stale = self.step in (4, 5)
            sequence = 2 if stale else self.step
            stamp = now - (1_000_000_000 if stale else 150_000_000)
            image = CameraSample(np.zeros((4, 4, 3), np.uint8), stamp, sequence)
            tracking = replace(
                ControllerPairSample.empty("vive"), right_tracked=True,
                right_pose_valid=True,
            )
            gripper = GripperObservation(0.5, True, stamp, sequence)
            return image, tracking, gripper, now - 150_000_000, not stale

    @contextmanager
    def fake_keyboard():
        """依次等待首帧、开始、写帧、断线时保存、等待、恢复及保存。"""
        keys = iter(["", " ", "", " ", " ", "", " ", "", " ", "", "\x1b"])
        yield lambda: next(keys)

    def open_dataset(*args):
        """创建清理过程读取的最小元数据。"""
        (output / "meta").mkdir(parents=True)
        (output / "meta/info.json").write_text("{}")
        return dataset

    monkeypatch.setattr(umi_capture, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {"tracker_camera": np.eye(4).tolist()})
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "NativeUmi", FakeDevice)
    monkeypatch.setattr(umi_capture, "keyboard", fake_keyboard)
    monkeypatch.setattr(umi_capture, "_open_recording_dataset", open_dataset)
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", lambda *args: None)
    monkeypatch.setattr(umi_capture, "_prepare_streaming_episode", lambda *_: {})
    monkeypatch.setattr(umi_capture.time, "sleep", lambda *_: None)
    umi_capture.main(["--calibration", str(calibration), "--output-dir", str(output)])

    assert saved[0] == ["已有已保存段"]
    assert len(saved) == 2 and len(saved[1]) == 2
    assert dataset.clear_episode_buffer.call_count == 1
    dataset.finalize.assert_called_once()


def test_status_text_shows_camera_recovery_phase():
    """状态面板区分重连、等待首帧和恢复失败。"""
    from types import SimpleNamespace

    health = umi_capture.CaptureHealth(True, True, 500_000_000, 350_000_000)
    tracking = SimpleNamespace(right_tracked=True, right_pose_valid=True)
    gripper = SimpleNamespace(valid=True, reason="")
    for status, text in (
        ("reconnecting", "正在自动重连"),
        ("waiting", "等待新画面"),
        ("error", "重连失败，请检查终端"),
    ):
        result = umi_capture.capture_status_text(
            mode="record", health=health, tracking=tracking, gripper=gripper,
            healthy=False, camera_status=status,
        )
        assert text in result and "等待设备就绪" in result
