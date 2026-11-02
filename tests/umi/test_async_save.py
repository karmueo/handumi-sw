"""验证 Vive 录制的后台保存、实时预览及退出时资源顺序。"""

import json
import os
import signal
import threading
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.cameras.base import CameraSample
from handumi.grippers.base import GripperObservation
from handumi.scripts import umi_capture
from handumi.tracking.base import ControllerPairSample


class CaptureHarness:
    """通过事件控制后台写入，用合成设备运行真实录制主循环。"""

    def __init__(self, tmp_path, monkeypatch, *, preview=True, fail=False):
        """设置设备、数据集及 Viewer 替身，不连接实际硬件。"""
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.statuses = []
        self.preview_calls = []
        self.events = []
        self.frames = []
        self.saved = []
        self.read = lambda: ""
        self.sequence = 0
        self.healthy = True
        self.sample_hook = None
        self.main_thread = threading.get_ident()
        self.fail = fail
        self.output = tmp_path / "dataset"
        calibration = tmp_path / "calibration.yaml"
        calibration.write_text("{}")
        self.args = ["--calibration", str(calibration), "--output-dir", str(self.output)]
        if not preview:
            self.args.append("--no-preview")
        self.viewer = Mock() if preview else None
        self.dataset = Mock()

        def check_idle(name):
            """所有非保存操作都不得与后台保存同时执行。"""
            assert not self.started.is_set() or self.finished.is_set(), name
            self.events.append(name)

        def add(frame):
            """记录每段帧并检查不会在保存期间追加。"""
            check_idle("add")
            self.frames.append(frame)

        def clear():
            """清除当前片段并检查互斥。"""
            check_idle("clear")
            self.frames.clear()

        def save():
            """等待主线程放行，模拟耗时或失败的真实保存。"""
            self.events.append("save_start")
            self.started.set()
            try:
                if not self.release.wait(5):
                    raise TimeoutError("测试未放行保存线程")
                if isinstance(self.fail, BaseException):
                    raise self.fail
                if self.fail:
                    raise OSError("模拟保存故障")
                self.saved.append(self.frames.copy())
                self.frames.clear()
            finally:
                self.events.append("save_end")
                self.finished.set()

        def finalize():
            """验证在后台线程退出后才关闭数据集。"""
            check_idle("finalize")

        self.dataset.add_frame.side_effect = add
        self.dataset.clear_episode_buffer.side_effect = clear
        self.dataset.save_episode.side_effect = save
        self.dataset.finalize.side_effect = finalize
        harness = self

        class Device:
            """按测试控制的健康状态不断产生新样本。"""

            def __init__(self, *args, **kwargs):
                """不打开硬件。"""

            def __enter__(self):
                """进入模拟设备会话。"""
                return self

            def __exit__(self, *args):
                """退出模拟设备会话。"""

            def sample(self, now, *, after_sequence=None):
                """生成具有独立相机序号的同步样本。"""
                harness.sequence += 1
                if harness.sample_hook is not None:
                    harness.sample_hook()
                stamp = now - 150_000_000
                healthy = harness.healthy
                image = CameraSample(np.zeros((4, 4, 3), np.uint8), stamp, harness.sequence)
                pose = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
                tracking = replace(
                    ControllerPairSample.empty("vive"), right_tracked=healthy,
                    right_pose_valid=healthy, right_controller_pose=pose, right_tcp_pose=pose,
                )
                gripper = GripperObservation(0.5, healthy, stamp, harness.sequence)
                return image, tracking, gripper, stamp, healthy

        @contextmanager
        def keyboard():
            """以测试提供的函数模拟终端按键。"""
            yield lambda: harness.read()

        def open_dataset(*args):
            """准备收尾时需要读取的元数据。"""
            (self.output / "meta").mkdir(parents=True)
            (self.output / "meta/info.json").write_text(json.dumps({}))
            return self.dataset

        def log_status(viewer, **status):
            """记录实时状态以及调用线程。"""
            assert threading.get_ident() == self.main_thread
            self.statuses.append(status.copy())

        def log_preview(*args, **kwargs):
            """验证预览始终在主线程刷新。"""
            assert threading.get_ident() == self.main_thread
            self.preview_calls.append(self.sequence)

        monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {"tracker_camera": np.eye(4).tolist()})
        monkeypatch.setattr(umi_capture, "metadata", lambda *a: {})
        monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})
        monkeypatch.setattr(umi_capture, "NativeUmi", Device)
        monkeypatch.setattr(umi_capture, "keyboard", keyboard)
        monkeypatch.setattr(
            umi_capture, "_prepare_streaming_episode",
            lambda *_: {"encoder_flush_s": 0.0, "video_verify_s": 0.0},
        )
        monkeypatch.setattr(umi_capture, "_open_recording_dataset", open_dataset)
        monkeypatch.setattr(umi_capture, "_start_camera_viewer", lambda *a: self.viewer)
        monkeypatch.setattr(umi_capture, "log_capture_status", log_status)
        monkeypatch.setattr(umi_capture, "log_live_preview", log_preview)

    def run(self):
        """运行录制，异常时保证后台测试线程不会被永久阻塞。"""
        try:
            umi_capture.main(self.args)
        finally:
            self.release.set()


def test_preview_keeps_up_and_space_during_save_is_discarded(tmp_path, monkeypatch):
    """保存期间继续采样、显示失效状态并忽略空格，完成后可开始新段。"""
    harness = CaptureHarness(tmp_path, monkeypatch)
    count = 0

    def read():
        """控制保存期间的预览、失效、恢复及下一段录制。"""
        nonlocal count
        count += 1
        if count == 1:
            return " "
        if count == 2:
            return " "
        if count == 3:
            assert harness.started.wait(2)
            return " "
        if count == 4:
            harness.healthy = False
        if count == 6:
            harness.healthy = True
        if count == 8:
            assert len([s for s in harness.statuses if s.get("saving")]) >= 2
            assert len(harness.preview_calls) >= 3
            assert any(not s["healthy"] and s.get("saving") for s in harness.statuses)
            harness.release.set()
        if count > 8 and any(s["saved"] == 1 for s in harness.statuses):
            if not any(s["recording"] and s["saved"] == 1 for s in harness.statuses):
                return " "
            return "\x1b" if harness.dataset.clear_episode_buffer.call_count == 0 else "\x03"
        assert count < 120
        return ""

    harness.read = read
    harness.run()
    assert len(harness.saved) == 1 and len(harness.saved[0]) == 1
    assert harness.dataset.save_episode.call_count == 1
    assert harness.dataset.clear_episode_buffer.call_count == 1
    assert harness.events.index("save_end") < harness.events.index("finalize")
    assert harness.viewer.disconnect.call_count == 1


@pytest.mark.parametrize("control", ["escape", "ctrl_c_key", "sigint", "keyboard_interrupt"])
def test_exit_during_save_waits_and_keeps_preview(tmp_path, monkeypatch, control):
    """退出请求和重复中断都不终止在途保存，完成前继续刷新。"""
    harness = CaptureHarness(tmp_path, monkeypatch)
    count = 0

    def read():
        """在后台保存期间发出退出请求，再观察状态后放行。"""
        nonlocal count
        count += 1
        if count in (1, 2):
            return " "
        if count in (3, 4):
            assert harness.started.wait(2)
            if control == "escape":
                return "\x1b"
            if control == "ctrl_c_key":
                return "\x03"
            if control == "sigint":
                os.kill(os.getpid(), signal.SIGINT)
                return ""
            raise KeyboardInterrupt
        if count == 7:
            assert any(s.get("saving") and s.get("exit_requested") for s in harness.statuses)
            assert harness.sequence >= 7
            harness.release.set()
        assert count < 120
        return ""

    harness.read = read
    previous = signal.getsignal(signal.SIGINT)
    harness.run()
    assert signal.getsignal(signal.SIGINT) == previous
    assert len(harness.saved) == 1
    assert harness.dataset.clear_episode_buffer.call_count == 0
    assert harness.events.index("save_end") < harness.events.index("finalize")


def test_failed_save_wins_over_finalize_error(tmp_path, monkeypatch):
    """后台保存失败被报告，且不会被关闭数据集时的错误覆盖。"""
    harness = CaptureHarness(tmp_path, monkeypatch, fail=True)
    count = 0

    def read():
        """放行失败任务；后续循环收取后台异常。"""
        nonlocal count
        count += 1
        if count in (1, 2):
            return " "
        if count == 3:
            assert harness.started.wait(2)
            harness.release.set()
        assert count < 120
        return ""

    def bad_finalize():
        """模拟收尾阶段同时出现错误。"""
        assert harness.finished.is_set()
        raise ValueError("模拟收尾故障")

    harness.read = read
    harness.dataset.finalize.side_effect = bad_finalize
    with pytest.raises(RuntimeError, match="保存当前段失败：模拟保存故障") as exc:
        harness.run()
    assert isinstance(exc.value.__cause__, OSError)
    assert harness.dataset.save_episode.call_count == 1
    assert harness.dataset.add_frame.call_count == 1
    assert harness.dataset.clear_episode_buffer.call_count == 0
    assert not any(s["saved"] for s in harness.statuses)
    harness.viewer.disconnect.assert_called_once()


def test_no_preview_episode_limit_waits_for_save(tmp_path, monkeypatch):
    """禁用预览且达到段数上限时，保存结束后才退出。"""
    harness = CaptureHarness(tmp_path, monkeypatch, preview=False)
    harness.args += ["--num-episodes", "1"]
    count = 0

    def read():
        """等待采样继续后完成后台保存。"""
        nonlocal count
        count += 1
        if count in (1, 2):
            return " "
        if count == 3:
            assert harness.started.wait(2)
        if count == 6:
            harness.release.set()
        assert count < 120
        return ""

    harness.read = read
    harness.run()
    assert count >= 6 and len(harness.saved) == 1
    assert harness.events.index("save_end") < harness.events.index("finalize")


def test_worker_interrupt_during_exception_cleanup_has_priority(tmp_path, monkeypatch):
    """采样异常触发清理时，后台抛出的 KeyboardInterrupt 不会无限等待。"""
    harness = CaptureHarness(tmp_path, monkeypatch, fail=KeyboardInterrupt("后台故障"))
    count = 0

    def read():
        """先开始并提交保存。"""
        nonlocal count
        count += 1
        return " " if count <= 2 else ""

    def fail_sample():
        """保存已启动后让主循环异常，同时放行后台线程。"""
        if harness.sequence == 3:
            assert harness.started.wait(2)
            harness.release.set()
            raise ValueError("采样故障")

    harness.read = read
    harness.sample_hook = fail_sample
    with pytest.raises(RuntimeError, match="保存当前段失败：后台故障") as exc:
        harness.run()
    assert isinstance(exc.value.__cause__, KeyboardInterrupt)
    assert harness.dataset.finalize.call_count == 1
    assert harness.events.index("save_end") < harness.events.index("finalize")


def test_sigint_during_exception_cleanup_waits_for_save(tmp_path, monkeypatch):
    """采样失败后收到真实重复 SIGINT，也须先等待在途保存结束。"""
    harness = CaptureHarness(tmp_path, monkeypatch)
    sample_failed = threading.Event()
    count = 0

    def read():
        """开始一段并提交保存。"""
        nonlocal count
        count += 1
        return " " if count <= 2 else ""

    def fail_sample():
        """触发采样错误，主线程随后进入清理等待。"""
        if harness.sequence == 3:
            assert harness.started.wait(2)
            sample_failed.set()
            raise ValueError("采样故障")

    def interrupt_and_release():
        """在后台保存尚未完成时向主线程发出多次真实信号。"""
        assert sample_failed.wait(2)
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGINT)
        harness.release.set()

    harness.read = read
    harness.sample_hook = fail_sample
    helper = threading.Thread(target=interrupt_and_release)
    helper.start()
    previous = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(ValueError, match="采样故障"):
            harness.run()
    finally:
        helper.join(timeout=3)
    assert not helper.is_alive()
    assert signal.getsignal(signal.SIGINT) == previous
    assert len(harness.saved) == 1
    assert harness.events.index("save_end") < harness.events.index("finalize")


def test_no_fresh_frame_rejects_save_without_committing(tmp_path, monkeypatch):
    """没有新相机帧时按保存仍须丢段，且不调用后台保存。"""
    harness = CaptureHarness(tmp_path, monkeypatch)
    keys = iter([" ", " ", "\x1b"])
    harness.read = lambda: next(keys)

    def duplicate():
        """第二轮采样沿用上次成功写入的相机序号。"""
        if harness.sequence == 2:
            harness.sequence = 1

    harness.sample_hook = duplicate
    harness.run()
    assert harness.dataset.add_frame.call_count == 1
    assert harness.dataset.clear_episode_buffer.call_count == 1
    harness.dataset.save_episode.assert_not_called()
    assert any("相机帧重复" in status["last_result"] for status in harness.statuses)
