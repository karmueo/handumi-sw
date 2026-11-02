"""验证 USB 相机停帧后的后台重连、帧代次及退出清理。"""

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from handumi.cameras.opencv import CameraFrameUnavailable, OpenCVCameraDevice


class FakeBackend:
    """模拟带锁、真实时间戳及可控连接行为的相机后端。"""

    def __init__(self, *, fail=False, gate=None, timestamp=None):
        """设置打开结果及首帧。"""
        self.frame_lock = threading.Lock()
        self.latest_frame = None if timestamp is None else np.ones((2, 4, 3), np.uint8)
        self.latest_timestamp = timestamp
        self.fail = fail
        self.gate = gate
        self.entered = threading.Event()
        self.opened = threading.Event()
        self.closed = threading.Event()
        self.close_count = 0

    def connect(self):
        """按测试事件控制打开完成。"""
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(3)
        if self.fail:
            raise ConnectionError("模拟打开失败")
        self.opened.set()

    def disconnect(self):
        """记录资源释放。"""
        self.close_count += 1
        self.closed.set()

    def frame(self, timestamp, value):
        """发布一张新的底层图像。"""
        with self.frame_lock:
            self.latest_timestamp = timestamp
            self.latest_frame = np.full((2, 4, 3), value, np.uint8)


def wait_for(predicate, timeout=3):
    """等待跨线程条件成立，超时给出清晰失败。"""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("等待相机测试条件超时")
        time.sleep(0.002)


def make_device(monkeypatch, backends):
    """注入一组顺序使用的后端并缩短重试时间。"""
    device = OpenCVCameraDevice("/dev/v4l/by-path/camera-video-index0", 30, 4, 2, "MJPG")
    device._stall_timeout_s = 0.08
    device._retry_interval_s = 0.02
    remaining = deque(backends)

    def create():
        """获取下一个后端，耗尽时模拟设备仍未出现。"""
        if remaining:
            return remaining.popleft()
        raise ConnectionError("设备未出现")

    monkeypatch.setattr(device, "_make_backend", create)
    return device


def test_reconnect_preserves_stale_frame_and_replaces_generation(monkeypatch):
    """停帧不刷新旧时间；失败清理后同时间戳的新代次仍递增发布。"""
    stamp = time.perf_counter()
    first = FakeBackend(timestamp=stamp)
    failed = FakeBackend(fail=True)
    recovered = FakeBackend(timestamp=stamp)
    device = make_device(monkeypatch, [first, failed, recovered])
    device.connect()
    try:
        wait_for(lambda: device.status == "streaming")
        old = device.sample_at()
        assert first.closed.wait(2)
        assert device.sample_at() is old
        assert failed.closed.wait(2)
        wait_for(lambda: recovered.opened.is_set() and device.status == "streaming")
        new = device.sample_at(old.capture_time_ns, after_sequence=old.sequence)
        assert new.sequence > old.sequence
        assert device.sample_at(old.capture_time_ns).sequence == new.sequence
        assert len(device._samples) == 1
        assert failed.close_count == 1
    finally:
        device.disconnect()
    assert first.close_count == recovered.close_count == 1
    assert device.status == "closed"


def test_no_initial_frame_is_explicit_and_reconnects(monkeypatch):
    """首次无帧会重连，调用方持续收到有界无帧异常直至真实画面恢复。"""
    first = FakeBackend()
    gate = threading.Event()
    recovered = FakeBackend(gate=gate, timestamp=time.perf_counter())
    device = make_device(monkeypatch, [first, recovered])
    device.connect()
    try:
        assert recovered.entered.wait(2)
        with pytest.raises(CameraFrameUnavailable):
            device.sample_at()
        gate.set()
        wait_for(lambda: device.status == "streaming")
        assert device.sample_at().sequence == 1
    finally:
        gate.set()
        device.disconnect()


def test_disconnect_during_slow_open_cleans_late_candidate(monkeypatch):
    """关闭与后台打开交错时，不发布迟到帧并释放刚打开的后端。"""
    first = FakeBackend(timestamp=time.perf_counter())
    gate = threading.Event()
    second = FakeBackend(gate=gate, timestamp=time.perf_counter())
    device = make_device(monkeypatch, [first, second])
    device.connect()
    assert second.entered.wait(2)
    stopped = threading.Event()
    errors = []

    def close():
        """在单独线程中等待相机关闭。"""
        try:
            device.disconnect()
        except Exception as error:
            errors.append(error)
        finally:
            stopped.set()

    thread = threading.Thread(target=close)
    thread.start()
    wait_for(lambda: device.status == "closed")
    gate.set()
    assert stopped.wait(2)
    thread.join()
    assert not errors
    assert second.close_count == 1
    assert device._camera is None and not device._samples
    with pytest.raises(RuntimeError, match="not connected"):
        device.sample_at()


def test_real_configuration_preserves_path_and_capture_settings(monkeypatch):
    """每次构造均使用原始设备路径、编码及采集参数。"""
    import lerobot.cameras.opencv as opencv

    configs = []
    monkeypatch.setattr(opencv, "OpenCVCamera", lambda config: configs.append(config))
    device = OpenCVCameraDevice("/dev/v4l/by-path/camera-video-index0", 30, 4, 2, "MJPG")
    device._make_backend()
    device._make_backend()
    assert len(configs) == 2
    assert all(config.index_or_path == Path(device.index_or_path) for config in configs)
    assert all((c.fps, c.width, c.height, c.fourcc) == (30, 4, 2, "MJPG") for c in configs)


def test_repeated_reconnect_preserves_transform_and_sequence(monkeypatch):
    """连续断线都应用图像处理钩子，且帧序号跨各代递增。"""
    from handumi.cameras.zedmini import ZedMiniCameraDevice

    backends = [FakeBackend(timestamp=time.perf_counter()) for _ in range(3)]
    device = ZedMiniCameraDevice("/dev/v4l/by-id/camera", 30, 4, 2)
    device._stall_timeout_s = 0.12
    device._retry_interval_s = 0.02
    remaining = deque(backends)
    monkeypatch.setattr(device, "_make_backend", lambda: remaining.popleft())
    device.connect()
    try:
        sequences = []
        for backend in backends:
            wait_for(lambda: backend.opened.is_set() and device._camera is backend)
            wait_for(lambda: device.status == "streaming")
            sample = device.sample_at()
            sequences.append(sample.sequence)
            assert sample.image.shape == (2, 2, 3)
        assert sequences == [1, 2, 3]
    finally:
        device.disconnect()
    assert [backend.close_count for backend in backends] == [1, 1, 1]


def test_pending_driver_reader_prevents_reopen_and_reports_on_shutdown(monkeypatch):
    """旧底层读线程未退出时不重开，退出也不会声称已完全清理。"""
    finish_reader = threading.Event()
    reader_started = threading.Event()

    def read_forever():
        """模拟 USB 驱动中的阻塞读取。"""
        reader_started.set()
        finish_reader.wait(3)

    reader = threading.Thread(target=read_forever, daemon=True)
    reader.start()
    assert reader_started.wait(1)
    first = FakeBackend(timestamp=time.perf_counter())
    first.thread = reader
    second = FakeBackend(timestamp=time.perf_counter())
    device = make_device(monkeypatch, [first, second])
    device.connect()
    try:
        assert first.closed.wait(2)
        with pytest.raises(TimeoutError, match="读取线程尚未退出"):
            device.disconnect()
        assert not second.entered.is_set()
        with pytest.raises(RuntimeError, match="read thread is still running"):
            device.connect()
    finally:
        finish_reader.set()
        reader.join(2)
        device.disconnect()


def test_waiting_status_is_set_before_first_frame(monkeypatch):
    """首帧到来前正确暴露等待状态。"""
    first = FakeBackend()
    device = make_device(monkeypatch, [first])
    device._stall_timeout_s = 5
    device.connect()
    try:
        assert device.status == "waiting"
    finally:
        device.disconnect()


def test_failed_open_keeps_live_reader_until_it_exits(monkeypatch):
    """失败后仍存活的读取线程退出前，不再次打开设备。"""
    release = threading.Event()
    reader = threading.Thread(target=lambda: release.wait(3), daemon=True)
    reader.start()
    first = FakeBackend(timestamp=time.perf_counter())
    failed = FakeBackend(fail=True)
    failed.thread = reader
    last = FakeBackend(timestamp=time.perf_counter())
    device = make_device(monkeypatch, [first, failed, last])
    device.connect()
    try:
        assert failed.closed.wait(2)
        assert not last.entered.wait(0.1)
        release.set()
        reader.join(2)
        assert last.entered.wait(2)
        wait_for(lambda: device.status == "streaming")
    finally:
        release.set()
        reader.join(2)
        device.disconnect()


def test_failed_cleanup_retains_backend_for_retry(monkeypatch):
    """后端关闭失败时保留引用并阻止再次连接，允许退出时重试清理。"""
    first = FakeBackend(timestamp=time.perf_counter())
    failed = FakeBackend(fail=True)
    attempts = []

    def close():
        """模拟两次清理失败后恢复。"""
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("模拟释放失败")
        failed.closed.set()

    failed.disconnect = close
    device = make_device(monkeypatch, [first, failed])
    device.connect()
    assert failed.entered.wait(2)
    wait_for(lambda: device.status == "error")
    assert device._unfinished_backend is failed
    with pytest.raises(RuntimeError, match="not fully released"):
        device.connect()
    device.disconnect()
    assert failed.closed.is_set() and device._unfinished_backend is None
    assert len(attempts) == 2


def test_disconnect_while_backend_factory_blocked(monkeypatch):
    """工厂调用晚返回时，停止信号阻止其结果成为活动相机。"""
    first = FakeBackend(timestamp=time.perf_counter())
    second = FakeBackend(timestamp=time.perf_counter())
    factory_entered = threading.Event()
    factory_gate = threading.Event()
    device = make_device(monkeypatch, [])
    calls = []

    def create():
        """首次直接返回，重连时等待退出请求。"""
        calls.append(1)
        if len(calls) == 1:
            return first
        factory_entered.set()
        assert factory_gate.wait(3)
        return second

    monkeypatch.setattr(device, "_make_backend", create)
    device.connect()
    assert factory_entered.wait(2)
    errors = []

    def close():
        """异步请求相机关闭。"""
        try:
            device.disconnect()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=close)
    thread.start()
    wait_for(lambda: device.status == "closed")
    factory_gate.set()
    thread.join(2)
    assert not thread.is_alive() and not errors
    assert second.close_count == 1 and device._camera is None
    assert not device._samples


def test_initial_connect_and_cleanup_failure_preserves_backend(monkeypatch):
    """首次打开失败且清理失败时保留对象，并在下一次关闭时重试。"""
    backend = FakeBackend(fail=True)
    calls = []

    def close():
        """第一次报告释放失败，第二次完成释放。"""
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("模拟释放失败")
        backend.closed.set()

    backend.disconnect = close
    device = make_device(monkeypatch, [backend])
    with pytest.raises(ConnectionError, match="模拟打开失败") as error:
        device.connect()
    assert "清理也失败" in error.value.__notes__[0]
    assert device._unfinished_backend is backend
    with pytest.raises(RuntimeError, match="not fully released"):
        device.connect()
    device.disconnect()
    assert backend.closed.is_set() and len(calls) == 2


@pytest.mark.parametrize(
    ("path", "should_warn"),
    [(0, True), ("/dev/video12", True), ("/dev/v4l/by-path/camera", False)],
)
def test_unstable_device_path_warning_only_for_direct_nodes(monkeypatch, caplog, path, should_warn):
    """停帧重连时仅对可能变更身份的数字编号建议使用稳定路径。"""
    first = FakeBackend(timestamp=time.perf_counter())
    device = make_device(monkeypatch, [first])
    device.index_or_path = path
    device.connect()
    try:
        assert first.closed.wait(2)
        messages = [record.message for record in caplog.records]
        warnings = [message for message in messages if "稳定路径" in message]
        assert bool(warnings) is should_warn
        assert len(warnings) <= 1
    finally:
        device.disconnect()
