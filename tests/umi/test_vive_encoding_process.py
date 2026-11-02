"""验证 VIVE 流式编码子进程的限流、故障传播与资源回收。"""

import queue
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.scripts._record_encoding import _vive_rgb_encoder_config
from handumi.scripts._vive_encoding_process import ViveProcessEncoder


def _encoder():
    """创建真实软件编码子进程，不连接硬件。"""
    return ViveProcessEncoder(fps=30, rgb_encoder=_vive_rgb_encoder_config("h264"),
                              queue_size=30, encoder_threads=2, control_timeout=20)


def test_cancel_and_reuse_process_removes_temporary_data(tmp_path):
    """取消清理当前目录；同一子进程可重新开始并输出完整视频。"""
    import av

    encoder = _encoder()
    try:
        encoder.start_episode(["camera"], tmp_path)
        temporary = encoder._episode_dir
        encoder.feed_frame("camera", np.zeros((48, 64, 3), dtype=np.uint8))
        encoder.cancel_episode()
        assert not temporary.exists()
        encoder.start_episode(["camera"], tmp_path)
        for value in range(4):
            encoder.feed_frame("camera", np.full((48, 64, 3), value, dtype=np.uint8))
        result = encoder.finish_episode()
        video = result["camera"][0]
        with av.open(str(video)) as container:
            assert sum(1 for _ in container.decode(video=0)) == 4
    finally:
        encoder.close()
    assert not list(tmp_path.glob("vive-encoding-*"))
    assert not encoder._process.is_alive()


def test_child_death_fails_feed_and_close_removes_staging(tmp_path):
    """子进程意外死亡后立刻报告，关闭仍能清理并且不阻塞。"""
    encoder = _encoder()
    encoder.start_episode(["camera"], tmp_path)
    temporary = encoder._episode_dir
    encoder._process.terminate()
    encoder._process.join(timeout=5)
    assert not encoder._process.is_alive()
    with pytest.raises(RuntimeError, match="编码子进程已退出"):
        encoder.feed_frame("camera", np.zeros((48, 64, 3), dtype=np.uint8))
    started = time.monotonic()
    encoder.close()
    assert time.monotonic() - started < 3
    assert not temporary.exists()


def test_ipc_full_and_async_failure_are_not_silent():
    """IPC 队列饱和及已到达的子进程异常都不能被当作成功入队。"""
    encoder = ViveProcessEncoder.__new__(ViveProcessEncoder)
    encoder._commands = queue.Queue(maxsize=1)
    encoder._replies = queue.Queue()
    encoder._process = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    encoder._failure = None
    encoder._frame_timeout = 0.01
    encoder._commands.put("occupied")
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="队列已满"):
        encoder.feed_frame("camera", image)
    encoder._replies.put(("async_error", 0, "编码线程异常"))
    with pytest.raises(RuntimeError, match="编码线程异常"):
        encoder.feed_frame("camera", image)


def test_cancel_timeout_stops_worker_before_removing_directory(tmp_path, monkeypatch):
    """取消失败时先结束子进程，再清理暂存目录并保留取消错误。"""
    encoder = ViveProcessEncoder.__new__(ViveProcessEncoder)
    temporary = tmp_path / "staging"
    temporary.mkdir()
    encoder._episode_dir = temporary
    encoder._process = SimpleNamespace(is_alive=lambda: True)
    encoder._failure = None
    monkeypatch.setattr(encoder, "_send", Mock(side_effect=TimeoutError("取消超时")))
    stopped = []

    def shutdown():
        """确认终止时目录仍存在，避免进程继续写入已删目录。"""
        assert temporary.exists()
        stopped.append(True)

    monkeypatch.setattr(encoder, "_shutdown", shutdown)
    with pytest.raises(TimeoutError, match="取消超时"):
        encoder.cancel_episode()
    assert stopped and not temporary.exists()


def test_spawn_failure_closes_ipc_queues(monkeypatch):
    """子进程无法启动时仍关闭已建立的队列资源。"""
    from handumi.scripts import _vive_encoding_process

    command_queue, response_queue = Mock(), Mock()
    process = Mock(pid=None)
    process.start.side_effect = OSError("spawn failed")
    context = Mock()
    context.Queue.side_effect = [command_queue, response_queue]
    context.Process.return_value = process
    monkeypatch.setattr(_vive_encoding_process.mp, "get_context", lambda *_: context)
    with pytest.raises(OSError, match="spawn failed"):
        ViveProcessEncoder(fps=30, rgb_encoder=None)
    command_queue.close.assert_called_once()
    response_queue.close.assert_called_once()
