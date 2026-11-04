"""验证批量回放进度的限频、非阻塞及非交互输出。"""

from __future__ import annotations

import io
import multiprocessing
import os
import queue
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import pytest
from rich.console import Console

from handumi.scripts.replay import replay_progress, replay_scheduler


def _flood_worker():
    """在真实 spawn worker 中持续写入无人消费的小队列。"""
    progress = replay_progress.WorkerProgress(0, replay_scheduler._PROGRESS_EVENTS, interval=0)
    for index in range(20000):
        progress("逐帧求解", current=index, total=20000, payload="x" * 65536)
    return progress.finish()["elapsed"]


def _spawn_flood_pool():
    """在独立监控进程中等待真实工作进程和事件队列关闭。"""
    context = multiprocessing.get_context("spawn")
    events = context.Queue(maxsize=2)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=context,
        initializer=replay_scheduler._worker_init, initargs=(events,),
    ) as executor:
        assert executor.submit(_flood_worker).result(timeout=20) >= 0
    events.close()
    events.cancel_join_thread()


def test_spawn_queue_flood_exits_without_consumer():
    """真实 spawn 队列饱和且无人消费时，进程池依然能够退出。"""
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_spawn_flood_pool)
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise AssertionError("进程池在饱和进度队列下未退出")
    assert process.exitcode == 0


def test_worker_progress_limits_and_drops_full_queue(monkeypatch):
    """持续帧事件限频，队列满时仍继续并可靠累计阶段时间。"""
    clock = [100.0]
    monkeypatch.setattr(replay_progress.time, "monotonic", lambda: clock[0])
    events = queue.Queue(maxsize=1)
    progress = replay_progress.WorkerProgress(3, events)
    progress("逐帧求解", current=1, total=100)
    for frame in range(2, 30):
        clock[0] += 0.001
        progress("逐帧求解", current=frame, total=100)
    assert events.qsize() == 1
    clock[0] += 0.3
    progress("逐帧求解", current=30, total=100)
    progress("严格验收")
    assert events.qsize() == 1
    summary = progress.finish()
    assert summary["elapsed"] > 0.3
    assert summary["stage_seconds"]["逐帧求解"] > 0.3


def test_non_terminal_progress_and_off():
    """非 TTY 只写纯文本；关闭进度时无输出。"""
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)
    scheduler = SimpleNamespace(
        drain_events=lambda: {1: {"stage": "逐帧求解", "current": 2, "total": 9}},
        running={object(): 1}, completed={0}, failures={}, cache_hits=1,
    )
    with replay_progress.BatchProgress(3, "auto", console=console) as progress:
        progress.update(scheduler, force=True)
    output = stream.getvalue()
    assert "1/3" in output and "2/9" in output and "\x1b" not in output
    stream.seek(0)
    stream.truncate()
    with replay_progress.BatchProgress(3, "off", console=console) as progress:
        progress.update(scheduler, force=True)
    assert stream.getvalue() == ""


def test_worker_stderr_is_bounded_and_restored():
    """原生 stderr 连续写入时持续排空，超额只保留有限内容。"""
    with replay_scheduler._BoundedStderr(limit=100) as capture:
        os.write(2, b"e" * 50000)
    assert len(capture.data) == 100 and capture.dropped == 49900
    assert "truncated" in capture.text()


def test_terminal_progress_translates_cache_and_closes_on_interrupt():
    """交互终端使用中文阶段，异常时关闭动态显示而不伪造完成数。"""
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None)
    scheduler = SimpleNamespace(
        drain_events=lambda: {0: {"episode": 0, "stage": "cache_hit"}},
        running={object(): 0}, completed=set(), failures={}, cache_hits=0,
    )
    progress = replay_progress.BatchProgress(1, "on", console=console)
    with pytest.raises(KeyboardInterrupt):
        with progress:
            progress.update(scheduler)
            raise KeyboardInterrupt()
    assert progress.live is not None and not progress.live.is_started
    assert "缓存命中" in stream.getvalue()


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_dataset_progress_non_terminal_limits_repeated_updates(monkeypatch, mode):
    """非交互终端在阶段切换时输出，并对频繁帧事件限频。"""
    clock = [100.0]
    monkeypatch.setattr(replay_progress.time, "monotonic", lambda: clock[0])
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)
    with replay_progress.DatasetPlacementProgress(mode, console=console) as display:
        display.update("准备输入", 0, 2, detail="episode 1/2")
        first = stream.getvalue()
        assert "准备输入 0/2" in first
        clock[0] += 1
        display.update("准备输入", 1, 2, detail="episode 2/2")
        assert stream.getvalue() == first
        clock[0] += 5
        display.update("准备输入", 2, 2)
        display.update("生成候选", 0, 16, detail="朝向 1/8")
        display.update("代表帧预筛", 0, 8, detail="候选 1/8", frame_current=2, frame_total=9)
        display.update("完整轨迹验证", 1, 16, detail="候选 1/8 · episode 2/2",
                       frame_current=3, frame_total=20)
    output = stream.getvalue()
    assert "准备输入 2/2" in output and "生成候选 0/16" in output
    assert "代表帧预筛 0/8" in output and "帧 2/9" in output
    assert "完整轨迹验证 1/16" in output and "帧 3/20" in output
    assert "\x1b" not in output


def test_dataset_progress_off_suppresses_all_status():
    """关闭模式不会启动动态显示，也不会输出各阶段文本。"""
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None)
    with replay_progress.DatasetPlacementProgress("off", console=console) as display:
        display.update("准备输入", 0, 2, force=True)
        display.update("完整轨迹验证", 1, 8, frame_current=5, frame_total=10)
    assert display.live is None and stream.getvalue() == ""


def test_dataset_terminal_progress_closes_on_error_without_fake_completion():
    """真实动态进度可展示当前帧，并在异常后关闭且保留真实完成量。"""
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None)
    display = replay_progress.DatasetPlacementProgress("on", console=console)
    with pytest.raises(RuntimeError, match="failure"):
        with display:
            display.update("完整轨迹验证", 1, 8, detail="候选 2/8 · episode 1/2",
                           frame_current=3, frame_total=10, force=True)
            assert display.progress.tasks[display.task].completed == 1
            assert display.progress.tasks[display.frame_task].completed == 3
            raise RuntimeError("failure")
    assert display.live is not None and not display.live.is_started
    assert display.progress.tasks[display.task].completed == 1
    assert "完整轨迹验证" in stream.getvalue()
    assert "候选 2/8" in stream.getvalue()
