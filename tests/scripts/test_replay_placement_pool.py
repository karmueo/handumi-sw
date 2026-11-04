"""验证共享摆放并行调度的排序、错误传播与输入隔离。"""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from handumi.scripts.replay import replay_placement_pool as placement_pool


def _pool():
    """创建含两段纯输入的可测试调度器。"""
    context = {name: None for name in placement_pool.PlacementPool._FIELDS}
    args = SimpleNamespace(robot="rm75b", _placement_scratch_dir="/tmp/scratch",
                           _episode_runtime=object())
    return placement_pool.PlacementPool(args, [context, context], None, 2)


def test_pool_keeps_only_explicitly_allowed_private_arguments():
    """只发送正式参数及轨迹暂存目录，不发送模型和调用回调。"""
    pool = _pool()
    assert pool.values == {"robot": "rm75b", "_placement_scratch_dir": "/tmp/scratch"}
    assert "input_identity" in pool.contexts[0]


def test_map_preserves_order_and_propagates_failure(monkeypatch):
    """任务乱序完成时结果仍按输入排序，异常向协调者传播。"""
    pool = _pool()
    seen = []

    def task(value):
        """使首项稍晚完成并在负数任务时模拟 worker 故障。"""
        if value == 0:
            time.sleep(0.03)
        if value < 0:
            raise RuntimeError("worker error")
        seen.append(value)
        return value * 2

    monkeypatch.setattr(placement_pool, "_task", task)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pool.executor = executor
        assert pool._map(iter([0, 1, 2]), 3, "测试") == [0, 2, 4]
        assert seen[0] == 1
        with pytest.raises(RuntimeError, match="worker error"):
            pool._map(iter([-1]), 1, "测试")


def test_aggregates_sample_scores_in_candidate_and_episode_order(monkeypatch):
    """多段计分之和不能因为 worker 完成顺序而对应到错误候选。"""
    pool = _pool()
    monkeypatch.setattr(pool, "_map", lambda tasks, total, stage: [(1, 2), (0, 3), (1, 4), (1, 5)])
    assert pool.sample_scores([object(), object()]) == [(1, 5), (2, 9)]


def test_first_frames_uses_original_eight_yaw_poses(monkeypatch):
    """八个朝向及同朝向内的 episode 次序与串行生成一致。"""
    pool = _pool()

    def capture(tasks, total, stage):
        """返回每个任务的朝向四元数和 episode 编号。"""
        return [(episode, pose.copy()) for _, episode, pose in tasks]

    monkeypatch.setattr(pool, "_map", capture)
    results = pool.first_frames(np.array([0, 0, 0, 0, 0, 0, 1]))
    assert len(results) == 8
    for yaw, per_episode in enumerate(results):
        assert [entry[0] for entry in per_episode] == [0, 1]
        np.testing.assert_allclose(per_episode[0][1][3:],
                                   [0, 0, np.sin(yaw * np.pi / 8), np.cos(yaw * np.pi / 8)])


def test_spawn_worker_propagates_failure_and_closes():
    """真正的 spawn worker 可接收纯快照并向父进程传递异常。"""
    pool = _pool()
    with pytest.raises(ValueError, match="Unknown placement task"):
        with pool:
            pool._map(iter([("bad", 0, None)]), 1, "测试")
    assert pool.executor is None


def test_worker_progress_is_nonblocking_and_forwarded(monkeypatch):
    """worker 报告帧进度，父进程在等待任务时显示且完成数不变。"""
    import queue

    events = queue.Queue(maxsize=2)
    shown = []

    class Recorder:
        """记录实际送到显示器的帧级状态。"""
        def update(self, *args, **kwargs):
            """保存父进程汇总的进度。"""
            shown.append((args, kwargs))

    monkeypatch.setattr(placement_pool, "_EVENTS", events)
    monkeypatch.setattr(placement_pool, "_LAST_PROGRESS", 0.0)
    monkeypatch.setattr(placement_pool, "_LAST_STAGE", None)
    placement_pool._emit(0, "逐帧求解", 12, 100)
    # 满队列不能阻塞 worker。
    events.put_nowait((1, "逐帧求解", 4, 40))
    placement_pool._emit(0, "新阶段", 0, 100)
    pool = _pool()
    pool.events = events
    pool.progress = Recorder()
    pool._drain_progress("完整轨迹验证", 1, 4)
    assert len(shown) == 2
    assert shown[0][0] == ("完整轨迹验证", 1, 4)
    assert shown[0][1]["frame_current"] == 12
    assert shown[0][1]["frame_total"] == 100


def test_abort_terminates_running_spawn_worker():
    """异常退出应终止正在执行的长任务，不等待它自然完成。"""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    pool = _pool()
    pool.executor = ProcessPoolExecutor(max_workers=1,
                                        mp_context=multiprocessing.get_context("spawn"))
    future = pool.executor.submit(time.sleep, 20)
    deadline = time.monotonic() + 5
    while not future.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.running()
    started = time.monotonic()
    pool.__exit__(RuntimeError, RuntimeError("abort"), None)
    assert time.monotonic() - started < 5
    assert pool.executor is None
