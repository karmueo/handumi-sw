"""验证离线任务的并行顺序、进度事件及进程清理。"""

from __future__ import annotations

import multiprocessing
import os
import time

import pytest

from handumi.dataset.parallel import (
    MIN_EPISODES_PER_JOB,
    describe_jobs,
    iter_chunks,
    map_episodes,
    resolve_jobs,
)


def _setup(**kwargs):
    return {"scale": kwargs.get("scale", 1)}


def _task(episode: int, state: dict) -> int:
    return episode * state["scale"]


def _uneven_task(episode: int, state: dict) -> int:
    """让第二段先完成，同时报告带帧数的工作状态。"""
    state["_progress"]("逐帧求解", current=1, total=2)
    time.sleep(0.8 if episode == 0 else 0.05)
    state["_progress"]("碰撞评分", current=2, total=2)
    return episode * state["scale"]


def _environment_task(episode: int, state: dict) -> tuple[str | None, ...]:
    """返回子进程在导入主模块期间继承的线程配置。"""
    return tuple(os.environ.get(name) for name in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS"))


def _flood_task(episode: int, state: dict) -> int:
    """密集发送事件以填满队列，最终仍可靠返回计算结果。"""
    for index in range(10000):
        state["_progress"]("阶段甲" if index % 2 else "阶段乙", current=index, total=10000)
    return episode


def test_sequential_path_runs_in_the_calling_process() -> None:
    """`jobs=1` is the loop the command has always run, not a pool of one.

    Keeping it in-process is what makes the sequential result impossible to
    drift from by accident.
    """
    seen: list[int] = []
    results = map_episodes(
        [3, 1, 2],
        setup=_setup,
        setup_kwargs={"scale": 10},
        task=_task,
        jobs=1,
        on_result=lambda episode, _value: seen.append(episode),
    )
    assert results == [30, 10, 20]
    # Order follows the episodes as given, not their numeric order.
    assert seen == [3, 1, 2]


def test_jobs_default_leaves_room_for_the_solver_threads() -> None:
    cores = os.cpu_count() or 1
    resolved = resolve_jobs(None, 1000)
    assert resolved == max(1, cores // 3)


def test_jobs_never_exceed_the_work_or_the_machine() -> None:
    cores = os.cpu_count() or 1
    assert resolve_jobs(1000, 1000) == cores
    assert resolve_jobs(8, 40) <= 8


def test_small_datasets_stay_sequential() -> None:
    """Each worker pays a JIT compilation before its first episode."""
    assert resolve_jobs(8, 1) == 1
    assert resolve_jobs(8, MIN_EPISODES_PER_JOB - 1) == 1
    assert resolve_jobs(8, MIN_EPISODES_PER_JOB * 2) == 2


def test_no_episodes_needs_no_workers() -> None:
    assert resolve_jobs(8, 0) == 1


def test_jobs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        resolve_jobs(0, 10)


def test_describe_jobs_says_what_will_happen() -> None:
    assert describe_jobs(1, 12) == "12 episodes, one at a time"
    assert describe_jobs(4, 12) == "12 episodes across 4 workers"


def test_chunks_cover_every_episode_in_order() -> None:
    assert iter_chunks(range(7), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert iter_chunks([], 3) == []
    with pytest.raises(ValueError, match="at least 1"):
        iter_chunks([1], 0)


def test_spawn_reports_completed_tasks_without_waiting_for_input_order() -> None:
    """慢任务在前时，结果回调立即收到快任务，返回值仍按输入排序。"""
    seen = []
    events = []
    values = map_episodes(
        [0, 1], setup=_setup, setup_kwargs={"scale": 10}, task=_uneven_task,
        jobs=2, on_result=lambda episode, _: seen.append(episode),
        on_progress=events.append,
    )
    assert values == [0, 10]
    assert seen == [1, 0]
    assert any(event.get("stage") == "逐帧求解" and event.get("total") == 2 for event in events)


def test_spawn_limits_threads_and_restores_parent_environment(monkeypatch) -> None:
    """数值库线程设置先于 spawn 生效，父进程原值在完成后恢复。"""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    values = map_episodes(
        [0, 1], setup=_setup, setup_kwargs={}, task=_environment_task, jobs=2,
    )
    assert all(value[:3] == ("1", "1", "1") for value in values)
    assert all("--xla_cpu_multi_thread_eigen=false" in value[3] for value in values)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert "XLA_FLAGS" not in os.environ


def test_interrupt_terminates_workers_and_restores_environment(monkeypatch) -> None:
    """回调中断时立即回收工作进程，并保留原始异常。"""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "9")
    before = {process.pid for process in multiprocessing.active_children()}

    def interrupt(_event):
        """模拟收到首条进度时发生 Ctrl+C。"""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        map_episodes([0, 1], setup=_setup, setup_kwargs={}, task=_uneven_task,
                     jobs=2, on_progress=interrupt)
    assert os.environ["OPENBLAS_NUM_THREADS"] == "9"
    assert {process.pid for process in multiprocessing.active_children()} <= before


def test_full_progress_queue_does_not_block_workers() -> None:
    """队列饱和时允许丢弃中间事件，但不能阻塞或丢失最终结果。"""
    values = map_episodes(
        [0, 1], setup=_setup, setup_kwargs={}, task=_flood_task, jobs=2,
        on_progress=lambda _: time.sleep(0.002),
    )
    assert values == [0, 1]
