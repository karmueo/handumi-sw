"""调度独立的离线片段任务，并在主进程汇总结果与进度。

单进程模式在调用进程中顺序执行；多进程模式按实际完成顺序通知结果，
返回值保持输入顺序。工作进程使用 spawn，避免 fork 已初始化的 JAX 状态。
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# Below this, the per-worker JIT compilation costs more than the parallelism
# returns. Measured on this repo's solvers, a worker spends a couple of seconds
# compiling before its first episode; a handful of episodes never earns that
# back.
MIN_EPISODES_PER_JOB = 4

THREADS_PER_WORKER: int | None = 1

_STATE: dict[str, Any] = {}
_EVENTS = None


def resolve_jobs(
    requested: int | None,
    episode_count: int,
    *,
    min_episodes_per_job: int = MIN_EPISODES_PER_JOB,
) -> int:
    """确定工作进程数，并按数据量和可用 CPU 数限制。

    requested 为 None 时默认每三个 CPU 分配一个进程；
    min_episodes_per_job 控制每个进程所需的最少段数，设为 1 可关闭小批次降档。
    无待处理段时返回 1，供调用方选择不启动进程池。
    """
    if min_episodes_per_job < 1:
        raise ValueError("min_episodes_per_job must be at least 1")
    if requested is not None and requested < 1:
        raise ValueError(f"--jobs must be at least 1, got {requested}")
    if episode_count <= 0:
        return 1
    available = os.cpu_count() or 1
    if requested is None:
        requested = max(1, available // 3)
    capped = min(requested, episode_count, available)
    if capped > 1 and episode_count < capped * min_episodes_per_job:
        capped = max(1, episode_count // min_episodes_per_job)
    return max(1, capped)


def _worker_init(
    setup: Callable[..., dict[str, Any]],
    kwargs: dict[str, Any],
    threads: int,
    events=None,
) -> None:
    """在独立进程中构造可复用状态，并避免退出时等待进度队列。"""
    global _EVENTS
    _EVENTS = events
    if events is not None:
        events.cancel_join_thread()
    _STATE.clear()
    _STATE.update(setup(**kwargs))


def _worker_run(payload: tuple[Callable[..., Any], int, int]) -> tuple[int, Any]:
    """执行一段并限频发送尽力而为的状态事件。"""
    task, position, episode = payload
    last_stage = None
    last_sent = 0.0

    def progress(stage: str, **details) -> None:
        """队列满时丢弃中间状态，绝不阻塞计算。"""
        nonlocal last_stage, last_sent
        if _EVENTS is None:
            return
        now = time.monotonic()
        if stage == last_stage and now - last_sent < 0.2:
            return
        last_stage, last_sent = stage, now
        try:
            _EVENTS.put_nowait({"episode": episode, "stage": stage, **details})
        except (queue.Full, OSError, ValueError):
            pass

    if _EVENTS is not None:
        _STATE["_progress"] = progress
    try:
        progress("模型初始化完成")
        return position, task(episode, _STATE)
    finally:
        _STATE.pop("_progress", None)


@contextlib.contextmanager
def _spawn_environment(threads: int):
    """在 spawn 导入主模块之前设置数值库线程限制，退出后恢复环境。"""
    count = str(max(1, threads))
    values = {
        "OMP_NUM_THREADS": count,
        "OPENBLAS_NUM_THREADS": count,
        "MKL_NUM_THREADS": count,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu"),
    }
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen" not in flags:
        flags += " --xla_cpu_multi_thread_eigen=false"
    if "intra_op_parallelism_threads=" not in flags:
        flags += f" intra_op_parallelism_threads={count}"
    values["XLA_FLAGS"] = flags.strip()
    before = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _terminate_pool(pool: ProcessPoolExecutor) -> None:
    """异常时终止运行中的进程，避免等待长时间求解任务。"""
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            pass
    for process in processes:
        try:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        except (OSError, ValueError):
            pass
    try:
        pool.shutdown(wait=True, cancel_futures=True)
    except (OSError, ValueError):
        pass


def map_episodes(
    episodes: Sequence[int],
    *,
    setup: Callable[..., dict[str, Any]],
    setup_kwargs: dict[str, Any],
    task: Callable[[int, dict[str, Any]], Any],
    jobs: int = 1,
    on_result: Callable[[int, Any], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_tick: Callable[[], None] | None = None,
) -> list[Any]:
    """按实际完成顺序通知结果，返回值保持输入顺序。

    setup 每个进程执行一次。提供 on_progress 时，task 可通过
    state['_progress'](阶段, **细节) 发送可丢弃的进度事件；状态回调
    与结果回调都只在调用进程执行。
    """
    ordered = list(episodes)
    if not ordered:
        return []
    if jobs <= 1:
        state = setup(**setup_kwargs)
        results: list[Any] = []
        for episode in ordered:
            def progress(stage: str, **details) -> None:
                """单进程中直接通知调用者，并保持定时刷新。"""
                if on_progress is not None:
                    on_progress({"episode": episode, "stage": stage, **details})
                if on_tick is not None:
                    on_tick()

            if on_progress is not None:
                state["_progress"] = progress
            try:
                progress("开始处理")
                value = task(episode, state)
            finally:
                state.pop("_progress", None)
            if on_result is not None:
                on_result(episode, value)
            results.append(value)
            if on_tick is not None:
                on_tick()
        return results

    threads = 1 if THREADS_PER_WORKER is None else THREADS_PER_WORKER
    context = mp.get_context("spawn")
    events = context.Queue(maxsize=max(16, jobs * 8)) if on_progress else None
    completed = queue.SimpleQueue()
    collected: list[Any] = [None] * len(ordered)
    pending = {}
    next_position = 0

    def drain() -> None:
        """消费当前可用事件；丢失的状态不会影响任务结果。"""
        if events is not None and on_progress is not None:
            while True:
                try:
                    on_progress(events.get_nowait())
                except queue.Empty:
                    break
        if on_tick is not None:
            on_tick()

    try:
        with _spawn_environment(threads):
            pool = ProcessPoolExecutor(
                max_workers=jobs, mp_context=context, initializer=_worker_init,
                initargs=(setup, setup_kwargs, threads, events),
            )
            try:
                while next_position < len(ordered) or pending:
                    while next_position < len(ordered) and len(pending) < jobs:
                        episode = ordered[next_position]
                        future = pool.submit(_worker_run, (task, next_position, episode))
                        pending[future] = (next_position, episode)
                        future.add_done_callback(completed.put)
                        if on_progress is not None:
                            on_progress({"episode": episode, "stage": "等待工作进程"})
                        next_position += 1
                    drain()
                    try:
                        future = completed.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    _, episode = pending.pop(future)
                    position, value = future.result()
                    collected[position] = value
                    drain()
                    if on_result is not None:
                        on_result(episode, value)
                drain()
            except BaseException:
                _terminate_pool(pool)
                raise
            else:
                pool.shutdown(wait=True)
    finally:
        if events is not None:
            try:
                events.close()
                events.cancel_join_thread()
            except (OSError, ValueError):
                pass
    return collected


def describe_jobs(jobs: int, episodes: int) -> str:
    if jobs <= 1:
        return f"{episodes} episodes, one at a time"
    return f"{episodes} episodes across {jobs} workers"


def iter_chunks(values: Iterable[int], size: int) -> list[list[int]]:
    """Split episodes into contiguous blocks, for callers that batch."""
    if size < 1:
        raise ValueError(f"chunk size must be at least 1, got {size}")
    chunk: list[int] = []
    chunks: list[list[int]] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks
