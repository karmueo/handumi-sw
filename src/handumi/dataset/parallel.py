"""Spread an offline per-episode solve across processes.

Episodes are independent: the IK warm-starts each frame from the previous one
*within* an episode and never across them, so solving several at once changes
nothing about the numbers. What it changes is wall-clock, and the offline
reviews are where that matters -- screening solves every episode of a dataset,
and conversion solves every episode it did not find in the screening cache.

Nothing here touches the live path. ``BimanualKinematicsSolver.ik`` is what
teleoperation runs frame by frame and is not involved; this module only decides
which process runs an episode's solve, never how that solve works.

Two properties are deliberate:

``--jobs 1`` runs the identical code in the calling process. It is not a pool of
one -- it is the loop the command has always run -- so the sequential path
cannot drift from the parallel one by accident.

Workers start with *spawn*, never fork. JAX installs threads and accelerator
state at import, and forking a process that has already imported it deadlocks
rather than failing, so a fresh interpreter per worker is the only safe option.
It costs one JIT compilation per worker, so callers may limit parallelism for
small datasets.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Below this, the per-worker JIT compilation costs more than the parallelism
# returns. Measured on this repo's solvers, a worker spends a couple of seconds
# compiling before its first episode; a handful of episodes never earns that
# back.
MIN_EPISODES_PER_JOB = 4

THREADS_PER_WORKER: int | None = 1

_STATE: dict[str, Any] = {}


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
) -> None:
    # Set before the worker imports JAX: each process gets a slice of the
    # machine, so N workers do not each try to use every core.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if threads > 0:
        os.environ["XLA_FLAGS"] = (
            f"{os.environ.get('XLA_FLAGS', '')} "
            f"--xla_cpu_multi_thread_eigen=false "
            f"intra_op_parallelism_threads={threads}"
        ).strip()
        os.environ["OMP_NUM_THREADS"] = str(threads)
    _STATE.clear()
    _STATE.update(setup(**kwargs))


def _worker_run(payload: tuple[Callable[..., Any], int]) -> tuple[int, Any]:
    task, episode = payload
    return episode, task(episode, _STATE)


def map_episodes(
    episodes: Sequence[int],
    *,
    setup: Callable[..., dict[str, Any]],
    setup_kwargs: dict[str, Any],
    task: Callable[[int, dict[str, Any]], Any],
    jobs: int = 1,
    on_result: Callable[[int, Any], None] | None = None,
) -> list[Any]:
    """Run ``task`` for every episode and return the results in episode order.

    ``setup`` builds whatever the task needs that is expensive and reusable --
    the robot model, the compiled collision functions -- once per process.
    """
    ordered = list(episodes)
    if jobs <= 1:
        state = setup(**setup_kwargs)
        results: list[Any] = []
        for episode in ordered:
            value = task(episode, state)
            if on_result is not None:
                on_result(episode, value)
            results.append(value)
        return results

    threads = 0 if THREADS_PER_WORKER is None else max(1, (os.cpu_count() or jobs) // jobs)
    context = mp.get_context("spawn")
    from concurrent.futures import ProcessPoolExecutor

    collected: dict[int, Any] = {}
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_worker_init,
        initargs=(setup, setup_kwargs, threads),
    ) as pool:
        # Results are reported as they land so a long run shows progress, and
        # reordered on the way out so the report never depends on timing.
        for episode, value in pool.map(
            _worker_run, [(task, episode) for episode in ordered]
        ):
            collected[episode] = value
            if on_result is not None:
                on_result(episode, value)
    return [collected[episode] for episode in ordered]


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
