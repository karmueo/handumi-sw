"""使用有限 spawn 进程准备 episode，交互请求优先且不阻塞播放线程。"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing
import os
import queue
import threading
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_RUNTIME_CACHE: dict[tuple[str, str, str, str], Any] = {}
_PROGRESS_EVENTS = None


@contextlib.contextmanager
def _spawn_environment():
    """在创建子进程前传递线程限制，覆盖 spawn 的 __main__ 提前导入。"""
    values = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu"),
    }
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen" not in flags:
        flags += " --xla_cpu_multi_thread_eigen=false"
    if "intra_op_parallelism_threads=" not in flags:
        flags += " intra_op_parallelism_threads=1"
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


def _worker_init(events=None) -> None:
    """限制每个 spawn worker 的数值库线程及 CPU 亲和性。"""
    global _PROGRESS_EVENTS
    _PROGRESS_EVENTS = events
    if events is not None:
        # 进度可丢失，worker 退出时不得等待无人消费的管道。
        events.cancel_join_thread()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))
            if cpus:
                start = os.getpid() % len(cpus)
                os.sched_setaffinity(
                    0, {cpus[(start + i) % len(cpus)] for i in range(min(2, len(cpus)))}
                )
        except OSError:
            pass


def _runtime_for(args, identity):
    """只在机器人配置或 URDF 摘要相同时复用已加载的求解模型。"""
    from handumi.robots.registry import load_embodiment

    key = (
        args.robot,
        str(identity["robot_config_sha256"]),
        str(identity["urdf_sha256"]),
        str(args.robot == "rm75b" and args.retarget_mode == "absolute-tcp"),
    )
    if key not in _RUNTIME_CACHE:
        _RUNTIME_CACHE.clear()
        if args.robot == "rm75b" and args.retarget_mode == "absolute-tcp":
            _RUNTIME_CACHE[key] = load_embodiment(args.robot, collision_kind="gripper-box")
        else:
            _RUNTIME_CACHE[key] = load_embodiment(args.robot)
    return _RUNTIME_CACHE[key]


class _WarningCapture(io.TextIOBase):
    """只保留有限条需要转发的警告，避免长任务积累全部日志。"""

    def __init__(self):
        """初始化限长行缓冲和警告列表。"""
        self.pending = ""
        self.warnings = []

    def write(self, text):
        """提取严格误差警告并限制保存量。"""
        lines = (self.pending + text).split("\n")
        self.pending = lines.pop()[:65536]
        for line in lines:
            if line.startswith(("[replay] warning: IK fidelity threshold exceeded",
                                "[replay] cache write skipped:",
                                "[replay] cache revalidation skipped:",
                                "[replay] source or calibration changed during solve")):
                if len(self.warnings) < 10:
                    self.warnings.append(line[:65536])
        return len(text)


class _BoundedStderr:
    """读取 worker 的原生和 Python stderr，限制转发日志所占内存。"""

    def __init__(self, limit: int = 16384):
        """设置单任务 stderr 最大保留字节数。"""
        self.limit = limit
        self.data = bytearray()
        self.dropped = 0

    def __enter__(self):
        """用管道接收描述符 2，后台线程持续排空管道。"""
        read_fd, write_fd = os.pipe()
        self.saved = os.dup(2)
        os.dup2(write_fd, 2)
        os.close(write_fd)

        def collect():
            """持续消费写入数据，并丢弃超出限额的部分。"""
            with os.fdopen(read_fd, "rb", buffering=0) as stream:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        return
                    remaining = max(0, self.limit - len(self.data))
                    self.data.extend(chunk[:remaining])
                    self.dropped += max(0, len(chunk) - remaining)

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        """恢复 stderr 后等待管道排空；异常也会恢复原描述符。"""
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.thread.join(timeout=1)

    def text(self):
        """返回有限日志，并提示丢弃的字节数。"""
        result = self.data.decode("utf-8", errors="replace").rstrip()
        if self.dropped:
            result += f"\n[replay] worker stderr truncated ({self.dropped} bytes omitted)"
        return result


class _BatchQueue:
    """按编号惰性生成批量任务，优先选中段且不存储全部编号。"""

    def __init__(self, count: int, current: int):
        """记录选中段和后续编号游标。"""
        self.count = count
        self.current = current
        self.first = True
        self.next = 0
        self.remaining = count

    def __bool__(self):
        """返回是否仍有任务未提交。"""
        return self.remaining > 0

    def popleft(self):
        """先取选中段，再按序取其余段。"""
        if not self.remaining:
            raise IndexError("empty batch queue")
        self.remaining -= 1
        if self.first:
            self.first = False
            return self.current
        if self.next == self.current:
            self.next += 1
        result = self.next
        self.next += 1
        return result

    def clear(self):
        """停止生成后续段。"""
        self.remaining = 0


def _worker_prepare(
    values: dict[str, Any], episode: int, refresh: bool
) -> dict[str, Any]:
    """在 worker 中准备一段；返回数组或可显示错误，不访问 Viser。"""
    from handumi.scripts.replay.replay_prepare import prepare_episode
    from handumi.scripts.replay.replay_progress import WorkerProgress

    args = argparse.Namespace(**values)
    args.episode = episode
    args.refresh_cache = refresh
    captured = _WarningCapture()
    progress = WorkerProgress(episode, _PROGRESS_EVENTS)
    args._replay_progress = progress
    stderr_capture = _BoundedStderr() if values.get("_batch_compact") else None
    stderr_context = stderr_capture if stderr_capture is not None else contextlib.nullcontext()
    try:
        with contextlib.redirect_stdout(captured), stderr_context:
            rollout = prepare_episode(args, runtime_provider=_runtime_for)
        response = {"warnings": captured.warnings, **progress.finish()}
        if stderr_capture is not None:
            response["logs"] = stderr_capture.text()
        if not values.get("_batch_compact") or episode == values.get("_batch_current"):
            response["rollout"] = rollout
        return response
    except (Exception, SystemExit) as exc:
        return {"error": str(exc), "logs": stderr_capture.text() if stderr_capture else "",
                **progress.finish()}


class EpisodeScheduler:
    """维护有限在途任务、选择优先队列及少量内存中的准备结果。"""

    def __init__(
        self,
        args,
        episode_count: int,
        *,
        current: int,
        workers: int = 8,
        background: bool = True,
        executor=None,
        batch: bool = False,
        progress: bool = False,
    ):
        """创建 spawn 池；后台队列只保存编号，不预先提交全部任务。"""
        if workers < 1:
            raise ValueError("prepare workers 必须至少为 1")
        if episode_count <= 0:
            raise ValueError("数据集没有 episode。")
        if not 0 <= current < episode_count:
            raise ValueError(f"episode {current} 超出范围 0..{episode_count - 1}。")
        self.values = {
            key: value for key, value in vars(args).items() if not key.startswith("_")
        }
        self.workers = workers
        self.batch = batch
        if batch:
            self.values.update(_batch_compact=True, _batch_current=current)
        context = multiprocessing.get_context("spawn")
        self.events = context.Queue(maxsize=max(16, workers * 8)) if progress else None
        self.executor = executor or ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(self.events,),
        )
        self.queue = (_BatchQueue(episode_count, current) if batch and background else deque(
            ep for ep in range(episode_count) if background and ep != current
        ))
        self.running: dict[Any, int] = {}
        self.completed = set() if batch else {current}
        self.failures: dict[int, str] = {}
        self.cache_hits = 0
        self.stage_seconds: dict[str, float] = {}
        self.warning_sink = print
        self.warnings: dict[int, list[str]] = {}
        self.results: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.result_capacity = max(2, workers + 1)
        self.closed = False

    def request(self, episode: int) -> None:
        """将用户选择移到待处理队首；已有在途任务不会重复提交。"""
        if self.closed or episode in self.running.values():
            return
        if episode in self.results:
            self.results.move_to_end(episode)
            return
        self.failures.pop(episode, None)
        try:
            self.queue.remove(episode)
        except ValueError:
            pass
        self.queue.appendleft(episode)

    def poll(self) -> None:
        """回收完成任务，并仅为可用 worker 提交优先队首。"""
        if self.closed:
            return
        for future, episode in list(self.running.items()):
            if not future.done():
                continue
            del self.running[future]
            try:
                response = future.result()
            except Exception as exc:
                response = {"error": f"worker failed: {exc}"}
            self.cache_hits += int(response.get("cache_hit", False))
            for stage, seconds in response.get("stage_seconds", {}).items():
                self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + seconds
            if response.get("logs"):
                self.warning_sink(f"[replay] episode {episode} stderr:\n{response['logs']}")
            if "error" in response:
                self.failures[episode] = str(response["error"])
            else:
                if "rollout" in response:
                    self.results[episode] = response["rollout"]
                warnings = response.get("warnings", [])
                if warnings:
                    if self.values.get("prepare_mode") == "all":
                        for warning in warnings:
                            self.warning_sink(f"[replay] episode {episode}: {warning}")
                    else:
                        self.warnings[episode] = warnings
                if episode in self.results:
                    self.results.move_to_end(episode)
                while len(self.results) > self.result_capacity:
                    discarded, _ = self.results.popitem(last=False)
                    # 对应结果已不在内存；再次选择时会从磁盘重建 warning。
                    self.warnings.pop(discarded, None)
            self.completed.add(episode)
        while self.queue and len(self.running) < self.workers:
            episode = self.queue.popleft()
            # 刷新模式每段只强制一次；已刷新且被 LRU 淘汰时可重读磁盘。
            refresh = bool(
                self.values.get("refresh_cache", False)
                and episode not in self.completed
            )
            try:
                with _spawn_environment():
                    future = self.executor.submit(
                        _worker_prepare, self.values, episode, refresh
                    )
                self.running[future] = episode
            except Exception as exc:
                self.failures[episode] = f"worker unavailable: {exc}"
                self.completed.add(episode)

    def drain_events(self) -> dict[int, dict]:
        """合并有限事件队列，只返回每个 episode 的最新状态。"""
        latest = {}
        if self.events is not None:
            while True:
                try:
                    event = self.events.get_nowait()
                except (queue.Empty, OSError, ValueError):
                    break
                latest[event["episode"]] = event
        return latest

    def take(self, episode: int):
        """取走所选结果；尚未完成返回 None，失败抛出 RuntimeError。"""
        if episode in self.failures:
            raise RuntimeError(self.failures[episode])
        result = self.results.pop(episode, None)
        if result is not None:
            for warning in self.warnings.pop(episode, []):
                print(f"[replay] episode {episode}: {warning}")
        return result

    def close(self) -> None:
        """停止提交、清空待处理队列，不等待已经运行的任务。"""
        if self.closed:
            return
        self.closed = True
        self.queue.clear()
        if not self.running:
            # 全部工作已结束时让空闲 worker 正常退出，回收进程间信号量。
            self.executor.shutdown(wait=True, cancel_futures=True)
        else:
            processes = list((getattr(self.executor, "_processes", None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            self.executor.shutdown(wait=False, cancel_futures=True)
        if self.events is not None:
            self.events.close()
            self.events.cancel_join_thread()
