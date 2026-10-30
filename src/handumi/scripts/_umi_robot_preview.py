"""Vive 录制的后台机器人 IK 预览：最新帧求解、代次隔离和 15 Hz 显示。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from handumi.robots.umi_ik import RelativeToolIk, ToolIkFrame

log = logging.getLogger("handumi.scripts.umi_robot_preview")

class PreviewView(Protocol):
    """机器人预览显示接口，测试可注入假 Rerun 视图。"""

    def start(self) -> None:
        """启动查看器。"""
        ...

    def log_frame(
        self,
        *,
        sample_time_ns: int,
        q: np.ndarray,
        target_tcp_pose7: np.ndarray,
        actual_tcp_pose7: np.ndarray,
        valid: bool,
        position_error_m: float,
        orientation_error_deg: float,
        failure_reason: str = "",
    ) -> None:
        """记录一帧求解结果。"""
        ...

    def close(self) -> None:
        """关闭查看器。"""
        ...


@dataclass(frozen=True)
class _Sample:
    """等待后台求解的一份源 TCP 快照。"""

    generation: int
    capture_time_ns: int
    source_tcp_pose7: np.ndarray | None


@dataclass(frozen=True)
class _Anchor:
    """新片段首帧锚点；必须优先于容量一队列中的最新帧。"""

    generation: int
    capture_time_ns: int
    source_tcp_pose7: np.ndarray | None


class RobotIkPreview:
    """后台求解并显示最新机器人 IK 结果，不阻塞 30 Hz 采集。

    求解线程是唯一修改 ``RelativeToolIk`` 状态的地方。``begin_episode()``
    只递增代次并放入独立的首帧锚点；线程在代次变化后先重置控制器，再从
    锚点建立新起点。容量一队列只会丢弃旧预览帧，不会覆盖锚点。旧代次
    结果在求解前后各检查一次，绝不提交给显示层。
    """

    def __init__(
        self,
        controller: RelativeToolIk,
        view: PreviewView,
        *,
        display_hz: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        """创建最新帧队列和线程状态；不启动线程或查看器。"""
        if not np.isfinite(display_hz) or display_hz <= 0:
            raise ValueError("display_hz 必须是正的有限值")
        self._controller = controller
        self._view = view
        self._display_period_s = 1.0 / float(display_hz)
        self._clock = clock
        self._logger = logger or log
        self._samples: queue.Queue[_Sample | object] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._view_lock = threading.Lock()
        self._stop = threading.Event()
        self._work = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._failed = False
        self._view_failed = False
        self._generation = 0
        self._active = False
        self._pending_anchor: _Anchor | None = None
        self._controller_generation: int | None = None
        self._next_display_s = 0.0
        self._dropped_samples = 0
        self._last_result: ToolIkFrame | None = None

    @property
    def started(self) -> bool:
        """后台线程是否已启动。"""
        return self._started

    @property
    def healthy(self) -> bool:
        """后台求解和显示是否仍可用。"""
        return (
            self._started
            and not self._closed
            and not self._failed
            and not self._view_failed
            and self._thread is not None
            and self._thread.is_alive()
        )

    @property
    def generation(self) -> int:
        """当前片段/等待预览代次。"""
        with self._lock:
            return self._generation

    @property
    def dropped_samples(self) -> int:
        """容量一队列丢弃的旧预览帧数量。"""
        return self._dropped_samples

    @property
    def last_result(self) -> ToolIkFrame | None:
        """最近一次有效提交给显示层的求解结果。"""
        return self._last_result

    def start(self) -> None:
        """预热 controller、启动显示层和后台线程。

        预热发生在查看器启动和录制循环之前；模型/网格的加载由调用方在
        目录处理前完成。启动失败会可靠关闭已启动的显示层。
        """
        if self._started:
            return
        if self._closed:
            raise RuntimeError("机器人预览已关闭")
        try:
            self._controller.warmup()
            self._view.start()
        except BaseException:
            try:
                self._view.close()
            except Exception:
                self._logger.debug("关闭预览显示层失败", exc_info=True)
            raise
        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-robot-preview",
            daemon=True,
        )
        self._started = True
        self._active = True
        self._thread.start()

    def begin_waiting(self) -> None:
        """进入等待录制的预览模式；首个样本会建立临时起点。"""
        with self._lock:
            self._generation += 1
            self._pending_anchor = None
            self._active = True
            _drain(self._samples)
            self._work.set()

    def begin_episode(
        self,
        anchor_source_tcp_pose7: np.ndarray | None,
        capture_time_ns: int,
    ) -> None:
        """开始新片段并提交录制首帧锚点。

        锚点存放在独立槽位，优先于容量一的最新帧队列处理，因此即使预览
        跳帧，录制首帧仍会建立正确的相对映射起点。
        """
        anchor = None
        if anchor_source_tcp_pose7 is not None:
            anchor = np.asarray(anchor_source_tcp_pose7, dtype=np.float32).copy()
            if anchor.shape != (7,):
                raise ValueError("anchor_source_tcp_pose7 必须是长度 7 的 pose7")
        with self._lock:
            self._generation += 1
            self._pending_anchor = _Anchor(
                generation=self._generation,
                capture_time_ns=int(capture_time_ns),
                source_tcp_pose7=anchor,
            )
            self._active = True
            _drain(self._samples)
            self._work.set()

    def end_episode(self) -> None:
        """结束当前片段，回到等待录制的独立代次。"""
        self.begin_waiting()

    def cancel_episode(self) -> None:
        """丢弃当前片段，旧代次的在途结果不会再发布。"""
        with self._lock:
            self._generation += 1
            self._pending_anchor = None
            self._active = False
            _drain(self._samples)
            self._work.set()

    def submit(
        self,
        capture_time_ns: int,
        source_tcp_pose7: np.ndarray | None,
        *,
        tracked: bool = True,
    ) -> None:
        """非阻塞提交最新源 TCP；队列满时替换旧预览帧。"""
        if self._closed or self._failed or self._view_failed:
            return
        with self._lock:
            if not self._active:
                return
            generation = self._generation
        pose = None
        if tracked and source_tcp_pose7 is not None:
            pose = np.asarray(source_tcp_pose7, dtype=np.float32).copy()
            if pose.shape != (7,):
                raise ValueError("source_tcp_pose7 必须是长度 7 的 pose7")
        sample = _Sample(
            generation=generation,
            capture_time_ns=int(capture_time_ns),
            source_tcp_pose7=pose,
        )
        try:
            self._samples.put_nowait(sample)
            self._work.set()
            return
        except queue.Full:
            pass
        try:
            self._samples.get_nowait()
            self._dropped_samples += 1
        except queue.Empty:
            pass
        try:
            self._samples.put_nowait(sample)
            self._work.set()
        except queue.Full:
            self._dropped_samples += 1

    def pause(self, *, timeout_s: float = 2.0) -> bool:
        """暂停求解并等待在途帧结束，供离线补算/保存阶段使用。"""
        self._running.clear()
        self._work.set()
        if not self._started:
            return True
        return self._idle.wait(timeout=max(0.0, timeout_s))

    def resume(self) -> None:
        """恢复后台求解。"""
        if self._closed:
            return
        self._running.set()
        self._work.set()

    def wait_idle(self, *, timeout_s: float = 2.0) -> bool:
        """等待求解线程当前帧完成，不改变暂停状态。"""
        if not self._started:
            return True
        return self._idle.wait(timeout=max(0.0, timeout_s))

    def close(self, *, timeout_s: float = 2.0) -> None:
        """停止后台线程并关闭预览显示；重复调用幂等。

        ``timeout_s`` 是有限等待：超时后线程可能仍在慢 solver 内，但代次
        已被推进，后续结果不会提交；``_view_lock`` 保证 ``view.close()``
        之后不再出现迟到的 ``log_frame``。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._active = False
            self._pending_anchor = None
            _drain(self._samples)
        self._stop.set()
        self._running.set()
        self._work.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout_s))
            if self._thread.is_alive():
                self._logger.warning("机器人预览线程未在 %.1f 秒内退出", timeout_s)
        with self._view_lock:
            try:
                self._view.close()
            except Exception:
                self._logger.debug("关闭机器人预览显示层失败", exc_info=True)

    def _run(self) -> None:
        """从独立锚点/容量一队列取最新工作，优先处理同代次锚点。"""
        try:
            while not self._stop.is_set():
                if not self._running.is_set():
                    self._idle.set()
                    self._work.wait(timeout=0.05)
                    self._work.clear()
                    continue
                self._work.wait(timeout=0.05)
                self._work.clear()
                work = self._take_next_work()
                if work is None:
                    self._idle.set()
                    continue
                self._idle.clear()
                if isinstance(work, _Anchor):
                    self._process(work.generation, work.capture_time_ns, work.source_tcp_pose7)
                else:
                    if self._is_current(work.generation):
                        self._process(
                            work.generation,
                            work.capture_time_ns,
                            work.source_tcp_pose7,
                        )
                self._idle.set()
        except Exception:
            self._failed = True
            self._logger.exception("机器人预览线程异常；原始录制继续")
        finally:
            self._idle.set()

    def _take_next_work(self) -> _Anchor | _Sample | None:
        """原子地优先取同代次锚点，否则取队列中的最新样本。"""
        with self._lock:
            anchor = self._pending_anchor
            if anchor is not None:
                self._pending_anchor = None
                if anchor.generation == self._generation:
                    return anchor
            try:
                return self._samples.get_nowait()
            except queue.Empty:
                return None

    def _is_current(self, generation: int) -> bool:
        """判断结果代次是否仍为当前代次且预览未关闭。"""
        with self._lock:
            return not self._closed and generation == self._generation

    def _process(
        self,
        generation: int,
        capture_time_ns: int,
        source_tcp_pose7: np.ndarray | None,
    ) -> None:
        """在求解线程内更新 controller，并在代次仍有效时提交显示。"""
        if self._closed:
            return
        if self._controller_generation != generation:
            self._controller.reset()
            self._controller_generation = generation
        if not self._is_current(generation):
            return
        frame = self._controller.process(source_tcp_pose7)
        # 求解期间可能已经开始新片段或关闭预览；旧结果绝不发布。
        if not self._is_current(generation):
            return
        now = self._clock()
        if now < self._next_display_s:
            return
        self._next_display_s = now + self._display_period_s
        with self._view_lock:
            if self._closed or not self._is_current(generation):
                return
            try:
                self._view.log_frame(
                    sample_time_ns=capture_time_ns,
                    q=frame.q,
                    target_tcp_pose7=frame.target_tcp_pose7,
                    actual_tcp_pose7=frame.actual_tcp_pose7,
                    valid=frame.valid,
                    position_error_m=frame.position_error_m,
                    orientation_error_deg=frame.orientation_error_deg,
                    failure_reason=frame.failure_reason,
                )
            except Exception:
                self._view_failed = True
                self._logger.exception("机器人 Rerun 显示失败；原始录制继续")
                return
            self._last_result = frame


def _drain(samples: queue.Queue[Any]) -> None:
    """清空容量一队列中的旧预览帧。"""
    while True:
        try:
            samples.get_nowait()
        except queue.Empty:
            return


__all__ = ["PreviewView", "RobotIkPreview"]
