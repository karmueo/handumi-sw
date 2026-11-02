"""在独立进程运行 VIVE 流式编码，隔离编码初始化对采集线程的影响。"""

from __future__ import annotations

import multiprocessing as mp
import queue
import shutil
import signal
import tempfile
import time
from pathlib import Path

import numpy as np


def _encoder_worker(commands, replies, fps, rgb_encoder, depth_encoder, queue_size, threads):
    """顺序处理编码命令，传播异步帧错误并在退出时清理编码器。"""
    # 终端 Ctrl+C 由主进程协调；保存中的编码进程须继续完成提交。
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from lerobot.datasets.video_utils import StreamingVideoEncoder

    encoder = None
    failure = None
    try:
        encoder = StreamingVideoEncoder(
            fps=fps, rgb_encoder=rgb_encoder, depth_encoder=depth_encoder,
            queue_maxsize=queue_size, encoder_threads=threads,
        )
        replies.put(("ready", 0, None))
        while True:
            operation, request_id, payload = commands.get()
            try:
                if operation == "start":
                    failure = None
                    encoder.start_episode(**payload)
                    replies.put(("ok", request_id, None))
                elif operation == "frame":
                    if failure is None:
                        key, image = payload
                        previous = encoder._dropped_frames.get(key, 0)
                        encoder.feed_frame(key, image)
                        if encoder._dropped_frames.get(key, 0) != previous:
                            raise RuntimeError(f"encoder queue dropped a frame for {key}")
                elif operation == "finish":
                    if failure is not None:
                        raise RuntimeError(failure)
                    result = encoder.finish_episode()
                    replies.put(("ok", request_id, result))
                elif operation == "cancel":
                    encoder.cancel_episode()
                    failure = None
                    replies.put(("ok", request_id, None))
                elif operation == "close":
                    encoder.close()
                    replies.put(("ok", request_id, None))
                    return
            except BaseException as exc:
                message = f"{type(exc).__name__}: {exc}"
                if operation == "frame":
                    failure = message
                    replies.put(("async_error", request_id, message))
                else:
                    replies.put(("error", request_id, message))
    except BaseException as exc:
        replies.put(("fatal", 0, f"{type(exc).__name__}: {exc}"))
    finally:
        if encoder is not None:
            encoder.close()


class ViveProcessEncoder:
    """以有界进程队列包装 LeRobot 编码器，并保证故障不可静默丢帧。"""

    def __init__(self, *, fps, rgb_encoder, depth_encoder=None, queue_size=30,
                 encoder_threads=None, control_timeout=180, frame_timeout=0.1):
        """启动编码子进程并等待就绪；控制与帧传输有明确超时。"""
        context = mp.get_context("spawn")
        # 首次打开编码器可能占用子进程数帧时间，留出有限的 IPC 缓冲。
        self._commands = context.Queue(maxsize=8)
        self._replies = context.Queue()
        self._process = context.Process(
            target=_encoder_worker,
            args=(self._commands, self._replies, fps, rgb_encoder, depth_encoder,
                  queue_size, encoder_threads),
            name="handumi-vive-encoder", daemon=True,
        )
        self._timeout = control_timeout
        self._frame_timeout = frame_timeout
        self._next_id = 1
        self._failure = None
        self._episode_dir = None
        self._closed = False
        self._dropped_frames = {}  # 丢帧由子进程转换为异步错误，供严格包装器兼容。
        try:
            self._process.start()
            self._wait(0, min(control_timeout, 30), ready=True)
        except BaseException:
            self._shutdown()
            raise

    def _alive(self):
        """确认子进程运行中，异常退出时报告退出码。"""
        if not self._process.is_alive():
            raise RuntimeError(f"VIVE 编码子进程已退出（退出码 {self._process.exitcode}）")

    def _poll(self, *, ignore_failure=False):
        """收取后台编码错误；清理操作可忽略已知错误以完成释放。"""
        while True:
            try:
                status, _, value = self._replies.get_nowait()
            except queue.Empty:
                break
            if status in {"async_error", "fatal", "error"}:
                self._failure = str(value)
        if self._failure is not None and not ignore_failure:
            raise RuntimeError(f"VIVE 编码失败：{self._failure}")
        self._alive()

    def _wait(self, request_id, timeout, *, ready=False, ignore_failure=False):
        """等待指定控制响应并检查进程死亡与其他异步错误。"""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("等待 VIVE 编码子进程响应超时")
            try:
                status, received_id, value = self._replies.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                self._alive()
                continue
            if status in {"async_error", "fatal"}:
                self._failure = str(value)
                if not ignore_failure:
                    raise RuntimeError(f"VIVE 编码失败：{value}")
                continue
            if received_id != request_id:
                continue
            if status == "error":
                self._failure = str(value)
                raise RuntimeError(f"VIVE 编码失败：{value}")
            if status == ("ready" if ready else "ok"):
                return value

    def _send(self, operation, payload=None, *, ignore_failure=False):
        """将控制命令顺序入队并等待确认。"""
        self._poll(ignore_failure=ignore_failure)
        request_id = self._next_id
        self._next_id += 1
        timeout = min(self._timeout, 5) if operation in {"cancel", "close"} else self._timeout
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"VIVE 编码命令队列超时：{operation}")
            try:
                self._commands.put(
                    (operation, request_id, payload), timeout=min(0.1, remaining),
                )
                break
            except queue.Full:
                self._poll(ignore_failure=ignore_failure)
        return self._wait(request_id, max(0, deadline - time.monotonic()),
                          ignore_failure=ignore_failure)

    def start_episode(self, video_keys, temp_dir, depth_video_keys=None):
        """为当前片段建立可清理目录并启动子进程编码。"""
        self._remove_episode_dir()
        self._episode_dir = Path(tempfile.mkdtemp(prefix="vive-encoding-", dir=temp_dir))
        try:
            self._send("start", dict(video_keys=video_keys, temp_dir=self._episode_dir,
                                     depth_video_keys=depth_video_keys))
        except BaseException:
            self._remove_episode_dir()
            raise

    def feed_frame(self, video_key, image):
        """复制后有界投递帧；队列超时或编码故障立即报错。"""
        self._poll()
        try:
            self._commands.put(("frame", 0, (video_key, np.array(image, copy=True))),
                               timeout=self._frame_timeout)
        except queue.Full as exc:
            self._alive()
            raise RuntimeError(f"VIVE 编码进程队列已满：{video_key}") from exc
        self._poll()

    def finish_episode(self):
        """等待所有帧编码完成并返回临时视频与统计信息。"""
        return self._send("finish")

    def cancel_episode(self):
        """取消当前编码并删除临时文件；必要时终止异常子进程。"""
        cancelled = False
        try:
            if self._process.is_alive():
                self._send("cancel", ignore_failure=True)
                self._failure = None
                cancelled = True
        finally:
            if not cancelled:
                self._shutdown()
            self._remove_episode_dir()

    def close(self):
        """关闭子进程和队列，清理未提交的临时目录。"""
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._send("close", ignore_failure=True)
        finally:
            self._shutdown()
            self._remove_episode_dir()

    def _remove_episode_dir(self):
        """删除当前片段的进程私有暂存目录。"""
        if self._episode_dir is not None:
            shutil.rmtree(self._episode_dir, ignore_errors=True)
            self._episode_dir = None

    def _shutdown(self):
        """按时关闭子进程，超时则终止并回收 IPC 资源。"""
        if self._closed:
            return
        self._closed = True
        if self._process.pid is not None:
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
        for channel in (self._commands, self._replies):
            channel.cancel_join_thread()
            channel.close()
