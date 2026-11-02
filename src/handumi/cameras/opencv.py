"""使用 LeRobot OpenCV 后端采集带时间戳图像并处理 USB 重连。"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from sys import platform

import numpy as np

from handumi.cameras.base import CameraDevice, CameraSample

log = logging.getLogger(__name__)


class CameraFrameUnavailable(TimeoutError):
    """相机尚未产生任何可供采集的真实帧。"""


@dataclass
class OpenCVCameraDevice(CameraDevice):
    """管理一个 OpenCV 相机、时间戳帧缓存及后台断线恢复。"""

    index_or_path: int | str
    fps: int
    width: int
    height: int
    fourcc: str | None = None

    def __post_init__(self) -> None:
        """初始化同步资源和跨重连递增的帧序号。"""
        self._camera = None
        self._samples: deque[CameraSample] = deque(
            maxlen=max(8, min(16, int(self.fps or 30)))
        )
        self._samples_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._pending_reader = None  # LeRobot 释放后可能仍未退出的读线程。
        self._unfinished_backend = None  # 清理失败时保留对象，供关闭时重试。
        self._cleanup_error: Exception | None = None
        self._perf_to_monotonic_ns = 0
        self._sequence = 0
        self._state = "closed"  # 连接状态供采集界面显示。
        self._stall_timeout_s = 1.0
        self._retry_interval_s = 1.0
        self._unstable_path_warned = False  # 设备身份提示只输出一次。

    @property
    def output_width(self) -> int:
        """返回输出图像宽度。"""
        return int(self.width)

    @property
    def output_height(self) -> int:
        """返回输出图像高度。"""
        return int(self.height)

    @property
    def status(self) -> str:
        """返回 closed、waiting、reconnecting、streaming 或 error。"""
        with self._samples_lock:
            return self._state

    def _make_backend(self):
        """按原始设备路径和参数创建新的 LeRobot 相机。"""
        from lerobot.cameras.opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import (
            Cv2Backends,
            OpenCVCameraConfig,
        )

        backend = (
            Cv2Backends.V4L2
            if platform.startswith("linux")
            and isinstance(self.index_or_path, str)
            and self.index_or_path.startswith(("/dev/video", "/dev/v4l/"))
            else Cv2Backends.ANY
        )
        cfg = OpenCVCameraConfig(
            index_or_path=(
                Path(self.index_or_path)
                if isinstance(self.index_or_path, str)
                else self.index_or_path
            ),
            fps=self.fps,
            width=self.width,
            height=self.height,
            fourcc=self.fourcc,
            backend=backend,
        )
        return OpenCVCamera(cfg)

    @staticmethod
    def _close_backend(camera):
        """释放后端并返回可能仍在运行的底层读取线程。"""
        from lerobot.utils.errors import DeviceNotConnectedError

        reader = getattr(camera, "thread", None)
        try:
            camera.disconnect()
        except DeviceNotConnectedError:
            # LeRobot 在部分连接失败时已自行释放资源。
            pass
        return reader if reader is not None and reader.is_alive() else None

    def _release_backend(self, camera):
        """保存待清理对象，成功释放后记录可能仍在运行的读取线程。"""
        self._unfinished_backend = camera
        reader = self._close_backend(camera)
        self._pending_reader = reader
        self._unfinished_backend = None
        return reader

    def connect(self) -> None:
        """同步打开首个后端；成功后将生命周期交给监控线程。"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            raise RuntimeError("Camera monitor is still running.")
        if self._pending_reader is not None and self._pending_reader.is_alive():
            raise RuntimeError("Previous camera read thread is still running.")
        if self._unfinished_backend is not None:
            raise RuntimeError("Previous camera backend is not fully released.")
        self._cleanup_error = None
        self._monitor_stop.clear()
        camera = self._make_backend()
        self._camera = camera
        try:
            camera.connect()
        except BaseException as error:
            try:
                self._release_backend(camera)
            except Exception as cleanup_error:
                error.add_note(f"相机连接失败后的清理也失败：{cleanup_error}")
            self._camera = None
            raise
        self._perf_to_monotonic_ns = time.monotonic_ns() - time.perf_counter_ns()
        with self._samples_lock:
            self._samples.clear()
            self._state = "waiting"
        self._monitor_thread = threading.Thread(
            target=self._monitor_frames,
            name=f"handumi_camera_{self.index_or_path}",
            daemon=True,
        )
        self._monitor_thread.start()

    def async_read(self) -> np.ndarray:
        """返回最近真实帧的图像。"""
        return self.sample_at().image

    def sample_at(
        self, target_time_ns: int | None = None, *, after_sequence: int | None = None,
    ) -> CameraSample:
        """按目标时间选最近真实帧；可优先选择序号大于给定值的帧。"""
        deadline = time.monotonic() + 0.25
        while True:
            with self._samples_lock:
                samples = tuple(self._samples)
                state = self._state
                stopped = self._monitor_stop.is_set()
            # 兼容仅注入缓存的内部测试，同时在重连期间继续提供旧帧。
            if stopped or (state == "closed" and self._camera is None):
                raise RuntimeError("OpenCV camera is not connected.")
            if samples:
                break
            if time.monotonic() >= deadline:
                raise CameraFrameUnavailable(
                    f"Camera {self.index_or_path!r} produced no timestamped frame."
                )
            time.sleep(0.005)

        if after_sequence is not None:
            newer = tuple(sample for sample in samples if sample.sequence > after_sequence)
            if newer:
                samples = newer
        if target_time_ns is None:
            return samples[-1]
        return min(
            samples, key=lambda sample: abs(sample.capture_time_ns - target_time_ns)
        )

    def disconnect(self) -> None:
        """停止重连并等待后台释放；驱动阻塞时明确报告清理未完成。"""
        with self._samples_lock:
            self._state = "closed"
            self._monitor_stop.set()
        monitor = self._monitor_thread
        if monitor is not None:
            # LeRobot 释放设备时两次等待底层读线程，各最多两秒。
            monitor.join(timeout=5.0)
            if monitor.is_alive():
                raise TimeoutError("相机后台线程尚未退出，设备释放仍在进行。")
            self._monitor_thread = None
        elif self._camera is not None:
            self._release_backend(self._camera)
            self._camera = None
        if self._unfinished_backend is not None:
            self._release_backend(self._unfinished_backend)
            self._cleanup_error = None
        with self._samples_lock:
            self._samples.clear()
        if self._pending_reader is not None and self._pending_reader.is_alive():
            raise TimeoutError("相机底层读取线程尚未退出，无法确认设备已完全释放。")
        if self._cleanup_error is not None:
            raise RuntimeError("相机后台清理失败。") from self._cleanup_error

    def _monitor_frames(self) -> None:
        """镜像真实帧；停帧时在本线程内顺序关闭并重开后端。"""
        camera = self._camera
        last_timestamp: float | None = None
        last_new_frame = time.monotonic()
        first_frame = True
        poll_s = min(0.005, 1.0 / max(float(self.fps or 30) * 4.0, 1.0))
        try:
            while not self._monitor_stop.is_set():
                if camera is not None:
                    try:
                        with camera.frame_lock:
                            image = camera.latest_frame
                            timestamp = camera.latest_timestamp
                        if image is not None and timestamp is not None and timestamp != last_timestamp:
                            prepared = self._prepare_image(image)
                            with self._samples_lock:
                                if self._monitor_stop.is_set():
                                    break
                                last_timestamp = float(timestamp)
                                last_new_frame = time.monotonic()
                                self._sequence += 1
                                sample = CameraSample(
                                    image=prepared,
                                    capture_time_ns=(
                                        int(round(last_timestamp * 1e9))
                                        + self._perf_to_monotonic_ns
                                    ),
                                    sequence=self._sequence,
                                )
                                if first_frame:
                                    self._samples.clear()
                                    first_frame = False
                                self._samples.append(sample)
                                self._state = "streaming"
                    except Exception:
                        # 无法读取或转换帧时让停帧计时器触发重连。
                        log.debug("读取相机帧失败", exc_info=True)
                if time.monotonic() - last_new_frame < self._stall_timeout_s:
                    self._monitor_stop.wait(poll_s)
                    continue

                with self._samples_lock:
                    if self._monitor_stop.is_set():
                        break
                    self._state = "reconnecting"
                log.warning("相机 %s 已停帧，正在尝试自动重连。", self.index_or_path)
                device_path = self.index_or_path
                if (
                    not self._unstable_path_warned
                    and (
                        isinstance(device_path, int)
                        or (
                            isinstance(device_path, str)
                            and device_path.startswith("/dev/video")
                            and device_path.removeprefix("/dev/video").isdigit()
                        )
                    )
                ):
                    log.warning(
                        "相机使用的设备编号可能在 USB 重连后改变；"
                        "建议使用 /dev/v4l/by-path 或唯一的 by-id 稳定路径。"
                    )
                    self._unstable_path_warned = True
                if camera is not None:
                    reader = self._release_backend(camera)
                    camera = None
                    self._camera = None
                    # 驱动未退出前不能创建第二个采集者。
                    while reader is not None and reader.is_alive():
                        if self._monitor_stop.wait(self._retry_interval_s):
                            break
                    if self._monitor_stop.is_set():
                        break
                while not self._monitor_stop.wait(self._retry_interval_s):
                    candidate = None
                    try:
                        candidate = self._make_backend()
                        if self._monitor_stop.is_set():
                            self._release_backend(candidate)
                            break
                        candidate.connect()
                    except Exception as error:
                        if candidate is not None:
                            reader = self._release_backend(candidate)
                            while reader is not None and reader.is_alive():
                                if self._monitor_stop.wait(self._retry_interval_s):
                                    break
                        log.debug("相机重连失败：%s", error)
                        continue
                    if self._monitor_stop.is_set():
                        self._release_backend(candidate)
                        break
                    camera = candidate
                    self._camera = candidate
                    self._perf_to_monotonic_ns = time.monotonic_ns() - time.perf_counter_ns()
                    last_timestamp = None
                    last_new_frame = time.monotonic()
                    first_frame = True
                    with self._samples_lock:
                        if not self._monitor_stop.is_set():
                            self._state = "waiting"
                    break
        except Exception as error:
            log.exception("相机 %s 的后台恢复失败。", self.index_or_path)
            self._cleanup_error = error
            with self._samples_lock:
                if not self._monitor_stop.is_set():
                    self._state = "error"
        finally:
            if camera is not None:
                try:
                    reader = self._release_backend(camera)
                    if reader is not None:
                        log.error("相机底层读取线程尚未退出，设备释放未完全确认。")
                except Exception as error:
                    self._cleanup_error = error
                    log.exception("相机 %s 关闭失败。", self.index_or_path)
            self._camera = None

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """转换后端图像；子类可按相机格式覆盖。"""
        return np.asarray(image)
