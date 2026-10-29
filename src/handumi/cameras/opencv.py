"""OpenCV/LeRobot camera backend."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from sys import platform

import numpy as np

from handumi.cameras.base import CameraDevice, CameraSample


@dataclass
class OpenCVCameraDevice(CameraDevice):
    """CameraDevice adapter around LeRobot's OpenCV camera implementation."""

    index_or_path: int | str
    fps: int
    width: int
    height: int
    fourcc: str | None = None

    def __post_init__(self) -> None:
        self._camera = None
        self._samples: deque[CameraSample] = deque(
            maxlen=max(8, min(16, int(self.fps or 30)))
        )
        self._samples_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._perf_to_monotonic_ns = 0
        self._sequence = 0

    @property
    def output_width(self) -> int:
        return int(self.width)

    @property
    def output_height(self) -> int:
        return int(self.height)

    def connect(self) -> None:
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
        self._camera = OpenCVCamera(cfg)
        self._camera.connect()
        self._perf_to_monotonic_ns = time.monotonic_ns() - time.perf_counter_ns()
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_frames,
            name=f"handumi_camera_{self.index_or_path}",
            daemon=True,
        )
        self._monitor_thread.start()

    def async_read(self) -> np.ndarray:
        return self.sample_at().image

    def sample_at(self, target_time_ns: int | None = None) -> CameraSample:
        if self._camera is None:
            raise RuntimeError("OpenCV camera is not connected.")

        deadline = time.monotonic() + 0.25
        while True:
            with self._samples_lock:
                samples = tuple(self._samples)
            if samples:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Camera {self.index_or_path!r} produced no timestamped frame."
                )
            time.sleep(0.005)

        if target_time_ns is None:
            return samples[-1]
        return min(
            samples, key=lambda sample: abs(sample.capture_time_ns - target_time_ns)
        )

    def disconnect(self) -> None:
        """关闭相机及采集线程；连接尚未完成时也允许安全清理。"""
        if self._camera is None:
            return
        from lerobot.utils.errors import DeviceNotConnectedError

        self._monitor_stop.set()
        monitor = self._monitor_thread
        if monitor is not None:
            monitor.join(timeout=1.0)
        self._monitor_thread = None
        try:
            self._camera.disconnect()
        except DeviceNotConnectedError:
            # 底层打开失败后已释放设备，无需再次断开。
            pass
        finally:
            self._camera = None
            with self._samples_lock:
                self._samples.clear()

    def _monitor_frames(self) -> None:
        """Mirror LeRobot's native capture buffer without resampling it."""
        last_timestamp: float | None = None
        poll_s = min(0.005, 1.0 / max(float(self.fps or 30) * 4.0, 1.0))
        while not self._monitor_stop.is_set():
            camera = self._camera
            if camera is None:
                break
            try:
                with camera.frame_lock:
                    image = camera.latest_frame
                    timestamp = camera.latest_timestamp
                if (
                    image is not None
                    and timestamp is not None
                    and timestamp != last_timestamp
                ):
                    last_timestamp = float(timestamp)
                    self._sequence += 1
                    capture_time_ns = (
                        int(round(last_timestamp * 1e9)) + self._perf_to_monotonic_ns
                    )
                    sample = CameraSample(
                        image=self._prepare_image(image),
                        capture_time_ns=capture_time_ns,
                        sequence=self._sequence,
                    )
                    with self._samples_lock:
                        self._samples.append(sample)
            except Exception:
                # The recorder health gate observes the resulting stale sample.
                pass
            self._monitor_stop.wait(poll_s)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize one backend frame before exposing it to consumers."""
        return np.asarray(image)
