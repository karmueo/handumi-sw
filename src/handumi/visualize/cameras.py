"""Non-blocking live camera views for physical teleoperation."""

from __future__ import annotations

import logging
import queue
import threading
import time
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from handumi.cameras import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_frames,
    resolve_camera_ids,
)

log = logging.getLogger("handumi.visualize.cameras")

_STOP = object()


class RerunCameraViewer:
    """Send only the newest camera frame batch to Rerun on a worker thread.

    The queue deliberately has one slot. When visualization falls behind, an
    old preview batch is replaced by the newest one; camera capture, robot
    control, and dataset writes never wait for Rerun or JPEG compression.
    """

    def __init__(
        self,
        camera_names: list[str],
        *,
        application_id: str,
        jpeg_quality: int = 75,
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.application_id = application_id
        self.jpeg_quality = jpeg_quality
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._failed = False
        self.dropped_batches = 0

    @property
    def healthy(self) -> bool:
        return not self._failed

    def start(self) -> None:
        if self._thread is not None or self._closed:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-rerun-cameras",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        frames: Mapping[str, np.ndarray],
        *,
        capture_time_ns: int | None = None,
    ) -> None:
        """Queue copied image arrays without ever waiting for the viewer."""
        if self._closed or self._failed or not frames:
            return
        images = {
            key: np.ascontiguousarray(frame).copy()
            for key, frame in frames.items()
            if key.startswith("observation.images.")
        }
        if not images:
            return
        item = (capture_time_ns or time.monotonic_ns(), images)
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped_batches += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped_batches += 1

    def close(self, *, timeout_s: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout_s))

    def _run(self) -> None:
        rr = None
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            rr.init(self.application_id, spawn=True)
            rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Grid(
                        *[
                            rrb.Spatial2DView(
                                origin=f"/observation.images.{name}", name=name
                            )
                            for name in self.camera_names
                        ]
                    ),
                    rrb.BlueprintPanel(state="collapsed"),
                    rrb.SelectionPanel(state="collapsed"),
                    rrb.TimePanel(state="collapsed"),
                ),
                make_active=True,
                make_default=True,
            )
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                capture_time_ns, images = item
                rr.set_time("capture_time", timestamp=np.datetime64(int(capture_time_ns), "ns"))
                for key, image in images.items():
                    rr.log(
                        key,
                        rr.Image(image).compress(jpeg_quality=self.jpeg_quality),
                    )
        except Exception:
            self._failed = True
            log.exception(
                "Rerun camera view failed; teleoperation and recording continue."
            )
        finally:
            if rr is not None:
                try:
                    rr.rerun_shutdown()
                except Exception:
                    log.debug("Rerun shutdown failed.", exc_info=True)


class OpenCVCameraViewer:
    """Show a disposable, low-overhead horizontal camera preview.

    Only explicitly selected images are copied into the one-slot queue.  The
    caller therefore remains non-blocking and an overloaded desktop merely
    drops preview frames instead of adding latency to teleoperation.
    """

    def __init__(
        self,
        camera_names: list[str],
        *,
        title: str = "HandUMI wrist cameras",
        window_width: int = 1600,
        window_height: int = 600,
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.title = title
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._failed = False
        self.dropped_batches = 0

    @property
    def healthy(self) -> bool:
        return not self._failed

    def start(self) -> None:
        if self._thread is not None or self._closed or not self.camera_names:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-opencv-cameras",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        frames: Mapping[str, np.ndarray],
        *,
        capture_time_ns: int | None = None,
    ) -> None:
        del capture_time_ns
        if self._closed or self._failed:
            return
        images = tuple(
            np.ascontiguousarray(frames[key]).copy()
            for name in self.camera_names
            if (key := f"observation.images.{name}") in frames
        )
        if not images:
            return
        try:
            self._queue.put_nowait(images)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
            self.dropped_batches += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(images)
        except queue.Full:
            self.dropped_batches += 1

    def close(self, *, timeout_s: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, timeout_s))

    @staticmethod
    def _side_by_side(cv2: Any, images: tuple[np.ndarray, ...]) -> np.ndarray:
        """Compose RGB frames without resizing when their heights already match."""
        target_height = min(int(image.shape[0]) for image in images)
        resized = []
        for image in images:
            if image.shape[0] == target_height:
                resized.append(image)
                continue
            target_width = max(
                1, int(round(image.shape[1] * target_height / image.shape[0]))
            )
            resized.append(
                cv2.resize(
                    image,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            )
        return np.hstack(resized)

    def _run(self) -> None:
        cv2 = None
        window_open = False
        try:
            import cv2

            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.title, self.window_width, self.window_height)
            window_open = True
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                images = item
                preview_rgb = self._side_by_side(cv2, images)
                cv2.imshow(
                    self.title,
                    cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    window_open = False
                    self._closed = True
                    break
        except Exception:
            self._failed = True
            log.exception(
                "OpenCV camera view failed; teleoperation and recording continue."
            )
        finally:
            if cv2 is not None and window_open:
                try:
                    cv2.destroyWindow(self.title)
                    cv2.waitKey(1)
                except Exception:
                    log.debug("OpenCV window shutdown failed.", exc_info=True)


class LiveCameraViews:
    """Own physical cameras and their best-effort, non-blocking preview."""

    def __init__(
        self,
        *,
        cameras: list[Any],
        camera_names: list[str],
        width: int,
        height: int,
        viewer: RerunCameraViewer | OpenCVCameraViewer,
    ) -> None:
        self.cameras = cameras
        self.camera_names = camera_names
        self.width = width
        self.height = height
        self.viewer = viewer
        self._read_failure_reported = False

    @classmethod
    def from_args(
        cls, args: Namespace, *, application_id: str
    ) -> LiveCameraViews | None:
        if args.no_rerun or args.skip_cameras:
            return None
        camera_names = list(args.cameras or ("left_wrist", "right_wrist"))
        camera_ids = resolve_camera_ids(
            None, args.rig_config, camera_names=camera_names
        )
        duplicates = {
            camera_id for camera_id in camera_ids if camera_ids.count(camera_id) > 1
        }
        if duplicates:
            mappings = ", ".join(
                f"{name}={camera_id}"
                for name, camera_id in zip(camera_names, camera_ids, strict=True)
            )
            raise SystemExit(
                f"Selected cameras must use distinct devices ({mappings})."
            )
        specs, _ = build_camera_specs(
            camera_ids,
            camera_names=camera_names,
            laptop_camera=False,
            laptop_cam_id=0,
            laptop_cam_name="laptop",
            rig_config=args.rig_config,
            default_fps=args.cam_fps,
            default_width=args.cam_width,
            default_height=args.cam_height,
        )
        cameras = connect_cameras(
            specs,
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            zero_non_laptop=False,
        )
        viewer = RerunCameraViewer(camera_names, application_id=application_id)
        viewer.start()
        return cls(
            cameras=cameras,
            camera_names=camera_names,
            width=args.cam_width,
            height=args.cam_height,
            viewer=viewer,
        )

    def update(self) -> dict[str, np.ndarray]:
        """Read the camera caches and enqueue a disposable preview batch."""
        try:
            frames = read_camera_frames(
                self.cameras,
                self.camera_names,
                width=self.width,
                height=self.height,
            )
        except Exception:
            if not self._read_failure_reported:
                log.exception(
                    "Camera preview read failed; robot control and recording continue."
                )
                self._read_failure_reported = True
            return {}
        self._read_failure_reported = False
        self.viewer.submit(frames)
        return frames

    def close(self) -> None:
        self.viewer.close()
        disconnect_cameras(self.cameras)


__all__ = ["LiveCameraViews", "OpenCVCameraViewer", "RerunCameraViewer"]
