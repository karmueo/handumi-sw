"""Record joint-level real-robot teleoperation demonstrations.

This is the recording sibling of ``handumi teleop-real``. The operator drives
the real robot with HandUMI tracking and Feetech gripper widths, while each
LeRobot row stores canonical robot joints directly:

* ``observation.state`` is the robot feedback read from the real backend.
* ``action`` is the next joint command produced by the teleop controller.

Before recording, controller->TCP calibration and Feetech calibration must be
available. Episode control leaves time to reset the physical task between runs:

* double-squeeze right while waiting: start the episode from home
* double-squeeze right while recording: save the episode and return home
* double-squeeze left while recording: reset the same episode and return home
* double-squeeze both grippers: discard the active episode and finish the session
* ``Esc`` / ``Ctrl+C``: discard the active attempt and stop

Gripper commands may remain synchronized while recording is waiting to start,
but arm teleoperation and capture are disabled in READY. Once recording starts,
opening a HandUMI gripper activates and anchors that arm;
holding the corresponding HandUMI gripper fully closed for two seconds parks it
at home (disable with ``--no-park-on-hold``). Episode gestures control
recording only and never activate an arm.

PICO tracking uses ADB.

Examples
--------
::

    handumi teleop-record --device pico --robot piper --output-dir outputs/piper-demo
    handumi teleop-record --device pico --robot openarmv1 --output-dir outputs/openarm-demo
    handumi teleop-record --device pico --robot piper --side right --output-dir outputs/right-arm-demo
    handumi teleop-record --device pico --robot piper \
        --output-dir outputs/piper-demo --resume

Common options:

* ``--device``        pico|meta tracking device.
* ``--robot``         Registered real backend, for example piper or openarmv1.
* ``--side``          left|right|both enabled arms.
* ``--fps``           Dataset/camera capture frequency in Hz.
* ``--control-fps``   Tracking and IK frequency (DLS defaults to 72 Hz).
* ``--num-episodes``  Number of episodes to record; 0 means until stopped.
* ``--episode-time-s`` Auto-save duration; 0 means until a save gesture.
* ``--task``          Task description stored in the dataset.
* ``--output-dir``    Destination directory, for example ``outputs/piper-demo``.
* ``--resume``        Append episodes from an existing dataset in that directory.
"""

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Real-time IK favors CPU tail latency; an explicit environment override wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from handumi.audio import (
    AudioCaptureError,
    PicoAudioRecorder,
    audio_features,
    audio_metadata,
)
from handumi.cameras import (
    build_camera_specs,
    camera_output_size,
    connect_cameras,
    disconnect_cameras,
    read_camera_samples,
    resolve_camera_ids,
)
from handumi.dataset.canonical import canonical_joint_layout, canonicalize_command
from handumi.dataset.capture import (
    CAMERA_STALE_TIMEOUT_S,
    GRIPPER_STALE_TIMEOUT_S,
    MAX_SYNC_SKEW_S,
    SENSOR_LOSS_TIMEOUT_S,
    SYNC_LAG_S,
    TRACKING_LOSS_TIMEOUT_S,
)
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    camera_health_features,
    capture_timing_features,
    feetech_features,
)
from handumi.feetech import FeetechGripperPair, FeetechGripperSampler, GripperWidths
from handumi.real.registry import make_real_backend
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.scripts.record import (
    _SOFTWARE_VIDEO_CODEC,
    StreamingEncodingError,
    _EscapeStopListener,
    _install_strict_streaming_encoder,
    _prepare_streaming_episode,
    _RecordingDashboard,
    _robot_metadata,
    _select_video_encoder,
    _validate_finalized_lerobot_dataset,
    _write_dataset_readme,
    build_tracker,
    connect_feetech,
)
from handumi.synchronization import (
    SustainedHealthGate,
    capture_timing_frame,
    synchronized_gripper_frame,
)
from handumi.teleop.common import (
    AdaptiveJointFilter,
    BestEffortPeriodicWorker,
    JointMotionDiagnostics,
    KeyboardSpaceListener,
    TeleopLoopTimer,
)
from handumi.teleop.common import (
    enabled_sides as _enabled_sides,
)
from handumi.teleop.common import (
    enabled_tracking_ok as _enabled_tracking_ok,
)
from handumi.teleop.common import (
    latest_widths as _latest_widths,
)
from handumi.teleop.common import (
    tracking_world_map as _tracking_world_map,
)
from handumi.teleop.core import TeleopController
from handumi.teleop.dls import (
    make_real_teleop_dls_solver,
    resolve_real_teleop_timing,
)
from handumi.teleop.hardware import (
    load_required_controller_tcp_calibration as _load_required_calibration,
)
from handumi.teleop.hardware import (
    validate_feetech_ready as _validate_feetech_ready,
)
from handumi.teleop.motion import (
    TeleopMotionConfig,
    validate_teleop_motion_args,
)
from handumi.teleop.physical import (
    DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S,
    add_physical_teleop_arguments,
    validate_physical_teleop_args,
)
from handumi.teleop.session import TeleopSession
from handumi.teleop.standby import (
    GRIPPER_PARK_HOLD_S,
    GripperHomeStandby,
)
from handumi.teleop.tracking import LatestTrackingSampler, TrackingRecoveryPolicy
from handumi.teleop.trajectory import TeleopCommandStream
from handumi.tracking.base import TrackingProvider
from handumi.tracking.gestures import BilateralClapArbiter, DoubleClapDetector
from handumi.utils.speech import log_say
from handumi.visualize import LiveCameraViews, OpenCVCameraViewer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
record_log = logging.getLogger("handumi.record_teleop")


class DatasetWriteError(RuntimeError):
    """A background LeRobot write failed or could not keep up."""


class SensorHealthError(RuntimeError):
    """A required capture stream remained unhealthy for too long."""


@dataclass(frozen=True)
class _CaptureSnapshot:
    base_q: np.ndarray
    action_q: np.ndarray
    openings: dict[str, float]
    target_time_ns: int
    record_time_ns: int
    tracking_time_ns: int
    submitted_s: float


@dataclass(frozen=True)
class CaptureTimingStats:
    samples: int
    queue_delay_avg_ms: float
    queue_delay_max_ms: float
    capture_avg_ms: float
    capture_max_ms: float
    robot_read_avg_ms: float
    robot_read_max_ms: float
    cameras_avg_ms: float
    cameras_max_ms: float
    gripper_avg_ms: float
    gripper_max_ms: float
    frame_build_avg_ms: float
    frame_build_max_ms: float


class _DatasetRequest:
    def __init__(self, operation: str, payload: Any = None) -> None:
        self.operation = operation
        self.payload = payload
        self.done = threading.Event()
        self.error: BaseException | None = None


class AsyncLeRobotWriter:
    """Serialize LeRobot operations away from the real-time control loop."""

    def __init__(
        self,
        dataset: Any,
        *,
        max_pending_frames: int,
        use_videos: bool,
    ) -> None:
        if max_pending_frames <= 0:
            raise ValueError("max_pending_frames must be > 0")
        self.dataset = dataset
        self.use_videos = bool(use_videos)
        self._queue: queue.Queue[_DatasetRequest] = queue.Queue(
            maxsize=max_pending_frames
        )
        self._episode_error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._deferred_total_s = 0.0
        self._deferred_max_s = 0.0
        self._dataset_total_s = 0.0
        self._dataset_max_s = 0.0
        self._timing_samples = 0
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-lerobot-writer",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    @property
    def pending_frames(self) -> int:
        return self._queue.qsize()

    def timing_stats(
        self, *, reset: bool = False
    ) -> tuple[float, float, float, float]:
        """Return deferred-build and LeRobot add avg/max times in milliseconds."""
        with self._timing_lock:
            samples = self._timing_samples
            result = (
                self._deferred_total_s / samples * 1000.0 if samples else 0.0,
                self._deferred_max_s * 1000.0,
                self._dataset_total_s / samples * 1000.0 if samples else 0.0,
                self._dataset_max_s * 1000.0,
            )
            if reset:
                self._deferred_total_s = 0.0
                self._deferred_max_s = 0.0
                self._dataset_total_s = 0.0
                self._dataset_max_s = 0.0
                self._timing_samples = 0
            return result

    def add_frame(
        self,
        frame: dict[str, Any] | Callable[[], dict[str, Any]],
    ) -> None:
        """Queue a row or a deferred row builder.

        Deferred builders are useful for timestamp-aligned peripheral work
        (notably audio), which must not run in the latency-sensitive IK loop.
        """
        self.raise_if_failed()
        try:
            self._queue.put_nowait(_DatasetRequest("frame", frame))
        except queue.Full as exc:
            raise DatasetWriteError(
                "LeRobot writer queue is full; recording cannot remain frame-aligned"
            ) from exc

    def flush(self) -> None:
        """Wait until every previously queued frame has been materialized."""
        self._request("flush")

    def save_episode(self, expected_frames: int) -> None:
        self._request("save", int(expected_frames))

    def clear_episode(self) -> None:
        self._request("clear")

    def finalize(self) -> None:
        try:
            self._request("finalize")
        finally:
            self._stop_thread()

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._episode_error
        if error is not None:
            raise DatasetWriteError(
                "Background LeRobot writer failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.clear_episode()
        except Exception:
            pass
        self._stop_thread()

    def _request(self, operation: str, payload: Any = None) -> None:
        if self._closed:
            raise DatasetWriteError("LeRobot writer is closed")
        request = _DatasetRequest(operation, payload)
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise DatasetWriteError(
                f"LeRobot writer operation {operation!r} failed"
            ) from request.error

    def _stop_thread(self) -> None:
        if self._closed:
            return
        self._closed = True
        request = _DatasetRequest("stop")
        self._queue.put(request)
        request.done.wait()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            try:
                if request.operation == "stop":
                    return
                if request.operation == "frame":
                    with self._error_lock:
                        episode_error = self._episode_error
                    if episode_error is None:
                        try:
                            payload = request.payload
                            deferred_start_s = time.perf_counter()
                            frame = payload() if callable(payload) else payload
                            deferred_elapsed_s = time.perf_counter() - deferred_start_s
                            dataset_start_s = time.perf_counter()
                            self.dataset.add_frame(frame)
                            dataset_elapsed_s = time.perf_counter() - dataset_start_s
                            with self._timing_lock:
                                self._deferred_total_s += deferred_elapsed_s
                                self._deferred_max_s = max(
                                    self._deferred_max_s, deferred_elapsed_s
                                )
                                self._dataset_total_s += dataset_elapsed_s
                                self._dataset_max_s = max(
                                    self._dataset_max_s, dataset_elapsed_s
                                )
                                self._timing_samples += 1
                        except BaseException as exc:
                            with self._error_lock:
                                self._episode_error = exc
                    continue
                if request.operation == "flush":
                    self.raise_if_failed()
                    continue
                if request.operation == "clear":
                    try:
                        self.dataset.clear_episode_buffer()
                    finally:
                        with self._error_lock:
                            self._episode_error = None
                    continue
                if request.operation == "save":
                    self.raise_if_failed()
                    if self.use_videos:
                        _prepare_streaming_episode(self.dataset, request.payload)
                    self.dataset.save_episode()
                    continue
                if request.operation == "finalize":
                    self.raise_if_failed()
                    self.dataset.finalize()
                    continue
                raise RuntimeError(f"Unknown dataset operation: {request.operation}")
            except BaseException as exc:
                request.error = exc
            finally:
                request.done.set()
                self._queue.task_done()


class AsyncEpisodeCapture:
    """Capture feedback and sensors without blocking the DLS control loop.

    The command loop only publishes small immutable snapshots.  This worker
    performs robot feedback reads, synchronized camera/gripper selection,
    canonicalization, and dataset enqueueing in order.  A bounded queue turns
    sustained capture overload into an explicit rejected attempt instead of
    silently adding latency to robot control.
    """

    def __init__(
        self,
        *,
        real_env,
        runtime,
        dataset_writer: AsyncLeRobotWriter | None,
        cameras: list[Any],
        camera_names: list[str],
        camera_width: int,
        camera_height: int,
        camera_stale_timeout_s: float,
        grippers: FeetechGripperSampler | FeetechGripperPair | None,
        gripper_stale_timeout_s: float,
        max_sync_skew_s: float,
        sensor_loss_timeout_s: float,
        task: str,
        audio_recorder: PicoAudioRecorder | None,
        max_pending_captures: int = 2,
    ) -> None:
        self.real_env = real_env
        self.runtime = runtime
        self.dataset_writer = dataset_writer
        self.cameras = cameras
        self.camera_names = camera_names
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.camera_stale_timeout_s = float(camera_stale_timeout_s)
        self.grippers = grippers
        self.gripper_stale_timeout_s = float(gripper_stale_timeout_s)
        self.max_sync_skew_s = float(max_sync_skew_s)
        self.max_sync_skew_ns = int(max_sync_skew_s * 1e9)
        self.health_gate = SustainedHealthGate(sensor_loss_timeout_s)
        self.task = task
        self.audio_recorder = audio_recorder
        self._queue: queue.Queue[_DatasetRequest] = queue.Queue(
            maxsize=max_pending_captures
        )
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._data_lock = threading.Lock()
        self._observations: list[np.ndarray] = []
        self._commands: list[np.ndarray] = []
        self._pending_dataset_frame: tuple[dict[str, Any], int, int] | None = None
        self._timing_lock = threading.Lock()
        self._timing = {
            "samples": 0,
            "queue_total": 0.0,
            "queue_max": 0.0,
            "capture_total": 0.0,
            "capture_max": 0.0,
            "robot_total": 0.0,
            "robot_max": 0.0,
            "camera_total": 0.0,
            "camera_max": 0.0,
            "gripper_total": 0.0,
            "gripper_max": 0.0,
            "build_total": 0.0,
            "build_max": 0.0,
        }
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-episode-capture",
            daemon=True,
        )
        self._thread.start()

    @property
    def pending_captures(self) -> int:
        return self._queue.qsize()

    @property
    def captured_frames(self) -> int:
        with self._data_lock:
            return len(self._observations)

    def submit(self, snapshot: _CaptureSnapshot) -> None:
        self.raise_if_failed()
        try:
            self._queue.put_nowait(_DatasetRequest("capture", snapshot))
        except queue.Full as exc:
            error = DatasetWriteError(
                "capture worker queue is full; capture cannot keep up with dataset FPS"
            )
            with self._error_lock:
                self._error = error
            raise error from exc

    def finish(self) -> tuple[np.ndarray, np.ndarray, int]:
        self._request("flush")
        with self._data_lock:
            if len(self._observations) < 2:
                size = canonical_joint_layout(self.runtime).size
                return (
                    np.empty((0, size), dtype=np.float32),
                    np.empty((0, size), dtype=np.float32),
                    0,
                )
            states = np.asarray(self._observations[:-1], dtype=np.float32)
            actions = np.asarray(self._commands[1:], dtype=np.float32)
        return states, actions, len(states)

    def timing_stats(self, *, reset: bool = False) -> CaptureTimingStats:
        with self._timing_lock:
            timing = dict(self._timing)
            samples = int(timing["samples"])
            divisor = max(samples, 1)
            result = CaptureTimingStats(
                samples=samples,
                queue_delay_avg_ms=timing["queue_total"] / divisor * 1000.0,
                queue_delay_max_ms=timing["queue_max"] * 1000.0,
                capture_avg_ms=timing["capture_total"] / divisor * 1000.0,
                capture_max_ms=timing["capture_max"] * 1000.0,
                robot_read_avg_ms=timing["robot_total"] / divisor * 1000.0,
                robot_read_max_ms=timing["robot_max"] * 1000.0,
                cameras_avg_ms=timing["camera_total"] / divisor * 1000.0,
                cameras_max_ms=timing["camera_max"] * 1000.0,
                gripper_avg_ms=timing["gripper_total"] / divisor * 1000.0,
                gripper_max_ms=timing["gripper_max"] * 1000.0,
                frame_build_avg_ms=timing["build_total"] / divisor * 1000.0,
                frame_build_max_ms=timing["build_max"] * 1000.0,
            )
            if reset:
                for key in self._timing:
                    self._timing[key] = 0
            return result

    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        request = _DatasetRequest("stop")
        self._queue.put(request)
        request.done.wait()
        self._thread.join(timeout=2.0)

    def _request(self, operation: str) -> None:
        if self._closed:
            raise DatasetWriteError("episode capture worker is closed")
        request = _DatasetRequest(operation)
        self._queue.put(request)
        request.done.wait()
        if request.error is not None:
            raise request.error

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            try:
                if request.operation == "stop":
                    return
                if request.operation == "capture":
                    with self._error_lock:
                        error = self._error
                    if error is None:
                        try:
                            self._capture(request.payload)
                        except BaseException as exc:
                            with self._error_lock:
                                self._error = exc
                    continue
                if request.operation == "flush":
                    self.raise_if_failed()
                    continue
                raise RuntimeError(
                    f"Unknown episode capture operation: {request.operation}"
                )
            except BaseException as exc:
                request.error = exc
            finally:
                request.done.set()
                self._queue.task_done()

    def _capture(self, snapshot: _CaptureSnapshot) -> None:
        capture_start_s = time.perf_counter()
        queue_delay_s = max(0.0, capture_start_s - snapshot.submitted_s)

        robot_start_s = time.perf_counter()
        observation_q = self.real_env.read(base_q=snapshot.base_q)
        robot_s = time.perf_counter() - robot_start_s

        camera_start_s = time.perf_counter()
        camera_frame, camera_health = read_camera_samples(
            self.cameras,
            self.camera_names,
            target_time_ns=snapshot.target_time_ns,
            record_time_ns=snapshot.record_time_ns,
            width=self.camera_width,
            height=self.camera_height,
            stale_timeout_s=self.camera_stale_timeout_s,
            max_sync_skew_s=self.max_sync_skew_s,
        )
        camera_s = time.perf_counter() - camera_start_s

        gripper_start_s = time.perf_counter()
        gripper_frame = synchronized_gripper_frame(
            self.grippers,
            target_time_ns=snapshot.target_time_ns,
            record_time_ns=snapshot.record_time_ns,
            stale_timeout_s=self.gripper_stale_timeout_s,
            max_sync_skew_s=self.max_sync_skew_s,
        )
        gripper_s = time.perf_counter() - gripper_start_s

        tracking_sync_ok = bool(
            snapshot.tracking_time_ns > 0
            and abs(snapshot.tracking_time_ns - snapshot.target_time_ns)
            <= self.max_sync_skew_ns
        )
        sensor_health = {
            **camera_health,
            "feetech": gripper_frame.healthy_for_gate,
            "tracking": tracking_sync_ok,
        }
        _, timed_out_sensors = self.health_gate.update(
            sensor_health, snapshot.record_time_ns
        )
        if timed_out_sensors:
            names = ", ".join(sorted(timed_out_sensors))
            raise SensorHealthError(f"Sensor health timed out: {names}")

        build_start_s = time.perf_counter()
        canonical_observation = canonicalize_command(
            observation_q,
            runtime=self.runtime,
            openings={
                "left": gripper_frame.widths.left_normalized,
                "right": gripper_frame.widths.right_normalized,
            },
        )
        canonical_action = canonicalize_command(
            snapshot.action_q,
            runtime=self.runtime,
            openings=snapshot.openings,
        )
        if self.dataset_writer is not None and self._pending_dataset_frame is not None:
            pending_frame, pending_target_ns, pending_record_ns = (
                self._pending_dataset_frame
            )
            self.dataset_writer.add_frame(
                _deferred_audio_frame(
                    {
                        **pending_frame,
                        "action": canonical_action,
                        "task": self.task,
                    },
                    audio_recorder=self.audio_recorder,
                    target_time_ns=pending_target_ns,
                    record_time_ns=pending_record_ns,
                    stale_timeout_s=self.camera_stale_timeout_s,
                    max_sync_skew_s=self.max_sync_skew_s,
                )
            )
        self._pending_dataset_frame = (
            {
                **camera_frame,
                "observation.state": canonical_observation,
                **_gripper_width_frame(gripper_frame.widths),
                **gripper_frame.frame,
                **capture_timing_frame(
                    snapshot.target_time_ns, snapshot.record_time_ns
                ),
                "calibration_id": np.array([-1], dtype=np.int64),
                "source_kind": np.array([1], dtype=np.int64),
            },
            snapshot.target_time_ns,
            snapshot.record_time_ns,
        )
        with self._data_lock:
            self._observations.append(canonical_observation)
            self._commands.append(canonical_action)
        build_s = time.perf_counter() - build_start_s
        capture_s = time.perf_counter() - capture_start_s
        with self._timing_lock:
            timing = self._timing
            timing["samples"] += 1
            for prefix, elapsed_s in (
                ("queue", queue_delay_s),
                ("capture", capture_s),
                ("robot", robot_s),
                ("camera", camera_s),
                ("gripper", gripper_s),
                ("build", build_s),
            ):
                timing[f"{prefix}_total"] += elapsed_s
                timing[f"{prefix}_max"] = max(
                    timing[f"{prefix}_max"], elapsed_s
                )


def _log_episode_interface(
    episode: int,
    total: str,
    *,
    waiting: bool,
    space_start: bool,
    park_hold_s: float,
    park_on_hold: bool,
) -> None:
    state = "READY AT HOME" if waiting else "RECORDING"
    start = "double-squeeze RIGHT"
    if space_start:
        start += " or press SPACE"
    park_line = (
        f"│ During REC: HandUMI gripper closed {park_hold_s:.1f} s parks that arm\n"
        if park_on_hold
        else ""
    )
    record_log.info(
        "\n"
        "┌─ HandUMI teleop recording "
        "─────────────────────────────\n"
        "│ Episode %d/%s  •  %s\n"
        "│ Start: %s\n"
        "│ Start / save: double-squeeze RIGHT\n"
        "│ Reset same episode + home: double-squeeze LEFT\n"
        "│ Finish session: double-squeeze BOTH  •  Stop: Esc / Ctrl+C\n"
        "│ After save/reset: wait for HOME, then start with RIGHT\n"
        "%s"
        "└────────────────────────────"
        "─────────────────────────────",
        episode,
        total,
        state,
        start,
        park_line,
    )


# ``_BilateralClapArbiter`` lives in ``handumi.tracking.gestures`` as
# ``BilateralClapArbiter`` so ``record.py`` can share it; keep this alias so
# the rest of this module (and its tests) can refer to it unchanged.
_BilateralClapArbiter = BilateralClapArbiter


class _EpisodePhase(Enum):
    """Operator-visible states for a single episode attempt."""

    READY = "ready"
    RECORDING = "recording"


def _episode_gesture_action(
    gesture: str | None, *, recording: bool
) -> str | None:
    """Map a resolved gripper gesture to the episode state transition."""
    if gesture == "both":
        return "finish"
    if gesture == "right":
        return "save" if recording else "start"
    if gesture == "left" and recording:
        return "reset"
    return None


def _deferred_audio_frame(
    frame: dict[str, Any],
    *,
    audio_recorder: PicoAudioRecorder | None,
    target_time_ns: int,
    record_time_ns: int,
    stale_timeout_s: float,
    max_sync_skew_s: float,
) -> Callable[[], dict[str, Any]]:
    """Build audio alignment in the dataset thread, after packet look-ahead.

    The PICO can deliver a PCM packet slightly after the video/control row it
    covers.  Resolving it synchronously in the IK loop both adds avoidable work
    and can falsely report a missing stream.  The writer naturally provides a
    small look-ahead while preserving the original row timestamps.
    """
    frozen_frame = dict(frame)

    def build() -> dict[str, Any]:
        if audio_recorder is None:
            return frozen_frame
        audio_frame, _ = audio_recorder.synchronized_frame(
            target_time_ns,
            record_time_ns,
            stale_timeout_s=stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        return {**frozen_frame, **audio_frame}

    return build


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    joint_names: list[str] | tuple[str, ...],
    camera_specs: list[dict[str, object]] | None = None,
    record_audio: bool = False,
) -> dict[str, Any]:
    img_dtype = "video" if use_videos else "image"
    features: dict[str, Any] = {}
    specs_by_name = {str(spec["name"]): spec for spec in (camera_specs or [])}
    for cam in cam_names:
        width, height = camera_output_size(
            specs_by_name.get(cam, {}),
            default_width=cam_width,
            default_height=cam_height,
        )
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    state_action = joint_state_feature(joint_names)
    features["observation.state"] = state_action
    features["action"] = dict(state_action)
    features.update(feetech_features())
    features.update(capture_timing_features())
    features.update(camera_health_features(cam_names))
    if record_audio:
        features.update(audio_features())
    features["calibration_id"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["calibration_id"],
    }
    features["source_kind"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["source_kind"],
    }
    return features


def joint_state_feature(joint_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    names = list(joint_names)
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }


def build_joint_frame(
    *,
    observation_q: np.ndarray,
    action_q: np.ndarray,
    widths: GripperWidths,
) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.asarray(observation_q, dtype=np.float32).copy(),
        "action": np.asarray(action_q, dtype=np.float32).copy(),
        **_gripper_width_frame(widths),
    }


def _gripper_width_frame(widths: GripperWidths) -> dict[str, np.ndarray]:
    """Build the calibrated Feetech values required by every dataset row."""
    return {
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array(
            [widths.right_ticks], dtype=np.int64
        ),
        "observation.feetech.left_width_mm": np.array(
            [widths.left_mm], dtype=np.float32
        ),
        "observation.feetech.right_width_mm": np.array(
            [widths.right_mm], dtype=np.float32
        ),
        "observation.feetech.left_normalized": np.array(
            [widths.left_normalized], dtype=np.float32
        ),
        "observation.feetech.right_normalized": np.array(
            [widths.right_normalized], dtype=np.float32
        ),
    }


def _gripper_openings(widths: GripperWidths) -> dict[str, float]:
    return {
        "left": float(widths.left_normalized),
        "right": float(widths.right_normalized),
    }


def _home_between_episodes(
    *,
    real_env,
    controller: TeleopController,
    home_q: np.ndarray,
    enabled_sides: tuple[str, ...],
    command_stream: TeleopCommandStream,
    joint_filter: AdaptiveJointFilter,
    home_standby: GripperHomeStandby,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    motion_joint_indices: tuple[int, ...],
    dashboard: _RecordingDashboard | None,
    episode: int,
    episode_total: str,
    play_sounds: bool,
    ready_after: bool = True,
) -> None:
    """Synchronously home the robot and leave every enabled arm inactive."""
    if dashboard is not None:
        dashboard.set_episode(
            episode,
            episode_total,
            state="HOMING",
            detail="Returning both arms home; wait before resetting the task",
        )
    record_log.info("Returning enabled arms home before the next episode ...")
    log_say("returning home", play_sounds=play_sounds)
    command_stream.stop()
    # ``home()`` is the one-time hardware/streamer initialization performed
    # during startup.  Calling it again here would create a second backend
    # command streamer while the original one can still be publishing the
    # previous episode, making the two streams fight over the robot.  Reuse
    # the initialized streamer for every episode transition instead.
    real_env.move_home(home_q)
    reset_q = controller.reset()
    joint_filter.reset(reset_q)
    home_standby.enter_standby(enabled_sides)
    command_stream.clear_joint_rate_limits(motion_joint_indices)
    widths = _latest_widths(grippers)
    command_stream.submit(
        reset_q,
        _gripper_openings(widths),
        time_s=time.perf_counter(),
        active=True,
        new_epoch=True,
    )
    if ready_after:
        record_log.info(
            "Robot is at home; double-squeeze RIGHT when the task is ready."
        )
    else:
        record_log.info("Robot is at home; recording session finished.")
    if dashboard is not None:
        dashboard.set_episode(
            episode,
            episode_total,
            state="READY" if ready_after else "FINISHED",
            detail=(
                "Robot at home; double-squeeze RIGHT to start"
                if ready_after
                else "Robot at home; recording session finished"
            ),
        )


def _parse_record_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    p = argparse.ArgumentParser(
        description="Record real-robot HandUMI teleoperation demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--help-advanced", action="store_true", help="Show expert hardware options."
    )
    add_physical_teleop_arguments(p)
    p.add_argument(
        "--ik-solver",
        choices=("dls", "lm"),
        default="dls",
        help="IK follower; defaults to the same incremental DLS as teleop-real.",
    )
    p.add_argument(
        "--control-fps",
        type=float,
        default=None,
        help="Tracking/IK rate; default 72 for DLS and 30 for LM.",
    )
    # The dataset stays at --fps (normally 30), while the real-teleop control
    # path gets its own cadence and interpolation delay.
    p.set_defaults(trajectory_delay_ms=None)
    p.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Episodes to save; 0 runs until double-squeeze BOTH.",
    )
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=0.0,
        help="Auto-save active episodes after this duration; 0 waits for RIGHT.",
    )
    p.add_argument("--task", type=str, default="HandUMI real teleop recording")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local dataset directory, for example outputs/piper-demo.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to the finalized dataset in --output-dir.",
    )
    p.add_argument(
        "--record-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record the PICO app microphone stream as one WAV per episode.",
    )
    if not show_advanced:
        normal = {
            "help",
            "help_advanced",
            "device",
            "robot",
            "side",
            "space_start",
            "no_park_on_hold",
            "no_sounds",
            "num_episodes",
            "episode_time_s",
            "ik_solver",
            "control_fps",
            "task",
            "output_dir",
            "resume",
            "record_audio",
            "cameras",
            "skip_cameras",
            "no_rerun",
        }
        for action in p._actions:
            if action.dest not in normal:
                action.help = argparse.SUPPRESS
    else:
        p.print_help()
        raise SystemExit(0)
    args = p.parse_args(raw_argv)
    args.control_fps, args.trajectory_delay_ms = resolve_real_teleop_timing(
        args.ik_solver,
        input_rate_hz=args.control_fps,
        trajectory_delay_ms=args.trajectory_delay_ms,
    )
    _apply_recording_defaults(args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_record_args(argv)


def _apply_recording_defaults(args: argparse.Namespace) -> None:
    args.sync_lag_s = SYNC_LAG_S
    args.max_sync_skew_s = MAX_SYNC_SKEW_S
    args.gripper_stale_timeout_s = GRIPPER_STALE_TIMEOUT_S
    args.sensor_loss_timeout_s = SENSOR_LOSS_TIMEOUT_S
    # Match teleop-real: keep serial reads off the control loop and sample at
    # least at 100 Hz (or at the input rate when configured above 100 Hz).
    args.feetech_sample_hz = max(100.0, float(args.control_fps))
    args.tracking_loss_timeout_s = TRACKING_LOSS_TIMEOUT_S
    args.camera_stale_timeout_s = CAMERA_STALE_TIMEOUT_S
    # PICO defaults to the same ADB transport used by the recording workflow,
    # while an explicit --pico-wifi selection remains available.
    if not args.pico_wifi:
        args.pico_adb = True


def _validate_record_args(args: argparse.Namespace) -> None:
    validate_teleop_motion_args(args)
    validate_physical_teleop_args(args)
    if args.num_episodes < 0:
        raise SystemExit("--num-episodes must be >= 0.")
    if args.episode_time_s < 0.0:
        raise SystemExit("--episode-time-s must be >= 0.")
    if args.control_fps <= 0.0:
        raise SystemExit("--control-fps must be > 0.")
    if args.record_audio and args.device != "pico":
        raise SystemExit("--record-audio currently requires --device pico.")
    for name in (
        "sync_lag_s",
        "max_sync_skew_s",
        "gripper_stale_timeout_s",
        "camera_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
        "tracking_loss_timeout_s",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero.")


def _wait_for_tracking_sampler(
    tracking_sampler: LatestTrackingSampler,
    stop_event: threading.Event,
    *,
    enabled_sides: tuple[str, ...],
    tracking_stale_ms: float,
    poll_s: float = 0.05,
) -> bool:
    """Wait for a fresh sample without polling the tracking SDK inline."""
    last_report = float("-inf")
    while not stop_event.is_set():
        now = time.monotonic()
        snapshot = tracking_sampler.latest()
        if snapshot is not None:
            sample = snapshot.sample
            side_tracked = {
                "left": sample.left_tracked,
                "right": sample.right_tracked,
            }
            if snapshot.age_s(
                now
            ) <= tracking_stale_ms / 1000.0 and _enabled_tracking_ok(
                side_tracked, enabled_sides
            ):
                record_log.info("Enabled controllers tracked; recording gate open.")
                return True
        if now - last_report >= 2.0:
            record_log.warning("Waiting for fresh controller tracking ...")
            last_report = now
        stop_event.wait(poll_s)
    return False


def record_episode(
    *,
    tracker: TrackingProvider,
    tracking_sampler: LatestTrackingSampler,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    real_env,
    controller: TeleopController,
    runtime,
    home_q: np.ndarray,
    enabled_sides: tuple[str, ...],
    space_listener: KeyboardSpaceListener,
    clap_detector: DoubleClapDetector,
    clap_arbiter: _BilateralClapArbiter,
    fps: int,
    task: str,
    stop_event: threading.Event,
    play_sounds: bool,
    initial_start_sides: tuple[str, ...],
    sync_lag_s: float,
    max_sync_skew_s: float,
    gripper_stale_timeout_s: float,
    sensor_loss_timeout_s: float,
    tracking_loss_timeout_s: float,
    tracking_stale_ms: float,
    command_stream: TeleopCommandStream,
    control_fps: float | None = None,
    episode_time_s: float = 0.0,
    park_hold_s: float = GRIPPER_PARK_HOLD_S,
    park_max_joint_speed_deg_s: float = DEFAULT_PARK_MAX_JOINT_SPEED_DEG_S,
    joint_filter: AdaptiveJointFilter | None = None,
    home_standby: GripperHomeStandby | None = None,
    dataset_writer: AsyncLeRobotWriter | None = None,
    cameras: list[Any] | None = None,
    camera_names: list[str] | None = None,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_stale_timeout_s: float = CAMERA_STALE_TIMEOUT_S,
    episode_number: int = 1,
    episode_total: str = "?",
    audio_recorder: PicoAudioRecorder | None = None,
    dashboard: _RecordingDashboard | None = None,
) -> tuple[np.ndarray, np.ndarray, int, str, np.ndarray]:
    resolved_control_fps = float(fps if control_fps is None else control_fps)
    loop_timer = TeleopLoopTimer(resolved_control_fps)
    capture_interval_s = 1.0 / float(fps)
    next_capture_s: float | None = None
    n_frames = 0
    phase = _EpisodePhase.READY
    recording_started_s: float | None = None
    episode_start_ns: int | None = None
    status = "interrupted"
    del initial_start_sides
    tracking_recovery = TrackingRecoveryPolicy()
    sync_lag_ns = int(sync_lag_s * 1e9)
    q = controller.q.copy()
    if joint_filter is None:
        finger_indices = {
            finger.index
            for fingers in (runtime.finger_joints or {}).values()
            for finger in fingers
        }
        indices = tuple(
            dict.fromkeys(
                index
                for side in enabled_sides
                for index in runtime.arm_joint_indices(side)
                if index not in finger_indices
            )
        )
        joint_filter = TeleopMotionConfig(
            input_rate_hz=resolved_control_fps
        ).make_joint_filter(filtered_indices=indices)
    joint_filter.reset(q)
    teleop_session = TeleopSession(controller, joint_filter)
    if home_standby is None:
        home_standby = GripperHomeStandby(initial_standby=True)
    joint_diagnostics = JointMotionDiagnostics(
        runtime.joint_names,
        tuple(joint_filter.filtered_indices or ()),
    )
    last_processed_tracking_time_ns: int | None = None
    timing_next_log_s = time.perf_counter() + 5.0
    timing_ik_total_s = 0.0
    timing_ik_max_s = 0.0
    timing_ik_samples = 0
    timing_tracking_age_max_s = 0.0
    timing_discarded_ik = 0
    timing_playback_counts = (0, 0, 0)
    cameras = cameras or []
    camera_names = camera_names or []
    capture_worker = AsyncEpisodeCapture(
        real_env=real_env,
        runtime=runtime,
        dataset_writer=dataset_writer,
        cameras=cameras,
        camera_names=camera_names,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_stale_timeout_s=camera_stale_timeout_s,
        grippers=grippers,
        gripper_stale_timeout_s=gripper_stale_timeout_s,
        max_sync_skew_s=max_sync_skew_s,
        sensor_loss_timeout_s=sensor_loss_timeout_s,
        task=task,
        audio_recorder=audio_recorder,
    )
    timing_window_start_s = time.perf_counter()
    timing_control_max_s = 0.0

    while True:
        loop_start, _ = loop_timer.tick()
        record_time_ns = time.monotonic_ns()

        try:
            capture_worker.raise_if_failed()
        except SensorHealthError as exc:
            status = "sensor_unhealthy"
            record_log.error("%s; discarding this attempt.", exc)
            break
        except (DatasetWriteError, StreamingEncodingError) as exc:
            status = "dataset_unhealthy"
            record_log.error(
                "Asynchronous capture failed; discarding this attempt: %s", exc
            )
            break

        if stop_event.is_set():
            status = "interrupted"
            break

        tracking_snapshot = tracking_sampler.latest()
        if tracking_snapshot is None:
            immediate_widths = _latest_widths(grippers)
            if dashboard is not None:
                dashboard.update_grippers(
                    immediate_widths.left_mm, immediate_widths.right_mm
                )
            command_stream.update_openings(_gripper_openings(immediate_widths))
            space_listener.consume_space()
            clap_detector.update_sides(
                immediate_widths.left_mm, immediate_widths.right_mm, loop_start
            )
            clap_arbiter.reset()
            loop_timer.sleep(loop_start)
            continue
        sample = tracking_snapshot.sample
        tracking_age_s = tracking_snapshot.age_s(loop_start)
        timing_tracking_age_max_s = max(timing_tracking_age_max_s, tracking_age_s)
        tracking_stale = tracking_age_s > tracking_stale_ms / 1000.0
        side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
        tracking_ok = not tracking_stale and _enabled_tracking_ok(
            side_tracked, enabled_sides
        )

        if not controller.active and not tracking_ok:
            tracking_recovery.reset()
            immediate_widths = _latest_widths(grippers)
            if dashboard is not None:
                dashboard.update_grippers(
                    immediate_widths.left_mm, immediate_widths.right_mm
                )
            command_stream.update_openings(_gripper_openings(immediate_widths))
            space_listener.consume_space()
            clap_detector.update_sides(
                immediate_widths.left_mm, immediate_widths.right_mm, loop_start
            )
            clap_arbiter.reset()
            loop_timer.sleep(loop_start)
            continue

        if not tracking_ok:
            immediate_widths = _latest_widths(grippers)
            if dashboard is not None:
                dashboard.update_grippers(
                    immediate_widths.left_mm, immediate_widths.right_mm
                )
            if tracking_recovery.note_missing(
                loop_start,
                observed_since=(
                    tracking_snapshot.fresh_at_s if tracking_stale else None
                ),
            ):
                command_stream.stop()
                held = real_env.hold(q)
                controller.tracking_lost(held)
                home_standby.enter_standby(enabled_sides)
                joint_filter.reset(held)
                q = held
                command_stream.submit(
                    held,
                    _gripper_openings(immediate_widths),
                    time_s=loop_start,
                    active=True,
                    new_epoch=True,
                )
                record_log.warning(
                    "Tracking lost%s; robot command held and this attempt will reset.",
                    (
                        f" (sample age {tracking_age_s * 1000.0:.0f} ms)"
                        if tracking_stale
                        else ""
                    ),
                )
                log_say("tracking lost", play_sounds=play_sounds)
            if (
                tracking_recovery.lost
                and tracking_recovery.lost_for(loop_start) >= tracking_loss_timeout_s
            ):
                status = "tracking_lost"
                break
            recover = getattr(tracker, "recover", None)
            if callable(recover) and tracking_recovery.should_recover(loop_start):
                acquired, recovered = tracking_sampler.try_source_call(recover)
                if acquired and recovered:
                    record_log.info(
                        "Tracking recovered; open the HandUMI gripper to re-anchor."
                    )
                    log_say("tracking recovered", play_sounds=play_sounds)
            command_stream.update_openings(_gripper_openings(immediate_widths))
            loop_timer.sleep(loop_start)
            continue
        if tracking_recovery.lost:
            record_log.info("Tracking stream recovered; checking gripper state.")
        tracking_recovery.reset()

        immediate_widths = _latest_widths(grippers)
        if dashboard is not None:
            dashboard.update_grippers(
                immediate_widths.left_mm, immediate_widths.right_mm
            )
        start_sides: tuple[str, ...] = ()
        if space_listener.consume_space():
            if phase is _EpisodePhase.READY:
                # With no HandUMI grippers, Space is the explicit arm-start
                # fallback.  Otherwise it starts recording only; opening each
                # gripper remains responsible for waking its parked arm.
                if grippers is None:
                    start_sides = controller.idle_sides()
                phase = _EpisodePhase.RECORDING
                recording_started_s = loop_start
                next_capture_s = loop_start
                episode_start_ns = record_time_ns
                if audio_recorder is not None:
                    audio_recorder.begin_episode(episode_start_ns)
                # Space may follow a partial squeeze made while waiting.  Do
                # not let it turn into an episode action after this boundary.
                clap_detector.reset()
                clap_arbiter.reset()
                home_standby.reset_close_timers()
                command_stream.clear_joint_rate_limits(
                    tuple(joint_filter.filtered_indices or ())
                )
                record_log.info("Recording episode started from home with Space.")
                if dashboard is not None:
                    dashboard.recording()
                log_say("recording episode", play_sounds=play_sounds)
        clap_gesture = clap_arbiter.update(
            clap_detector,
            immediate_widths.left_mm,
            immediate_widths.right_mm,
            loop_start,
        )
        if clap_detector.last_clap_edges:
            record_log.info(
                "Clap edge registered: %s (left=%.1f mm, right=%.1f mm).",
                "/".join(clap_detector.last_clap_edges),
                immediate_widths.left_mm,
                immediate_widths.right_mm,
            )
        gesture_action = _episode_gesture_action(
            clap_gesture, recording=phase is _EpisodePhase.RECORDING
        )
        if clap_gesture is not None:
            record_log.info(
                "Resolved double clap: %s%s (left=%.1f mm, right=%.1f mm).",
                clap_gesture,
                " (episode ready)" if phase is _EpisodePhase.READY else "",
                immediate_widths.left_mm,
                immediate_widths.right_mm,
            )
        if gesture_action == "start":
            phase = _EpisodePhase.RECORDING
            recording_started_s = loop_start
            next_capture_s = loop_start
            episode_start_ns = record_time_ns
            if audio_recorder is not None:
                audio_recorder.begin_episode(episode_start_ns)
            # A completed start gesture and any partial gesture on the other
            # side belong to READY, never to the new episode.  Both hands must
            # reopen before a save/reset gesture can begin.
            clap_detector.reset()
            clap_arbiter.reset()
            # The episode gesture must not wake an arm.  The triggering right
            # gripper normally ends closed, so it stays at home until the
            # operator reopens it; other open grippers wake via standby.update.
            home_standby.reset_close_timers()
            command_stream.clear_joint_rate_limits(
                tuple(joint_filter.filtered_indices or ())
            )
            record_log.info(
                "● REC | episode %d/%s | started from home by double-squeeze "
                "RIGHT | 0 frames.",
                episode_number,
                episode_total,
            )
            if dashboard is not None:
                dashboard.recording()
            log_say("recording episode", play_sounds=play_sounds)
        elif gesture_action == "save":
            status = "saved"
            break
        elif gesture_action == "reset":
            status = "reset"
            break
        elif gesture_action == "finish":
            status = "session_finished"
            break

        if (
            phase is _EpisodePhase.RECORDING
            and episode_time_s > 0.0
            and recording_started_s is not None
            and loop_start - recording_started_s >= episode_time_s
        ):
            status = "saved"
            record_log.info(
                "Episode %d reached %.1f s; saving automatically.",
                episode_number,
                episode_time_s,
            )
            break

        inputs = teleop_session.inputs(sample, immediate_widths)
        if phase is _EpisodePhase.READY:
            # READY is deliberately not teleoperation: before RIGHT starts the
            # attempt there is no feedback read, IK, anchoring, or capture.
            # The home stream may mirror gripper openings, but both arms stay
            # fixed and inactive.
            command_stream.update_openings(inputs.openings)
            real_env.check_health()
            loop_timer.sleep(loop_start)
            continue

        # During recording, the physical HandUMI gripper is the single source
        # for both close-to-park and reopen-to-wake transitions.  This avoids
        # stale robot feedback parking an arm while the operator holds it open.
        if grippers is not None:
            park_sides, wake_sides = home_standby.update(
                inputs.openings,
                loop_start,
                enabled_sides,
            )
        else:
            park_sides, wake_sides = (), ()
        if park_sides:
            parked = controller.park(park_sides)
            if parked:
                park_indices = tuple(
                    index
                    for index in joint_filter.filtered_indices or ()
                    if any(index in controller.side_indices[side] for side in parked)
                )
                command_stream.limit_joint_rates(
                    park_indices,
                    np.deg2rad(park_max_joint_speed_deg_s),
                )
                record_log.info(
                    "HandUMI %s gripper remained fully closed for %.1fs; "
                    "arm returning home "
                    "and entering standby while recording continues.",
                    "/".join(parked),
                    park_hold_s,
                )
                for side in parked:
                    log_say(f"{side} arm standby", play_sounds=play_sounds)
        if wake_sides:
            wake_indices = tuple(
                index
                for index in joint_filter.filtered_indices or ()
                if any(index in controller.side_indices[side] for side in wake_sides)
            )
            command_stream.clear_joint_rate_limits(wake_indices)
            start_sides += tuple(
                side for side in wake_sides if side not in start_sides
            )
            record_log.info(
                "HandUMI %s gripper open; waking and re-anchoring arm.",
                "/".join(wake_sides),
            )
        fresh_tracking = (
            last_processed_tracking_time_ns is None
            or tracking_snapshot.source_time_ns != last_processed_tracking_time_ns
        )
        if not fresh_tracking and not start_sides and not park_sides:
            command_stream.update_openings(inputs.openings)
            real_env.check_health()
            loop_timer.sleep(loop_start)
            continue

        capture_this_step = bool(
            next_capture_s is None or loop_start >= next_capture_s
        )
        capture_base_q = q.copy()
        controller_q_before_ik = controller.q.copy()
        filter_before_ik = teleop_session.snapshot_filter()
        set_solver_timestep = getattr(controller.solver, "set_timestep", None)
        if callable(set_solver_timestep):
            source_dt_s = (
                1.0 / resolved_control_fps
                if last_processed_tracking_time_ns is None
                else (
                    tracking_snapshot.source_time_ns
                    - last_processed_tracking_time_ns
                )
                / 1e9
            )
            set_solver_timestep(source_dt_s)
        ik_start_s = time.perf_counter()
        teleop_frame = teleop_session.advance(
            inputs,
            now_s=loop_start,
            start_sides=start_sides,
            exact_sides=park_sides,
        )
        ik_elapsed_s = time.perf_counter() - ik_start_s
        timing_ik_total_s += ik_elapsed_s
        timing_ik_max_s = max(timing_ik_max_s, ik_elapsed_s)
        timing_ik_samples += 1
        newest_after_ik = tracking_sampler.latest()
        if (
            newest_after_ik is not None
            and newest_after_ik.source_time_ns != tracking_snapshot.source_time_ns
            and not start_sides
            and not park_sides
        ):
            controller.q = controller_q_before_ik
            teleop_session.restore_filter(filter_before_ik)
            timing_discarded_ik += 1
            real_env.check_health()
            continue
        anchored = teleop_frame.anchored_sides
        if anchored:
            record_log.info(
                "Teleop arm anchored from home: %s.", "/".join(anchored)
            )

        action_q = teleop_frame.q
        joint_diagnostics.observe(teleop_frame.step.q, action_q)
        command_stream.submit(
            action_q,
            teleop_frame.inputs.openings,
            time_s=tracking_snapshot.fresh_at_s,
            active=controller.active or bool(park_sides),
            new_epoch=bool(anchored),
        )
        # Grippers remain live even when every arm is parked. Arm activity
        # only controls IK motion, never gripper communication.
        command_stream.update_openings(teleop_frame.inputs.openings)
        last_processed_tracking_time_ns = tracking_snapshot.source_time_ns
        real_env.check_health()
        q = action_q

        timing_now_s = time.perf_counter()
        timing_control_max_s = max(timing_control_max_s, timing_now_s - loop_start)
        if command_stream.running and timing_now_s >= timing_next_log_s:
            output_stats = command_stream.stats()
            playback_counts = (
                output_stats.interpolated_commands,
                output_stats.extrapolated_commands,
                output_stats.held_commands,
            )
            playback_window = tuple(
                max(0, current - previous)
                for current, previous in zip(
                    playback_counts,
                    timing_playback_counts,
                    strict=True,
                )
            )
            # Wall-clock time since the episode started (real time elapsed),
            # as opposed to the effective data time below, which is derived
            # from frames actually captured and can fall behind if frames are
            # ever missed or discarded.
            episode_elapsed_s = max(
                0,
                int(timing_now_s - (recording_started_s or timing_now_s)),
            )
            n_frames = capture_worker.captured_frames
            episode_data_s = max(0, int(n_frames / fps))
            queued_frames = (
                dataset_writer.pending_frames if dataset_writer is not None else 0
            )
            capture_timing = capture_worker.timing_stats(reset=True)
            writer_timing = (
                dataset_writer.timing_stats(reset=True)
                if dataset_writer is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            record_log.info(
                "%s | episode %d/%s | elapsed %02d:%02d | data %02d:%02d | %d frames | "
                "writer_queue=%d | output=%.1f Hz, missed=%d, "
                "tracking_age_max=%.0f ms, control_max=%.1f ms, "
                "IK_rate=%.1f Hz, IK_avg/max=%.1f/%.1f ms, "
                "backend_write_max=%.1f ms, output_lateness_max=%.1f ms, "
                "target_age=%.0f ms, playback(i/x/h)=%d/%d/%d, "
                "stale_IK_discarded=%d; %s.",
                "● REC",
                episode_number,
                episode_total,
                episode_elapsed_s // 60,
                episode_elapsed_s % 60,
                episode_data_s // 60,
                episode_data_s % 60,
                n_frames,
                queued_frames,
                output_stats.effective_rate_hz,
                output_stats.missed_deadlines,
                timing_tracking_age_max_s * 1000.0,
                timing_control_max_s * 1000.0,
                timing_ik_samples
                / max(timing_now_s - timing_window_start_s, 1e-6),
                (
                    timing_ik_total_s / timing_ik_samples * 1000.0
                    if timing_ik_samples
                    else 0.0
                ),
                timing_ik_max_s * 1000.0,
                output_stats.max_write_ms,
                output_stats.max_lateness_ms,
                output_stats.latest_target_age_ms,
                *playback_window,
                timing_discarded_ik,
                joint_diagnostics.summary(),
            )
            record_log.info(
                "Recording worker: samples=%d queue=%d, "
                "queue_delay_avg/max=%.1f/%.1f ms, "
                "capture_avg/max=%.1f/%.1f ms, robot_read_avg/max=%.1f/%.1f ms, "
                "cameras_avg/max=%.1f/%.1f ms, gripper_avg/max=%.1f/%.1f ms, "
                "frame_build_avg/max=%.1f/%.1f ms, "
                "audio_deferred_avg/max=%.1f/%.1f ms, "
                "lerobot_add_avg/max=%.1f/%.1f ms, writer_queue=%d.",
                capture_timing.samples,
                capture_worker.pending_captures,
                capture_timing.queue_delay_avg_ms,
                capture_timing.queue_delay_max_ms,
                capture_timing.capture_avg_ms,
                capture_timing.capture_max_ms,
                capture_timing.robot_read_avg_ms,
                capture_timing.robot_read_max_ms,
                capture_timing.cameras_avg_ms,
                capture_timing.cameras_max_ms,
                capture_timing.gripper_avg_ms,
                capture_timing.gripper_max_ms,
                capture_timing.frame_build_avg_ms,
                capture_timing.frame_build_max_ms,
                *writer_timing,
                queued_frames,
            )
            timing_next_log_s = timing_now_s + 5.0
            timing_window_start_s = timing_now_s
            timing_ik_total_s = 0.0
            timing_ik_max_s = 0.0
            timing_ik_samples = 0
            timing_tracking_age_max_s = 0.0
            timing_discarded_ik = 0
            timing_playback_counts = playback_counts
            joint_diagnostics.reset()
            timing_control_max_s = 0.0

        played_command = command_stream.latest()
        if played_command is None:
            played_action_q = action_q
            played_openings = teleop_frame.inputs.openings
        else:
            played_action_q, played_openings = played_command

        if not capture_this_step:
            loop_timer.sleep(loop_start)
            continue
        assert episode_start_ns is not None
        if next_capture_s is None:
            next_capture_s = loop_start
        missed_capture_slots = max(
            0,
            int((loop_start - next_capture_s) // capture_interval_s),
        )
        next_capture_s += (missed_capture_slots + 1) * capture_interval_s
        target_time_ns = max(episode_start_ns, record_time_ns - sync_lag_ns)
        tracking_time_ns = int(sample.aligned_time_ns or sample.pc_monotonic_ns)
        try:
            capture_worker.submit(
                _CaptureSnapshot(
                    base_q=capture_base_q,
                    action_q=np.asarray(played_action_q, dtype=np.float32).copy(),
                    openings=dict(played_openings),
                    target_time_ns=target_time_ns,
                    record_time_ns=record_time_ns,
                    tracking_time_ns=tracking_time_ns,
                    submitted_s=time.perf_counter(),
                )
            )
        except DatasetWriteError as exc:
            status = "dataset_unhealthy"
            record_log.error("%s; discarding this attempt.", exc)
            break
        n_frames = capture_worker.captured_frames
        if dashboard is not None:
            dashboard.update_frames(n_frames)
        loop_timer.sleep(loop_start)

    try:
        states, actions, n_frames = capture_worker.finish()
    except SensorHealthError as exc:
        if status == "saved":
            status = "sensor_unhealthy"
        record_log.error("Asynchronous capture did not finish cleanly: %s", exc)
        size = canonical_joint_layout(runtime).size
        states = np.empty((0, size), dtype=np.float32)
        actions = np.empty((0, size), dtype=np.float32)
        n_frames = capture_worker.captured_frames
    except (DatasetWriteError, StreamingEncodingError) as exc:
        if status == "saved":
            status = "dataset_unhealthy"
        record_log.error("Asynchronous capture did not finish cleanly: %s", exc)
        size = canonical_joint_layout(runtime).size
        states = np.empty((0, size), dtype=np.float32)
        actions = np.empty((0, size), dtype=np.float32)
        n_frames = capture_worker.captured_frames
    finally:
        capture_worker.close()
    return states, actions, n_frames, status, q


def _run_record() -> None:
    # Captured before any hardware/dataset setup so the dashboard's "program
    # runtime" reflects the whole run, not just the recording loop.
    program_start_s = time.perf_counter()
    args = _parse_record_args()
    args.output_dir = Path(args.output_dir)
    args.repo_id = f"local/{args.output_dir.name}"
    if args.resume:
        _validate_resume_dataset(args.output_dir)
        info = json.loads((args.output_dir / "meta" / "info.json").read_text())
        handumi = info.get("handumi") or {}
        audio = handumi.get("audio") if isinstance(handumi, dict) else None
        if not isinstance(audio, dict):
            audio = {}
        dataset_records_audio = bool(audio.get("enabled"))
        if dataset_records_audio and not bool(audio.get("frame_aligned")):
            raise SystemExit(
                "Cannot resume this legacy audio dataset: it has episode-level "
                "WAV files but no frame-aligned audio timing features."
            )
        if (
            args.record_audio is not None
            and bool(args.record_audio) != dataset_records_audio
        ):
            raise SystemExit(
                "Cannot resume with a different audio setting; this dataset has "
                f"audio {'enabled' if dataset_records_audio else 'disabled'}."
            )
        args.record_audio = dataset_records_audio
    elif args.record_audio is None:
        args.record_audio = False
    _validate_record_args(args)
    play_sounds = not args.no_sounds
    stop_event = threading.Event()

    record_log.info("Loading %s robot kinematics.", args.robot)
    runtime = load_embodiment(args.robot)
    import jax

    record_log.info("IK compute platform: %s.", jax.default_backend())
    try:
        home_pose_name, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    enabled_sides = _enabled_sides(args.side)
    controller = TeleopController(
        runtime,
        home_q=home_q,
        enabled_sides=enabled_sides,
        source_world_to_robot_world=_tracking_world_map(args.device),
        translation_scale=args.translation_scale,
    )
    if args.ik_solver == "dls":
        controller.solver = make_real_teleop_dls_solver(
            runtime,
            controller.solver,
            home_q,
            input_rate_hz=float(args.control_fps),
        )
        for side in enabled_sides:
            limits = controller.solver.side_joint_speed_limits_rad_s[side]
            record_log.info(
                "DLS %s joint-speed limits: %.1f..%.1f deg/s.",
                side,
                float(np.rad2deg(np.min(limits))),
                float(np.rad2deg(np.max(limits))),
            )
    record_log.info(
        "Warming %s IK solver at %.1f Hz before touching hardware.",
        args.ik_solver.upper(),
        args.control_fps,
    )
    controller.warmup()
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    real_env = make_real_backend(
        args.robot,
        runtime=runtime,
        rig_config=args.rig_config,
        active_sides=enabled_sides,
    )
    gripper_pair = None
    grippers = None
    tracker_started = False
    tracking_sampler: LatestTrackingSampler | None = None
    cameras: list[Any] = []
    camera_names = [] if args.skip_cameras else list(args.cameras)
    camera_specs: list[dict[str, Any]] = []
    camera_views: LiveCameraViews | None = None
    camera_worker: BestEffortPeriodicWorker | None = None
    dataset = None
    dataset_writer: AsyncLeRobotWriter | None = None
    audio_recorder: PicoAudioRecorder | None = None
    dashboard: _RecordingDashboard | None = None
    dataset_log_handler: logging.FileHandler | None = None
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    motion_config = TeleopMotionConfig.from_args(
        args,
        input_rate_hz=args.control_fps,
    )
    finger_indices = {
        finger.index
        for fingers in (runtime.finger_joints or {}).values()
        for finger in fingers
    }
    motion_joint_indices = tuple(
        dict.fromkeys(
            index
            for side in enabled_sides
            for index in runtime.arm_joint_indices(side)
            if index not in finger_indices
        )
    )
    joint_filter = motion_config.make_joint_filter(
        filtered_indices=motion_joint_indices
    )
    command_stream = motion_config.make_command_stream(real_env.write)

    def _on_signal(signum, frame):
        del signum, frame
        record_log.info("Signal received - discarding active episode and stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    escape_listener = _EscapeStopListener(stop_event)
    escape_listener.start()

    try:
        record_log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        tracking_sampler = LatestTrackingSampler(
            tracker.latest,
            sample_rate_hz=motion_config.input_rate_hz,
        )
        tracking_sampler.start()
        if camera_names:
            camera_ids = resolve_camera_ids(
                None,
                args.rig_config,
                camera_names=camera_names,
            )
            if len(camera_ids) != len(set(camera_ids)):
                mappings = ", ".join(
                    f"{name}={camera_id}"
                    for name, camera_id in zip(
                        camera_names, camera_ids, strict=True
                    )
                )
                raise SystemExit(
                    f"Selected cameras must use distinct devices ({mappings})."
                )
            camera_specs, _ = build_camera_specs(
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
                camera_specs,
                fps=args.cam_fps,
                width=args.cam_width,
                height=args.cam_height,
                zero_non_laptop=False,
            )
        preview_camera_names = [
            name for name in ("left_wrist", "right_wrist") if name in camera_names
        ]
        if cameras and preview_camera_names and not args.no_rerun:
            viewer = OpenCVCameraViewer(preview_camera_names)
            viewer.start()
            camera_views = LiveCameraViews(
                cameras=cameras,
                camera_names=camera_names,
                width=args.cam_width,
                height=args.cam_height,
                viewer=viewer,
            )
            camera_worker = BestEffortPeriodicWorker(
                camera_views.update,
                rate_hz=min(args.cam_fps, 15),
                thread_name="handumi-record-camera-preview",
            )
            camera_worker.start()
        gripper_pair = connect_feetech(args)
        if gripper_pair is not None:
            grippers = FeetechGripperSampler(
                gripper_pair,
                sample_hz=args.feetech_sample_hz,
            )
            grippers.start()

        real_env.setup(repair=not args.skip_can_repair)
        real_env.connect()
        record_log.info("Selected home pose: %s", home_pose_name)
        real_env.home(home_q)
        joint_filter.reset(home_q)
        startup_widths = _latest_widths(grippers)
        command_stream.submit(
            home_q,
            _gripper_openings(startup_widths),
            time_s=time.perf_counter(),
            active=True,
            new_epoch=True,
        )
        record_log.info(
            "Continuous gripper synchronization started; arms remain at home "
            "until their HandUMI gripper opens."
        )
        record_log.info(
            "Control pipeline: %s at %.1f Hz; dataset capture %.1f FPS; "
            "trajectory playback %.1f Hz, %.1f ms delay, "
            "%.0f ms max bridge, %.0f ms EMA; adaptive IK filter "
            "cutoff=%.1f Hz + %.1f*|dq/dt|.",
            args.ik_solver.upper(),
            args.control_fps,
            args.fps,
            args.command_rate_hz,
            args.trajectory_delay_ms,
            args.max_extrapolation_ms,
            args.motion_smoothing_time_constant_s * 1000.0,
            args.joint_filter_min_cutoff_hz,
            args.joint_filter_velocity_coefficient,
        )
        space_listener.start()

        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        layout = canonical_joint_layout(runtime)
        use_videos = bool(camera_names)
        features = build_features(
            camera_names,
            args.cam_width,
            args.cam_height,
            use_videos,
            layout.names,
            camera_specs=camera_specs,
            record_audio=bool(args.record_audio),
        )
        encoder_selection = None
        dataset_vcodec = _SOFTWARE_VIDEO_CODEC
        if use_videos:
            output_sizes = [
                camera_output_size(
                    spec,
                    default_width=args.cam_width,
                    default_height=args.cam_height,
                )
                for spec in camera_specs
            ]
            encoder_selection = _select_video_encoder(
                policy="auto",
                requested_vcodec=None,
                width=max(width for width, _ in output_sizes),
                height=max(height for _, height in output_sizes),
                fps=args.fps,
                camera_count=len(camera_names),
                requested_threads=None,
            )
            dataset_vcodec = encoder_selection.vcodec
            record_log.info(
                "LeRobot video encoder: %s (%s).",
                dataset_vcodec,
                "hardware" if encoder_selection.hardware else "CPU",
            )
        dataset_kwargs = {
            "repo_id": args.repo_id,
            "root": args.output_dir,
            "image_writer_processes": 0,
            "image_writer_threads": 0 if use_videos else max(1, 4 * len(camera_names)),
            "rgb_encoder": RGBEncoderConfig(vcodec=dataset_vcodec) if use_videos else None,
            "streaming_encoding": use_videos,
            "encoder_queue_maxsize": max(1, args.fps),
            "encoder_threads": (
                encoder_selection.threads if encoder_selection is not None else None
            ),
        }
        if args.resume:
            dataset = LeRobotDataset.resume(**dataset_kwargs)
        else:
            dataset = LeRobotDataset.create(
                **dataset_kwargs,
                fps=args.fps,
                robot_type=runtime.config.kind,
                features=features,
                use_videos=use_videos,
            )
        from handumi.dataset.lerobot_compat import preserve_dataset_info

        preserve_dataset_info(dataset)
        if use_videos:
            _install_strict_streaming_encoder(dataset)
        dataset_writer = AsyncLeRobotWriter(
            dataset,
            max_pending_frames=max(2, args.fps * 2),
            use_videos=use_videos,
        )
        if args.record_audio:
            audio_recorder = PicoAudioRecorder(
                lambda: getattr(tracker, "xrt", None),
                Path(dataset.root),
            )
            audio_recorder.start()
        existing_episodes = int(dataset.num_episodes)
        diagnostics_dir = Path(dataset.root) / "logs"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = diagnostics_dir / "teleop_record.log"
        dataset_log_handler = logging.FileHandler(
            diagnostics_path,
            mode="a",
            encoding="utf-8",
        )
        dataset_log_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.getLogger().addHandler(dataset_log_handler)
        record_log.info("Recording LeRobot dataset at: %s", dataset.root)
        record_log.info(
            "Persistent diagnostics: %s | solver=%s control=%.1f Hz "
            "capture=%.1f FPS cameras=%s audio=%s.",
            diagnostics_path,
            args.ik_solver,
            args.control_fps,
            args.fps,
            ",".join(camera_names) or "none",
            bool(args.record_audio),
        )
        record_log.info(
            "Timing targets: control_max < %.1f ms, IK_rate near %.1f Hz, "
            "capture queue_delay < %.1f ms, writer/capture queues normally 0, "
            "stale_IK_discarded and missed deadlines near 0.",
            1000.0 / args.control_fps,
            args.control_fps,
            1000.0 / args.fps,
        )

        start_control = "Double-squeeze RIGHT"
        if args.space_start:
            start_control += " / SPACE"
        episode_controls = [
            (start_control, "Start from home"),
            ("Double-squeeze RIGHT while REC", "Save and return home"),
            ("Double-squeeze LEFT while REC", "Reset same episode and home"),
            (
                "Double-squeeze BOTH",
                "Discard active attempt and end the session",
            ),
            ("Esc / Ctrl+C", "Discard current episode and stop"),
        ]
        if not args.no_park_on_hold:
            episode_controls.append(
                (
                    f"HandUMI gripper closed {args.gripper_park_hold_s:.1f}s",
                    "Park that arm at home",
                )
            )
        if args.episode_time_s > 0.0:
            episode_controls.insert(
                2,
                (
                    f"{args.episode_time_s:.1f}s while REC",
                    "Auto-save and return home",
                ),
            )
        dashboard = _RecordingDashboard(
            task=args.task,
            dataset=str(dataset.root),
            fps=args.fps,
            existing_episodes=existing_episodes,
            existing_frames=int(getattr(dataset, "num_frames", 0)),
            started_at_s=program_start_s,
            controls=tuple(episode_controls),
        )
        dashboard.start()

        clap_detector = DoubleClapDetector()
        clap_arbiter = _BilateralClapArbiter()
        home_standby = GripperHomeStandby(
            hold_s=args.gripper_park_hold_s,
            initial_standby=True,
            park_on_hold=not args.no_park_on_hold,
        )

        def home_for_next_episode(
            episode: int,
            episode_total: str,
            *,
            ready_after: bool = True,
        ) -> None:
            _home_between_episodes(
                real_env=real_env,
                controller=controller,
                home_q=home_q,
                enabled_sides=enabled_sides,
                command_stream=command_stream,
                joint_filter=joint_filter,
                home_standby=home_standby,
                grippers=grippers,
                motion_joint_indices=motion_joint_indices,
                dashboard=dashboard,
                episode=episode,
                episode_total=episode_total,
                play_sounds=play_sounds,
                ready_after=ready_after,
            )

        recorded = 0
        while (
            args.num_episodes <= 0 or recorded < args.num_episodes
        ) and not stop_event.is_set():
            ep_num = existing_episodes + recorded + 1
            ep_total = (
                "inf"
                if args.num_episodes <= 0
                else str(existing_episodes + args.num_episodes)
            )
            if not _wait_for_tracking_sampler(
                tracking_sampler,
                stop_event,
                enabled_sides=enabled_sides,
                tracking_stale_ms=args.tracking_stale_ms,
            ):
                break
            record_log.info("--- Episode %d/%s ---", ep_num, ep_total)
            _log_episode_interface(
                ep_num,
                ep_total,
                waiting=True,
                space_start=args.space_start,
                park_hold_s=args.gripper_park_hold_s,
                park_on_hold=not args.no_park_on_hold,
            )
            dashboard.set_episode(
                ep_num,
                ep_total,
                state="READY",
                detail="Robot at home; double-squeeze RIGHT to start",
            )
            log_say(f"Episode {ep_num} ready", play_sounds=play_sounds)
            if not args.space_start:
                record_log.info(
                    "  Double-squeeze RIGHT to start episode %d from home ...",
                    ep_num,
                )
            else:
                record_log.info(
                    "  Double-squeeze RIGHT or press Space to start episode %d ...",
                    ep_num,
                )
            clap_arbiter.reset()
            clap_detector.reset()
            try:
                _, _, n_frames, status, _ = record_episode(
                    tracker=tracker,
                    tracking_sampler=tracking_sampler,
                    grippers=grippers,
                    real_env=real_env,
                    controller=controller,
                    runtime=runtime,
                    home_q=home_q,
                    enabled_sides=enabled_sides,
                    space_listener=space_listener,
                    clap_detector=clap_detector,
                    clap_arbiter=clap_arbiter,
                    fps=args.fps,
                    episode_time_s=args.episode_time_s,
                    task=args.task,
                    stop_event=stop_event,
                    play_sounds=play_sounds,
                    initial_start_sides=(),
                    sync_lag_s=args.sync_lag_s,
                    max_sync_skew_s=args.max_sync_skew_s,
                    gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                    sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                    tracking_loss_timeout_s=args.tracking_loss_timeout_s,
                    tracking_stale_ms=args.tracking_stale_ms,
                    command_stream=command_stream,
                    control_fps=args.control_fps,
                    park_hold_s=args.gripper_park_hold_s,
                    park_max_joint_speed_deg_s=args.park_max_joint_speed_deg_s,
                    joint_filter=joint_filter,
                    home_standby=home_standby,
                    dataset_writer=dataset_writer,
                    cameras=cameras,
                    camera_names=camera_names,
                    camera_width=args.cam_width,
                    camera_height=args.cam_height,
                    camera_stale_timeout_s=args.camera_stale_timeout_s,
                    episode_number=ep_num,
                    episode_total=ep_total,
                    audio_recorder=audio_recorder,
                    dashboard=dashboard,
                )
            except Exception as exc:
                # A hardware disconnect (cable pull, USB drop, robot power
                # loss, ...) or any other unexpected failure must not lose
                # already-saved episodes: discard only the one in flight and
                # stop the whole session, matching handumi.scripts.record.
                record_log.exception(
                    "Unexpected failure during episode %d; discarding it and stopping.",
                    ep_num,
                )
                dataset_writer.clear_episode()
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                dashboard.discarded(0, f"unexpected error: {exc}")
                log_say("Episode discarded", play_sounds=play_sounds)
                stop_event.set()
                break
            if status == "session_finished":
                record_log.info(
                    "Bilateral double clap detected; finishing recording session."
                )
                dataset_writer.clear_episode()
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                dashboard.discarded(n_frames, "session finished")
                log_say("Recording session finished", play_sounds=play_sounds)
                if controller.active:
                    home_for_next_episode(
                        ep_num,
                        ep_total,
                        ready_after=False,
                    )
                break
            if status == "reset":
                record_log.warning(
                    "Resetting episode %d by left double clap (%d frames removed).",
                    ep_num,
                    n_frames,
                )
                dataset_writer.clear_episode()
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                dashboard.set_episode(
                    ep_num,
                    ep_total,
                    state="RESETTING",
                    detail="Left double-squeeze; returning both arms home",
                )
                log_say(f"Resetting episode {ep_num}", play_sounds=play_sounds)
                home_for_next_episode(ep_num, ep_total)
                continue
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "dataset_unhealthy",
                "encoder_unhealthy",
                "interrupted",
            }:
                record_log.warning(
                    "Episode %d attempt rejected (%s, %d frames); resetting it.",
                    ep_num,
                    status,
                    n_frames,
                )
                dataset_writer.clear_episode()
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                if status == "interrupted":
                    dashboard.discarded(n_frames, status)
                    break
                dashboard.set_episode(
                    ep_num,
                    ep_total,
                    state="RESETTING",
                    detail=f"Attempt rejected ({status}); returning home",
                )
                log_say(f"Resetting episode {ep_num}", play_sounds=play_sounds)
                home_for_next_episode(ep_num, ep_total)
                continue
            if status != "saved":
                raise RuntimeError(
                    f"Unexpected episode terminal state before save: {status}"
                )
            audio_path: Path | None = None
            try:
                # Deferred audio alignment and all LeRobot frame additions must
                # finish before the recorder snapshots its frame-target list.
                dataset_writer.flush()
                if audio_recorder is not None:
                    audio_recorder.prepare_episode(expected_frames=n_frames)
                audio_path = (
                    audio_recorder.commit_episode(
                        existing_episodes + recorded,
                        chunks_size=int(dataset.meta.chunks_size),
                    )
                    if audio_recorder is not None
                    else None
                )
                dataset_writer.save_episode(n_frames)
            except (AudioCaptureError, DatasetWriteError) as exc:
                if audio_path is not None:
                    audio_path.unlink(missing_ok=True)
                record_log.error("Episode discarded before commit: %s", exc)
                dataset_writer.clear_episode()
                if audio_recorder is not None:
                    audio_recorder.cancel_episode()
                dashboard.set_episode(
                    ep_num,
                    ep_total,
                    state="RESETTING",
                    detail="Save failed; returning home and retrying this episode",
                )
                log_say(f"Resetting episode {ep_num}", play_sounds=play_sounds)
                home_for_next_episode(ep_num, ep_total)
                continue
            recorded += 1
            dashboard.saved(n_frames)
            record_log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(
                f"Episode {ep_num} saved, {n_frames} frames",
                play_sounds=play_sounds,
            )
            has_next_episode = (
                args.num_episodes <= 0 or recorded < args.num_episodes
            )
            next_episode = existing_episodes + recorded + 1
            home_for_next_episode(
                next_episode if has_next_episode else ep_num,
                ep_total,
                ready_after=has_next_episode,
            )

        dataset_writer.finalize()
        from handumi.dataset import update_handumi_metadata

        updated_info = update_handumi_metadata(
            dataset.root,
            {
                "recording_device": args.device,
                "ik_solver": args.ik_solver,
                "control_rate_hz": args.control_fps,
                "diagnostics_log": "logs/teleop_record.log",
                "audio": audio_metadata(bool(args.record_audio)),
                "episode_time_s": args.episode_time_s,
                "capture_schema": HANDUMI_CAPTURE_SCHEMA,
                "state_layout": "yaml_arm_joints_plus_logical_gripper_width_m",
                "state_semantics": "real_robot_joint_feedback",
                "action_semantics": "next_step_teleop_joint_command",
                "trajectory_command_rate_hz": args.command_rate_hz,
                "trajectory_delay_ms": args.trajectory_delay_ms,
                "trajectory_max_extrapolation_ms": args.max_extrapolation_ms,
                "trajectory_ema_time_constant_s": (
                    args.motion_smoothing_time_constant_s
                ),
                "joint_filter_min_cutoff_hz": args.joint_filter_min_cutoff_hz,
                "joint_filter_velocity_coefficient": (
                    args.joint_filter_velocity_coefficient
                ),
                "joint_filter_derivative_cutoff_hz": (
                    args.joint_filter_derivative_cutoff_hz
                ),
                "gripper_park_on_hold": not bool(args.no_park_on_hold),
                "gripper_park_hold_s": args.gripper_park_hold_s,
                "park_max_joint_speed_deg_s": args.park_max_joint_speed_deg_s,
                "translation_scale": args.translation_scale,
                "observation_action_alignment": (
                    "observation.state[t] is canonical backend feedback; "
                    "action[t] is the next recorded teleop command."
                ),
                "source_kind_ids": {"converted": 0, "teleop": 1, "unknown": -1},
                "calibration_id_semantics": (
                    "-1 means no per-episode calibration artifact is referenced"
                ),
                "sync_lag_s": args.sync_lag_s,
                "max_sync_skew_s": args.max_sync_skew_s,
                "joint_names": layout.names,
                "target_robot": _robot_metadata(args.robot),
                "repo_id": args.repo_id,
                "video_keys": [
                    f"observation.images.{name}" for name in camera_names
                ],
            },
        )
        preserve_dataset_info(dataset, updated_info)
        _write_dataset_readme(
            Path(dataset.root),
            repo_id=args.repo_id,
            task=args.task,
            license_id="other",
        )
        if existing_episodes + recorded > 0:
            _validate_finalized_lerobot_dataset(Path(dataset.root))
            record_log.info("LeRobot v3 integrity validation passed.")
        else:
            record_log.warning(
                "Session finished without completed episodes; skipping dataset "
                "integrity validation."
            )
        record_log.info(
            "Done. Recorded %d episode(s). Dataset at: %s", recorded, args.output_dir
        )
    finally:
        if dashboard is not None:
            dashboard.stop()
        escape_listener.stop()
        space_listener.close()
        try:
            command_stream.stop()
        finally:
            try:
                if dataset_writer is not None:
                    dataset_writer.close()
                if audio_recorder is not None:
                    audio_recorder.close()
                real_env.disconnect()
            finally:
                if grippers is not None:
                    grippers.stop()
                if gripper_pair is not None:
                    gripper_pair.close()
                if tracker_started:
                    if tracking_sampler is not None:
                        tracking_sampler.stop()
                    tracker.stop()
                if camera_worker is not None:
                    camera_worker.close()
                if camera_views is not None:
                    camera_views.close()
                elif cameras:
                    disconnect_cameras(cameras)
                log_say("Exiting", play_sounds=play_sounds, blocking=True)
                if dataset_log_handler is not None:
                    logging.getLogger().removeHandler(dataset_log_handler)
                    dataset_log_handler.close()


def _existing_episode_count(root: Path) -> int:
    info_path = Path(root) / "meta" / "info.json"
    return int(json.loads(info_path.read_text()).get("total_episodes", 0))


def _validate_resume_dataset(root: Path) -> None:
    """Require a finalized vector dataset before appending teleop episodes."""
    root = Path(root)
    info_path = root / "meta" / "info.json"
    required_paths = (
        root / "meta" / "episodes",
        root / "meta" / "tasks.parquet",
        root / "data",
    )
    if not root.is_dir():
        raise SystemExit(f"Cannot resume: dataset directory does not exist: {root}")
    if not all(path.exists() for path in required_paths):
        raise SystemExit(f"Cannot resume incomplete dataset at {root}.")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot resume: invalid {info_path}: {exc}") from exc
    if not isinstance(info, dict) or "total_episodes" not in info:
        raise SystemExit(f"Cannot resume: {info_path} is not a HandUMI dataset.")


def main() -> None:
    _run_record()


if __name__ == "__main__":
    main()
