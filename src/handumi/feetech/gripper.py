"""采集 Feetech 夹爪编码器并换算开度，支持后台采样和诊断日志。"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.feetech.diagnostics import log_diagnostic

log = logging.getLogger("handumi.record")


_ENCODER_RESOLUTION = 4096
_HALF_TURN = _ENCODER_RESOLUTION // 2


@dataclass(frozen=True)
class GripperWidths:
    left: float
    right: float
    left_mm: float
    right_mm: float
    left_normalized: float
    right_normalized: float
    left_ticks: int
    right_ticks: int

    @classmethod
    def zero(cls) -> GripperWidths:
        """All-zero widths, used when Feetech is skipped or unavailable."""
        return cls(
            left=0.0,
            right=0.0,
            left_mm=0.0,
            right_mm=0.0,
            left_normalized=0.0,
            right_normalized=0.0,
            left_ticks=0,
            right_ticks=0,
        )


@dataclass(frozen=True)
class GripperSample:
    """One aperture sample timestamped on the workstation monotonic clock."""

    widths: GripperWidths
    sample_time_ns: int
    sequence: int
    enabled: bool = True


class FeetechGripperSampler:
    """Continuously sample both encoders and retain a short native-rate buffer."""

    def __init__(
        self,
        grippers: FeetechGripperPair,
        *,
        sample_hz: float = 100.0,
        buffer_seconds: float = 1.0,
        reconnect_after_errors: int = 2,
    ) -> None:
        if sample_hz <= 0:
            raise ValueError("sample_hz must be greater than zero.")
        if reconnect_after_errors <= 0:
            raise ValueError("reconnect_after_errors must be greater than zero.")
        self.grippers = grippers
        self.sample_hz = float(sample_hz)
        self.reconnect_after_errors = int(reconnect_after_errors)
        self._samples: deque[GripperSample] = deque(
            maxlen=max(8, int(round(sample_hz * buffer_seconds)))
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._last_error: str | None = None
        self._consecutive_errors = 0
        self._total_errors = 0
        self._reconnect_count = 0

    def start(self, *, timeout_s: float = 2.0) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="handumi_feetech_sampler",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return
            time.sleep(0.01)
        error = self.last_error or "no encoder sample received"
        self.stop()
        raise RuntimeError(f"Feetech sampler failed to start: {error}")

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def latest(self) -> GripperSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def sample_at(self, target_time_ns: int | None = None) -> GripperSample | None:
        with self._lock:
            samples = tuple(self._samples)
        if not samples:
            return None
        if target_time_ns is None:
            return samples[-1]
        return min(
            samples, key=lambda sample: abs(sample.sample_time_ns - target_time_ns)
        )

    def samples(self) -> tuple[GripperSample, ...]:
        """Return a stable snapshot of the native-rate sample history."""
        with self._lock:
            return tuple(self._samples)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def consecutive_errors(self) -> int:
        with self._lock:
            return self._consecutive_errors

    @property
    def reconnect_count(self) -> int:
        with self._lock:
            return self._reconnect_count

    @property
    def total_errors(self) -> int:
        with self._lock:
            return self._total_errors

    def _run(self) -> None:
        interval_s = 1.0 / self.sample_hz
        next_sample = time.perf_counter()
        while not self._stop.is_set():
            started_ns = time.monotonic_ns()
            try:
                read_fast = getattr(
                    self.grippers,
                    "read_normalized_widths_fast",
                    self.grippers.read_normalized_widths,
                )
                widths = read_fast()
                finished_ns = time.monotonic_ns()
                self._sequence += 1
                sample = GripperSample(
                    widths=widths,
                    sample_time_ns=(started_ns + finished_ns) // 2,
                    sequence=self._sequence,
                )
                with self._lock:
                    recovered_after = self._consecutive_errors
                    self._samples.append(sample)
                    self._last_error = None
                    self._consecutive_errors = 0
                if recovered_after:
                    log.info(
                        "Feetech sampling recovered after %d failed read(s).",
                        recovered_after,
                    )
            except Exception as exc:  # noqa: BLE001 - health gate owns recovery.
                with self._lock:
                    self._last_error = str(exc)
                    self._consecutive_errors += 1
                    self._total_errors += 1
                    count = self._consecutive_errors
                if count == 1:
                    log.warning("Feetech sampling failed: %s", exc)
                if count % self.reconnect_after_errors == 0:
                    reconnect = getattr(self.grippers, "reconnect", None)
                    if callable(reconnect):
                        try:
                            reconnect()
                        except Exception as reconnect_exc:  # noqa: BLE001
                            with self._lock:
                                self._last_error = (
                                    f"{exc}; reconnect failed: {reconnect_exc}"
                                )
                            log.warning("Feetech reconnect failed: %s", reconnect_exc)
                        else:
                            with self._lock:
                                self._reconnect_count += 1
                            log.info(
                                "Feetech buses reopened after %d consecutive read errors.",
                                count,
                            )

            next_sample += interval_s
            delay = next_sample - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_sample = time.perf_counter()


def zero_gripper_widths() -> GripperWidths:
    """Backend-neutral zero widths (thin wrapper over :meth:`GripperWidths.zero`)."""
    return GripperWidths.zero()


class _EncoderUnwrapper:
    """Turn raw 0-4095 Feetech readings into a continuous tick stream.

    The servo reports ``Present_Position`` modulo 4096, so a gripper whose range
    crosses the 0/4095 seam (like the right HandUMI gripper) makes the raw value
    jump a full revolution between consecutive frames. We sample fast enough that
    real motion never exceeds half a turn per frame, so any jump larger than that
    is a wraparound we cancel by accumulating turns.

    The first frame is trusted as-is (``turns == 0``) rather than guessed from the
    calibration: any guess is ambiguous when the range hugs the seam, and a wrong
    guess latches the whole stream onto the wrong revolution. Start a recording
    with the grippers roughly closed (away from the seam) and tracking is exact.
    """

    def __init__(self) -> None:
        self._prev_raw: int | None = None
        self._turns = 0

    def __call__(self, raw: int) -> int:
        if self._prev_raw is not None:
            delta = raw - self._prev_raw
            if delta > _HALF_TURN:
                self._turns -= 1
            elif delta < -_HALF_TURN:
                self._turns += 1
        self._prev_raw = raw
        return raw + self._turns * _ENCODER_RESOLUTION


class FeetechGripperPair:
    def __init__(
        self, config: FeetechConfig, *, active_sides: tuple[str, ...] = ("left", "right")
    ) -> None:
        from handumi.config import dataset_active_sides

        self.config = config
        self.active_sides = dataset_active_sides({"active_sides": active_sides})
        self._ports = {side: _side_port(config, getattr(config, side)) for side in self.active_sides}
        self._buses: dict[str, FeetechBus] = {}
        for port in set(self._ports.values()):
            self._buses[port] = FeetechBus(
                port=port,
                baudrate=config.baudrate,
                protocol_version=config.protocol_version,
            )
        self._left_port = self._ports.get("left")
        self._right_port = self._ports.get("right")
        self._left_unwrap = _EncoderUnwrapper()
        self._right_unwrap = _EncoderUnwrapper()

    def open(self) -> None:
        opened: list[FeetechBus] = []
        try:
            for bus in self._buses.values():
                bus.open()
                opened.append(bus)
        except BaseException:
            for bus in reversed(opened):
                bus.close()
            raise

    def close(self) -> None:
        for bus in self._buses.values():
            bus.close()

    def reconnect(self) -> None:
        """Clear and reopen both adapters after an SDK/USB communication stall.

        A Feetech ping performs two serial transactions and proved less reliable
        than the position read itself on the HandUMI adapters.  The sampler's
        next real position read is therefore the reconnect health check.
        """
        for bus in self._buses.values():
            try:
                bus.reset_io()
            except RuntimeError:
                pass
        self.close()
        time.sleep(0.02)
        self.open()

    def __enter__(self) -> FeetechGripperPair:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_normalized_widths(self) -> GripperWidths:
        return self._read_normalized_widths(retries=4, retry_delay_s=0.05)

    def read_normalized_widths_fast(self) -> GripperWidths:
        """One transaction per side for latency-sensitive background sampling."""
        return self._read_normalized_widths(retries=0, retry_delay_s=0.0)

    def _read_normalized_widths(
        self, *, retries: int, retry_delay_s: float
    ) -> GripperWidths:
        widths = {side: {"width_m": 0.0, "width_mm": 0.0, "normalized": 0.0, "ticks": 0}
                  for side in ("left", "right")}
        for side in self.active_sides:
            widths[side] = _read_width(
                self._buses[self._ports[side]], getattr(self.config, side),
                getattr(self, f"_{side}_unwrap"), retries=retries,
                retry_delay_s=retry_delay_s, side=side,
            )
        left, right = widths["left"], widths["right"]
        return GripperWidths(
            left=left["width_m"],
            right=right["width_m"],
            left_mm=left["width_mm"],
            right_mm=right["width_mm"],
            left_normalized=left["normalized"],
            right_normalized=right["normalized"],
            left_ticks=int(left["ticks"]),
            right_ticks=int(right["ticks"]),
        )


def _read_width(
    bus: FeetechBus,
    calibration: GripperCalibration,
    unwrap: _EncoderUnwrapper,
    *,
    retries: int = 4,
    retry_delay_s: float = 0.05,
    side: str | None = None,
) -> dict[str, float | int]:
    """读取单侧位置并换算开度，诊断时记录展开及限幅前后的数值。"""
    # 分别保留原始位置与展开位置，以辨认跨圈和瞬时跳变。
    raw_ticks = bus.read_position(
        calibration.servo_id, retries=retries, retry_delay_s=retry_delay_s,
    )
    ticks = unwrap(raw_ticks)
    normalized = calibration.normalized_width(ticks)
    width_mm = calibration.width_mm(ticks)
    log_diagnostic(
        "width_conversion", side=side, port=getattr(bus, "port", calibration.port),
        servo_id=calibration.servo_id, raw_ticks=raw_ticks, ticks=ticks,
        closed_ticks=calibration.closed_ticks, open_ticks=calibration.open_ticks,
        max_width_mm=calibration.max_width_mm,
        unclamped_normalized=(ticks - calibration.closed_ticks)
        / (calibration.open_ticks - calibration.closed_ticks),
        normalized=normalized, width_mm=width_mm,
    )
    return {
        "ticks": ticks,
        "normalized": normalized,
        "width_mm": width_mm,
        "width_m": width_mm / 1000.0,
    }


def _side_port(config: FeetechConfig, calibration: GripperCalibration) -> str:
    port = calibration.port or config.port
    if not port:
        raise ValueError(
            "Feetech port is not configured. Set a shared `port` or per-side `left.port` / `right.port`."
        )
    return port
