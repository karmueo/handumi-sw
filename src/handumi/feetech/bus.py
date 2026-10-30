"""提供夹爪 Feetech 编码器底层通信及可选通信诊断日志。"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from types import MethodType
from typing import Any

from handumi.feetech.diagnostics import log_diagnostic


class FeetechUnavailableError(RuntimeError):
    """Raised when the Feetech SDK is not available in the active environment."""


@dataclass
class FeetechBus:
    port: str
    baudrate: int = 1_000_000
    protocol_version: int = 0

    def __post_init__(self) -> None:
        self._sdk: Any | None = None
        self._port_handler: Any | None = None
        self._packet_handler: Any | None = None

    def open(self) -> None:
        try:
            import scservo_sdk as sdk
        except ImportError as exc:
            raise FeetechUnavailableError(
                "Feetech SDK is not importable. Install/sync lerobot[feetech] "
                "or feetech-servo-sdk in this environment."
            ) from exc

        port_handler = sdk.PortHandler(self.port)
        _install_reliable_packet_timeout(port_handler)
        # PortHandler.openPort() already delegates to setBaudRate() and the
        # latter reopens the serial device. Calling both toggles every adapter
        # twice, adds substantial reconnect latency, and can reset USB boards
        # through DTR. Configure and open exactly once.
        if not port_handler.setBaudRate(self.baudrate):
            port_handler.closePort()
            raise RuntimeError(
                f"Could not open Feetech port {self.port} at "
                f"{self.baudrate} baud."
            )

        self._sdk = sdk
        self._port_handler = port_handler
        self._packet_handler = sdk.PacketHandler(self.protocol_version)

    def close(self) -> None:
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._sdk = None
        self._port_handler = None
        self._packet_handler = None

    def reset_io(self) -> None:
        """Release a wedged SDK transaction and discard partial reply bytes."""
        port = self._require_port()
        clear = getattr(port, "clearPort", None)
        if callable(clear):
            clear()
        if hasattr(port, "is_using"):
            port.is_using = False

    def __enter__(self) -> FeetechBus:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ping(self, servo_id: int) -> bool:
        packet = self._require_packet()
        try:
            _, comm, error = packet.ping(self._require_port(), int(servo_id))
        except _RETRYABLE_SDK_EXCEPTIONS:
            return False
        return _comm_success(comm, self._sdk) and _no_error(error)

    def ping_model(self, servo_id: int, *, retries: int = 2) -> int | None:
        packet = self._require_packet()
        port = self._require_port()
        for attempt in range(retries + 1):
            try:
                model_number, comm, error = packet.ping(port, int(servo_id))
            except _RETRYABLE_SDK_EXCEPTIONS:
                self.reset_io()
            else:
                if _comm_success(comm, self._sdk) and _no_error(error):
                    return int(model_number)
            if attempt < retries:
                self.reset_io()
        return None

    def scan(self, ids: Iterable[int]) -> list[int]:
        return [int(servo_id) for servo_id in ids if self.ping(int(servo_id))]

    def read_position(
        self, servo_id: int, *, retries: int = 4, retry_delay_s: float = 0.05
    ) -> int:
        packet = self._require_packet()
        port = self._require_port()
        last_error = "no response"
        for attempt in range(retries + 1):
            try:
                value, comm, error = packet.read2ByteTxRx(
                    port, int(servo_id), _PRESENT_POSITION_ADDR
                )
            except _RETRYABLE_SDK_EXCEPTIONS as exc:
                # The SCServo SDK indexes the reply buffer before checking the
                # comm result, so a truncated response (e.g. the servo is still
                # busy right after an EEPROM middle-position write) crashes with
                # IndexError instead of reporting a clean failure. PySerial can
                # also raise OSError/SerialException when the port reports ready
                # but returns no bytes. Treat both as retryable bus I/O failures.
                # These exceptional exits can bypass the SDK's normal release
                # of its per-port transaction flag and make later calls fail as
                # if another operation still owned the port.
                if hasattr(port, "is_using"):
                    port.is_using = False
                last_error = _format_sdk_exception(exc)
                log_diagnostic(
                    "position_read", port=self.port, servo_id=servo_id,
                    attempt=attempt + 1, raw_ticks=None, comm=None, error=None,
                    success=False, exception=last_error,
                )
            else:
                # 通信成功时也保留原始值，以识别被 SDK 接受的异常位置。
                success = _comm_success(comm, self._sdk) and _no_error(error)
                log_diagnostic(
                    "position_read", port=self.port, servo_id=servo_id,
                    attempt=attempt + 1, raw_ticks=value, comm=comm, error=error,
                    success=success,
                )
                if success:
                    return int(value)
                last_error = f"comm={comm}, error={error}"
            if attempt < retries:
                time.sleep(retry_delay_s)
        raise RuntimeError(
            f"Failed to read Present_Position from servo {servo_id} after "
            f"{retries + 1} attempts ({last_error})."
        )

    def write_servo_id(self, old_id: int, new_id: int) -> None:
        self.disable_torque(old_id)
        self._write_1_byte(old_id, _ID_ADDR, new_id, "ID")

    def disable_torque(self, servo_id: int) -> None:
        self._write_1_byte(servo_id, _TORQUE_ENABLE_ADDR, 0, "Torque_Enable")
        try:
            self._write_1_byte(servo_id, _LOCK_ADDR, 0, "Lock")
        except RuntimeError:
            pass

    def set_middle_position(self, servo_id: int) -> bool:
        """Re-home the servo so its current shaft angle reads ~2048 (centre).

        Feetech STS/SMS servos treat a write of 128 to ``Torque_Enable`` as a
        "middle position calibration": the controller stores a position
        correction in EEPROM so that ``Present_Position`` reports 2048 at the
        current physical position. We unlock EEPROM first and re-lock after so
        the correction persists across power cycles.

        Some firmware does not ACK the middle-calibration write (the servo is
        busy recalibrating), which surfaces as a write error even though the
        EEPROM update went through. The writes are therefore best-effort: the
        caller must confirm success by reading the position back. Returns False
        if any write was rejected.
        """
        ok = True
        try:
            self._write_1_byte(servo_id, _LOCK_ADDR, 0, "Lock(unlock)")  # unlock EEPROM
        except RuntimeError:
            ok = False
        try:
            self._write_1_byte(
                servo_id, _TORQUE_ENABLE_ADDR, _MIDDLE_CALIBRATION, "Torque_Enable(middle)"
            )
        except RuntimeError:
            ok = False
        time.sleep(0.2)  # let the EEPROM write commit before re-locking / reading back
        try:
            self._write_1_byte(servo_id, _LOCK_ADDR, 1, "Lock(relock)")  # re-lock EEPROM
        except RuntimeError:
            ok = False
        return ok

    def _write_1_byte(
        self,
        servo_id: int,
        address: int,
        value: int,
        name: str,
        *,
        retries: int = 4,
        retry_delay_s: float = 0.05,
    ) -> None:
        packet = self._require_packet()
        port = self._require_port()
        last_error = "no response"
        for attempt in range(retries + 1):
            try:
                comm, error = packet.write1ByteTxRx(
                    port,
                    int(servo_id),
                    int(address),
                    int(value),
                )
            except _RETRYABLE_SDK_EXCEPTIONS as exc:
                last_error = _format_sdk_exception(exc)
            else:
                if _comm_success(comm, self._sdk) and _no_error(error):
                    return
                last_error = f"comm={comm}, error={error}"
            if attempt < retries:
                time.sleep(retry_delay_s)
        raise RuntimeError(
            f"Failed to write {name}={value} on servo {servo_id} after "
            f"{retries + 1} attempts ({last_error})."
        )

    def _require_packet(self):
        if self._packet_handler is None:
            raise RuntimeError("Feetech bus is not open.")
        return self._packet_handler

    def _require_port(self):
        if self._port_handler is None:
            raise RuntimeError("Feetech bus is not open.")
        return self._port_handler


def _comm_success(result: Any, sdk: Any | None) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, tuple):
        if len(result) >= 2:
            comm_result = result[-2]
        elif result:
            comm_result = result[-1]
        else:
            return False
    else:
        comm_result = result
    success = getattr(sdk, "COMM_SUCCESS", 0)
    return comm_result == success or comm_result == 0


def _no_error(error: Any) -> bool:
    return int(error) == 0


def _format_sdk_exception(exc: BaseException) -> str:
    if isinstance(exc, IndexError):
        return "truncated response"
    return f"{exc.__class__.__name__}: {exc}"


def _install_reliable_packet_timeout(port: Any) -> None:
    """Apply LeRobot's Feetech SDK timeout workaround to one port handler.

    The PyPI SDK has shipped several incompatible timeout calculations.  A
    50 ms receive allowance is long enough to survive USB scheduler jitter,
    while runtime reads remain non-blocking to teleop because they happen in
    the sampler thread and use a single transaction.
    """

    def set_packet_timeout(self: Any, packet_length: int) -> None:
        self.packet_start_time = self.getCurrentTime()
        transmission_ms = self.tx_time_per_byte * float(packet_length)
        self.packet_timeout = transmission_ms + (self.tx_time_per_byte * 3.0) + 50.0

    port.setPacketTimeout = MethodType(set_packet_timeout, port)


_RETRYABLE_SDK_EXCEPTIONS = (IndexError, OSError)

_ID_ADDR = 5
_TORQUE_ENABLE_ADDR = 40
_LOCK_ADDR = 55
_PRESENT_POSITION_ADDR = 56
_MIDDLE_CALIBRATION = 128  # write to Torque_Enable to set current pos as 2048
