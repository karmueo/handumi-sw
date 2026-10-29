"""通过 SteamVR/OpenVR 采集单台 Vive Tracker，提供有效性严格的时间插值。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from handumi.tracking.base import ControllerPairSample, apply_tcp_calibration_pose7


class ViveTrackingProvider:
    """按序列号绑定 Tracker；SDK 可注入以便离线测试，不依赖 ROS。"""

    device = "vive"

    def __init__(self, serial: str, calibration, *, rate_hz=100, sdk=None):
        """保存设备绑定与 TCP 标定；构造时不会连接硬件。"""
        if not serial or rate_hz <= 0:
            raise ValueError("需要 Tracker 序列号和正采样率")
        self.serial = serial
        self.calibration = calibration
        self.rate_hz = rate_hz
        self.sdk = sdk
        self._history = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        """启动后台 OpenVR 会话；初始化失败直接报告。"""
        if self._thread is not None:
            raise RuntimeError("Tracker 已启动")
        if self.sdk is None:
            import openvr

            self.sdk = openvr
        self._system = self.sdk.init(self.sdk.VRApplication_Background)
        self._stop.clear()
        self.error = None
        self._history.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """持续发布查询中点时间，失效时仍保存无效样本阻断跨缺口插值。"""
        try:
            while not self._stop.is_set():
                begin = time.monotonic_ns()
                poses = self._system.getDeviceToAbsoluteTrackingPose(
                    self.sdk.TrackingUniverseRawAndUncalibrated,
                    0,
                    self.sdk.k_unMaxTrackedDeviceCount,
                )
                stamp = (begin + time.monotonic_ns()) // 2
                sample = SimpleNamespace(**vars(ControllerPairSample.empty("vive")))
                for index, pose in enumerate(poses):
                    if not pose.bDeviceIsConnected:
                        continue
                    if (
                        self._system.getTrackedDeviceClass(index)
                        != self.sdk.TrackedDeviceClass_GenericTracker
                    ):
                        continue
                    serial = self._system.getStringTrackedDeviceProperty(
                        index, self.sdk.Prop_SerialNumber_String
                    )
                    if serial != self.serial:
                        continue
                    sample.connected = bool(pose.bDeviceIsConnected)
                    sample.right_device_tracked = sample.connected
                    sample.right_pose_valid = bool(
                        pose.bPoseIsValid and sample.connected
                    )
                    sample.right_tracked = sample.right_pose_valid
                    if sample.right_tracked:
                        matrix = np.asarray(
                            getattr(
                                pose.mDeviceToAbsoluteTracking,
                                "m",
                                pose.mDeviceToAbsoluteTracking,
                            ),
                            dtype=float,
                        )
                        anchor = np.r_[
                            matrix[:, 3], Rotation.from_matrix(matrix[:, :3]).as_quat()
                        ].astype(np.float32)
                        sample.right_controller_pose = anchor
                        sample.right_device_controller_pose = anchor.copy()
                        sample.left_tcp_pose, sample.right_tcp_pose = (
                            apply_tcp_calibration_pose7(
                                sample.left_controller_pose, anchor, self.calibration
                            )
                        )
                    break
                sample = ControllerPairSample(
                    **(
                        vars(sample)
                        | dict(
                            pc_monotonic_ns=stamp,
                            aligned_time_ns=stamp,
                            streaming=sample.connected,
                            clock_synced=True,
                            sequence=begin,
                        )
                    )
                )
                with self._lock:
                    self._history.append(sample)
                self._stop.wait(
                    max(0, 1 / self.rate_hz - (time.monotonic_ns() - begin) / 1e9)
                )
        except Exception as exc:
            self.error = exc

    def latest(self):
        """返回最近观测；超过 100 ms 的观测不再视为有效。"""
        with self._lock:
            sample = (
                self._history[-1]
                if self._history
                else ControllerPairSample.empty("vive")
            )
        if self.error or time.monotonic_ns() - sample.pc_monotonic_ns > 100_000_000:
            return replace(sample, right_tracked=False, right_pose_valid=False)
        return sample

    def sample_at(self, target_time_ns):
        """仅在两端有效且间隔不超过 50 ms 时插值，禁止外推。"""
        with self._lock:
            history = list(self._history)
        for a, b in zip(history, history[1:]):
            if a.pc_monotonic_ns <= target_time_ns <= b.pc_monotonic_ns:
                gap = b.pc_monotonic_ns - a.pc_monotonic_ns
                if (
                    not a.right_tracked
                    or not b.right_tracked
                    or not 0 < gap <= 50_000_000
                ):
                    break
                fraction = (target_time_ns - a.pc_monotonic_ns) / gap
                pa, pb = a.right_controller_pose, b.right_controller_pose
                rotation = Slerp([0, 1], Rotation.from_quat([pa[3:], pb[3:]]))(
                    [fraction]
                ).as_quat()[0]
                anchor = np.r_[
                    pa[:3] * (1 - fraction) + pb[:3] * fraction, rotation
                ].astype(np.float32)
                left, right = apply_tcp_calibration_pose7(
                    a.left_controller_pose, anchor, self.calibration
                )
                return replace(
                    a,
                    right_controller_pose=anchor,
                    right_device_controller_pose=anchor.copy(),
                    left_tcp_pose=left,
                    right_tcp_pose=right,
                    aligned_time_ns=target_time_ns,
                    pc_monotonic_ns=target_time_ns,
                )
        return ControllerPairSample.empty("vive")

    def stop(self):
        """停止采集并释放 OpenVR 会话。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
            self.sdk.shutdown()
