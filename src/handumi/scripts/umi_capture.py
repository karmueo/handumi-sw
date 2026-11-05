"""单台 Vive/USB/ArUco 的原生诊断、预览和严格录制入口。"""

from __future__ import annotations

import argparse
import json
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_to_dict,
)
from handumi.calibration.umi import estimator, pose7, verify
from handumi.calibration.umi_table import session_core
from handumi.cameras.opencv import CameraFrameUnavailable, OpenCVCameraDevice
from handumi.dataset.raw import (
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
    camera_health_features,
    capture_timing_features,
    pose7_feature,
    raw_state_feature,
    raw_tracking_features,
    scalar_feature,
)
from handumi.grippers.base import GripperObservation, VisualGripperProvider
from handumi.robots.utils import pose7_to_mat
from handumi.scripts._record_encoding import (
    _install_strict_streaming_encoder,
    _prepare_streaming_episode,
    _select_video_encoder,
    _vive_rgb_encoder_config,
)
from handumi.scripts._record_output import (
    _check_output_directory,
    _overwrite_arg,
    _prepare_output_directory,
)
from handumi.scripts._umi_preview import capture_blueprint
from handumi.scripts._umi_session import (
    DEFAULT_CALIBRATION as DEFAULT_CALIBRATION,
)
from handumi.scripts._umi_session import (
    calibration_paths,
    load_session,
)
from handumi.tracking.vive import ViveTrackingProvider


@dataclass(frozen=True)
class CaptureHealth:
    """保存严格健康判定所用的各项检查，供录制与预览共同使用。"""

    tracker: bool
    gripper: bool
    camera_age_ns: int
    synchronization_error_ns: int

    @property
    def camera_ok(self):
        """相机帧龄在原有的有效范围内。"""
        return 0 <= self.camera_age_ns <= 350_000_000

    @property
    def synchronization_ok(self):
        """校正后的相机时间与目标时间之差未超过原有限制。"""
        return abs(self.synchronization_error_ns) <= 60_000_000

    @property
    def healthy(self):
        """返回原有的严格采集有效性，不额外检查位姿标志。"""
        return bool(self.tracker and self.gripper and self.camera_ok and self.synchronization_ok)


def capture_health(image, tracking, gripper, target, corrected, checked_now):
    """按采样时刻计算健康检查，供录制判定和状态解释复用。"""
    return CaptureHealth(
        tracker=tracking.right_tracked,
        gripper=gripper.valid,
        camera_age_ns=checked_now - image.capture_time_ns,
        synchronization_error_ns=corrected - target,
    )


@contextmanager
def keyboard():
    """临时使用非阻塞终端按键；退出或异常均恢复终端设置。"""
    if not sys.stdin.isatty():
        raise RuntimeError("键盘录制需要交互终端")
    previous = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield (
            lambda: (
                sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else ""
            )
        )
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous)


class NativeUmi:
    """独占管理一台相机和 Tracker，向所有消费者提供同一组选定样本。"""

    def __init__(
        self, snapshot, *, tracker=None, camera=None, session=None,
        gripper_hold_ns=500_000_000,
    ):
        """注入设备便于测试；默认按已验证快照创建原生设备。"""
        verify(snapshot, require_time=False)
        self.snapshot = snapshot
        self.offset_ns = round((snapshot.get("time") or {}).get("offset_s", 0) * 1e9)
        identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        calibration = ControllerTcpCalibration(identity, pose7(snapshot["tracker_tcp"]))
        self.tracker = tracker or ViveTrackingProvider(snapshot["serial"], calibration)
        if session is not None:
            self.tracker.set_workspace_from_device_pose(pose7(session["table_from_device"]))
        width, height = snapshot["resolution"]
        self.camera = camera or OpenCVCameraDevice(
            snapshot["camera_device"],
            snapshot["fps"],
            width,
            height,
            snapshot["fourcc"],
        )
        self.gripper = VisualGripperProvider(
            estimator(snapshot), hold_timeout_ns=gripper_hold_ns
        )

    def __enter__(self):
        """打开设备；任一步失败均释放已经取得的资源。"""
        try:
            self.tracker.start()
            self.camera.connect()
        except BaseException as error:
            try:
                self.__exit__(None, None, None)
            except Exception as cleanup_error:
                error.add_note(f"设备清理时也发生错误：{cleanup_error}")
            raise
        return self

    def __exit__(self, *args):
        """关闭相机和定位后台线程。"""
        try:
            self.camera.disconnect()
        finally:
            self.tracker.stop()

    def sample(self, now_ns=None, *, after_sequence=None):
        """选取延迟 150 ms 的图像，在其校正时刻插值定位并检查有效性。"""
        if getattr(self.tracker, "error", None) is not None:
            raise RuntimeError("Vive 采集线程失败") from self.tracker.error
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        started = time.monotonic_ns()
        target = now_ns - 150_000_000
        if after_sequence is None:
            image = self.camera.sample_at(target - self.offset_ns)
        else:
            image = self.camera.sample_at(target - self.offset_ns, after_sequence=after_sequence)
        corrected = image.capture_time_ns + self.offset_ns
        tracking = self.tracker.sample_at(corrected)
        gripper = self.gripper.observe(image, offset_ns=self.offset_ns)
        checked_now = now_ns + time.monotonic_ns() - started
        self.last_health = capture_health(
            image, tracking, gripper, target, corrected, checked_now,
        )
        return image, tracking, gripper, target, self.last_health.healthy


def metadata(snapshot, session=None):
    """生成独立 UMI 身份和原始数据契约，不依赖目标机器人夹爪。"""
    identity = np.array([0, 0, 0, 0, 0, 0, 1])
    tcp = calibration_to_dict(left=identity, right=pose7(snapshot["tracker_tcp"]))[
        "calibration"
    ]
    result = {
        "recording_device": "vive",
        "active_sides": ["right"],
        "tracking_schema": HANDUMI_TRACKING_SCHEMA,
        "capture_schema": "synchronized_sources_v2",
        "state_semantics": HANDUMI_STATE_SEMANTICS,
        "tracking_workspace": "table" if session is not None else "openvr_raw",
        "tcp_pose": {
            "feature": "observation.tracking.right_tcp_pose",
            "frame": "table" if session is not None else "openvr_raw",
            "sample_time_feature": "observation.tracking.aligned_time_ns",
        },
        "controller_tcp_calibration": {
            "schema_version": 2,
            "applied_to_state": False,
            "source_tool": "fastumi",
            "source_gripper": "aruco",
            "tracking_device": "vive",
            "controller_mount": snapshot["geometry_digest"],
            "sha256": snapshot["geometry_digest"],
            **tcp,
        },
        "umi_calibration": snapshot,
        "sources": {
            "gripper": {
                "left": {
                    "enabled": False,
                    "width_available": False,
                    "normalized_available": False,
                },
                "right": {
                    "enabled": True,
                    "backend": "aruco",
                    "camera": "right_wrist",
                    "width_available": False,
                    "normalized_available": True,
                },
            },
            "cameras": {"right_wrist": {"enabled": True}},
        },
    }
    if session is not None:
        result["umi_table_session"] = deepcopy(session)
    return result


def features(snapshot):
    """构造与旧 16 维布局兼容、带显式夹爪能力的 v2 特征。"""
    result = {
        "observation.state": raw_state_feature(),
        "action": raw_state_feature(),
        **raw_tracking_features(),
        "observation.tracking.right_tcp_pose": pose7_feature(),
        **capture_timing_features(),
        **camera_health_features(["right_wrist"]),
    }
    for side in ("left", "right"):
        for name in ("normalized", "valid", "sample_time_ns", "sequence"):
            result[f"observation.gripper.{side}.{name}"] = scalar_feature(
                "float32" if name == "normalized" else "int64"
            )
    result["observation.camera.right_wrist.capture_time_ns"] = scalar_feature("int64")
    width, height = snapshot["resolution"]
    result["observation.images.right_wrist"] = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    for feature in result.values():
        feature["shape"] = tuple(feature["shape"])
    return result


def assemble_frame(image, tracking, gripper, target, now, task):
    """装配共享图像、原始锚点、校正时间及开度；未知净宽保持有限占位。"""
    state = np.r_[
        tracking.left_controller_pose, tracking.right_controller_pose, 0, 0
    ].astype(np.float32)

    def scalar(value):
        """将诊断数值编码为单元素整数列。"""
        return np.array([int(value)], dtype=np.int64)

    return {
        "observation.state": state,
        "action": state.copy(),
        "task": task,
        "observation.images.right_wrist": image.image,
        **tracking.tracking_frame(),
        "observation.tracking.right_tcp_pose": tracking.right_tcp_pose.copy(),
        **gripper.frame(),
        **GripperObservation(0, False, 0, 0).frame("left"),
        "observation.sync.target_time_ns": scalar(target),
        "observation.sync.record_time_ns": scalar(now),
        "observation.camera.right_wrist.capture_time_ns": scalar(image.capture_time_ns),
        "observation.camera.right_wrist.sample_time_ns": scalar(gripper.sample_time_ns),
        "observation.camera.right_wrist.sequence": scalar(image.sequence),
        "observation.camera.right_wrist.healthy": scalar(True),
    }


def transition(recording, key, healthy):
    """执行严格录制状态机，失效优先于保存；返回新状态和动作。"""
    if key == "\x03":
        return False, "exit"
    if recording and not healthy:
        return False, "discard"
    if key == "\x1b":
        return False, "discard" if recording else "exit"
    if key == " ":
        if recording:
            return False, "save"
        return (True, "start") if healthy else (False, "wait")
    return recording, "frame" if recording else "wait"



def validate_resume_target(root: Path, expected_metadata: dict) -> dict:
    """启动设备前检查本地续录元数据；无已保存段或元数据缺失时拒绝续录。"""
    try:
        info = json.loads((root / "meta/info.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法续录：无法读取 {root / 'meta/info.json'}") from exc
    if info.get("total_episodes", 0) == 0:
        raise ValueError(
            f"无法续录：{root} 尚未保存任何录制段。"
            "请使用新的 --output-dir 并去掉 --resume 开始录制；原目录保持不变。"
        )
    if not (root / "meta/tasks.parquet").is_file():
        raise ValueError("无法续录：本地缺少 meta/tasks.parquet，请检查或恢复原数据集。")
    if not any((root / "meta/episodes").rglob("*.parquet")):
        raise ValueError("无法续录：本地缺少 meta/episodes 段元数据，请检查或恢复原数据集。")
    tcp_name = "observation.tracking.right_tcp_pose"
    available = info.get("features")
    tcp_feature = available.get(tcp_name) if isinstance(available, dict) else None
    if tcp_feature is None:
        raise ValueError(
            "旧数据集未保存 TCP 位姿，无法续录；请使用新的 --output-dir。"
            "已有数据仍可读取和转换。"
        )
    expected_feature = pose7_feature()
    if (
        not isinstance(tcp_feature, dict)
        or tcp_feature.get("dtype") != expected_feature["dtype"]
        or tcp_feature.get("shape") not in ([7], (7,))
        or tcp_feature.get("names") != expected_feature["names"]
    ):
        raise ValueError("无法续录：TCP 位姿字段的数据类型、形状或名称不匹配")

    def stable(value):
        """排除桌面复验报告变化，仍严格比较会话身份和其他数据契约。"""
        result = deepcopy(value)
        if isinstance(result, dict) and "umi_table_session" in result:
            result["umi_table_session"] = session_core(result["umi_table_session"])
        return result

    if stable(info.get("handumi")) != stable(expected_metadata):
        raise ValueError("续录的设备、数据契约或标定不一致；重新求解桌面后请创建新数据集")
    return info["handumi"]


def _coordinate_axes(viewer, length):
    """在局部原点绘制 X 红、Y 深绿、Z 深蓝三根指定长度的箭头。"""
    return viewer.Arrows3D(
        origins=np.zeros((3, 3)), vectors=np.eye(3) * length,
        radii=viewer.Radius.ui_points(1.0),
        colors=[[255, 0, 0], [0, 128, 0], [0, 0, 139]], show_labels=False,
    )


def _capture_issues(health, duplicate_frame):
    """列出严格录制判定和录制期重复帧所依据的异常。"""
    issues = []
    if not health.tracker:
        issues.append("Tracker 未追踪")
    if not health.gripper:
        issues.append("夹爪检测无效")
    if health.camera_age_ns < 0:
        issues.append("相机时间异常")
    elif not health.camera_ok:
        issues.append("相机帧过旧")
    if not health.synchronization_ok:
        issues.append("相机同步超限")
    if duplicate_frame:
        issues.append("相机帧重复")
    return issues


def capture_status_text(
    *, mode, health, tracking, gripper, healthy, duplicate_frame=False,
    recording=False, saving=False, exit_requested=False, frame_count=0, fps=30, saved=0, last_result="",
    camera_status=None, total_episodes=0,
):
    """生成采集状态；总数仅包含已保存段，零起始 ID 对应当前或下一段。"""
    issues = _capture_issues(health, duplicate_frame)
    if not healthy and not issues:
        issues.append("采集状态无效")
    if mode == "record":
        phase = "保存中" if saving else (
            "录制中" if recording else ("可开始录制" if healthy else "等待设备就绪")
        )
        lines = [
            f"## {phase}",
            f"当前段：{frame_count} 帧 · 数据时长 {frame_count / fps:.1f} 秒  ",
            f"本次已保存：{saved} 段  ",
            f"episode 总数（已保存）：{total_episodes}  ",
            f"{'当前' if recording or saving else '下一段'} episode ID：{total_episodes}（从 0 开始）",
        ]
        if last_result:
            lines.extend(["", f"最近结果：{last_result}"])
    else:
        label = "设备诊断" if mode == "doctor" else "采集预览"
        lines = [f"## {label} · {'设备就绪' if healthy else '等待设备就绪'}"]
    camera = "正常" if health.camera_ok else (
        "时间异常" if health.camera_age_ns < 0 else "帧过旧"
    )
    if camera_status == "reconnecting":
        camera += "、正在自动重连"
    elif camera_status == "waiting":
        camera += "、等待新画面"
    elif camera_status == "error":
        camera += "、重连失败，请检查终端"
    if duplicate_frame:
        camera += "、帧重复"
    tracker = "追踪正常" if tracking.right_tracked else "未追踪"
    pose = "位姿有效" if tracking.right_pose_valid else "位姿无效"
    gripper_status = (
        getattr(gripper, "reason", "") or ("检测正常" if gripper.valid else "检测无效")
    )
    sync = "正常" if health.synchronization_ok else "超限"
    lines.extend([
        "",
        f"相机：{camera} · 帧龄 {health.camera_age_ns / 1e6:.0f} ms  ",
        "采集包含约 150 ms 同步缓冲  ",
        f"Tracker：{tracker} · {pose}  ",
        f"夹爪：{gripper_status}  ",
        f"同步：{sync} · 偏差 {health.synchronization_error_ns / 1e6:+.0f} ms",
    ])
    if issues:
        lines.extend(["", "不可录制原因：" + "；".join(issues)])
    if mode == "record" and saving:
        lines.extend(["", "保存完成后退出" if exit_requested else "正在保存，请稍候"])
    elif mode == "record":
        lines.extend([
            "",
            "空格：保存当前段" if recording else "空格：开始录制",
            "  ",
            "Esc：丢弃当前段" if recording else "Esc：退出",
            "  ",
            "请在启动命令的终端中使用按键",
        ])
    return "\n".join(lines)


def log_capture_status(viewer, **status):
    """静态更新状态文档，使其不受时间游标位置影响。"""
    viewer.log(
        "capture_status", viewer.TextDocument(capture_status_text(**status),
                                              media_type="text/markdown"),
        static=True,
    )


def log_live_preview(
    viewer, image, tracking, gripper, healthy, camera_age_s, axis_length=0.08,
    *, session=None, tracker_from_camera=None,
):
    """记录最新图像与同一采集时间的 Tracker、TCP 和相机坐标。

    tracker_from_camera 是 T_Tracker_Camera；None 保留旧内部调用的行为。
    相机轴长固定为 0.06 米，其节点也是可选视锥与图像的父节点。
    """
    viewer.set_time("capture", duration=image.capture_time_ns / 1e9)
    viewer.log(
        "camera/right_wrist",
        viewer.Image(image.image).compress(jpeg_quality=80),
        static=True,
    )
    viewer.log("gripper/right", viewer.Scalars(gripper.normalized))
    viewer.log("health/valid", viewer.Scalars(int(healthy)))
    viewer.log("health/camera_sequence", viewer.Scalars(image.sequence))
    viewer.log("health/camera_age_s", viewer.Scalars(camera_age_s))
    if tracking.right_tracked and tracking.right_pose_valid:
        for name, pose, length, color, label in (
            ("tracker", tracking.right_controller_pose, 0.06, [255, 190, 50], "Tracker"),
            ("tcp", tracking.right_tcp_pose, axis_length, [80, 220, 255], "TCP"),
        ):
            viewer.log(f"tracking/{name}", viewer.Transform3D(
                translation=pose[:3],
                rotation=viewer.Quaternion(xyzw=pose[3:]),
            ))
            # 箭头和名称均使用父坐标系局部坐标，定位失效由父级递归清除。
            viewer.log(f"tracking/{name}/axes", _coordinate_axes(viewer, length))
            viewer.log(f"tracking/{name}/origin", viewer.Points3D(
                [[0, 0, 0]], radii=0.005, colors=color,
                labels=[label], show_labels=True,
            ))
        if tracker_from_camera is not None:
            # Tracker 已位于桌面或 OpenVR 参考系；外参将相机坐标映射至 Tracker。
            reference_from_camera = (
                pose7_to_mat(tracking.right_controller_pose) @ tracker_from_camera
            )
            viewer.log("tracking/camera", viewer.Transform3D(
                translation=reference_from_camera[:3, 3],
                mat3x3=reference_from_camera[:3, :3],
            ))
            viewer.log("tracking/camera/axes", _coordinate_axes(viewer, 0.06))
            viewer.log("tracking/camera/origin", viewer.Points3D(
                [[0, 0, 0]], radii=0.005, colors=[110, 180, 255],
                labels=["Camera"], show_labels=True,
            ))
    else:
        viewer.log("tracking/tracker", viewer.Clear(recursive=True))
        viewer.log("tracking/tcp", viewer.Clear(recursive=True))
        if tracker_from_camera is not None:
            viewer.log("tracking/camera", viewer.Clear(recursive=True))


def init_tracking_preview(viewer, session):
    """显示固定参考系三轴；未标定诊断明确使用 OpenVR 原始参考系。"""
    viewer.log("tracking", (
        viewer.ViewCoordinates.RIGHT_HAND_Z_UP if session is not None
        else viewer.ViewCoordinates.RIGHT_HAND_Y_UP
    ), static=True)
    viewer.log("tracking/reference", viewer.Transform3D(translation=[0, 0, 0]), static=True)
    viewer.log("tracking/reference/axes", _coordinate_axes(viewer, 0.12), static=True)
    viewer.log(
        "tracking/reference/origin",
        viewer.Points3D(
            [[0, 0, 0]], radii=0.005, colors=[230, 230, 230],
            labels=["Table" if session is not None else "OpenVR"], show_labels=True,
        ),
        static=True,
    )
    # 低调角点稳定初始自动取景范围，不表示真实桌面边界。
    if session is not None:
        bounds = [[x * 0.75, y * 0.75, z * 0.4]
                  for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    else:
        bounds = [[x * 0.75, y * 0.4, z * 0.75]
                  for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    viewer.log(
        "tracking/bounds",
        viewer.Points3D(
            bounds, colors=[[128, 100, 100, 90]] * 8, radii=0.004, show_labels=False,
        ),
        static=True,
    )
    reference = (
        "Desktop frame: X right, Y away from operator, Z up"
        if session is not None
        else "Raw OpenVR frame (desktop not calibrated)"
    )
    viewer.log("reference_frame", viewer.TextDocument(reference), static=True)


def _parse_gripper_hold_ms(value):
    """校验漏检保持期限为整数毫秒；0 表示关闭保持。"""
    try:
        milliseconds = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "--gripper-hold-ms 必须是非负整数毫秒"
        ) from error
    if milliseconds < 0:
        raise argparse.ArgumentTypeError("--gripper-hold-ms 必须是非负整数毫秒")
    return milliseconds


class _RerunCaptureViewer:
    """将显式 RecordingStream 适配为采集预览使用的 rr 模块接口。"""

    def __init__(self, stream, rr) -> None:
        """包装相机与追踪预览共享的记录流。"""
        self.stream = stream
        self._rr = rr

    def __getattr__(self, name):
        """把 Rerun archetype 访问委托给 rerun 模块。"""
        return getattr(self._rr, name)

    def log(self, entity_path, entity, *, static=False) -> None:
        """把实体写入共享记录流。"""
        self.stream.log(entity_path, entity, static=static)

    def set_time(self, timeline, **kwargs) -> None:
        """设置当前线程的采集时间轴。"""
        self.stream.set_time(timeline, **kwargs)

    def flush(self) -> None:
        """在长时间保存前推送最新状态。"""
        self.stream.flush()

    def disconnect(self) -> None:
        """刷新并关闭共享记录流。"""
        try:
            self.stream.flush()
        finally:
            self.stream.disconnect()


def _start_camera_viewer(args, session):
    """启动相机、追踪、夹爪与健康状态共享的 Rerun 窗口。"""
    if args.no_preview:
        return None
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun_cli

    # 避免连接已有的旧版 Viewer，使用当前 SDK 自带的二进制及独立端口。
    viewer_dir = Path(rerun_cli.__file__).parent
    if sys.platform == "darwin" and (viewer_dir / "Rerun.app").exists():
        executable = viewer_dir / "Rerun.app/Contents/MacOS/Rerun"
    else:
        executable = viewer_dir / ("rerun.exe" if sys.platform == "win32" else "rerun")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stream = rr.RecordingStream("handumi_umi")
    rr.spawn(port=port, executable_path=str(executable), recording=stream)
    viewer = _RerunCaptureViewer(stream, rr)
    stream.send_blueprint(
        capture_blueprint(rrb, table_frame=session is not None),
        make_active=True,
        make_default=True,
    )
    init_tracking_preview(viewer, session)
    return viewer


def _open_recording_dataset(args, snapshot, recording_metadata):
    """创建或续录 LeRobot 数据集；测试可替换此函数。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from handumi.dataset.lerobot_compat import preserve_dataset_info

    width, height = snapshot["resolution"]
    selection = _select_video_encoder(
        policy=args.encoder, requested_vcodec=None, width=width, height=height,
        fps=args.fps, camera_count=1, requested_threads=None, h264_only=True,
        streaming_probe=True,
    )
    kind = "硬件" if selection.hardware else "软件"
    print(f"视频编码：{selection.vcodec}（{kind}，流式）；策略：{args.encoder}。", flush=True)
    options = dict(
        rgb_encoder=_vive_rgb_encoder_config(selection.vcodec),
        streaming_encoding=True, encoder_queue_maxsize=30,
        encoder_threads=selection.threads, image_writer_threads=0,
    )
    if args.resume:
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id, root=args.output_dir,
            **options,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output_dir,
            fps=30,
            robot_type="handumi_raw",
            features=features(snapshot),
            use_videos=True,
            **options,
        )
    _install_strict_streaming_encoder(
        dataset, verify_video_frames=True, isolate_process=True, rgb_vcodec=selection.vcodec,
    )
    dataset._handumi_encoding = {"codec": selection.vcodec, "hardware": selection.hardware}
    preserve_dataset_info(dataset, {"handumi": recording_metadata})
    return dataset


def _save_streaming_episode(dataset, frame_count):
    """先结束并校验流式视频，再提交数据；返回各阶段耗时（秒）。"""
    start = time.monotonic()
    timings = _prepare_streaming_episode(dataset, frame_count)
    prepared = time.monotonic()
    dataset.save_episode()
    finished = time.monotonic()
    return {"encode_finish_s": prepared - start, **timings, "commit_s": finished - prepared}


def _capture_metrics(frame_times, camera_times, add_durations, save_durations, total_s):
    """汇总录制实测速度及保存阶段耗时，时间均来自单调时钟。"""
    frame_intervals = np.diff(frame_times)
    camera_intervals = np.diff(camera_times)
    elapsed = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0.0
    return {
        "frames": len(frame_times),
        "capture_fps": (len(frame_times) - 1) / elapsed if elapsed > 0 else None,
        "frame_interval_ms_p95": float(np.percentile(frame_intervals, 95) * 1000) if len(frame_intervals) else None,
        "camera_interval_ms_p95": float(np.percentile(camera_intervals, 95) * 1000) if len(camera_intervals) else None,
        "add_frame_ms_p95": float(np.percentile(add_durations, 95) * 1000) if add_durations else None,
        **save_durations,
        "save_total_s": total_s,
    }


def _next_capture_deadline(deadline, now, fps):
    """计算下一次采样节拍；超期时舍弃积压，避免快速补跑重复帧。"""
    return max(deadline + 1 / fps, now)


def main(argv=None, *, mode="record"):
    """通过共享运行时执行诊断、图像预览或 LeRobot 录制。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["vive"], default="vive")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--session-calibration", type=Path)
    parser.add_argument("--tcp-axis-length-m", type=float, default=0.08)
    parser.add_argument("--rig-config", type=Path, default=Path("configs/rig.yaml"))
    parser.add_argument("--side", choices=["right"], default="right")
    parser.add_argument("--fps", type=int, choices=[30], default=30)
    parser.add_argument(
        "--gripper-hold-ms", type=_parse_gripper_hold_ms, default=500, metavar="MS",
        help="ArUco 短暂漏检保持期限（整数毫秒，默认 500；0 关闭保持）",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-id", default="local/umi")
    parser.add_argument("--task", default="UMI demonstration")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=0)
    preview_options = parser.add_mutually_exclusive_group()
    preview_options.add_argument("--no-preview", action="store_true")
    preview_options.add_argument(
        "--spatial-view", action="store_true",
        help="在三维预览中显示相机视锥和去畸变画面（默认关闭）",
    )
    if mode == "record":
        parser.add_argument(
            "--encoder", choices=("auto", "cpu", "gpu"), default="auto",
            help="视频编码策略：自动优先硬件、CPU 或仅 GPU（默认 auto）",
        )
        parser.add_argument(
            "--overwrite", type=_overwrite_arg, default=True, metavar="{true,false}",
            help="默认 true：删除整个已有输出目录；false：目录存在时报错。--resume 忽略此开关并追加。",
        )
        parser.add_argument("--dry-run", action="store_true", help="仅展示录制计划，不修改目录或连接硬件。")
    args = parser.parse_args(argv)
    if not np.isfinite(args.tcp_axis_length_m) or args.tcp_axis_length_m <= 0:
        parser.error("--tcp-axis-length-m 必须为正的有限数值")
    explicit_session = args.session_calibration is not None
    args.calibration, args.session_calibration = calibration_paths(args)
    snapshot = verify(
        yaml.safe_load(args.calibration.read_text()), require_time=mode != "visualize"
    )
    if mode == "record" and args.output_dir is None:
        parser.error("录制需要 --output-dir")
    try:
        session = load_session(
            args.session_calibration, snapshot, required=mode == "record",
            explicit=explicit_session,
        )
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    recording_metadata = metadata(snapshot, session)
    print("参考坐标系：桌面" if session else "参考坐标系：OpenVR 原始坐标系（尚未标定桌面）")
    if mode == "record" and args.resume:
        try:
            recording_metadata = validate_resume_target(args.output_dir, recording_metadata)
        except ValueError as exc:
            parser.error(str(exc))
    if mode == "record":
        _check_output_directory(args)
        operation = "resume" if args.resume else (
            "overwrite entire directory" if args.output_dir.exists() else "new dataset"
        )
        print(f"Recording plan: {args.output_dir} ({operation})")
        if args.dry_run:
            return
        _prepare_output_directory(args)
    dataset = None
    recording = False
    previous_recorded_sequence = None  # 仅跟踪本段成功写入的真实相机帧。
    camera_stalled = False  # 仅在停帧和恢复时提示，避免终端刷屏。
    camera_waiting_first_frame = False  # 首帧尚不可用时只提示一次。
    saved = 0
    total_episodes = 0  # 主线程维护已提交总数，保存期间保持当前段 ID 稳定。
    frame_count = 0
    frame_times = []  # 每个成功录制帧的主机单调时间。
    camera_times = []  # 对应相机采集时间，用于计算原始帧间隔。
    add_durations = []  # 每帧添加至数据集所花时间。
    save_requested_at = 0.0  # 保存请求时刻，含等待后台线程的时间。
    encoding_info = {}  # 数据集创建时记录编码器，不在后台保存时读取。
    last_result = ""
    viewer = None
    next_preview = 0.0  # 限制显示频率，降低预览传输压力。
    save_executor = None  # 唯一后台写入线程；仅保存片段时创建。
    save_future = None  # 未完成或尚未收取结果的保存任务。
    save_failure = None  # 保存失败优先于清理期间的其他异常。
    exit_requested = False  # 保存完成后退出，避免中断写入。
    interrupted = False  # 信号处理器只置位，由采集循环处理。
    previous_sigint = None
    if mode == "record" and threading.current_thread() is threading.main_thread():
        previous_sigint = signal.getsignal(signal.SIGINT)

        def request_interrupt(signum, frame):
            """延迟处理中断，确保提交任务及清理期间不会打断数据集写入。"""
            nonlocal interrupted
            interrupted = True

        signal.signal(signal.SIGINT, request_interrupt)

    try:
        spatial_preview = None
        if args.spatial_view:
            from handumi.scripts._umi_spatial_preview import SpatialCameraPreview

            spatial_preview = SpatialCameraPreview(snapshot)
        viewer = _start_camera_viewer(args, session)
        # 相机外参在会话内固定，首次初始化后供每次预览复用。
        tracker_from_camera = (
            np.asarray(snapshot["tracker_camera"], dtype=float)
            if viewer is not None else None
        )
        with (
            NativeUmi(
                snapshot, session=session,
                gripper_hold_ns=args.gripper_hold_ms * 1_000_000,
            ) as devices,
            nullcontext(lambda: "") if mode == "doctor" else keyboard() as read_key,
        ):
            if mode == "record":
                dataset = _open_recording_dataset(
                    args, snapshot, recording_metadata
                )
                total_episodes = dataset.meta.total_episodes if args.resume else 0
                print(
                    f"episode 总数（已保存）：{total_episodes}；"
                    f"下一段 episode ID：{total_episodes}（从 0 开始）",
                    flush=True,
                )
                info = getattr(dataset, "_handumi_encoding", None)
                if isinstance(info, dict):
                    encoding_info = info
            print("空格开始/保存；Esc 取消/退出；Ctrl+C 退出。无效帧立即丢弃当前段。")
            deadline = time.monotonic()
            readiness_deadline = deadline + 10
            while True:
                try:
                    now = time.monotonic_ns()
                    try:
                        if recording and previous_recorded_sequence is not None:
                            image, tracking, gripper, target, healthy = devices.sample(
                                now, after_sequence=previous_recorded_sequence,
                            )
                        else:
                            image, tracking, gripper, target, healthy = devices.sample(now)
                    except CameraFrameUnavailable:
                        if not camera_waiting_first_frame:
                            print("相机尚无画面，正在等待并自动尝试恢复。", flush=True)
                            camera_waiting_first_frame = True
                        key = read_key()
                        if interrupted or key in {"\x1b", "\x03"}:
                            break
                        if mode == "doctor" and time.monotonic() > readiness_deadline:
                            raise RuntimeError("10 秒内相机未提供画面，请检查 USB 连接")
                        continue
                    camera_waiting_first_frame = False
                    camera_age_s = (time.monotonic_ns() - image.capture_time_ns) / 1e9
                    # NativeUmi 留下与健康判定相同的一次检查结果；可注入设备按相同规则计算。
                    health = getattr(devices, "last_health", None)
                    if health is None:
                        health = capture_health(
                            image, tracking, gripper, target,
                            gripper.sample_time_ns, image.capture_time_ns + round(camera_age_s * 1e9),
                        )
                    stalled = camera_age_s > 1.0
                    if stalled and not camera_stalled:
                        print(
                            f"相机画面已停更 {camera_age_s:.1f} 秒，当前显示缓存帧；"
                            "正在尝试自动重连，请检查相机 USB 连接。",
                            flush=True,
                        )
                    elif camera_stalled and not stalled:
                        print("相机画面已恢复更新。", flush=True)
                    camera_stalled = stalled
                    duplicate_frame = (
                        recording and previous_recorded_sequence is not None
                        and image.sequence <= previous_recorded_sequence
                    )
                    if duplicate_frame:
                        healthy = False
                    was_recording = recording
                    key = read_key()
                    if interrupted:
                        key = "\x03"
                        interrupted = False
                    saving_at_key = save_future is not None
                    if saving_at_key:
                        # 保存开始后消费空格，完成当轮也不能把它当成下一段开始。
                        if key in {"\x1b", "\x03"}:
                            exit_requested = True
                            next_preview = 0.0
                            print("正在保存，完成后退出。", flush=True)
                        action = "wait"
                        recording = False
                        if save_future.done():
                            completed = save_future
                            save_future = None
                            try:
                                save_durations = completed.result()
                            except BaseException as exc:
                                save_failure = exc
                                raise RuntimeError(f"保存当前段失败：{exc}") from exc
                            saved += 1
                            total_episodes += 1
                            metrics = _capture_metrics(
                                frame_times, camera_times, add_durations, save_durations,
                                time.monotonic() - save_requested_at,
                            )
                            metrics.update(encoding_info)
                            print("录制性能：" + json.dumps(metrics, ensure_ascii=False), flush=True)
                            last_result = f"已保存 {frame_count} 帧"
                            frame_count = 0
                            next_preview = 0.0
                            print(
                                f"本次已保存 {saved} 段；已保存 episode ID：{total_episodes - 1}；"
                                f"episode 总数（已保存）：{total_episodes}；"
                                f"下一段 episode ID：{total_episodes}", flush=True,
                            )
                            if args.num_episodes > 0 and saved >= args.num_episodes:
                                exit_requested = True
                    else:
                        recording, action = transition(recording, key, healthy)
                    if mode != "record":
                        recording = False
                    if action == "exit":
                        if dataset is not None and was_recording:
                            dataset.clear_episode_buffer()
                        break

                    status = dict(
                        mode=mode, health=health, tracking=tracking, gripper=gripper,
                        healthy=healthy, duplicate_frame=duplicate_frame,
                        recording=recording, frame_count=frame_count, fps=args.fps,
                        saved=saved, last_result=last_result, total_episodes=total_episodes,
                        camera_status=getattr(getattr(devices, "camera", None), "status", None),
                    )
                    if dataset is not None:
                        if action == "start":
                            print(
                                f"开始录制；当前 episode ID：{total_episodes}；"
                                f"episode 总数（已保存）：{total_episodes}", flush=True,
                            )
                            frame_count = 0
                            previous_recorded_sequence = None
                            frame_times.clear()
                            camera_times.clear()
                            add_durations.clear()
                            last_result = ""
                        if action == "discard":
                            dataset.clear_episode_buffer()
                            previous_recorded_sequence = None
                            causes = _capture_issues(health, duplicate_frame)
                            last_result = "当前段已丢弃：" + (
                                "；".join(causes) if not healthy and causes
                                else "采集状态无效" if not healthy else "用户取消"
                            )
                            frame_count = 0
                            print(f"{last_result}；按空格重新开始。", flush=True)
                        elif action == "save":
                            previous_recorded_sequence = None
                            save_requested_at = time.monotonic()
                            if viewer is not None:
                                status.update(recording=True, saving=True)
                                log_capture_status(viewer, **status)
                                viewer.flush()
                            if save_executor is None:
                                save_executor = ThreadPoolExecutor(max_workers=1)
                            save_future = save_executor.submit(_save_streaming_episode, dataset, frame_count)
                        elif recording:
                            frame = assemble_frame(
                                image, tracking, gripper, target,
                                time.monotonic_ns(), args.task,
                            )
                            add_start = time.monotonic()
                            dataset.add_frame(frame)
                            add_end = time.monotonic()
                            previous_recorded_sequence = image.sequence
                            frame_times.append(now / 1e9)
                            camera_times.append(image.capture_time_ns / 1e9)
                            add_durations.append(add_end - add_start)
                            frame_count += 1
                    if viewer is not None and (
                        time.monotonic() >= next_preview or action in {"start", "save", "discard"}
                    ):
                        status.update(
                            recording=recording, saving=save_future is not None,
                            exit_requested=exit_requested, frame_count=frame_count,
                            saved=saved, last_result=last_result,
                        )
                        log_capture_status(viewer, **status)
                        log_live_preview(
                            viewer, image, tracking, gripper, healthy, camera_age_s,
                            args.tcp_axis_length_m, session=session,
                            tracker_from_camera=tracker_from_camera,
                        )
                        if spatial_preview is not None:
                            spatial_preview.log(viewer, image, tracking, camera_age_s)
                        next_preview = time.monotonic() + 1 / 15
                    if exit_requested and save_future is None:
                        if viewer is not None:
                            viewer.flush()
                        break
                    if mode == "doctor":
                        print(
                            f"Tracker={tracking.right_tracked}; ArUco={gripper.valid}; synchronized={healthy}"
                        )
                        if healthy:
                            break
                        if time.monotonic() > readiness_deadline:
                            raise RuntimeError(
                                "10 秒内设备未达到可录制状态，请检查 Tracker、双标签和同步"
                            )
                    loop_end = time.monotonic()
                    deadline = _next_capture_deadline(deadline, loop_end, 30)
                    time.sleep(max(0, deadline - loop_end))
                except KeyboardInterrupt:
                    # 兼容显式抛出的中断；真正的 SIGINT 由处理器置位。
                    if save_future is not None:
                        exit_requested = True
                        next_preview = 0.0
                        print("正在保存，完成后退出。", flush=True)
                        continue
                    if recording and dataset is not None:
                        dataset.clear_episode_buffer()
                        recording = False
                    break
    finally:
        # 保存完成前不得关闭或修改数据集；逐项收尾并保留首个异常。
        active_error = sys.exception()
        cleanup_error = None
        try:
            if save_future is not None:
                while True:
                    try:
                        save_future.result()
                        break
                    except KeyboardInterrupt as exc:
                        if save_future.done():
                            save_failure = exc
                            break
                        continue
                    except BaseException as exc:
                        save_failure = exc
                        break
            if save_executor is not None:
                try:
                    save_executor.shutdown(wait=True)
                except BaseException as exc:
                    cleanup_error = exc
            if dataset is not None:
                if recording:
                    try:
                        dataset.clear_episode_buffer()
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
                try:
                    dataset.finalize()
                    path = args.output_dir / "meta/info.json"
                    info = json.loads(path.read_text())
                    info["handumi"] = recording_metadata
                    path.write_text(json.dumps(info, indent=2) + "\n")
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            if viewer is not None:
                try:
                    viewer.disconnect()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
        finally:
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)
        if save_failure is not None and not (
            isinstance(active_error, RuntimeError) and active_error.__cause__ is save_failure
        ):
            raise RuntimeError(f"保存当前段失败：{save_failure}") from save_failure
        if active_error is None and cleanup_error is not None:
            raise cleanup_error
