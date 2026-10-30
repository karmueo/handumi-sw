"""单台 Vive/USB/ArUco 的原生诊断、预览和严格录制入口。"""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_to_dict,
)
from handumi.calibration.umi import estimator, pose7, verify
from handumi.calibration.umi_table import session_core
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.dataset.raw import (
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
    camera_health_features,
    capture_timing_features,
    raw_state_feature,
    raw_tracking_features,
    scalar_feature,
)
from handumi.grippers.base import GripperObservation, VisualGripperProvider
from handumi.scripts._record_output import (
    _check_output_directory,
    _overwrite_arg,
    _prepare_output_directory,
)
from handumi.scripts._umi_session import (
    DEFAULT_CALIBRATION as DEFAULT_CALIBRATION,
)
from handumi.scripts._umi_session import (
    calibration_paths,
    load_session,
)
from handumi.tracking.vive import ViveTrackingProvider


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

    def sample(self, now_ns=None):
        """选取延迟 150 ms 的图像并在校正时刻插值定位，返回严格有效性。"""
        if getattr(self.tracker, "error", None) is not None:
            raise RuntimeError("Vive 采集线程失败") from self.tracker.error
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        started = time.monotonic_ns()
        target = now_ns - 150_000_000
        image = self.camera.sample_at(target - self.offset_ns)
        corrected = image.capture_time_ns + self.offset_ns
        tracking = self.tracker.sample_at(corrected)
        gripper = self.gripper.observe(image, offset_ns=self.offset_ns)
        checked_now = now_ns + time.monotonic_ns() - started
        healthy = bool(
            tracking.right_tracked
            and gripper.valid
            and 0 <= checked_now - image.capture_time_ns <= 350_000_000
            and abs(corrected - target) <= 60_000_000
        )
        return image, tracking, gripper, target, healthy


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
    def stable(value):
        """排除桌面复验报告变化，仍严格比较会话身份和其他数据契约。"""
        result = deepcopy(value)
        if isinstance(result, dict) and "umi_table_session" in result:
            result["umi_table_session"] = session_core(result["umi_table_session"])
        return result

    if stable(info.get("handumi")) != stable(expected_metadata):
        raise ValueError("续录的设备、数据契约或标定不一致；重新求解桌面后请创建新数据集")
    return info["handumi"]


def log_live_preview(viewer, image, tracking, gripper, healthy, camera_age_s, axis_length=0.08):
    """相机始终显示最新帧，其余曲线和 TCP 保留原有时间轴语义。"""
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
        viewer.log("tracking/tcp", viewer.Transform3D(
            translation=tracking.right_tcp_pose[:3],
            rotation=viewer.Quaternion(xyzw=tracking.right_tcp_pose[3:]),
            axis_length=axis_length,
        ))
    else:
        viewer.log("tracking/tcp", viewer.Clear(recursive=True))


def init_tracking_preview(viewer, session):
    """显示固定参考系三轴；未标定诊断明确使用 OpenVR 原始参考系。"""
    viewer.log("tracking", (
        viewer.ViewCoordinates.RIGHT_HAND_Z_UP if session is not None
        else viewer.ViewCoordinates.RIGHT_HAND_Y_UP
    ), static=True)
    viewer.log("tracking/reference", viewer.Transform3D(axis_length=0.12), static=True)
    reference = (
        "Desktop frame: X right, Y away from operator, Z up"
        if session
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


def _positive_finite_float(value):
    """解析正的有限浮点参数； argparse 在硬件和目录处理前失败。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("必须是正的有限数值") from error
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是正的有限数值")
    return number


@dataclass(frozen=True)
class _RobotPreviewSetup:
    """机器人预览的重型运行前预检结果。"""

    runtime: Any
    tool: Any
    thresholds: Any
    live_controller: Any
    view: Any
    offline_solver: Any


def _validate_robot_options(parser, args):
    """校验机器人预览开关组合和阈值，并在解析阶段给出清晰错误。"""
    if not args.robot_preview:
        if (
            args.robot_tool_calibration is not None
            or args.robot_position_tolerance_m is not None
            or args.robot_orientation_tolerance_deg is not None
        ):
            parser.error("--robot-* 参数仅能与 --robot-preview 一起使用")
        args.robot_position_tolerance_m = 0.01
        args.robot_orientation_tolerance_deg = 5.0
        return
    if args.no_preview:
        parser.error("--robot-preview 与 --no-preview 不能同时指定")
    if args.robot_tool_calibration is None:
        parser.error("--robot-preview 需要 --robot-tool-calibration")
    if args.robot_position_tolerance_m is None:
        args.robot_position_tolerance_m = 0.01
    if args.robot_orientation_tolerance_deg is None:
        args.robot_orientation_tolerance_deg = 5.0


def _validate_robot_calibration(args):
    """读取并验证工具标定和阈值；不加载 JAX 模型或启动任何线程。"""
    from handumi.robots.umi_ik import IkThresholds, ToolCalibration

    try:
        tool = ToolCalibration.from_yaml(args.robot_tool_calibration)
        thresholds = IkThresholds(
            args.robot_position_tolerance_m,
            args.robot_orientation_tolerance_deg,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"机器人预览预检失败：{exc}") from exc
    return tool, thresholds


def _prepare_robot_preview(args, tool, thresholds):
    """删除输出目录前完成模型加载、完整网格加载和两套 solver 预热。"""
    from handumi.robots.registry import load_embodiment
    from handumi.robots.umi_ik import RelativeToolIk
    from handumi.visualize.robot import RerunRobotView, validate_rerun_api

    # 在覆盖输出目录之前确认实际安装的 Rerun API 可用，而不是等到
    # preview.start() 才发现缺包或版本不兼容。
    validate_rerun_api()
    runtime = load_embodiment(args.robot)
    offline_solver = runtime.solver_cls(config=runtime.config.ik_weights)
    live_controller = RelativeToolIk.from_runtime(
        runtime,
        tool_calibration=tool,
        thresholds=thresholds,
    )
    live_controller.warmup()
    offline_probe = RelativeToolIk.from_runtime(
        runtime,
        tool_calibration=tool,
        thresholds=thresholds,
        solver=offline_solver,
    )
    offline_probe.warmup()
    view = RerunRobotView(runtime, axis_length=args.tcp_axis_length_m)
    view.load()
    return _RobotPreviewSetup(
        runtime=runtime,
        tool=tool,
        thresholds=thresholds,
        live_controller=live_controller,
        view=view,
        offline_solver=offline_solver,
    )


class _RerunCaptureViewer:
    """把显式 RecordingStream 适配为现有预览函数使用的 rr 模块接口。

    相机和机器人共用一个记录流，但 Rerun 的时间戳是线程本地的：主线程
    记录 capture 时间轴，后台求解线程记录 robot_ik 时间轴，互不污染。
    """

    def __init__(self, stream, rr) -> None:
        """包装显式记录流；相机与机器人共享同一 Rerun 记录。"""
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

    def disconnect(self) -> None:
        """刷新并关闭共享记录流。"""
        try:
            self.stream.flush()
        finally:
            self.stream.disconnect()


def _start_robot_preview(setup, viewer):
    """在现有采集窗口中绑定机器人 3D 场景并启动后台求解线程。"""
    from handumi.scripts._umi_robot_preview import RobotIkPreview

    if viewer is None or not hasattr(viewer, "stream"):
        raise RuntimeError("机器人预览需要现有采集 Rerun 窗口")
    setup.view.bind_stream(viewer.stream, send_blueprint=False)
    preview = RobotIkPreview(setup.live_controller, setup.view)
    preview.start()
    return preview


def _create_robot_ik_recorder(root: Path, setup):
    """构造逐帧补算记录器；每集新控制器共享已预热的离线 solver。"""
    from handumi.dataset.robot_ik import RobotIkEpisodeRecorder
    from handumi.robots.umi_ik import RelativeToolIk

    def factory():
        return RelativeToolIk.from_runtime(
            setup.runtime,
            tool_calibration=setup.tool,
            thresholds=setup.thresholds,
            solver=setup.offline_solver,
        )

    return RobotIkEpisodeRecorder(root, factory)


def _start_camera_viewer(args, session):
    """启动采集 Rerun 窗口；机器人预览共享同一显式记录流。"""
    if args.no_preview:
        return None
    import rerun as rr
    import rerun.blueprint as rrb

    stream = rr.RecordingStream("handumi_umi")
    stream.spawn()
    viewer = _RerunCaptureViewer(stream, rr)
    if getattr(args, "robot_preview", False):
        from handumi.visualize.robot import capture_blueprint

        stream.send_blueprint(
            capture_blueprint(rrb),
            make_active=True,
            make_default=True,
        )
    init_tracking_preview(viewer, session)
    return viewer


def _open_recording_dataset(args, snapshot, recording_metadata):
    """创建或续录 LeRobot 数据集；测试可替换此函数。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.resume:
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.output_dir)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output_dir,
            fps=30,
            robot_type="handumi_raw",
            features=features(snapshot),
            use_videos=True,
            image_writer_threads=4,
            vcodec="h264",
        )
    dataset.meta.info["handumi"] = recording_metadata
    return dataset


def _print_robot_ik_progress(done: int, total: int) -> None:
    """在终端显示逐帧补算进度，避免长时间保存看起来停住。"""
    interval = max(1, total // 10)
    if done == total or done % interval == 0:
        print(f"  robot_ik 补算 {done}/{total}", end="\r", flush=True)


def _warn_incomplete_robot_ik(root: Path) -> None:
    """启动时报告遗留暂存/恢复文件；只读，不补算历史 episode。"""
    from handumi.dataset.robot_ik import find_incomplete_records

    for record in find_incomplete_records(root):
        print(
            f"警告：发现未完成的 robot_ik 记录 ({record.kind}): "
            f"{record.path} - {record.detail}",
            flush=True,
        )


def main(argv=None, *, mode="record"):
    """通过共享运行时执行诊断、图像预览或 LeRobot 录制。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["vive"], default="vive")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--session-calibration", type=Path)
    parser.add_argument("--tcp-axis-length-m", type=float, default=0.08)
    parser.add_argument("--rig-config", type=Path, default=Path("configs/rig.yaml"))
    parser.add_argument("--robot", choices=["rm75b"], default="rm75b")
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
    parser.add_argument("--no-preview", action="store_true")
    if mode == "record":
        parser.add_argument(
            "--overwrite", type=_overwrite_arg, default=True, metavar="{true,false}",
            help="默认 true：删除整个已有输出目录；false：目录存在时报错。--resume 忽略此开关并追加。",
        )
        parser.add_argument("--dry-run", action="store_true", help="仅展示录制计划，不修改目录或连接硬件。")
        parser.add_argument(
            "--robot-preview",
            action="store_true",
            help="启用 RM75B IK 预览和逐帧附加记录；仅 Vive 录制可用。",
        )
        parser.add_argument(
            "--robot-tool-calibration",
            type=Path,
            help="法兰 Link7 中的工具 TCP 标定 YAML（T_end_tcp）。",
        )
        parser.add_argument(
            "--robot-position-tolerance-m",
            type=_positive_finite_float,
            default=None,
            help="机器人 IK 有效位置误差上限（米，默认 0.01）。",
        )
        parser.add_argument(
            "--robot-orientation-tolerance-deg",
            type=_positive_finite_float,
            default=None,
            help="机器人 IK 有效姿态误差上限（度，默认 5）。",
        )
    args = parser.parse_args(argv)
    if not np.isfinite(args.tcp_axis_length_m) or args.tcp_axis_length_m <= 0:
        parser.error("--tcp-axis-length-m 必须为正的有限数值")
    if mode == "record":
        _validate_robot_options(parser, args)
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
    robot_tool = None
    robot_thresholds = None
    robot_setup = None
    if mode == "record" and args.robot_preview:
        try:
            robot_tool, robot_thresholds = _validate_robot_calibration(args)
            if not args.dry_run:
                robot_setup = _prepare_robot_preview(
                    args, robot_tool, robot_thresholds
                )
        except Exception as exc:
            parser.error(f"机器人预览预检失败：{exc}")
    if mode == "record":
        _check_output_directory(args)
        operation = "resume" if args.resume else (
            "overwrite entire directory" if args.output_dir.exists() else "new dataset"
        )
        print(f"Recording plan: {args.output_dir} ({operation})")
        if args.robot_preview:
            print(
                "Robot preview: "
                f"tool={args.robot_tool_calibration}, "
                f"pos<={robot_thresholds.max_position_error_m:.4f} m, "
                f"rot<={robot_thresholds.max_orientation_error_deg:.3f} deg"
            )
        if args.dry_run:
            return
        _prepare_output_directory(args)
        if args.robot_preview:
            _warn_incomplete_robot_ik(args.output_dir)
    dataset = None
    recording = False
    previous_sequence = None
    camera_stalled = False  # 仅在停帧和恢复时提示，避免终端刷屏。
    saved = 0
    viewer = None
    preview = None
    robot_ik_recorder = None
    current_episode_index = None
    robot_ik_cache_failed = False
    robot_ik_publish_failed = False
    robot_ik_complete = 0
    robot_ik_incomplete = 0
    next_preview = 0.0  # 限制显示频率，降低预览传输压力。

    def stop_robot_preview(reason):
        """预览失败只停止显示，不中断原始录制。"""
        nonlocal preview
        if preview is None:
            return
        print(f"警告：机器人预览已停止（{reason}）；原始录制继续。", flush=True)
        try:
            preview.close()
        except Exception as exc:
            print(f"警告：机器人预览关闭失败：{exc}", flush=True)
        preview = None

    try:
        viewer = _start_camera_viewer(args, session)
        if robot_setup is not None:
            preview = _start_robot_preview(robot_setup, viewer)
            if not preview.healthy:
                raise RuntimeError("机器人预览线程未正常启动")
            robot_ik_recorder = _create_robot_ik_recorder(
                args.output_dir, robot_setup
            )
            print("机器人已就绪：按空格开始录制时启动实时 IK。")
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
            print("空格开始/保存；Esc 取消/退出；Ctrl+C 退出。无效帧立即丢弃当前段。")
            deadline = time.monotonic()
            readiness_deadline = deadline + 10
            while True:
                now = time.monotonic_ns()
                image, tracking, gripper, target, healthy = devices.sample(now)
                tracked = bool(getattr(tracking, "right_tracked", False)) and bool(
                    getattr(tracking, "right_pose_valid", True)
                )
                tcp_pose = getattr(tracking, "right_tcp_pose", None) if tracked else None
                # 附加记录和预览必须使用实际保存图像的采集时间，而不是
                # NativeUmi 的计划同步目标 target；二者允许存在同步偏差。
                frame_time_ns = int(image.capture_time_ns)
                camera_age_s = (time.monotonic_ns() - image.capture_time_ns) / 1e9
                stalled = camera_age_s > 1.0
                if stalled and not camera_stalled:
                    print(
                        f"相机画面已停更 {camera_age_s:.1f} 秒，当前显示缓存帧；"
                        "请检查相机 USB 连接及采集线程错误。",
                        flush=True,
                    )
                elif camera_stalled and not stalled:
                    print("相机画面已恢复更新。", flush=True)
                camera_stalled = stalled
                if recording and image.sequence == previous_sequence:
                    healthy = False
                was_recording = recording
                key = read_key()
                recording, action = transition(recording, key, healthy)
                if mode != "record":
                    recording = False
                if action == "exit":
                    if dataset is not None and was_recording:
                        dataset.clear_episode_buffer()
                    if was_recording and robot_ik_recorder is not None:
                        robot_ik_recorder.cancel_episode()
                    if was_recording and preview is not None:
                        preview.cancel_episode()
                    break

                if dataset is not None:
                    if action == "discard":
                        dataset.clear_episode_buffer()
                        if robot_ik_recorder is not None:
                            robot_ik_recorder.cancel_episode()
                        if preview is not None:
                            # 使在途结果失效，等待下次录制再建立起点。
                            preview.cancel_episode()
                            preview.resume()
                        print("当前段已丢弃：取消或传感器/同步无效；按空格重新开始。")
                    elif action == "save":
                        if preview is not None:
                            if not preview.pause(timeout_s=2.0):
                                print(
                                    "警告：机器人预览未在 2 秒内暂停；"
                                    "保存仍使用独立 solver 继续。",
                                    flush=True,
                                )
                        robot_ik_publish_failed = False
                        episode_index = (
                            current_episode_index
                            if current_episode_index is not None
                            else int(dataset.num_episodes)
                        )
                        prepared = None
                        if robot_ik_recorder is not None and not robot_ik_cache_failed:
                            try:
                                prepared = robot_ik_recorder.prepare_episode(
                                    episode_index,
                                    progress=_print_robot_ik_progress,
                                )
                                print()
                            except Exception as exc:
                                print(
                                    "警告：robot_ik 逐帧补算失败，仍保存原始 "
                                    f"episode：{exc}",
                                    flush=True,
                                )
                                try:
                                    robot_ik_recorder.cancel_episode()
                                except Exception:
                                    pass
                        elif robot_ik_recorder is not None:
                            # 缓存期间就失败：本段没有完整逐帧数据，不能补算。
                            try:
                                robot_ik_recorder.cancel_episode()
                            except Exception:
                                pass

                        try:
                            dataset.save_episode()
                        except BaseException:
                            if prepared is not None and robot_ik_recorder is not None:
                                robot_ik_recorder.discard_prepared()
                            elif robot_ik_recorder is not None:
                                robot_ik_recorder.cancel_episode()
                            raise

                        saved += 1
                        published = False
                        if prepared is not None and robot_ik_recorder is not None:
                            from handumi.dataset.robot_ik import RobotIkPublishError

                            try:
                                published_paths = robot_ik_recorder.publish_episode()
                                published = True
                                print(
                                    "robot_ik 附加记录已发布: "
                                    f"{published_paths.npz_path.name}, "
                                    f"{published_paths.json_path.name}",
                                    flush=True,
                                )
                            except RobotIkPublishError as exc:
                                robot_ik_publish_failed = True
                                recovery = exc.recovery_path or exc.recovery.get(
                                    "recovery_file"
                                )
                                print(
                                    "警告：原始 episode 已保存，但 robot_ik 附加"
                                    f"记录发布失败；恢复文件保留: {recovery or '暂存目录'}",
                                    flush=True,
                                )
                                try:
                                    robot_ik_recorder.release_prepared()
                                except Exception:
                                    pass
                            except Exception as exc:
                                robot_ik_publish_failed = True
                                print(
                                    "警告：原始 episode 已保存，但 robot_ik 附加"
                                    f"记录发布异常：{exc}",
                                    flush=True,
                                )
                                try:
                                    robot_ik_recorder.release_prepared()
                                except Exception:
                                    pass
                        elif prepared is not None:
                            print(
                                "警告：原始 episode 已保存，但缺少附加记录记录器。",
                                flush=True,
                            )

                        record_complete = prepared is not None and published
                        if args.robot_preview:
                            if record_complete:
                                robot_ik_complete += 1
                            else:
                                robot_ik_incomplete += 1
                        if preview is not None:
                            try:
                                preview.cancel_episode()
                                preview.resume()
                            except Exception as exc:
                                stop_robot_preview(str(exc))
                        # 保存可能耗时数秒；重新对齐采集节拍，避免追赶忙循环。
                        deadline = time.monotonic()
                        next_preview = 0.0
                        if args.robot_preview and not record_complete:
                            print(
                                f"已保存 {saved} 段（原始数据完整；本段 robot_ik "
                                "附加记录未完成）"
                            )
                        elif robot_ik_publish_failed:
                            print(
                                f"已保存 {saved} 段（robot_ik 附加记录未完整发布）"
                            )
                        else:
                            print(f"已保存 {saved} 段")
                        if args.num_episodes > 0 and saved >= args.num_episodes:
                            break
                    elif recording:
                        frame = assemble_frame(
                            image,
                            tracking,
                            gripper,
                            target,
                            time.monotonic_ns(),
                            args.task,
                        )
                        dataset.add_frame(frame)
                        current_episode_index = (
                            int(dataset.num_episodes)
                            if current_episode_index is None
                            else current_episode_index
                        )
                        if action == "start":
                            current_episode_index = int(dataset.num_episodes)
                            robot_ik_cache_failed = False
                            if robot_ik_recorder is not None:
                                try:
                                    robot_ik_recorder.begin_episode()
                                except Exception as exc:
                                    print(
                                        "警告：robot_ik 片段缓存未能开始，本段"
                                        f"不生成附加记录：{exc}",
                                        flush=True,
                                    )
                                    robot_ik_recorder = None
                                    robot_ik_cache_failed = True
                            if (
                                preview is not None
                                and preview.healthy
                            ):
                                try:
                                    preview.begin_episode(tcp_pose, frame_time_ns)
                                except Exception as exc:
                                    stop_robot_preview(str(exc))
                        if robot_ik_recorder is not None:
                            try:
                                robot_ik_recorder.add_frame(
                                    frame_time_ns / 1e9,
                                    tcp_pose,
                                    tracked=tracked,
                                )
                            except Exception as exc:
                                robot_ik_cache_failed = True
                                print(
                                    "警告：robot_ik 逐帧缓存失败，本段将只保存"
                                    f"原始 episode：{exc}",
                                    flush=True,
                                )
                        if (
                            preview is not None
                            and preview.healthy
                            and action != "start"
                        ):
                            try:
                                preview.submit(frame_time_ns, tcp_pose, tracked=tracked)
                            except Exception as exc:
                                stop_robot_preview(str(exc))
                previous_sequence = image.sequence
                if viewer is not None and time.monotonic() >= next_preview:
                    log_live_preview(
                        viewer, image, tracking, gripper, healthy, camera_age_s,
                        args.tcp_axis_length_m,
                    )
                    next_preview = time.monotonic() + 1 / 15
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
                deadline += 1 / 30
                time.sleep(max(0, deadline - time.monotonic()))
    finally:
        if preview is not None:
            try:
                preview.close()
            except Exception as exc:
                print(f"警告：机器人预览关闭失败：{exc}", flush=True)
            preview = None
        if getattr(args, "robot_preview", False):
            print(
                "robot_ik 附加记录摘要："
                f"完整 {robot_ik_complete} 段，未完成 {robot_ik_incomplete} 段",
                flush=True,
            )
        if dataset is not None:
            if recording:
                dataset.clear_episode_buffer()
            if robot_ik_recorder is not None:
                if recording:
                    robot_ik_recorder.cancel_episode()
                elif robot_ik_recorder.prepared is not None:
                    # 原始 episode 已经或可能已保存；保留暂存/恢复文件，
                    # 让启动检查或人工恢复处理。finally 不得删除它们。
                    robot_ik_recorder.release_prepared()
            try:
                dataset.finalize()
                path = args.output_dir / "meta/info.json"
                info = json.loads(path.read_text())
                info["handumi"] = recording_metadata
                path.write_text(json.dumps(info, indent=2) + "\n")
            finally:
                if viewer is not None:
                    viewer.disconnect()
        elif viewer is not None:
            viewer.disconnect()
