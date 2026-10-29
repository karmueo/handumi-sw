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
from pathlib import Path

import numpy as np
import yaml

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_to_dict,
)
from handumi.calibration.umi import estimator, pose7, verify
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
from handumi.tracking.vive import ViveTrackingProvider

DEFAULT_CALIBRATION = Path("configs/calibration/umi/local/calibration.yaml")


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

    def __init__(self, snapshot, *, tracker=None, camera=None):
        """注入设备便于测试；默认按已验证快照创建原生设备。"""
        verify(snapshot, require_time=False)
        self.snapshot = snapshot
        self.offset_ns = round((snapshot.get("time") or {}).get("offset_s", 0) * 1e9)
        identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        calibration = ControllerTcpCalibration(identity, pose7(snapshot["tracker_tcp"]))
        self.tracker = tracker or ViveTrackingProvider(snapshot["serial"], calibration)
        width, height = snapshot["resolution"]
        self.camera = camera or OpenCVCameraDevice(
            snapshot["camera_device"],
            snapshot["fps"],
            width,
            height,
            snapshot["fourcc"],
        )
        self.gripper = VisualGripperProvider(estimator(snapshot))

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


def metadata(snapshot):
    """生成独立 UMI 身份和原始数据契约，不依赖目标机器人夹爪。"""
    identity = np.array([0, 0, 0, 0, 0, 0, 1])
    tcp = calibration_to_dict(left=identity, right=pose7(snapshot["tracker_tcp"]))[
        "calibration"
    ]
    return {
        "recording_device": "vive",
        "active_sides": ["right"],
        "tracking_schema": HANDUMI_TRACKING_SCHEMA,
        "capture_schema": "synchronized_sources_v2",
        "state_semantics": HANDUMI_STATE_SEMANTICS,
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



def validate_resume_target(root: Path, expected_metadata: dict) -> None:
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
    if info.get("handumi") != expected_metadata:
        raise ValueError("续录的设备、数据契约或标定不一致")


def log_live_preview(viewer, image, tracking, gripper, healthy, camera_age_s):
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
    viewer.log("tracking/tcp", viewer.Points3D([tracking.right_tcp_pose[:3]]))


def main(argv=None, *, mode="record"):
    """通过共享运行时执行诊断、图像预览或 LeRobot 录制。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["vive"], default="vive")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--rig-config", type=Path, default=Path("configs/rig.yaml"))
    parser.add_argument("--robot", choices=["rm75b"], default="rm75b")
    parser.add_argument("--side", choices=["right"], default="right")
    parser.add_argument("--fps", type=int, choices=[30], default=30)
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
    args = parser.parse_args(argv)
    if args.calibration is None:
        rig = (
            yaml.safe_load(args.rig_config.read_text())
            if args.rig_config.exists()
            else {}
        )
        args.calibration = Path(
            (rig or {}).get("umi", {}).get("calibration", DEFAULT_CALIBRATION)
        )
    snapshot = verify(
        yaml.safe_load(args.calibration.read_text()), require_time=mode != "visualize"
    )
    if mode == "record" and args.output_dir is None:
        parser.error("录制需要 --output-dir")
    if mode == "record" and args.resume:
        try:
            validate_resume_target(args.output_dir, metadata(snapshot))
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
    previous_sequence = None
    camera_stalled = False  # 仅在停帧和恢复时提示，避免终端刷屏。
    saved = 0
    viewer = None
    next_preview = 0.0  # 限制显示频率，降低预览传输压力。
    if not args.no_preview:
        import rerun as rr

        rr.init("handumi_umi", spawn=True)
        viewer = rr
    try:
        with (
            NativeUmi(snapshot) as devices,
            nullcontext(lambda: "") if mode == "doctor" else keyboard() as read_key,
        ):
            if mode == "record":
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                if args.resume:
                    dataset = LeRobotDataset.resume(
                        repo_id=args.repo_id, root=args.output_dir
                    )
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
                dataset.meta.info["handumi"] = metadata(snapshot)
            print("空格开始/保存；Esc 取消/退出；Ctrl+C 退出。无效帧立即丢弃当前段。")
            deadline = time.monotonic()
            readiness_deadline = deadline + 10
            while True:
                now = time.monotonic_ns()
                image, tracking, gripper, target, healthy = devices.sample(now)
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
                key = read_key()
                recording, action = transition(recording, key, healthy)
                if mode != "record":
                    recording = False
                if action == "exit":
                    if dataset is not None:
                        dataset.clear_episode_buffer()
                    break
                if dataset is not None:
                    if action == "discard":
                        dataset.clear_episode_buffer()
                        print("当前段已丢弃：取消或传感器/同步无效；按空格重新开始。")
                    elif action == "save":
                        dataset.save_episode()
                        saved += 1
                        print(f"已保存 {saved} 段")
                        if args.num_episodes > 0 and saved >= args.num_episodes:
                            break
                    elif recording:
                        dataset.add_frame(
                            assemble_frame(
                                image,
                                tracking,
                                gripper,
                                target,
                                time.monotonic_ns(),
                                args.task,
                            )
                        )
                previous_sequence = image.sequence
                if viewer is not None and time.monotonic() >= next_preview:
                    log_live_preview(
                        viewer, image, tracking, gripper, healthy, camera_age_s
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
        if dataset is not None:
            if recording:
                dataset.clear_episode_buffer()
            dataset.finalize()
            path = args.output_dir / "meta/info.json"
            info = json.loads(path.read_text())
            info["handumi"] = metadata(snapshot)
            path.write_text(json.dumps(info, indent=2) + "\n")
        if viewer is not None:
            viewer.disconnect()
