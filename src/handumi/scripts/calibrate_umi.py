"""提供 UMI 几何导入、原生时间标定采集和验证命令。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from handumi.calibration.umi import import_calibration, verify
from handumi.calibration.umi_time import detect_board_details, solve_time
from handumi.scripts.umi_capture import DEFAULT_CALIBRATION, NativeUmi


def board_preview(rgb, markers, accepted, frame_count, remaining):
    """在图像副本上绘制本帧标签、观测状态和剩余时间，不修改原始采集图像。"""
    canvas = rgb.copy()
    color = (60, 255, 60) if accepted else (255, 210, 0)
    for tag_id, corners in markers:
        polygon = np.rint(corners).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        for point in polygon:
            cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(tag_id),
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    status = "OBSERVATION ACCEPTED" if accepted else "NO OBSERVATION"
    lines = (
        f"AprilGrid: {status}",
        f"Tags: {len(markers)} (need >=6)  Frames: {frame_count}  "
        f"Remaining: {max(0, remaining):.1f}s",
    )
    # OpenCV 使用英文文本，避免依赖系统中文字库。
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 58), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (8, 21 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def collect(snapshot, duration, *, preview=True):
    """采集标定观测和 Tracker 时间线，可实时预览，但不将图像写入采集文件。"""
    fx, fy, cx, cy = snapshot["intrinsics"]
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    distortion = np.array(snapshot["distortion"], float)
    times, observations, tracks = [], [], {}
    previous = None
    viewer = None
    if preview:
        import rerun as rr

        rr.init("handumi_umi_time_calibration", spawn=True)
        viewer = rr
    try:
        with NativeUmi(snapshot) as devices:
            end = time.monotonic() + duration
            print("固定 AprilGrid，保持可见并进行变速、多轴平移和旋转。")
            while time.monotonic() < end:
                image = devices.camera.sample_at()
                if image.sequence != previous:
                    previous = image.sequence
                    if image.sequence % 2 == 0:
                        found, markers = detect_board_details(
                            image.image, matrix, distortion
                        )
                        if found is not None:
                            times.append(image.capture_time_ns / 1e9)
                            observations.append([part.tolist() for part in found])
                        if viewer is not None:
                            annotated = board_preview(
                                image.image,
                                markers,
                                found is not None,
                                len(observations),
                                end - time.monotonic(),
                            )
                            viewer.set_time(
                                "capture", duration=image.capture_time_ns / 1e9
                            )
                            viewer.log(
                                "calibration/camera",
                                viewer.Image(annotated).compress(jpeg_quality=75),
                            )
                with devices.tracker._lock:
                    for sample in devices.tracker._history:
                        tracks[sample.pc_monotonic_ns] = (
                            sample.right_controller_pose.tolist(),
                            sample.right_tracked,
                        )
                time.sleep(0.005)
    finally:
        if viewer is not None:
            viewer.disconnect()
    stamps = sorted(tracks)
    return {
        "camera_times": times,
        "observations": observations,
        "tracker_times": [stamp / 1e9 for stamp in stamps],
        "tracker_poses": [tracks[stamp][0] for stamp in stamps],
        "tracker_valid": [tracks[stamp][1] for stamp in stamps],
        "geometry_digest": snapshot["geometry_digest"],
    }


def main(argv=None):
    """命令失败不覆盖原标定；仅通过质量门后保存新时间结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["import", "time", "verify"])
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--tcp", type=Path)
    parser.add_argument("--gripper", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--camera-device")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--no-preview", action="store_true", help="采集时关闭实时预览")
    args = parser.parse_args(argv)
    if args.command == "import":
        if not all(
            (args.camera, args.tcp, args.gripper, args.serial, args.camera_device)
        ):
            parser.error(
                "import 需要 --camera --tcp --gripper --serial --camera-device"
            )
        snapshot = import_calibration(
            args.camera,
            args.tcp,
            args.gripper,
            serial=args.serial,
            camera_device=args.camera_device,
        )
    else:
        snapshot = verify(
            yaml.safe_load(args.calibration.read_text()),
            require_time=args.command == "verify",
        )
        if args.command == "verify":
            print("几何与时间标定绑定检查通过；实物安装仍需按操作文档核验。")
            return
        if args.duration < 4:
            parser.error("采集至少需要 4 秒")
        data = (
            json.loads(args.input.read_text())
            if args.input
            else collect(snapshot, args.duration, preview=not args.no_preview)
        )
        if data.get("geometry_digest") != snapshot["geometry_digest"]:
            raise ValueError("标定观测与当前设备配置不一致")
        if args.capture_output:
            args.capture_output.parent.mkdir(parents=True, exist_ok=True)
            args.capture_output.write_text(json.dumps(data))
        observations = [
            tuple(np.asarray(part, float) for part in obs)
            for obs in data["observations"]
        ]
        snapshot["time"] = solve_time(
            data["camera_times"],
            observations,
            data["tracker_times"],
            data["tracker_poses"],
            data["tracker_valid"],
            snapshot,
        )
        print(json.dumps(snapshot["time"], indent=2))
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.calibration.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump(snapshot, sort_keys=False))
    temporary.replace(args.calibration)
