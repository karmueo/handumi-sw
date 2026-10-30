"""提供 UMI 几何导入、时间与桌面标定，并保存桌面采集诊断。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from handumi.calibration.umi import digest, import_calibration, verify
from handumi.calibration.umi_table import board_from_table, solve_table, verify_table
from handumi.calibration.umi_time import detect_board_details, solve_time
from handumi.scripts._umi_session import calibration_paths, load_session, save_session
from handumi.scripts._umi_table_diagnostics import TableDiagnostics
from handumi.scripts._umi_table_sampling import (
    CURRENT_VIEW_COMPLETE,
    STATIONARY_POSITION_M,
    STATIONARY_ROTATION_DEG,
    STATIONARY_WINDOW_S,
    TABLE_TARGET_VIEWS,
    StationaryViewCapture,
    TrackerJournal,
    image_quality,
    stationary,
)
from handumi.scripts.umi_capture import NativeUmi


def board_preview(rgb, markers, accepted, frame_count, remaining, *, reasons=None):
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
    if reasons:
        status = ", ".join(reasons)
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


def table_preview(rgb, markers, view_capture):
    """在图像副本上绘制标定板轮廓、标签及简化英文状态，不写入原始图像。

    Args:
        rgb: 相机采集的 RGB 原图。
        markers: 当前帧检测到的标签 ID 与角点。
        view_capture: 维护当前视角和完成进度的停稳采集状态。

    Returns:
        带标注的新图像；原图保持不变。
    """
    canvas = rgb.copy()
    complete = view_capture.status == CURRENT_VIEW_COMPLETE
    color = (60, 255, 60) if complete else (255, 210, 0)
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
    lines = (view_capture.status, view_capture.progress_text)
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


def collect_table(
    snapshot,
    duration=None,
    *,
    preview=True,
    diagnostics=None,
    warmup_s=5.0,
    require_stationary=False,
    target_views=None,
):
    """采集同步 Tracker/AprilGrid 对；目标视角收满后自动结束。

    Args:
        snapshot: 已验证的 UMI 标定快照。
        duration: 计时采集时长；target_views 非空时忽略，可为 None。
        preview: 是否启动 Rerun 实时预览。
        diagnostics: 可选的逐帧诊断记录器。
        warmup_s: 开始计数前的预热秒数。
        require_stationary: 是否按停稳视角收帧。
        target_views: 可选的目标独立视角数；仅 table 传入 15。

    Returns:
        包含观测、Tracker 原始时间线和采样策略的采集数据。
    """
    if target_views is not None and (
        not isinstance(target_views, int) or target_views < 1
    ):
        raise ValueError("目标视角数必须为正整数")
    if target_views is not None and not require_stationary:
        raise ValueError("目标视角数需要启用停稳采集")
    if target_views is None and (
        duration is None or not np.isfinite(duration) or duration < 0
    ):
        raise ValueError("计时采集需要非负有限时长")
    fx, fy, cx, cy = snapshot["intrinsics"]
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    distortion = np.array(snapshot["distortion"], float)
    data = {
        "camera_times": [], "observations": [], "tracker_poses": [],
        "geometry_digest": snapshot["geometry_digest"],
        "time_digest": digest(snapshot["time"]),
    }
    if diagnostics is not None:
        diagnostics.data = data
    data["sampling_policy"] = {
        "warmup_s": warmup_s, "require_stationary": require_stationary,
        "target_views": target_views, "end_reason": None,
        "stationary_window_s": STATIONARY_WINDOW_S,
        "position_span_m": STATIONARY_POSITION_M,
        "rotation_span_deg": STATIONARY_ROTATION_DEG, "single_frame_median_px": 1.0,
        "single_frame_p95_px": 2.0,
    }
    view_capture = (
        (
            StationaryViewCapture(target_views=target_views)
            if target_views is not None
            else StationaryViewCapture()
        )
        if require_stationary else None
    )
    if view_capture is not None:
        data["sampling_policy"]["frames_per_view"] = view_capture.frames_per_view
    journal = TrackerJournal(data, diagnostics)
    viewer = None
    previous = None
    if preview:
        import rerun as rr

        rr.init("handumi_umi_table_calibration", spawn=True)
        if require_stationary:
            import rerun.blueprint as rrb

            rr.send_blueprint(rrb.Blueprint(rrb.Vertical(
                rrb.TextDocumentView(origin="/calibration/status", name="Table calibration status"),
                rrb.Spatial2DView(origin="/calibration/camera", name="Calibration camera"),
                row_shares=[1, 3],
            )), make_active=True, make_default=True)
            initial_status = (
                f"# CAPTURING\n\n{view_capture.progress_text}"
                if target_views is not None
                else "# PREPARING STATIONARY CAPTURE\n\n"
                "After warmup, hold roughly steady until VIEW COMPLETE appears."
            )
            rr.log("calibration/status", rr.TextDocument(
                initial_status,
                media_type="text/markdown",
            ))
        viewer = rr
    try:
        with NativeUmi(snapshot) as devices:
            ready = time.monotonic() + warmup_s
            end = None if target_views is not None else ready + duration
            if target_views is not None:
                print(
                    f"预热 {warmup_s:g} 秒；随后换视角并保持大致稳定，"
                    f"每个视角收满 {view_capture.frames_per_view} 帧合格观测；"
                    f"完成 {target_views} 个独立视角后自动结束，不使用 --duration。"
                )
            elif require_stationary:
                print(f"预热 {warmup_s:g} 秒；随后换视角并保持大致稳定，轻微手抖允许；每个视角收满 15 帧后提示完成。")
            else:
                print(f"预热 {warmup_s:g} 秒；随后手持缓慢移动相机，连续采样，无需停稳。")
            print("将 AprilGrid 平放并固定：IDs 0–5 靠近自己且从左到右递增。")
            if target_views is not None:
                print(
                    f"依次覆盖 {target_views} 个不同视角；相对所有已完成视角"
                    "至少移动 5 cm 或旋转 10°。"
                )
            else:
                print("缓慢移动相机覆盖至少四个视角，间距至少 5 cm 或角度至少 10°。")
            while end is None or time.monotonic() < end:
                history = journal.drain(devices.tracker)
                if devices.tracker.error is not None:
                    raise RuntimeError("Vive 采集失败") from devices.tracker.error
                target = time.monotonic_ns() - 150_000_000
                frame = devices.camera.sample_at(target - devices.offset_ns)
                if frame.sequence == previous:
                    time.sleep(0.005)
                    continue
                previous = frame.sequence
                corrected = frame.capture_time_ns + devices.offset_ns
                tracking = devices.tracker.sample_at(corrected)
                detection_started = time.monotonic_ns()  # 检测前帧龄不含角点计算耗时。
                detection = {}  # 检测器原始标签数和被拒候选数，仅写入诊断。
                found, markers = detect_board_details(
                    frame.image, matrix, distortion, stats=detection
                )
                checked_at = time.monotonic_ns()
                frame_age = checked_at - frame.capture_time_ns
                pairing_error = corrected - target
                rejected_reasons = []  # 保留所有拒绝原因，不改变原有验收条件。
                if found is None:
                    rejected_reasons.append("board_not_found_or_pnp_failed")
                if not tracking.right_tracked:
                    rejected_reasons.append("tracker_unavailable")
                if not 0 <= frame_age <= 350_000_000:
                    rejected_reasons.append("frame_age_out_of_range")
                if abs(pairing_error) > 60_000_000:
                    rejected_reasons.append("pairing_error_too_large")
                quality = image_quality(found, matrix, distortion)
                if quality is not None and (not np.isfinite(quality).all() or quality[0] > 1 or quality[1] > 2):
                    rejected_reasons.append("single_frame_quality_failed")
                if warmup_s and time.monotonic() < ready:
                    rejected_reasons.append("warming_up")
                accepted = not rejected_reasons
                if view_capture is not None:
                    stable = stationary(history, corrected)
                    accepted = view_capture.update(
                        tracking.right_device_controller_pose, stable, rejected_reasons
                    )
                    if not accepted and not rejected_reasons:
                        rejected_reasons.append("view_complete" if stable else "not_stationary")
                if accepted:
                    data["camera_times"].append(frame.capture_time_ns / 1e9)
                    data["observations"].append([part.tolist() for part in found])
                    data["tracker_poses"].append(tracking.right_device_controller_pose.tolist())
                if diagnostics is not None:
                    image_path = diagnostics.save_frame_image(frame, len(markers))
                    diagnostics.record_frame({
                        "sequence": int(frame.sequence),
                        "capture_time_ns": int(frame.capture_time_ns),
                        "corrected_time_ns": int(corrected),
                        "target_time_ns": int(target),
                        "tag_ids": [int(tag_id) for tag_id, _ in markers],
                        "tag_count": len(markers),
                        "aruco_tag_count": detection.get("aruco_tag_count"),
                        "rejected_candidate_count": detection.get("rejected_candidate_count"),
                        "image": image_path,
                        "single_frame_error_px": quality,
                        "tracker_valid": bool(tracking.right_tracked),
                        "accepted": accepted,
                        "rejected_reasons": rejected_reasons,
                        "detection_ms": (checked_at - detection_started) / 1e6,
                        "frame_age_before_ms": (detection_started - frame.capture_time_ns) / 1e6,
                        "frame_age_after_ms": frame_age / 1e6,
                        "pairing_error_ms": pairing_error / 1e6,
                    }, found, tracking)
                if viewer is not None:
                    if target_views is not None:
                        viewer.log("calibration/status", viewer.TextDocument(
                            view_capture.markdown(),
                            media_type="text/markdown",
                        ))
                        viewer.log("calibration/camera", viewer.Image(table_preview(
                            frame.image, markers, view_capture,
                        )).compress(jpeg_quality=75))
                    else:
                        if view_capture is not None:
                            viewer.log("calibration/status", viewer.TextDocument(
                                view_capture.markdown(end - time.monotonic()),
                                media_type="text/markdown",
                            ))
                        viewer.log("calibration/camera", viewer.Image(board_preview(
                            frame.image, markers, accepted, len(data["observations"]),
                            end - time.monotonic(),
                            reasons=rejected_reasons,
                        )).compress(jpeg_quality=75))
                # 最后一帧已写入观测、诊断和 15/15 预览后再退出采集。
                if view_capture is not None and view_capture.target_reached:
                    data["sampling_policy"]["end_reason"] = "target_views_reached"
                    break
            else:
                data["sampling_policy"]["end_reason"] = "duration_elapsed"
            journal.drain(devices.tracker)
    finally:
        if view_capture is not None:
            data["sampling_policy"]["completed_views"] = len(view_capture.completed)
        if "devices" in locals():
            journal.drain(devices.tracker)
        if viewer is not None:
            viewer.disconnect()
    return data


def table_command(args, snapshot):
    """求解新会话或独立复验；验证中断/失败立即使已有会话不可用于录制。"""
    if args.command == "table":
        if args.board_height_m is None:
            raise ValueError("table 必须显式指定 --board-height-m（板面距真实桌面的米制高度）")
        board_from_table(args.board_height_m)
        session = None
    else:
        session = load_session(args.session_calibration, snapshot, require_verified=False)
        session["verification"] = {"accepted": False, "reason": "独立验证尚未完成"}
        save_session(args.session_calibration, session)
    print(f"实际标定文件：{getattr(args, 'calibration', None)}；桌面文件：{args.session_calibration.resolve()}")
    print(f"相机：{snapshot['camera_device']}；Tracker：{snapshot['serial']}；分辨率：{snapshot['resolution']}；帧率：{snapshot['fps']}")
    print(f"实际时间偏移：{snapshot['time']['offset_s'] * 1000:.6f} ms；采用方式：{snapshot['time'].get('acceptance', 'automatic')}")
    diagnostics = TableDiagnostics(args, snapshot, session)
    error = None  # 保留原始异常，避免诊断写入失败覆盖真正的标定错误。
    try:
        data = collect_table(
            snapshot,
            None if args.command == "table" else getattr(args, "duration", None),
            preview=not args.no_preview,
            diagnostics=diagnostics,
            require_stationary=(
                True if args.command == "table"
                else bool(getattr(args, "stationary", False))
            ),
            target_views=(
                TABLE_TARGET_VIEWS if args.command == "table" else None
            ),
        )
        diagnostics.data = data
        if args.command == "table":
            session = solve_table(data, snapshot, args.board_height_m)
        else:
            session["verification"] = verify_table(data, snapshot, session)
        save_session(args.session_calibration, session)
    except BaseException as exc:
        error = exc
        if args.command == "verify-table":
            session["verification"] = {"accepted": False, "reason": str(exc) or type(exc).__name__}
            save_session(args.session_calibration, session)
        raise
    finally:
        try:
            diagnostics.finish(session=session, error=error)
            capture_output = getattr(args, "capture_output", None)
            if capture_output is not None and diagnostics.data is not None:
                capture_output.parent.mkdir(parents=True, exist_ok=True)
                capture_output.write_text(json.dumps(diagnostics.data), encoding="utf-8")
        except OSError as diagnostic_error:
            message = f"桌面诊断保存不完整：{diagnostic_error}"
            if error is not None:
                error.add_note(message)
            print(message, flush=True)
    print(f"桌面会话已保存：{args.session_calibration}")
    if args.command == "table":
        print("请保持标定板固定，再运行 handumi calibrate umi verify-table 独立验证。")
    else:
        print("桌面独立验证通过；移走板后可录制。OpenVR 原点变化时需重新放板复验。")


def main(argv=None):
    """命令失败不覆盖原标定；仅通过质量门后保存新时间结果。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "table：预热 5 秒后按停稳视角采集，每个视角 15 帧，"
            "完成 15 个独立视角后自动结束并忽略 --duration；"
            "verify-table 默认仍按 --duration 连续采集，可加 --stationary 启用停稳收帧。"
            "table 后必须再运行 verify-table 独立验证。"
        ),
    )
    parser.add_argument("command", choices=["import", "time", "verify", "table", "verify-table"])
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--session-calibration", type=Path)
    parser.add_argument("--rig-config", type=Path, default=Path("configs/rig.yaml"))
    parser.add_argument(
        "--board-height-m", type=float,
        help="table 必填：标签印刷平面距真实桌面的高度（米，非负有限值）",
    )
    parser.add_argument("--camera", type=Path)
    parser.add_argument("--tcp", type=Path)
    parser.add_argument("--gripper", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--camera-device")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument(
        "--duration", type=float, default=30,
        help="time/verify-table 的采集时长（秒）；table 忽略此参数",
    )
    parser.add_argument("--no-preview", action="store_true", help="采集时关闭实时预览")
    parser.add_argument(
        "--stationary", action="store_true",
        help="兼容参数：table 默认启用停稳采集；verify-table 可显式启用",
    )
    args = parser.parse_args(argv)
    if args.stationary and args.command not in {"table", "verify-table"}:
        parser.error("--stationary 仅用于 table 和 verify-table")
    args.calibration, args.session_calibration = calibration_paths(args)
    if args.command != "table" and (
        not np.isfinite(args.duration) or args.duration < 4
    ):
        parser.error("采集时长必须为至少 4 秒的有限数值")
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
            require_time=args.command in {"verify", "table", "verify-table"},
        )
        if args.command in {"table", "verify-table"}:
            if args.input:
                parser.error("桌面求解和独立验证必须重新采集，不能使用 --input")
            table_command(args, snapshot)
            return
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
