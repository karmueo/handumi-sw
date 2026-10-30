"""保存桌面标定的逐帧诊断、观测和结果，供失败后离线分析。"""

from __future__ import annotations

import json
import traceback
from collections import Counter
from datetime import UTC, datetime
from uuid import uuid4

import cv2
import numpy as np
import yaml

from handumi.calibration.umi_table import LIMITS
from handumi.scripts._umi_table_sampling import TABLE_TARGET_VIEWS

IMAGE_POLICY = {  # 低标签帧原图保存策略，限制采集时的磁盘和编码开销。
    "max_tag_count": 20, "max_images": 150, "min_interval_ms": 250,
    "jpeg_quality": 95,
}


class TableDiagnostics:
    """每次采集创建独立目录；逐帧落盘，正常退出或异常时补齐汇总。"""

    def __init__(self, args, snapshot, session):
        """在打开设备前保存配置快照；目录不可写时立即报错。"""
        started = datetime.now(UTC)
        run_id = f"{started:%Y%m%dT%H%M%S.%fZ}-{uuid4().hex[:8]}"
        self.directory = args.session_calibration.parent / "table-diagnostics" / run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.data = None  # 指向采集中的观测，确保 Ctrl+C 后仍可保存已有数据。
        self.frames = []  # 汇总仅保留轻量字段；角点和位姿写入逐帧文件。
        self.saved_images = 0  # 已保存的低标签原图数量。
        self.last_image_ns = None  # 上一张保存原图的相机采集时间。
        self.report = {
            "schema": "umi_table_diagnostics_v1",
            "started_at": started.isoformat(),
            "command": args.command,
            "duration_s": (
                None if args.command == "table" else getattr(args, "duration", None)
            ),
            "target_views": (
                TABLE_TARGET_VIEWS if args.command == "table" else None
            ),
            "end_reason": None,
            "preview": not args.no_preview,
            "board_height_m": (
                args.board_height_m if session is None else session["board_height_m"]
            ),
            "session_path": str(args.session_calibration.resolve()),
            "limits": dict(LIMITS),
            "capture_limits": {
                "target_delay_ms": 150, "max_frame_age_ms": 350,
                "max_pairing_error_ms": 60,
            },
            "image_policy": dict(IMAGE_POLICY),
            "status": "collecting",
        }
        (self.directory / "calibration.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
        )
        if session is not None:
            (self.directory / "session-input.yaml").write_text(
                yaml.safe_dump(session, sort_keys=False), encoding="utf-8"
            )
        self._write_json("report.json", self.report)
        print(f"桌面诊断目录：{self.directory.resolve()}", flush=True)

    def _write_json(self, filename, value):
        """以 UTF-8 写入可读诊断文件，允许保留非有限指标用于追查数值故障。"""
        (self.directory / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_frame_image(self, frame, tag_count):
        """按策略保存低标签帧的 RGB 原图，返回诊断目录内相对路径或 None。

        仅在标签数低于阈值、未达数量上限且距上次保存足够久时写入 JPEG，
        便于离线复现检测失败而不拖慢采集。
        """
        if (
            tag_count >= IMAGE_POLICY["max_tag_count"]
            or self.saved_images >= IMAGE_POLICY["max_images"]
            or (
                self.last_image_ns is not None
                and frame.capture_time_ns - self.last_image_ns
                < IMAGE_POLICY["min_interval_ms"] * 1_000_000
            )
        ):
            return None
        relative = f"images/{int(frame.sequence):06d}.jpg"
        path = self.directory / relative
        path.parent.mkdir(exist_ok=True)
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, IMAGE_POLICY["jpeg_quality"]],
        )
        if not ok:
            raise OSError(f"无法编码诊断图像：{relative}")
        path.write_bytes(encoded.tobytes())
        self.saved_images += 1
        self.last_image_ns = frame.capture_time_ns
        return relative

    def record_frame(self, details, observation, tracking):
        """立即保存本帧判定和原始几何观测，进程意外退出时保留已完成记录。"""
        self.frames.append(details)
        record = dict(details)
        record["tracker_pose"] = (
            tracking.right_device_controller_pose.tolist()
            if tracking.right_tracked else None
        )
        record["observation"] = (
            [part.tolist() for part in observation] if observation is not None else None
        )
        with (self.directory / "frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, *, session=None, error=None):
        """保存部分或完整观测、全部指标及异常；不修改任何标定结果。"""
        if self.data is not None:
            self._write_json("observations.json", self.data)
        reasons = Counter(
            reason for frame in self.frames for reason in frame["rejected_reasons"]
        )
        self.report.update({
            "finished_at": datetime.now(UTC).isoformat(),
            "status": "passed" if error is None else (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            ),
            "processed_frames": len(self.frames),
            "accepted_frames": len((self.data or {}).get("observations", [])),
            "rejected_frames": sum(not frame["accepted"] for frame in self.frames),
            "rejected_reasons": dict(reasons),
            "saved_images": self.saved_images,
        })
        policy = (self.data or {}).get("sampling_policy") or {}
        if "target_views" in policy:
            self.report["target_views"] = policy["target_views"]
        if "completed_views" in policy:
            self.report["completed_views"] = policy["completed_views"]
        if policy.get("end_reason") is not None:
            self.report["end_reason"] = policy["end_reason"]
        elif error is not None:
            self.report["end_reason"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
        if self.frames:
            # 对比检测前后帧龄，可区分取帧延迟和检测耗时造成的拒绝。
            self.report["timing_ms"] = {
                key: dict(zip(
                    ["median", "p95", "max"],
                    np.percentile([frame[key] for frame in self.frames], [50, 95, 100]).tolist(),
                    strict=True,
                ))
                for key in ("detection_ms", "frame_age_before_ms", "frame_age_after_ms")
            }
        if error is not None:
            self.report["error"] = {
                "type": type(error).__name__, "message": str(error),
            }
            if hasattr(error, "metrics"):
                self.report["metrics"] = error.metrics
            (self.directory / "error.txt").write_text(
                "".join(traceback.format_exception(error)), encoding="utf-8"
            )
        elif session is not None:
            self.report["metrics"] = (
                session["solve_metrics"] if self.report["command"] == "table"
                else session["verification"]["metrics"]
            )
            (self.directory / "session-result.yaml").write_text(
                yaml.safe_dump(session, sort_keys=False), encoding="utf-8"
            )
        self._write_json("report.json", self.report)
        print(
            f"桌面诊断已保存：{self.directory.resolve()} "
            f"（有效观测 {self.report['accepted_frames']} 帧）",
            flush=True,
        )
