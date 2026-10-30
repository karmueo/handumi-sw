"""验证桌面标定诊断落盘、逐帧拒绝原因和异常时的数据保留。"""

import json
import threading
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import yaml

from handumi.calibration.umi_table import TableQualityError, check_metrics
from handumi.cameras.base import CameraSample
from handumi.scripts import _umi_table_diagnostics, calibrate_umi
from handumi.scripts._umi_table_diagnostics import TableDiagnostics


def command_args(tmp_path):
    """构造隔离于真实标定目录的命令参数。"""
    return SimpleNamespace(
        command="table", board_height_m=0.0082, duration=30, no_preview=True,
        session_calibration=tmp_path / "session.yaml", capture_output=None,
    )


def test_quality_error_keeps_all_metrics():
    """多项超限时报告所有问题，同时保留未超限项。"""
    metrics = {
        "frames": 40, "views": 4, "translation_rms_m": 0.02,
        "rotation_rms_deg": 0.1, "median_px": 3.1, "p95_px": 10.1,
    }
    with pytest.raises(TableQualityError) as caught:
        check_metrics(metrics)
    assert caught.value.metrics == metrics
    assert all(key in str(caught.value) for key in (
        "translation_rms_m", "median_px", "p95_px",
    ))


def test_failed_solve_saves_observations_snapshot_and_metrics(table_setup, tmp_path, monkeypatch):
    """真实求解失败保留原会话，日志足以重算并包含全部指标。"""
    profile, data, _, _ = table_setup
    data = deepcopy(data)
    data["tracker_poses"][10][0] += 0.1
    args = command_args(tmp_path)
    args.capture_output = tmp_path / "capture.json"
    args.session_calibration.write_text("keep old session", encoding="utf-8")
    monkeypatch.setattr(calibrate_umi, "collect_table", Mock(return_value=data))

    with pytest.raises(TableQualityError):
        calibrate_umi.table_command(args, profile)

    directory, = (tmp_path / "table-diagnostics").iterdir()
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["accepted_frames"] == 40
    assert report["metrics"]["translation_rms_m"] > 0.008
    assert set(report["limits"]) <= set(report["metrics"])
    assert json.loads((directory / "observations.json").read_text()) == data
    assert json.loads(args.capture_output.read_text()) == data
    assert yaml.safe_load((directory / "calibration.yaml").read_text()) == profile
    assert "TableQualityError" in (directory / "error.txt").read_text()
    assert args.session_calibration.read_text() == "keep old session"


def test_successful_runs_keep_separate_diagnostics(table_setup, tmp_path, monkeypatch):
    """多次成功采集保留独立日志，记录求解指标及待独立验证的结果。"""
    profile, data, _, _ = table_setup
    args = command_args(tmp_path)
    monkeypatch.setattr(calibrate_umi, "collect_table", Mock(return_value=data))
    for _ in range(2):
        calibrate_umi.table_command(args, profile)
    directories = list((tmp_path / "table-diagnostics").iterdir())
    assert len(directories) == 2
    for directory in directories:
        report = json.loads((directory / "report.json").read_text())
        result = yaml.safe_load((directory / "session-result.yaml").read_text())
        assert report["status"] == "passed"
        assert report["metrics"] == result["solve_metrics"]
        assert result["verification"]["accepted"] is False


def test_interrupt_keeps_partial_capture_and_invalidates_verification(table_setup, tmp_path, monkeypatch):
    """Ctrl+C 后仍保存部分观测和错误堆栈，独立验证保持失败状态。"""
    profile, data, session, _ = table_setup
    args = command_args(tmp_path)
    args.command = "verify-table"
    args.session_calibration.write_text(yaml.safe_dump(session), encoding="utf-8")
    partial = deepcopy(data)
    for key in ("camera_times", "observations", "tracker_poses"):
        partial[key] = partial[key][:2]

    def interrupt(*args, diagnostics, **kwargs):
        """模拟采集已有两帧后用户中断。"""
        diagnostics.data = partial
        raise KeyboardInterrupt

    monkeypatch.setattr(calibrate_umi, "collect_table", interrupt)
    with pytest.raises(KeyboardInterrupt):
        calibrate_umi.table_command(args, profile)
    directory, = (tmp_path / "table-diagnostics").iterdir()
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["end_reason"] == "interrupted"
    assert report["accepted_frames"] == 2
    assert json.loads((directory / "observations.json").read_text()) == partial
    assert (directory / "session-input.yaml").exists()
    assert "KeyboardInterrupt" in (directory / "error.txt").read_text()
    assert yaml.safe_load(args.session_calibration.read_text())["verification"]["accepted"] is False


@pytest.mark.parametrize("problem,reason", [
    ("none", None),
    ("moving", "not_stationary"),
    ("warming", "warming_up"),
    ("slow_detection", "frame_age_out_of_range"),
    ("board", "board_not_found_or_pnp_failed"),
    ("tracker", "tracker_unavailable"),
    ("pairing", "pairing_error_too_large"),
])
def test_capture_records_rejection_and_detection_cost(table_setup, tmp_path, monkeypatch, problem, reason):
    """采集记录真实判定原因，并区分取帧时新鲜但检测结束后超龄的情况。"""
    profile, data, _, _ = table_setup
    args = command_args(tmp_path)
    diagnostics = TableDiagnostics(args, profile, None)
    elapsed = [0.0]  # 模拟检测耗时，避免测试依赖真实计时或硬件。
    monkeypatch.setattr(calibrate_umi, "time", SimpleNamespace(
        monotonic=lambda: elapsed[0],
        monotonic_ns=lambda: 1_000_000_000 + round(elapsed[0] * 1e9),
    ))
    if problem == "warming":
        ticks = iter([0, 0, 0, 6])
        calibrate_umi.time.monotonic = lambda: next(ticks)
    capture_time = 770_000_000 if problem == "pairing" else 850_000_000
    frame = CameraSample(np.zeros((48, 64, 3), np.uint8), capture_time, 1)
    tracking = SimpleNamespace(
        right_tracked=problem != "tracker",
        right_device_controller_pose=np.array(data["tracker_poses"][0]),
    )
    devices = SimpleNamespace(
        offset_ns=0,
        camera=SimpleNamespace(sample_at=Mock(return_value=frame)),
        tracker=SimpleNamespace(_lock=threading.Lock(), _history=[], error=None, sample_at=Mock(return_value=tracking)),
    )
    monkeypatch.setattr(calibrate_umi, "NativeUmi", lambda _: nullcontext(devices))

    def detect(*args, stats=None):
        """返回固定角点、写入检测统计并推进模拟检测时钟。"""
        stats.update(aruco_tag_count=7, rejected_candidate_count=3)
        elapsed[0] += 0.25 if problem == "slow_detection" else 0.05
        observation = None if problem == "board" else tuple(
            np.asarray(part) for part in data["observations"][0]
        )
        return observation, [(index, np.zeros((4, 2))) for index in range(6)]

    monkeypatch.setattr(calibrate_umi, "detect_board_details", detect)
    captured = calibrate_umi.collect_table(profile, 0.01, preview=False, diagnostics=diagnostics, warmup_s=5 if problem == "warming" else 0, require_stationary=problem == "moving")
    # 逐帧文件在汇总前已经落盘，异常退出仍有已处理帧的证据。
    record = json.loads((diagnostics.directory / "frames.jsonl").read_text())
    assert record["rejected_reasons"] == ([] if reason is None else [reason])
    assert record["accepted"] is (reason is None)
    assert len(captured["observations"]) == (1 if reason is None else 0)
    assert record["tag_count"] == 6
    assert record["aruco_tag_count"] == 7
    assert record["rejected_candidate_count"] == 3
    assert record["image"] == "images/000001.jpg"
    assert (diagnostics.directory / record["image"]).stat().st_size > 0
    if problem == "slow_detection":
        assert record["frame_age_before_ms"] == 150
        assert record["frame_age_after_ms"] == 400
        assert record["detection_ms"] == 250
    diagnostics.finish(error=ValueError("test capture only"))
    report = json.loads((diagnostics.directory / "report.json").read_text())
    assert report["processed_frames"] == 1
    assert report["saved_images"] == 1
    assert report["rejected_reasons"] == ({} if reason is None else {reason: 1})


def test_diagnostic_write_failure_does_not_hide_capture_error(table_setup, tmp_path, monkeypatch):
    """磁盘写入失败不覆盖设备异常，终端提示日志不完整。"""
    profile, _, _, _ = table_setup
    args = command_args(tmp_path)
    monkeypatch.setattr(calibrate_umi, "collect_table", Mock(side_effect=RuntimeError("camera failed")))
    monkeypatch.setattr(TableDiagnostics, "finish", Mock(side_effect=OSError("disk full")))
    with pytest.raises(RuntimeError, match="camera failed") as caught:
        calibrate_umi.table_command(args, profile)
    assert "诊断保存不完整" in caught.value.__notes__[0]


def test_low_tag_images_follow_save_policy(table_setup, tmp_path, monkeypatch):
    """仅保存低标签帧，并遵守最小间隔和数量上限。"""
    profile, _, _, _ = table_setup
    monkeypatch.setitem(_umi_table_diagnostics.IMAGE_POLICY, "max_images", 2)
    diagnostics = TableDiagnostics(command_args(tmp_path), profile, None)
    image = np.zeros((48, 64, 3), np.uint8)
    image[..., 0] = 255  # 红色像素用于确认 RGB 到 BGR 的转换。

    def frame(sequence, time_ms):
        """构造指定序号和采集时间的帧。"""
        return CameraSample(image, time_ms * 1_000_000, sequence)

    assert diagnostics.save_frame_image(frame(1, 0), 20) is None
    assert diagnostics.save_frame_image(frame(2, 0), 19) == "images/000002.jpg"
    assert diagnostics.save_frame_image(frame(3, 100), 5) is None
    assert diagnostics.save_frame_image(frame(4, 300), 5) == "images/000004.jpg"
    assert diagnostics.save_frame_image(frame(5, 900), 5) is None
    saved = cv2.imread(str(diagnostics.directory / "images/000002.jpg"))
    assert saved[..., 2].mean() > 200 and saved[..., 0].mean() < 50
    assert sorted(path.name for path in (diagnostics.directory / "images").iterdir()) == [
        "000002.jpg", "000004.jpg",
    ]
