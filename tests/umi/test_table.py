"""验证桌面几何、绑定、独立验收、失败保护和稳定续录身份。"""

import json
import threading
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from handumi.calibration.umi import digest
from handumi.calibration.umi_table import (
    board_from_table,
    session_core,
    solve_table,
    validate_session,
    verify_table,
)
from handumi.cameras.base import CameraSample
from handumi.scripts import calibrate_umi, umi_capture
from handumi.scripts._umi_session import calibration_paths, load_session, save_session
from handumi.scripts._umi_table_diagnostics import TableDiagnostics


def test_table_geometry_and_height(table_setup):
    """板中心向下偏移真实板高；两段非交换变换按正确方向组合。"""
    profile, data, session, device_board = table_setup
    expected = np.linalg.inv(device_board @ board_from_table(0.012))
    np.testing.assert_allclose(session["table_from_device"], expected, atol=1e-7)
    np.testing.assert_allclose(board_from_table(0.012)[:3, :3], np.eye(3))
    np.testing.assert_allclose(board_from_table(0.012)[:3, 3], [0.20625, 0.20625, -0.012])
    another = solve_table(data, profile, 0.012)
    assert another["session_id"] != session["session_id"]
    with pytest.raises(ValueError, match="独立验证"):
        validate_session(another, profile)


@pytest.mark.parametrize("height", [-0.001, float("nan"), float("inf")])
def test_invalid_board_height(height):
    """非法板高不得生成看似有效的桌面变换。"""
    with pytest.raises(ValueError, match="高度"):
        board_from_table(height)


@pytest.mark.parametrize("field", ["serial", "geometry_digest", "time_digest", "transform"])
def test_session_rejects_binding_and_content_changes(table_setup, field):
    """设备、几何、时间结果与变换内容均必须匹配桌面身份。"""
    profile, _, session, _ = table_setup
    changed = deepcopy(session)
    if field == "transform":
        changed["table_from_device"][0][3] += 0.01
    else:
        changed[field] = "other"
    with pytest.raises(ValueError, match="不一致|摘要无效"):
        validate_session(changed, profile)


def test_table_rejects_sparse_static_and_displaced_observations(table_setup):
    """足够帧数仍需多视角；独立验证不能拟合掉桌面/追踪原点的变化。"""
    profile, data, session, _ = table_setup
    sparse = deepcopy(data)
    sparse["observations"] = sparse["observations"][:29]
    with pytest.raises(ValueError, match="30 帧"):
        solve_table(sparse, profile, 0)
    static = deepcopy(data)
    static["tracker_poses"] = [static["tracker_poses"][0]] * 40
    with pytest.raises(ValueError, match="4 个不同视角"):
        solve_table(static, profile, 0)
    shifted = deepcopy(data)
    for pose in shifted["tracker_poses"]:
        pose[0] += 0.02
    with pytest.raises(ValueError, match="translation_rms_m"):
        verify_table(shifted, profile, session)
    inconsistent = deepcopy(data)
    inconsistent["tracker_poses"][10][0] += 0.1
    with pytest.raises(ValueError, match="验证失败"):
        solve_table(inconsistent, profile, 0)


def test_cli_verification_failure_and_recovery_preserve_identity(table_setup, tmp_path, monkeypatch):
    """真实命令失败会撤销验收，重验通过后恢复，几何文件和桌面身份保持不变。"""
    profile, data, session, _ = table_setup
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(yaml.safe_dump(profile))
    before = calibration.read_bytes()
    target = tmp_path / "session.yaml"
    save_session(target, session)
    core = session_core(session)
    collect = Mock(side_effect=RuntimeError("camera failed"))
    monkeypatch.setattr(calibrate_umi, "collect_table", collect)
    flags = ["verify-table", "--calibration", str(calibration), "--session-calibration", str(target), "--no-preview"]
    with pytest.raises(RuntimeError, match="camera failed"):
        calibrate_umi.main(flags)
    with pytest.raises(ValueError, match="独立验证"):
        load_session(target, profile)
    assert collect.call_args.args[1] == 30
    assert collect.call_args.kwargs["require_stationary"] is False
    assert collect.call_args.kwargs["target_views"] is None
    failed = yaml.safe_load(target.read_text())
    assert session_core(failed) == core
    collect.side_effect = None
    collect.return_value = data
    calibrate_umi.main(flags)
    recovered = load_session(target, profile)
    assert session_core(recovered) == core
    assert calibration.read_bytes() == before
    assert collect.call_count == 2


def test_failed_solve_does_not_replace_existing_session(table_setup, tmp_path, monkeypatch):
    """重新求解失败保留旧文件；板高必须显式给出才可打开设备。"""
    profile, _, session, _ = table_setup
    target = tmp_path / "session.yaml"
    save_session(target, session)
    before = target.read_bytes()
    collect = Mock(side_effect=RuntimeError("no observations"))
    monkeypatch.setattr(calibrate_umi, "collect_table", collect)
    args = SimpleNamespace(command="table", board_height_m=None, session_calibration=target, duration=30, no_preview=True)
    with pytest.raises(ValueError, match="显式指定"):
        calibrate_umi.table_command(args, profile)
    collect.assert_not_called()
    args.board_height_m = 0
    with pytest.raises(RuntimeError, match="observations"):
        calibrate_umi.table_command(args, profile)
    assert target.read_bytes() == before


@pytest.mark.parametrize("problem", ["missing", "unverified", "wrong_binding"])
def test_record_failure_precedes_directory_and_hardware(table_setup, tmp_path, monkeypatch, problem):
    """无效桌面标定在覆盖输出目录、启动硬件和预览之前拒绝录制。"""
    profile, _, session, _ = table_setup
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(yaml.safe_dump(profile))
    target = tmp_path / "session.yaml"
    if problem == "unverified":
        session["verification"]["accepted"] = False
    if problem == "wrong_binding":
        session["time_digest"] = "different"
    if problem != "missing":
        save_session(target, session)
    root = tmp_path / "recording"
    root.mkdir()
    (root / "keep").write_text("original")
    hardware = Mock(side_effect=AssertionError("must not open hardware"))
    monkeypatch.setattr(umi_capture, "NativeUmi", hardware)
    with pytest.raises(SystemExit) as error:
        umi_capture.main([
            "--calibration", str(calibration), "--session-calibration", str(target),
            "--output-dir", str(root), "--no-preview",
        ])
    assert error.value.code == 2
    assert (root / "keep").read_text() == "original"
    hardware.assert_not_called()


def test_stable_resume_preserves_initial_verification(table_setup, tmp_path):
    """重复验收和路径变化允许续录；新身份、核心变换或旧原始系数据均拒绝。"""
    profile, data, session, _ = table_setup
    original = umi_capture.metadata(profile, session)
    (tmp_path / "meta/episodes").mkdir(parents=True)
    (tmp_path / "meta/tasks.parquet").touch()
    (tmp_path / "meta/episodes/file-000.parquet").touch()
    info = tmp_path / "meta/info.json"
    info.write_text(json.dumps({"total_episodes": 1, "handumi": original}))
    changed = deepcopy(session)
    changed["verification"]["checked_at"] = "later"
    changed["path"] = "moved.yaml"
    assert umi_capture.validate_resume_target(tmp_path, umi_capture.metadata(profile, changed)) == original
    fresh = solve_table(data, profile, 0.012)
    with pytest.raises(ValueError, match="新数据集"):
        umi_capture.validate_resume_target(tmp_path, umi_capture.metadata(profile, fresh))
    changed["table_from_device"][0][3] += 0.01
    changed["core_digest"] = digest(session_core(changed))
    with pytest.raises(ValueError, match="不一致"):
        umi_capture.validate_resume_target(tmp_path, umi_capture.metadata(profile, changed))
    info.write_text(json.dumps({"total_episodes": 1, "handumi": umi_capture.metadata(profile)}))
    with pytest.raises(ValueError, match="不一致"):
        umi_capture.validate_resume_target(tmp_path, original)


def test_session_path_priority_and_explicit_missing(table_setup, tmp_path):
    """CLI 覆盖 rig，隐式缺失只允许诊断降级，显式路径缺失必须报错。"""
    profile, _, _, _ = table_setup
    rig = tmp_path / "rig.yaml"
    rig.write_text("umi:\n  calibration: geometry.yaml\n  session_calibration: table.yaml\n")
    args = SimpleNamespace(rig_config=rig, calibration=None, session_calibration=None)
    assert [str(path) for path in calibration_paths(args)] == ["geometry.yaml", "table.yaml"]
    args.session_calibration = tmp_path / "chosen.yaml"
    assert calibration_paths(args)[1] == args.session_calibration
    assert load_session(args.session_calibration, profile, required=False) is None
    with pytest.raises(ValueError, match="无法读取"):
        load_session(args.session_calibration, profile, required=False, explicit=True)


def test_table_capture_uses_time_offset_without_gripper(table_setup, monkeypatch):
    """桌面采集按相机时间加偏移配对原始位姿，完全不访问夹爪检测。"""
    profile, data, _, _ = table_setup
    frame = CameraSample(np.zeros((48, 64, 3), np.uint8), 830_000_000, 1)
    tracking = SimpleNamespace(right_tracked=True, right_device_controller_pose=np.array(data["tracker_poses"][0]))
    devices = SimpleNamespace(
        offset_ns=20_000_000,
        camera=SimpleNamespace(sample_at=Mock(return_value=frame)),
        tracker=SimpleNamespace(_lock=threading.Lock(), _history=[], error=None, sample_at=Mock(return_value=tracking)),
    )
    monkeypatch.setattr(calibrate_umi, "NativeUmi", lambda _: nullcontext(devices))
    monkeypatch.setattr(calibrate_umi.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(calibrate_umi.time, "monotonic_ns", lambda: 1_000_000_000)
    monkeypatch.setattr(calibrate_umi, "detect_board_details", lambda *a, **k: (tuple(np.asarray(part) for part in data["observations"][0]), []))
    result = calibrate_umi.collect_table(profile, 1, preview=False, warmup_s=0)
    devices.camera.sample_at.assert_called_once_with(830_000_000)
    devices.tracker.sample_at.assert_called_once_with(850_000_000)
    assert result["sampling_policy"]["require_stationary"] is False
    assert result["sampling_policy"]["target_views"] is None
    assert result["sampling_policy"]["end_reason"] == "duration_elapsed"
    assert result["camera_times"] == [0.83]
    np.testing.assert_allclose(result["tracker_poses"], [tracking.right_device_controller_pose])


@pytest.mark.parametrize("axis_length", ["0", "-0.1", "nan", "inf"])
def test_invalid_axis_length_precedes_loading_files(axis_length):
    """非法显示尺度在加载机器配置和启动设备前报错。"""
    with pytest.raises(SystemExit) as error:
        umi_capture.main(["--tcp-axis-length-m", axis_length], mode="visualize")
    assert error.value.code == 2


def test_stationary_preview_displays_completion(table_setup, monkeypatch):
    """真实 Rerun 文本和布局可构造，停稳收帧后向面板发布完成提示。"""
    import rerun as rr

    from handumi.scripts._umi_table_sampling import StationaryViewCapture

    profile, data, _, _ = table_setup
    frame = CameraSample(np.zeros((48, 64, 3), np.uint8), 830_000_000, 1)
    tracking = SimpleNamespace(right_tracked=True, right_device_controller_pose=np.array(data["tracker_poses"][0]))
    devices = SimpleNamespace(
        offset_ns=20_000_000,
        camera=SimpleNamespace(sample_at=Mock(return_value=frame)),
        tracker=SimpleNamespace(_lock=threading.Lock(), _history=[], error=None, sample_at=Mock(return_value=tracking)),
    )
    monkeypatch.setattr(calibrate_umi, "NativeUmi", lambda _: nullcontext(devices))
    monkeypatch.setattr(calibrate_umi.time, "monotonic", Mock(side_effect=[0, 0, 0, 0, 2]))
    monkeypatch.setattr(calibrate_umi.time, "monotonic_ns", lambda: 1_000_000_000)
    monkeypatch.setattr(calibrate_umi, "detect_board_details", lambda *a, **k: (tuple(np.asarray(part) for part in data["observations"][0]), []))
    monkeypatch.setattr(calibrate_umi, "stationary", lambda *a: True)
    monkeypatch.setattr(calibrate_umi, "StationaryViewCapture", lambda: StationaryViewCapture(1))
    for name in ["init", "log", "send_blueprint", "disconnect"]:
        monkeypatch.setattr(rr, name, Mock())
    result = calibrate_umi.collect_table(profile, 1, warmup_s=0, require_stationary=True)
    assert result["camera_times"] == [0.83]
    rr.send_blueprint.assert_called_once()
    statuses = [call.args[1] for call in rr.log.call_args_list if call.args[0] == "calibration/status"]
    assert "VIEW COMPLETE" in str(statuses[-1].text.as_arrow_array().to_pylist())
    rr.disconnect.assert_called_once()


def target_capture_devices(data, frames_per_view=15):
    """构造每收满一个视角就平移 6 cm 的合成设备，并统计取帧次数。"""
    image = np.zeros((48, 64, 3), np.uint8)
    state = {"frames": 0, "clock_ns": 0}
    base_pose = np.array(data["tracker_poses"][0], float)

    def monotonic_ns():
        """推进模拟单调时钟，保证采集帧时间递增且满足同步门限。"""
        state["clock_ns"] += 1_000_000
        return state["clock_ns"]

    def sample_at(target):
        """返回与目标时间一致的新序号帧。"""
        state["frames"] += 1
        return CameraSample(image, int(target), state["frames"])

    def sample_pose(_):
        """按已取帧数切换独立视角，每个视角保持静止。"""
        pose = base_pose.copy()
        pose[0] = ((state["frames"] - 1) // frames_per_view) * 0.06
        return SimpleNamespace(
            right_tracked=True, right_device_controller_pose=pose,
        )

    devices = SimpleNamespace(
        offset_ns=0,
        camera=SimpleNamespace(sample_at=sample_at),
        tracker=SimpleNamespace(
            _lock=threading.Lock(), _history=[], error=None, sample_at=sample_pose,
        ),
    )
    return state, devices, monotonic_ns


def test_collect_table_target_stops_after_fifteen_views(table_setup, monkeypatch):
    """完成第 14 个视角后继续采集，第 15 个视角最后一帧后立即结束。"""
    profile, data, _, _ = table_setup
    state, devices, monotonic_ns = target_capture_devices(data)
    monkeypatch.setattr(calibrate_umi, "NativeUmi", lambda _: nullcontext(devices))
    monkeypatch.setattr(calibrate_umi.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(calibrate_umi.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(calibrate_umi.time, "sleep", lambda _: None)
    monkeypatch.setattr(calibrate_umi, "stationary", lambda *args: True)
    monkeypatch.setattr(
        calibrate_umi, "detect_board_details",
        lambda *args, **kwargs: (
            tuple(np.asarray(part) for part in data["observations"][0]), [],
        ),
    )

    result = calibrate_umi.collect_table(
        profile, target_views=15, preview=False, warmup_s=0,
        require_stationary=True,
    )
    assert state["frames"] == 15 * 15
    assert len(result["camera_times"]) == 15 * 15
    assert result["sampling_policy"]["target_views"] == 15
    assert result["sampling_policy"]["completed_views"] == 15
    assert result["sampling_policy"]["end_reason"] == "target_views_reached"


def test_target_preview_and_diagnostics_save_last_frame(table_setup, tmp_path, monkeypatch):
    """第 15 个视角完成后保存最后一帧与诊断，并把预览刷新为 15/15。"""
    import rerun as rr

    profile, data, _, _ = table_setup
    state, devices, monotonic_ns = target_capture_devices(data)
    args = SimpleNamespace(
        command="table", board_height_m=0.0082, duration=30, no_preview=False,
        session_calibration=tmp_path / "session.yaml",
    )
    diagnostics = TableDiagnostics(args, profile, None)
    monkeypatch.setattr(calibrate_umi, "NativeUmi", lambda _: nullcontext(devices))
    monkeypatch.setattr(calibrate_umi.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(calibrate_umi.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(calibrate_umi.time, "sleep", lambda _: None)
    monkeypatch.setattr(calibrate_umi, "stationary", lambda *args: True)
    monkeypatch.setattr(
        calibrate_umi, "detect_board_details",
        lambda *args, **kwargs: (
            tuple(np.asarray(part) for part in data["observations"][0]), [],
        ),
    )
    for name in ["init", "log", "send_blueprint", "disconnect"]:
        monkeypatch.setattr(rr, name, Mock())

    result = calibrate_umi.collect_table(
        profile, target_views=15, preview=True, diagnostics=diagnostics,
        warmup_s=0, require_stationary=True,
    )
    diagnostics.finish()

    assert state["frames"] == 15 * 15
    assert len(result["camera_times"]) == 15 * 15
    records = [
        json.loads(line) for line in
        (diagnostics.directory / "frames.jsonl").read_text().splitlines()
    ]
    assert records[-1]["sequence"] == 15 * 15
    assert records[-1]["accepted"] is True
    saved = json.loads((diagnostics.directory / "observations.json").read_text())
    assert saved["camera_times"] == result["camera_times"]
    report = json.loads((diagnostics.directory / "report.json").read_text())
    assert report["target_views"] == 15
    assert report["completed_views"] == 15
    assert report["end_reason"] == "target_views_reached"
    assert report["accepted_frames"] == 15 * 15
    assert report["duration_s"] is None

    statuses = [
        call.args[1] for call in rr.log.call_args_list
        if call.args[0] == "calibration/status"
    ]
    rendered = [status.text.as_arrow_array().to_pylist()[0] for status in statuses]
    assert {text.splitlines()[0] for text in rendered} <= {
        "# CAPTURING", "# CURRENT VIEW COMPLETE",
    }
    assert all("Completed views:" in text for text in rendered)
    assert all(
        "Remaining" not in text and "frames" not in text for text in rendered
    )
    assert "# CURRENT VIEW COMPLETE" in rendered[-1]
    assert "Completed views: 15/15" in rendered[-1]
    rr.disconnect.assert_called_once()


def test_table_preview_keeps_labels_and_uses_simplified_status(monkeypatch):
    """图像叠字保留标签 ID，只显示两种状态和完成进度。"""
    from handumi.scripts._umi_table_sampling import StationaryViewCapture

    capture = StationaryViewCapture(frames_per_view=1, target_views=15)
    texts = []
    monkeypatch.setattr(
        calibrate_umi.cv2, "putText",
        lambda image, text, *args, **kwargs: texts.append(text),
    )
    marker = np.array([[10, 80], [20, 80], [20, 90], [10, 90]], np.float32)
    canvas = calibrate_umi.table_preview(
        np.zeros((120, 160, 3), np.uint8), [(3, marker)], capture,
    )
    assert texts == ["3", "CAPTURING", "Completed views: 0/15"]
    assert canvas.shape == (120, 160, 3)
    assert np.any(canvas)


def test_table_command_defaults_to_15_views_and_verify_table_stays_timed(
    table_setup, tmp_path, monkeypatch,
):
    """table 总是传入 15 视角和停稳采集；verify-table 默认仍按时长采集。"""
    profile, data, session, _ = table_setup
    target_data = deepcopy(data)
    target_data["sampling_policy"] = {
        "target_views": 15, "completed_views": 15,
        "end_reason": "target_views_reached",
    }
    collect = Mock(return_value=target_data)
    monkeypatch.setattr(calibrate_umi, "collect_table", collect)
    args = SimpleNamespace(
        command="table", board_height_m=0, duration=1, stationary=False,
        session_calibration=tmp_path / "session.yaml", no_preview=True,
    )
    calibrate_umi.table_command(args, profile)
    assert collect.call_args.args[1] is None
    assert collect.call_args.kwargs["require_stationary"] is True
    assert collect.call_args.kwargs["target_views"] == 15
    report = json.loads(
        (next((tmp_path / "table-diagnostics").iterdir()) / "report.json").read_text()
    )
    assert report["duration_s"] is None
    assert report["target_views"] == 15
    assert report["end_reason"] == "target_views_reached"

    timed_data = deepcopy(data)
    timed_data["sampling_policy"] = {
        "target_views": None, "completed_views": 4,
        "end_reason": "duration_elapsed",
    }
    collect.return_value = timed_data
    args.command = "verify-table"
    args.duration = 60
    args.stationary = False
    calibrate_umi.table_command(args, profile)
    assert collect.call_args.args[1] == 60
    assert collect.call_args.kwargs["require_stationary"] is False
    assert collect.call_args.kwargs["target_views"] is None
    reports = [
        json.loads((directory / "report.json").read_text())
        for directory in (tmp_path / "table-diagnostics").iterdir()
    ]
    assert any(
        report["command"] == "verify-table"
        and report["target_views"] is None
        and report["end_reason"] == "duration_elapsed"
        for report in reports
    )


@pytest.mark.parametrize("duration", ["0", "nan"])
def test_table_does_not_validate_duration_but_verify_table_does(
    tmp_path, monkeypatch, duration,
):
    """table 忽略非法时长；verify-table 保留原有最短时长校验。"""
    captured = []
    monkeypatch.setattr(calibrate_umi, "verify", lambda snapshot, **kwargs: snapshot)
    monkeypatch.setattr(
        calibrate_umi, "table_command",
        lambda args, snapshot: captured.append(args),
    )
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    common = [
        "--calibration", str(calibration),
        "--rig-config", str(tmp_path / "missing-rig.yaml"),
    ]
    calibrate_umi.main(
        ["table", "--board-height-m", "0.0082", "--duration", duration, *common]
    )
    assert len(captured) == 1
    if duration == "nan":
        assert np.isnan(captured[0].duration)
    else:
        assert captured[0].duration == 0
    with pytest.raises(SystemExit) as error:
        calibrate_umi.main(["verify-table", "--duration", duration, *common])
    assert error.value.code == 2


def test_target_capture_interrupt_releases_devices_and_saves_partial_diagnostics(
    table_setup, tmp_path, monkeypatch,
):
    """目标视角采集中断仍释放设备与预览，并保存已接收帧的诊断。"""
    import rerun as rr

    profile, data, _, _ = table_setup
    state, devices, monotonic_ns = target_capture_devices(data)
    sample_at = devices.camera.sample_at

    def interrupted_sample(target):
        """前五帧正常返回，随后模拟用户 Ctrl+C。"""
        if state["frames"] >= 5:
            raise KeyboardInterrupt
        return sample_at(target)

    devices.camera.sample_at = interrupted_sample
    closed = []

    class Hardware:
        """记录设备上下文是否被释放。"""

        def __init__(self, snapshot):
            """接收与真实设备相同的标定快照。"""

        def __enter__(self):
            """返回模拟设备。"""
            return devices

        def __exit__(self, *args):
            """记录设备释放。"""
            closed.append(True)

    args = SimpleNamespace(
        command="table", board_height_m=0.0082, duration=30, no_preview=False,
        session_calibration=tmp_path / "session.yaml",
    )
    diagnostics = TableDiagnostics(args, profile, None)
    monkeypatch.setattr(calibrate_umi, "NativeUmi", Hardware)
    monkeypatch.setattr(calibrate_umi.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(calibrate_umi.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(calibrate_umi.time, "sleep", lambda _: None)
    monkeypatch.setattr(calibrate_umi, "stationary", lambda *args: True)
    monkeypatch.setattr(
        calibrate_umi, "detect_board_details",
        lambda *args, **kwargs: (
            tuple(np.asarray(part) for part in data["observations"][0]), [],
        ),
    )
    for name in ["init", "log", "send_blueprint", "disconnect"]:
        monkeypatch.setattr(rr, name, Mock())

    with pytest.raises(KeyboardInterrupt):
        calibrate_umi.collect_table(
            profile, target_views=15, preview=True, diagnostics=diagnostics,
            warmup_s=0, require_stationary=True,
        )
    assert closed == [True]
    rr.disconnect.assert_called_once()
    assert len(diagnostics.data["observations"]) == 5
    assert diagnostics.data["sampling_policy"]["completed_views"] == 0
    diagnostics.finish(error=KeyboardInterrupt())
    report = json.loads((diagnostics.directory / "report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["end_reason"] == "interrupted"
    assert report["accepted_frames"] == 5
