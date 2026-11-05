"""验证 Vive 纯数据采集入口及共享 Rerun 布局。"""

import json
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.cameras.base import CameraSample
from handumi.grippers.base import GripperObservation
from handumi.scripts import umi_capture
from handumi.tracking.base import ControllerPairSample


def test_capture_deadline_drops_backlog_after_slow_preview():
    """预览超期后丢弃积压，短工作循环不以短间隔连续补跑。"""
    period = 1 / 30
    deadline = 0.0
    # 首轮预览耗时 130 ms，下一轮从当前时刻开始，不追赶过去的采样点。
    deadline = umi_capture._next_capture_deadline(deadline, 0.130, 30)
    assert deadline == pytest.approx(0.130)
    first_start = deadline
    # 下一轮只耗时 9 ms，接下来的开始时刻应距上一轮 1/30 秒。
    deadline = umi_capture._next_capture_deadline(deadline, first_start + 0.009, 30)
    assert deadline - first_start == pytest.approx(period)
    second_start = deadline
    deadline = umi_capture._next_capture_deadline(deadline, second_start + 0.010, 30)
    assert deadline - second_start == pytest.approx(period)
    # 正常工作循环沿原定采样节拍前进。
    assert umi_capture._next_capture_deadline(1.0, 1.010, 30) == pytest.approx(1.0 + period)


@pytest.mark.parametrize("option", [
    ["--robot-preview"],
    ["--robot-tool-calibration", "tool.yaml"],
    ["--robot-position-tolerance-m", "0.01"],
    ["--robot-orientation-tolerance-deg", "5"],
    ["--robot", "rm75b"],
])
def test_removed_robot_options_fail_before_output_or_device(tmp_path, monkeypatch, option):
    """旧采集参数在读取标定及触及输出目录或硬件之前被拒绝。"""
    output = tmp_path / "dataset"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    monkeypatch.setattr(umi_capture, "calibration_paths", Mock(side_effect=AssertionError))
    with pytest.raises(SystemExit) as error:
        umi_capture.main(["--output-dir", str(output), *option])
    assert error.value.code == 2
    assert marker.read_text() == "original"


def test_four_panel_blueprint_and_no_preview(monkeypatch):
    """显式布局包含四种视图；禁用预览时不创建记录流。"""
    import rerun as rr

    real_recording_stream = rr.RecordingStream
    stream = Mock()
    make_stream = Mock(return_value=stream)
    spawn = Mock()
    monkeypatch.setattr(rr, "RecordingStream", make_stream)
    monkeypatch.setattr(rr, "spawn", spawn)
    assert umi_capture._start_camera_viewer(SimpleNamespace(no_preview=True), None) is None
    make_stream.assert_not_called()
    viewer = umi_capture._start_camera_viewer(SimpleNamespace(no_preview=False), {})
    assert viewer.stream is stream
    spawn.assert_called_once()
    assert spawn.call_args.kwargs["recording"] is stream
    assert spawn.call_args.kwargs["port"] > 0
    assert "rerun_cli" in spawn.call_args.kwargs["executable_path"]
    blueprint = stream.send_blueprint.call_args.args[0]
    root = blueprint.root_container
    assert root.grid_columns == 2
    assert [view.origin for view in root.contents] == [
        "/camera/right_wrist", "/tracking", "/gripper", "/capture_status",
    ]
    assert [view.class_identifier for view in root.contents] == [
        "2D", "3D", "TimeSeries", "TextDocument",
    ]
    assert blueprint.blueprint_panel.state.name == "Expanded"
    assert getattr(blueprint, "selection_panel", None) is None
    assert getattr(blueprint, "time_panel", None) is None
    view = root.contents[1]
    background = view.properties["Background"]
    assert background.color.as_arrow_array().to_pylist() == [
        int.from_bytes(bytes([40, 8, 8, 255]), "big")
    ]
    assert view.properties["LineGrid3D"].visible.as_arrow_array().to_pylist() == [False]
    # 实际编码蓝图，覆盖属性能否序列化到 Rerun 记录流。
    recording = real_recording_stream("capture_blueprint_test", make_default=False)
    memory = recording.memory_recording()
    try:
        recording.send_blueprint(blueprint)
        recording.flush()
        assert len(memory.drain_as_bytes()) > 0
    finally:
        recording.disconnect()


def test_dry_run_does_not_open_devices_or_modify_output(tmp_path, monkeypatch):
    """预演解析标定后退出，已有输出目录及其内容不变。"""
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    output = tmp_path / "dataset"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("original")
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {"tracker_camera": np.eye(4).tolist()})
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", Mock(side_effect=AssertionError))
    monkeypatch.setattr(umi_capture, "NativeUmi", Mock(side_effect=AssertionError))
    umi_capture.main(["--calibration", str(calibration), "--output-dir", str(output), "--dry-run"])
    assert marker.read_text() == "original"


class ImmediateExecutor:
    """让既有状态测试固定收取结果的时序，并由并发测试覆盖真实线程。"""

    def __init__(self, **kwargs):
        """匹配单线程执行器构造接口。"""

    def submit(self, fn, *args):
        """立即结束模拟任务，主循环仍在下一轮收取结果。"""
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True):
        """模拟关闭无后台线程的执行器。"""



@pytest.mark.parametrize("early_exit", [False, True])
def test_record_save_discard_and_exit_without_ik(tmp_path, monkeypatch, early_exit, capsys):
    """采集状态机保存有效片段、丢弃失效片段，并正常释放数据集。"""
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    output = tmp_path / "dataset"
    dataset = Mock()
    frames = []
    saved = []
    events = []
    viewer = Mock()
    viewer.flush.side_effect = lambda: events.append(("flush", None))
    dataset.add_frame.side_effect = frames.append

    def save_episode():
        """记录保存发生的时刻，验证保存前状态已刷新。"""
        events.append(("save", None))
        saved.append(frames.copy())
        frames.clear()

    dataset.save_episode.side_effect = save_episode
    dataset.clear_episode_buffer.side_effect = frames.clear

    def open_dataset(*args):
        """准备资源释放时读取的模拟元数据。"""
        (output / "meta").mkdir(parents=True)
        (output / "meta/info.json").write_text(json.dumps({}))
        return dataset

    base = umi_capture.time.monotonic_ns()
    samples = []
    for i, healthy in enumerate([True, True, True, True, True, False, True]):
        stamp = base + i * 1_000_000
        pose = np.array([i, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        tracking = replace(
            ControllerPairSample.empty("vive"), right_tracked=healthy,
            right_pose_valid=healthy, right_controller_pose=pose, right_tcp_pose=pose.copy(),
        )
        image = CameraSample(np.zeros((4, 4, 3), dtype=np.uint8), stamp, i + 1)
        samples.append((image, tracking, GripperObservation(0.5, healthy, stamp, i + 1), stamp, healthy))

    class FakeDevice:
        """逐次交付固定观测，不打开设备。"""

        def __init__(self, *args, **kwargs):
            """初始化固定样本迭代器。"""
            self.samples = iter(samples)

        def __enter__(self):
            """返回模拟设备。"""
            return self

        def __exit__(self, *args):
            """结束模拟设备会话。"""

        def sample(self, now, *, after_sequence=None):
            """返回下一组合成观测。"""
            return next(self.samples)

    @contextmanager
    def fake_keyboard():
        """模拟开始、保存、再次开始、失效丢弃及退出。"""
        keys = iter([" ", "", " ", " ", " ", "", "\x1b"])
        yield lambda: next(keys)

    monkeypatch.setattr(umi_capture, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {"tracker_camera": np.eye(4).tolist()})
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {"recording_device": "vive"})
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "NativeUmi", FakeDevice)
    monkeypatch.setattr(umi_capture, "keyboard", fake_keyboard)
    monkeypatch.setattr(
        umi_capture, "_prepare_streaming_episode",
        lambda *_: {"encoder_flush_s": 0.0, "video_verify_s": 0.0},
    )
    monkeypatch.setattr(umi_capture, "_open_recording_dataset", open_dataset)
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", lambda *args: viewer)
    monkeypatch.setattr(umi_capture, "log_live_preview", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        umi_capture, "log_capture_status",
        lambda _viewer, **status: events.append(("status", status)),
    )
    monkeypatch.setattr(umi_capture.time, "sleep", lambda *_: None)
    args = ["--calibration", str(calibration), "--output-dir", str(output)]
    if early_exit:
        args += ["--num-episodes", "1"]
    umi_capture.main(args)
    assert len(saved) == 1
    assert len(saved[0]) == 2
    assert all("observation.tracking.right_tcp_pose" in frame for frame in saved[0])
    assert dataset.clear_episode_buffer.call_count == (0 if early_exit else 1)
    dataset.finalize.assert_called_once()
    assert not (output / "robot_ik").exists()
    save_index = next(i for i, event in enumerate(events) if event[0] == "save")
    assert events[save_index - 1] == ("flush", None)
    assert events[save_index - 2][1]["saving"] is True
    assert events[save_index - 2][1]["frame_count"] == 2
    statuses = [status for name, status in events if name == "status"]
    assert any(s["saved"] == 1 and s["last_result"] == "已保存 2 帧" for s in statuses)
    assert any(s["recording"] and s["frame_count"] == 1 and not s["last_result"] for s in statuses)
    if early_exit:
        assert events[-1] == ("flush", None)
        assert events[-2][1]["last_result"] == "已保存 2 帧"
    else:
        assert any("Tracker 未追踪" in s["last_result"] for s in statuses)
        output_text = capsys.readouterr().out
        assert "当前段已丢弃：Tracker 未追踪" in output_text
        assert "按空格重新开始" in output_text
    viewer.disconnect.assert_called_once()


@pytest.mark.parametrize("resume", [False, True])
def test_status_persists_results_and_resets_for_each_session(tmp_path, monkeypatch, resume, capsys):
    """普通刷新保留结果，新段清除结果，续录也只统计本次保存段数。"""
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    output = tmp_path / "dataset"
    output.mkdir()
    dataset = Mock()
    viewer = Mock()
    statuses = []

    def open_dataset(*args):
        """构造已有数据集所需的模拟元数据文件。"""
        (output / "meta").mkdir(exist_ok=True)
        (output / "meta/info.json").write_text("{}")
        # 假设续录数据集已有多段；面板不应把它们计入本次保存数。
        dataset.meta.total_episodes = 8 if resume else 0
        return dataset

    class FakeDevice:
        """为每轮提供新的有效相机帧及对应的健康检查。"""

        def __init__(self, *args, **kwargs):
            """初始化帧序号。"""
            self.sequence = 0
            self.last_health = umi_capture.CaptureHealth(True, True, 150_000_000, 0)

        def __enter__(self):
            """返回模拟设备。"""
            return self

        def __exit__(self, *args):
            """结束模拟设备会话。"""

        def sample(self, now, *, after_sequence=None):
            """生成匹配目标时间的样本，避免重复帧触发自动丢弃。"""
            self.sequence += 1
            stamp = now - 150_000_000
            image = CameraSample(np.zeros((4, 4, 3), dtype=np.uint8), stamp, self.sequence)
            pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            tracking = replace(
                ControllerPairSample.empty("vive"), right_tracked=True,
                right_pose_valid=True, right_controller_pose=pose, right_tcp_pose=pose.copy(),
            )
            gripper = GripperObservation(0.5, True, stamp, self.sequence)
            return image, tracking, gripper, stamp, True

    @contextmanager
    def fake_keyboard():
        """保存、等待、开始新段、手动取消、等待，再退出。"""
        keys = iter(["", " ", " ", "", "", " ", "\x1b", "", "\x1b"])
        yield lambda: next(keys)

    monotonic_values = count()
    monkeypatch.setattr(umi_capture.time, "monotonic", lambda: next(monotonic_values) / 10)
    monkeypatch.setattr(umi_capture.time, "sleep", lambda *_: None)
    monkeypatch.setattr(umi_capture, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {"tracker_camera": np.eye(4).tolist()})
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {"recording_device": "vive"})
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})
    validate = Mock(return_value={"recording_device": "vive"})
    monkeypatch.setattr(umi_capture, "validate_resume_target", validate)
    monkeypatch.setattr(umi_capture, "_check_output_directory", lambda *args: None)
    monkeypatch.setattr(umi_capture, "_prepare_output_directory", lambda *args: None)
    monkeypatch.setattr(umi_capture, "NativeUmi", FakeDevice)
    monkeypatch.setattr(umi_capture, "keyboard", fake_keyboard)
    monkeypatch.setattr(
        umi_capture, "_prepare_streaming_episode",
        lambda *_: {"encoder_flush_s": 0.0, "video_verify_s": 0.0},
    )
    monkeypatch.setattr(umi_capture, "_open_recording_dataset", open_dataset)
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", lambda *args: viewer)
    monkeypatch.setattr(umi_capture, "log_live_preview", lambda *a, **k: None)
    monkeypatch.setattr(
        umi_capture, "log_capture_status",
        lambda _viewer, **status: statuses.append(status),
    )
    args = ["--calibration", str(calibration), "--output-dir", str(output)]
    if resume:
        args.append("--resume")
    umi_capture.main(args)

    assert validate.call_count == int(resume)
    normal = [status for status in statuses if not status.get("saving", False)]
    assert len(normal) == 7
    ready, first_start, saved, saved_wait, next_start, discarded, discarded_wait = normal
    initial_total = 8 if resume else 0  # 续录从已有段数继续编号。
    assert ready["total_episodes"] == initial_total
    assert first_start["total_episodes"] == initial_total
    assert all(s["total_episodes"] == initial_total for s in statuses if s.get("saving"))
    assert all(s["total_episodes"] == initial_total + 1 for s in normal[2:])
    terminal = capsys.readouterr().out
    assert f"当前 episode ID：{initial_total}；" in terminal
    assert f"已保存 episode ID：{initial_total}；" in terminal
    assert f"episode 总数（已保存）：{initial_total + 1}；" in terminal
    assert ready["saved"] == 0 and not ready["last_result"]
    assert first_start["recording"] and first_start["frame_count"] == 1
    assert saved["saved"] == 1 and saved["last_result"] == "已保存 1 帧"
    assert saved_wait["saved"] == 1 and saved_wait["last_result"] == saved["last_result"]
    assert next_start["recording"] and next_start["frame_count"] == 1
    assert not next_start["last_result"] and next_start["saved"] == 1
    assert discarded["last_result"] == "当前段已丢弃：用户取消"
    assert not discarded["recording"] and discarded["frame_count"] == 0
    assert discarded_wait["last_result"] == discarded["last_result"]
    assert discarded_wait["saved"] == 1
    dataset.save_episode.assert_called_once()
    dataset.clear_episode_buffer.assert_called_once()
