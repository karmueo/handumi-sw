"""Vive 录制入口的机器人预览/附加记录生命周期集成测试。"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from handumi.cameras.base import CameraSample
from handumi.dataset.robot_ik import (
    RobotIkEpisodeRecorder,
    RobotIkPublishError,
    find_incomplete_records,
)
from handumi.grippers.base import GripperObservation
from handumi.scripts import umi_capture
from handumi.scripts._umi_robot_preview import RobotIkPreview
from handumi.tracking.base import ControllerPairSample


def _tool_matrix() -> np.ndarray:
    """合成非平凡工具矩阵，不复制本机私有标定。"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", [10.0, -20.0, 30.0], degrees=True
    ).as_matrix()
    matrix[:3, 3] = [0.012, -0.008, 0.195]
    return matrix


def _tool_yaml(path: Path) -> Path:
    """写入带 T_end_tcp 的合成 YAML。"""
    path.write_text(
        yaml.safe_dump({"T_end_tcp": _tool_matrix().tolist()}),
        encoding="utf-8",
    )
    return path


class FakeDataset:
    """记录帧顺序和保存缓冲区的 LeRobot 替身。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.meta = SimpleNamespace(info={})
        self.num_episodes = 0
        self.current: list[dict] = []
        self.saved_frames: list[list[dict]] = []
        self.clears = 0
        self.finalized = False
        self.add_times: list[float] = []

    def add_frame(self, frame: dict) -> None:
        self.current.append(frame)
        self.add_times.append(time.monotonic())

    def clear_episode_buffer(self) -> None:
        self.clears += 1
        self.current.clear()

    def save_episode(self) -> None:
        self.saved_frames.append(list(self.current))
        self.current.clear()
        self.num_episodes += 1

    def finalize(self) -> None:
        self.finalized = True


class FakeViewer:
    """现有采集窗口的替身，带共享记录流属性。"""

    def __init__(self) -> None:
        self.stream = object()
        self.disconnected = 0
        self.logged: list[tuple] = []
        self.times: list[tuple] = []

    def __getattr__(self, name):
        import rerun as rr

        return getattr(rr, name)

    def set_time(self, timeline, **kwargs) -> None:
        self.times.append((timeline, kwargs))

    def log(self, entity_path, entity, *, static=False) -> None:
        self.logged.append((entity_path, entity, static))

    def disconnect(self) -> None:
        self.disconnected += 1


class FakePreview:
    """记录预览生命周期调用，不启动线程。"""

    def __init__(self) -> None:
        self.healthy = True
        self.events: list[tuple] = []
        self.closed = 0

    def begin_episode(self, pose, capture_time_ns) -> None:
        self.events.append(("begin_episode", None if pose is None else pose.copy(), int(capture_time_ns)))

    def submit(self, capture_time_ns, pose, *, tracked=True) -> None:
        self.events.append(
            ("submit", int(capture_time_ns), None if pose is None else pose.copy(), bool(tracked))
        )

    def pause(self, *, timeout_s=2.0) -> bool:
        self.events.append(("pause", timeout_s))
        return True

    def resume(self) -> None:
        self.events.append(("resume",))

    def end_episode(self) -> None:
        self.events.append(("end_episode",))

    def cancel_episode(self) -> None:
        self.events.append(("cancel_episode",))

    def close(self) -> None:
        self.closed += 1


class FakeDevice:
    """按脚本返回样本，不连接硬件。"""

    def __init__(self, samples) -> None:
        self.samples = iter(samples)
        self.closed = 0

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.closed += 1

    def sample(self, now_ns=None):
        del now_ns
        return next(self.samples)


class EntryIkController:
    """不依赖 JAX 的补算控制器替身。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.processed: list[float | None] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def process(self, pose):
        self.processed.append(None if pose is None else float(pose[0]))
        if self.fail:
            raise RuntimeError("simulated prepare failure")
        q = np.zeros(7, dtype=np.float32)
        return SimpleNamespace(
            q=q,
            target_tcp_pose7=q.copy(),
            actual_tcp_pose7=q.copy(),
            position_error_m=0.0,
            orientation_error_deg=0.0,
            valid=True,
            failure_reason="",
            failure_reasons=(),
        )

    def metadata(self) -> dict:
        identity = np.eye(4).tolist()
        return {
            "robot": {
                "name": "rm75b",
                "side": "right",
                "joint_names": [f"joint{index}" for index in range(1, 8)],
                "home_q_rad": [0.0] * 7,
                "home_end_pose7": [0, 0, 0, 0, 0, 0, 1],
                "home_tcp_pose7": [0, 0, 0, 0, 0, 0, 1],
            },
            "tool_calibration": {
                "key": "T_end_tcp",
                "matrix": identity,
                "source": "synthetic",
                "sha256": None,
                "notes": [],
            },
            "relative_mapping": {
                "mode": "local-relative",
                "compose_source": "commanded",
                "initial_alignment": "source_local_frame_to_home_tool_tcp",
                "translation_scale": 1.0,
                "source_world_to_robot_world": np.eye(3).tolist(),
                "max_reach_m": None,
                "anchor_source_tcp_pose7": None,
                "adapter_rotation": None,
                "home_tcp_pose7": [0, 0, 0, 0, 0, 0, 1],
            },
            "ik": {"parameters": {"max_joint_delta": 0.12}, "solver": "Fake"},
            "validity": {
                "max_position_error_m": 0.01,
                "max_orientation_error_deg": 5.0,
                "joint_limits_checked": True,
                "joint_limit_tolerance_rad": 1e-6,
            },
        }


def _sample(index: int, base_ns: int, *, healthy: bool = True):
    """构造与实际 capture_time 不同的计划 target，供单位回归使用。"""
    capture = base_ns + index * 33_000_000
    target = capture + 123_000_000
    tracker_pose = np.array(
        [0.001 * index, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
    )
    tracking = replace(
        ControllerPairSample.empty("vive"),
        right_tracked=healthy,
        right_pose_valid=healthy,
        right_device_tracked=healthy,
        connected=True,
        streaming=True,
        clock_synced=True,
        right_controller_pose=tracker_pose,
        right_tcp_pose=tracker_pose.copy(),
        pc_monotonic_ns=capture,
        aligned_time_ns=capture,
        sequence=index + 1,
    )
    image = CameraSample(
        np.zeros((8, 8, 3), dtype=np.uint8),
        capture,
        index + 1,
    )
    gripper = GripperObservation(0.5, healthy, capture, index + 1)
    return image, tracking, gripper, target, healthy


class SlowController:
    """首帧求解阻塞的假 controller，用于证明采集不等待预览 IK。"""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.processed = 0

    def warmup(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def process(self, pose):
        del pose
        self.processed += 1
        if self.processed == 1:
            self.started.set()
            self.release.wait(timeout=5.0)
        q = np.zeros(7, dtype=np.float32)
        return SimpleNamespace(
            q=q,
            target_tcp_pose7=q.copy(),
            actual_tcp_pose7=q.copy(),
            position_error_m=0.0,
            orientation_error_deg=0.0,
            valid=True,
            failure_reason="",
            failure_reasons=(),
        )


class SlowPreviewView:
    """Real RobotIkPreview 的假 Rerun 视图。"""

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.frames = 0

    def start(self) -> None:
        self.started += 1

    def log_frame(self, **kwargs) -> None:
        del kwargs
        self.frames += 1

    def close(self) -> None:
        self.closed += 1


@contextmanager
def _scripted_keyboard(keys):
    """把脚本按键提供给 umi_capture.keyboard 上下文。"""
    iterator = iter(keys)

    def read_key():
        try:
            return next(iterator)
        except StopIteration:
            return "\x1b"

    yield read_key


class Harness:
    """集中安装入口测试的替身。"""

    def __init__(self, tmp_path, monkeypatch, *, dataset, device, keys, preview):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.dataset = dataset
        self.device = device
        self.preview = preview
        self.viewer = FakeViewer()
        self.calibration = tmp_path / "calibration.yaml"
        self.calibration.write_text("{}", encoding="utf-8")
        self.tool = _tool_yaml(tmp_path / "T_end_tcp.yaml")
        self.resume_seen: list[bool] = []
        monkeypatch.setattr(umi_capture, "verify", lambda *args, **kwargs: {})
        monkeypatch.setattr(umi_capture, "metadata", lambda *args, **kwargs: {})
        monkeypatch.setattr(umi_capture, "load_session", lambda *args, **kwargs: None)
        monkeypatch.setattr(umi_capture, "validate_resume_target", lambda *args: {})
        monkeypatch.setattr(umi_capture, "NativeUmi", lambda *args, **kwargs: device)
        monkeypatch.setattr(umi_capture, "keyboard", lambda: _scripted_keyboard(keys))
        monkeypatch.setattr(umi_capture, "_start_camera_viewer", lambda *args: self.viewer)
        monkeypatch.setattr(
            umi_capture,
            "_prepare_robot_preview",
            lambda *args: SimpleNamespace(
                runtime=None,
                tool=None,
                thresholds=None,
                live_controller=None,
                view=None,
                offline_solver=None,
            ),
        )
        monkeypatch.setattr(
            umi_capture,
            "_start_robot_preview",
            lambda setup, viewer: preview,
        )

        def open_dataset(args, snapshot, recording_metadata):
            self.resume_seen.append(bool(args.resume))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "meta").mkdir(parents=True, exist_ok=True)
            (args.output_dir / "meta/info.json").write_text(
                json.dumps({"total_episodes": dataset.num_episodes, "handumi": {}}),
                encoding="utf-8",
            )
            dataset.meta.info["handumi"] = recording_metadata
            return dataset

        monkeypatch.setattr(umi_capture, "_open_recording_dataset", open_dataset)

    def argv(self, root: Path, *extra: str):
        return [
            "--calibration",
            str(self.calibration),
            "--rig-config",
            str(self.tmp_path / "missing-rig.yaml"),
            "--output-dir",
            str(root),
            "--robot-preview",
            "--robot-tool-calibration",
            str(self.tool),
            *extra,
        ]


def _install_fake_recorder(monkeypatch, controller: EntryIkController):
    """让入口使用真实附加记录器 + 假控制器。"""

    def make_recorder(root, setup):
        del setup
        return RobotIkEpisodeRecorder(root, lambda: controller)

    monkeypatch.setattr(umi_capture, "_create_robot_ik_recorder", make_recorder)


def test_entry_start_save_publishes_and_uses_capture_time_units(
    tmp_path, monkeypatch, capsys
) -> None:
    """start/save 顺序、帧数、NPZ 秒单位与预览纳秒单位均正确。"""
    base = time.monotonic_ns()
    samples = [_sample(index, base) for index in range(4)]
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice(samples),
        keys=[" ", "", " ", "\x1b"],
        preview=preview,
    )
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)

    root = tmp_path / "dataset"
    umi_capture.main(harness.argv(root))
    output = capsys.readouterr().out

    assert dataset.finalized
    assert len(dataset.saved_frames) == 1
    assert len(dataset.saved_frames[0]) == 2
    assert dataset.clears == 0
    planned_target = int(
        dataset.saved_frames[0][0]["observation.sync.target_time_ns"][0]
    )
    assert planned_target == samples[0][3]
    assert harness.viewer.disconnected == 1
    assert preview.closed == 1
    events = {event[0] for event in preview.events}
    assert {"begin_episode", "submit", "pause", "cancel_episode", "resume"} <= events
    begin = next(event for event in preview.events if event[0] == "begin_episode")
    submit = next(event for event in preview.events if event[0] == "submit")
    captures = [image.capture_time_ns for image, *_ in samples]
    targets = [target for *_, target, _ in samples]
    assert begin[2] == captures[0]
    assert submit[1] == captures[1]
    assert begin[2] not in targets
    assert submit[1] not in targets

    paths = root / "robot_ik"
    npz = paths / "episode_000000.npz"
    assert npz.is_file(), output
    assert (paths / "episode_000000.json").is_file()
    with np.load(npz) as data:
        assert data["timestamp_s"].shape == (2,)
        np.testing.assert_allclose(
            data["timestamp_s"],
            np.asarray(captures[:2], dtype=np.float64) / 1e9,
        )
        assert not np.any(
            np.abs(data["timestamp_s"] - np.asarray(targets[:2]) / 1e9) < 1e-3
        )
    assert "完整 1 段，未完成 0 段" in output


def test_entry_preflight_failure_preserves_existing_output(
    tmp_path, monkeypatch
) -> None:
    """模型/网格预检失败发生在覆盖目录之前，不碰硬件。"""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}", encoding="utf-8")
    tool = _tool_yaml(tmp_path / "T_end_tcp.yaml")
    monkeypatch.setattr(umi_capture, "verify", lambda *args, **kwargs: {})
    monkeypatch.setattr(umi_capture, "metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *args, **kwargs: None)

    def fail(*args, **kwargs):
        raise RuntimeError("mesh preflight failed")

    def unexpected(*args, **kwargs):
        pytest.fail("预检失败后不得连接硬件/创建数据集")

    monkeypatch.setattr(umi_capture, "_prepare_robot_preview", fail)
    monkeypatch.setattr(umi_capture, "_prepare_output_directory", unexpected)
    monkeypatch.setattr(umi_capture, "NativeUmi", unexpected)
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", unexpected)
    with pytest.raises(SystemExit) as error:
        umi_capture.main(
            [
                "--calibration",
                str(calibration),
                "--rig-config",
                str(tmp_path / "missing-rig.yaml"),
                "--output-dir",
                str(root),
                "--robot-preview",
                "--robot-tool-calibration",
                str(tool),
            ]
        )
    assert error.value.code == 2
    assert (root / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_entry_rerun_dependency_checked_before_output_overwrite(
    tmp_path, monkeypatch
) -> None:
    """Rerun API 缺失在覆盖目录前报告，不进入模型/硬件阶段。"""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}", encoding="utf-8")
    tool = _tool_yaml(tmp_path / "T_end_tcp.yaml")
    monkeypatch.setattr(umi_capture, "verify", lambda *args, **kwargs: {})
    monkeypatch.setattr(umi_capture, "metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *args, **kwargs: None)

    def missing(*args, **kwargs):
        raise RuntimeError("missing rerun API")

    monkeypatch.setattr(
        "handumi.visualize.robot.validate_rerun_api", missing
    )
    monkeypatch.setattr(umi_capture, "_prepare_output_directory", missing)
    with pytest.raises(SystemExit) as error:
        umi_capture.main(
            [
                "--calibration",
                str(calibration),
                "--rig-config",
                str(tmp_path / "missing-rig.yaml"),
                "--output-dir",
                str(root),
                "--robot-preview",
                "--robot-tool-calibration",
                str(tool),
            ]
        )
    assert error.value.code == 2
    assert (root / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_entry_dry_run_has_no_viewer_hardware_threads_or_directory_changes(
    tmp_path, monkeypatch
) -> None:
    """--dry-run 在机器人预览下只做参数/标定预检，不启动任何运行资源。"""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep.txt").write_text("keep", encoding="utf-8")
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}", encoding="utf-8")
    tool = _tool_yaml(tmp_path / "T_end_tcp.yaml")
    monkeypatch.setattr(umi_capture, "verify", lambda *args, **kwargs: {})
    monkeypatch.setattr(umi_capture, "metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *args, **kwargs: None)

    def unexpected(*args, **kwargs):
        pytest.fail("dry-run 不得启动 viewer、硬件、线程或修改目录")

    for name in (
        "_prepare_robot_preview",
        "_prepare_output_directory",
        "_start_camera_viewer",
        "_start_robot_preview",
        "_open_recording_dataset",
        "NativeUmi",
        "keyboard",
    ):
        monkeypatch.setattr(umi_capture, name, unexpected)
    umi_capture.main(
        [
            "--calibration",
            str(calibration),
            "--rig-config",
            str(tmp_path / "missing-rig.yaml"),
            "--output-dir",
            str(root),
            "--robot-preview",
            "--robot-tool-calibration",
            str(tool),
            "--dry-run",
        ]
    )
    assert (root / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_entry_prepare_failure_saves_original_and_reports_incomplete(
    tmp_path, monkeypatch, capsys
) -> None:
    """附加记录补算失败不得丢原始 episode，最终摘要必须标未完成。"""
    base = time.monotonic_ns()
    samples = [_sample(index, base) for index in range(4)]
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice(samples),
        keys=[" ", "", " ", "\x1b"],
        preview=preview,
    )
    controller = EntryIkController(fail=True)
    _install_fake_recorder(monkeypatch, controller)

    root = tmp_path / "dataset"
    umi_capture.main(harness.argv(root))
    output = capsys.readouterr().out
    assert len(dataset.saved_frames) == 1
    assert len(dataset.saved_frames[0]) == 2
    assert not (root / "robot_ik/episode_000000.npz").exists()
    assert "附加记录未完成" in output
    assert "完整 0 段，未完成 1 段" in output


def test_entry_waiting_does_not_start_preview(
    tmp_path, monkeypatch
) -> None:
    """等待录制时按 R 也不会启动实时计算。"""
    base = time.monotonic_ns()
    samples = [_sample(0, base), _sample(1, base)]
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice(samples),
        keys=["r", ""],
        preview=preview,
    )
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)
    umi_capture.main(harness.argv(tmp_path / "dataset"))
    submits = [event for event in preview.events if event[0] == "submit"]
    assert not submits
    assert dataset.saved_frames == []


def test_entry_record_automatically_starts_preview_and_publishes_ik(
    tmp_path, monkeypatch, capsys
) -> None:
    """空格自动启动实时预览，保存后停止提交并发布补算结果。"""
    base = time.monotonic_ns()
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path, monkeypatch, dataset=dataset,
        device=FakeDevice([_sample(i, base, healthy=i != 5) for i in range(8)]),
        keys=["", " ", "", " ", " ", "", "", "\x1b"], preview=preview,
    )
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)
    root = tmp_path / "dataset"
    umi_capture.main(harness.argv(root))
    assert sum(event[0] == "begin_episode" for event in preview.events) == 2
    assert sum(event[0] == "submit" for event in preview.events) == 1
    assert (root / "robot_ik/episode_000000.npz").exists()
    assert "按空格开始录制时启动实时 IK" in capsys.readouterr().out


def test_entry_preview_only_runs_during_episodes(
    tmp_path, monkeypatch, capsys
) -> None:
    """等待期间忽略 R，保存或丢弃后停止，下次录制重新启动。"""
    base = time.monotonic_ns()
    samples = [_sample(i, base, healthy=i != 6) for i in range(9)]
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path, monkeypatch, dataset=dataset, device=FakeDevice(samples),
        keys=["R", "r", " ", "", " ", " ", "", "", "\x1b"],
        preview=preview,
    )
    _install_fake_recorder(monkeypatch, EntryIkController())
    umi_capture.main(harness.argv(tmp_path / "dataset"))
    begins = [event for event in preview.events if event[0] == "begin_episode"]
    assert [event[2] for event in begins] == [
        samples[2][0].capture_time_ns, samples[5][0].capture_time_ns
    ]
    submits = [event[1] for event in preview.events if event[0] == "submit"]
    assert samples[0][0].capture_time_ns not in submits
    assert samples[1][0].capture_time_ns not in submits
    assert samples[7][0].capture_time_ns not in submits
    assert submits == [samples[3][0].capture_time_ns]
    assert sum(event[0] == "cancel_episode" for event in preview.events) == 2


def test_entry_discard_clears_robot_ik_cache_without_saving(
    tmp_path, monkeypatch
) -> None:
    """无效传感器帧丢弃片段时清理附加记录缓存，不发布文件。"""
    base = time.monotonic_ns()
    samples = [
        _sample(0, base),
        _sample(1, base),
        _sample(2, base, healthy=False),
        _sample(3, base),
    ]
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice(samples),
        keys=[" ", "", "", "\x1b"],
        preview=preview,
    )
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)
    root = tmp_path / "dataset"
    umi_capture.main(harness.argv(root))
    assert dataset.saved_frames == []
    assert dataset.clears >= 1
    assert not (root / "robot_ik/episode_000000.npz").exists()
    assert ("cancel_episode",) in preview.events


def test_entry_slow_solver_does_not_block_frame_capture(
    tmp_path, monkeypatch
) -> None:
    """后台首帧求解阻塞时，采集仍按顺序追加帧并保存原始片段。"""
    base = time.monotonic_ns()
    samples = [_sample(index, base) for index in range(6)]
    dataset = FakeDataset(tmp_path / "dataset")
    slow = SlowController()
    slow_view = SlowPreviewView()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice(samples),
        keys=[" ", "", "", "", " ", "\x1b"],
        preview=FakePreview(),
    )

    def start_real_preview(setup, viewer):
        del setup, viewer
        preview = RobotIkPreview(slow, slow_view, display_hz=15.0)
        preview.start()
        return preview

    monkeypatch.setattr(umi_capture, "_start_robot_preview", start_real_preview)
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)
    timer = threading.Timer(1.0, slow.release.set)
    timer.start()
    try:
        umi_capture.main(harness.argv(tmp_path / "dataset"))
    finally:
        timer.cancel()
        slow.release.set()
    assert slow.started.is_set()
    assert len(dataset.saved_frames) == 1
    assert len(dataset.saved_frames[0]) == 4
    # 首帧 IK 阻塞 1 秒；四帧采集应在阻塞结束前完成（约 0.1 秒）。
    span = dataset.add_times[-1] - dataset.add_times[0]
    assert span < 0.75
    assert slow_view.closed == 1


def test_entry_publish_failure_keeps_staging_and_reports_incomplete(
    tmp_path, monkeypatch, capsys
) -> None:
    """原始保存成功后发布失败：不删暂存、不显示完整成功。"""
    base = time.monotonic_ns()
    samples = [_sample(index, base) for index in range(4)]
    dataset = FakeDataset(tmp_path / "dataset")
    preview = FakePreview()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice(samples),
        keys=[" ", "", " ", ""],
        preview=preview,
    )
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)

    def fail_publish(self, episode_index=None):
        del self, episode_index
        raise RobotIkPublishError(
            "simulated publish failure",
            episode_index=0,
            recovery={"recovery_file": None},
            recovery_path=None,
        )

    monkeypatch.setattr(RobotIkEpisodeRecorder, "publish_episode", fail_publish)
    root = tmp_path / "dataset"
    umi_capture.main(harness.argv(root))
    output = capsys.readouterr().out
    assert len(dataset.saved_frames) == 1
    assert "附加记录未完成" in output
    assert "完整 0 段，未完成 1 段" in output
    staging = root / "robot_ik/.staging"
    assert list(staging.glob("*.npz"))
    assert list(staging.glob("*.json"))
    assert {record.kind for record in find_incomplete_records(root)} == {"staged"}


def test_entry_resume_preserves_records_and_warns_incomplete(
    tmp_path, monkeypatch, capsys
) -> None:
    """续录保留既有目录/遗留记录并给出启动警告。"""
    root = tmp_path / "dataset"
    staging = root / "robot_ik/.staging"
    staging.mkdir(parents=True)
    (staging / "episode_000000-deadbeef.npz").write_bytes(b"staged")
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(
        json.dumps({"total_episodes": 1, "handumi": {}}), encoding="utf-8"
    )
    base = time.monotonic_ns()
    dataset = FakeDataset(root)
    preview = FakePreview()
    harness = Harness(
        tmp_path,
        monkeypatch,
        dataset=dataset,
        device=FakeDevice([_sample(0, base)]),
        keys=["\x1b"],
        preview=preview,
    )
    controller = EntryIkController()
    _install_fake_recorder(monkeypatch, controller)
    umi_capture.main(harness.argv(root, "--resume"))
    output = capsys.readouterr().out
    assert harness.resume_seen == [True]
    assert "未完成的 robot_ik" in output
    assert (staging / "episode_000000-deadbeef.npz").is_file()
    assert "完整 0 段，未完成 0 段" in output


def test_camera_viewer_uses_explicit_stream_and_combined_blueprint(
    monkeypatch,
) -> None:
    """采集窗口使用显式记录流；机器人开关时发送合并布局，不启动第二应用。"""
    import rerun as rr

    class FakeRecordingStream:
        def __init__(self, application_id):
            self.application_id = application_id
            self.spawned = 0
            self.blueprint = None
            self.logs = []
            self.times = []
            self.flushed = 0
            self.disconnected = 0

        def log(self, entity_path, entity, *, static=False):
            self.logs.append((entity_path, entity, static))

        def set_time(self, timeline, **kwargs):
            self.times.append((timeline, kwargs))

        def send_blueprint(self, blueprint, **kwargs):
            self.blueprint = blueprint

        def spawn(self):
            self.spawned += 1

        def flush(self):
            self.flushed += 1

        def disconnect(self):
            self.disconnected += 1

    stream = FakeRecordingStream("handumi_umi")
    monkeypatch.setattr(rr, "RecordingStream", lambda application_id: stream)
    viewer = umi_capture._start_camera_viewer(
        SimpleNamespace(no_preview=False, robot_preview=True),
        None,
    )
    assert viewer.stream is stream
    assert stream.spawned == 1
    assert stream.blueprint is not None
    viewer.disconnect()
    assert stream.flushed == 1
    assert stream.disconnected == 1


def test_entry_rejects_conflicting_or_incomplete_robot_options_before_startup(
    tmp_path, monkeypatch
) -> None:
    """冲突开关、缺工具路径和非法阈值都在启动前以退出码 2 拒绝。"""
    root = tmp_path / "dataset"
    root.mkdir()
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}", encoding="utf-8")
    tool = _tool_yaml(tmp_path / "T_end_tcp.yaml")

    def unexpected(*args, **kwargs):
        pytest.fail("非法机器人参数不得进入启动流程")

    for name in (
        "verify",
        "_check_output_directory",
        "_prepare_output_directory",
        "NativeUmi",
    ):
        monkeypatch.setattr(umi_capture, name, unexpected)
    base = [
        "--calibration",
        str(calibration),
        "--rig-config",
        str(tmp_path / "missing-rig.yaml"),
        "--output-dir",
        str(root),
    ]
    cases = [
        [*base, "--robot-preview", "--no-preview", "--robot-tool-calibration", str(tool)],
        [*base, "--robot-preview"],
        [*base, "--robot-preview", "--robot-tool-calibration", str(tool), "--robot-position-tolerance-m", "-1"],
        [*base, "--robot-tool-calibration", str(tool)],
    ]
    for argv in cases:
        with pytest.raises(SystemExit) as error:
            umi_capture.main(argv)
        assert error.value.code == 2
