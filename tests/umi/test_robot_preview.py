"""机器人 Rerun 网格显示和后台预览线程的合成测试。"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from handumi.robots.registry import load_embodiment
from handumi.scripts._umi_robot_preview import RobotIkPreview
from handumi.visualize.robot import RerunRobotView, capture_blueprint, load_robot_meshes


class FakeStream:
    """记录 Rerun 调用，不启动查看器。"""

    def __init__(self) -> None:
        self.logs: list[tuple[str, object, bool]] = []
        self.times: list[tuple[str, int]] = []
        self.blueprint = None
        self.spawned = 0
        self.flushed = 0
        self.disconnected = 0

    def log(self, entity_path, entity, *, static=False) -> None:
        self.logs.append((entity_path, entity, bool(static)))

    def set_time(self, timeline, **kwargs) -> None:
        self.times.append((timeline, dict(kwargs)))

    def send_blueprint(self, blueprint, **kwargs) -> None:
        self.blueprint = blueprint

    def spawn(self) -> None:
        self.spawned += 1

    def flush(self) -> None:
        self.flushed += 1

    def disconnect(self) -> None:
        self.disconnected += 1


class FakeView:
    """记录预览提交时间与结果的假显示层。"""

    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.frames: list[dict] = []

    def start(self) -> None:
        self.started += 1

    def log_frame(self, **kwargs) -> None:
        self.frames.append(kwargs)

    def close(self) -> None:
        self.closed += 1


class BlockingController:
    """可控阻塞/记录的假 RelativeToolIk。"""

    def __init__(self) -> None:
        self.warmups = 0
        self.resets = 0
        self.processed: list[np.ndarray | None] = []
        self.events: list[str | float] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.processed_event = threading.Event()

        def noop_reset() -> None:
            self.resets += 1
            self.events.append("reset")

        self.reset = noop_reset

    def warmup(self) -> None:
        self.warmups += 1

    def process(self, source):
        self.started.set()
        self.processed.append(None if source is None else np.asarray(source).copy())
        self.events.append(0.0 if source is None else float(source[0]))
        self.release.wait(timeout=5.0)
        self.processed_event.set()
        value = 0.0 if source is None else float(source[0])
        return SimpleNamespace(
            q=np.full(7, value, dtype=np.float32),
            target_tcp_pose7=np.full(7, value, dtype=np.float32),
            actual_tcp_pose7=np.full(7, value, dtype=np.float32),
            valid=True,
            position_error_m=0.0,
            orientation_error_deg=0.0,
            failure_reason="",
        )


def _wait_until(predicate, timeout_s: float = 3.0) -> None:
    """等待线程状态，避免固定 sleep 导致不稳定。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("等待条件超时")


def test_rerun_robot_view_logs_meshes_once_and_transforms_per_frame() -> None:
    """网格静态记录一次；每帧每个连杆只记录一个父级 Transform3D。"""
    runtime = load_embodiment("rm75b")
    stream = FakeStream()
    view = RerunRobotView(runtime, stream=stream)
    view.load()
    assert view.mesh_count == 8
    view.start()
    assert stream.blueprint is not None
    mesh_logs = [
        (path, entity)
        for path, entity, static in stream.logs
        if static and type(entity).__name__ == "Mesh3D"
    ]
    assert len(mesh_logs) == view.mesh_count
    assert len({path for path, _ in mesh_logs}) == view.mesh_count
    initial_transforms = [
        path for path, entity, static in stream.logs
        if not static and type(entity).__name__ == "Transform3D"
    ]
    assert len(set(initial_transforms)) == view.mesh_count
    stream.logs[:] = [entry for entry in stream.logs if entry[2]]
    stream.times.clear()

    home = runtime.home_q()
    _, home_end = runtime.solver_cls().fk_pose7(home)
    target = home_end.copy()
    actual = home_end.copy()
    view.log_frame(
        sample_time_ns=123_456_789,
        q=home,
        target_tcp_pose7=target,
        actual_tcp_pose7=actual,
        valid=True,
        position_error_m=0.001,
        orientation_error_deg=0.5,
    )
    assert len(stream.times) == 1
    assert stream.times[0][0] == "capture"
    assert stream.times[0][1]["duration"] == pytest.approx(123_456_789 / 1e9)
    transform_logs = [
        (path, entity)
        for path, entity, static in stream.logs
        if not static and type(entity).__name__ == "Transform3D"
        and path.startswith("/robot/link/")
    ]
    assert len(transform_logs) == view.mesh_count
    assert len({path for path, _ in transform_logs}) == view.mesh_count
    assert any(path == "/robot/target_tcp" for path, _, _ in stream.logs)
    assert any(path == "/robot/actual_tcp" for path, _, _ in stream.logs)
    assert any(path == "/robot/status/valid" for path, _, _ in stream.logs)
    assert any(path == "/robot/status_text" for path, _, _ in stream.logs)
    # marker 是 TCP Transform3D 的子节点，位置必须为局部原点，否则会被
    # 父变换二次平移。
    marker_points = [
        entity
        for path, entity, _ in stream.logs
        if path == "/robot/target_tcp/marker"
        and type(entity).__name__ == "Points3D"
    ]
    assert len(marker_points) == 1
    assert marker_points[0].positions.as_arrow_array().to_pylist() == [[0.0, 0.0, 0.0]]

    # 第二帧只追加连杆/状态变换，不重复静态网格。
    view.log_frame(
        sample_time_ns=123_456_999,
        q=home,
        target_tcp_pose7=target,
        actual_tcp_pose7=actual,
        valid=False,
        position_error_m=0.02,
        orientation_error_deg=8.0,
        failure_reason="position_error",
    )
    assert sum(
        1
        for _, entity, static in stream.logs
        if static and type(entity).__name__ == "Mesh3D"
    ) == view.mesh_count
    view.close()
    assert stream.flushed == 1
    # 注入的 stream 不归 view 所有，不主动断开外部记录流。
    assert stream.disconnected == 0


def test_capture_and_robot_share_stream_and_capture_timeline() -> None:
    """相机与机器人写同一记录流的 capture 时间轴，使用实际样本时间。"""
    from handumi.scripts.umi_capture import _RerunCaptureViewer

    runtime = load_embodiment("rm75b")
    stream = FakeStream()
    capture_view = _RerunCaptureViewer(stream, __import__("rerun"))
    robot_view = RerunRobotView(runtime, stream=stream)
    robot_view.start()

    capture_samples_ns = 111_000_000_000
    stream.times.clear()
    robot_sample_ns = 222_000_000_000
    capture_view.set_time("capture", duration=capture_samples_ns / 1e9)
    robot_view.log_frame(
        sample_time_ns=robot_sample_ns,
        q=runtime.home_q(),
        target_tcp_pose7=np.zeros(7, dtype=np.float32),
        actual_tcp_pose7=np.zeros(7, dtype=np.float32),
        valid=True,
        position_error_m=0.0,
        orientation_error_deg=0.0,
    )
    assert [timeline for timeline, _ in stream.times] == ["capture", "capture"]
    assert stream.times[0][1]["duration"] == pytest.approx(capture_samples_ns / 1e9)
    assert stream.times[1][1]["duration"] == pytest.approx(robot_sample_ns / 1e9)
    robot_view.close()


def test_robot_view_binds_shared_stream_without_second_app_or_blueprint() -> None:
    """机器人视图复用现有显式记录流，不启动第二个应用/覆盖相机布局。"""
    runtime = load_embodiment("rm75b")
    stream = FakeStream()
    view = RerunRobotView(runtime, stream=stream)
    view.bind_stream(stream, send_blueprint=False)
    view.start()
    assert stream.spawned == 0
    assert stream.blueprint is None
    assert any(
        type(entity).__name__ == "Mesh3D"
        for _, entity, static in stream.logs
        if static
    )
    view.close()
    assert stream.disconnected == 0


def test_real_rerun_rrd_accepts_shared_capture_timeline(tmp_path) -> None:
    """真实 Rerun SDK 无 GUI 写入临时 RRD：相机与机器人同流同轴。"""
    import rerun as rr
    import rerun.blueprint as rrb

    from handumi.scripts.umi_capture import _RerunCaptureViewer

    runtime = load_embodiment("rm75b")
    path = tmp_path / "robot_preview.rrd"
    stream = rr.RecordingStream("handumi_robot_capture_test")
    stream.save(str(path))
    stream.send_blueprint(capture_blueprint(rrb), make_active=True, make_default=True)
    capture_view = _RerunCaptureViewer(stream, rr)
    robot_view = RerunRobotView(runtime, stream=stream)
    robot_view.start()
    capture_view.set_time("capture", duration=1.25)
    capture_view.log(
        "camera/right_wrist",
        rr.Image(np.zeros((2, 2, 3), dtype=np.uint8)),
        static=True,
    )
    pose = np.zeros(7, dtype=np.float32)
    pose[6] = 1.0
    robot_view.log_frame(
        sample_time_ns=1_250_000_000,
        q=runtime.home_q(),
        target_tcp_pose7=pose,
        actual_tcp_pose7=pose,
        valid=True,
        position_error_m=0.0,
        orientation_error_deg=0.0,
    )
    stream.flush()
    robot_view.close()
    stream.disconnect()
    assert path.is_file() and path.stat().st_size > 0


def _iter_blueprint_views(container):
    """递归收集蓝图中的实际视图对象。"""
    if hasattr(container, "class_identifier"):
        return [container]
    views = []
    for item in getattr(container, "contents", ()):
        views.extend(_iter_blueprint_views(item))
    return views


def test_capture_blueprint_shows_gripper_health_and_default_side_panels() -> None:
    """布局实际包含相机/tracking/机器人/夹爪/健康/IK，且不折叠侧栏。"""
    import rerun.blueprint as rrb

    blueprint = capture_blueprint(rrb).to_blueprint()
    views = {
        view.origin: (view.class_identifier, view.name)
        for view in _iter_blueprint_views(blueprint.root_container)
    }
    assert views == {
        "/camera/right_wrist": ("2D", "Camera"),
        "/tracking": ("3D", "Raw tracking"),
        "/robot": ("3D", "RM75B preview"),
        "/gripper": ("TimeSeries", "Gripper openness"),
        "/health": ("TimeSeries", "Capture health"),
        "/robot/status": ("TimeSeries", "IK status"),
        "/robot/status_text": ("TextLog", "Robot state"),
    }
    # 旧采集窗口的左侧栏必须显式恢复为展开，而不是依赖默认状态。
    assert blueprint.blueprint_panel is not None
    assert blueprint.blueprint_panel.state == rrb.PanelState.Expanded
    assert blueprint.collapse_panels is False


def test_rerun_robot_view_rejects_bad_mesh_runtime() -> None:
    """没有可视化网格的运行时在加载阶段明确失败。"""

    class NoMeshRuntime:
        def load_urdf(self, *, load_meshes):
            del load_meshes
            return SimpleNamespace(scene=None)

    with pytest.raises(ValueError, match="网格"):
        load_robot_meshes(NoMeshRuntime())


def test_preview_start_and_close_stop_thread() -> None:
    """启动预热一次，关闭可靠停止后台线程且可重复调用。"""
    controller = BlockingController()
    view = FakeView()
    preview = RobotIkPreview(controller, view)
    preview.start()
    assert controller.warmups == 1
    assert view.started == 1
    thread = preview._thread
    assert thread is not None and thread.is_alive()
    preview.close()
    assert not thread.is_alive()
    assert view.closed == 1
    preview.close()
    assert view.closed == 1


def test_close_invalidates_inflight_result_and_never_logs_to_closed_view() -> None:
    """close 超时后慢 solver 返回，不得向已关闭 view 写迟到日志。"""
    controller = BlockingController()
    controller.release.clear()
    view = FakeView()
    preview = RobotIkPreview(controller, view, display_hz=1000.0)
    preview.start()
    thread = preview._thread
    try:
        preview.begin_episode(np.full(7, 1, dtype=np.float32), 100)
        _wait_until(lambda: controller.started.is_set())
        preview.close(timeout_s=0.0)
        assert view.closed == 1
        assert thread is not None and thread.is_alive()
        controller.release.set()
        _wait_until(lambda: not thread.is_alive())
        assert view.frames == []
    finally:
        controller.release.set()
        preview.close()


def test_preview_anchor_wins_over_inflight_latest_sample() -> None:
    """等待帧在途时开始新片段，首帧锚点仍先于最新帧处理。"""
    controller = BlockingController()
    view = FakeView()
    preview = RobotIkPreview(controller, view, display_hz=1000.0)
    preview.start()
    try:
        # 先制造一个在途等待帧，使 worker 停在 process 内。
        preview.submit(50, np.full(7, 9, dtype=np.float32))
        _wait_until(lambda: controller.started.is_set())
        preview.begin_episode(np.full(7, 1, dtype=np.float32), 100)
        preview.submit(101, np.full(7, 2, dtype=np.float32))
        preview.submit(102, np.full(7, 3, dtype=np.float32))
        controller.release.set()
        _wait_until(lambda: controller.processed and controller.processed[0] is not None)
        # 旧等待帧是 gen0；新代次 reset 后的第一帧必须是锚点 1，而不是 2/3。
        _wait_until(lambda: 1.0 in controller.events)
        reset_index = max(
            index for index, event in enumerate(controller.events) if event == "reset"
        )
        assert controller.events[reset_index + 1] == 1.0
        assert controller.resets >= 1
    finally:
        controller.release.set()
        preview.close()
    assert view.closed == 1


def test_preview_discards_stale_generation_result() -> None:
    """求解期间进入新片段，旧代次结果不得提交给显示层。"""
    controller = BlockingController()
    view = FakeView()
    preview = RobotIkPreview(controller, view, display_hz=1000.0)
    controller.release.clear()
    preview.start()
    try:
        preview.begin_episode(np.full(7, 1, dtype=np.float32), 100)
        _wait_until(lambda: controller.started.is_set())
        preview.begin_episode(np.full(7, 2, dtype=np.float32), 200)
        controller.release.set()
        _wait_until(lambda: len(controller.processed) >= 2)
        _wait_until(lambda: any(frame["sample_time_ns"] == 200 for frame in view.frames))
        assert all(frame["sample_time_ns"] != 100 for frame in view.frames)
    finally:
        controller.release.set()
        preview.close()


def test_preview_submit_is_nonblocking_while_solver_is_slow() -> None:
    """容量一 submit 不等待慢 solver；只保留最新预览帧。"""
    controller = BlockingController()
    view = FakeView()
    preview = RobotIkPreview(controller, view, display_hz=1000.0)
    preview.start()
    try:
        preview.begin_episode(np.full(7, 1, dtype=np.float32), 100)
        _wait_until(lambda: controller.started.is_set())
        started = time.monotonic()
        for index in range(30):
            preview.submit(100 + index, np.full(7, index, dtype=np.float32))
        elapsed = time.monotonic() - started
        assert elapsed < 0.25
        assert preview.dropped_samples > 0
    finally:
        controller.release.set()
        preview.close()


def test_preview_display_rate_limit_uses_sample_time() -> None:
    """显示最多 15 Hz，但 Rerun 时间使用实际求解样本时间。"""
    controller = BlockingController()
    view = FakeView()
    fake_clock = [0.0]
    preview = RobotIkPreview(
        controller,
        view,
        display_hz=15.0,
        clock=lambda: fake_clock[0],
    )
    preview.begin_waiting()
    generation = preview.generation
    for index in range(6):
        fake_clock[0] = index / 30.0
        preview._process(
            generation,
            1000 + index,
            np.full(7, index, dtype=np.float32),
        )
    # 30 Hz 输入按 15 Hz 显示：索引 0、2、4 三帧，时间戳保持样本时间。
    assert len(view.frames) == 3
    assert [frame["sample_time_ns"] for frame in view.frames] == [1000, 1002, 1004]
    preview.close()
