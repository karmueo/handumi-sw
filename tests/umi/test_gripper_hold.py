"""验证 ArUco 短暂漏检时按相机采集时间保持最近成功开度。"""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.cameras.base import CameraSample
from handumi.feetech import diagnostics
from handumi.grippers.base import VisualGripperProvider
from handumi.scripts import umi_capture
from handumi.scripts.umi_capture import NativeUmi, transition

FRAME = np.zeros((2, 2, 3), dtype=np.uint8)
T0 = 1_000_000_000
HOLD_NS = 500_000_000


@pytest.fixture
def diagnostic_path(tmp_path, monkeypatch):
    """为每个测试创建独立诊断日志，并关闭文件句柄。"""
    path = tmp_path / "diagnostics.jsonl"
    monkeypatch.setenv("HANDUMI_FEETECH_DIAGNOSTICS", str(path))
    diagnostics._diagnostic_logger.cache_clear()
    yield path
    logger = diagnostics._diagnostic_logger()
    if logger:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    diagnostics._diagnostic_logger.cache_clear()


def _estimate(openness):
    """构造带常见诊断字段的视觉估计结果。"""
    return SimpleNamespace(
        openness=openness,
        raw_openness=openness,
        distance_mm=70.0,
        left_pose=SimpleNamespace(translation_mm=(1, 2, 3)),
        right_pose=SimpleNamespace(translation_mm=(71, 2, 3)),
        roi=(0, 0, 1, 1),
    )


def _make_provider(*results, **kwargs):
    """按顺序返回估计结果的提供者，保留实际检测调用次数。"""
    estimator = SimpleNamespace(
        estimate=Mock(side_effect=list(results)),
        last_detected_marker_count=2,
        closed_distance_mm=40,
        open_distance_mm=120,
    )
    return VisualGripperProvider(estimator, **kwargs), estimator


def _observe(provider, capture_time_ns, sequence, offset_ns=0):
    """用指定相机采集时间输入一帧。"""
    return provider.observe(
        CameraSample(FRAME, capture_time_ns, sequence), offset_ns=offset_ns
    )


def _records(path):
    """读取诊断 JSONL 记录。"""
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_first_frame_failure_stays_invalid(diagnostic_path):
    """尚无成功观测时，漏检帧没有可保持的开度。"""
    provider, estimator = _make_provider(None)

    result = _observe(provider, T0, 1)

    assert not result.valid
    assert result.normalized == 0
    assert (result.sample_time_ns, result.sequence) == (T0, 1)
    assert "未检测到" in result.reason
    assert provider.hold_timeout_ns == HOLD_NS
    assert estimator.estimate.call_count == 1
    record = _records(diagnostic_path)[0]
    assert record["outcome"] == "invalid"
    assert record["detected"] is False
    assert record["last_success_time_ns"] is None
    assert record["last_success_age_ns"] is None
    assert record["hold_timeout_ns"] == HOLD_NS


def test_hold_at_exact_deadline_uses_current_frame_identity(diagnostic_path):
    """默认恰好 500 毫秒仍可保持，但帧序号和校正时间属于当前帧。"""
    provider, _ = _make_provider(_estimate(0.7), None)
    _observe(provider, T0, 1)

    held = _observe(provider, T0 + HOLD_NS, 2, offset_ns=25)

    assert held.valid and held.normalized == 0.7
    assert held.sequence == 2
    assert held.sample_time_ns == T0 + HOLD_NS + 25
    assert held.width_m is None
    assert "保持" in held.reason
    record = _records(diagnostic_path)[1]
    assert record["outcome"] == "held"
    assert record["detected"] is False
    assert record["valid"] is True
    assert record["normalized"] == 0.7
    assert record["capture_time_ns"] == T0 + HOLD_NS
    assert record["sample_time_ns"] == T0 + HOLD_NS + 25
    assert record["last_success_time_ns"] == T0
    assert record["last_success_age_ns"] == HOLD_NS
    assert record["hold_timeout_ns"] == HOLD_NS
    assert record["distance_mm"] is None
    assert record["left_position_mm"] is None
    assert record["right_position_mm"] is None


def test_expiry_and_prior_holds_do_not_extend_deadline(diagnostic_path):
    """保持复用不刷新成功时间，超过默认 500 毫秒后仍立即失效。"""
    provider, _ = _make_provider(_estimate(0.6), None, None, None)
    _observe(provider, T0, 1)
    assert _observe(provider, T0 + 150_000_000, 2).valid
    assert _observe(provider, T0 + HOLD_NS, 3).valid

    expired = _observe(provider, T0 + HOLD_NS + 1, 4)

    assert not expired.valid
    assert expired.normalized == 0
    assert "保持期限" in expired.reason
    records = _records(diagnostic_path)
    assert [row["outcome"] for row in records] == [
        "measured", "held", "held", "invalid",
    ]
    assert records[3]["last_success_time_ns"] == T0
    assert records[3]["last_success_age_ns"] == HOLD_NS + 1
    assert records[3]["hold_timeout_ns"] == HOLD_NS


@pytest.mark.parametrize("hold_ns", [50_000_000, 200_000_000], ids=["50ms", "200ms"])
def test_custom_timeout_boundary(diagnostic_path, hold_ns):
    """自定义较短期限精确保持到边界，并保留原有 200 毫秒边界。"""
    provider, _ = _make_provider(
        _estimate(0.65), None, None, hold_timeout_ns=hold_ns
    )
    _observe(provider, T0, 1)

    held = _observe(provider, T0 + hold_ns, 2)
    expired = _observe(provider, T0 + hold_ns + 1, 3)

    assert held.valid and held.normalized == 0.65
    assert not expired.valid and expired.normalized == 0
    records = _records(diagnostic_path)
    assert records[1]["hold_timeout_ns"] == hold_ns
    assert records[1]["last_success_age_ns"] == hold_ns
    assert records[2]["last_success_age_ns"] == hold_ns + 1


def test_zero_timeout_fully_disables_hold(diagnostic_path):
    """零期限关闭保持，相同采集时间但不同序号的新帧也不得复用。"""
    provider, _ = _make_provider(
        _estimate(0.7), None, None, hold_timeout_ns=0
    )
    _observe(provider, T0, 1)

    same_time = _observe(provider, T0, 2)
    later = _observe(provider, T0 + 1, 3)

    assert not same_time.valid and same_time.normalized == 0
    assert not later.valid and later.normalized == 0
    records = _records(diagnostic_path)
    assert [row["outcome"] for row in records] == [
        "measured", "invalid", "invalid",
    ]
    assert records[1]["last_success_age_ns"] == 0
    assert records[1]["hold_timeout_ns"] == 0


def test_recovery_replaces_held_value(diagnostic_path):
    """恢复检测后立即使用新估计，下一次保持以新成功时间为基准。"""
    provider, estimator = _make_provider(
        _estimate(0.8), None, _estimate(0.2), None
    )
    _observe(provider, T0, 1)
    assert _observe(provider, T0 + 100_000_000, 2).normalized == 0.8

    recovered = _observe(provider, T0 + 150_000_000, 3)

    assert recovered.valid and recovered.normalized == 0.2
    held = _observe(provider, T0 + 200_000_000, 4)
    assert held.valid and held.normalized == 0.2
    assert estimator.estimate.call_count == 4
    records = _records(diagnostic_path)
    assert [row["outcome"] for row in records] == [
        "measured", "held", "measured", "held",
    ]
    assert records[2]["last_success_time_ns"] == T0 + 150_000_000
    assert records[3]["last_success_time_ns"] == T0 + 150_000_000
    assert records[3]["last_success_age_ns"] == 50_000_000


def test_real_zero_openness_is_measured_and_held(diagnostic_path):
    """真实测得的零开度有效，并与无效结果的占位零值区分。"""
    provider, _ = _make_provider(_estimate(0.0), None, None)

    measured = _observe(provider, T0, 1)
    held = _observe(provider, T0 + 100_000_000, 2)
    expired = _observe(provider, T0 + HOLD_NS + 1, 3)

    assert measured.valid and measured.normalized == 0
    assert measured.reason == ""
    assert held.valid and held.normalized == 0
    assert not expired.valid and expired.normalized == 0
    frame = measured.frame()
    assert frame["observation.gripper.right.valid"].tolist() == [1]
    assert frame["observation.gripper.right.normalized"].tolist() == [0]
    records = _records(diagnostic_path)
    assert [(row["outcome"], row["valid"]) for row in records] == [
        ("measured", True), ("held", True), ("invalid", False),
    ]
    assert [row["normalized"] for row in records] == [0, 0, 0]


def test_same_frame_is_cached_without_refreshing_hold(diagnostic_path):
    """同一帧重复读取复用原结果，不再次估计也不延长保持期。"""
    provider, estimator = _make_provider(_estimate(0.5), None, None)
    frame = CameraSample(FRAME, T0, 1)
    first = provider.observe(frame)
    assert provider.observe(frame) is first

    miss = CameraSample(FRAME, T0 + 100_000_000, 2)
    held = provider.observe(miss)
    assert held.valid and held.normalized == 0.5
    assert provider.observe(miss) is held
    assert estimator.estimate.call_count == 2

    expired = provider.observe(CameraSample(FRAME, T0 + HOLD_NS + 1, 3))
    assert not expired.valid
    assert estimator.estimate.call_count == 3
    assert len(_records(diagnostic_path)) == 3


def test_time_rollback_clears_hold_state(diagnostic_path):
    """相机时间相对上一输入帧倒退时清除旧开度，避免跨时间段复用。"""
    provider, _ = _make_provider(
        _estimate(0.9), None, None, _estimate(0.3), None
    )
    _observe(provider, T0, 1)
    assert _observe(provider, T0 + 100_000_000, 2).valid

    # 上一输入帧是 T0+100ms，本帧倒退到 T0+50ms；若不清除会误保持。
    rollback = _observe(provider, T0 + 50_000_000, 3)

    assert not rollback.valid
    assert rollback.normalized == 0
    recovered = _observe(provider, T0 + 60_000_000, 4)
    assert recovered.valid and recovered.normalized == 0.3
    held = _observe(provider, T0 + 160_000_000, 5)
    assert held.valid and held.normalized == 0.3
    records = _records(diagnostic_path)
    assert [row["outcome"] for row in records] == [
        "measured", "held", "invalid", "measured", "held",
    ]
    assert records[2]["last_success_time_ns"] is None
    assert records[2]["last_success_age_ns"] is None
    assert records[4]["last_success_time_ns"] == T0 + 60_000_000


class _QueueCamera:
    """按顺序返回预置帧，模拟相机采集时间。"""

    def __init__(self, samples):
        self._samples = list(samples)

    def sample_at(self, target_time_ns):
        """忽略目标时间，直接返回下一帧。"""
        return self._samples.pop(0)


def test_native_health_gate_and_transition_during_hold(diagnostic_path):
    """保持期内 NativeUmi 健康且录制继续，超期后健康失败并丢弃当前段。"""
    provider, _ = _make_provider(None, _estimate(0.4), None, None)
    samples = [
        CameraSample(FRAME, T0, 1),
        CameraSample(FRAME, T0 + 100_000_000, 2),
        CameraSample(FRAME, T0 + 100_000_000 + HOLD_NS, 3),
        CameraSample(FRAME, T0 + 100_000_000 + HOLD_NS + 1, 4),
    ]
    devices = NativeUmi.__new__(NativeUmi)
    devices.offset_ns = 0
    devices.gripper = provider
    devices.camera = _QueueCamera(samples)
    devices.tracker = SimpleNamespace(
        error=None, sample_at=lambda _: SimpleNamespace(right_tracked=True)
    )

    def sample(item):
        """让相机采集时间落在 NativeUmi 的 150 毫秒延迟窗口内。"""
        return devices.sample(item.capture_time_ns + 150_000_000)

    _, _, first, _, first_healthy = sample(samples[0])
    assert not first.valid
    assert first_healthy is False
    assert transition(True, "", first_healthy) == (False, "discard")

    _, _, measured, _, measured_healthy = sample(samples[1])
    assert measured.valid and measured.normalized == 0.4
    assert measured_healthy is True
    assert transition(True, "", measured_healthy) == (True, "frame")

    _, _, held, _, held_healthy = sample(samples[2])
    assert held.valid and held.normalized == 0.4
    assert held.sequence == 3
    assert held_healthy is True
    assert transition(True, "", held_healthy) == (True, "frame")

    _, _, expired, _, expired_healthy = sample(samples[3])
    assert not expired.valid
    assert expired_healthy is False
    assert transition(True, "", expired_healthy) == (False, "discard")

    records = _records(diagnostic_path)
    assert [row["outcome"] for row in records] == [
        "invalid", "measured", "held", "invalid",
    ]
    assert records[2]["last_success_time_ns"] == T0 + 100_000_000
    assert records[2]["last_success_age_ns"] == HOLD_NS
    assert records[2]["hold_timeout_ns"] == HOLD_NS
    assert records[3]["last_success_age_ns"] == HOLD_NS + 1


@pytest.mark.parametrize(
    ("extra", "expected_ns"),
    [
        ([], 500_000_000),
        (["--gripper-hold-ms", "250"], 250_000_000),
        (["--gripper-hold-ms", "0"], 0),
    ],
    ids=["default", "250ms", "disabled"],
)
def test_cli_gripper_hold_reaches_provider(
    tmp_path, monkeypatch, extra, expected_ns
):
    """命令行毫秒值经 NativeUmi 真实构造后传给 VisualGripperProvider。"""
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")

    class FakeTracker:
        """替代真实 Tracker，供 NativeUmi 启停和查询。"""

        error = None

        def start(self):
            """无需启动后台线程。"""

        def stop(self):
            """无需停止后台线程。"""

        def sample_at(self, _):
            """始终报告定位有效。"""
            return SimpleNamespace(right_tracked=True)

    class FakeCamera:
        """替代真实相机，返回接近目标时刻的预置帧。"""

        def connect(self):
            """无需打开设备。"""

        def disconnect(self):
            """无需关闭设备。"""

        def sample_at(self, _):
            """返回满足 NativeUmi 同步窗口的一帧。"""
            return CameraSample(FRAME, time.monotonic_ns() - 150_000_000, 1)

    class FakeEstimator:
        """始终返回有效开度，使 doctor 首轮即可退出。"""

        last_detected_marker_count = 2

        def estimate(self, _):
            """返回固定测试开度。"""
            return SimpleNamespace(openness=0.5)

    snapshot = {
        "serial": "synthetic",
        "camera_device": "synthetic",
        "resolution": [64, 48],
        "fps": 30,
        "fourcc": "MJPG",
        "tracker_tcp": np.eye(4).tolist(),
        "time": None,
    }
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: snapshot)
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: None)
    monkeypatch.setattr(umi_capture, "metadata", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "estimator", lambda _: FakeEstimator())

    real_native = umi_capture.NativeUmi
    seen = {}

    def build(snapshot_value, *, session=None, gripper_hold_ns=500_000_000):
        """使用真实 NativeUmi 构造链路，仅替换硬件和视觉算法。"""
        devices = real_native(
            snapshot_value,
            tracker=FakeTracker(),
            camera=FakeCamera(),
            session=session,
            gripper_hold_ns=gripper_hold_ns,
        )
        seen["provider"] = devices.gripper
        return devices

    monkeypatch.setattr(umi_capture, "NativeUmi", build)
    umi_capture.main(
        [
            "--calibration", str(calibration),
            "--rig-config", str(tmp_path / "missing-rig.yaml"),
            "--no-preview", *extra,
        ],
        mode="doctor",
    )

    assert seen["provider"].hold_timeout_ns == expected_ns


def test_invalid_gripper_hold_rejected_before_startup(tmp_path, monkeypatch):
    """负数、浮点数和文本在读取标定、处理输出或连接设备前被拒绝。"""
    # 故意使用不存在的标定文件：参数若通过校验，读取阶段会先失败。
    calibration = tmp_path / "missing-calibration.yaml"
    dataset = tmp_path / "dataset"

    def unexpected(*args, **kwargs):
        """非法参数不应进入后续启动流程。"""
        pytest.fail("非法保持期限不得进入启动流程")

    monkeypatch.setattr(umi_capture, "verify", unexpected)
    monkeypatch.setattr(umi_capture, "load_session", unexpected)
    monkeypatch.setattr(umi_capture, "NativeUmi", unexpected)
    monkeypatch.setattr(umi_capture, "_check_output_directory", unexpected)
    monkeypatch.setattr(umi_capture, "_prepare_output_directory", unexpected)
    for value in ("-1", "1.5", "abc"):
        with pytest.raises(SystemExit) as error:
            umi_capture.main([
                "--calibration", str(calibration),
                "--output-dir", str(dataset),
                "--rig-config", str(tmp_path / "missing-rig.yaml"),
                "--gripper-hold-ms", value,
            ])
        assert error.value.code == 2
    assert not dataset.exists()
