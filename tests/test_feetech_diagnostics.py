"""验证夹爪诊断日志保留通信失败及限幅前后的异常位置。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from handumi.feetech import diagnostics
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import GripperCalibration
from handumi.feetech.gripper import _EncoderUnwrapper, _read_width


@pytest.fixture
def diagnostic_path(tmp_path, monkeypatch):
    """为每个测试创建独立日志，并关闭文件句柄。"""
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


def test_failed_reads_and_clamped_width(diagnostic_path):
    """通信失败不得产生开度；成功但越界的读数须保留限幅前值。"""
    bus = FeetechBus("test-port")
    bus._port_handler = SimpleNamespace(is_using=False)
    bus._sdk = SimpleNamespace(COMM_SUCCESS=0)
    bus._packet_handler = SimpleNamespace(read2ByteTxRx=Mock(side_effect=[
        (0, -1, 0), IndexError("short packet"), (500, 0, 0),
    ]))
    calibration = GripperCalibration(servo_id=1, closed_ticks=1000,
                                     open_ticks=2000, max_width_mm=90)
    widths = _read_width(bus, calibration, _EncoderUnwrapper(), retries=2,
                         retry_delay_s=0, side="right")
    records = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert [row["success"] for row in records[:3]] == [False, False, True]
    assert records[0]["comm"] == -1
    assert records[1]["exception"] == "truncated response"
    assert records[2]["raw_ticks"] == 500
    assert records[3]["side"] == "right"
    assert records[3]["ticks"] == 500
    assert records[3]["unclamped_normalized"] == -0.5
    assert records[3]["width_mm"] == widths["width_mm"] == 0
    assert all(row["monotonic_ns"] > 0 for row in records)


def test_disabled_diagnostics(monkeypatch):
    """未配置环境变量时不创建日志器。"""
    monkeypatch.delenv("HANDUMI_FEETECH_DIAGNOSTICS", raising=False)
    diagnostics._diagnostic_logger.cache_clear()
    try:
        diagnostics.log_diagnostic("test")
        assert diagnostics._diagnostic_logger() is None
    finally:
        diagnostics._diagnostic_logger.cache_clear()


def test_visual_missing_markers_are_logged(diagnostic_path):
    """确认视觉漏检的零值与真实有效开度可区分，重复帧不重复估计。"""
    import numpy as np

    from handumi.cameras.base import CameraSample
    from handumi.grippers.base import VisualGripperProvider

    estimator = SimpleNamespace(
        estimate=Mock(side_effect=[SimpleNamespace(
            openness=0.8, raw_openness=0.81, distance_mm=70,
            left_pose=SimpleNamespace(translation_mm=(1, 2, 3)),
            right_pose=SimpleNamespace(translation_mm=(71, 2, 3)),
        ), None]),
        last_detected_marker_count=2,
    )
    provider = VisualGripperProvider(estimator)
    frame = CameraSample(np.zeros((2, 2, 3), dtype=np.uint8), 100, 1)
    provider.observe(frame)
    provider.observe(frame)
    estimator.last_detected_marker_count = 1
    # 超过默认 500 毫秒保持期限的漏检仍是无测量，不能沿用旧开度。
    result = provider.observe(CameraSample(frame.image, 100 + 500_000_001, 2))
    records = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["outcome"] == "measured"
    assert records[0]["detected"] is True
    assert records[0]["valid"] is True
    assert records[0]["distance_mm"] == 70
    assert records[0]["left_position_mm"] == [1, 2, 3]
    assert records[0]["last_success_time_ns"] == 100
    assert records[0]["last_success_age_ns"] == 0
    assert records[0]["hold_timeout_ns"] == 500_000_000
    assert records[1]["outcome"] == "invalid"
    assert records[1]["detected"] is False
    assert records[1]["valid"] is False
    assert records[1]["detected_marker_count"] == 1
    assert records[1]["distance_mm"] is None
    assert records[1]["left_position_mm"] is None
    assert records[1]["normalized"] == result.normalized == 0
    assert records[1]["last_success_time_ns"] == 100
    assert records[1]["last_success_age_ns"] == 500_000_001
    assert records[1]["hold_timeout_ns"] == 500_000_000
    assert records[1]["sequence"] == 2
