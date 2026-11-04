"""验证筛选基准的样本、快照、完整帧检查与结果对比。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark_screening.py"
SPEC = importlib.util.spec_from_file_location("screening_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_select_episodes_is_stable_and_uses_eight_quantiles():
    """长度相同时按编号排序，选择包含两端的八个均匀分位。"""
    rows = [(i, i // 2) for i in range(200)]
    expected = [rows[round(199 * i / 7)] for i in range(8)]
    assert benchmark.select_episodes(rows[::-1]) == expected
    with pytest.raises(ValueError, match="至少"):
        benchmark.select_episodes(rows[:7])
    with pytest.raises(ValueError, match="重复"):
        benchmark.select_episodes(rows[:8] + [rows[0]])


def test_snapshot_copies_videos_and_rejects_changed_files(tmp_path):
    """数据与视频必须复制隔离，任一文件被修改后快照不能用于基准。"""
    dataset = tmp_path / "source"
    for path in (dataset / "data", dataset / "meta/episodes", dataset / "videos/camera",
                 dataset / ".handumi/deployment"):
        path.mkdir(parents=True)
    (dataset / "data/data.parquet").write_bytes(b"data")
    pq.write_table(pa.table({"episode_index": list(range(8)), "length": list(range(1, 9))}),
                   dataset / "meta/episodes/meta.parquet")
    for name in ("info.json", "stats.json", "tasks.parquet"):
        (dataset / "meta" / name).write_bytes(name.encode())
    (dataset / "videos/camera/video.mp4").write_bytes(b"video")
    (dataset / ".handumi/deployment/rm75b.yaml").write_bytes(b"deployment")
    calibration = tmp_path / "robot.yaml"
    calibration.write_bytes(b"robot")
    output = tmp_path / "snapshot"
    manifest = benchmark.make_snapshot(dataset, output, calibration)
    assert len(manifest["episodes"]) == 8
    assert (output / "dataset/videos/camera/video.mp4").read_bytes() == b"video"
    assert benchmark.validate_snapshot(output) == manifest
    with pytest.raises(ValueError, match="已存在"):
        benchmark.make_snapshot(dataset, output, calibration)
    (output / "dataset/videos/camera/video.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="摘要不匹配"):
        benchmark.validate_snapshot(output)


def _report_fixture(directory, *, qpos=0.0, status="accepted"):
    """创建一个有完整帧与验收事件的筛选报告。"""
    directory.mkdir()
    report = directory / "report.json"
    payload = {"episodes": [{"episode_index": 3, "frame_count": 2, "status": status,
                             "metrics": {"self_collision_frames": 0}}]}
    report.write_text(json.dumps(payload))
    arrays = {"3/qpos": np.full((2, 2), qpos), "3/frame_indices": np.arange(2)}
    for side in ("left", "right"):
        arrays[f"3/{side}_pos_error_m"] = np.full(2, 0.001)
        arrays[f"3/{side}_rot_error_deg"] = np.full(2, 1.0)
    np.savez_compressed(directory / "report_solves.npz", **arrays)
    manifest = {"episodes": [{"episode": 3, "frames": 2}]}
    events = [{"event": "strict_validation", "episode": 3, "valid_collision_summary": True}]
    return report, manifest, events, arrays


def test_verify_output_checks_frames_thresholds_and_collision(tmp_path):
    """只接受完整帧、有限误差以及确有碰撞终验证据的求解结果。"""
    report, manifest, events, arrays = _report_fixture(tmp_path / "one")
    assert benchmark.verify_output(report, manifest, events)["ok"]
    missing_proof = benchmark.verify_output(report, manifest, [])
    assert not missing_proof["ok"] and "碰撞终验证据" in missing_proof["errors"][0]
    arrays["3/right_pos_error_m"][1] = 0.01
    np.savez_compressed(report.with_name("report_solves.npz"), **arrays)
    assert not benchmark.verify_output(report, manifest, events)["ok"]
    arrays["3/right_pos_error_m"][1] = 0.001
    arrays["3/frame_indices"] = np.array([0, 2])
    np.savez_compressed(report.with_name("report_solves.npz"), **arrays)
    assert not benchmark.verify_output(report, manifest, events)["ok"]


def test_compare_reports_records_status_and_numerical_differences(tmp_path):
    """状态差异和允许的轨迹差异均可追溯，不把数值变化误称为一致。"""
    before, _, _, _ = _report_fixture(tmp_path / "before")
    after, _, _, _ = _report_fixture(tmp_path / "after", qpos=0.1, status="rejected")
    result = benchmark.compare_reports(before, after)
    assert result["status_changes"][3] == ["accepted", "rejected"]
    assert result["trajectory"][3]["frame_indices"]["equal"]
    assert result["trajectory"][3]["qpos"]["max_abs_difference"] == pytest.approx(0.1)
