"""验证回放基准的小样本选择和冻结候选规则。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark_replay_placement.py"
SPEC = importlib.util.spec_from_file_location("replay_placement_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_select_episodes_uses_stable_length_and_index_order():
    """长度相同按 episode 编号排序，分位选择稳定且返回完整帧数。"""
    rows = [(i, i // 2) for i in range(200)]
    assert benchmark.select_episodes(rows[::-1]) == [rows[i] for i in (20, 70, 129, 179)]
    with pytest.raises(ValueError):
        benchmark.select_episodes(rows[:3])


def test_candidate_subset_keeps_reference_and_covers_yaws():
    """代表帧覆盖八个朝向，完整验证包含精确参考和另一方向。"""
    from handumi.retargeting.dataset_placement import Candidate

    reference = np.array([0, 0, 0, 0, 0, 0, 1.0])
    candidates = [Candidate(reference.copy(), 0)]
    candidates += [Candidate(np.array([i, 0, 0, 0, 0, 0, 1.0]), i) for i in range(1, 8)]
    sample, full = benchmark.candidate_subset(candidates, reference)
    assert [candidate.yaw_index for candidate in sample] == list(range(8))
    assert np.array_equal(full[0].pose, reference)
    assert full[1].yaw_index == 4
    with pytest.raises(ValueError):
        benchmark.candidate_subset(candidates[:7], reference)


def test_snapshot_digest_detects_changed_data(tmp_path):
    """快照内容发生变化时禁止进入性能或正确性对比。"""
    import json

    from handumi.dataset.episode_solve_cache import array_digest

    states = np.zeros((2, 16), dtype=np.float32)
    np.savez_compressed(tmp_path / "episode-0.npz", states=states)
    entry = {"index": 10, "length": 2, "fps": 30, "file": "episode-0.npz",
             "states": array_digest(states), "openings": None}
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": [entry], "info": {}}))
    entries, loaded = benchmark.read_snapshot(tmp_path)
    assert entries[0]["index"] == 10 and len(loaded[0][0]) == 2
    np.savez_compressed(tmp_path / "episode-0.npz", states=states + 1)
    with pytest.raises(ValueError, match="校验失败"):
        benchmark.read_snapshot(tmp_path)


def test_compare_results_requires_same_discrete_outcome():
    """布尔和失败帧必须相同，浮点误差遵循约定容差。"""
    def result(passed=True, position=0.001):
        """构造最小结果用于比较。"""
        return {"runs": [{"generated": [{"yaw_index": 0, "pose7": [position]}],
                          "sample_results": [[1, 3]],
                          "full_results": [[{"passed": passed, "prefix_frames": 1}]],
                          "rank": [[-1, -1]]}]}

    assert not benchmark.compare_results(result(), result(position=0.001001))
    assert benchmark.compare_results(result(), result(passed=False))
    assert benchmark.compare_results(result(), result(position=0.01))


def test_compare_captures_checks_each_rollout_array(tmp_path):
    """完整轨迹的位置、关节与误差任一变化都应被发现。"""
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    fields = ["qpos", "target_left_pose7_robot_world", "target_right_pose7_robot_world",
              "achieved_left_pose7_robot_world", "achieved_right_pose7_robot_world",
              "left_pos_error_m", "right_pos_error_m", "left_rot_error_deg", "right_rot_error_deg"]
    arrays = {key: np.zeros((2, 3)) for key in fields}
    np.savez_compressed(left / "one.npz", **arrays)
    np.savez_compressed(right / "one.npz", **arrays)
    assert benchmark.compare_captures(left, right) == []
    arrays["qpos"] = np.ones((2, 3))
    np.savez_compressed(right / "one.npz", **arrays)
    assert benchmark.compare_captures(left, right) == ["one.npz:qpos mismatch"]


def test_snapshot_requires_empty_output_and_copies_calibration(tmp_path, monkeypatch):
    """快照必须复制并摘要真实标定，已有内容不得覆盖。"""
    import hashlib
    from types import SimpleNamespace

    import pyarrow.parquet as pq

    from handumi.dataset import selection
    from handumi.scripts.replay import replay_in_sim as replay

    dataset = tmp_path / "dataset"
    meta = dataset / "meta" / "episodes"
    meta.mkdir(parents=True)
    (meta / "episodes.parquet").touch()
    calibration = tmp_path / "robot.yaml"
    calibration.write_text("known calibration", encoding="utf-8")
    output = tmp_path / "snapshot"

    class Column:
        """提供测试所需的 Arrow 列接口。"""
        def __init__(self, values):
            """保存列数据。"""
            self.values = values

        def to_pylist(self):
            """返回普通列表。"""
            return self.values

    monkeypatch.setattr(pq, "read_table", lambda *a, **k: {
        "episode_index": Column([0, 1, 2, 3]), "length": Column([2, 3, 4, 5])})
    monkeypatch.setattr(selection, "resolve_dataset_selection", lambda path: SimpleNamespace(
        repo_id="benchmark/local", root=dataset))
    monkeypatch.setattr(benchmark, "replay_args", lambda root: SimpleNamespace())
    monkeypatch.setattr(replay, "load_episode_states", lambda args: (
        np.zeros((args.episode + 2, 16), dtype=np.float32), 30.0, {}, None))
    manifest = benchmark.make_snapshot(dataset, output, calibration)
    assert (output / "robot_tcp.yaml").read_bytes() == calibration.read_bytes()
    assert manifest["robot_tcp_calibration"]["sha256"] == hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert len(benchmark.read_snapshot(output)[0]) == 4
    with pytest.raises(ValueError, match="禁止覆盖"):
        benchmark.make_snapshot(dataset, output, calibration)
    (output / "robot_tcp.yaml").write_text("changed")
    with pytest.raises(ValueError, match="标定摘要"):
        benchmark.read_snapshot(output)


def test_benchmark_reports_provenance_and_rss(monkeypatch):
    """外层观测加入结果且不改变传入的基准参数。"""
    seen = []
    def fake_benchmark(*args, **kwargs):
        """模拟内部阶段，记录输入并返回现有输出。"""
        seen.append((args, kwargs))
        return {"runs": [{"total_seconds": 0.1}], "cold_total_seconds": 0.2}
    monkeypatch.setattr(benchmark, "_benchmark", fake_benchmark)
    monkeypatch.setattr(benchmark, "code_provenance", lambda: {"git_head": "test"})
    result = benchmark.benchmark("snapshot", "candidates", 2, 3)
    assert seen[0][0][:4] == ("snapshot", "candidates", 2, 3)
    assert result["code_provenance"]["git_head"] == "test"
    assert "peak_kib" in result["rss_process_tree_sampled"]
    assert "sampled lower bound" in result["rss_process_tree_sampled"]["semantics"]
