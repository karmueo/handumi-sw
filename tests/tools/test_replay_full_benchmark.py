"""验证全样本基准的完整性、限时事件与共同任务比较规则。"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("full_replay_benchmark", TOOLS / "benchmark_replay_full.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
sys.path.remove(str(TOOLS))


def event(index, elapsed, result=None):
    """构造一个持久化测试事件。"""
    return {"event": "task_done", "index": index, "elapsed_seconds": elapsed,
            "result": [index] if result is None else result}


def write(path, rows):
    """写出 JSONL 事件供比较函数读取。"""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_uses_only_identical_contiguous_tasks(tmp_path):
    """优化版额外完成的任务不会被错算为共同前缀。"""
    baseline, optimized = tmp_path / "a", tmp_path / "b"
    write(baseline, [event(0, 1), event(1, 2), event(2, 3)])
    write(optimized, [event(1, 0.5), event(3, 0.7), event(0, 1.0)])
    result = benchmark.compare_events(baseline, optimized)
    assert result["共同连续任务数"] == 2
    assert result["基线墙钟秒"] == 2
    assert result["优化墙钟秒"] == 1
    assert result["墙钟加速比"] == 2
    assert result["结果完全一致"]


def test_missing_or_duplicate_first_frames_are_never_frozen(tmp_path):
    """首帧任务未完整或出现重复时，禁止输出完整候选输入。"""
    events, target = tmp_path / "events", tmp_path / "first.json"
    write(events, [event(i, i + 1, []) for i in range(7)])
    with pytest.raises(ValueError, match="未完整"):
        benchmark.freeze_first(events, target, 1)
    assert not target.exists()
    write(events, [event(i, i + 1, []) for i in range(8)] + [event(0, 9, [])])
    with pytest.raises(ValueError, match="重复"):
        benchmark.freeze_first(events, target, 1)
    assert not target.exists()
    write(events, [event(i, i + 1, []) for i in reversed(range(8))])
    benchmark.freeze_first(events, target, 1)
    assert json.loads(target.read_text())["episode_count"] == 1


def test_interrupted_partial_json_line_is_ignored(tmp_path):
    """超时中断后只恢复已完整写出的任务事件。"""
    path = tmp_path / "events"
    path.write_text(json.dumps(event(0, 1)) + '\n{"event":', encoding="utf-8")
    assert list(benchmark.task_ids(benchmark.load_events(path))) == [0]


def test_finalists_require_all_sample_tasks(tmp_path, monkeypatch):
    """代表帧未全部完成时不能将候选冒充正式决选。"""
    from types import SimpleNamespace

    import numpy as np

    from handumi.retargeting.dataset_placement import Candidate
    from handumi.scripts.replay import replay_dataset_placement as placement

    candidates = [Candidate(np.array([0., 0., 0., 0., 0., 0., 1.]), 0)]
    monkeypatch.setattr(benchmark, "read_snapshot", lambda _snapshot: ([{}, {}], []))
    monkeypatch.setattr(benchmark, "load_candidates", lambda _path: candidates)
    monkeypatch.setattr(benchmark, "replay_args", lambda _snapshot: object())
    monkeypatch.setattr(placement, "_reference_calibration", lambda _args: SimpleNamespace(
        pose7=np.array([0., 0., 0., 0., 0., 0., 1.])))
    events, output = tmp_path / "events", tmp_path / "finalists"
    write(events, [event(0, 1, [1, 9])])
    with pytest.raises(ValueError, match="不完整"):
        benchmark.select_finalists("snapshot", "candidates", events, output)
    assert not output.exists()
    write(events, [event(0, 1, [1, 9]), event(1, 2, [1, 9])])
    benchmark.select_finalists("snapshot", "candidates", events, output)
    assert len(json.loads(output.read_text())) == 1
