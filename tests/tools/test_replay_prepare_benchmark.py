"""验证正式回放基准的调用计数和隔离缓存诊断。"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("replay_prepare_benchmark", TOOLS / "benchmark_replay_prepare.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)
sys.path.remove(str(TOOLS))


def test_call_counters_restore_on_error(monkeypatch):
    """两处模型别名各计一次，异常后恢复类方法与两个别名。"""
    from handumi.robots import kinematics, registry
    from handumi.scripts.replay import replay_in_sim as replay

    def ik(*args, **kwargs):
        """模拟 IK 输出。"""
        return "ik"
    def model(*args, **kwargs):
        """模拟模型初始化。"""
        return "model"
    monkeypatch.setattr(kinematics.BimanualKinematicsSolver, "ik", ik)
    monkeypatch.setattr(registry, "load_embodiment", model)
    monkeypatch.setattr(replay, "load_embodiment", model)
    with pytest.raises(RuntimeError, match="stop"):
        with benchmark.count_replay_calls() as counts:
            assert kinematics.BimanualKinematicsSolver.ik(None) == "ik"
            assert registry.load_embodiment("x") == "model"
            assert replay.load_embodiment("x") == "model"
            assert counts == {"ik_calls": 1, "model_initializations": 2}
            raise RuntimeError("stop")
    assert kinematics.BimanualKinematicsSolver.ik is ik
    assert registry.load_embodiment is model
    assert replay.load_embodiment is model


def test_existing_cache_is_copied_and_hit_has_no_ik(tmp_path, monkeypatch):
    """已有缓存只读复制到独立输出；命中诊断不调用 IK。"""
    from handumi.calibration import deployment
    from handumi.scripts.replay import replay_prepare
    source = tmp_path / "existing"
    source.mkdir()
    (source / "cache.bin").write_text("original")
    output = tmp_path / "out"
    monkeypatch.setattr(benchmark, "read_snapshot", lambda snapshot: (
        [{"index": 7, "length": 2}], [(np.zeros((2, 16)), 30, {}, None)]))
    monkeypatch.setattr(benchmark, "replay_args", lambda snapshot: SimpleNamespace(robot="rm75b"))
    monkeypatch.setattr(deployment, "load_deployment_calibration", lambda *a, **k: object())
    monkeypatch.setattr(benchmark, "snapshot_loader", lambda loaded: __import__("contextlib").nullcontext())
    def prepare(args, **kwargs):
        """模拟命中并写入副本，证明原缓存不会变更。"""
        print("[replay] cache hit episode=0")
        (args.cache_dir / "cache.bin").write_text("copy")
        return {"qpos": np.zeros((2, 9))}
    monkeypatch.setattr(replay_prepare, "prepare_episode", prepare)
    result = benchmark.benchmark_prepare("snapshot", tmp_path / "calibration", output, [0],
                                         count_calls=True, existing_cache=source)
    assert result["episodes"][0]["cache_hit"]
    assert result["counts"]["ik_calls"] == 0
    assert (source / "cache.bin").read_text() == "original"
    assert (output / "cache" / "cache.bin").read_text() == "copy"


def test_snapshot_output_and_cache_must_be_independent(tmp_path):
    """路径保护基于快照和输出的关系，与机器目录结构无关。"""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "cache").mkdir()
    with pytest.raises(ValueError, match="输出和快照"):
        benchmark.benchmark_prepare(snapshot, "calibration", snapshot / "out", [0])
    with pytest.raises(ValueError, match="输出和快照"):
        benchmark.benchmark_prepare(snapshot, "calibration", tmp_path, [0])
    with pytest.raises(ValueError, match="缓存和快照"):
        benchmark.benchmark_prepare(snapshot, "calibration", tmp_path / "out", [0],
                                    existing_cache=snapshot / "cache")
