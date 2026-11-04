"""验证转换的并行进程配置、缓存选择与失败回退。"""

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from handumi.dataset import parallel
from handumi.scripts import conversion
from handumi.scripts.replay import replay_prepare


def _read_worker_config(episode: int, state: dict) -> tuple[int, int, int]:
    """在真实子进程读取转换配置，同时模拟乱序完成。"""
    if episode == 3:
        time.sleep(0.15)
    return episode, state["args"].marker, os.getpid()


def _spawn_replay_args(args: argparse.Namespace, episode: int) -> tuple[int, int]:
    """为子进程测试构造轻量回放参数。"""
    return args.marker, episode


def _spawn_solve(replay_args: tuple[int, int]) -> dict:
    """模拟一个失败段与其他可成功求解的段。"""
    marker, episode = replay_args
    if episode == 1:
        raise ValueError("invalid segment")
    return {"qpos": np.array([marker, episode])}


def _spawn_presolve(episode: int, state: dict) -> dict:
    """在 spawn 子进程中替换重型求解器并调用真实预求解入口。"""
    conversion._replay_args_for_episode = _spawn_replay_args
    replay_prepare.prepare_episode = _spawn_solve
    return conversion._presolve_one(episode, state)


def test_jobs_parser_defaults_overrides_and_rejects_invalid_values() -> None:
    """默认八进程，显式参数和无效值在解析时生效。"""
    parser = conversion.build_parser()
    assert parser.parse_args(["dataset", "--robot", "rm75b"]).jobs == 8
    assert parser.parse_args(["dataset", "--robot", "rm75b", "--jobs", "1"]).jobs == 1
    assert parser.parse_args(["dataset", "--robot", "rm75b", "--jobs", "12"]).jobs == 12
    for value in ("0", "-2", "abc"):
        with pytest.raises(SystemExit):
            parser.parse_args(["dataset", "--robot", "rm75b", "--jobs", value])


def test_conversion_worker_limit_respects_cpus_and_small_batches(monkeypatch) -> None:
    """转换允许小批次并行，其他调用保留原降档策略。"""
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 3)
    assert parallel.resolve_jobs(8, 2, min_episodes_per_job=1) == 2
    assert parallel.resolve_jobs(8, 10, min_episodes_per_job=1) == 3
    assert parallel.resolve_jobs(1, 10, min_episodes_per_job=1) == 1
    assert parallel.resolve_jobs(8, 2) == 1


def test_spawn_receives_conversion_config_and_preserves_input_order() -> None:
    """真实 spawn 子进程取得初始化配置，结果保持输入顺序。"""
    args = argparse.Namespace(marker=17)
    results = parallel.map_episodes(
        [3, 1, 2],
        setup=conversion._presolve_worker_state,
        setup_kwargs={"args": args},
        task=_read_worker_config,
        jobs=2,
    )
    assert [(episode, marker) for episode, marker, _ in results] == [
        (3, 17), (1, 17), (2, 17)
    ]
    assert all(pid != os.getpid() for _, _, pid in results)


def test_presolve_one_uses_worker_state_and_defers_failure(monkeypatch) -> None:
    """求解调用读取进程状态；单段异常和 SystemExit 都交回主流程。"""
    args = argparse.Namespace(marker=17)
    monkeypatch.setattr(
        conversion, "_replay_args_for_episode",
        lambda config, episode: (config.marker, episode),
    )

    def solve(replay_args):
        """按段返回结果或模拟求解错误。"""
        marker, episode = replay_args
        if episode == 1:
            raise ValueError("invalid segment")
        if episode == 2:
            raise SystemExit("strict failure")
        return {"qpos": np.array([marker, episode]), "extra": "ignored"}

    monkeypatch.setattr(replay_prepare, "prepare_episode", solve)
    state = conversion._presolve_worker_state(args=args)
    np.testing.assert_array_equal(conversion._presolve_one(0, state)["rollout"]["qpos"], [17, 0])
    assert "invalid segment" in conversion._presolve_one(1, state)["error"]
    assert "strict failure" in conversion._presolve_one(2, state)["error"]


def test_spawn_continues_after_one_episode_fails() -> None:
    """真实进程池中一段求解异常不会中断其他段。"""
    results = parallel.map_episodes(
        [0, 1, 2],
        setup=conversion._presolve_worker_state,
        setup_kwargs={"args": argparse.Namespace(marker=17)},
        task=_spawn_presolve,
        jobs=2,
    )
    assert [item["episode"] for item in results] == [0, 1, 2]
    assert "invalid segment" in results[1]["error"]
    np.testing.assert_array_equal(results[0]["rollout"]["qpos"], [17, 0])
    np.testing.assert_array_equal(results[2]["rollout"]["qpos"], [17, 2])


def test_presolve_mixed_cache_order_and_failed_task(monkeypatch, capsys) -> None:
    """已缓存的段不进入池，失败段留给主流程重试，成功结果按源编号索引。"""
    args = argparse.Namespace()
    monkeypatch.setattr(conversion, "_cached_solve_for", lambda _args, ep: {} if ep == 2 else None)
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 8)
    calls = []

    def run(episodes, **kwargs):
        """模拟并行任务返回，检查初始化配置。"""
        calls.append((episodes, kwargs))
        return [
            {"episode": 3, "rollout": {"qpos": np.array([3])}},
            {"episode": 1, "error": "failed"},
        ]

    monkeypatch.setattr(parallel, "map_episodes", run)
    result = conversion.presolve_episodes(args, [3, 2, 1], jobs=8)
    assert list(result) == [3]
    assert calls[0][0] == [3, 1]
    assert calls[0][1]["jobs"] == 2
    assert calls[0][1]["setup_kwargs"]["args"] is args
    assert "2 episodes across 2 workers" in capsys.readouterr().out


def test_cached_and_sequential_batches_do_not_create_pool(monkeypatch) -> None:
    """全部缓存命中或只有一个进程时不创建进程池。"""
    monkeypatch.setattr(parallel, "map_episodes", lambda *args, **kwargs: pytest.fail("unexpected pool"))
    monkeypatch.setattr(conversion, "_cached_solve_for", lambda *_: {})
    assert conversion.presolve_episodes(argparse.Namespace(), [0, 1], jobs=8) == {}
    monkeypatch.setattr(conversion, "_cached_solve_for", lambda *_: None)
    assert conversion.presolve_episodes(argparse.Namespace(), [0, 1], jobs=1) == {}


def test_complete_cache_and_refresh(monkeypatch) -> None:
    """完整缓存命中无需预求解；刷新和禁用缓存跳过所有缓存。"""
    args = argparse.Namespace(
        no_solve_cache=False, refresh_cache=False, root="dataset", embodiment="rm75b"
    )
    monkeypatch.setattr(conversion, "_replay_args_for_episode", lambda _args, ep: SimpleNamespace(episode=ep))
    monkeypatch.setattr(conversion, "solver_signature", lambda *args, **kwargs: "key")
    monkeypatch.setattr(conversion, "load_cached_solve", lambda *args, **kwargs: None)
    cached = {"qpos": np.array([1])}
    monkeypatch.setattr(replay_prepare, "load_prepared_episode", lambda replay_args: cached)
    assert conversion._cached_solve_for(args, 1) is cached
    args.refresh_cache = True
    assert conversion._cached_solve_for(args, 1) is None
    args.refresh_cache = False
    args.no_solve_cache = True
    assert conversion._cached_solve_for(args, 1) is None
