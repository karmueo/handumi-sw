"""验证逐段缓存内容身份、原子持久化与有限优先调度。"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from handumi.dataset.episode_solve_cache import EpisodeSolveCache
from handumi.scripts.replay import replay_in_sim as replay
from handumi.scripts.replay import replay_prepare as prepare
from handumi.scripts.replay import replay_scheduler
from handumi.scripts.replay.replay_scheduler import EpisodeScheduler


def _args_and_loaded(tmp_path, monkeypatch, *, collision_enabled=False):
    """构建不依赖真实数据的 RM75B 完整缓存输入。"""
    # 此组测试缓存机制；独立关闭真实 RM75B 碰撞约束以使用合成轨迹。
    from dataclasses import replace

    import handumi.robots.registry as registry

    original_load = registry.load_robot_config

    def load_test_config(name):
        """为缓存测试提供未启用碰撞的隔离配置。"""
        config = original_load(name)
        return replace(config, ik_weights=replace(config.ik_weights, self_collision_weight=0))

    if not collision_enabled:
        monkeypatch.setattr(registry, "load_robot_config", load_test_config)
    tool = tmp_path / "tool.yaml"
    tool.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "handumi_robot_tcp_calibration",
                "robot": "rm75b",
                "verified": False,
                "calibration": {
                    "arms": {
                        "right": {"ee_link": "Link7", "ee_to_tcp": np.eye(4).tolist()}
                    }
                },
            }
        )
    )
    args = replay.build_parser().parse_args(
        [
            str(tmp_path),
            "--robot",
            "rm75b",
            "--retarget-mode",
            "absolute-tcp",
            "--base-placement",
            "fixed",
            "--robot-tcp-calibration",
            str(tool),
        ]
    )
    args.dataset_root, args.repo_id, args.rig_config = (
        tmp_path,
        "local/test",
        tmp_path / "missing.yaml",
    )
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 6] = states[:, 13] = 1
    info = {
        "total_episodes": 1,
        "total_frames": 2,
        "handumi": {
            "recording_device": "vive",
            "tracking_workspace": "table",
            "active_sides": ["right"],
        },
    }
    loaded = [states, 30.0, info, None]
    monkeypatch.setattr(replay, "load_episode_states", lambda args: tuple(loaded))
    monkeypatch.setattr(
        prepare, "_source_calibration", lambda *args: {"left": [0], "right": [1]}
    )
    return args, loaded, tool


def _rollout(error=0.0):
    """生成帧数一致的完整最小回放结果。"""
    poses = np.tile([0, 0, 0, 0, 0, 0, 1], (2, 1)).astype(np.float32)
    result = {
        "qpos": np.zeros((2, 9), dtype=np.float32),
        "frame_indices": np.arange(2),
        "fps": np.asarray([30.0]),
        "gripper_normalized": np.zeros((2, 2)),
        "retarget_mode": np.asarray(["absolute-tcp"]),
    }
    for side in ("left", "right"):
        for kind in ("target", "achieved"):
            result[f"{kind}_{side}_pose7_robot_world"] = poses.copy()
        result[f"{side}_pos_error_m"] = np.full(2, error if side == "right" else 0)
        result[f"{side}_rot_error_deg"] = np.zeros(2)
        result[f"raw_{side}_pose7_ground_truth"] = poses.copy()
        result[f"raw_{side}_controller_pose7"] = poses.copy()
        result[f"calibrated_{side}_tcp_pose7"] = poses.copy()
    for name in (
        "base_placement_policy",
        "base_placement_scope",
        "base_placement_diagnostics_json",
        "controller_device",
        "controller_tcp_calibration",
        "robot_tcp_calibration_json",
        "gripper_source",
        "tracking_workspace",
        "retarget_mode_requested",
    ):
        result[name] = np.asarray([""])
    for name in ("robot_from_table_pose7", "table_from_robot_pose7"):
        result[name] = poses[:1].copy()
    for name in (
        "base_placement_verified",
        "initial_solve_iterations",
        "initial_max_position_error_m",
        "initial_max_rotation_error_deg",
    ):
        result[name] = np.asarray([0])
    return result


def test_cache_hit_strict_gate_and_approach_are_independent(
    tmp_path, monkeypatch, capsys
):
    """严格检查在命中时重做，渲染过渡按当前设置恢复。"""
    args, _, _ = _args_and_loaded(tmp_path, monkeypatch)
    args.approach_seconds = 1.0
    calls = []
    monkeypatch.setattr(
        replay,
        "solve_episode",
        lambda args: calls.append(args.strict_ik) or _rollout(0.021),
    )
    first = prepare.prepare_episode(args)
    assert calls == [False] and len(first["approach_qpos"]) > 0
    args.approach_seconds = 0
    second = prepare.prepare_episode(args)
    assert calls == [False] and len(second["approach_qpos"]) == 0
    assert "warning: IK fidelity threshold exceeded" in capsys.readouterr().out
    args.strict_ik = True
    with pytest.raises(SystemExit, match="IK fidelity threshold exceeded"):
        prepare.prepare_episode(args)
    assert calls == [False]


def test_cache_identity_is_per_episode_and_tracks_geometry(tmp_path, monkeypatch):
    """追加其他段保持命中；当前数据、求解参数、工具和 URDF 改变均失效。"""
    args, loaded, tool = _args_and_loaded(tmp_path, monkeypatch)
    key, _ = prepare.episode_identity(args, tuple(loaded))
    duplicate = tmp_path / "same_tool.yaml"
    duplicate.write_bytes(tool.read_bytes())
    args.robot_tcp_calibration = duplicate
    assert prepare.episode_identity(args, tuple(loaded))[0] != key
    args.robot_tcp_calibration = tool
    loaded[2]["total_episodes"] = 2
    loaded[2]["total_frames"] = 20
    assert prepare.episode_identity(args, tuple(loaded))[0] == key
    loaded[0][0, 7] += 0.1
    assert prepare.episode_identity(args, tuple(loaded))[0] != key
    loaded[0][0, 7] -= 0.1
    args.max_ik_position_error_m = 0.004
    assert prepare.episode_identity(args, tuple(loaded))[0] != key
    args.max_ik_position_error_m = 0.005
    tool.write_text(tool.read_text() + "\n# changed\n")
    assert prepare.episode_identity(args, tuple(loaded))[0] != key
    current, _ = prepare.episode_identity(args, tuple(loaded))
    import handumi.robots.registry as registry

    old = registry.load_robot_config
    urdf = tmp_path / "model.urdf"
    urdf.write_text("one")

    def changed_config(name):
        """把测试机器人模型指向独立 URDF。"""
        from dataclasses import replace

        return replace(old(name), urdf=urdf)

    monkeypatch.setattr(registry, "load_robot_config", changed_config)
    assert prepare.episode_identity(args, tuple(loaded))[0] != current
    first, _ = prepare.episode_identity(args, tuple(loaded))
    urdf.write_text("two")
    assert prepare.episode_identity(args, tuple(loaded))[0] != first


def test_atomic_cache_damage_and_write_failure(tmp_path, monkeypatch):
    """截断当作 miss；替换失败不能破坏已存在的完整文件。"""
    cache = EpisodeSolveCache(tmp_path)
    key = "a" * 64
    cache.store(key, _rollout())
    before = cache.path(key).read_bytes()
    import handumi.dataset.episode_solve_cache as module

    monkeypatch.setattr(
        module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("read-only"))
    )
    with pytest.raises(OSError):
        cache.store(key, _rollout(0.1))
    assert cache.path(key).read_bytes() == before
    cache.path(key).write_bytes(before[:50])
    assert cache.load(key) == (None, None)


def test_corrupted_complete_rollout_is_recomputed(tmp_path, monkeypatch):
    """缓存截断后自动重算并原子恢复，不将损坏当作有效结果。"""
    args, _, _ = _args_and_loaded(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        replay, "solve_episode", lambda args: calls.append(1) or _rollout()
    )
    prepare.prepare_episode(args)
    path = next(prepare.cache_directory(args).rglob("*.npz"))
    path.write_bytes(path.read_bytes()[:40])
    prepare.prepare_episode(args)
    assert len(calls) == 2
    assert (
        EpisodeSolveCache(prepare.cache_directory(args)).load(path.stem)[0] is not None
    )


def test_readonly_cache_and_no_cache_return_rollout(tmp_path, monkeypatch):
    """缓存不可写与显式禁用时，前台仍返回求解结果。"""
    args, _, _ = _args_and_loaded(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        replay, "solve_episode", lambda args: calls.append(1) or _rollout()
    )
    monkeypatch.setattr(
        EpisodeSolveCache,
        "store",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only")),
    )
    assert len(prepare.prepare_episode(args)["qpos"]) == 2
    args.no_cache = True
    assert len(prepare.prepare_episode(args)["qpos"]) == 2
    assert len(calls) == 2


def test_only_deterministic_failures_are_cached_and_refresh_retries(
    tmp_path, monkeypatch
):
    """确定性首帧失败复用结果，强制刷新重试；临时错误不被持久化。"""
    args, _, _ = _args_and_loaded(tmp_path, monkeypatch)
    calls = []

    def fail(_):
        """模拟一个可缓存的确定性求解失败。"""
        calls.append(1)
        raise SystemExit("Unable to prepare replay start pose: unreachable")

    monkeypatch.setattr(replay, "solve_episode", fail)
    for _ in range(2):
        with pytest.raises(SystemExit, match="Unable to prepare"):
            prepare.prepare_episode(args)
    assert len(calls) == 1
    args.refresh_cache = True
    with pytest.raises(SystemExit, match="Unable to prepare"):
        prepare.prepare_episode(args)
    assert len(calls) == 2

    def transient(_):
        """模拟不应缓存的临时错误。"""
        calls.append(1)
        raise SystemExit("temporary input error")

    args.cache_dir = tmp_path / "transient"
    args.refresh_cache = False
    monkeypatch.setattr(replay, "solve_episode", transient)
    for _ in range(2):
        with pytest.raises(SystemExit, match="temporary input"):
            prepare.prepare_episode(args)
    assert len(calls) == 4


class FakeExecutor:
    """可控的 worker executor，用于验证排队和关闭行为。"""

    def __init__(self):
        """保存提交顺序和人工完成的 future。"""
        self.jobs = []
        self.shutdown_args = None

    def submit(self, fn, values, episode, refresh):
        """记录提交而不执行工作。"""
        future = Future()
        self.jobs.append((episode, future, refresh))
        return future

    def shutdown(self, **kwargs):
        """记录非阻塞关闭。"""
        self.shutdown_args = kwargs


def test_prepare_all_continues_when_initial_episode_fails(monkeypatch, tmp_path):
    """全量 headless 的初始失败不能阻止其他段准备。"""
    import sys

    observed = []
    monkeypatch.setattr(
        sys, "argv", ["handumi", str(tmp_path), "--headless", "--prepare-mode", "all"]
    )
    monkeypatch.setattr(
        replay,
        "resolve_dataset_selection",
        lambda *args, **kwargs: SimpleNamespace(repo_id="local/test", root=tmp_path),
    )
    monkeypatch.setattr(
        replay,
        "_prepare_all_headless",
        lambda args: (observed.append(args.episode) or None, {0: "initial unreachable"}),
    )
    with pytest.raises(SystemExit, match="Preparation failed"):
        replay.main()
    assert observed == [0]


def test_scheduler_close_stops_running_workers():
    """关闭时清除未提交任务并终止仍在运行的准备进程。"""
    terminated = []
    process = SimpleNamespace(
        is_alive=lambda: True, terminate=lambda: terminated.append(True)
    )
    executor = FakeExecutor()
    executor._processes = {1: process}
    scheduler = EpisodeScheduler(
        SimpleNamespace(refresh_cache=False), 3, current=0, workers=1, executor=executor
    )
    scheduler.poll()
    scheduler.close()
    assert terminated == [True] and not scheduler.queue
    assert executor.shutdown_args == {"wait": False, "cancel_futures": True}


def test_worker_warning_is_emitted_when_selected(monkeypatch, capsys):
    """非严格超差警告随 worker 结果返回，并在选择该段时输出一次。"""

    def simulate(args, runtime_provider=None):
        """模拟后台固定摆放超差但允许预览。"""
        print("[replay] warning: IK fidelity threshold exceeded: frame 9")
        return _rollout(0.01)

    monkeypatch.setattr(prepare, "prepare_episode", simulate)
    response = replay_scheduler._worker_prepare({"refresh_cache": False}, 1, False)
    executor = FakeExecutor()
    scheduler = EpisodeScheduler(
        SimpleNamespace(refresh_cache=False), 2, current=0, workers=1, executor=executor
    )
    scheduler.poll()
    executor.jobs[0][1].set_result(response)
    scheduler.poll()
    assert "warning" not in capsys.readouterr().out
    assert scheduler.take(1) is not None
    assert "warning: IK fidelity threshold exceeded" in capsys.readouterr().out
    assert scheduler.take(1) is None
    assert "warning" not in capsys.readouterr().out
    scheduler.close()


def test_scheduler_limits_inflight_prioritizes_and_deduplicates():
    """选中已排队段优先于后台，运行中重复选择不会重复提交。"""
    executor = FakeExecutor()
    scheduler = EpisodeScheduler(
        SimpleNamespace(refresh_cache=False), 6, current=0, workers=2, executor=executor
    )
    scheduler.poll()
    assert [item[0] for item in executor.jobs] == [1, 2]
    scheduler.request(2)
    scheduler.request(5)
    scheduler.request(5)
    executor.jobs[0][1].set_result({"rollout": _rollout()})
    scheduler.poll()
    assert [item[0] for item in executor.jobs] == [1, 2, 5]
    assert len(scheduler.running) == 2
    scheduler.close()
    assert not scheduler.queue and executor.shutdown_args == {
        "wait": False,
        "cancel_futures": True,
    }


@pytest.mark.parametrize("current", [0, 2, 4])
def test_batch_scheduler_all_tasks_and_compact_results(current):
    """任意选中段优先、在途有界，失败不阻断后续且仅保留选中轨迹。"""
    executor = FakeExecutor()
    scheduler = EpisodeScheduler(
        SimpleNamespace(refresh_cache=True, prepare_mode="all"), 5,
        current=current, workers=2, executor=executor, batch=True,
    )
    scheduler.poll()
    assert executor.jobs[0][0] == current
    submitted = []
    while scheduler.running or scheduler.queue:
        for future, episode in list(scheduler.running.items()):
            submitted.append(episode)
            response = {"error": "bad"} if episode == (current + 1) % 5 else {}
            if episode == current:
                response["rollout"] = _rollout()
            response.update(cache_hit=episode == current, stage_seconds={"求解": 0.1})
            future.set_result(response)
        scheduler.poll()
        assert len(scheduler.running) <= 2 and len(scheduler.results) <= 1
    assert sorted(submitted) == list(range(5))
    assert all(job[2] for job in executor.jobs)
    assert len(scheduler.completed) == 5 and len(scheduler.failures) == 1
    assert scheduler.cache_hits == 1 and scheduler.take(current) is not None
    assert scheduler.stage_seconds["求解"] == pytest.approx(0.5)
    scheduler.close()


@pytest.mark.parametrize("count,current", [(0, 0), (2, -1), (2, 2)])
def test_scheduler_rejects_empty_or_invalid_episode(count, current):
    """空数据和越界请求在启动任务前清楚失败。"""
    with pytest.raises(ValueError):
        EpisodeScheduler(SimpleNamespace(), count, current=current, executor=FakeExecutor())


def test_batch_worker_returns_only_selected_rollout(monkeypatch):
    """非选中段返回紧凑摘要；缓存不能写入时选中段仍携带本次结果。"""
    monkeypatch.setattr(prepare, "prepare_episode", lambda *a, **k: _rollout())
    values = {"_batch_compact": True, "_batch_current": 1}
    other = replay_scheduler._worker_prepare(values, 0, False)
    chosen = replay_scheduler._worker_prepare(values, 1, False)
    assert "rollout" not in other
    assert "rollout" in chosen and len(chosen["rollout"]["qpos"]) == 2


def test_batch_queue_is_lazy_and_single_episode_works():
    """大规模批次只保留游标，单段只提交一次。"""
    pending = replay_scheduler._BatchQueue(1_000_000, 50)
    assert pending.popleft() == 50 and pending.popleft() == 0
    assert len(vars(pending)) == 5
    single = replay_scheduler._BatchQueue(1, 0)
    assert single.popleft() == 0 and not single


def test_batch_selected_worker_returns_real_result_when_cache_write_fails(tmp_path, monkeypatch):
    """真实准备流程的缓存写入失败时仍返回选中段，且不重新求解。"""
    args, _, _ = _args_and_loaded(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(replay, "solve_episode", lambda _: calls.append(1) or _rollout())
    monkeypatch.setattr(replay_scheduler, "_runtime_for", lambda *a: object())
    monkeypatch.setattr(
        EpisodeSolveCache, "store",
        lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")),
    )
    values = {**vars(args), "_batch_compact": True, "_batch_current": 0}
    response = replay_scheduler._worker_prepare(values, 0, False)
    assert "error" not in response and len(response["rollout"]["qpos"]) == 2
    assert len(calls) == 1 and any("cache write skipped" in x for x in response["warnings"])


@pytest.mark.parametrize("fail_selected", [True, False])
def test_headless_batch_orchestration_continues_and_preserves_selected(
    tmp_path, monkeypatch, fail_selected
):
    """混合完成顺序和失败时仍调度全部段，只取选中段结果并关闭资源。"""
    monkeypatch.setattr(replay, "ensure_metadata", lambda _: {"total_episodes": 5})
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    schedulers = []

    class ScriptedScheduler(EpisodeScheduler):
        """用交错完成的可控 future 模拟长短任务。"""
        def __init__(self, *args, **kwargs):
            """保留真实调度行为，替换实际进程池。"""
            kwargs["executor"] = FakeExecutor()
            super().__init__(*args, **kwargs)
            self.cycle = 0
            self.was_closed = False
            schedulers.append(self)

        def poll(self):
            """交替完成尾部短任务和首部较长任务。"""
            self.cycle += 1
            running = list(self.running.items())
            if running:
                future, episode = running[-1 if self.cycle % 3 else 0]
                failed = episode == (2 if fail_selected else 3)
                response = {"error": "unreachable"} if failed else {}
                if episode == 2 and not failed:
                    response["rollout"] = _rollout()
                future.set_result(response)
            super().poll()

        def close(self):
            """记录清理调用并执行真实关闭路径。"""
            self.was_closed = True
            super().close()

    monkeypatch.setattr(replay_scheduler, "EpisodeScheduler", ScriptedScheduler)
    args = SimpleNamespace(repo_id="local/test", dataset_root=tmp_path, revision="main",
                           episode=2, prepare_workers=2, progress="off", refresh_cache=True,
                           prepare_mode="all")
    rollout, failures = replay._prepare_all_headless(args)
    scheduler = schedulers[0]
    assert sorted(job[0] for job in scheduler.executor.jobs) == list(range(5))
    assert scheduler.executor.jobs[0][0] == 2
    assert all(job[2] for job in scheduler.executor.jobs)
    assert failures == ({2: "unreachable"} if fail_selected else {3: "unreachable"})
    assert (rollout is None) == fail_selected
    assert scheduler.was_closed


def test_headless_batch_interrupt_closes_scheduler(tmp_path, monkeypatch):
    """键盘中断从批量流程退出时仍调用调度器关闭。"""
    monkeypatch.setattr(replay, "ensure_metadata", lambda _: {"total_episodes": 1})
    schedulers = []
    class InterruptedScheduler(EpisodeScheduler):
        """构造立即中断的批次。"""
        def __init__(self, *args, **kwargs):
            """以受控 executor 初始化可中断调度器。"""
            kwargs["executor"] = FakeExecutor()
            super().__init__(*args, **kwargs)
            schedulers.append(self)
        def poll(self):
            """模拟用户在调度期间按下中断键。"""
            raise KeyboardInterrupt()
    monkeypatch.setattr(replay_scheduler, "EpisodeScheduler", InterruptedScheduler)
    args = SimpleNamespace(repo_id="local/test", dataset_root=tmp_path, revision="main",
                           episode=0, prepare_workers=1, progress="off")
    with pytest.raises(KeyboardInterrupt):
        replay._prepare_all_headless(args)
    assert schedulers[0].closed


def test_main_headless_batch_saves_selected_once_and_preserves_output(tmp_path, monkeypatch):
    """命令行 headless 全量成功时只导出选中段一次，保留显式路径。"""
    import sys
    output = tmp_path / "selected.npz"
    monkeypatch.setattr(sys, "argv", ["handumi", str(tmp_path), "--headless",
        "--prepare-mode", "all", "--episode", "3", "--output", str(output)])
    monkeypatch.setattr(replay, "resolve_dataset_selection", lambda *a, **k:
        SimpleNamespace(repo_id="local/test", root=tmp_path))
    selected = _rollout()
    monkeypatch.setattr(replay, "_prepare_all_headless", lambda args: (selected, {}))
    saved = []
    monkeypatch.setattr(replay, "save_rollout", lambda args, rollout:
        saved.append((args.episode, args.output, rollout)))
    replay.main()
    assert saved == [(3, output, selected)]


def test_rm75b_cache_rejects_missing_collision_summary(tmp_path, monkeypatch):
    """未经碰撞认证的结果不得写入成功缓存。"""
    args, _, _ = _args_and_loaded(tmp_path, monkeypatch, collision_enabled=True)
    monkeypatch.setattr(replay, "solve_episode", lambda args: _rollout())
    with pytest.raises(SystemExit, match="collision validation summary"):
        prepare.prepare_episode(args)
    assert not list(prepare.cache_directory(args).rglob("*.npz"))


def test_rm75b_approach_failure_does_not_poison_direct_start_cache(tmp_path, monkeypatch):
    """接近段搜索失败可复用，但关闭接近段后必须重新求解。"""
    from handumi.dataset.episode_solve_cache import array_digest
    from handumi.robots.self_collision import GRIPPER_BOX_VALIDATION_VERSION

    args, _, _ = _args_and_loaded(tmp_path, monkeypatch, collision_enabled=True)
    args.base_placement = "auto"
    args.approach_seconds = 1.0
    calls = []  # 记录实际求解使用的接近时长。
    rollout = _rollout()
    rollout["self_collision_validation"] = np.asarray([GRIPPER_BOX_VALIDATION_VERSION])
    rollout["self_collision_qpos_sha256"] = np.asarray([
        array_digest(rollout["qpos"])["sha256"]
    ])
    rollout["self_collision_min_sampled_m"] = np.asarray([0.1])
    rollout["self_collision_min_links"] = np.asarray(["Link2", "gripper_link"])

    def solve(run_args):
        """模拟接近段碰撞导致搜索失败、直接首帧开始则成功。"""
        calls.append(run_args.approach_seconds)
        if run_args.approach_seconds > 0:
            raise SystemExit("Unable to find strict episode XY placement: approach collision")
        return rollout

    monkeypatch.setattr(replay, "solve_episode", solve)
    # 本测试仅隔离缓存路由，碰撞复验由独立测试覆盖。
    monkeypatch.setattr(prepare, "_finish_rollout", lambda args, result, **kwargs: result)
    for _ in range(2):
        with pytest.raises(SystemExit, match="approach collision"):
            prepare.prepare_episode(args)
    assert calls == [1.0]

    args.approach_seconds = 0.0
    for _ in range(2):
        np.testing.assert_array_equal(prepare.prepare_episode(args)["qpos"], rollout["qpos"])
    assert calls == [1.0, 0.0]


def test_rm75b_collision_mesh_changes_cache_identity(tmp_path, monkeypatch):
    """仅更改碰撞网格摘要也会使 RM75B 缓存失效。"""
    args, loaded, _ = _args_and_loaded(tmp_path, monkeypatch, collision_enabled=True)
    original, payload = prepare.episode_identity(args, tuple(loaded))
    assert any("rm75_gripper_base.stl" in name for name in payload["collision_meshes"])
    digest = prepare.digest_file

    def changed(path):
        """模拟单个夹爪 STL 内容变更。"""
        if path is not None and str(path).endswith("rm75_gripper_base.stl"):
            return "changed"
        return digest(path)

    monkeypatch.setattr(prepare, "digest_file", changed)
    assert prepare.episode_identity(args, tuple(loaded))[0] != original


def test_rm75b_gripper_model_changes_cache_identity(tmp_path, monkeypatch):
    """夹爪 URDF 或机器人配置变化都会使 RM75B 缓存失效。"""
    args, loaded, _ = _args_and_loaded(tmp_path, monkeypatch)
    original, payload = prepare.episode_identity(args, tuple(loaded))
    assert payload["urdf_sha256"] and payload["robot_config_sha256"]
    digest = prepare.digest_file
    for suffix in ("rm75b.urdf", "rm75b.yaml"):
        def changed(path, *, suffix=suffix):
            """模拟单个模型或配置文件内容变更。"""
            if path is not None and str(path).endswith(suffix):
                return "changed"
            return digest(path)

        with monkeypatch.context() as context:
            context.setattr(prepare, "digest_file", changed)
            assert prepare.episode_identity(args, tuple(loaded))[0] != original


def test_rm75b_cached_approach_is_rechecked(tmp_path, monkeypatch):
    """缓存结果每次使用时重新检查当前设置生成的接近轨迹。"""
    import handumi.robots.registry as registry
    import handumi.robots.self_collision as collision
    from handumi.dataset.episode_solve_cache import array_digest
    from handumi.robots.self_collision import (
        GRIPPER_BOX_VALIDATION_VERSION,
        CollisionResult,
    )

    args, _, _ = _args_and_loaded(tmp_path, monkeypatch, collision_enabled=True)
    args.approach_seconds = 1.0
    rollout = _rollout()
    rollout["self_collision_validation"] = np.asarray([GRIPPER_BOX_VALIDATION_VERSION])
    rollout["self_collision_qpos_sha256"] = np.asarray([array_digest(rollout["qpos"])["sha256"]])
    rollout["self_collision_min_sampled_m"] = np.asarray([0.1])
    rollout["self_collision_min_links"] = np.asarray(["Link2", "gripper_link"])

    class Checker:
        """合成安全轨迹及不安全的 home 接近段。"""
        def __init__(self, *args):
            """接受与真实碰撞检查器一致的参数。"""

        def check_pose(self, q):
            """所有端点姿态均安全。"""
            return CollisionResult(True, 0.1, ("Link2", "gripper_link"), "clear")

        def check_segment(self, start, end):
            """只有 home 起始段判定为碰撞。"""
            is_home = np.any(start)
            return CollisionResult(not is_home, -0.01 if is_home else 0.1,
                                   ("Link2", "gripper_link"), "collision")

    fake_runtime = SimpleNamespace(robot=None, solver_cls=lambda: SimpleNamespace(robot_collision=None),
                                   load_urdf=lambda: None)
    monkeypatch.setattr(registry, "load_embodiment", lambda _, **kwargs: fake_runtime)
    monkeypatch.setattr(collision, "GripperBoxCollisionChecker", Checker)
    with pytest.raises(SystemExit, match="approach.*Link2/gripper_link"):
        prepare._finish_rollout(args, rollout, cached=True)


def test_collision_summary_rejects_other_model_version(tmp_path, monkeypatch):
    """夹爪盒与完整机械臂模型的缓存认证不可混用。"""
    from handumi.dataset.episode_solve_cache import array_digest
    from handumi.robots.self_collision import (
        COLLISION_VALIDATION_VERSION,
        GRIPPER_BOX_VALIDATION_VERSION,
    )

    args, _, _ = _args_and_loaded(tmp_path, monkeypatch, collision_enabled=True)
    rollout = _rollout()
    rollout["self_collision_qpos_sha256"] = np.asarray([array_digest(rollout["qpos"])["sha256"]])
    rollout["self_collision_min_sampled_m"] = np.asarray([0.1])
    rollout["self_collision_min_links"] = np.asarray(["Link2", "gripper_link"])
    rollout["self_collision_validation"] = np.asarray([COLLISION_VALIDATION_VERSION])
    assert not prepare._valid_collision_summary(args, rollout)
    rollout["self_collision_validation"] = np.asarray([GRIPPER_BOX_VALIDATION_VERSION])
    assert prepare._valid_collision_summary(args, rollout)
    args.retarget_mode = "anchored"
    assert not prepare._valid_collision_summary(args, rollout)


def test_rm75b_direct_start_checks_first_frame_without_home(tmp_path, monkeypatch):
    """直接起始不依赖 home 安全性，仍验收首帧与帧间运动。"""
    import handumi.robots.registry as registry
    import handumi.robots.self_collision as collision
    from handumi.dataset.episode_solve_cache import array_digest

    args, _, _ = _args_and_loaded(tmp_path, monkeypatch, collision_enabled=True)
    replay.resolve_replay_options(args)
    args.approach_seconds = 0.0
    rollout = _rollout()
    rollout["self_collision_validation"] = np.asarray([collision.GRIPPER_BOX_VALIDATION_VERSION])
    rollout["self_collision_qpos_sha256"] = np.asarray([array_digest(rollout["qpos"])["sha256"]])
    rollout["self_collision_min_sampled_m"] = np.asarray([0.1])
    rollout["self_collision_min_links"] = np.asarray(["Link2", "gripper_link"])
    calls = []

    class Checker:
        """模拟不安全 home 和安全的首帧及运动段。"""

        def __init__(self, *args):
            """接受真实检查器的初始化参数。"""

        def check_pose(self, q):
            """记录姿态；仅非零 home 姿态不安全。"""
            calls.append("pose")
            safe = not np.any(q)
            return collision.CollisionResult(safe, 0.1 if safe else 0.0, ("Link2", "gripper_link"), "test")

        def check_segment(self, start, end):
            """记录轨迹段检查。"""
            calls.append("segment")
            return collision.CollisionResult(True, 0.1, ("Link2", "gripper_link"), "clear")

    runtime = SimpleNamespace(robot=None, solver_cls=lambda: SimpleNamespace(robot_collision=None), load_urdf=lambda: None)
    monkeypatch.setattr(registry, "load_embodiment", lambda _, **kwargs: runtime)
    monkeypatch.setattr(collision, "GripperBoxCollisionChecker", Checker)
    prepare._finish_rollout(args, rollout, cached=True)
    assert calls == ["pose", "segment"]
    args.approach_seconds = 1.0
    with pytest.raises(SystemExit, match="home.*Link2/gripper_link"):
        prepare._finish_rollout(args, rollout, cached=True)
