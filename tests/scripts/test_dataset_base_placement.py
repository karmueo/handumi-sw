"""验证数据集共享底座的评分、固定标定与完整帧约束。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.calibration.deployment import DeploymentCalibration
from handumi.retargeting.dataset_placement import (
    Candidate,
    EpisodeTargets,
    choose_finalists,
    generate_candidates,
    rank_result,
    representative_frames,
)
from handumi.robots.utils import pose_inv
from handumi.scripts.replay import replay_dataset_placement as dataset_placement
from handumi.scripts.replay import replay_in_sim as replay


def _reference():
    """构造测试用仿真参考标定。"""
    return DeploymentCalibration(
        robot="rm75b", pose7=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        path=None, scope="simulation", source="test", verified=False,
        profile="sim", lab=None,
    )


def _args(tmp_path, *extra):
    """构造与真实 CLI 一致的共享摆放参数。"""
    args = replay.build_parser().parse_args([
        str(tmp_path), "--robot", "rm75b", "--retarget-mode", "absolute-tcp",
        "--base-placement", "dataset-xy-yaw", *extra,
    ])
    args.repo_id = "test/data"
    args.dataset_root = tmp_path
    replay.resolve_replay_options(args)
    return args


@pytest.mark.parametrize("extra", [
    ("--stride", "2"), ("--start-frame", "1"), ("--max-frames", "3"),
    ("--only-manipulation",), ("--deployment-calibration", "other.yaml"),
    ("--placement-search-candidates", "0"),
])
def test_mode_rejects_cropped_or_conflicting_options(tmp_path, extra):
    """全数据集模式拒绝裁剪与无法确定摆放来源的参数。"""
    with pytest.raises(SystemExit):
        _args(tmp_path, *extra)


def test_mode_requires_rm75b_absolute_tcp(tmp_path):
    """仅支持约定的机器人和重定向方式。"""
    parser = replay.build_parser()
    for extra in (("--robot", "piper", "--retarget-mode", "absolute-tcp"),
                  ("--robot", "rm75b", "--retarget-mode", "anchored")):
        args = parser.parse_args([str(tmp_path), "--base-placement", "dataset-xy-yaw", *extra])
        with pytest.raises(SystemExit):
            replay.resolve_replay_options(args)


def test_unresolved_dataset_policy_cannot_fall_back_to_episode_search(tmp_path):
    """worker 若未收到共享标定就报错，避免悄悄逐段重新摆放。"""
    with pytest.raises(SystemExit, match="has not been resolved"):
        replay.solve_episode(_args(tmp_path))


def test_cache_entrypoints_reject_missing_or_physical_dataset_placement(tmp_path):
    """缓存入口也必须显式收到共享仿真标定，避免命中其他摆放。"""
    from dataclasses import replace

    from handumi.scripts.replay import replay_prepare

    args = _args(tmp_path)
    for entrypoint in (replay_prepare.prepare_episode, replay_prepare.load_prepared_episode):
        with pytest.raises(SystemExit, match="has not been resolved"):
            entrypoint(args)
        args.deployment_selection = replace(_reference(), scope="physical")
        with pytest.raises(SystemExit, match="simulation calibration"):
            entrypoint(args)
        del args.deployment_selection


def test_candidate_generation_preserves_height_tilt_and_reference():
    """八个 yaw 覆盖全周且保留参考位姿，Z 和局部倾斜保持不变。"""
    reference = np.r_[[0.1, -0.2, 0.13], Rotation.from_euler("xyz", [0.12, -0.15, 0.3]).as_quat()]
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [0.1, 0, 0, 0, 0, 0, 1]])
    episode = EpisodeTargets(poses, np.array([[0.2, 0.1, 0.3], [0.3, 0.1, 0.3]]), 0.0, 1.0)
    candidates = generate_candidates(reference, [episode], 0.24, lambda pose, i: [])
    assert {candidate.yaw_index for candidate in candidates} == set(range(8))
    assert any(np.allclose(candidate.pose, reference) for candidate in candidates)
    assert len(candidates) <= 32
    initial = Rotation.from_quat(reference[3:])
    for candidate in candidates:
        assert candidate.pose[2] == reference[2]
        expected = Rotation.from_euler("z", candidate.yaw_index * np.pi / 4) * initial
        assert (expected.inv() * Rotation.from_quat(candidate.pose[3:])).magnitude() < 1e-7


def test_representative_frames_and_finalists_are_deterministic():
    """代表帧包含两端；入围预算包含参考并优先保留各方向。"""
    poses = np.zeros((20, 7))
    poses[:, 0] = np.linspace(0, 1, 20)
    poses[:, 6] = 1
    frames = representative_frames(poses, 0.02, 10)
    assert len(frames) == 9 and frames[0] == 0 and frames[-1] == 19
    assert len(set(frames)) == 9
    reference = poses[0].copy()
    candidates = []
    for yaw in range(8):
        pose = reference.copy()
        pose[0] = yaw * 0.01
        pose[3:] = Rotation.from_euler("z", yaw * np.pi / 4).as_quat()
        candidates.append(Candidate(pose, yaw))
    winners, records = choose_finalists(candidates, reference, 8, lambda c: (c.yaw_index, 5))
    assert len(winners) == len(records) == 8
    assert any(np.allclose(item.pose, reference) for item in winners)
    assert {item.yaw_index for item in winners} == set(range(8))


def test_full_episode_score_prioritizes_complete_then_prefix():
    """较多整段优先于更长前缀，前缀并列再选择较近摆放。"""
    reference = np.array([0, 0, 0, 0, 0, 0, 1.0])
    near = Candidate(reference.copy(), 0)
    far_pose = reference.copy()
    far_pose[0] = 0.4
    far = Candidate(far_pose, 0)
    one_complete = [{"passed": True, "prefix_frames": 3}, {"passed": False, "prefix_frames": 0}]
    no_complete = [{"passed": False, "prefix_frames": 20}, {"passed": False, "prefix_frames": 20}]
    assert rank_result(far, reference, one_complete) < rank_result(near, reference, no_complete)
    assert rank_result(near, reference, one_complete) < rank_result(far, reference, one_complete)


def test_prefix_observer_rebuilds_from_cached_rollout(tmp_path):
    """完整结果与缓存命中均以首个超差帧冻结同一前缀。"""
    args = _args(tmp_path)
    observer = dataset_placement.PrefixObserver()
    for frame, within in enumerate((True, True, False, True)):
        observer.start_frame(frame, 4)
        observer.frame_checked(frame, within)
    assert observer.prefix == 2 and observer.failed_frame == 2
    rollout = {"frame_indices": np.arange(4), "left_pos_error_m": np.zeros(4),
               "right_pos_error_m": np.array([0, 0, 0.03, 0]),
               "left_rot_error_deg": np.zeros(4), "right_rot_error_deg": np.zeros(4)}
    observer.rollout_checked(rollout, args)
    assert observer.prefix == 2 and observer.failed_frame == 2 and observer.total_frames == 4


def test_fixed_placement_is_reused_and_failed_refresh_keeps_old_file(tmp_path, monkeypatch):
    """首次结果发布后永久复用；显式重估零成功时保留旧标定。"""
    args = _args(tmp_path)
    reference = _reference()
    monkeypatch.setattr(dataset_placement, "_reference_calibration", lambda args: reference)
    pose = np.array([0.2, -0.3, 0, 0, 0, 0, 1.0])
    candidate = Candidate(pose, 0)
    report = {"passed_episodes": 1, "episode_count": 2}
    calls = []

    def estimate(args, selected):
        """返回一次成功搜索供持久化测试。"""
        calls.append(1)
        return candidate, dict(report)

    monkeypatch.setattr(dataset_placement, "_estimate", estimate)
    selected = dataset_placement.resolve_dataset_placement(args)
    np.testing.assert_allclose(pose_inv(selected.pose7), pose, atol=1e-6)
    path = tmp_path / ".handumi" / "base_placement" / "rm75b.yaml"
    original = path.read_bytes()
    dataset_placement.resolve_dataset_placement(args)
    assert len(calls) == 1 and path.read_bytes() == original
    args.refresh_cache = True
    dataset_placement.resolve_dataset_placement(args)
    assert len(calls) == 1
    args.refresh_base_placement = True
    monkeypatch.setattr(dataset_placement, "_estimate", lambda args, ref: (candidate, {"passed_episodes": 0}))
    with pytest.raises(SystemExit, match="no fully passing episode"):
        dataset_placement.resolve_dataset_placement(args)
    assert path.read_bytes() == original
    assert json.loads((path.parent / "rm75b.latest.json").read_text())["status"] == "search_failed"


def test_physical_calibration_is_rejected(tmp_path, monkeypatch):
    """实体标定不能在搜索期间被转换成仿真摆放。"""
    args = _args(tmp_path)
    from dataclasses import replace

    monkeypatch.setattr(dataset_placement, "resolve_deployment_calibration", lambda *a, **k: replace(_reference(), scope="physical"))
    with pytest.raises(SystemExit, match="physical"):
        dataset_placement._reference_calibration(args)


def test_full_scoring_uses_standard_non_strict_solve_and_final_gate(tmp_path, monkeypatch):
    """候选评分使用固定摆放、HOME 默认种子以及正式严格门禁。"""
    from handumi.scripts.replay import replay_prepare

    args = _args(tmp_path)
    args.refresh_base_placement = True
    pose = np.array([0.1, 0, 0, 0, 0, 0, 1.0])
    selection = dataset_placement._candidate_calibration(_reference(), pose)
    poses = np.array([[0, 0, 0, 0, 0, 0, 1]] * 3)
    context = {"loaded": (np.zeros((3, 16)),), "runtime": object(), "collision_checker": None,
               "targets_for_search": EpisodeTargets(poses, np.zeros((3, 3)), 0, 1)}
    seen = []
    progress_events = []

    def solve(trial):
        """模拟逐帧求解和一次精度超差。"""
        replay.resolve_replay_options(trial)
        seen.append((trial.base_placement, trial.strict_ik, hasattr(trial, "_initial_q_seed"), trial.deployment_selection))
        trial._replay_progress("逐帧求解", current=1, total=3)
        for frame, within in enumerate((True, False, True)):
            trial._placement_score.start_frame(frame, 3)
            trial._placement_score.frame_checked(frame, within)
        return {}

    def finish(trial, rollout, **kwargs):
        """模拟正式严格验收拒绝超差轨迹。"""
        assert trial.strict_ik
        raise SystemExit("IK fidelity threshold exceeded; frames: 1")

    monkeypatch.setattr(replay, "solve_episode", solve)
    monkeypatch.setattr(replay_prepare, "_finish_rollout", finish)
    result = dataset_placement._full_episode(
        args, context, 0, selection,
        progress_callback=lambda stage, **details: progress_events.append((stage, details)),
    )
    assert seen == [("fixed", False, False, selection)]
    assert progress_events == [("逐帧求解", {"current": 1, "total": 3})]
    assert not hasattr(args, "_replay_progress")
    assert result["passed"] is False and result["prefix_frames"] == 1
    assert result["first_failed_frame"] == 1 and result["total_frames"] == 3


def test_estimation_reports_real_phase_and_frame_progress(tmp_path, monkeypatch):
    """估计协调器报告各阶段及当前帧，并且不把显示回调留在全局参数上。"""
    from types import SimpleNamespace

    from handumi.scripts.replay import replay_progress

    args = _args(tmp_path, "--placement-workers", "1")
    reference = _reference()
    pose = np.array([0, 0, 0, 0, 0, 0, 1.0])
    targets = EpisodeTargets(np.array([pose] * 3), np.zeros((3, 3)), 0, 1)
    context = {"runtime": object(), "tool": object(), "targets": {}, "locked": (),
               "targets_for_search": targets, "loaded": (np.zeros((3, 16)),),
               "input_identity": "test-identity"}
    displays = []

    class Recorder:
        """模拟显示器并记录生命周期及实际阶段事件。"""

        def __init__(self, mode):
            """保存模式，等待进入上下文。"""
            self.mode = mode
            self.events = []
            self.closed = False
            displays.append(self)

        def __enter__(self):
            """返回模拟显示器。"""
            return self

        def __exit__(self, *args):
            """记录显示器关闭。"""
            self.closed = True

        def update(self, stage, current, total, **details):
            """保存阶段、实际完成量和帧细节。"""
            self.events.append((stage, current, total, details))

    def contexts(args, reference, progress):
        """模拟两项输入准备事件。"""
        progress.update("准备输入", 0, 1, detail="episode 1/1")
        progress.update("准备输入", 1, 1, detail="episode 1/1")
        return [context], (0.24, 0.46, 0.18, 0.14)

    def generate(reference, episodes, shoulder, first_xy):
        """触发真实候选生成回调并保留参考位姿。"""
        for _ in range(8):
            first_xy(reference, 0)
        return [Candidate(reference.copy(), 0)]

    def sample_factory(args, contexts, reference, *, frame_progress):
        """在代表帧评分中发出真实帧完成事件。"""
        def sample(candidate):
            """模拟一个代表帧通过。"""
            frame_progress(0, 0, 1)
            frame_progress(0, 1, 1)
            return 1, 1
        return sample

    def full(args, context, episode, selection, *, progress_callback):
        """模拟首帧、逐帧与严格验收的独立进度事件。"""
        progress_callback("首帧求解（可能含编译）")
        progress_callback("逐帧求解", current=2, total=3)
        progress_callback("严格验收")
        return {"episode": episode, "passed": True, "prefix_frames": 3, "total_frames": 3}

    monkeypatch.setattr(replay_progress, "DatasetPlacementProgress", Recorder)
    monkeypatch.setattr(dataset_placement, "_episode_contexts", contexts)
    monkeypatch.setattr(dataset_placement, "generate_candidates", generate)
    monkeypatch.setattr(dataset_placement, "first_frame_candidates", lambda *a, **kw: [SimpleNamespace(xy=[0, 0])])
    monkeypatch.setattr(dataset_placement, "_sample_evaluator", sample_factory)
    monkeypatch.setattr(dataset_placement, "_full_episode", full)
    winner, report = dataset_placement._estimate(args, reference)
    assert winner is not None and report["passed_episodes"] == 1
    assert displays[0].closed and displays[0].mode == "auto"
    events = displays[0].events
    assert {event[0] for event in events} == {"准备输入", "生成候选", "代表帧预筛", "完整轨迹验证"}
    assert any(stage == "生成候选" and current == total == 8 for stage, current, total, _ in events)
    assert any(stage == "代表帧预筛" and details.get("frame_current") == 1
               for stage, _, _, details in events)
    assert any(stage == "完整轨迹验证" and details.get("frame_current") == 2
               and details.get("frame_total") == 3 for stage, _, _, details in events)
    assert not hasattr(args, "_replay_progress")


def test_sample_evaluator_reports_each_sample_and_keeps_ik_scoring(tmp_path):
    """代表帧进度不改变 IK 判定，并分别报告每帧开始与完成。"""
    from types import SimpleNamespace

    from handumi.robots.kinematics import KinematicsConfig

    pose = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)

    class Solver:
        """模拟保持目标位姿的独立无步长 IK。"""

        def __init__(self, *, config, locked_joint_indices):
            """记录传入的关节限制与无步长配置。"""
            assert config.max_joint_delta is None and not locked_joint_indices

        def ik(self, q, **kwargs):
            """直接保持单位位姿。"""
            return q.copy()

        def fk_pose7(self, q):
            """返回与目标一致的末端位姿。"""
            return q.copy(), q.copy()

    runtime = SimpleNamespace(
        config=SimpleNamespace(ik_weights=KinematicsConfig(), replay_ik_weights={}, home_q=pose),
        solver_cls=Solver,
        robot=SimpleNamespace(joints=SimpleNamespace(lower_limits=np.full(7, -10),
                                                     upper_limits=np.full(7, 10))),
    )
    tool = SimpleNamespace(target_ee=lambda side, value: value,
                           achieved_tcp=lambda side, value: value)
    context = {"runtime": runtime, "collision_checker": None, "tool": tool,
               "locked": (), "samples": [0, 2], "gripper_openings": None,
               "targets_for_search": EpisodeTargets(np.array([pose] * 3), np.zeros((3, 3)), 0, 1)}
    events = []
    evaluate = dataset_placement._sample_evaluator(
        _args(tmp_path), [context], _reference(),
        frame_progress=lambda episode, current, total: events.append((episode, current, total)),
    )
    assert evaluate(Candidate(pose.copy(), 0)) == (1, 2)
    assert events == [(0, 0, 2), (0, 1, 2), (0, 1, 2), (0, 2, 2)]


def test_main_estimates_once_freezes_all_workers_and_reports_before_exit(tmp_path, monkeypatch):
    """主入口混合失败时仍处理全段；追加段后复用同一标定。"""
    import sys
    from concurrent.futures import Future
    from types import SimpleNamespace

    from handumi.scripts.replay import replay_scheduler

    count = [3]
    monkeypatch.setattr(replay, "ensure_metadata", lambda _: {"total_episodes": count[0]})
    monkeypatch.setattr(dataset_placement, "ensure_metadata", lambda _: {"total_episodes": count[0]})
    monkeypatch.setattr(replay, "resolve_dataset_selection", lambda *a, **k:
                        SimpleNamespace(repo_id="local/test", root=tmp_path))
    monkeypatch.setattr(replay.time, "sleep", lambda _: None)
    monkeypatch.setattr(dataset_placement, "_reference_calibration", lambda args: _reference())
    estimates = []
    base = np.array([0.2, -0.3, 0, 0, 0, 0, 1.0])

    def estimate(args, reference):
        """返回一段严格通过的模拟全数据集估计结果。"""
        estimates.append(count[0])
        return Candidate(base.copy(), 0), {"passed_episodes": 1, "episode_count": count[0]}

    monkeypatch.setattr(dataset_placement, "_estimate", estimate)
    submissions = []

    class ImmediateExecutor:
        """以确定的混合成功和失败模拟工作进程并记录收到的摆放。"""

        def submit(self, fn, values, episode, refresh):
            """立即完成一个提交，并保留传给 worker 的公共参数。"""
            submissions.append((episode, values["deployment_selection"].pose7.copy()))
            future = Future()
            metrics = {"prefix_frames": 1 if episode == 1 else 2, "total_frames": 2,
                       "solve_identity": f"episode-{episode}"}
            response = {"placement_metrics": metrics, "stage_seconds": {"solve": 1}}
            if episode == 1:
                response["error"] = "first frame unreachable"
            elif episode == values["_batch_current"]:
                response["rollout"] = {"qpos": np.zeros((2, 1))}
            future.set_result(response)
            return future

        def shutdown(self, **kwargs):
            """兼容批处理调度器的资源清理。"""

    original_scheduler = replay_scheduler.EpisodeScheduler

    class InlineScheduler(original_scheduler):
        """保留真实队列和结果汇总，仅替换进程执行器。"""

        def __init__(self, *args, **kwargs):
            """注入同步 executor，避免测试启动硬件和 JAX worker。"""
            kwargs["executor"] = ImmediateExecutor()
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(replay_scheduler, "EpisodeScheduler", InlineScheduler)
    monkeypatch.setattr(sys, "argv", ["handumi", str(tmp_path), "--robot", "rm75b",
        "--retarget-mode", "absolute-tcp", "--base-placement", "dataset-xy-yaw",
        "--headless", "--prepare-mode", "all", "--progress", "off"])
    for expected_count in (3, 4):
        count[0] = expected_count
        submissions.clear()
        with pytest.raises(SystemExit, match="Preparation failed"):
            replay.main()
        assert [item[0] for item in submissions] == list(range(expected_count))
        for _, pose in submissions:
            np.testing.assert_allclose(pose_inv(pose), base, atol=1e-6)
        report_path = tmp_path / ".handumi" / "base_placement" / "rm75b.latest.json"
        report = json.loads(report_path.read_text())
        assert report["evaluated_episodes"] == expected_count
        assert report["failed_episodes"] == 1 and report["total_frames"] == 2 * expected_count
        assert report["prefix_frames"] == 2 * expected_count - 1
        assert report["verified_prefix_rate"] == pytest.approx((2 * expected_count - 1) / (2 * expected_count))
        assert report["episodes"][1]["error"] == "first frame unreachable"
        assert report["episodes"][0]["solve_identity"] == "episode-0"
    assert estimates == [3]


def test_initialization_failure_writes_diagnostic_and_preserves_calibration(tmp_path, monkeypatch):
    """输入初始化异常写明搜索失败，且不替换已有固定标定。"""
    args = _args(tmp_path)
    args.refresh_base_placement = True
    directory = tmp_path / ".handumi" / "base_placement"
    directory.mkdir(parents=True)
    calibration = directory / "rm75b.yaml"
    calibration.write_text("old calibration")
    monkeypatch.setattr(dataset_placement, "_reference_calibration", lambda args: _reference())

    def fail(args, reference):
        """模拟读取某一 episode 时出现非法输入。"""
        raise ValueError("episode 2: invalid input")

    monkeypatch.setattr(dataset_placement, "_estimate", fail)
    with pytest.raises(ValueError, match="invalid input"):
        dataset_placement.resolve_dataset_placement(args)
    assert calibration.read_text() == "old calibration"
    report = json.loads((directory / "rm75b.latest.json").read_text())
    assert report["status"] == "search_error" and "episode 2" in report["error"]


def test_parallel_search_preserves_serial_ranking_and_report(tmp_path, monkeypatch):
    """相同首帧和评分输入下，并行协调器与串行路径保留相同搜索结果。"""
    from types import SimpleNamespace

    from handumi.scripts.replay import replay_placement_pool

    args = _args(tmp_path, "--placement-workers", "1", "--placement-search-candidates", "3")
    reference = _reference()
    pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    contexts = []
    for index in range(2):
        poses = np.array([pose, pose + np.array([0.05, 0, 0, 0, 0, 0, 0])])
        targets = EpisodeTargets(poses, np.array([[0.3 + index * 0.1, 0, 0.1]] * 2), 0, 0.9)
        contexts.append({"runtime": object(), "tool": object(), "targets": {"index": index},
                         "locked": (), "targets_for_search": targets,
                         "loaded": (np.zeros((2, 16)),), "input_identity": f"ep{index}"})

    def first(_runtime, _tool, targets, _base, _locked, max_seeds):
        """返回与段相关且与执行顺序无关的首帧候选。"""
        return [SimpleNamespace(xy=np.array([0.1 * targets["index"], 0.2]))]

    def sample(_args, _contexts, _reference, *, frame_progress):
        """使用方向和水平位置决定代表帧分数。"""
        return lambda candidate: (2 - candidate.yaw_index % 2, 10 - candidate.yaw_index)

    def result(episode, selection):
        """给定同一摆放时返回稳定的整段评分。"""
        x = float(pose_inv(selection.pose7)[0])
        return {"episode": episode, "passed": x < 0.2,
                "prefix_frames": 2 if x < 0.2 else 1, "total_frames": 2}

    def full(_args, _context, episode, selection, *, progress_callback):
        """模拟串行整段评分。"""
        return result(episode, selection)

    class Pool:
        """按稳定索引模拟并行池输出，检验上层归并与报告。"""
        def __init__(self, args, given_contexts, ref, workers, progress):
            """保存池所使用的参考标定及上下文。"""
            assert given_contexts is contexts and workers == 3
            self.reference = ref

        def __enter__(self):
            """进入模拟池。"""
            return self

        def __exit__(self, *unused):
            """退出模拟池。"""

        def first_frames(self, reference_pose):
            """输出与串行首帧求解器一致的候选。"""
            return [[[np.array([0.1 * episode, 0.2])] for episode in range(2)]
                    for _ in range(8)]

        def sample_scores(self, candidates):
            """返回与串行路径相同的代表帧评分。"""
            return [(2 - candidate.yaw_index % 2, 10 - candidate.yaw_index)
                    for candidate in candidates]

        def full_scores(self, candidates):
            """返回与串行路径相同的完整轨迹评分。"""
            return [[result(episode, dataset_placement._candidate_calibration(
                self.reference, candidate.pose)) for episode in range(2)]
                    for candidate in candidates]

    class Progress:
        """丢弃测试期间的进度事件。"""
        def update(self, *args, **kwargs):
            """不显示进度。"""

    monkeypatch.setattr(dataset_placement, "_episode_contexts", lambda *args: (contexts, (0.2,)))
    monkeypatch.setattr(dataset_placement, "first_frame_candidates", first)
    monkeypatch.setattr(dataset_placement, "_sample_evaluator", sample)
    monkeypatch.setattr(dataset_placement, "_full_episode", full)
    monkeypatch.setattr(replay_placement_pool, "PlacementPool", Pool)
    serial_winner, serial = dataset_placement._estimate_with_progress(args, reference, Progress())
    args.placement_workers = 3
    parallel_winner, parallel = dataset_placement._estimate_with_progress(args, reference, Progress())
    np.testing.assert_allclose(parallel_winner.pose, serial_winner.pose, atol=1e-5, rtol=1e-5)
    for key in ("sample_candidates", "candidates", "passed_episodes", "prefix_frames",
                "verified_prefix_rate", "table_from_robot_pose7"):
        assert parallel[key] == serial[key]


def test_placement_workers_must_be_positive(tmp_path):
    """通过 CLI 和内部选项解析均拒绝非正数的摆放进程数。"""
    with pytest.raises(ValueError, match="placement-workers"):
        _args(tmp_path, "--placement-workers", "0")
