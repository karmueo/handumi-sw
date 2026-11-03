"""验证绝对 TCP 映射、标定选择、严格误差与缓存闭合。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.calibration.deployment import (
    DeploymentCalibration,
    resolve_deployment_calibration,
)
from handumi.calibration.robot_tcp import load_robot_tcp_calibration
from handumi.dataset import screening
from handumi.dataset.screening import robot_fingerprint, solver_signature
from handumi.retargeting.base_placement import (
    PlacementCandidate,
    first_frame_candidates,
    search_feasible_placement,
)
from handumi.robots.kinematics import KinematicsConfig
from handumi.robots.registry import load_embodiment
from handumi.robots.utils import IDENTITY_POSE7, mat_to_pose7, pose7_to_mat, pose_mul
from handumi.scripts import conversion
from handumi.scripts.replay import replay_in_sim as replay


def test_first_frame_optimizer_has_verifiable_minimum_and_respects_lock():
    """笛卡尔 FK 的最小位移可解析求出，同时验证锁关节和完整姿态。"""
    names = tuple(f"q{i}" for i in range(6))

    class CartesianModel:
        """用前三维平移、后三维旋转向量表示工具位姿。"""

        def update_cfg(self, values):
            """保存输入关节。"""
            self.q = np.array([values[name] for name in names])

        def get_transform(self, link):
            """计算精确的齐次 FK。"""
            matrix = np.eye(4)
            matrix[:3, 3] = self.q[:3]
            matrix[:3, :3] = Rotation.from_rotvec(self.q[3:]).as_matrix()
            return matrix

    arm = SimpleNamespace(ee_link="tool")
    lower = np.array([-0.1] * 3 + [-3.0] * 3)
    upper = -lower
    runtime = SimpleNamespace(
        load_urdf=lambda **kwargs: CartesianModel(),
        joint_names=names,
        config=SimpleNamespace(home_q=np.zeros(6), arms={"right": arm}),
        robot=SimpleNamespace(
            joints=SimpleNamespace(lower_limits=lower, upper_limits=upper)
        ),
        arms={"right": arm},
    )
    tool = SimpleNamespace(matrices={"right": np.eye(4)})
    target = np.array([0.3, 0.05, 0, 0, 0, 0, 1])
    candidates = first_frame_candidates(
        runtime, tool, {"right": target}, IDENTITY_POSE7
    )
    assert candidates
    best = candidates[0]
    np.testing.assert_allclose(best.xy, [0.2, 0], atol=1e-5)
    np.testing.assert_allclose(best.q0[:3] + [*best.xy, 0], target[:3], atol=1e-5)
    np.testing.assert_allclose(
        Rotation.from_rotvec(best.q0[3:]).as_matrix(), np.eye(3), atol=1e-5
    )
    locked = first_frame_candidates(
        runtime, tool, {"right": target}, IDENTITY_POSE7, locked=(1,)
    )
    assert locked
    np.testing.assert_allclose(locked[0].xy, [0.2, 0.05], atol=1e-5)
    assert locked[0].q0[1] == pytest.approx(0)


def test_search_keeps_distinct_joint_branches_at_same_xy():
    """同一位置的不同关节分支都需全段验证，不能只凭位置去重。"""
    candidates = [
        PlacementCandidate(np.array([0.1, 0.0]), np.array([value]))
        for value in (0.0, 1.0)
    ]
    checked = []

    def evaluate(candidate):
        """只允许第二个分支，且要求不少于 10 cm 水平移动。"""
        checked.append((candidate.xy.copy(), candidate.q0.copy()))
        return (
            "pass" if candidate.q0[0] == 1 and candidate.xy[0] >= 0.1 - 1e-8 else None
        )

    best, result, diagnostics = search_feasible_placement(candidates, evaluate)
    assert result == "pass" and best.q0[0] == 1
    assert np.linalg.norm(best.xy) == pytest.approx(0.1, abs=0.002)
    assert {q[0] for xy, q in checked if np.allclose(xy, [0.1, 0])} == {0, 1}
    assert diagnostics["oracle_calls"] == len(checked)


def test_search_rejects_failed_full_trajectory_and_refines_nearest_pass():
    """首帧可行但整段失败的候选不得通过，向内细化最小可行半径。"""
    initial = PlacementCandidate(np.array([0.04, 0.0]), np.array([0.0]))
    best, result, diagnostics = search_feasible_placement(
        [initial], lambda candidate: "pass" if candidate.xy[0] >= 0.12 - 1e-8 else None
    )
    assert result == "pass" and 0.12 - 1e-8 <= best.xy[0] <= 0.122
    assert diagnostics["search_resolution_m"] >= 0.002


def test_search_expands_beyond_large_initial_radius():
    """桌面原点较远时，搜索仍从首帧半径向外扩展。"""
    initial = PlacementCandidate(np.array([0.5, 0.0]), np.array([0.0]))
    best, result, diagnostics = search_feasible_placement(
        [initial], lambda candidate: "pass" if candidate.xy[0] >= 0.56 - 1e-8 else None
    )
    assert result == "pass" and best.xy[0] >= 0.56 - 1e-8
    assert diagnostics["oracle_calls"] <= diagnostics["oracle_budget"]


def test_search_fallback_finds_distant_strict_candidate():
    """近端预算耗尽后，粗环仍可找到完整轨迹达标的远处摆放。"""
    initial = PlacementCandidate(np.array([0.04, 0.0]), np.array([0.0]))
    visited = []

    def evaluate(candidate):
        """仅远处负 Y 附近的整段轨迹通过。"""
        visited.append(candidate.xy.copy())
        return "pass" if abs(candidate.xy[0]) < .001 and candidate.xy[1] <= -.25 else None

    best, result, diagnostics = search_feasible_placement([initial], evaluate)
    assert result == "pass"
    assert best.xy[1] <= -.25
    assert diagnostics["oracle_calls"] > 68
    assert any(np.allclose(xy, [0, -.25]) for xy in visited)


def _tool_file(path, matrix):
    """写入只供测试使用的工具标定。"""
    data = {
        "schema_version": 1,
        "kind": "handumi_robot_tcp_calibration",
        "robot": "rm75b",
        "verified": False,
        "calibration": {
            "arms": {"right": {"ee_link": "Link7", "ee_to_tcp": matrix.tolist()}}
        },
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _fixture_replay(
    monkeypatch, tmp_path, *, tool_matrix=None, bad_rotation=False, later_miss=False
):
    """构建以位姿为状态的测试机器人，便于独立检查目标变换。"""
    matrix = np.eye(4) if tool_matrix is None else tool_matrix
    tool_path = _tool_file(tmp_path / "tool.yaml", matrix)
    home = IDENTITY_POSE7.copy()

    class Solver:
        """可直接命中链接位姿的虚拟求解器。"""

        def __init__(self, config=None, locked_joint_indices=()):
            """记录本次求解是否具有逐帧限速。"""
            self.limited = config.max_joint_delta is not None

        def fk_pose7(self, q):
            """返回链接位姿。"""
            return home.copy(), np.asarray(q).copy()

        def ik(self, q, *, left_pose, right_pose):
            """根据测试设置命中或故意错过目标。"""
            if later_miss and self.limited:
                return q.copy()
            result = np.concatenate(right_pose).astype(np.float32)
            if bad_rotation:
                result[3:] = home[3:]
            return result

    arm = SimpleNamespace(ee_link="Link7")
    cfg = SimpleNamespace(
        replay_gripper_mode="normalized",
        gripper_max_width_m=0.1,
        ik_weights=KinematicsConfig(max_joint_delta=0.2),
        replay_max_joint_delta=None,
        replay_ik_weights={},
        home_q=home,
        posture_q=home,
        arms={"right": arm},
    )
    runtime = SimpleNamespace(
        config=cfg,
        arms={"right": arm},
        active_sides=("right",),
        solver_cls=Solver,
        set_finger_positions=lambda *args: None,
    )
    monkeypatch.setattr(replay, "load_embodiment", lambda robot: runtime)
    source_cal = ControllerTcpCalibration(left=home, right=home)
    selection = replay.TcpCalibrationSelection(
        calibration=source_cal,
        source="explicit test",
        source_robot="rm75b",
        source_gripper="test",
        tracking_device="vive",
        controller_mount="test",
        trusted_dataset_snapshot=False,
    )
    monkeypatch.setattr(
        replay, "_resolved_tcp_calibration", lambda *args, **kwargs: selection
    )
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 6] = 1.0
    states[:, 7:10] = [[0.2, 0.1, 0.3], [0.22, 0.1, 0.3]]
    states[:, 10:14] = Rotation.from_euler("y", 90, degrees=True).as_quat()
    info = {
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "tracking_workspace": "table",
        }
    }
    monkeypatch.setattr(
        replay, "load_episode_states", lambda args: (states, 30.0, info, None)
    )
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "rm75b",
            "--retarget-mode",
            "absolute-tcp",
            "--base-placement",
            "fixed",
            "--robot-tcp-calibration",
            str(tool_path),
            "--approach-seconds",
            "0",
        ]
    )
    table = np.r_[
        [0.4, -0.2, 0.1], Rotation.from_euler("z", 30, degrees=True).as_quat()
    ]
    args.deployment_selection = DeploymentCalibration(
        "rm75b", table, None, "assumed", "test", False, "identity", None
    )
    return args, states, table, tool_path


def test_auto_keeps_zero_placement_and_rejects_physical_override(monkeypatch, tmp_path):
    """零位移整段达标不搜索，显式实测标定不可被虚拟摆放覆盖。"""
    args, states, _, _ = _fixture_replay(monkeypatch, tmp_path)
    args.base_placement = "auto"
    args.deployment_selection = DeploymentCalibration(
        "rm75b",
        IDENTITY_POSE7.copy(),
        None,
        "assumed",
        "identity",
        False,
        "identity",
        None,
    )
    monkeypatch.setattr(
        replay,
        "first_frame_candidates",
        lambda *args: pytest.fail("零位移达标不应启动搜索"),
    )
    rollout = replay.solve_episode(args)
    np.testing.assert_allclose(rollout["table_from_robot_pose7"][0], IDENTITY_POSE7)
    np.testing.assert_allclose(
        pose7_to_mat(rollout["target_right_pose7_robot_world"]),
        pose7_to_mat(states[:, 7:14]),
        atol=1e-6,
    )
    assert json.loads(rollout["base_placement_diagnostics_json"][0])["zero_preserved"]
    args.base_placement = "episode-xy"
    args.deployment_selection = DeploymentCalibration(
        "rm75b",
        IDENTITY_POSE7.copy(),
        None,
        "physical",
        "measured",
        True,
        "explicit",
        None,
    )
    with pytest.raises(SystemExit, match="cannot modify a physical"):
        replay.solve_episode(args)


def test_base_placement_policy_enters_cache_signature(tmp_path):
    """自动、固定与逐段摆放不能交叉复用同一求解缓存。"""
    path = _tool_file(tmp_path / "tool.yaml", np.eye(4))
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "rm75b",
            "--retarget-mode",
            "absolute-tcp",
            "--robot-tcp-calibration",
            str(path),
        ]
    )
    replay.resolve_replay_options(args)
    auto_signature = solver_signature(args)
    args.base_placement = "fixed"
    assert solver_signature(args) != auto_signature


def test_absolute_tcp_transform_and_start_orientation(monkeypatch, tmp_path):
    """桌面、工具均有旋转和平移，TCP 目标和 FK 必须完整一致。"""
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_euler("x", 35, degrees=True).as_matrix()
    tool[:3, 3] = [0.03, -0.04, 0.2]
    args, states, table, _ = _fixture_replay(monkeypatch, tmp_path, tool_matrix=tool)
    result = replay.solve_episode(args)
    expected = np.array([pose_mul(table, row[7:14]) for row in states])
    np.testing.assert_allclose(
        result["target_right_pose7_robot_world"], expected, atol=1e-6
    )
    np.testing.assert_allclose(
        pose7_to_mat(result["achieved_right_pose7_robot_world"]),
        pose7_to_mat(expected),
        atol=1e-6,
    )
    # home 的单位姿态和目标首帧相差约 90°，不应被首帧对齐抹除。
    assert np.linalg.norm(expected[0, 3:6]) > 0.5
    np.testing.assert_allclose(result["right_pos_error_m"], 0, atol=1e-6)


def test_start_rotation_failure_and_strict_later_frame(monkeypatch, tmp_path):
    """首帧朝向错误必须失败，后续超差则遵循 strict 开关。"""
    args, _, _, _ = _fixture_replay(monkeypatch, tmp_path, bad_rotation=True)
    with pytest.raises(SystemExit, match="rotation errors"):
        replay.solve_episode(args)
    args, _, _, _ = _fixture_replay(monkeypatch, tmp_path, later_miss=True)
    result = replay.solve_episode(args)
    assert result["right_pos_error_m"][1] > 0.005
    args.strict_ik = True
    with pytest.raises(SystemExit, match=r"1: .*mm/.*deg"):
        replay.solve_episode(args)


def test_initial_solve_uses_home_and_only_active_arm(monkeypatch, tmp_path):
    """单臂首帧从 HOME 求解，未启用侧的运动不参与误差判定。"""
    args, _, _, _ = _fixture_replay(monkeypatch, tmp_path)
    runtime = replay.load_embodiment("rm75b")
    runtime.config.posture_q = np.array([0.9, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    original_solver = runtime.solver_cls
    seeds = []

    class WitnessSolver(original_solver):
        """记录种子并让未启用侧的 FK 随右臂发生偏移。"""

        def ik(self, q, *, left_pose, right_pose):
            """记录首帧输入，再委托基类命中右臂目标。"""
            seeds.append(q.copy())
            return super().ik(q, left_pose=left_pose, right_pose=right_pose)

        def fk_pose7(self, q):
            """为未启用侧生成不参与验收的虚拟位置。"""
            left, right = super().fk_pose7(q)
            left[0] += q[0]
            return left, right

    runtime.solver_cls = WitnessSolver
    result = replay.solve_episode(args)
    np.testing.assert_allclose(seeds[0], runtime.config.home_q)
    assert result["initial_max_position_error_m"][0] < 0.005


def test_nonfinite_source_and_cached_values_rejected(monkeypatch, tmp_path):
    """源姿态和缓存只要包含 NaN 就不能通过 TCP 验收。"""
    args, states, _, _ = _fixture_replay(monkeypatch, tmp_path)
    states[1, 7] = np.nan
    with pytest.raises(SystemExit, match="non-finite.*frame 1"):
        replay.solve_episode(args)
    cache = {"qpos": np.zeros((2, 7)), "frame_indices": np.array([3, 4])}
    for side in ("left", "right"):
        cache[f"{side}_pos_error_m"] = np.zeros(2)
        cache[f"{side}_rot_error_deg"] = np.zeros(2)
    cache["right_rot_error_deg"][1] = np.nan
    resolve = SimpleNamespace(
        retarget_mode="absolute-tcp",
        max_ik_position_error_m=0.005,
        max_ik_rotation_error_deg=5.0,
    )
    with pytest.raises(SystemExit, match="non-finite.*frame 4"):
        conversion._require_absolute_tcp_fidelity(resolve, cache)


def test_nonfinite_ik_output_is_rejected_at_first_frame(monkeypatch, tmp_path):
    """首帧求解器返回非有限关节时不能绕过误差阈值。"""
    args, _, _, _ = _fixture_replay(monkeypatch, tmp_path)
    runtime = replay.load_embodiment("rm75b")
    original_solver = runtime.solver_cls

    class NonFiniteSolver(original_solver):
        """模拟数值求解产生无效关节的故障。"""

        def ik(self, q, *, left_pose, right_pose):
            """返回无穷值供检查。"""
            return np.full_like(q, np.inf)

    runtime.solver_cls = NonFiniteSolver
    with pytest.raises(
        SystemExit, match="non-finite initial IK joint or FK at frame 0"
    ):
        replay.solve_episode(args)


def test_rejects_controller_layout_that_claims_tcp_already_applied(
    monkeypatch, tmp_path
):
    """controller raw 布局不能同时声称已将标定应用于 state。"""
    args, states, _, _ = _fixture_replay(monkeypatch, tmp_path)
    info = {
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "tracking_workspace": "table",
            "controller_tcp_calibration": {"applied_to_state": True},
        }
    }
    monkeypatch.setattr(
        replay, "load_episode_states", lambda args: (states, 30.0, info, None)
    )
    with pytest.raises(SystemExit, match="already-applied"):
        replay.solve_episode(args)


def test_tool_matrix_validation_and_cache_digest(tmp_path):
    """工具位姿方向、非法矩阵、内容变更和缓存签名均受检测。"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [30, 40, 10], degrees=True).as_matrix()
    matrix[:3, 3] = [0.1, 0.2, 0.3]
    path = _tool_file(tmp_path / "tool.yaml", matrix)
    arms = {"right": SimpleNamespace(ee_link="Link7")}
    calibration = load_robot_tcp_calibration("rm75b", arms, explicit_path=path)
    target = np.r_[
        [0.4, -0.3, 0.5], Rotation.from_euler("z", 40, degrees=True).as_quat()
    ]
    ee = calibration.target_ee("right", target)
    np.testing.assert_allclose(
        pose7_to_mat(ee) @ matrix, pose7_to_mat(target), atol=1e-6
    )
    np.testing.assert_allclose(calibration.achieved_tcp("right", ee), target, atol=1e-6)
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "rm75b",
            "--retarget-mode",
            "absolute-tcp",
            "--robot-tcp-calibration",
            str(path),
        ]
    )
    replay.resolve_replay_options(args)
    old_signature = solver_signature(args)
    old_fingerprint = robot_fingerprint("rm75b", robot_tcp_path=path)
    matrix[0, 3] += 0.01
    _tool_file(path, matrix)
    assert solver_signature(args) != old_signature
    assert robot_fingerprint("rm75b", robot_tcp_path=path) != old_fingerprint
    matrix[0, 0] = 3.0
    _tool_file(path, matrix)
    with pytest.raises(SystemExit, match=r"SE\(3\)"):
        load_robot_tcp_calibration("rm75b", arms, explicit_path=path)


def test_identity_fallback_and_option_defaults(tmp_path):
    """仅新模式在缺失桌面标定时采用有来源标记的单位变换。"""
    placement = resolve_deployment_calibration(
        "rm75b", rig_config=tmp_path / "missing.yaml", identity_fallback=True
    )
    assert (
        placement.path is None
        and placement.source == "identity_assumed_table_equals_base"
    )
    np.testing.assert_allclose(placement.pose7, IDENTITY_POSE7)
    with pytest.raises(SystemExit, match="Missing deployment calibration"):
        resolve_deployment_calibration(
            "rm75b", explicit_path=tmp_path / "missing.yaml", identity_fallback=True
        )
    args = replay.build_parser().parse_args(
        ["unused", "--retarget-mode", "absolute-tcp"]
    )
    replay.resolve_replay_options(args)
    assert (
        args.absolute_orientation,
        args.max_ik_position_error_m,
        args.max_ik_rotation_error_deg,
    ) == ("table-absolute", 0.005, 5.0)
    conflicting = replay.build_parser().parse_args(
        [
            "unused",
            "--retarget-mode",
            "absolute-tcp",
            "--absolute-orientation",
            "relative-start",
        ]
    )
    with pytest.raises(SystemExit, match="relative-start conflicts"):
        replay.resolve_replay_options(conflicting)
    old = replay.build_parser().parse_args(
        ["unused", "--retarget-mode", "local-relative"]
    )
    replay.resolve_replay_options(old)
    assert (
        old.absolute_orientation,
        old.max_ik_position_error_m,
        old.max_ik_rotation_error_deg,
    ) == ("relative-start", 0.03, 45.0)

    # 即使机器人存在模拟摆放，auto 也必须按确认的规则采用单位变换。
    piper = resolve_deployment_calibration(
        "piper", rig_config=tmp_path / "missing.yaml", identity_fallback=True
    )
    assert piper.profile == "identity"
    simulation = resolve_deployment_calibration(
        "piper",
        profile="sim",
        rig_config=tmp_path / "missing.yaml",
        identity_fallback=True,
    )
    assert simulation.profile == "sim" and simulation.path.is_file()


@pytest.mark.parametrize("identity_fallback", [False, True])
def test_explicit_local_requires_measured_placement(tmp_path, identity_fallback):
    """显式 local 缺失时必须报错，不能被单位变换回退掩盖。"""
    with pytest.raises(SystemExit, match="No lab-local table calibration"):
        resolve_deployment_calibration(
            "rm75b",
            profile="local",
            rig_config=tmp_path / "missing.yaml",
            identity_fallback=identity_fallback,
        )


def test_default_controller_calibration_invalidates_screening(tmp_path, monkeypatch):
    """默认控制器标定改变后，门禁和旧筛选求解缓存同时失效。"""
    from dataclasses import replace

    from handumi.robots import registry

    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {"total_episodes": 1, "fps": 30, "handumi": {"recording_device": "vive"}}
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    source_path = tmp_path / "controller.yaml"
    tool_path = _tool_file(tmp_path / "robot_tool.yaml", np.eye(4))
    config = replace(
        registry.load_robot_config("rm75b"),
        controller_tcp_calibrations={"vive": source_path},
    )
    monkeypatch.setattr(registry, "load_robot_config", lambda robot: config)
    args = replay.build_parser().parse_args(
        [
            str(root), "--robot", "rm75b", "--retarget-mode", "absolute-tcp",
            "--robot-tcp-calibration", str(tool_path),
        ]
    )
    args.dataset_root = root
    replay.resolve_replay_options(args)

    def write_source(offset):
        """更新默认控制器到 TCP 的平移，保持配置路径不变。"""
        pose = {"position": [offset, 0, 0], "quaternion": [0, 0, 0, 1]}
        source_path.write_text(
            yaml.safe_dump({"calibration": {
                "controller_to_gripper_tcp": {"left": pose, "right": pose}
            }}),
            encoding="utf-8",
        )

    write_source(0.01)
    signature = solver_signature(args)
    assert signature == solver_signature(args, source_info=info)
    payload = {
        "kind": screening.SCREENING_KIND,
        "schema_version": screening.SCREENING_SCHEMA_VERSION,
        "robot": "rm75b",
        "payload_manifest": screening.dataset_payload_manifest(root),
        "robot_fingerprint": screening.robot_fingerprint(
            "rm75b", robot_tcp_path=tool_path
        ),
        "solver_signature": signature,
        "episodes": [{"episode_index": 0, "findings": []}],
        "_solves": {"0/qpos": np.zeros((2, 7), dtype=np.float32)},
    }
    screening.write_screening_report(
        screening.screening_report_path(root, "rm75b"), payload
    )
    assert screening.load_cached_solve(root, robot="rm75b", episode=0, signature=signature)
    assert (
        screening.evaluate_screening_gate(
            root, robot="rm75b", expected_solver_signature=signature,
            expected_robot_tcp_path=tool_path,
        ).status
        == "clear"
    )

    write_source(0.02)
    changed = solver_signature(args)
    assert changed != signature
    assert (
        screening.load_cached_solve(root, robot="rm75b", episode=0, signature=changed)
        is None
    )
    assert (
        screening.evaluate_screening_gate(
            root, robot="rm75b", expected_solver_signature=changed,
            expected_robot_tcp_path=tool_path,
        ).status
        == "stale"
    )

    # 显式选择优先于默认文件；未生效的默认标定变化不应使签名失效。
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_bytes(source_path.read_bytes())
    args.controller_tcp_calibration = explicit_path
    explicit_signature = solver_signature(args)
    write_source(0.03)
    assert solver_signature(args) == explicit_signature
    explicit_path.write_bytes(source_path.read_bytes())
    assert solver_signature(args) != explicit_signature


def test_real_rm75b_ik_on_reachable_tcp_trajectory(monkeypatch, tmp_path):
    """用真实 RM75B FK 构造连续可达示教，要求 TCP 误差通过新门槛。"""
    runtime = load_embodiment("rm75b")
    solver = runtime.solver_cls()
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_euler("xyz", [25, -15, 60], degrees=True).as_matrix()
    tool[:3, 3] = [0.02, -0.03, 0.12]
    tool_path = _tool_file(tmp_path / "tool.yaml", tool)
    tool_pose = mat_to_pose7(tool)
    states = np.zeros((3, 16), dtype=np.float32)
    states[:, 6] = 1
    for i in range(3):
        q = runtime.config.home_q.copy()
        q[0] += 0.01 * i
        q[3] -= 0.005 * i
        _, flange = solver.fk_pose7(q)
        states[i, 7:14] = pose_mul(flange, tool_pose)
    info = {
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "tracking_workspace": "table",
        }
    }
    monkeypatch.setattr(
        replay, "load_episode_states", lambda args: (states, 30.0, info, None)
    )
    identity = ControllerTcpCalibration(left=IDENTITY_POSE7, right=IDENTITY_POSE7)
    selection = replay.TcpCalibrationSelection(
        calibration=identity,
        source="explicit test",
        source_robot="rm75b",
        source_gripper="test",
        tracking_device="vive",
        controller_mount="test",
        trusted_dataset_snapshot=False,
    )
    monkeypatch.setattr(
        replay, "_resolved_tcp_calibration", lambda *args, **kwargs: selection
    )
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "rm75b",
            "--retarget-mode",
            "absolute-tcp",
            "--strict-ik",
            "--robot-tcp-calibration",
            str(tool_path),
            "--rig-config",
            str(tmp_path / "missing.yaml"),
            "--approach-seconds",
            "0",
        ]
    )
    rollout = replay.solve_episode(args)
    assert np.max(rollout["right_pos_error_m"]) < 0.005
    assert np.max(rollout["right_rot_error_deg"]) < 5
    np.testing.assert_allclose(
        pose7_to_mat(rollout["target_right_pose7_robot_world"]),
        pose7_to_mat(states[:, 7:14]),
        atol=1e-6,
    )


def test_conversion_rechecks_cached_and_presolved_and_forces_strict(
    monkeypatch, tmp_path
):
    """转换缓存和预解均不能绕过超差检查，新求解强制 strict。"""
    from handumi.scripts.replay import replay_prepare

    monkeypatch.setattr(replay_prepare, "load_prepared_episode", lambda args: None)
    parser = conversion.build_parser()
    args = parser.parse_args(
        ["unused", "--robot", "rm75b", "--retarget-mode", "absolute-tcp"]
    )
    conversion._resolve_cli_profile(parser, args)
    args.root, args.repo_id, args.revision = tmp_path, "local/test", "main"
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}", encoding="utf-8")
    args.presolved = {}
    rollout = {"qpos": np.zeros((2, 7)), "frame_indices": np.array([8, 9])}
    for side in ("left", "right"):
        rollout[f"{side}_pos_error_m"] = np.zeros(2)
        rollout[f"{side}_rot_error_deg"] = np.zeros(2)
    rollout["right_pos_error_m"][1] = 0.01
    monkeypatch.setattr(
        conversion, "load_cached_solve", lambda *args, **kwargs: rollout
    )
    with pytest.raises(SystemExit, match=r"9: .*mm/.*deg"):
        conversion._solve_with_replay_pipeline(args=args, source_episode_index=0)
    monkeypatch.setattr(conversion, "load_cached_solve", lambda *args, **kwargs: None)
    args.presolved = {0: rollout}
    with pytest.raises(SystemExit, match=r"9: .*mm/.*deg"):
        conversion._solve_with_replay_pipeline(args=args, source_episode_index=0)
    args.presolved = {}
    observed = []
    args.no_solve_cache = True
    monkeypatch.setattr(
        replay,
        "solve_episode",
        lambda replay_args: observed.append(replay_args.strict_ik) or rollout,
    )
    conversion._solve_with_replay_pipeline(args=args, source_episode_index=0)
    assert observed == [True]


def test_screening_passes_tcp_mode_and_records_failure(monkeypatch, tmp_path):
    """筛选把新模式和门槛传给求解器，并将超差标记为拒绝。"""
    observed = []

    def fail_solve(args):
        """模拟轨迹第二帧超差。"""
        observed.append(
            (
                args.retarget_mode,
                args.strict_ik,
                args.max_ik_position_error_m,
                args.max_ik_rotation_error_deg,
                args.robot_tcp_calibration,
            )
        )
        raise SystemExit(
            "IK fidelity threshold exceeded; frames (position/rotation): 1: 10.00mm/6.00deg"
        )

    monkeypatch.setattr(replay, "solve_episode", fail_solve)
    tool_path = tmp_path / "tool.yaml"
    state = {
        "args_template": {
            "root": tmp_path,
            "repo_id": "local/test",
            "revision": "main",
            "robot": "rm75b",
            "deployment_profile": "auto",
            "rig_config": tmp_path / "rig.yaml",
            "retarget_mode": "absolute-tcp",
            "robot_tcp_calibration": tool_path,
            "max_position_error_m": 0.005,
            "max_rotation_error_deg": 5.0,
            "no_cache": True,
        }
    }
    item = screening._screen_one_episode(0, state)
    assert observed == [("absolute-tcp", True, 0.005, 5.0, tool_path)]
    report = screening._grade(
        item, cfg=screening.RetargetScreeningConfig(), fps=30, fences={}
    )
    assert not report.accepted
    assert report.findings[0].code == "retarget_ik_fidelity"
    assert "1: 10.00mm/6.00deg" in report.findings[0].metrics["solver_message"]


def test_screen_all_rejected_retains_signature(monkeypatch, tmp_path):
    """即使没有成功轨迹，筛选报告也能匹配求解设置并标记拒绝。"""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {"total_episodes": 1, "fps": 30, "handumi": {"recording_device": "vive"}}
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    tool_path = _tool_file(tmp_path / "tool.yaml", np.eye(4))
    failed = {
        "episode_index": 0,
        "unreachable": "IK fidelity threshold exceeded; frames (position/rotation): 4: 7mm/6deg",
        "frame_count": 0,
        "metrics": {},
        "solves": {},
        "resolved": {},
    }
    monkeypatch.setattr(screening, "map_episodes", lambda *args, **kwargs: [failed])
    payload = screening.screen_dataset(
        root,
        robot="rm75b",
        retarget_mode="absolute-tcp",
        robot_tcp_calibration=tool_path,
        rig_config=tmp_path / "missing.yaml",
        jobs=1,
        progress=False,
    )
    assert payload["solver_signature"]
    assert payload["summary"]["reject_episode_indices"] == [0]
    screening.write_screening_report(
        screening.screening_report_path(root, "rm75b"), payload
    )
    gate = screening.evaluate_screening_gate(
        root,
        robot="rm75b",
        expected_solver_signature=payload["solver_signature"],
        expected_robot_tcp_path=tool_path,
    )
    assert gate.status == "flagged"


def test_tcp_conversion_gate_rejects_different_screening_signature(tmp_path):
    """旧模式报告不能替代 TCP 筛选，相同配置报告继续通过门禁。"""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text('{"total_episodes": 1}', encoding="utf-8")
    payload = {
        "payload_manifest": screening.dataset_payload_manifest(root),
        "robot_fingerprint": screening.robot_fingerprint("rm75b"),
        "solver_signature": "old-mode-signature",
        "episodes": [{"episode_index": 0, "findings": []}],
    }
    report_path = screening.screening_report_path(root, "rm75b")
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert screening.evaluate_screening_gate(root, robot="rm75b").status == "clear"
    gate = screening.evaluate_screening_gate(
        root, robot="rm75b", expected_solver_signature="tcp-signature"
    )
    assert gate.status == "stale" and "absolute-tcp" in gate.detail
    payload["solver_signature"] = "tcp-signature"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        screening.evaluate_screening_gate(
            root, robot="rm75b", expected_solver_signature="tcp-signature"
        ).status
        == "clear"
    )

    # 显式快照可位于仓库外；门禁应使用本次请求的真实路径校验摘要。
    tool_path = _tool_file(tmp_path / "external.yaml", np.eye(4))
    payload["robot_tcp_calibration_path"] = tool_path.name
    payload["robot_fingerprint"] = screening.robot_fingerprint(
        "rm75b", robot_tcp_path=tool_path
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        screening.evaluate_screening_gate(
            root,
            robot="rm75b",
            expected_solver_signature="tcp-signature",
            expected_robot_tcp_path=tool_path,
        ).status
        == "clear"
    )
