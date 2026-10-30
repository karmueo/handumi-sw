"""工具 TCP 相对 IK 与逐帧附加记录的合成数据测试。

测试使用位姿回显假求解器，不读取采集设备、不启动 Rerun，也不依赖真实
机器人。最后一项集成测试复用仓库现有的 RM75B 机器人运行时。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.dataset.robot_ik import (
    NPZ_FIELDS,
    RobotIkEpisodeRecorder,
    RobotIkPublishError,
    RobotIkStateError,
    find_incomplete_records,
    has_complete_record,
    record_paths,
)
from handumi.robots.registry import load_embodiment
from handumi.robots.umi_ik import (
    FAILURE_FK_SOLVER_ERROR,
    FAILURE_IK_SOLVER_ERROR,
    FAILURE_INVALID_JOINT_SHAPE,
    FAILURE_JOINT_LIMIT_VIOLATION,
    FAILURE_NON_FINITE_JOINT_ANGLES,
    FAILURE_NON_FINITE_SOURCE_POSE,
    FAILURE_ORIENTATION_ERROR,
    FAILURE_POSITION_ERROR,
    FAILURE_TRACKING_LOST,
    IkThresholds,
    RelativeMappingConfig,
    RelativeToolIk,
    ToolCalibration,
    load_tool_calibration,
)
from handumi.robots.utils import pose_mul

HOME_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
LIMITS = (-10.0, 10.0)
IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
ROTATION_X_90 = Rotation.from_euler("x", 90.0, degrees=True).as_quat().astype(
    np.float32
)


def _tool_matrix(z_offset: float = 0.195) -> np.ndarray:
    """合成带旋转和 Z 偏移的工具矩阵，避免读取本机标定。"""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", [12.0, -7.0, 25.0], degrees=True
    ).as_matrix()
    matrix[:3, 3] = [0.012, -0.008, z_offset]
    return matrix


def _pure_z_tool(z_offset: float = 0.195) -> ToolCalibration:
    """仅沿法兰 Z 轴平移的工具标定，便于验证杆臂效应。"""
    matrix = np.eye(4, dtype=np.float64)
    matrix[2, 3] = z_offset
    return ToolCalibration.from_matrix(matrix, source="synthetic_z_tool.yaml")


class EchoSolver:
    """把法兰目标直接当关节状态返回的假求解器。

    该假模型不是真实关节空间，但能精确验证工具偏移、相对映射和阈值。
    ``*_after`` 参数控制从第几次 IK/FK 调用开始注入失败。
    """

    def __init__(
        self,
        *,
        offset: np.ndarray | None = None,
        offset_after: int = 0,
        quaternion_delta_deg: float | None = None,
        quaternion_delta_after: int = 0,
        bad_q: np.ndarray | None = None,
        bad_q_after: int = 0,
        hold_q: np.ndarray | None = None,
        hold_ik_at: set[int] | None = None,
        raise_ik_after: int | None = None,
        raise_fk_after: int | None = None,
    ) -> None:
        self.offset = None if offset is None else np.asarray(offset, np.float32)
        self.offset_after = offset_after
        self.quaternion_delta_deg = quaternion_delta_deg
        self.quaternion_delta_after = quaternion_delta_after
        self.bad_q = None if bad_q is None else np.asarray(bad_q, np.float32)
        self.bad_q_after = bad_q_after
        self.hold_q = None if hold_q is None else np.asarray(hold_q, np.float32)
        self.hold_ik_at = set(hold_ik_at or ())
        self.raise_ik_after = raise_ik_after
        self.raise_fk_after = raise_fk_after
        self.ik_calls = 0
        self.fk_calls = 0

    def ik(
        self,
        q_current: np.ndarray,
        *,
        left_pose=None,
        right_pose=None,
    ) -> np.ndarray:
        """返回目标法兰位姿（或注入的失败值）。"""
        del q_current, left_pose
        self.ik_calls += 1
        if self.raise_ik_after is not None:
            if self.ik_calls > self.raise_ik_after:
                raise RuntimeError("simulated IK failure")
        assert right_pose is not None
        if self.hold_q is not None:
            return self.hold_q.copy()
        if self.ik_calls in self.hold_ik_at:
            return HOME_Q.copy()
        q = np.r_[right_pose[0], right_pose[1]].astype(np.float32)
        if self.bad_q is not None and self.ik_calls > self.bad_q_after:
            return self.bad_q.copy()
        if self.offset is not None and self.ik_calls > self.offset_after:
            q[:3] += self.offset
        if (
            self.quaternion_delta_deg is not None
            and self.ik_calls > self.quaternion_delta_after
        ):
            delta = Rotation.from_euler(
                "x", self.quaternion_delta_deg, degrees=True
            ).as_quat()
            q[3:7] = Rotation.from_quat(q[3:7]).as_quat()
            q[3:7] = (Rotation.from_quat(q[3:7]) * Rotation.from_quat(delta)).as_quat()
        return q.astype(np.float32)

    def fk_pose7(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把关节状态原样作为左右法兰位姿返回。"""
        self.fk_calls += 1
        if self.raise_fk_after is not None:
            if self.fk_calls > self.raise_fk_after:
                raise RuntimeError("simulated FK failure")
        pose = np.asarray(q, dtype=np.float32).copy()
        return pose, pose


def _controller(
    *,
    solver=None,
    tool: ToolCalibration | None = None,
    thresholds: IkThresholds | None = None,
    mapping: RelativeMappingConfig | None = None,
) -> RelativeToolIk:
    """构造使用假求解器的单臂工具 IK 控制器。"""
    return RelativeToolIk(
        solver=EchoSolver() if solver is None else solver,
        home_q=HOME_Q,
        joint_names=JOINT_NAMES,
        lower_limits=np.full(7, LIMITS[0], dtype=np.float32),
        upper_limits=np.full(7, LIMITS[1], dtype=np.float32),
        tool_calibration=(
            ToolCalibration.from_matrix(_tool_matrix(), source="synthetic_tool.yaml")
            if tool is None
            else tool
        ),
        thresholds=thresholds,
        relative_mapping=mapping,
        side="right",
        robot_id="rm75b-test",
        ik_parameters={
            "pos_weight": 100.0,
            "ori_weight": 15.0,
            "rest_weight": 1.0,
            "max_joint_delta": 0.12,
            "max_reach": 0.9,
        },
    )


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """构造单位旋转、给定平移的源位姿。"""
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _source_frames(count: int, step: float = 0.01) -> list[tuple[float, np.ndarray]]:
    """生成按原始帧顺序排列的时间戳和源 TCP 序列。"""
    return [
        (10.0 + index / 30.0, _pose(x=index * step))
        for index in range(count)
    ]


def _make_recorder(
    root: Path,
    *,
    solver_factory=EchoSolver,
    thresholds: IkThresholds | None = None,
) -> RobotIkEpisodeRecorder:
    """构造每次补算都新建控制器的记录器。"""

    def factory() -> RelativeToolIk:
        return _controller(
            solver=solver_factory(),
            tool=_pure_z_tool(),
            thresholds=thresholds,
        )

    return RobotIkEpisodeRecorder(root, factory)


def _local_fastumi_tool() -> ToolCalibration:
    """读取本机 fastumi 工具标定；文件不存在时跳过真实模型测试。"""
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "calibration"
        / "umi"
        / "local"
        / "fastumi"
        / "T_end_tcp.yaml"
    )
    if not path.is_file():
        pytest.skip("本机 fastumi 工具标定不存在")
    return load_tool_calibration(path)


@pytest.fixture(scope="module")
def rm75b_runtime():
    """加载一次 RM75B 运行时，供真实模型回归测试复用。"""
    return load_embodiment("rm75b")


# ---------------------------------------------------------------------------
# 工具标定与接口校验
# ---------------------------------------------------------------------------


def test_tool_calibration_roundtrip_and_validation() -> None:
    """工具矩阵校验、pose7 转换、逆变换和来源字段保持一致。"""
    calibration = ToolCalibration.from_matrix(
        _tool_matrix(), source="synthetic.yaml", notes=("人工调整 Z=0.195",)
    )
    assert calibration.matrix.shape == (4, 4)
    assert calibration.matrix[2, 3] == pytest.approx(0.195)
    assert calibration.tool_pose7[2] == pytest.approx(0.195)
    assert calibration.to_metadata()["notes"] == ["人工调整 Z=0.195"]
    identity = pose_mul(calibration.tool_pose7, calibration.inverse_pose7)
    np.testing.assert_allclose(identity[:3], 0.0, atol=1e-6)
    assert abs(float(np.dot(identity[3:7], [0, 0, 0, 1]))) == pytest.approx(1.0)


def test_tool_calibration_yaml_preserves_source_comments(tmp_path) -> None:
    """YAML 读取保留文件摘要和顶部人工调整注释。"""
    matrix = _tool_matrix()
    values = "\n".join(
        "  - [" + ", ".join(f"{value:.15g}" for value in row) + "]"
        for row in matrix
    )
    path = tmp_path / "T_end_tcp.yaml"
    path.write_text(
        "# 工具 TCP 标定。\n"
        "# Z 平移按用户要求改为 0.195 m。\n"
        "T_end_tcp:\n"
        f"{values}\n",
        encoding="utf-8",
    )
    calibration = load_tool_calibration(path)
    assert calibration.sha256 is not None and len(calibration.sha256) == 64
    assert calibration.source == str(path)
    assert any("0.195" in note for note in calibration.notes)
    np.testing.assert_allclose(calibration.matrix, matrix, atol=1e-12)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        np.array([[np.nan, 0, 0, 0]] * 4),
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
            dtype=float,
        ),
        np.array(
            [[1, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        ),
        np.diag([1.0, 1.0, -1.0, 1.0]),
    ],
)
def test_tool_calibration_rejects_invalid_transforms(matrix) -> None:
    """非 4×4、非有限、坏齐次行、非正交和镜像矩阵都被拒绝。"""
    with pytest.raises(ValueError):
        ToolCalibration.from_matrix(matrix)


def test_local_deployment_tool_calibration_has_expected_offset() -> None:
    """本机部署标定的 Z 偏移为 0.195 m，且保留人工调整说明。"""
    calibration = _local_fastumi_tool()
    assert calibration.matrix[2, 3] == pytest.approx(0.195)
    assert any("0.195" in note for note in calibration.notes)


def test_thresholds_and_mapping_reject_invalid_values() -> None:
    """阈值、缩放和世界旋转只接受有限且物理上有效的值。"""
    for position, orientation in ((0.0, 5.0), (-0.01, 5.0), (0.01, np.nan)):
        with pytest.raises(ValueError):
            IkThresholds(position, orientation)
    for scale in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            RelativeMappingConfig(translation_scale=scale)
    with pytest.raises(ValueError):
        RelativeMappingConfig(source_world_to_robot_world=np.ones((3, 3)))


# ---------------------------------------------------------------------------
# 相对映射、工具偏移与杆臂效应
# ---------------------------------------------------------------------------


def test_first_frame_aligns_to_home_tool_pose() -> None:
    """首个有效 TCP 建立起点，但目标固定在 home 工具 TCP。"""
    solver = EchoSolver()
    controller = _controller(solver=solver, tool=_pure_z_tool())
    frame = controller.process(_pose(x=0.4, y=-0.3, z=0.2))
    assert frame.valid
    np.testing.assert_allclose(frame.target_tcp_pose7, controller.home_tcp_pose7)
    np.testing.assert_allclose(frame.q, HOME_Q)
    assert frame.failure_reason == ""
    assert controller.started
    assert solver.ik_calls == 0


def test_warmup_compiles_without_changing_state() -> None:
    """预热只编译 IK 路径，不改变控制器状态或触发首帧 IK。"""
    solver = EchoSolver()
    controller = _controller(solver=solver)
    q_before = controller.q
    controller.warmup()
    assert solver.ik_calls == 1
    assert controller.frames_processed == 0
    assert not controller.started
    np.testing.assert_allclose(controller.q, q_before)
    frame = controller.process(IDENTITY_POSE)
    assert frame.valid
    assert solver.ik_calls == 1


def test_translation_maps_one_to_one_and_applies_tool_offset_once() -> None:
    """平移增量 1:1 映射，工具偏移只在目标/实际换算中出现一次。"""
    tool = _pure_z_tool()
    controller = _controller(tool=tool)
    first = controller.process(IDENTITY_POSE)
    assert first.valid
    np.testing.assert_allclose(
        first.target_tcp_pose7[:3], [0.0, 0.0, 0.195], atol=1e-6
    )

    second = controller.process(_pose(x=0.01, y=-0.02, z=0.03))
    assert second.valid
    np.testing.assert_allclose(
        second.target_tcp_pose7[:3], [0.01, -0.02, 0.225], atol=1e-6
    )
    np.testing.assert_allclose(
        second.target_end_pose7[:3], [0.01, -0.02, 0.03], atol=1e-6
    )
    np.testing.assert_allclose(second.actual_tcp_pose7, second.target_tcp_pose7)
    offset_once = pose_mul(second.actual_end_pose7, tool.tool_pose7)
    np.testing.assert_allclose(offset_once, second.actual_tcp_pose7, atol=1e-6)
    assert second.position_error_m < 1e-6


def test_rotation_moves_flange_with_tool_lever_arm() -> None:
    """工具 TCP 原地旋转时，法兰目标必须包含杆臂补偿。"""
    controller = _controller(tool=_pure_z_tool())
    first = controller.process(IDENTITY_POSE)
    assert first.valid
    rotation_pose = np.r_[0.0, 0.0, 0.0, ROTATION_X_90]
    second = controller.process(rotation_pose)
    assert second.valid
    np.testing.assert_allclose(
        second.target_tcp_pose7[:3], [0.0, 0.0, 0.195], atol=1e-6
    )
    np.testing.assert_allclose(
        second.target_end_pose7[:3], [0.0, 0.195, 0.195], atol=1e-5
    )
    assert np.linalg.norm(
        second.target_end_pose7[:3] - second.target_tcp_pose7[:3]
    ) == pytest.approx(0.195, abs=1e-5)
    np.testing.assert_allclose(second.actual_tcp_pose7, second.target_tcp_pose7)


def test_position_and_orientation_thresholds_hold_last_valid_frame() -> None:
    """位置/姿态超阈值时判无效，并保留上一有效关节状态。"""
    position_ok = _controller(
        solver=EchoSolver(offset=np.array([0.009, 0.0, 0.0]), offset_after=0),
        tool=_pure_z_tool(),
    )
    position_ok.process(IDENTITY_POSE)
    ok_frame = position_ok.process(_pose(x=0.02))
    assert ok_frame.valid
    assert ok_frame.position_error_m == pytest.approx(0.009, abs=1e-5)

    position_bad = _controller(
        solver=EchoSolver(offset=np.array([0.0105, 0.0, 0.0]), offset_after=0),
        tool=_pure_z_tool(),
    )
    position_bad.process(IDENTITY_POSE)
    bad_frame = position_bad.process(_pose(x=0.02))
    assert not bad_frame.valid
    assert FAILURE_POSITION_ERROR in bad_frame.failure_reasons
    np.testing.assert_allclose(bad_frame.q, HOME_Q)

    identity_tool = ToolCalibration.from_matrix(np.eye(4))
    orientation_ok = _controller(
        solver=EchoSolver(quaternion_delta_deg=4.0, quaternion_delta_after=0),
        tool=identity_tool,
    )
    orientation_ok.process(IDENTITY_POSE)
    assert orientation_ok.process(_pose(x=0.02)).valid

    orientation_bad = _controller(
        solver=EchoSolver(quaternion_delta_deg=6.0, quaternion_delta_after=0),
        tool=identity_tool,
    )
    orientation_bad.process(IDENTITY_POSE)
    bad_orientation = orientation_bad.process(_pose(x=0.02))
    assert not bad_orientation.valid
    assert FAILURE_ORIENTATION_ERROR in bad_orientation.failure_reasons
    assert bad_orientation.attempted_orientation_error_deg > 5.0


def test_failed_frame_holds_pose_then_continues_same_mapping() -> None:
    """失败帧保持姿态，恢复后仍按上一目标累计同段相对运动。"""
    controller = _controller(
        solver=EchoSolver(hold_ik_at={1}), tool=_pure_z_tool()
    )
    first = controller.process(IDENTITY_POSE)
    failed = controller.process(_pose(x=0.05))
    recovered = controller.process(_pose(x=0.10))
    assert first.valid
    assert not failed.valid
    assert failed.failure_reason == FAILURE_POSITION_ERROR
    np.testing.assert_allclose(failed.q, first.q)
    np.testing.assert_allclose(failed.actual_tcp_pose7, first.actual_tcp_pose7)
    assert recovered.valid
    np.testing.assert_allclose(
        recovered.target_tcp_pose7[:3], [0.10, 0.0, 0.195], atol=1e-6
    )
    np.testing.assert_allclose(recovered.actual_tcp_pose7, recovered.target_tcp_pose7)


def test_tracking_loss_and_non_finite_source_do_not_advance_mapping() -> None:
    """跟踪丢失/非有限源位姿不推进上一源，恢复后继续同段映射。"""
    controller = _controller(tool=_pure_z_tool())
    first = controller.process(IDENTITY_POSE)
    lost = controller.process(None)
    recovered = controller.process(_pose(x=0.02))
    non_finite = _pose(x=0.04)
    non_finite[0] = np.nan
    invalid = controller.process(non_finite)
    recovered_again = controller.process(_pose(x=0.04))

    assert lost.failure_reason == FAILURE_TRACKING_LOST
    np.testing.assert_allclose(lost.q, first.q)
    np.testing.assert_allclose(lost.target_tcp_pose7, first.target_tcp_pose7)
    assert recovered.valid
    np.testing.assert_allclose(
        recovered.target_tcp_pose7[:3], [0.02, 0.0, 0.195], atol=1e-6
    )
    assert invalid.failure_reason == FAILURE_NON_FINITE_SOURCE_POSE
    np.testing.assert_allclose(invalid.q, recovered.q)
    np.testing.assert_allclose(invalid.target_tcp_pose7, recovered.target_tcp_pose7)
    assert recovered_again.valid
    np.testing.assert_allclose(
        recovered_again.target_tcp_pose7[:3], [0.04, 0.0, 0.195], atol=1e-6
    )


@pytest.mark.parametrize(
    "solver, expected_reason",
    [
        (EchoSolver(raise_ik_after=0), FAILURE_IK_SOLVER_ERROR),
        (
            EchoSolver(bad_q=np.full(7, np.nan), bad_q_after=0),
            FAILURE_NON_FINITE_JOINT_ANGLES,
        ),
        (
            EchoSolver(bad_q=np.full(7, 100.0), bad_q_after=0),
            FAILURE_JOINT_LIMIT_VIOLATION,
        ),
        (
            EchoSolver(bad_q=np.zeros(6), bad_q_after=0),
            FAILURE_INVALID_JOINT_SHAPE,
        ),
        (EchoSolver(raise_fk_after=2), FAILURE_FK_SOLVER_ERROR),
    ],
)
def test_solver_failures_are_reported_and_held(solver, expected_reason) -> None:
    """IK/FK 异常、非有限关节、越界和错误形状都保持上一有效状态。"""
    controller = _controller(solver=solver, tool=_pure_z_tool())
    first = controller.process(IDENTITY_POSE)
    failed = controller.process(_pose(x=0.02))
    assert first.valid
    assert not failed.valid
    assert expected_reason in failed.failure_reasons
    np.testing.assert_allclose(failed.q, first.q)
    np.testing.assert_allclose(failed.actual_tcp_pose7, first.actual_tcp_pose7)


def test_metadata_records_configuration_and_current_anchor() -> None:
    """元信息包含机器人、工具、映射、IK 参数、阈值和本段起点。"""
    controller = _controller(tool=_pure_z_tool())
    controller.process(_pose(x=0.1))
    metadata = controller.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata["robot"]["name"] == "rm75b-test"
    assert metadata["robot"]["joint_names"] == list(JOINT_NAMES)
    assert metadata["robot"]["home_q_rad"] == HOME_Q.tolist()
    assert metadata["tool_calibration"]["matrix"][2][3] == pytest.approx(0.195)
    assert metadata["relative_mapping"]["translation_scale"] == 1.0
    assert metadata["relative_mapping"]["max_reach_m"] is None
    assert metadata["relative_mapping"]["compose_source"] == "commanded"
    assert metadata["relative_mapping"]["anchor_source_tcp_pose7"] == _pose(
        x=0.1
    ).tolist()
    assert np.asarray(metadata["relative_mapping"]["adapter_rotation"]).shape == (
        3,
        3,
    )
    assert metadata["ik"]["parameters"]["max_joint_delta"] == 0.12
    assert metadata["validity"]["max_position_error_m"] == 0.01
    assert metadata["validity"]["max_orientation_error_deg"] == 5.0


# ---------------------------------------------------------------------------
# 逐帧补算、NPZ/JSON 与发布生命周期
# ---------------------------------------------------------------------------


def test_recorder_recomputes_every_frame_and_publishes(tmp_path) -> None:
    """完整逐帧补算保持帧序/时间戳，并生成 NPZ/JSON 后原子发布。"""
    frames = _source_frames(4)
    recorder = _make_recorder(tmp_path)
    recorder.begin_episode()
    for timestamp, pose in frames:
        recorder.add_frame(timestamp, pose)
    progress: list[tuple[int, int]] = []
    prepared = recorder.prepare_episode(0, progress=lambda done, total: progress.append((done, total)))
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert prepared.frame_count == 4
    assert prepared.valid_count == 4
    assert prepared.invalid_count == 0
    assert prepared.failure_reasons == {}
    assert prepared.npz_path.is_file()
    assert prepared.json_path.is_file()
    assert not record_paths(tmp_path, 0).npz.exists()
    assert not has_complete_record(tmp_path, 0)

    with np.load(prepared.npz_path) as data:
        assert set(data.files) == set(NPZ_FIELDS)
        np.testing.assert_array_equal(data["frame_index"], np.arange(4))
        np.testing.assert_allclose(
            data["timestamp_s"], [timestamp for timestamp, _ in frames]
        )
        np.testing.assert_allclose(data["q_rad"][0], HOME_Q)
        assert data["q_rad"].shape == (4, 7)
        assert data["valid"].all()
        assert list(data["failure_reason"]) == [""] * 4
        np.testing.assert_allclose(
            data["target_tcp_pose7"][:3, :3], [[0, 0, 0.195], [0.01, 0, 0.195], [0.02, 0, 0.195]]
        )
        np.testing.assert_allclose(
            data["target_tcp_pose7"], data["actual_tcp_pose7"], atol=1e-6
        )
        assert float(np.max(data["position_error_m"])) < 1e-6

    document = json.loads(prepared.json_path.read_text(encoding="utf-8"))
    assert document["format"] == "handumi_robot_ik"
    assert document["format_version"] == 1
    assert document["episode_index"] == 0
    assert document["frame_count"] == 4
    assert document["valid_frame_count"] == 4
    assert document["invalid_frame_count"] == 0
    assert document["npz_file"] == "episode_000000.npz"
    assert document["source"]["tool_offset_applications"] == 1
    assert document["robot"]["name"] == "rm75b-test"
    assert document["tool_calibration"]["matrix"][2][3] == pytest.approx(0.195)
    assert document["relative_mapping"]["max_reach_m"] is None
    assert document["ik"]["parameters"]["max_reach"] == 0.9
    assert document["validity"]["max_position_error_m"] == 0.01
    assert document["validity"]["max_orientation_error_deg"] == 5.0

    published = recorder.publish_episode()
    assert published.npz_path == record_paths(tmp_path, 0).npz
    assert published.json_path == record_paths(tmp_path, 0).json
    assert published.npz_path.is_file()
    assert published.json_path.is_file()
    assert not prepared.npz_path.exists()
    assert not prepared.json_path.exists()
    assert has_complete_record(tmp_path, 0)
    assert find_incomplete_records(tmp_path) == ()
    assert recorder.frame_count == 0
    assert recorder.prepared is None


def test_cancel_discards_cache_and_staged_files(tmp_path) -> None:
    """取消片段丢弃缓存和暂存文件，不生成最终记录。"""
    recorder = _make_recorder(tmp_path)
    recorder.begin_episode()
    for timestamp, pose in _source_frames(3):
        recorder.add_frame(timestamp, pose)
    prepared = recorder.prepare_episode(0)
    recorder.cancel_episode()
    assert recorder.frame_count == 0
    assert recorder.prepared is None
    assert not prepared.npz_path.exists()
    assert not prepared.json_path.exists()
    assert not has_complete_record(tmp_path, 0)
    assert find_incomplete_records(tmp_path) == ()
    with pytest.raises(RobotIkStateError):
        recorder.prepare_episode(0)
    with pytest.raises(RobotIkStateError):
        recorder.add_frame(0.0, IDENTITY_POSE)


def test_recorder_rejects_invalid_lifecycle_calls(tmp_path) -> None:
    """未开始、空片段、重复补算和重复开始都会明确报错。"""
    recorder = _make_recorder(tmp_path)
    with pytest.raises(RobotIkStateError):
        recorder.add_frame(0.0, IDENTITY_POSE)
    with pytest.raises(RobotIkStateError):
        recorder.prepare_episode(0)
    recorder.begin_episode()
    with pytest.raises(RobotIkStateError):
        recorder.begin_episode()
    with pytest.raises(RobotIkStateError):
        recorder.prepare_episode(0)
    recorder.add_frame(0.0, IDENTITY_POSE)
    recorder.prepare_episode(0)
    with pytest.raises(RobotIkStateError):
        recorder.prepare_episode(0)
    recorder.discard_prepared()


def test_recorder_records_hold_state_for_invalid_frames(tmp_path) -> None:
    """补算中无效帧保留上一有效关节角/实际位姿，但目标继续前进。"""
    frames = [
        (10.0, _pose(x=0.0)),
        (10.1, _pose(x=0.05)),
        (10.2, _pose(x=0.10)),
        (10.3, _pose(x=0.15)),
    ]
    recorder = _make_recorder(
        tmp_path, solver_factory=lambda: EchoSolver(hold_ik_at={1, 2, 3})
    )
    recorder.begin_episode()
    for timestamp, pose in frames:
        recorder.add_frame(timestamp, pose)
    prepared = recorder.prepare_episode(0)
    assert prepared.valid_count == 1
    assert prepared.invalid_count == 3
    assert prepared.failure_reasons == {FAILURE_POSITION_ERROR: 3}
    with np.load(prepared.npz_path) as data:
        np.testing.assert_array_equal(data["valid"], [True, False, False, False])
        np.testing.assert_allclose(
            data["q_rad"][1:],
            np.tile(data["q_rad"][0], (3, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            data["actual_tcp_pose7"][1:],
            np.tile(data["actual_tcp_pose7"][0], (3, 1)),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            data["target_tcp_pose7"][:, 0], [0.0, 0.05, 0.10, 0.15], atol=1e-6
        )
        assert list(data["failure_reason"])[1:] == [FAILURE_POSITION_ERROR] * 3


def test_recorder_records_tracking_lost_frames(tmp_path) -> None:
    """无跟踪帧仍占原始帧序号，并保留上一目标与状态。"""
    recorder = _make_recorder(tmp_path)
    recorder.begin_episode()
    recorder.add_frame(10.0, IDENTITY_POSE)
    recorder.add_frame(10.1, None, tracked=False)
    recorder.add_frame(10.2, _pose(x=0.02))
    prepared = recorder.prepare_episode(0)
    with np.load(prepared.npz_path) as data:
        np.testing.assert_array_equal(data["valid"], [True, False, True])
        assert list(data["failure_reason"]) == ["", FAILURE_TRACKING_LOST, ""]
        np.testing.assert_allclose(data["target_tcp_pose7"][1], data["target_tcp_pose7"][0])
        np.testing.assert_allclose(data["q_rad"][1], data["q_rad"][0])
        np.testing.assert_allclose(
            data["target_tcp_pose7"][2, :3], [0.02, 0.0, 0.195], atol=1e-6
        )


def test_publish_failure_keeps_staging_and_recovery_metadata(tmp_path, monkeypatch) -> None:
    """发布失败回滚最终文件，保留暂存结果和恢复元信息。"""
    recorder = _make_recorder(tmp_path)
    recorder.begin_episode()
    for timestamp, pose in _source_frames(2):
        recorder.add_frame(timestamp, pose)
    prepared = recorder.prepare_episode(0)
    marker = tmp_path / "original_episode.parquet"
    marker.write_bytes(b"original episode bytes")

    real_replace = os.replace

    def flaky_replace(source, destination):
        if Path(destination).suffix == ".json":
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(RobotIkPublishError) as info:
        recorder.publish_episode()
    error = info.value
    assert error.recovery["rollback"] == "restored"
    assert error.recovery["original_dataset_episode_saved"] is True
    assert error.recovery_path is not None and error.recovery_path.is_file()
    assert prepared.npz_path.is_file()
    assert prepared.json_path.is_file()
    assert not record_paths(tmp_path, 0).npz.exists()
    assert not record_paths(tmp_path, 0).json.exists()
    assert marker.read_bytes() == b"original episode bytes"
    recovery_doc = json.loads(error.recovery_path.read_text(encoding="utf-8"))
    assert recovery_doc["episode_index"] == 0
    assert recovery_doc["staged"]["npz"] == str(prepared.npz_path)
    kinds = {record.kind for record in find_incomplete_records(tmp_path)}
    assert kinds == {"staged", "recovery"}

    recorder.discard_prepared()
    assert not prepared.npz_path.exists()
    assert not prepared.json_path.exists()
    assert error.recovery_path.is_file()
    assert {record.kind for record in find_incomplete_records(tmp_path)} == {
        "recovery"
    }


def test_publish_refuses_to_overwrite_existing_record(tmp_path) -> None:
    """已有同集记录时拒绝覆盖，并保留原记录和恢复元信息。"""
    recorder = _make_recorder(tmp_path)
    recorder.begin_episode()
    for timestamp, pose in _source_frames(2):
        recorder.add_frame(timestamp, pose)
    recorder.prepare_episode(0)
    recorder.publish_episode()
    original_npz = record_paths(tmp_path, 0).npz.read_bytes()
    original_json = record_paths(tmp_path, 0).json.read_bytes()

    second = _make_recorder(tmp_path)
    second.begin_episode()
    for timestamp, pose in _source_frames(2, step=0.02):
        second.add_frame(timestamp, pose)
    prepared = second.prepare_episode(0)
    with pytest.raises(RobotIkPublishError) as info:
        second.publish_episode()
    assert info.value.recovery["rollback"] == "not_started"
    assert record_paths(tmp_path, 0).npz.read_bytes() == original_npz
    assert record_paths(tmp_path, 0).json.read_bytes() == original_json
    assert prepared.npz_path.is_file()
    assert info.value.recovery_path is not None
    assert info.value.recovery_path.is_file()


def test_existing_episode_record_is_not_rewritten_when_recording_another(tmp_path) -> None:
    """续录新片段只写新编号，不补算或改写历史 episode。"""
    first = _make_recorder(tmp_path)
    first.begin_episode()
    for timestamp, pose in _source_frames(2):
        first.add_frame(timestamp, pose)
    first.prepare_episode(0)
    first.publish_episode()
    episode_zero_npz = record_paths(tmp_path, 0).npz.read_bytes()
    episode_zero_json = record_paths(tmp_path, 0).json.read_bytes()

    second = _make_recorder(tmp_path)
    second.begin_episode()
    for timestamp, pose in _source_frames(3):
        second.add_frame(timestamp, pose)
    second.prepare_episode(1)
    second.publish_episode()

    assert record_paths(tmp_path, 0).npz.read_bytes() == episode_zero_npz
    assert record_paths(tmp_path, 0).json.read_bytes() == episode_zero_json
    document = json.loads(record_paths(tmp_path, 1).json.read_text(encoding="utf-8"))
    assert document["episode_index"] == 1
    assert document["frame_count"] == 3
    assert find_incomplete_records(tmp_path) == ()


def test_find_incomplete_records_reports_staged_and_orphan_files(tmp_path) -> None:
    """启动检查识别暂存、恢复元信息和残缺最终文件，忽略完整记录。"""
    record_dir = tmp_path / "robot_ik"
    staging = record_dir / ".staging"
    staging.mkdir(parents=True)
    (staging / "episode_000000-deadbeef.npz").write_bytes(b"staged")
    (staging / "recovery_episode_000000-deadbeef.json").write_text(
        "{}", encoding="utf-8"
    )
    (record_dir / "episode_000003.npz").write_bytes(b"orphan")
    (record_dir / "episode_000004.json").write_text("{}", encoding="utf-8")
    (record_dir / "episode_000005.npz").write_bytes(b"complete")
    (record_dir / "episode_000005.json").write_text("{}", encoding="utf-8")

    records = find_incomplete_records(tmp_path)
    kinds = {record.kind for record in records}
    assert kinds == {"staged", "recovery", "missing_json", "missing_npz"}
    paths = {record.path.name for record in records}
    assert "episode_000005.npz" not in paths
    assert "episode_000005.json" not in paths


# ---------------------------------------------------------------------------
# 与现有机器人运行时的集成
# ---------------------------------------------------------------------------


def test_from_runtime_uses_rm75b_home_and_tool_offset(rm75b_runtime) -> None:
    """现有 RM75B 运行时可直接构造控制器并校验 home 工具 TCP。"""
    runtime = rm75b_runtime
    controller = RelativeToolIk.from_runtime(
        runtime,
        tool_calibration=ToolCalibration.from_matrix(_tool_matrix()),
    )
    np.testing.assert_allclose(controller.home_q, runtime.home_q())
    tool = ToolCalibration.from_matrix(_tool_matrix())
    np.testing.assert_allclose(
        controller.home_tcp_pose7,
        pose_mul(controller.home_end_pose7, tool.tool_pose7),
        atol=1e-5,
    )
    assert controller.metadata()["robot"]["name"] == "rm75b"
    assert controller.metadata()["ik"]["parameters"]["max_reach"] == 0.9
    frame = controller.process(controller.home_tcp_pose7)
    assert frame.valid
    assert frame.position_error_m < 0.01
    assert frame.orientation_error_deg < 5.0


def test_real_rm75b_tracks_millimeter_steps_and_small_rotations(
    rm75b_runtime,
) -> None:
    """真实 RM75B 连续 1 mm 平移和小旋转均有效，关节分支保持连续。"""
    tool = ToolCalibration.from_matrix(
        _tool_matrix(), source="synthetic_rm75b_tool.yaml"
    )
    assert tool.matrix[2, 3] == pytest.approx(0.195)
    controller = RelativeToolIk.from_runtime(
        rm75b_runtime,
        tool_calibration=tool,
    )
    controller.warmup()
    metadata = controller.metadata()
    assert metadata["ik"]["parameters"]["max_joint_delta"] == pytest.approx(0.12)
    assert metadata["validity"]["max_position_error_m"] == pytest.approx(0.01)
    assert metadata["validity"]["max_orientation_error_deg"] == pytest.approx(5.0)
    home = controller.home_tcp_pose7.copy()
    sources = [home.copy()]
    for step in range(1, 6):
        pose = home.copy()
        pose[0] += 0.001 * step
        sources.append(pose)
    for degrees in (0.5, 1.0, 2.0):
        pose = home.copy()
        pose[0] += 0.005
        pose[3:7] = (
            Rotation.from_euler("x", degrees, degrees=True)
            * Rotation.from_quat(pose[3:7])
        ).as_quat().astype(np.float32)
        sources.append(pose)

    max_delta = rm75b_runtime.config.ik_weights.max_joint_delta
    assert max_delta is not None
    previous_q = controller.home_q
    frames = []
    for index, source in enumerate(sources):
        frame = controller.process(source)
        frames.append(frame)
        assert frame.valid, (index, frame.failure_reason)
        assert frame.position_error_m < 0.001
        assert frame.orientation_error_deg < 0.5
        np.testing.assert_allclose(
            pose_mul(frame.actual_end_pose7, tool.tool_pose7),
            frame.actual_tcp_pose7,
            atol=1e-5,
        )
        assert np.max(np.abs(frame.q - previous_q)) <= max_delta + 1e-5
        previous_q = frame.q

    np.testing.assert_allclose(frames[0].q, controller.home_q)
    np.testing.assert_allclose(frames[0].target_tcp_pose7, home)
    np.testing.assert_allclose(frames[-1].actual_tcp_pose7, sources[-1], atol=1e-4)


def test_real_rm75b_unreachable_freezes_and_recovers(rm75b_runtime) -> None:
    """真实 RM75B 不可达目标判无效并冻结，回到可达目标后恢复。"""
    tool = ToolCalibration.from_matrix(
        _tool_matrix(), source="synthetic_rm75b_tool.yaml"
    )
    controller = RelativeToolIk.from_runtime(
        rm75b_runtime,
        tool_calibration=tool,
    )
    controller.warmup()
    home = controller.home_tcp_pose7.copy()
    assert controller.process(home).valid
    near = home.copy()
    near[0] += 0.001
    near_frame = controller.process(near)
    assert near_frame.valid
    held_q = near_frame.q.copy()
    held_tcp = near_frame.actual_tcp_pose7.copy()

    unreachable = home.copy()
    unreachable[0] += 2.0
    failed = controller.process(unreachable)
    assert not failed.valid
    assert FAILURE_POSITION_ERROR in failed.failure_reasons
    np.testing.assert_allclose(failed.q, held_q)
    np.testing.assert_allclose(failed.actual_tcp_pose7, held_tcp)

    recovered = controller.process(home)
    assert recovered.valid
    np.testing.assert_allclose(recovered.actual_tcp_pose7, home, atol=1e-4)


def test_warmed_solver_can_be_shared_by_controllers(rm75b_runtime) -> None:
    """同一预热 solver 可被多个控制器共享，控制器状态仍相互独立。"""
    tool = ToolCalibration.from_matrix(
        _tool_matrix(), source="synthetic_rm75b_tool.yaml"
    )
    solver = rm75b_runtime.solver_cls(
        config=rm75b_runtime.config.ik_weights
    )
    first = RelativeToolIk.from_runtime(
        rm75b_runtime,
        tool_calibration=tool,
        solver=solver,
    )
    first.warmup()
    second = RelativeToolIk.from_runtime(
        rm75b_runtime,
        tool_calibration=tool,
        solver=solver,
    )
    assert second.frames_processed == 0
    assert not second.started
    home = second.home_tcp_pose7.copy()
    assert second.process(home).valid
    near = home.copy()
    near[0] += 0.001
    assert second.process(near).valid
    assert first.frames_processed == 0
    np.testing.assert_allclose(first.q, first.home_q)
    assert second.metadata()["ik"]["parameters"]["max_joint_delta"] == 0.12
