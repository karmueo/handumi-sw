"""工具 TCP 标定与单臂相对运动 IK 求解。

本模块只负责矩阵校验、相对 TCP 目标生成、IK/FK 以及有效性判断，不读取
采集设备、不依赖 Rerun 或数据集写入。实时预览和逐帧补算共用同一套实现，
因此两者的坐标与阈值行为保持一致。

求解器通过 :class:`ArmIkSolver` 协议注入；默认的
:meth:`RelativeToolIk.from_runtime` 使用现有 ``handumi.robots`` 中的机器人
运行时，测试则可以传入不依赖硬件的假求解器。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from handumi.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
)
from handumi.robots.utils import mat_to_pose7, pose_inv, pose_mul

TOOL_CALIBRATION_KEY = "T_end_tcp"

DEFAULT_POSITION_TOLERANCE_M = 0.01
DEFAULT_ORIENTATION_TOLERANCE_DEG = 5.0
JOINT_LIMIT_TOLERANCE_RAD = 1e-6

FAILURE_TRACKING_LOST = "tracking_lost"
FAILURE_NON_FINITE_SOURCE_POSE = "non_finite_source_pose"
FAILURE_INVALID_SOURCE_POSE = "invalid_source_pose"
FAILURE_NON_FINITE_TARGET_POSE = "non_finite_target_pose"
FAILURE_IK_SOLVER_ERROR = "ik_solver_error"
FAILURE_INVALID_JOINT_SHAPE = "invalid_joint_shape"
FAILURE_NON_FINITE_JOINT_ANGLES = "non_finite_joint_angles"
FAILURE_JOINT_LIMIT_VIOLATION = "joint_limit_violation"
FAILURE_FK_SOLVER_ERROR = "fk_solver_error"
FAILURE_NON_FINITE_FK_POSE = "non_finite_fk_pose"
FAILURE_POSITION_ERROR = "position_error"
FAILURE_ORIENTATION_ERROR = "orientation_error"


class ArmIkSolver(Protocol):
    """工具 IK 所需的最小求解器接口。

    方法与现有 ``BimanualKinematicsSolver`` 兼容：``ik`` 接收法兰目标，
    ``fk_pose7`` 返回左右末端位姿。这样既可以直接复用机器人运行时，也
    可以在测试中注入无硬件依赖的假求解器。
    """

    def ik(
        self,
        q_current: np.ndarray,
        *,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """从上一关节角求解一个或两个末端目标，返回关节角。"""
        ...

    def fk_pose7(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回左右末端的 ``[x, y, z, qx, qy, qz, qw]`` 位姿。"""
        ...


@dataclass(frozen=True)
class IkThresholds:
    """IK 结果的有效性阈值。

    ``max_position_error_m`` 为工具 TCP 位置误差上限（米），
    ``max_orientation_error_deg`` 为最短旋转角误差上限（度）。两项都必须
    是正的有限值。
    """

    max_position_error_m: float = DEFAULT_POSITION_TOLERANCE_M
    max_orientation_error_deg: float = DEFAULT_ORIENTATION_TOLERANCE_DEG

    def __post_init__(self) -> None:
        """校验并规范化位置/姿态误差阈值。"""
        position = float(self.max_position_error_m)
        orientation = float(self.max_orientation_error_deg)
        if not np.isfinite(position) or position <= 0.0:
            raise ValueError("max_position_error_m 必须是正的有限值")
        if not np.isfinite(orientation) or orientation <= 0.0:
            raise ValueError("max_orientation_error_deg 必须是正的有限值")
        object.__setattr__(self, "max_position_error_m", position)
        object.__setattr__(self, "max_orientation_error_deg", orientation)

    def to_metadata(self) -> dict[str, Any]:
        """返回可写入 JSON 的阈值快照。"""
        return {
            "max_position_error_m": self.max_position_error_m,
            "max_orientation_error_deg": self.max_orientation_error_deg,
            "joint_limits_checked": True,
            "joint_limit_tolerance_rad": JOINT_LIMIT_TOLERANCE_RAD,
        }


@dataclass(frozen=True)
class RelativeMappingConfig:
    """工具 TCP 相对运动映射参数。

    源位姿已经是采集流程输出的桌面坐标系工具 TCP，默认世界旋转为单位
    矩阵，与 Vive 回放的 local-relative 处理一致。首帧把源 TCP 的局部
    坐标系对齐到机器人 home 工具 TCP，之后把每一帧的源局部增量叠加到
    上一目标上。``max_reach_m`` 故意固定为 ``None``：不通过工作空间裁剪
    掩盖不可达目标。
    """

    translation_scale: float = 1.0
    source_world_to_robot_world: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        """校验平移缩放和世界旋转，并保存为运行时类型。"""
        scale = float(self.translation_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("translation_scale 必须是正的有限值")
        world_map = _validated_rotation(
            self.source_world_to_robot_world,
            name="source_world_to_robot_world",
        )
        object.__setattr__(self, "translation_scale", scale)
        object.__setattr__(self, "source_world_to_robot_world", world_map)

    def to_metadata(self) -> dict[str, Any]:
        """返回可写入 JSON 的相对映射快照。"""
        return {
            "mode": "local-relative",
            "compose_source": "commanded",
            "initial_alignment": "source_local_frame_to_home_tool_tcp",
            "translation_scale": self.translation_scale,
            "source_world_to_robot_world": (
                self.source_world_to_robot_world.tolist()
            ),
            "max_reach_m": None,
        }


@dataclass(frozen=True)
class ToolCalibration:
    """法兰坐标系中的工具 TCP 标定 ``T_end_tcp``。

    ``matrix`` 是 ``^Link7 T_tcp`` 齐次变换。``source`` 保存原始文件路径，
    ``sha256`` 保存文件内容摘要，``notes`` 保存 YAML 注释中的人工调整
    说明，便于附加记录保留工具标定的来源。
    """

    matrix: np.ndarray
    source: str = "memory"
    sha256: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验工具矩阵并规范化来源及注释字段。"""
        object.__setattr__(
            self,
            "matrix",
            _validated_transform(self.matrix, name=TOOL_CALIBRATION_KEY),
        )
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        source: str = "memory",
        notes: Sequence[str] = (),
    ) -> ToolCalibration:
        """从内存中的 4×4 矩阵构造标定。"""
        return cls(
            matrix=np.asarray(matrix, dtype=np.float64),
            source=source,
            sha256=None,
            notes=tuple(notes),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ToolCalibration:
        """读取带 ``T_end_tcp`` 键的 YAML 工具标定。"""
        calibration_path = Path(path)
        raw = calibration_path.read_bytes()
        text = raw.decode("utf-8")
        document = yaml.safe_load(text)
        if isinstance(document, Mapping):
            if TOOL_CALIBRATION_KEY not in document:
                raise ValueError(
                    f"{calibration_path} 缺少 {TOOL_CALIBRATION_KEY!r} 键"
                )
            value = document[TOOL_CALIBRATION_KEY]
        elif document is None:
            raise ValueError(f"{calibration_path} 为空，无法读取工具标定")
        else:
            value = document
        notes = tuple(
            line.strip().lstrip("#").strip()
            for line in text.splitlines()
            if line.strip().startswith("#") and line.strip().lstrip("#").strip()
        )
        return cls(
            matrix=value,
            source=str(calibration_path),
            sha256=hashlib.sha256(raw).hexdigest(),
            notes=notes,
        )

    @property
    def tool_pose7(self) -> np.ndarray:
        """工具 TCP 在法兰坐标系中的 ``[x, y, z, qx, qy, qz, qw]``。"""
        return mat_to_pose7(self.matrix)

    @property
    def inverse_pose7(self) -> np.ndarray:
        """``inverse(T_end_tcp)`` 的 pose7 形式。"""
        return pose_inv(self.tool_pose7)

    def to_metadata(self) -> dict[str, Any]:
        """返回可写入 JSON 的标定快照。"""
        return {
            "key": TOOL_CALIBRATION_KEY,
            "matrix": self.matrix.tolist(),
            "source": self.source,
            "sha256": self.sha256,
            "notes": list(self.notes),
        }


def load_tool_calibration(path: str | Path) -> ToolCalibration:
    """从 YAML 文件读取工具标定。"""
    return ToolCalibration.from_yaml(path)


@dataclass(frozen=True)
class ToolIkFrame:
    """一帧工具相对 IK 的结果。

    ``q`` 始终是这一段内的状态快照：有效帧为本次解，失败帧为上一有效
    关节角。``actual_*`` 与 ``position_error_m``/``orientation_error_deg``
    对应当前 ``q``，因此无效帧显示的是保持姿态相对当前目标的误差。
    ``attempted_*`` 保存本次求解的原始结果，仅用于诊断失败原因。
    """

    source_tcp_pose7: np.ndarray | None
    target_tcp_pose7: np.ndarray
    target_end_pose7: np.ndarray
    q: np.ndarray
    actual_tcp_pose7: np.ndarray
    actual_end_pose7: np.ndarray
    position_error_m: float
    orientation_error_deg: float
    valid: bool
    failure_reason: str = ""
    attempted_q: np.ndarray | None = None
    attempted_tcp_pose7: np.ndarray | None = None
    attempted_position_error_m: float = float("nan")
    attempted_orientation_error_deg: float = float("nan")
    error_detail: str | None = None

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        """把分号分隔的失败原因拆成稳定代码元组。"""
        if not self.failure_reason:
            return ()
        return tuple(
            reason for reason in self.failure_reason.split(";") if reason
        )


class RelativeToolIk:
    """单臂工具 TCP 的相对运动映射、IK 与有效性状态机。

    源位姿是采集流程输出的桌面坐标系工具 TCP。每段第一次有效调用建立
    新起点：目标固定在机器人 home 工具 TCP，并用现有回放的
    ``local_frame_adapter`` 对齐局部轴向。之后每帧把源局部增量叠加到上一
    目标上，转换为法兰目标后调用注入的求解器。

    失败帧不更新关节状态，但仍推进目标序列；恢复后继续使用同一段映射。
    这让预览能看到当前目标与保持姿态的差距，也让逐帧补算保留原始目标。
    """

    def __init__(
        self,
        *,
        solver: ArmIkSolver,
        home_q: np.ndarray,
        joint_names: Sequence[str],
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        tool_calibration: ToolCalibration,
        thresholds: IkThresholds | None = None,
        relative_mapping: RelativeMappingConfig | None = None,
        side: str = "right",
        robot_id: str = "unknown",
        ik_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """建立单臂工具 IK 状态；校验 home、关节限位和工具标定。"""
        if side not in ("left", "right"):
            raise ValueError("side 必须是 'left' 或 'right'")
        if not isinstance(tool_calibration, ToolCalibration):
            raise TypeError("tool_calibration 必须是 ToolCalibration")
        self._solver = solver
        self._side = side
        self._robot_id = str(robot_id)
        self._joint_names = tuple(str(name) for name in joint_names)
        self._home_q = np.asarray(home_q, dtype=np.float32).copy()
        joint_count = len(self._joint_names)
        if joint_count == 0:
            raise ValueError("joint_names 不能为空")
        if self._home_q.shape != (joint_count,):
            raise ValueError(
                f"home_q 长度为 {self._home_q.size}，期望 {joint_count}"
            )
        self._lower_limits = np.asarray(lower_limits, dtype=np.float32).copy()
        self._upper_limits = np.asarray(upper_limits, dtype=np.float32).copy()
        if self._lower_limits.shape != (joint_count,) or (
            self._upper_limits.shape != (joint_count,)
        ):
            raise ValueError("关节限位长度必须与 joint_names 一致")
        if not (
            np.isfinite(self._home_q).all()
            and np.isfinite(self._lower_limits).all()
            and np.isfinite(self._upper_limits).all()
        ):
            raise ValueError("home_q 与关节限位必须是有限值")
        if np.any(self._lower_limits > self._upper_limits):
            raise ValueError("关节下限不能大于上限")
        if np.any(self._home_q < self._lower_limits - JOINT_LIMIT_TOLERANCE_RAD) or (
            np.any(self._home_q > self._upper_limits + JOINT_LIMIT_TOLERANCE_RAD)
        ):
            raise ValueError("home_q 超出关节限位")

        self._tool_calibration = tool_calibration
        self._thresholds = thresholds or IkThresholds()
        self._relative_mapping = relative_mapping or RelativeMappingConfig()
        self._ik_parameters = _json_safe(ik_parameters or {})

        self._home_end_pose7 = self._fk(self._home_q)
        self._home_tcp_pose7 = pose_mul(
            self._home_end_pose7, self._tool_calibration.tool_pose7
        ).astype(np.float32)
        if not _finite_pose7(self._home_tcp_pose7):
            raise ValueError("home 工具 TCP 位姿不是有限值")

        self._last_valid_q = self._home_q.copy()
        self._last_valid_end_pose7 = self._home_end_pose7.copy()
        self._last_valid_tcp_pose7 = self._home_tcp_pose7.copy()
        self._previous_target_tcp_pose7 = self._home_tcp_pose7.copy()
        self._previous_source_tcp_pose7: np.ndarray | None = None
        self._anchor_source_tcp_pose7: np.ndarray | None = None
        self._adapter_rot: np.ndarray | None = None
        self._started = False
        self._frames_processed = 0
        self._last_frame: ToolIkFrame | None = None
        self._warmed = False

    @classmethod
    def from_runtime(
        cls,
        runtime,
        *,
        tool_calibration: ToolCalibration | str | Path,
        thresholds: IkThresholds | None = None,
        relative_mapping: RelativeMappingConfig | None = None,
        side: str = "right",
        ik_config=None,
        home_pose: str | None = None,
        solver: ArmIkSolver | None = None,
    ) -> RelativeToolIk:
        """用现有 ``RobotRuntime`` 构造控制器。

        ``runtime`` 需提供 ``solver_cls``、``arms``、``robot``、``joint_names``
        和 ``home_q``。求解器配置默认沿用机器人 YAML 的常规 IK 权重。

        传入已预热的 ``solver`` 可让实时预览与逐帧补算共享同一份 JIT
        编译结果；此时不能再同时提供 ``ik_config``。每个控制器仍有自己
        独立的 home/起点状态。
        """
        if side not in runtime.arms:
            raise ValueError(f"机器人 {runtime.name} 没有 {side!r} 臂")
        if isinstance(tool_calibration, ToolCalibration):
            calibration = tool_calibration
        else:
            calibration = load_tool_calibration(tool_calibration)
        if solver is not None and ik_config is not None:
            raise ValueError("solver 与 ik_config 只能提供一个")
        if solver is None:
            config = runtime.config.ik_weights if ik_config is None else ik_config
            solver = runtime.solver_cls(config=config)
        else:
            config = getattr(solver, "config", runtime.config.ik_weights)
        home_q = runtime.home_q(home_pose)
        parameters = asdict(config) if is_dataclass(config) else {}
        return cls(
            solver=solver,
            home_q=home_q,
            joint_names=runtime.joint_names,
            lower_limits=np.asarray(runtime.robot.joints.lower_limits),
            upper_limits=np.asarray(runtime.robot.joints.upper_limits),
            tool_calibration=calibration,
            thresholds=thresholds,
            relative_mapping=relative_mapping,
            side=side,
            robot_id=runtime.name,
            ik_parameters=parameters,
        )

    @property
    def home_q(self) -> np.ndarray:
        """home 关节角副本。"""
        return self._home_q.copy()

    @property
    def home_tcp_pose7(self) -> np.ndarray:
        """home 关节角对应的工具 TCP 位姿副本。"""
        return self._home_tcp_pose7.copy()

    @property
    def home_end_pose7(self) -> np.ndarray:
        """home 关节角对应的法兰位姿副本。"""
        return self._home_end_pose7.copy()

    @property
    def q(self) -> np.ndarray:
        """当前保持的关节角副本（失败时为上一有效解）。"""
        return self._last_valid_q.copy()

    @property
    def started(self) -> bool:
        """是否已经由首个有效 TCP 建立新起点。"""
        return self._started

    @property
    def frames_processed(self) -> int:
        """本段已处理的帧数（含失败帧）。"""
        return self._frames_processed

    @property
    def last_frame(self) -> ToolIkFrame | None:
        """最近一帧结果。"""
        return self._last_frame

    def reset(self) -> np.ndarray:
        """回到 home 并清除本段起点，返回 home 关节角副本。"""
        self._last_valid_q = self._home_q.copy()
        self._last_valid_end_pose7 = self._home_end_pose7.copy()
        self._last_valid_tcp_pose7 = self._home_tcp_pose7.copy()
        self._previous_target_tcp_pose7 = self._home_tcp_pose7.copy()
        self._previous_source_tcp_pose7 = None
        self._anchor_source_tcp_pose7 = None
        self._adapter_rot = None
        self._started = False
        self._frames_processed = 0
        self._last_frame = None
        return self._last_valid_q.copy()

    def warmup(self) -> None:
        """预编译一次 IK 路径，不改变任何片段状态。

        与首帧的 home 直通不同，这里真正调用一次求解器，让 JAX/JAXLS 完成
        编译。同一 ``solver`` 实例可以被后续实时预览和逐帧补算控制器共享，
        因此工厂只需在构造阶段预热一次。结果被丢弃，``q``/起点/帧计数均
        保持不变；重复调用不会重复求解。
        """
        if self._warmed:
            return
        self._ik(self._home_q, self._home_end_pose7)
        self._warmed = True

    def process(self, source_tcp_pose7: np.ndarray | None) -> ToolIkFrame:
        """处理一帧源工具 TCP，返回目标、保持状态和有效性。

        传入 ``None`` 表示跟踪丢失；传入非有限位姿会作为失败帧记录。两者
        都不更新起点或上一源位姿，恢复后仍沿用同一段映射。
        """
        self._frames_processed += 1
        reasons: list[str] = []
        detail: str | None = None
        source: np.ndarray | None = None
        was_started = self._started

        if source_tcp_pose7 is None:
            reasons.append(FAILURE_TRACKING_LOST)
            target = self._previous_target_tcp_pose7.copy()
        else:
            raw_source = np.asarray(source_tcp_pose7, dtype=np.float32)
            if raw_source.shape != (7,):
                raise ValueError(
                    "source_tcp_pose7 必须是长度为 7 的 "
                    "[x, y, z, qx, qy, qz, qw] 位姿"
                )
            source = raw_source.copy()
            if not _finite_pose7(source):
                reasons.append(FAILURE_NON_FINITE_SOURCE_POSE)
                target = self._previous_target_tcp_pose7.copy()
            else:
                normalized = _normalized_pose7(source)
                if normalized is None:
                    reasons.append(FAILURE_INVALID_SOURCE_POSE)
                    target = self._previous_target_tcp_pose7.copy()
                else:
                    source = normalized
                    target, reasons = self._target_for_source(source, reasons)

        target_end_pose7 = pose_mul(
            target, self._tool_calibration.inverse_pose7
        ).astype(np.float32)
        if not _finite_pose7(target_end_pose7):
            reasons.append(FAILURE_NON_FINITE_TARGET_POSE)
            target_end_pose7 = self._home_end_pose7.copy()

        attempted_q: np.ndarray | None = None
        attempted_end_pose7: np.ndarray | None = None
        attempted_tcp_pose7: np.ndarray | None = None
        attempted_position_error = float("nan")
        attempted_orientation_error = float("nan")

        if not reasons:
            # 首帧目标就是 home 工具 TCP。直接使用 home 关节角既符合
            # “机械臂回到 home”，也避免在腕部奇异点上做一次无意义的
            # 求解；后续帧仍使用注入求解器的常规 IK。
            first_home_frame = not was_started and np.allclose(
                target, self._home_tcp_pose7, atol=1e-9
            )
            if first_home_frame:
                attempted_q = self._home_q.copy()
            else:
                try:
                    attempted_q = self._ik(self._last_valid_q, target_end_pose7)
                except Exception as exc:  # noqa: BLE001 - 求解失败要转成状态
                    reasons.append(FAILURE_IK_SOLVER_ERROR)
                    detail = _exception_detail(exc)

        if attempted_q is not None:
            if attempted_q.shape != self._home_q.shape:
                reasons.append(FAILURE_INVALID_JOINT_SHAPE)
                detail = detail or (
                    f"求解结果长度为 {attempted_q.size}，"
                    f"期望 {self._home_q.size}"
                )
                attempted_q = None
            elif not np.isfinite(attempted_q).all():
                reasons.append(FAILURE_NON_FINITE_JOINT_ANGLES)
            else:
                outside = np.flatnonzero(
                    (attempted_q < self._lower_limits - JOINT_LIMIT_TOLERANCE_RAD)
                    | (attempted_q > self._upper_limits + JOINT_LIMIT_TOLERANCE_RAD)
                )
                if outside.size:
                    reasons.append(FAILURE_JOINT_LIMIT_VIOLATION)
                    detail = detail or f"越界关节索引 {outside.tolist()}"
                try:
                    attempted_end_pose7 = self._fk(attempted_q)
                except Exception as exc:  # noqa: BLE001 - FK 失败同样要保留状态
                    reasons.append(FAILURE_FK_SOLVER_ERROR)
                    detail = detail or _exception_detail(exc)
                if attempted_end_pose7 is not None:
                    if not _finite_pose7(attempted_end_pose7):
                        reasons.append(FAILURE_NON_FINITE_FK_POSE)
                    else:
                        attempted_tcp_pose7 = pose_mul(
                            attempted_end_pose7,
                            self._tool_calibration.tool_pose7,
                        ).astype(np.float32)
                        attempted_position_error = _position_error(
                            target, attempted_tcp_pose7
                        )
                        attempted_orientation_error = _rotation_error_deg(
                            target, attempted_tcp_pose7
                        )
                        if (
                            attempted_position_error
                            > self._thresholds.max_position_error_m
                        ):
                            reasons.append(FAILURE_POSITION_ERROR)
                        if (
                            attempted_orientation_error
                            > self._thresholds.max_orientation_error_deg
                        ):
                            reasons.append(FAILURE_ORIENTATION_ERROR)

        reasons = list(dict.fromkeys(reasons))
        valid = not reasons
        if valid:
            assert attempted_q is not None
            assert attempted_end_pose7 is not None
            assert attempted_tcp_pose7 is not None
            q = attempted_q.copy()
            actual_end_pose7 = attempted_end_pose7.copy()
            actual_tcp_pose7 = attempted_tcp_pose7.copy()
            position_error = attempted_position_error
            orientation_error = attempted_orientation_error
            self._last_valid_q = q.copy()
            self._last_valid_end_pose7 = actual_end_pose7.copy()
            self._last_valid_tcp_pose7 = actual_tcp_pose7.copy()
        else:
            q = self._last_valid_q.copy()
            actual_end_pose7 = self._last_valid_end_pose7.copy()
            actual_tcp_pose7 = self._last_valid_tcp_pose7.copy()
            position_error = _position_error(target, actual_tcp_pose7)
            orientation_error = _rotation_error_deg(target, actual_tcp_pose7)

        frame = ToolIkFrame(
            source_tcp_pose7=source,
            target_tcp_pose7=target.astype(np.float32),
            target_end_pose7=target_end_pose7.astype(np.float32),
            q=q.astype(np.float32),
            actual_tcp_pose7=actual_tcp_pose7.astype(np.float32),
            actual_end_pose7=actual_end_pose7.astype(np.float32),
            position_error_m=float(position_error),
            orientation_error_deg=float(orientation_error),
            valid=valid,
            failure_reason=";".join(reasons),
            attempted_q=(
                None if attempted_q is None else attempted_q.astype(np.float32)
            ),
            attempted_tcp_pose7=(
                None
                if attempted_tcp_pose7 is None
                else attempted_tcp_pose7.astype(np.float32)
            ),
            attempted_position_error_m=float(attempted_position_error),
            attempted_orientation_error_deg=float(attempted_orientation_error),
            error_detail=detail,
        )
        self._last_frame = frame
        return frame

    def _target_for_source(
        self,
        source: np.ndarray,
        reasons: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """生成当前工具 TCP 目标；非有限增量时保留上一目标。"""
        if not self._started:
            self._started = True
            self._anchor_source_tcp_pose7 = source.copy()
            self._adapter_rot = local_frame_adapter(
                source,
                self._home_tcp_pose7,
                source_world_to_robot_world=(
                    self._relative_mapping.source_world_to_robot_world
                ),
            ).astype(np.float32)
            self._previous_source_tcp_pose7 = source.copy()
            self._previous_target_tcp_pose7 = self._home_tcp_pose7.copy()
            return self._home_tcp_pose7.copy(), reasons

        assert self._previous_source_tcp_pose7 is not None
        assert self._adapter_rot is not None
        target = local_relative_robot_target_pose7(
            previous_source_pose7=self._previous_source_tcp_pose7,
            current_source_pose7=source,
            base_robot_pose7=self._previous_target_tcp_pose7,
            adapter_rot=self._adapter_rot,
            home_robot_pose7=self._home_tcp_pose7,
            translation_scale=self._relative_mapping.translation_scale,
            max_reach=None,
        )
        normalized = _normalized_pose7(target)
        if normalized is None:
            reasons.append(FAILURE_NON_FINITE_TARGET_POSE)
            return self._previous_target_tcp_pose7.copy(), reasons
        self._previous_source_tcp_pose7 = source.copy()
        self._previous_target_tcp_pose7 = normalized.copy()
        return normalized, reasons

    def _fk(self, q: np.ndarray) -> np.ndarray:
        """调用注入求解器的 FK，并取出本臂法兰位姿。"""
        left, right = self._solver.fk_pose7(np.asarray(q, dtype=np.float32))
        pose = right if self._side == "right" else left
        result = np.asarray(pose, dtype=np.float32).copy()
        if result.shape != (7,):
            raise ValueError("求解器 FK 必须返回长度为 7 的位姿")
        return result

    def _ik(self, q_seed: np.ndarray, target_end_pose7: np.ndarray) -> np.ndarray:
        """调用注入求解器的单臂 IK。"""
        pose = (
            np.asarray(target_end_pose7[:3], dtype=np.float32),
            np.asarray(target_end_pose7[3:7], dtype=np.float32),
        )
        if self._side == "right":
            result = self._solver.ik(q_seed, right_pose=pose)
        else:
            result = self._solver.ik(q_seed, left_pose=pose)
        return np.asarray(result, dtype=np.float32).copy()

    def metadata(self) -> dict[str, Any]:
        """返回本段工具标定、映射、IK 参数和阈值的 JSON 快照。"""
        return {
            "robot": {
                "name": self._robot_id,
                "side": self._side,
                "joint_names": list(self._joint_names),
                "home_q_rad": self._home_q.tolist(),
                "home_end_pose7": self._home_end_pose7.tolist(),
                "home_tcp_pose7": self._home_tcp_pose7.tolist(),
            },
            "tool_calibration": self._tool_calibration.to_metadata(),
            "relative_mapping": {
                **self._relative_mapping.to_metadata(),
                "anchor_source_tcp_pose7": (
                    None
                    if self._anchor_source_tcp_pose7 is None
                    else self._anchor_source_tcp_pose7.tolist()
                ),
                "adapter_rotation": (
                    None if self._adapter_rot is None else self._adapter_rot.tolist()
                ),
                "home_tcp_pose7": self._home_tcp_pose7.tolist(),
            },
            "ik": {
                "parameters": _json_safe(self._ik_parameters),
                "solver": type(self._solver).__name__,
            },
            "validity": self._thresholds.to_metadata(),
        }


def _validated_transform(matrix: Any, *, name: str) -> np.ndarray:
    """校验米制刚体齐次矩阵并返回 float64 副本。"""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} 必须是 4×4 齐次矩阵")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} 包含非有限值")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} 的齐次行无效")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} 的旋转必须属于 SO(3)")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} 的旋转行列式必须为 1")
    return value


def _validated_rotation(matrix: Any, *, name: str) -> np.ndarray:
    """校验 3×3 旋转矩阵并返回 float32 副本。"""
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"{name} 必须是 3×3 旋转矩阵")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} 包含非有限值")
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} 的正交性无效")
    if not np.isclose(np.linalg.det(value), 1.0, atol=1e-5):
        raise ValueError(f"{name} 的行列式必须为 1")
    return value.astype(np.float32)


def _finite_pose7(pose: np.ndarray) -> bool:
    """判断 pose7 是否全部为有限值。"""
    return bool(np.isfinite(np.asarray(pose, dtype=np.float64)).all())


def _normalized_pose7(pose: np.ndarray) -> np.ndarray | None:
    """归一化 pose7 四元数；四元数退化时返回 ``None``。"""
    normalized = np.asarray(pose, dtype=np.float32).copy()
    norm = float(np.linalg.norm(normalized[3:7]))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    normalized[3:7] /= norm
    return normalized


def _position_error(target: np.ndarray, actual: np.ndarray) -> float:
    """工具 TCP 位置误差（米）。"""
    return float(
        np.linalg.norm(
            np.asarray(target[:3], dtype=np.float64)
            - np.asarray(actual[:3], dtype=np.float64)
        )
    )


def _rotation_error_deg(target: np.ndarray, actual: np.ndarray) -> float:
    """两个 pose7 四元数之间的最短旋转角（度）。"""
    target_quat = np.asarray(target[3:7], dtype=np.float64)
    actual_quat = np.asarray(actual[3:7], dtype=np.float64)
    target_norm = float(np.linalg.norm(target_quat))
    actual_norm = float(np.linalg.norm(actual_quat))
    if target_norm < 1e-8 or actual_norm < 1e-8:
        return float("nan")
    target_quat = target_quat / target_norm
    actual_quat = actual_quat / actual_norm
    dot = abs(float(np.dot(target_quat, actual_quat)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def _exception_detail(exc: Exception) -> str:
    """把异常转成适合记录的一句话。"""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _json_safe(value: Any) -> Any:
    """递归转换为 JSON 可序列化的 Python 值。"""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ArmIkSolver",
    "FAILURE_FK_SOLVER_ERROR",
    "FAILURE_IK_SOLVER_ERROR",
    "FAILURE_INVALID_JOINT_SHAPE",
    "FAILURE_INVALID_SOURCE_POSE",
    "FAILURE_JOINT_LIMIT_VIOLATION",
    "FAILURE_NON_FINITE_FK_POSE",
    "FAILURE_NON_FINITE_JOINT_ANGLES",
    "FAILURE_NON_FINITE_SOURCE_POSE",
    "FAILURE_NON_FINITE_TARGET_POSE",
    "FAILURE_ORIENTATION_ERROR",
    "FAILURE_POSITION_ERROR",
    "FAILURE_TRACKING_LOST",
    "IkThresholds",
    "RelativeMappingConfig",
    "RelativeToolIk",
    "TOOL_CALIBRATION_KEY",
    "ToolCalibration",
    "ToolIkFrame",
    "load_tool_calibration",
]
