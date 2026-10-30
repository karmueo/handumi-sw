"""Rerun 机器人网格与连杆变换显示，不依赖采集设备。"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("handumi.visualize.robot")

TARGET_COLOR = (255, 90, 90)
ACTUAL_COLOR = (90, 170, 255)
MESH_COLOR = (150, 150, 150, 255)


class LogStream(Protocol):
    """Rerun ``RecordingStream`` 的最小可注入接口。"""

    def log(self, entity_path: str, entity: Any, *, static: bool = False) -> None:
        """记录一个 Rerun archetype。"""
        ...

    def set_time(self, timeline: str, **kwargs: Any) -> None:
        """设置当前线程的记录时间。"""
        ...

    def send_blueprint(self, blueprint: Any, **kwargs: Any) -> None:
        """发送视图布局。"""
        ...

    def spawn(self) -> None:
        """启动 Rerun 查看器。"""
        ...

    def flush(self) -> None:
        """刷新记录队列。"""
        ...

    def disconnect(self) -> None:
        """关闭记录流。"""
        ...


def validate_rerun_api() -> None:
    """在启动/删除输出目录前校验实际安装的 Rerun API。

    ``_prepare_robot_preview`` 在覆盖已有输出目录之前调用此函数，避免直到
    录制阶段才发现缺少 Rerun 或版本不兼容。
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError("机器人预览需要 rerun-sdk") from exc
    required_rr = (
        "RecordingStream",
        "Mesh3D",
        "Transform3D",
        "Points3D",
        "Scalars",
        "TextLog",
        "Quaternion",
        "ViewCoordinates",
    )
    required_rrb = (
        "Blueprint",
        "Spatial2DView",
        "Spatial3DView",
        "TimeSeriesView",
        "TextLogView",
        "Grid",
        "BlueprintPanel",
        "SelectionPanel",
        "TimePanel",
    )
    missing = [name for name in required_rr if not hasattr(rr, name)]
    missing += [name for name in required_rrb if not hasattr(rrb, name)]
    if not hasattr(rr.RecordingStream, "spawn"):
        missing.append("RecordingStream.spawn")
    if missing:
        raise RuntimeError(
            "当前 rerun-sdk 缺少机器人预览所需 API: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class RobotMesh:
    """一个已加载的 URDF 网格及其 Rerun 实体路径。"""

    key: str
    link_path: str
    mesh_path: str
    vertex_positions: np.ndarray
    triangle_indices: np.ndarray
    vertex_normals: np.ndarray | None


def _entity_name(value: str) -> str:
    """把网格文件名转换为稳定的 Rerun 实体名。"""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return name or "mesh"


def load_robot_meshes(runtime) -> tuple[Any, tuple[RobotMesh, ...]]:
    """加载完整 URDF 网格；任何缺失或空网格都明确报错。

    返回 ``(urdf, meshes)``。网格顶点保持连杆局部坐标，后续只通过父级
    ``Transform3D`` 更新位姿，避免把同一变换烘进网格两次。
    """
    urdf = runtime.load_urdf(load_meshes=True)
    if urdf.scene is None or not urdf.scene.geometry:
        raise ValueError("URDF 没有可用的可视化网格")
    meshes: list[RobotMesh] = []
    for key in sorted(urdf.scene.geometry):
        geometry = urdf.scene.geometry[key]
        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"网格 {key!r} 没有有效顶点")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"网格 {key!r} 没有有效三角面")
        normals = getattr(geometry, "vertex_normals", None)
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
            if normals.shape != vertices.shape or not np.isfinite(normals).all():
                normals = None
        name = _entity_name(key)
        meshes.append(
            RobotMesh(
                key=key,
                link_path=f"/robot/link/{name}",
                mesh_path=f"/robot/link/{name}/mesh",
                vertex_positions=vertices,
                triangle_indices=faces.astype(np.uint32),
                vertex_normals=normals,
            )
        )
    if not meshes:
        raise ValueError("URDF 没有可用的可视化网格")
    return urdf, tuple(meshes)


class RerunRobotView:
    """独立机器人场景：静态网格 + 逐帧连杆/TCP/误差显示。

    ``load()`` 只加载模型，不启动查看器，适合在删除已有输出目录前做完整
    网格预检。``start()`` 才创建/启动 Rerun 记录流。``log_frame()`` 使用
    求解样本的时间戳，而不是当前墙钟时间。
    """

    def __init__(
        self,
        runtime,
        *,
        application_id: str = "handumi_umi_robot",
        stream: LogStream | None = None,
        axis_length: float = 0.08,
        timeline: str = "capture",
    ) -> None:
        """保存机器人运行时、显示流和轴长配置；网格延迟到 load/start。"""
        self.runtime = runtime
        self.application_id = str(application_id)
        self.timeline = str(timeline)
        self.axis_length = float(axis_length)
        if not np.isfinite(self.axis_length) or self.axis_length <= 0:
            raise ValueError("axis_length 必须是正的有限值")
        self._stream = stream
        self._owns_stream = stream is None
        self._urdf = None
        self._meshes: tuple[RobotMesh, ...] = ()
        self._joint_names: tuple[str, ...] = ()
        self._joint_indices: dict[str, int] = {}
        self._rr = None
        self._send_blueprint = True
        self._started = False
        self._closed = False

    @property
    def loaded(self) -> bool:
        """是否已加载网格。"""
        return self._urdf is not None

    @property
    def started(self) -> bool:
        """是否已启动 Rerun 记录流。"""
        return self._started

    @property
    def healthy(self) -> bool:
        """记录流是否仍可用。"""
        return self._started and not self._closed

    @property
    def mesh_count(self) -> int:
        """已加载网格数量。"""
        return len(self._meshes)

    def load(self) -> RerunRobotView:
        """加载完整 URDF 网格和关节名；重复调用幂等。"""
        if self._urdf is not None:
            return self
        self._urdf, self._meshes = load_robot_meshes(self.runtime)
        self._joint_names = tuple(self._urdf.actuated_joint_names)
        self._joint_indices = {
            name: index for index, name in enumerate(self._joint_names)
        }
        return self

    def bind_stream(self, stream: LogStream, *, send_blueprint: bool = False) -> None:
        """绑定已有的显式记录流，避免启动第二个 Rerun 应用窗口。"""
        if self._started:
            raise RuntimeError("RerunRobotView 已启动，不能再绑定记录流")
        self._stream = stream
        self._owns_stream = False
        self._send_blueprint = bool(send_blueprint)

    def start(self) -> None:
        """静态记录网格并启动显示；已有共享流时不重复创建应用。"""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("RerunRobotView 已关闭")
        self.load()
        validate_rerun_api()
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        if self._stream is None:
            stream = rr.RecordingStream(self.application_id)
            stream.spawn()
            self._stream = stream
        stream = self._stream
        if self._send_blueprint:
            stream.send_blueprint(_robot_blueprint(rrb), make_active=True)
        stream.log(
            "/robot",
            rr.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
        )
        for mesh in self._meshes:
            stream.log(
                mesh.mesh_path,
                rr.Mesh3D(
                    vertex_positions=mesh.vertex_positions,
                    triangle_indices=mesh.triangle_indices,
                    vertex_normals=mesh.vertex_normals,
                    albedo_factor=MESH_COLOR,
                ),
                static=True,
            )
        # 等待手动开启 IK 时，也必须提供完整连杆位姿，避免局部网格重叠。
        # 使用时间数据，使后续实时变换可以覆盖初始姿态。
        # 将运行时关节顺序映射到 URDF 的关节顺序。
        home_by_name = dict(zip(self.runtime.joint_names, self.runtime.home_q()))
        self._urdf.update_cfg([float(home_by_name[name]) for name in self._joint_names])
        stream.set_time(self.timeline, duration=time.monotonic_ns() / 1e9)
        for mesh in self._meshes:
            transform = np.asarray(self._urdf.scene.graph.get(mesh.key)[0])
            stream.log(
                mesh.link_path,
                rr.Transform3D(
                    translation=transform[:3, 3],
                    mat3x3=transform[:3, :3],
                ),
            )
        self._started = True

    def log_frame(
        self,
        *,
        sample_time_ns: int,
        q: np.ndarray,
        target_tcp_pose7: np.ndarray,
        actual_tcp_pose7: np.ndarray,
        valid: bool,
        position_error_m: float,
        orientation_error_deg: float,
        failure_reason: str = "",
    ) -> None:
        """在求解样本时间记录连杆、目标/实际 TCP 与状态。"""
        if not self._started or self._stream is None or self._urdf is None:
            raise RuntimeError("RerunRobotView 尚未启动")
        rr = self._rr
        stream = self._stream
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (len(self._joint_names),):
            raise ValueError(
                f"q 长度为 {q.size}，期望 {len(self._joint_names)}"
            )
        config = [float(q[self._joint_indices[name]]) for name in self._joint_names]
        self._urdf.update_cfg(config)
        # 与相机/tracking 使用同一时间轴和 duration 类型。Rerun 的
        # set_time 是线程本地的，因此后台机器人结果不会污染相机时间。
        stream.set_time(
            self.timeline,
            duration=int(sample_time_ns) / 1e9,
        )
        for mesh in self._meshes:
            transform = np.asarray(self._urdf.scene.graph.get(mesh.key)[0])
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError(f"网格 {mesh.key!r} 的连杆变换无效")
            stream.log(
                mesh.link_path,
                rr.Transform3D(
                    translation=transform[:3, 3],
                    mat3x3=transform[:3, :3],
                ),
            )
        _log_tcp(stream, rr, "/robot/target_tcp", target_tcp_pose7, TARGET_COLOR, self.axis_length)
        _log_tcp(stream, rr, "/robot/actual_tcp", actual_tcp_pose7, ACTUAL_COLOR, self.axis_length)
        stream.log("/robot/status/position_error_mm", rr.Scalars(float(position_error_m) * 1000.0))
        stream.log("/robot/status/orientation_error_deg", rr.Scalars(float(orientation_error_deg)))
        stream.log("/robot/status/valid", rr.Scalars(float(bool(valid))))
        state = "OK" if valid else (failure_reason or "INVALID")
        stream.log(
            "/robot/status_text",
            rr.TextLog(
                f"{state} | pos={position_error_m * 1000.0:.2f} mm | "
                f"rot={orientation_error_deg:.2f} deg"
            ),
        )

    def close(self) -> None:
        """刷新并关闭记录流；由本对象创建的流才会 disconnect。"""
        if self._closed:
            return
        self._closed = True
        stream = self._stream
        if stream is not None:
            try:
                stream.flush()
            except Exception:
                log.debug("Rerun robot flush failed", exc_info=True)
            if self._owns_stream:
                try:
                    stream.disconnect()
                except Exception:
                    log.debug("Rerun robot disconnect failed", exc_info=True)
        self._started = False


def capture_blueprint(rrb: Any) -> Any:
    """采集窗口布局：显式展开左侧栏，保留相机/tracking/机器人及状态曲线。

    ``BlueprintPanel(expanded=True)`` 明确恢复旧采集窗口的左侧栏；其他
    面板不覆盖，使用 Rerun 默认状态。夹爪开合度和采集健康曲线作为显式
    视图，避免只能依赖侧栏树手动查找。
    """
    return rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(origin="/camera/right_wrist", name="Camera"),
            rrb.Spatial3DView(origin="/tracking", name="Raw tracking"),
            rrb.Spatial3DView(origin="/robot", name="RM75B preview"),
            rrb.TimeSeriesView(origin="/gripper", name="Gripper openness"),
            rrb.TimeSeriesView(origin="/health", name="Capture health"),
            rrb.TimeSeriesView(origin="/robot/status", name="IK status"),
            rrb.TextLogView(origin="/robot/status_text", name="Robot state"),
        ),
        rrb.BlueprintPanel(expanded=True),
    )


def _robot_blueprint(rrb: Any) -> Any:
    """独立机器人视图布局：3D 模型、IK 时间序列和状态文本。"""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/robot", name="RM75B preview"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/robot/status", name="IK status"),
                rrb.TextLogView(origin="/robot/status_text", name="State"),
            ),
        ),
    )


def _log_tcp(stream: LogStream, rr: Any, path: str, pose7: np.ndarray, color, axis_length: float) -> None:
    """在指定实体记录 TCP 坐标轴和彩色标记。"""
    pose = np.asarray(pose7, dtype=np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{path} 必须是有限 pose7")
    stream.log(
        path,
        rr.Transform3D(
            translation=pose[:3],
            rotation=rr.Quaternion(xyzw=pose[3:7]),
            axis_length=axis_length,
        ),
    )
    stream.log(
        f"{path}/marker",
        rr.Points3D(
            positions=np.zeros((1, 3), dtype=np.float32),
            colors=[color],
            radii=0.012,
        ),
    )


__all__ = [
    "ACTUAL_COLOR",
    "RobotMesh",
    "RerunRobotView",
    "TARGET_COLOR",
    "capture_blueprint",
    "load_robot_meshes",
    "validate_rerun_api",
]
