"""机器人 IK 逐帧附加记录：缓存、补算、暂存与发布。

本模块负责在录制期间缓存工具 TCP 与采集时间戳，在 episode 保存前从
home 开始按完整帧序列重新求解，并把结果写入独立目录 ``robot_ik/`` 下的
NPZ 与 JSON。原始 LeRobot 数据契约不变；附加记录先生成在
``robot_ik/.staging/``，只有原始 episode 保存成功后才发布。

失败帧保留上一有效关节角作为状态快照，但仍记录当前目标和误差。发布失败
时保留临时结果与 ``recovery_episode_*.json`` 恢复元信息，不把附加记录报告为完整
成功；旧 episode 的既有记录不会被改写。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from handumi.dataset.paths import portable_path
from handumi.robots.umi_ik import RelativeToolIk, ToolIkFrame

FORMAT_NAME = "handumi_robot_ik"
FORMAT_VERSION = 1
RECORD_DIR_NAME = "robot_ik"
STAGING_DIR_NAME = ".staging"
RECOVERY_PREFIX = "recovery_episode_"

NPZ_FIELDS: tuple[str, ...] = (
    "frame_index",
    "timestamp_s",
    "q_rad",
    "valid",
    "failure_reason",
    "target_tcp_pose7",
    "actual_tcp_pose7",
    "position_error_m",
    "orientation_error_deg",
)
NPZ_UNITS: dict[str, str] = {
    "frame_index": "原始采集帧顺序索引",
    "timestamp_s": "秒",
    "q_rad": "弧度",
    "valid": "bool",
    "failure_reason": "稳定代码，空字符串表示有效",
    "target_tcp_pose7": "米 + xyzw 四元数",
    "actual_tcp_pose7": "米 + xyzw 四元数",
    "position_error_m": "米",
    "orientation_error_deg": "度",
}

ControllerFactory = Callable[[], RelativeToolIk]


class RobotIkRecordError(RuntimeError):
    """附加记录生命周期错误基类。"""


class RobotIkStateError(RobotIkRecordError):
    """调用顺序不符合缓存/补算/发布状态机。"""


class RobotIkWriteError(RobotIkRecordError):
    """补算结果写入临时 NPZ/JSON 失败。"""


class RobotIkPublishError(RobotIkRecordError):
    """附加记录发布失败，附带可恢复的临时结果元信息。"""

    def __init__(
        self,
        message: str,
        *,
        episode_index: int,
        recovery: dict[str, Any],
        recovery_path: Path | None = None,
    ) -> None:
        """保存发布失败恢复元信息和恢复文件路径。"""
        super().__init__(message)
        self.episode_index = int(episode_index)
        self.recovery = recovery
        self.recovery_path = recovery_path


@dataclass(frozen=True)
class RobotIkRecordPaths:
    """一个 episode 的最终 NPZ/JSON 路径。"""

    episode_index: int
    npz: Path
    json: Path


@dataclass(frozen=True)
class PreparedRobotIk:
    """已写入暂存目录、等待发布的一集附加记录。"""

    episode_index: int
    npz_path: Path
    json_path: Path
    frame_count: int
    valid_count: int
    invalid_count: int
    failure_reasons: dict[str, int]
    prepared_at: str


@dataclass(frozen=True)
class PublishedRobotIk:
    """已发布的一集附加记录。"""

    episode_index: int
    npz_path: Path
    json_path: Path
    frame_count: int
    valid_count: int


@dataclass(frozen=True)
class IncompleteRobotIkRecord:
    """启动检查发现的遗留暂存或残缺发布记录。"""

    kind: str
    path: Path
    detail: str


def record_paths(root: str | Path, episode_index: int) -> RobotIkRecordPaths:
    """返回 ``<root>/robot_ik/episode_XXXXXX.{npz,json}`` 路径。"""
    index = _validated_episode_index(episode_index)
    directory = Path(root) / RECORD_DIR_NAME
    stem = f"episode_{index:06d}"
    return RobotIkRecordPaths(
        episode_index=index,
        npz=directory / f"{stem}.npz",
        json=directory / f"{stem}.json",
    )


def has_complete_record(root: str | Path, episode_index: int) -> bool:
    """判断某集是否已有完整的 NPZ 与 JSON。"""
    paths = record_paths(root, episode_index)
    return paths.npz.is_file() and paths.json.is_file()


def find_incomplete_records(
    root: str | Path,
) -> tuple[IncompleteRobotIkRecord, ...]:
    """检查遗留暂存文件、恢复元信息和残缺的最终文件。

    该函数只读，适合在开始新录制或续录前调用；它不会补算历史 episode，
    也不会修改任何文件。
    """
    record_dir = Path(root) / RECORD_DIR_NAME
    if not record_dir.is_dir():
        return ()
    records: list[IncompleteRobotIkRecord] = []
    staging = record_dir / STAGING_DIR_NAME
    if staging.is_dir():
        for path in sorted(staging.iterdir()):
            if not path.is_file():
                continue
            if path.name.startswith(RECOVERY_PREFIX):
                kind = "recovery"
                detail = "上一次发布失败留下的恢复元信息"
            else:
                kind = "staged"
                detail = "已补算但尚未发布的临时附加记录"
            records.append(
                IncompleteRobotIkRecord(kind=kind, path=path, detail=detail)
            )

    for path in sorted(record_dir.glob("episode_*.npz")):
        if not path.with_suffix(".json").is_file():
            records.append(
                IncompleteRobotIkRecord(
                    kind="missing_json",
                    path=path,
                    detail="最终 NPZ 缺少同名 JSON 元信息",
                )
            )
    for path in sorted(record_dir.glob("episode_*.json")):
        if not path.with_suffix(".npz").is_file():
            records.append(
                IncompleteRobotIkRecord(
                    kind="missing_npz",
                    path=path,
                    detail="最终 JSON 缺少同名 NPZ 帧数据",
                )
            )
    return tuple(sorted(records, key=lambda item: (str(item.path), item.kind)))


def warn_incomplete_records(
    root: str | Path,
    *,
    logger: logging.Logger | None = None,
) -> tuple[IncompleteRobotIkRecord, ...]:
    """检查并记录遗留附加记录，供录制启动时提示未完成状态。"""
    log = logger or logging.getLogger(__name__)
    records = find_incomplete_records(root)
    for record in records:
        log.warning(
            "检测到未完成的 robot_ik 附加记录 (%s): %s - %s",
            record.kind,
            record.path,
            record.detail,
        )
    return records


class RobotIkEpisodeRecorder:
    """一集机器人 IK 附加记录的缓存与发布状态机。

    典型集成顺序::

        recorder.begin_episode()
        ...  # 每帧调用 add_frame(timestamp_s, tcp_pose7, tracked=...)
        prepared = recorder.prepare_episode(episode_index, progress=...)
        dataset.save_episode()          # 先保存原始 episode
        recorder.publish_episode()      # 成功后再发布附加文件

    任一步取消或原始保存失败时调用 ``cancel_episode()``；发布失败时调用
    ``discard_prepared()`` 丢弃暂存，或根据异常中的恢复元信息手动处理。

    ``controller_factory`` 每次都必须返回一个全新的、从 home 开始的
    ``RelativeToolIk``；不要复用实时预览的实例，否则补算的 ``reset()``
    会打断预览状态。控制器可以共享同一个已 ``warmup()`` 的 solver
    （``from_runtime(..., solver=...)``），避免每个 episode 重复 JIT。
    """

    def __init__(
        self,
        root: str | Path,
        controller_factory: ControllerFactory,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """保存数据集根目录和每集新建控制器的工厂。"""
        if not callable(controller_factory):
            raise TypeError("controller_factory 必须可调用")
        self._root = Path(root)
        self._controller_factory = controller_factory
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._frames: list[tuple[float, np.ndarray | None]] = []
        self._prepared: PreparedRobotIk | None = None
        self._episode_open = False

    @property
    def root(self) -> Path:
        """数据集根目录。"""
        return self._root

    @property
    def frame_count(self) -> int:
        """当前缓存帧数。"""
        with self._lock:
            return len(self._frames)

    @property
    def prepared(self) -> PreparedRobotIk | None:
        """当前等待发布的暂存记录。"""
        with self._lock:
            return self._prepared

    @property
    def episode_open(self) -> bool:
        """是否处于可追加帧的活动片段。"""
        with self._lock:
            return self._episode_open

    def begin_episode(self) -> None:
        """开始缓存新片段；上一段的暂存记录必须已发布或丢弃。"""
        with self._lock:
            if self._prepared is not None:
                raise RobotIkStateError(
                    "上一段附加记录尚未发布或丢弃，不能开始新片段"
                )
            if self._episode_open:
                raise RobotIkStateError("当前片段尚未结束，不能重复开始")
            if self._frames:
                raise RobotIkStateError(
                    "上一片段缓存仍在，请先补算发布或取消"
                )
            self._frames.clear()
            self._episode_open = True

    def add_frame(
        self,
        timestamp_s: float,
        tcp_pose7: np.ndarray | None,
        *,
        tracked: bool = True,
    ) -> None:
        """缓存一帧采集时间戳与桌面坐标系工具 TCP。

        ``tracked=False`` 或 ``tcp_pose7=None`` 表示该帧没有有效跟踪；
        补算时会生成 ``tracking_lost`` 失败帧，仍占用原始帧序号。非有限
        位姿也会被缓存，由补算标记为 ``non_finite_source_pose``。
        """
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s 必须是有限值")
        with self._lock:
            if not self._episode_open:
                raise RobotIkStateError("尚未开始新片段，不能追加帧")
            if self._prepared is not None:
                raise RobotIkStateError("当前片段已经补算，不能继续追加帧")
            pose: np.ndarray | None
            if not tracked or tcp_pose7 is None:
                pose = None
            else:
                pose = np.asarray(tcp_pose7, dtype=np.float32)
                if pose.shape != (7,):
                    raise ValueError(
                        "tcp_pose7 必须是长度为 7 的 "
                        "[x, y, z, qx, qy, qz, qw] 位姿"
                    )
                pose = pose.copy()
            self._frames.append((timestamp, pose))

    def cancel_episode(self) -> None:
        """丢弃当前片段缓存和尚未发布的暂存文件。"""
        with self._lock:
            self._frames.clear()
            self._episode_open = False
            prepared = self._prepared
            self._prepared = None
        if prepared is not None:
            _remove_paths(prepared.npz_path, prepared.json_path)
        self._remove_empty_staging_dir()

    def discard_prepared(self) -> None:
        """丢弃当前暂存文件，保留已缓存帧以便重新补算。"""
        with self._lock:
            prepared = self._prepared
            self._prepared = None
        if prepared is not None:
            _remove_paths(prepared.npz_path, prepared.json_path)
        self._remove_empty_staging_dir()

    def release_prepared(self) -> None:
        """保留暂存/恢复文件，仅放弃内存引用以允许开始下一段。

        用于原始 episode 已保存、但附加记录发布失败的路径。暂存 NPZ/JSON
        和 recovery 元信息必须留给启动检查或人工恢复，不能在这里删除。
        """
        with self._lock:
            self._prepared = None
            self._frames.clear()
            self._episode_open = False

    def prepare_episode(
        self,
        episode_index: int,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> PreparedRobotIk:
        """从 home 按完整缓存序列重新求解，并写入暂存 NPZ/JSON。

        ``progress(done, total)`` 在每帧完成后调用，调用方可用它显示保存
        进度。原始 episode 保存成功前不会生成最终文件。
        """
        index = _validated_episode_index(episode_index)
        with self._lock:
            if self._prepared is not None:
                raise RobotIkStateError("当前片段已经补算，等待发布或丢弃")
            frames = list(self._frames)
            if not frames:
                raise RobotIkStateError("当前片段没有可补算的帧")
            # 关闭片段，防止补算过程中继续追加未被纳入的帧。
            self._episode_open = False

        controller = self._controller_factory()
        controller.reset()
        results: list[ToolIkFrame] = []
        total = len(frames)
        for done, (_, pose) in enumerate(frames, start=1):
            results.append(controller.process(pose))
            if progress is not None:
                progress(done, total)

        metadata = _portable_metadata(controller.metadata())
        document = _record_document(
            self._root,
            index,
            results=results,
            metadata=metadata,
        )
        staging, npz_path, json_path = _staging_paths(self._root, index)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            _write_npz(npz_path, frames=frames, results=results)
            _write_json(json_path, document)
        except Exception as exc:
            _remove_paths(npz_path, json_path)
            raise RobotIkWriteError(
                f"写入 episode {index} 的 robot_ik 临时文件失败: "
                f"{_exception_detail(exc)}"
            ) from exc
        except BaseException:
            # Ctrl+C 等中断也要清掉可能只写了一半的暂存文件。
            _remove_paths(npz_path, json_path)
            raise

        valid_count = sum(1 for frame in results if frame.valid)
        prepared = PreparedRobotIk(
            episode_index=index,
            npz_path=npz_path,
            json_path=json_path,
            frame_count=len(results),
            valid_count=valid_count,
            invalid_count=len(results) - valid_count,
            failure_reasons=_failure_reason_counts(results),
            prepared_at=_now_iso(),
        )
        with self._lock:
            if self._prepared is not None:
                _remove_paths(npz_path, json_path)
                raise RobotIkStateError("补算期间出现了另一份待发布记录")
            self._prepared = prepared
        return prepared

    def publish_episode(
        self,
        episode_index: int | None = None,
    ) -> PublishedRobotIk:
        """把暂存文件发布到 ``robot_ik/episode_XXXXXX.*``。

        调用方必须在原始 episode 保存成功后调用。发布失败时暂存文件与
        ``recovery_episode_*.json`` 会保留，异常中的 ``recovery`` 描述恢复路径。
        """
        with self._lock:
            prepared = self._prepared
        if prepared is None:
            raise RobotIkStateError("没有等待发布的附加记录")
        if episode_index is not None:
            requested = _validated_episode_index(episode_index)
            if requested != prepared.episode_index:
                raise RobotIkStateError(
                    f"待发布记录属于 episode {prepared.episode_index}，"
                    f"不是 {requested}"
                )

        destination = record_paths(self._root, prepared.episode_index)
        destination.npz.parent.mkdir(parents=True, exist_ok=True)
        missing = [
            path
            for path in (prepared.npz_path, prepared.json_path)
            if not path.is_file()
        ]
        if missing:
            recovery = self._recovery_document(
                prepared,
                destination,
                rollback="not_started",
                reason="暂存附加文件缺失，无法发布: "
                + ", ".join(str(path) for path in missing),
            )
            recovery_path, write_error = self._write_recovery(recovery)
            if recovery_path is not None:
                recovery["recovery_file"] = str(recovery_path)
            if write_error is not None:
                recovery["recovery_write_error"] = write_error
            raise RobotIkPublishError(
                f"暂存 robot_ik 记录不完整，拒绝发布: {missing}",
                episode_index=prepared.episode_index,
                recovery=recovery,
                recovery_path=recovery_path,
            )
        if destination.npz.exists() or destination.json.exists():
            recovery = self._recovery_document(
                prepared,
                destination,
                rollback="not_started",
                reason="目标附加记录已存在，拒绝覆盖旧记录",
            )
            recovery_path, write_error = self._write_recovery(recovery)
            if recovery_path is not None:
                recovery["recovery_file"] = str(recovery_path)
            if write_error is not None:
                recovery["recovery_write_error"] = write_error
            raise RobotIkPublishError(
                f"拒绝覆盖已存在的 robot_ik 记录: {destination.npz}",
                episode_index=prepared.episode_index,
                recovery=recovery,
                recovery_path=recovery_path,
            )

        published_npz: Path | None = None
        rollback = "not_started"
        rollback_error: str | None = None
        try:
            os.replace(prepared.npz_path, destination.npz)
            published_npz = destination.npz
            os.replace(prepared.json_path, destination.json)
        except Exception as exc:
            if published_npz is not None:
                try:
                    os.replace(destination.npz, prepared.npz_path)
                    rollback = "restored"
                except Exception as rollback_exc:  # noqa: BLE001 - 回滚失败要记录
                    rollback = "partial"
                    rollback_error = _exception_detail(rollback_exc)
            recovery = self._recovery_document(
                prepared,
                destination,
                rollback=rollback,
                reason=_exception_detail(exc),
                rollback_error=rollback_error,
            )
            recovery_path, write_error = self._write_recovery(recovery)
            if recovery_path is not None:
                recovery["recovery_file"] = str(recovery_path)
            if write_error is not None:
                recovery["recovery_write_error"] = write_error
            raise RobotIkPublishError(
                f"发布 episode {prepared.episode_index} 的 robot_ik 附加文件"
                f"失败；原始 episode 已保留，暂存结果仍可恢复: "
                f"{_exception_detail(exc)}",
                episode_index=prepared.episode_index,
                recovery=recovery,
                recovery_path=recovery_path,
            ) from exc

        with self._lock:
            if self._prepared is prepared:
                self._prepared = None
                self._frames.clear()
                self._episode_open = False
        self._remove_empty_staging_dir()
        self._logger.info(
            "Robot IK 附加记录已发布: %s, %s",
            destination.npz,
            destination.json,
        )
        return PublishedRobotIk(
            episode_index=prepared.episode_index,
            npz_path=destination.npz,
            json_path=destination.json,
            frame_count=prepared.frame_count,
            valid_count=prepared.valid_count,
        )

    def _recovery_document(
        self,
        prepared: PreparedRobotIk,
        destination: RobotIkRecordPaths,
        *,
        rollback: str,
        reason: str,
        rollback_error: str | None = None,
    ) -> dict[str, Any]:
        """构造发布失败恢复元信息。"""
        return {
            "kind": "robot_ik_publish_failure",
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "created_at": _now_iso(),
            "episode_index": prepared.episode_index,
            "frame_count": prepared.frame_count,
            "reason": reason,
            "staged": {
                "npz": str(prepared.npz_path),
                "json": str(prepared.json_path),
            },
            "destination": {
                "npz": str(destination.npz),
                "json": str(destination.json),
            },
            "rollback": rollback,
            "rollback_error": rollback_error,
            # publish_episode 的调用契约就是原始 episode 已保存成功。
            "original_dataset_episode_saved": True,
            "note": (
                "原始 episode 已保留；逐帧附加记录未完整发布。"
                "请根据 staged/destination 路径恢复，已有旧记录不会被改写。"
            ),
        }

    def _write_recovery(
        self,
        recovery: Mapping[str, Any],
    ) -> tuple[Path | None, str | None]:
        """把恢复元信息写入暂存目录，返回路径或写入错误。"""
        staging = self._root / RECORD_DIR_NAME / STAGING_DIR_NAME
        try:
            staging.mkdir(parents=True, exist_ok=True)
            path = staging / (
                f"{RECOVERY_PREFIX}{recovery['episode_index']:06d}-"
                f"{uuid.uuid4().hex[:8]}.json"
            )
            _write_json(path, recovery)
        except Exception as exc:  # noqa: BLE001 - 恢复信息写失败也要报告
            return None, _exception_detail(exc)
        return path, None

    def _remove_empty_staging_dir(self) -> None:
        """暂存目录为空时移除，避免留下无意义目录。"""
        staging = self._root / RECORD_DIR_NAME / STAGING_DIR_NAME
        try:
            staging.rmdir()
        except OSError:
            pass


def _failure_reason_counts(results: list[ToolIkFrame]) -> dict[str, int]:
    """按稳定失败代码统计帧数；一帧可能有多个原因。"""
    counts = Counter(
        reason for frame in results for reason in frame.failure_reasons
    )
    return dict(sorted(counts.items()))


def _record_document(
    root: Path,
    episode_index: int,
    *,
    results: list[ToolIkFrame],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """组装 episode JSON 元信息。"""
    destination = record_paths(root, episode_index)
    valid_count = sum(1 for frame in results if frame.valid)
    document: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "created_at": _now_iso(),
        "episode_index": episode_index,
        "frame_count": len(results),
        "valid_frame_count": valid_count,
        "invalid_frame_count": len(results) - valid_count,
        "failure_reason_counts": _failure_reason_counts(results),
        "frame_order": "acquisition",
        "joint_unit": "rad",
        "timestamp_unit": "s",
        "tcp_pose_format": "[x, y, z, qx, qy, qz, qw]",
        "npz_file": destination.npz.name,
        "npz_fields": list(NPZ_FIELDS),
        "npz_units": dict(NPZ_UNITS),
        "source": {
            "tcp_frame": "table",
            "pose_source": "recorded_umi_tcp",
            "timestamp_source": "acquisition",
            "tool_offset_applications": 1,
            "invalid_joint_state_policy": "hold_last_valid",
        },
    }
    document.update(metadata)
    return document


def _staging_paths(
    root: Path,
    episode_index: int,
) -> tuple[Path, Path, Path]:
    """生成唯一暂存目录与 NPZ/JSON 临时路径。"""
    staging = Path(root) / RECORD_DIR_NAME / STAGING_DIR_NAME
    token = uuid.uuid4().hex[:12]
    stem = f"episode_{episode_index:06d}-{token}"
    return (
        staging,
        staging / f"{stem}.npz",
        staging / f"{stem}.json",
    )


def _write_npz(
    path: Path,
    *,
    frames: list[tuple[float, np.ndarray | None]],
    results: list[ToolIkFrame],
) -> None:
    """按原始帧顺序写入 NPZ。"""
    arrays: dict[str, np.ndarray] = {
        "frame_index": np.arange(len(results), dtype=np.int64),
        "timestamp_s": np.asarray(
            [timestamp for timestamp, _ in frames], dtype=np.float64
        ),
        "q_rad": np.stack([frame.q for frame in results]).astype(np.float32),
        "valid": np.asarray([frame.valid for frame in results], dtype=bool),
        "failure_reason": np.asarray(
            [frame.failure_reason for frame in results], dtype=str
        ),
        "target_tcp_pose7": np.stack(
            [frame.target_tcp_pose7 for frame in results]
        ).astype(np.float32),
        "actual_tcp_pose7": np.stack(
            [frame.actual_tcp_pose7 for frame in results]
        ).astype(np.float32),
        "position_error_m": np.asarray(
            [frame.position_error_m for frame in results], dtype=np.float32
        ),
        "orientation_error_deg": np.asarray(
            [frame.orientation_error_deg for frame in results], dtype=np.float32
        ),
    }
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    """写入 UTF-8 JSON；拒绝 NaN/Infinity 以免产生不可移植元信息。"""
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    """把 NumPy/Path 值转换为 JSON 可序列化值。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化类型 {type(value).__name__}")


def _portable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """复制元信息并把工具标定来源改成可移植路径。"""
    snapshot = copy.deepcopy(dict(metadata))
    tool = snapshot.get("tool_calibration")
    if isinstance(tool, dict) and isinstance(tool.get("source"), str):
        tool["source"] = portable_path(tool["source"])
    return snapshot


def _validated_episode_index(episode_index: int) -> int:
    """校验 episode 编号是非负整数。"""
    index = int(episode_index)
    if index < 0 or index != episode_index:
        raise ValueError("episode_index 必须是非负整数")
    return index


def _remove_paths(*paths: Path) -> None:
    """删除存在的暂存文件。"""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _exception_detail(exc: Exception) -> str:
    """把异常转成适合记录的一句话。"""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _now_iso() -> str:
    """当前 UTC 时间（ISO 8601）。"""
    return datetime.now(UTC).isoformat()


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "IncompleteRobotIkRecord",
    "NPZ_FIELDS",
    "PreparedRobotIk",
    "PublishedRobotIk",
    "RECORD_DIR_NAME",
    "RobotIkEpisodeRecorder",
    "RobotIkPublishError",
    "RobotIkRecordError",
    "RobotIkRecordPaths",
    "RobotIkStateError",
    "RobotIkWriteError",
    "STAGING_DIR_NAME",
    "find_incomplete_records",
    "has_complete_record",
    "record_paths",
    "warn_incomplete_records",
]
