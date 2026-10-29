"""Join separately recorded sessions into one dataset, without reinterpreting them.

A merge is the one derivative operation that adds no judgement: every episode of
every source survives, in source order, with its frames, timestamps and video
untouched. What changes is bookkeeping -- episode and global frame indices are
renumbered, and the task table is unified -- so the provenance report records
which source each output episode came from, and the source datasets are left
exactly as they were found.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from handumi.dataset.analysis import dataset_payload_manifest
from handumi.dataset.curation import (
    STANDARD_INFO_KEYS,
    dataset_license,
    load_dataset_info,
    validate_dataset_integrity,
)
from handumi.dataset.paths import portable_path

MERGE_SCHEMA_VERSION = 1
MERGE_KIND = "handumi_dataset_merge"

# Compared across sources before anything is written. fps, robot_type and
# features are also checked by LeRobot's aggregation, but failing here names the
# offending dataset and costs nothing, where failing there happens after the
# videos have already been concatenated.
_COMPATIBILITY_KEYS = ("codebase_version", "robot_type", "fps", "chunks_size")


@dataclass(frozen=True)
class MergeSource:
    """One dataset entering a merge, as it was found on disk."""

    root: Path
    repo_id: str
    total_episodes: int
    total_frames: int
    tasks: tuple[str, ...]
    first_output_episode_index: int

    @property
    def output_episode_indices(self) -> range:
        return range(
            self.first_output_episode_index,
            self.first_output_episode_index + self.total_episodes,
        )


@dataclass(frozen=True)
class DatasetMergePlan:
    sources: tuple[MergeSource, ...]
    output_root: Path
    output_repo_id: str
    # None keeps every source's own wording, which yields one task per distinct
    # string. A merge of sessions that recorded the same task under different
    # wordings needs the unified string, or consumers see two tasks.
    task: str | None

    @property
    def total_episodes(self) -> int:
        return sum(source.total_episodes for source in self.sources)

    @property
    def total_frames(self) -> int:
        return sum(source.total_frames for source in self.sources)

    @property
    def source_tasks(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for source in self.sources:
            for task in source.tasks:
                seen.setdefault(task, None)
        return tuple(seen)


@dataclass(frozen=True)
class DatasetMergeResult:
    root: Path
    repo_id: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    report_path: Path
    validation: dict[str, Any]


def plan_dataset_merge(
    source_roots: list[str | Path] | tuple[str | Path, ...],
    *,
    output_root: str | Path,
    source_repo_ids: list[str] | tuple[str, ...] | None = None,
    output_repo_id: str | None = None,
    task: str | None = None,
) -> DatasetMergePlan:
    """Check that the sources can be joined, and fix the order they join in.

    Nothing is read beyond metadata and nothing is written, so a plan is safe to
    print and inspect before committing to the merge.
    """
    roots = [Path(root).resolve() for root in source_roots]
    if len(roots) < 2:
        raise ValueError("A merge needs at least two source datasets")
    duplicates = sorted({str(root) for root in roots if roots.count(root) > 1})
    if duplicates:
        raise ValueError(f"Source datasets are repeated: {duplicates}")

    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Source dataset does not exist: {root}")
        if output == root or output.is_relative_to(root):
            raise ValueError("Output must be a separate path outside every source")

    if source_repo_ids is not None and len(source_repo_ids) != len(roots):
        raise ValueError("Give one repo id per source dataset, or none at all")

    sources: list[MergeSource] = []
    reference: dict[str, Any] | None = None
    reference_root: Path | None = None
    next_index = 0
    for position, root in enumerate(roots):
        info = load_dataset_info(root)
        if reference is None:
            reference, reference_root = info, root
        else:
            _require_compatible(reference, info, reference_root, root)
        total_episodes = int(info.get("total_episodes", 0))
        total_frames = int(info.get("total_frames", 0))
        if total_episodes <= 0 or total_frames <= 0:
            raise ValueError(f"Source dataset has no episodes: {root}")
        repo_id = (
            source_repo_ids[position]
            if source_repo_ids is not None
            else f"local/{root.name}"
        )
        sources.append(
            MergeSource(
                root=root,
                repo_id=repo_id,
                total_episodes=total_episodes,
                total_frames=total_frames,
                tasks=_source_tasks(root),
                first_output_episode_index=next_index,
            )
        )
        next_index += total_episodes

    if task is not None and not task.strip():
        raise ValueError("Unified task must not be blank")
    return DatasetMergePlan(
        sources=tuple(sources),
        output_root=output,
        output_repo_id=output_repo_id or f"local/{output.name}",
        task=task,
    )


def merge_dataset(plan: DatasetMergePlan) -> DatasetMergeResult:
    """Execute a merge plan locally and atomically make its output visible."""
    if plan.output_root.exists():
        raise ValueError(f"Output path already exists: {plan.output_root}")
    plan.output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.output_root.name}.merging-",
            dir=plan.output_root.parent,
        )
    )
    build_root = temp_parent / "dataset"
    manifests_before = {
        source.repo_id: dataset_payload_manifest(source.root) for source in plan.sources
    }

    try:
        result = _build_merged_dataset(plan, build_root, manifests_before)
        for source in plan.sources:
            if dataset_payload_manifest(source.root) != manifests_before[source.repo_id]:
                raise RuntimeError(
                    f"Source dataset changed while merging: {source.root}"
                )
        if plan.output_root.exists():
            raise RuntimeError(f"Output path appeared during merge: {plan.output_root}")
        build_root.replace(plan.output_root)
        return DatasetMergeResult(
            root=plan.output_root,
            repo_id=plan.output_repo_id,
            total_episodes=result["total_episodes"],
            total_frames=result["total_frames"],
            total_tasks=result["total_tasks"],
            report_path=plan.output_root / "meta" / "handumi_merge.json",
            validation=result["validation"],
        )
    finally:
        if temp_parent.exists():
            shutil.rmtree(temp_parent)


def _build_merged_dataset(
    plan: DatasetMergePlan,
    build_root: Path,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from lerobot.datasets.dataset_tools import merge_datasets, modify_tasks
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import create_lerobot_dataset_card

    datasets = [
        LeRobotDataset(repo_id=source.repo_id, root=source.root, download_videos=True)
        for source in plan.sources
    ]
    merged = merge_datasets(datasets, plan.output_repo_id, output_dir=build_root)
    if plan.task is not None:
        modify_tasks(merged, new_task=plan.task)
        # modify_tasks rewrites task_index without touching stats.json, so the
        # column's statistics would still describe the pre-merge task table.
        _rewrite_constant_task_index_stats(build_root)

    merged_episodes = _episode_source_map(plan)
    output_info = load_dataset_info(build_root)
    output_info, dropped = _carry_source_info(plan, output_info)
    handumi = output_info.get("handumi")
    if not isinstance(handumi, dict):
        handumi = {}
        output_info["handumi"] = handumi
    handumi["merge"] = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "sources": [
            {
                "repo_id": source.repo_id,
                "total_episodes": source.total_episodes,
                "total_frames": source.total_frames,
                "output_episode_range": [
                    source.output_episode_indices.start,
                    source.output_episode_indices.stop,
                ],
            }
            for source in plan.sources
        ],
        "unified_task": plan.task,
        "source_tasks": list(plan.source_tasks),
    }
    (build_root / "meta" / "info.json").write_text(
        json.dumps(output_info, indent=4) + "\n",
        encoding="utf-8",
    )

    validation = validate_dataset_integrity(
        build_root,
        repo_id=plan.output_repo_id,
        load_with_lerobot=True,
    )
    report = {
        "schema_version": MERGE_SCHEMA_VERSION,
        "kind": MERGE_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                "repo_id": source.repo_id,
                "root": portable_path(source.root),
                "total_episodes": source.total_episodes,
                "total_frames": source.total_frames,
                "tasks": list(source.tasks),
                "output_episode_range": [
                    source.output_episode_indices.start,
                    source.output_episode_indices.stop,
                ],
                "payload_manifest": manifests[source.repo_id],
            }
            for source in plan.sources
        ],
        "output": {
            "repo_id": plan.output_repo_id,
            "root": portable_path(plan.output_root),
            "total_episodes": validation["total_episodes"],
            "total_frames": validation["total_frames"],
            "total_tasks": validation["total_tasks"],
        },
        "unified_task": plan.task,
        "source_tasks": list(plan.source_tasks),
        "dropped_info_keys": dropped,
        "episode_source": merged_episodes,
        "validation": validation,
        "output_payload_manifest": dataset_payload_manifest(build_root),
    }
    report_path = build_root / "meta" / "handumi_merge.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task_names = pq.read_table(build_root / "meta" / "tasks.parquet")["task"].to_pylist()
    sources_text = ", ".join(source.repo_id for source in plan.sources)
    card = create_lerobot_dataset_card(
        tags=["HandUMI", "merged"],
        dataset_info=output_info,
        license=dataset_license(plan.sources[0].root),
        repo_id=plan.output_repo_id,
        dataset_description=(
            f"HandUMI dataset merged locally from {len(plan.sources)} recording "
            f"sessions: {sources_text}. Tasks: "
            f"{', '.join(str(task) for task in task_names)}."
        ),
        url="https://github.com/murobotics-ai/handumi-sw",
    )
    card.save(build_root / "README.md")
    validate_dataset_integrity(
        build_root,
        repo_id=plan.output_repo_id,
        load_with_lerobot=True,
    )
    return {
        "total_episodes": validation["total_episodes"],
        "total_frames": validation["total_frames"],
        "total_tasks": validation["total_tasks"],
        "validation": validation,
    }


def _require_compatible(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_root: Path | None,
    candidate_root: Path,
) -> None:
    from handumi.config import dataset_active_sides

    if dataset_active_sides(reference.get("handumi", {})) != dataset_active_sides(candidate.get("handumi", {})):
        raise ValueError("Cannot merge recordings with different handumi.active_sides.")
    left_meta, right_meta = reference.get("handumi", {}), candidate.get("handumi", {})
    if "synchronized_sources_v2" in (left_meta.get("capture_schema"), right_meta.get("capture_schema")):
        for key in ("capture_schema", "sources", "umi_calibration"):
            if left_meta.get(key) != right_meta.get(key):
                raise ValueError(f"Cannot merge incompatible UMI {key}")
    for key in _COMPATIBILITY_KEYS:
        if reference.get(key) != candidate.get(key):
            raise ValueError(
                f"{candidate_root.name} has {key}={candidate.get(key)!r} where "
                f"{reference_root.name if reference_root else 'the first source'} "
                f"has {reference.get(key)!r}; merging them would mix incompatible "
                "recordings"
            )
    reference_features = reference.get("features", {})
    candidate_features = candidate.get("features", {})
    missing = sorted(set(reference_features) - set(candidate_features))
    extra = sorted(set(candidate_features) - set(reference_features))
    if missing or extra:
        raise ValueError(
            f"{candidate_root.name} does not have the same features: "
            f"missing {missing}, unexpected {extra}"
        )
    differing = sorted(
        key
        for key, feature in reference_features.items()
        if not _same_feature(feature, candidate_features[key])
    )
    if differing:
        raise ValueError(
            f"{candidate_root.name} declares different shape, dtype or video "
            f"encoding for: {differing}"
        )


def _same_feature(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Compare what a merge cannot reconcile, ignoring incidental ordering."""
    if reference.get("dtype") != candidate.get("dtype"):
        return False
    if list(reference.get("shape") or ()) != list(candidate.get("shape") or ()):
        return False
    if list(reference.get("names") or ()) != list(candidate.get("names") or ()):
        return False
    reference_info = reference.get("info") or {}
    candidate_info = candidate.get("info") or {}
    return all(
        reference_info.get(key) == candidate_info.get(key)
        for key in (
            "video.height",
            "video.width",
            "video.codec",
            "video.pix_fmt",
            "video.fps",
            "video.channels",
        )
    )


def _source_tasks(root: Path) -> tuple[str, ...]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise ValueError(f"Source dataset is missing meta/tasks.parquet: {root}")
    table = pq.read_table(path)
    column = "task" if "task" in table.column_names else table.column_names[0]
    return tuple(str(value) for value in table[column].to_pylist())


def _carry_source_info(
    plan: DatasetMergePlan,
    output_info: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Carry non-standard info keys forward only where every source agrees.

    ``handumi`` metadata such as the recording rig or a workspace pose belongs to
    the session that produced it. Copying one session's over a merge of three
    would attribute it to episodes it never described, so anything the sources
    state differently is dropped and named in the report instead.

    The comparison descends one level, into the entries of a block like
    ``handumi``, and no further. One level is needed because a merge of curated
    sources disagrees on exactly one entry -- each carries its own
    ``handumi.curation`` -- and judging the block whole would take
    ``tracking_schema`` and ``state_semantics`` with it, which every source does
    agree on and without which the dataset no longer identifies its own layout.
    Going deeper would be worse than either: each entry below is one record, and
    a session calibration reduced to the fields three sessions happen to share
    keeps its board definition while losing ``table_from_device``, which reads as
    a calibration that is present rather than one that is absent.
    """
    infos = [load_dataset_info(source.root) for source in plan.sources]
    carried, dropped = _agreed_entries(
        [
            {key: value for key, value in info.items() if key not in STANDARD_INFO_KEYS}
            for info in infos
        ],
        prefix="",
    )
    output_info.update(carried)
    return output_info, dropped


_MISSING = object()

# One level: the entries of a block like ``handumi`` are compared, what sits
# inside each entry is not. See _carry_source_info for why deeper is wrong.
_MAX_MERGE_DEPTH = 1


def _agreed_entries(
    values: list[dict[str, Any]],
    prefix: str,
    depth: int = 0,
) -> tuple[dict[str, Any], list[str]]:
    """Keep what every source states identically; name the rest by dotted path."""
    carried: dict[str, Any] = {}
    dropped: list[str] = []
    for key in sorted({key for value in values for key in value}):
        path = f"{prefix}{key}"
        present = [value.get(key, _MISSING) for value in values]
        if present[0] is not _MISSING and all(item == present[0] for item in present):
            carried[key] = copy.deepcopy(present[0])
            continue
        if depth < _MAX_MERGE_DEPTH and all(isinstance(item, dict) for item in present):
            nested, nested_dropped = _agreed_entries(present, f"{path}.", depth + 1)
            if nested:
                carried[key] = nested
            dropped.extend(nested_dropped)
            continue
        dropped.append(path)
    return carried, dropped


def _episode_source_map(plan: DatasetMergePlan) -> list[dict[str, Any]]:
    """One row per output episode, naming where it came from."""
    rows: list[dict[str, Any]] = []
    for source in plan.sources:
        for offset in range(source.total_episodes):
            rows.append(
                {
                    "output_episode_index": source.first_output_episode_index + offset,
                    "source_repo_id": source.repo_id,
                    "source_episode_index": offset,
                }
            )
    return rows


def _rewrite_constant_task_index_stats(root: Path) -> None:
    """Restate ``task_index`` statistics after every row was set to one task.

    The values are exact rather than recomputed: a single task means every row
    holds 0, so the whole distribution is known without reading the data back.
    """
    path = root / "meta" / "stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    entry = stats.get("task_index")
    if not isinstance(entry, dict):
        return
    count = entry.get("count")
    zero = [0.0]
    stats["task_index"] = {
        key: (copy.deepcopy(count) if key == "count" else list(zero))
        for key in entry
    }
    path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")


def merge_summary_lines(plan: DatasetMergePlan) -> list[str]:
    """The plan as a reviewer reads it, before any bytes are written."""
    lines = [
        "Dataset merge plan",
        f"  Output: {plan.output_root}",
        f"  Episodes: {plan.total_episodes}  Frames: {plan.total_frames}",
    ]
    for source in plan.sources:
        indices = source.output_episode_indices
        lines.append(
            f"  {source.repo_id}: {source.total_episodes} episodes, "
            f"{source.total_frames} frames -> output {indices.start}-{indices.stop - 1}"
        )
        lines.append(f"    tasks: {list(source.tasks)}")
    if plan.task is None:
        lines.append(f"  Tasks: kept as recorded {list(plan.source_tasks)}")
    else:
        lines.append(f"  Tasks: unified to {plan.task!r}")
    return lines
