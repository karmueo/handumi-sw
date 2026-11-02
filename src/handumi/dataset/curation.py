"""Create validated derivative datasets from auditable analysis reports."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from handumi.dataset.analysis import dataset_payload_manifest, load_analysis_report
from handumi.dataset.paths import portable_path

CURATION_SCHEMA_VERSION = 1
CURATION_KIND = "handumi_dataset_curation"

STANDARD_INFO_KEYS = {
    "codebase_version",
    "robot_type",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "chunks_size",
    "data_files_size_in_mb",
    "video_files_size_in_mb",
    "fps",
    "splits",
    "data_path",
    "video_path",
    "features",
}


@dataclass(frozen=True)
class DatasetCurationPlan:
    source_root: Path
    source_repo_id: str
    analysis_path: Path
    output_root: Path
    output_repo_id: str
    source_total_episodes: int
    source_total_frames: int
    excluded_source_episode_indices: tuple[int, ...]
    kept_source_episode_indices: tuple[int, ...]
    # Split by who decided, so the record shows which removals were mechanical
    # and which were a judgement call.
    auto_excluded_source_episode_indices: tuple[int, ...] = ()
    human_excluded_source_episode_indices: tuple[int, ...] = ()

    @property
    def output_total_episodes(self) -> int:
        return len(self.kept_source_episode_indices)


@dataclass(frozen=True)
class DatasetCurationResult:
    root: Path
    repo_id: str
    total_episodes: int
    total_frames: int
    excluded_source_episode_indices: tuple[int, ...]
    report_path: Path
    validation: dict[str, Any]


def plan_dataset_curation(
    source_root: str | Path,
    *,
    analysis_path: str | Path,
    output_root: str | Path,
    source_repo_id: str | None = None,
    output_repo_id: str | None = None,
    exclude_episode_indices: list[int] | tuple[int, ...] = (),
    exclude_rejected: bool = False,
) -> DatasetCurationPlan:
    """Build and validate a curation plan without changing either dataset."""
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    analysis = Path(analysis_path).resolve()
    if not source.is_dir():
        raise ValueError(f"Source dataset does not exist: {source}")
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be a separate path outside the source dataset")

    info = load_dataset_info(source)
    report = load_analysis_report(analysis)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    report_dataset = report.get("dataset", {})
    if int(report_dataset.get("total_episodes", -1)) != total_episodes:
        raise ValueError("Analysis report is stale: total_episodes no longer matches")
    if int(report_dataset.get("total_frames", -1)) != total_frames:
        raise ValueError("Analysis report is stale: total_frames no longer matches")
    report_indices = sorted(
        int(item["source_episode_index"])
        for item in report["episodes"]
        if isinstance(item, dict) and "source_episode_index" in item
    )
    if report_indices != list(range(total_episodes)):
        raise ValueError("Analysis report has incomplete or invalid episode indices")
    report_manifest = report_dataset.get("payload_manifest")
    current_manifest = dataset_payload_manifest(source)
    if report_manifest != current_manifest:
        raise ValueError(
            "Analysis report is stale: dataset payload fingerprint changed"
        )

    report_root = report_dataset.get("root")
    # Reports record the portable form; older ones recorded an absolute path,
    # and both have to keep identifying the dataset they were written for.
    if report_root and str(report_root) not in {
        portable_path(source),
        str(source),
    }:
        raise ValueError(
            f"Analysis report belongs to a different dataset root: {report_root}"
        )

    human = {int(value) for value in exclude_episode_indices}
    # Findings graded "reject" describe episodes no reasonable reviewer keeps --
    # corrupt state, an unreachable start pose, sensors down beyond warm-up --
    # so requiring a human to retype those indices adds transcription risk
    # without adding judgement. Warnings still need a person: whether a brief
    # self-intersection or an orientation outlier disqualifies an episode
    # depends on the task, and no threshold knows that.
    automatic: set[int] = set()
    if exclude_rejected:
        automatic = {
            int(item["source_episode_index"])
            for item in report["episodes"]
            if isinstance(item, dict) and item.get("status") == "rejected"
        }
    excluded = human | automatic
    invalid = sorted(
        value for value in excluded if value < 0 or value >= total_episodes
    )
    if invalid:
        raise ValueError(f"Episode indices out of range: {invalid}")
    if not excluded:
        raise ValueError(
            "Curation requires episode indices to exclude: pass --exclude, or "
            "--exclude-rejected when the analysis rejected any episode."
        )
    kept = tuple(value for value in range(total_episodes) if value not in excluded)
    if not kept:
        raise ValueError("Curation cannot remove every episode")

    resolved_source_repo = source_repo_id or str(
        report_dataset.get("repo_id") or f"local/{source.name}"
    )
    resolved_output_repo = output_repo_id or f"local/{output.name}"
    return DatasetCurationPlan(
        source_root=source,
        source_repo_id=resolved_source_repo,
        analysis_path=analysis,
        output_root=output,
        output_repo_id=resolved_output_repo,
        source_total_episodes=total_episodes,
        source_total_frames=total_frames,
        excluded_source_episode_indices=tuple(sorted(excluded)),
        kept_source_episode_indices=kept,
        auto_excluded_source_episode_indices=tuple(sorted(automatic - human)),
        human_excluded_source_episode_indices=tuple(sorted(human)),
    )


def curate_dataset(plan: DatasetCurationPlan) -> DatasetCurationResult:
    """Execute a curation plan locally and atomically make its output visible."""
    if plan.output_root.exists():
        raise ValueError(f"Output path already exists: {plan.output_root}")
    plan.output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.output_root.name}.curating-",
            dir=plan.output_root.parent,
        )
    )
    build_root = temp_parent / "dataset"
    source_manifest_before = dataset_payload_manifest(plan.source_root)

    try:
        result = _build_curated_dataset(plan, build_root, source_manifest_before)
        source_manifest_after = dataset_payload_manifest(plan.source_root)
        if source_manifest_after != source_manifest_before:
            raise RuntimeError("Source dataset changed while curation was running")
        if plan.output_root.exists():
            raise RuntimeError(
                f"Output path appeared during curation: {plan.output_root}"
            )
        build_root.replace(plan.output_root)
        return DatasetCurationResult(
            root=plan.output_root,
            repo_id=plan.output_repo_id,
            total_episodes=result["total_episodes"],
            total_frames=result["total_frames"],
            excluded_source_episode_indices=plan.excluded_source_episode_indices,
            report_path=plan.output_root / "meta" / "handumi_curation.json",
            validation=result["validation"],
        )
    finally:
        if temp_parent.exists():
            shutil.rmtree(temp_parent)


def validate_dataset_integrity(
    root: str | Path,
    *,
    repo_id: str | None = None,
    load_with_lerobot: bool = True,
) -> dict[str, Any]:
    """Cross-check LeRobot metadata, Parquet rows, stats, and video streams."""
    dataset_root = Path(root)
    info = load_dataset_info(dataset_root)
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    fps = float(info.get("fps", 0))
    if total_episodes <= 0 or total_frames <= 0 or fps <= 0:
        raise RuntimeError(
            "Dataset info contains invalid episode, frame, or FPS totals"
        )

    episode_files = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")
    )
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not episode_files or not data_files:
        raise RuntimeError("Dataset is missing episode or data Parquet files")
    episodes = pq.read_table(episode_files)
    if episodes.num_rows != total_episodes:
        raise RuntimeError("Episode metadata row count does not match info.json")
    episode_indices = np.asarray(episodes["episode_index"].to_pylist(), dtype=np.int64)
    lengths = np.asarray(episodes["length"].to_pylist(), dtype=np.int64)
    if not np.array_equal(episode_indices, np.arange(total_episodes)):
        raise RuntimeError("Episode indices are not continuous and zero-based")
    if int(lengths.sum()) != total_frames:
        raise RuntimeError("Episode lengths do not sum to total_frames")
    expected_to = np.cumsum(lengths)
    expected_from = np.concatenate((np.array([0], dtype=np.int64), expected_to[:-1]))
    dataset_from = np.asarray(
        episodes["dataset_from_index"].to_pylist(), dtype=np.int64
    )
    dataset_to = np.asarray(episodes["dataset_to_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(dataset_from, expected_from) or not np.array_equal(
        dataset_to, expected_to
    ):
        raise RuntimeError("Episode dataset offsets are stale")
    _validate_splits(info.get("splits"), total_episodes)

    data = pq.read_table(
        data_files,
        columns=["episode_index", "frame_index", "index", "timestamp", "task_index"],
    )
    if data.num_rows != total_frames:
        raise RuntimeError("Data Parquet rows do not match total_frames")
    data_episode = np.asarray(data["episode_index"].to_pylist(), dtype=np.int64)
    data_frame = np.asarray(data["frame_index"].to_pylist(), dtype=np.int64)
    data_index = np.asarray(data["index"].to_pylist(), dtype=np.int64)
    data_timestamp = np.asarray(data["timestamp"].to_pylist(), dtype=np.float64)
    data_task = np.asarray(data["task_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(data_index, np.arange(total_frames)):
        raise RuntimeError("Global frame indices are not continuous")
    counts = np.bincount(data_episode, minlength=total_episodes)
    if not np.array_equal(counts, lengths):
        raise RuntimeError("Per-episode Parquet frame counts do not match metadata")
    for episode_index, length in enumerate(lengths):
        frames = data_frame[data_episode == episode_index]
        if not np.array_equal(frames, np.arange(length)):
            raise RuntimeError(f"Frame indices are invalid for episode {episode_index}")
    timestamp_tolerance = max(1e-6, (1.0 / fps) * 1e-4)
    if not np.allclose(
        data_timestamp,
        data_frame / fps,
        rtol=0,
        atol=timestamp_tolerance,
    ):
        raise RuntimeError("Frame timestamps do not match frame_index / fps")

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    stats_path = dataset_root / "meta" / "stats.json"
    if not tasks_path.is_file() or not stats_path.is_file():
        raise RuntimeError("Dataset is missing tasks.parquet or stats.json")
    tasks = pq.read_table(tasks_path)
    task_count = tasks.num_rows
    if task_count != int(info.get("total_tasks", -1)):
        raise RuntimeError("Task count does not match info.json")
    task_indices = np.asarray(tasks["task_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(task_indices, np.arange(task_count)):
        raise RuntimeError("Task indices are not continuous and zero-based")
    if np.any(data_task < 0) or np.any(data_task >= task_count):
        raise RuntimeError("Data contains task indices outside tasks.parquet")
    task_names = [str(value) for value in tasks["task"].to_pylist()]
    episode_tasks = episodes["tasks"].to_pylist()
    for episode_index in range(total_episodes):
        used = sorted(set(data_task[data_episode == episode_index].tolist()))
        expected_names = {task_names[index] for index in used}
        if set(episode_tasks[episode_index] or []) != expected_names:
            raise RuntimeError(
                f"Episode task metadata is stale for episode {episode_index}"
            )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    count_features = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
    count_features.update(
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") not in {"image", "video"}
    )
    for feature in sorted(count_features):
        count = np.asarray(stats.get(feature, {}).get("count", []))
        if count.size == 0 or not np.all(count == total_frames):
            raise RuntimeError(f"Statistics count is stale for {feature}")

    video_report = _validate_videos(dataset_root, info, episodes, lengths)
    audio_report = _validate_audio(dataset_root, info, total_episodes)
    if load_with_lerobot:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        loaded = LeRobotDataset(
            repo_id=repo_id or f"local/{dataset_root.name}",
            root=dataset_root,
            download_videos=False,
        )
        if len(loaded) != total_frames or loaded.meta.total_episodes != total_episodes:
            raise RuntimeError("LeRobotDataset load disagrees with dataset metadata")

    return {
        "status": "ok",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": task_count,
        "min_duration_s": float(lengths.min() / fps),
        "max_duration_s": float(lengths.max() / fps),
        "video": video_report,
        "audio": audio_report,
        "lerobot_load": "ok" if load_with_lerobot else "skipped",
    }


def _validate_splits(value: Any, total_episodes: int) -> None:
    if not isinstance(value, dict) or not value:
        raise RuntimeError("Dataset split metadata is missing")
    coverage = np.zeros(total_episodes, dtype=np.int8)
    for name, interval in value.items():
        try:
            start_text, end_text = str(interval).split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid dataset split {name!r}: {interval!r}") from exc
        if start < 0 or end <= start or end > total_episodes:
            raise RuntimeError(f"Dataset split {name!r} is out of range")
        coverage[start:end] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Dataset splits must cover every episode exactly once")


def _build_curated_dataset(
    plan: DatasetCurationPlan,
    build_root: Path,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    from lerobot.datasets import dataset_tools
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import create_lerobot_dataset_card

    from handumi.dataset.lerobot_compat import DatasetCardInfo

    source_info = load_dataset_info(plan.source_root)
    source = LeRobotDataset(
        repo_id=plan.source_repo_id,
        root=plan.source_root,
        download_videos=True,
    )
    episode_mapping = {
        old_index: new_index
        for new_index, old_index in enumerate(plan.kept_source_episode_indices)
    }
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=plan.output_repo_id,
        fps=source.meta.fps,
        features=source.meta.features,
        robot_type=source.meta.robot_type,
        root=build_root,
        use_videos=bool(source.meta.video_keys),
        chunks_size=source.meta.chunks_size,
        data_files_size_in_mb=source.meta.data_files_size_in_mb,
        video_files_size_in_mb=source.meta.video_files_size_in_mb,
    )

    video_metadata = None
    if source.meta.video_keys:
        video_metadata = _copy_videos_losslessly(
            source, build_root, episode_mapping, source_info
        )
    if source.meta.video_keys and video_metadata is None:
        _source_video_encoding(plan.source_root, source_info)
        video_metadata = dataset_tools._copy_and_reindex_videos(
            source, new_meta, episode_mapping,
        )
    data_metadata = dataset_tools._copy_and_reindex_data(
        source,
        new_meta,
        episode_mapping,
    )
    dataset_tools._copy_and_reindex_episodes_metadata(
        source,
        new_meta,
        episode_mapping,
        data_metadata,
        video_metadata,
    )
    audio_files = _copy_and_reindex_audio(
        plan.source_root,
        build_root,
        source_info,
        episode_mapping,
    )

    analysis = load_analysis_report(plan.analysis_path)
    analysis_episodes = {
        int(item["source_episode_index"]): item for item in analysis["episodes"]
    }
    output_info_path = build_root / "meta" / "info.json"
    output_info = load_dataset_info(build_root)
    for key, value in source_info.items():
        if key not in STANDARD_INFO_KEYS:
            output_info[key] = copy.deepcopy(value)
    handumi = output_info.get("handumi")
    if not isinstance(handumi, dict):
        handumi = {}
        output_info["handumi"] = handumi
    curation_summary = {
        "schema_version": CURATION_SCHEMA_VERSION,
        "source_dataset": plan.source_repo_id,
        "source_total_episodes": plan.source_total_episodes,
        "source_total_frames": plan.source_total_frames,
        "analysis_report": plan.analysis_path.name,
        "removed_source_episode_indices": list(plan.excluded_source_episode_indices),
        "removed_automatically": list(plan.auto_excluded_source_episode_indices),
        "removed_by_reviewer": list(plan.human_excluded_source_episode_indices),
        "removed_frames": sum(
            int(analysis_episodes[index]["frame_count"])
            for index in plan.excluded_source_episode_indices
        ),
    }
    handumi["curation"] = curation_summary
    output_info_path.write_text(
        json.dumps(output_info, indent=4) + "\n",
        encoding="utf-8",
    )

    validation = validate_dataset_integrity(
        build_root,
        repo_id=plan.output_repo_id,
        load_with_lerobot=True,
    )
    curation_report = {
        "schema_version": CURATION_SCHEMA_VERSION,
        "kind": CURATION_KIND,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "repo_id": plan.source_repo_id,
            "root": portable_path(plan.source_root),
            "total_episodes": plan.source_total_episodes,
            "total_frames": plan.source_total_frames,
            "payload_manifest": source_manifest,
        },
        "output": {
            "repo_id": plan.output_repo_id,
            "root": portable_path(plan.output_root),
            "total_episodes": validation["total_episodes"],
            "total_frames": validation["total_frames"],
        },
        "analysis_report": portable_path(plan.analysis_path),
        "excluded_source_episode_indices": list(plan.excluded_source_episode_indices),
        "removed_episodes": [
            analysis_episodes[index] for index in plan.excluded_source_episode_indices
        ],
        "episode_index_mapping": {
            str(old): new for old, new in episode_mapping.items()
        },
        "audio_files_copied": audio_files,
        "validation": validation,
        "output_payload_manifest": dataset_payload_manifest(build_root),
        "analysis": analysis,
    }
    report_path = build_root / "meta" / "handumi_curation.json"
    report_path.write_text(
        json.dumps(curation_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task_names = pq.read_table(build_root / "meta" / "tasks.parquet")[
        "task"
    ].to_pylist()
    task_summary = ", ".join(str(task) for task in task_names)
    card = create_lerobot_dataset_card(
        tags=["HandUMI", "curated"],
        dataset_info=DatasetCardInfo(output_info),
        license=dataset_license(plan.source_root),
        repo_id=plan.output_repo_id,
        dataset_description=(
            "Curated HandUMI dataset derived locally from "
            f"{plan.source_repo_id}. Tasks: {task_summary}. Removed source episodes: "
            f"{list(plan.excluded_source_episode_indices)}."
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
        "validation": validation,
    }


def load_dataset_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset info {path}: {exc}") from exc


def _copy_videos_losslessly(
    source,
    build_root: Path,
    episode_mapping: dict[int, int],
    info: dict[str, Any],
) -> dict[int, dict[str, Any]] | None:
    """Rebuild each video from the episodes kept, copying compressed packets.

    Returns the same per-episode metadata LeRobot's re-encoding path produces,
    or ``None`` when the streams cannot be copied -- an episode not starting on
    a keyframe, or more than one file per camera -- so the caller falls back to
    re-encoding rather than writing something it cannot guarantee.

    Re-encoding a dataset to drop a few episodes costs minutes per camera and
    returns frames that differ from the ones the review graded. Copying is
    exact and, measured on a 201-episode dataset, took under a second.
    """
    from handumi.dataset.videocopy import (
        EpisodeSegment,
        copy_segments,
        keyframe_aligned,
    )

    fps = float(source.meta.fps)
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    if not video_keys:
        return {}
    kept = sorted(episode_mapping)
    metadata: dict[int, dict[str, Any]] = {
        episode_mapping[old]: {} for old in kept
    }
    for key in video_keys:
        files = sorted((Path(source.root) / "videos" / key).glob("chunk-*/*.mp4"))
        if len(files) != 1:
            return None
        segments = []
        for old in kept:
            episode = source.meta.episodes[old]
            first = round(float(episode[f"videos/{key}/from_timestamp"]) * fps)
            segments.append(EpisodeSegment(first, int(episode["length"])))
        if not keyframe_aligned(files[0], segments):
            return None
        destination = build_root / "videos" / key / "chunk-000" / "file-000.mp4"
        written = copy_segments(files[0], destination, segments)
        expected = sum(segment.frame_count for segment in segments)
        if written != expected:
            raise RuntimeError(
                f"Lossless copy wrote {written} frames for {key}, expected {expected}"
            )
        cumulative = 0.0
        for old in kept:
            new = episode_mapping[old]
            duration = int(source.meta.episodes[old]["length"]) / fps
            metadata[new][f"videos/{key}/chunk_index"] = 0
            metadata[new][f"videos/{key}/file_index"] = 0
            metadata[new][f"videos/{key}/from_timestamp"] = cumulative
            metadata[new][f"videos/{key}/to_timestamp"] = cumulative + duration
            cumulative += duration
    return metadata


def _source_video_encoding(root: Path, info: dict[str, Any]) -> tuple[str, str]:
    from lerobot.datasets.video_utils import get_video_info

    codecs: set[str] = set()
    pixel_formats: set[str] = set()
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        files = sorted((root / "videos" / key).glob("chunk-*/*.mp4"))
        if not files:
            raise RuntimeError(f"Video feature has no files: {key}")
        actual = get_video_info(files[0])
        declared = feature.get("info", {})
        codec = str(actual.get("video.codec", ""))
        pixel_format = str(actual.get("video.pix_fmt", ""))
        if declared.get("video.codec") != codec:
            raise RuntimeError(f"Declared and actual video codec disagree for {key}")
        if declared.get("video.pix_fmt") != pixel_format:
            raise RuntimeError(f"Declared and actual pixel format disagree for {key}")
        codecs.add(codec)
        pixel_formats.add(pixel_format)
    if len(codecs) != 1 or len(pixel_formats) != 1:
        raise RuntimeError("All video features must use one codec and pixel format")
    codec = next(iter(codecs))
    encoder = {
        "h264": "libx264",
        "hevc": "libx265",
        "h265": "libx265",
        "av1": "libsvtav1",
    }.get(codec)
    if encoder is None:
        raise RuntimeError(f"Unsupported source video codec for curation: {codec}")
    return encoder, next(iter(pixel_formats))


def _copy_and_reindex_audio(
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    episode_mapping: dict[int, int],
) -> int:
    handumi = info.get("handumi")
    audio = handumi.get("audio") if isinstance(handumi, dict) else None
    if not isinstance(audio, dict) or not audio.get("enabled"):
        return 0
    path_template = audio.get("path")
    audio_key = audio.get("key")
    if not isinstance(path_template, str) or not isinstance(audio_key, str):
        raise TypeError("Enabled audio metadata is missing path or key")
    chunks_size = int(info.get("chunks_size", 1000))
    copied = 0
    for old_index, new_index in episode_mapping.items():
        old_path = source_root / path_template.format(
            audio_key=audio_key,
            chunk_index=old_index // chunks_size,
            file_index=old_index % chunks_size,
        )
        new_path = output_root / path_template.format(
            audio_key=audio_key,
            chunk_index=new_index // chunks_size,
            file_index=new_index % chunks_size,
        )
        if not old_path.is_file():
            raise RuntimeError(f"Missing source episode audio: {old_path}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
        copied += 1
    return copied


def _validate_videos(
    root: Path,
    info: dict[str, Any],
    episodes: Any,
    lengths: np.ndarray,
) -> dict[str, Any]:
    fps = float(info["fps"])
    total_frames = int(info["total_frames"])
    result: dict[str, Any] = {}
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        files = sorted((root / "videos" / key).glob("chunk-*/*.mp4"))
        if not files:
            raise RuntimeError(f"Missing videos for {key}")
        frame_total = 0
        file_details = []
        declared = feature.get("info", {})
        for path in files:
            probed = _probe_video(path)
            if probed["codec"] != declared.get("video.codec"):
                raise RuntimeError(f"Video codec mismatch in {path}")
            if probed["pixel_format"] != declared.get("video.pix_fmt"):
                raise RuntimeError(f"Video pixel format mismatch in {path}")
            if abs(probed["fps"] - fps) > 1e-6:
                raise RuntimeError(f"Video FPS mismatch in {path}")
            if probed["width"] != int(declared.get("video.width", 0)):
                raise RuntimeError(f"Video width mismatch in {path}")
            if probed["height"] != int(declared.get("video.height", 0)):
                raise RuntimeError(f"Video height mismatch in {path}")
            frame_total += probed["frames"]
            file_details.append({"path": str(path.relative_to(root)), **probed})
        if frame_total != total_frames:
            raise RuntimeError(
                f"Video frames for {key}={frame_total}, expected {total_frames}"
            )

        prefix = f"videos/{key}"
        previous_by_file: dict[tuple[int, int], float] = {}
        for episode_index, length in enumerate(lengths.tolist()):
            chunk = int(episodes[f"{prefix}/chunk_index"][episode_index].as_py())
            file_index = int(episodes[f"{prefix}/file_index"][episode_index].as_py())
            start = float(episodes[f"{prefix}/from_timestamp"][episode_index].as_py())
            end = float(episodes[f"{prefix}/to_timestamp"][episode_index].as_py())
            if round((end - start) * fps) != int(length):
                raise RuntimeError(
                    f"Video duration mismatch for episode {episode_index}"
                )
            file_key = (chunk, file_index)
            previous = previous_by_file.get(file_key)
            if previous is not None and abs(start - previous) > 1e-6:
                raise RuntimeError(f"Non-contiguous video offsets for {key}")
            previous_by_file[file_key] = end
        result[key] = {"total_frames": frame_total, "files": file_details}
    return result


def _probe_video(path: Path) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect video {path}: {exc}") from exc
    streams = json.loads(output).get("streams", [])
    if not streams:
        raise RuntimeError(f"Video has no stream: {path}")
    stream = streams[0]
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": _frame_rate(stream),
        "frames": int(stream["nb_frames"]),
    }


def _frame_rate(stream: dict[str, Any]) -> float:
    """Read the coded frame rate, not the one derived from the header duration.

    ``avg_frame_rate`` is frames divided by the container's own duration, and
    that duration is stored in the movie header's coarser timescale. Rounding it
    puts the average a hair off a rate the stream actually holds exactly: a
    concatenation of 53834 frames whose presentation stamps are all 1/30 apart
    reports 107668/3589 = 29.99944. Comparing that against the dataset FPS
    rejects a video that is perfectly regular, so read ``r_frame_rate``, which is
    the rate the stream is coded at, and keep the average only as a fallback for
    a stream that does not declare one.
    """
    for key in ("r_frame_rate", "avg_frame_rate"):
        value = str(stream.get(key, "")).strip()
        if not value or "/" not in value:
            continue
        numerator, denominator = value.split("/", maxsplit=1)
        try:
            rate = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return rate
    raise RuntimeError(f"Video declares no usable frame rate: {stream}")


def _validate_audio(
    root: Path,
    info: dict[str, Any],
    total_episodes: int,
) -> dict[str, Any]:
    handumi = info.get("handumi")
    audio = handumi.get("audio") if isinstance(handumi, dict) else None
    if not isinstance(audio, dict) or not audio.get("enabled"):
        return {"enabled": False, "files": 0}
    from handumi.audio import validate_audio_files

    validate_audio_files(
        root,
        total_episodes,
        chunks_size=int(info.get("chunks_size", 1000)),
    )
    path_template = str(
        audio.get(
            "path",
            "audio/{audio_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.wav",
        )
    )
    directory = path_template.split("/{audio_key}", maxsplit=1)[0]
    files = list((root / directory).rglob("*.wav"))
    return {"enabled": True, "files": len(files)}


def dataset_license(root: Path) -> str:
    readme = root / "README.md"
    if not readme.is_file():
        return "other"
    for line in readme.read_text(encoding="utf-8").splitlines()[:20]:
        if line.startswith("license:"):
            value = line.partition(":")[2].strip()
            return value or "other"
    return "other"
