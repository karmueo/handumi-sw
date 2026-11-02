from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from handumi.dataset.merging import (
    merge_dataset,
    merge_summary_lines,
    plan_dataset_merge,
)

_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["episode", "frame"],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["episode", "frame"],
    },
}


def _create_dataset(
    root: Path,
    *,
    repo_id: str,
    lengths: list[int],
    task: str,
    fps: int = 1,
    features: dict | None = None,
) -> None:
    resolved = features or _FEATURES
    dataset = LeRobotDataset.create(
        repo_id,
        fps=fps,
        features=resolved,
        root=root,
        use_videos=False,
    )
    for episode_index, length in enumerate(lengths):
        for frame_index in range(length):
            frame = {"task": task}
            for name in ("observation.state", "action"):
                value = np.zeros(resolved[name]["shape"], dtype=np.float32)
                value[0], value[1] = episode_index, frame_index
                frame[name] = value
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()


def _read_data(root: Path):
    return pq.read_table(sorted((root / "data").glob("chunk-*/*.parquet")))


def test_merge_joins_sessions_in_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2, 3], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[4], task="pick")

    plan = plan_dataset_merge([first, second], output_root=tmp_path / "merged")
    assert plan.total_episodes == 3
    assert plan.total_frames == 9
    assert plan.sources[0].output_episode_indices == range(0, 2)
    assert plan.sources[1].output_episode_indices == range(2, 3)

    result = merge_dataset(plan)
    assert result.total_episodes == 3
    assert result.total_frames == 9
    assert result.total_tasks == 1

    data = _read_data(result.root)
    assert data.num_rows == 9
    episode_index = np.asarray(data["episode_index"].to_pylist())
    assert np.array_equal(np.unique(episode_index), np.arange(3))
    assert np.array_equal(
        np.asarray(data["index"].to_pylist()), np.arange(9)
    )
    # The second session's only episode keeps its frames, renumbered as episode 2.
    state = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)
    assert np.array_equal(state[episode_index == 2][:, 1], np.arange(4))


def test_merge_unifies_task_wording(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="put the t block")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="put the t")

    plan = plan_dataset_merge(
        [first, second],
        output_root=tmp_path / "merged",
        task="put the t block",
    )
    assert set(plan.source_tasks) == {"put the t block", "put the t"}
    result = merge_dataset(plan)

    assert result.total_tasks == 1
    tasks = pq.read_table(result.root / "meta" / "tasks.parquet")
    assert tasks["task"].to_pylist() == ["put the t block"]
    data = _read_data(result.root)
    assert set(data["task_index"].to_pylist()) == {0}
    stats = json.loads((result.root / "meta" / "stats.json").read_text())
    # Statistics must describe the rewritten column, not the pre-merge one.
    assert stats["task_index"]["max"] == [0.0]
    assert stats["task_index"]["count"] == [4]
    episodes = pq.read_table(
        sorted((result.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    )
    assert {task for row in episodes["tasks"].to_pylist() for task in row} == {
        "put the t block"
    }


def test_merge_keeps_distinct_tasks_without_unification(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="place")

    result = merge_dataset(
        plan_dataset_merge([first, second], output_root=tmp_path / "merged")
    )
    assert result.total_tasks == 2


def test_merge_report_names_the_source_of_every_episode(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2, 2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[3], task="pick")

    result = merge_dataset(
        plan_dataset_merge(
            [first, second],
            output_root=tmp_path / "merged",
            source_repo_ids=["team/first", "team/second"],
        )
    )
    report = json.loads(result.report_path.read_text())
    assert report["kind"] == "handumi_dataset_merge"
    assert [source["repo_id"] for source in report["sources"]] == [
        "team/first",
        "team/second",
    ]
    assert report["episode_source"] == [
        {
            "output_episode_index": 0,
            "source_repo_id": "team/first",
            "source_episode_index": 0,
        },
        {
            "output_episode_index": 1,
            "source_repo_id": "team/first",
            "source_episode_index": 1,
        },
        {
            "output_episode_index": 2,
            "source_repo_id": "team/second",
            "source_episode_index": 0,
        },
    ]
    info = json.loads((result.root / "meta" / "info.json").read_text())
    assert info["handumi"]["merge"]["sources"][1]["output_episode_range"] == [2, 3]


def test_merge_leaves_the_sources_untouched(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="place")
    before = {
        path: path.read_bytes()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    }

    merge_dataset(
        plan_dataset_merge(
            [first, second], output_root=tmp_path / "merged", task="pick"
        )
    )
    after = {
        path: path.read_bytes()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_merge_carries_shared_metadata_and_drops_conflicting(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="pick")
    for root, pose in ((first, [0.0, 1.0]), (second, [9.0, 9.0])):
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["handumi"] = {"rig": "handumi-v2", "workspace_pose": pose}
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    result = merge_dataset(
        plan_dataset_merge([first, second], output_root=tmp_path / "merged")
    )
    report = json.loads(result.report_path.read_text())
    # Only the entry the sources state differently is dropped. Judging the whole
    # block would discard the rig they agree on along with it.
    assert report["dropped_info_keys"] == ["handumi.workspace_pose"]
    info = json.loads((result.root / "meta" / "info.json").read_text())
    assert info["handumi"]["rig"] == "handumi-v2"
    assert "workspace_pose" not in info["handumi"]
    assert info["handumi"]["merge"]["sources"][0]["total_episodes"] == 1


def test_merge_keeps_the_schema_identity_of_agreeing_sources(tmp_path: Path) -> None:
    """A curated source carries its own curation record; that must not cost the
    schema keys every source shares, which are what identify the layout."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="pick")
    shared = {"tracking_schema": "controller_raw_compact", "capture_schema": "sync"}
    for root, extra in ((first, {}), (second, {"curation": {"removed": [3]}})):
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["handumi"] = {**shared, **extra}
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    result = merge_dataset(
        plan_dataset_merge([first, second], output_root=tmp_path / "merged")
    )
    info = json.loads((result.root / "meta" / "info.json").read_text())
    assert info["handumi"]["tracking_schema"] == "controller_raw_compact"
    assert info["handumi"]["capture_schema"] == "sync"
    assert "curation" not in info["handumi"]
    report = json.loads(result.report_path.read_text())
    assert report["dropped_info_keys"] == ["handumi.curation"]


def test_plan_refuses_incompatible_sources(tmp_path: Path) -> None:
    first = tmp_path / "first"
    other_fps = tmp_path / "other_fps"
    other_features = tmp_path / "other_features"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(other_fps, repo_id="local/fps", lengths=[2], task="pick", fps=30)
    _create_dataset(
        other_features,
        repo_id="local/features",
        lengths=[2],
        task="pick",
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["a", "b", "c"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["episode", "frame"]},
        },
    )

    with pytest.raises(ValueError, match="fps="):
        plan_dataset_merge([first, other_fps], output_root=tmp_path / "merged")
    with pytest.raises(ValueError, match="observation.state"):
        plan_dataset_merge([first, other_features], output_root=tmp_path / "merged")


def test_plan_refuses_degenerate_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="pick")

    with pytest.raises(ValueError, match="at least two"):
        plan_dataset_merge([first], output_root=tmp_path / "merged")
    with pytest.raises(ValueError, match="repeated"):
        plan_dataset_merge([first, first], output_root=tmp_path / "merged")
    with pytest.raises(ValueError, match="does not exist"):
        plan_dataset_merge(
            [first, tmp_path / "missing"], output_root=tmp_path / "merged"
        )
    with pytest.raises(ValueError, match="outside every source"):
        plan_dataset_merge([first, second], output_root=first / "inner")
    with pytest.raises(ValueError, match="already exists"):
        plan_dataset_merge([first, second], output_root=second)
    with pytest.raises(ValueError, match="blank"):
        plan_dataset_merge([first, second], output_root=tmp_path / "merged", task="  ")


def test_merge_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="pick")
    plan = plan_dataset_merge([first, second], output_root=tmp_path / "merged")
    plan.output_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="already exists"):
        merge_dataset(plan)


def test_summary_lines_report_the_split_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2, 2], task="put the t")
    _create_dataset(second, repo_id="local/second", lengths=[1], task="put the t block")

    plan = plan_dataset_merge(
        [first, second], output_root=tmp_path / "merged", task="put the t block"
    )
    text = "\n".join(merge_summary_lines(plan))
    assert "Episodes: 3  Frames: 5" in text
    assert "output 0-1" in text
    assert "output 2-2" in text
    assert "unified to 'put the t block'" in text
    assert not (tmp_path / "merged").exists()


def _create_video_dataset(
    root: Path, *, repo_id: str, lengths: list[int], task: str
) -> None:
    features = {
        **_FEATURES,
        "observation.images.test": {
            "dtype": "video",
            "shape": (16, 16, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id,
        fps=5,
        features=features,
        root=root,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    for episode_index, length in enumerate(lengths):
        for frame_index in range(length):
            value = np.array([episode_index, frame_index], dtype=np.float32)
            dataset.add_frame(
                {
                    "observation.state": value,
                    "action": value,
                    "observation.images.test": np.full(
                        (16, 16, 3), episode_index * 40 + frame_index, dtype=np.uint8
                    ),
                    "task": task,
                }
            )
        dataset.save_episode()
    dataset.finalize()


def test_merge_concatenates_video_streams(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_video_dataset(first, repo_id="local/first", lengths=[5, 6], task="put the t")
    _create_video_dataset(second, repo_id="local/second", lengths=[7], task="put the t")

    result = merge_dataset(
        plan_dataset_merge(
            [first, second], output_root=tmp_path / "merged", task="put the t"
        )
    )

    assert result.total_episodes == 3
    assert result.total_frames == 18
    # validate_dataset_integrity re-probes every video and checks that each
    # episode's from/to timestamps stay contiguous inside its file, which is
    # what a botched concatenation breaks.
    video = result.validation["video"]["observation.images.test"]
    assert video["total_frames"] == 18
    episodes = pq.read_table(
        sorted((result.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    )
    prefix = "videos/observation.images.test"
    starts = episodes[f"{prefix}/from_timestamp"].to_pylist()
    ends = episodes[f"{prefix}/to_timestamp"].to_pylist()
    assert starts[0] == 0.0
    assert starts[1:] == ends[:-1]


def test_merge_treats_each_metadata_record_as_a_whole(tmp_path: Path) -> None:
    """A calibration that differs in one field is dropped entirely. Keeping the
    fields the sessions happen to share would read as a calibration that is
    present while its actual transform is gone."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    _create_dataset(first, repo_id="local/first", lengths=[2], task="pick")
    _create_dataset(second, repo_id="local/second", lengths=[2], task="pick")
    for root, translation in ((first, [0.0, 0.0, 0.6]), (second, [0.1, 0.0, 0.6])):
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["handumi"] = {
            "tracking_schema": "controller_raw_compact",
            "spatial_session_calibration": {
                "board_id": "charuco_5x7",
                "table_from_device": {"translation_m": translation},
            },
        }
        info_path.write_text(json.dumps(info, indent=4) + "\n")

    result = merge_dataset(
        plan_dataset_merge([first, second], output_root=tmp_path / "merged")
    )
    info = json.loads((result.root / "meta" / "info.json").read_text())
    assert info["handumi"]["tracking_schema"] == "controller_raw_compact"
    assert "spatial_session_calibration" not in info["handumi"]
    report = json.loads(result.report_path.read_text())
    assert report["dropped_info_keys"] == ["handumi.spatial_session_calibration"]
