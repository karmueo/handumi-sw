from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from handumi.dataset.direction import (
    EpisodeDirection,
    analyze_dataset_direction,
    score_directions,
)

_SIZE = 32


def _scene(block_x: int, distractor_x: int) -> np.ndarray:
    """A workspace: a small object that the task moves, and a large operator.

    The operator is deliberately bigger and placed at a position unrelated to
    the task, because that is what defeats a naive whole-frame comparison.
    """
    frame = np.full((_SIZE, _SIZE, 3), 120, dtype=np.uint8)
    frame[4:16, distractor_x : distractor_x + 12] = 200
    frame[22:27, block_x : block_x + 5] = 30
    return frame


def _create_dataset(
    root: Path,
    *,
    reversed_episodes: set[int],
    count: int = 6,
    length: int = 9,
    camera: str = "workspace",
) -> None:
    video_key = f"observation.images.{camera}"
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        video_key: {
            "dtype": "video",
            "shape": (_SIZE, _SIZE, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/direction",
        fps=5,
        features=features,
        root=root,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    rng = np.random.default_rng(0)
    for episode_index in range(count):
        backwards = episode_index in reversed_episodes
        for frame_index in range(length):
            progress = frame_index / (length - 1)
            # Forward: the object travels left to right. Reversed: the same
            # motion, played the other way.
            block = 4 + int(round((1 - progress if backwards else progress) * 20))
            distractor = int(rng.integers(2, 18))
            value = np.array([episode_index, frame_index], dtype=np.float32)
            dataset.add_frame(
                {
                    "observation.state": value,
                    "action": value,
                    video_key: _scene(block, distractor),
                    "task": "move the object",
                }
            )
        dataset.save_episode()
    dataset.finalize()


def test_flags_only_the_episodes_running_backwards(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, reversed_episodes={1, 4})

    reports, summary = analyze_dataset_direction(root)

    assert summary["reversed_episode_indices"] == [1, 4]
    flagged = {r.episode_index for r in reports if r.findings}
    assert flagged == {1, 4}
    for report in reports:
        similarity = float(report.metrics["direction_similarity"])
        if report.episode_index in {1, 4}:
            assert similarity < 0
            assert report.findings[0].code == "reversed_demonstration"
            # A warning, not a rejection: a dataset may hold both directions on
            # purpose, and only a reviewer knows whether this one does.
            assert report.findings[0].severity == "warning"
        else:
            assert similarity > 0
            assert report.findings == ()


def test_a_top_camera_is_the_workspace_camera(tmp_path: Path) -> None:
    root = tmp_path / "top"
    _create_dataset(root, reversed_episodes={2}, camera="top")

    reports, summary = analyze_dataset_direction(root)

    assert summary["reversed_episode_indices"] == [2]
    assert len(reports) == 6


def test_a_dataset_without_a_workspace_camera_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "wrist"
    _create_dataset(root, reversed_episodes=set(), camera="left")

    with pytest.raises(ValueError, match="no workspace camera"):
        analyze_dataset_direction(root)


def test_a_dataset_that_runs_one_way_flags_nothing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, reversed_episodes=set())

    reports, summary = analyze_dataset_direction(root)

    assert summary["reversed_episode_indices"] == []
    assert all(report.findings == () for report in reports)


def test_the_operator_does_not_decide_the_verdict(tmp_path: Path) -> None:
    """The distractor is larger than the object and moves at random.

    Comparing whole frames lets it dominate; weighting each pixel by how
    consistently it changes is what keeps the small object in charge.
    """
    root = tmp_path / "dataset"
    _create_dataset(root, reversed_episodes={2}, count=8, length=11)

    reports, _ = analyze_dataset_direction(root)
    scores = {
        r.episode_index: float(r.metrics["direction_similarity"]) for r in reports
    }
    forward = [v for k, v in scores.items() if k != 2]
    assert scores[2] < 0 < min(forward)
    # The gap has to be wide enough that the sign is not a coin flip.
    assert min(forward) - scores[2] > 0.5


def test_an_episode_that_changes_nothing_is_not_judged() -> None:
    still = np.zeros((4, 4, 3), dtype=np.float32)
    moved = still.copy()
    moved[1, 1] = 40.0
    edges = {
        0: (still.copy(), moved.copy()),
        1: (still.copy(), moved.copy()),
        2: (still.copy(), moved.copy()),
        3: (still.copy(), still.copy()),
    }
    results = {
        item.episode_index: item
        for item in score_directions(edges, dict.fromkeys(range(4), 10))
    }
    assert not results[3].reversed_demonstration
    assert results[3].change_magnitude == 0.0


def test_direction_needs_a_majority_to_compare_against() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="at least three episodes"):
        score_directions({0: (frame, frame), 1: (frame, frame)}, {0: 5, 1: 5})


def test_reversed_verdict_needs_both_a_negative_score_and_real_change() -> None:
    assert EpisodeDirection(0, 10, -0.9, 1.0).reversed_demonstration
    assert not EpisodeDirection(0, 10, 0.9, 1.0).reversed_demonstration
    # Negative but nothing moved: the cosine of near-zero vectors is noise.
    assert not EpisodeDirection(0, 10, -0.9, 0.0).reversed_demonstration


def test_report_reaches_the_merged_analysis(tmp_path: Path) -> None:
    from handumi.dataset.analysis import discover_quality_reports
    from handumi.dataset.quality import EpisodeQualityConfig, write_quality_report

    root = tmp_path / "dataset"
    _create_dataset(root, reversed_episodes={3})
    reports, _ = analyze_dataset_direction(root)
    path = write_quality_report(
        root / "meta" / "handumi_direction.json",
        reports,
        config=EpisodeQualityConfig(),
        dataset="local/direction",
    )

    # Without this the finding never reaches analyze, and so never reaches the
    # reviewer who curates.
    assert path in discover_quality_reports(root)
    payload = json.loads(path.read_text())
    flagged = [
        e["episode_index"]
        for e in payload["episodes"]
        if any(f["code"] == "reversed_demonstration" for f in e["findings"])
    ]
    assert flagged == [3]
