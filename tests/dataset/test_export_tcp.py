"""验证 TCP 导出的坐标变换、图像处理和质量筛选。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.scripts import export_tcp
from handumi.scripts.export_tcp import letterbox, quality_selection, relative_tcp


def test_relative_tcp_uses_each_episode_initial_orientation():
    """非零初始旋转下平移和旋转均转换到各自起始坐标系。"""
    q = Rotation.from_euler("z", 90, degrees=True)
    poses = np.array(
        [
            [1, 2, 0, *q.as_quat()],
            [1, 3, 0, *(q * Rotation.from_euler("x", 45, degrees=True)).as_quat()],
        ]
    )
    values = relative_tcp(poses, [0.6, 0.9])
    np.testing.assert_allclose(values[0], [0, 0, 0, 0, 0, 0, 0.6], atol=1e-6)
    np.testing.assert_allclose(values[1], [1, 0, 0, np.pi / 4, 0, 0, 0.9], atol=1e-6)
    np.testing.assert_allclose(relative_tcp(poses[1:], [0.9])[0, :6], 0, atol=1e-6)


def test_letterbox_preserves_image_and_black_borders():
    """宽图像按比例缩放，顶部和底部填黑。"""
    image = np.full((108, 192, 3), 255, dtype=np.uint8)
    output = letterbox(image)
    assert output.shape == (224, 224, 3)
    assert not output[0].any()
    assert output[112].min() == 255


def test_quality_selection_preserves_rejected_reason(tmp_path):
    """质量报告必须覆盖全部回合，并保留拒绝原因。"""
    (tmp_path / "meta").mkdir()
    path = tmp_path / "meta" / "handumi_quality.json"
    rows = [
        {"episode_index": 0, "frame_count": 3, "status": "accepted"},
        {
            "episode_index": 1,
            "frame_count": 4,
            "status": "rejected",
            "findings": [{"code": "bad_tracking"}],
        },
    ]
    path.write_text(json.dumps({"dataset": "local/source", "episodes": rows}))
    episodes = [{"episode_index": 0, "length": 3}, {"episode_index": 1, "length": 4}]
    kept, rejected = quality_selection(tmp_path, "local/source", episodes)
    assert kept == [0]
    assert rejected[0]["findings"][0]["code"] == "bad_tracking"
    rows[1]["frame_count"] = 5
    path.write_text(json.dumps({"dataset": "local/source", "episodes": rows}))
    with pytest.raises(ValueError, match="帧数"):
        quality_selection(tmp_path, "local/source", episodes)

    path.write_text("{")
    with pytest.raises(ValueError, match="重新运行 handumi validate"):
        quality_selection(tmp_path, "local/source", episodes)
    rows[0]["status"] = "rejected"
    rows[0]["findings"] = []
    rows[1]["frame_count"] = 4
    path.write_text(json.dumps({"dataset": "local/source", "episodes": rows}))
    with pytest.raises(ValueError, match="所有回合"):
        quality_selection(tmp_path, "local/source", episodes)


def test_export_cleans_partial_dataset_on_short_video(tmp_path, monkeypatch):
    """解码提前结束时关闭写入器并清除未完成的数据集。"""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    import handumi.dataset.curation as curation

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    prefix = "videos/observation.images.right_wrist/"
    row = {
        "episode_index": 0,
        "length": 2,
        "tasks": ["task"],
        prefix + "chunk_index": 0,
        prefix + "file_index": 0,
        prefix + "from_timestamp": 0.0,
        prefix + "to_timestamp": 2 / 30,
    }
    info = {
        "fps": 30,
        "total_episodes": 1,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "capture_schema": "synchronized_sources_v2",
            "tcp_pose": {
                "frame": "table",
                "feature": "observation.tracking.right_tcp_pose",
            },
        },
    }
    monkeypatch.setattr(export_tcp, "load_info", lambda _: info)
    monkeypatch.setattr(
        export_tcp.pq, "read_table", lambda _: SimpleNamespace(to_pylist=lambda: [row])
    )
    monkeypatch.setattr(export_tcp, "quality_selection", lambda *args: ([0], []))
    poses = np.array([[0, 0, 0, 0, 0, 0, 1]] * 2)
    monkeypatch.setattr(
        export_tcp,
        "load_raw_episode",
        lambda **kwargs: SimpleNamespace(
            signals={
                "observation.tracking.right_tcp_pose": poses,
                "observation.gripper.right.normalized": [0.8, 0.7],
            }
        ),
    )

    class Writer:
        """记录异常路径上写入器是否结束。"""

        _is_finalized = False

        def add_frame(self, frame):
            """接受测试帧。"""

        def finalize(self):
            """模拟释放写入器。"""
            self._is_finalized = True

    writer = Writer()
    monkeypatch.setattr(
        lerobot_dataset.LeRobotDataset, "create", lambda *args, **kwargs: writer
    )
    monkeypatch.setattr(
        curation, "validate_dataset_integrity", lambda *args, **kwargs: None
    )

    class Capture:
        """模拟视频首帧有效而第二帧丢失。"""

        def __init__(self, path):
            self.count = 0

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def get(self, *args):
            return 2

        def read(self):
            self.count += 1
            return (
                (True, np.zeros((20, 20, 3), dtype=np.uint8))
                if self.count == 1
                else (False, None)
            )

        def release(self):
            pass

    monkeypatch.setattr(export_tcp.cv2, "VideoCapture", Capture)
    with pytest.raises(RuntimeError, match="视频帧数不足"):
        export_tcp.export(source, output, "local/output")
    assert writer._is_finalized
    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))


def test_export_shared_video_and_skipped_episode(tmp_path, monkeypatch):
    """真实写入器保留两个回合末帧，视频读取跨过被拒绝回合。"""
    import cv2
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    video_dir = source / "videos" / "observation.images.right_wrist" / "chunk-000"
    video_dir.mkdir(parents=True)
    path = video_dir / "file-000.mp4"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert video.isOpened()
    for intensity in [20, 40, 60, 80, 120, 160]:
        video.write(np.full((48, 64, 3), intensity, dtype=np.uint8))
    video.release()
    prefix = "videos/observation.images.right_wrist/"
    rows = [
        {
            "episode_index": i,
            "length": 2,
            "tasks": ["pick place"],
            prefix + "chunk_index": 0,
            prefix + "file_index": 0,
            prefix + "from_timestamp": i * 2 / 30,
            prefix + "to_timestamp": (i + 1) * 2 / 30,
        }
        for i in range(3)
    ]
    info = {
        "fps": 30,
        "total_episodes": 3,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "capture_schema": "synchronized_sources_v2",
            "tcp_pose": {
                "frame": "table",
                "feature": "observation.tracking.right_tcp_pose",
            },
        },
    }
    monkeypatch.setattr(export_tcp, "load_info", lambda _: info)
    original_read = pq.read_table

    def read_once(*args, **kwargs):
        monkeypatch.setattr(export_tcp.pq, "read_table", original_read)
        return SimpleNamespace(to_pylist=lambda: rows)

    monkeypatch.setattr(export_tcp.pq, "read_table", read_once)
    monkeypatch.setattr(
        export_tcp,
        "quality_selection",
        lambda *args: (
            [0, 2],
            [{"source_episode_index": 1, "findings": [{"code": "reject"}]}],
        ),
    )

    def raw(**kwargs):
        episode = kwargs["episode"]
        return SimpleNamespace(
            signals={
                "observation.tracking.right_tcp_pose": np.array(
                    [[episode, 0, 0, 0, 0, 0, 1], [episode + 0.1, 0, 0, 0, 0, 0, 1]]
                ),
                "observation.gripper.right.normalized": [0.8, 0.7],
            }
        )

    monkeypatch.setattr(export_tcp, "load_raw_episode", raw)
    output = tmp_path / "output"
    export_tcp.export(source, output, "local/output")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/output", root=output, video_backend="pyav")
    assert len(dataset) == 4
    for index in range(4):
        item = dataset[index]
        np.testing.assert_allclose(item["action"], item["observation.state"])
        assert (
            abs(float(item["observation.state"][0]) - (0.1 if index % 2 else 0)) < 1e-5
        )
        assert (
            abs(float(item["observation.state"][6]) - (0.7 if index % 2 else 0.8))
            < 1e-5
        )
        assert item["task"] == "pick place"
        expected = [20, 40, 120, 160][index] / 255
        assert (
            abs(float(item["observation.images.front"][:, 112, 112].mean()) - expected)
            < 0.05
        )
    contract = json.loads((output / "meta" / "fastumi.json").read_text())
    assert [row["source_episode_index"] for row in contract["episodes"]] == [0, 2]
    assert contract["rejected"][0]["source_episode_index"] == 1
