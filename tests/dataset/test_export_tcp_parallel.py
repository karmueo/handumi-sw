"""验证 TCP 导出并行、源数据检查及编码队列的可靠性。"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from handumi.dataset.raw import HANDUMI_STATE_SEMANTICS, HANDUMI_TRACKING_SCHEMA
from handumi.scripts import _export_tcp_runtime as runtime
from handumi.scripts import export_tcp


@pytest.fixture
def source_dataset(tmp_path):
    """构造跨两个数据文件和一个共享视频的四个回合。"""
    root = tmp_path / "source"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    videos = root / "videos" / "observation.images.right_wrist" / "chunk-000"
    videos.mkdir(parents=True)
    video = cv2.VideoWriter(
        str(videos / "file-000.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48)
    )
    assert video.isOpened()
    lengths = [1, 3, 3, 3]
    prefix = "videos/observation.images.right_wrist/"
    episodes, samples = [], []
    offset = 0
    for episode, length in enumerate(lengths):
        task = "alpha" if episode < 2 else "beta"
        episodes.append(
            {
                "episode_index": episode,
                "length": length,
                "tasks": [task],
                prefix + "chunk_index": 0,
                prefix + "file_index": 0,
                prefix + "from_timestamp": offset / 30,
                prefix + "to_timestamp": (offset + length) / 30,
            }
        )
        for frame in range(length):
            number = offset + frame + 1
            video.write(
                np.full((48, 64, 3), (episode * 40, frame * 50, 100), dtype=np.uint8)
            )
            row = {
                "episode_index": episode,
                "frame_index": frame,
                "observation.state": [0.0] * 16,
                "observation.valid": [1] * 8,
                "observation.tracking.right_tcp_pose": [
                    episode + frame * 0.1,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "observation.camera.right_wrist.sample_time_ns": number * 100,
                "observation.camera.right_wrist.sequence": number,
            }
            for side in ("left", "right"):
                row[f"observation.gripper.{side}.normalized"] = 0.25 + frame * 0.1
                row[f"observation.gripper.{side}.valid"] = 1
                row[f"observation.gripper.{side}.sample_time_ns"] = number * 100
                row[f"observation.gripper.{side}.sequence"] = number
            samples.append(row)
        offset += length
    video.release()
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(samples[:4]), root / "data/chunk-000/file-000.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist(samples[4:]), root / "data/chunk-000/file-001.parquet"
    )
    info = {
        "fps": 30,
        "total_episodes": len(episodes),
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "handumi": {
            "recording_device": "vive",
            "active_sides": ["right"],
            "capture_schema": "synchronized_sources_v2",
            "tracking_schema": HANDUMI_TRACKING_SCHEMA,
            "state_semantics": HANDUMI_STATE_SEMANTICS,
            "tcp_pose": {
                "frame": "table",
                "feature": "observation.tracking.right_tcp_pose",
            },
            "sources": {
                "gripper": {
                    "left": {
                        "enabled": False,
                        "normalized_available": False,
                        "width_available": False,
                    },
                    "right": {
                        "enabled": True,
                        "normalized_available": True,
                        "width_available": False,
                        "camera": "right_wrist",
                    },
                }
            },
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    quality = {
        "dataset": "local/source",
        "episodes": [
            {
                "episode_index": i,
                "frame_count": length,
                "status": "rejected" if i == 1 else "accepted",
                "findings": [],
            }
            for i, length in enumerate(lengths)
        ],
    }
    (root / "meta/handumi_quality.json").write_text(json.dumps(quality))
    return root


def _export_command(source: Path, output: Path, workers: int) -> list[str]:
    """生成 CPU 导出命令，子进程可真实启用 spawn。"""
    return [
        sys.executable,
        "-m",
        "handumi.scripts.export_tcp",
        str(source),
        "--output",
        str(output),
        "--repo-id",
        "local/result",
        "--encoder",
        "cpu",
        "--workers",
        str(workers),
        "--resize-mode",
        "stretch",
        "--image-size",
        "240",
        "134",
    ]


def test_parallel_export_preserves_data_and_reindexed_stats(source_dataset, tmp_path):
    """单帧及跳过的回合跨分片导出后，状态、帧序和统计一致。"""
    outputs = [tmp_path / "single", tmp_path / "parallel"]
    for root, workers in zip(outputs, (1, 3), strict=True):
        result = subprocess.run(
            _export_command(source_dataset, root, workers),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    tables = [
        pq.read_table(sorted((root / "data").glob("chunk-*/*.parquet")))
        for root in outputs
    ]
    for key in (
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
        "observation.state",
        "action",
    ):
        assert tables[0][key].to_pylist() == tables[1][key].to_pylist()
    assert tables[1]["episode_index"].to_pylist() == [0, 1, 1, 1, 2, 2, 2]
    for root in outputs:
        contract = json.loads((root / "meta/fastumi.json").read_text())
        assert [row["source_episode_index"] for row in contract["episodes"]] == [
            0,
            2,
            3,
        ]
        assert contract["rejected"][0]["source_episode_index"] == 1
        stats = json.loads((root / "meta/stats.json").read_text())
        assert np.asarray(stats["observation.images.front"]["mean"]).shape == (3, 1, 1)
        assert np.isfinite(np.asarray(stats["observation.images.front"]["mean"])).all()
        for key in ("episode_index", "index", "task_index"):
            actual = np.asarray(tables[1][key].to_pylist())
            for stat, expected in (
                ("min", actual.min()),
                ("max", actual.max()),
                ("mean", actual.mean()),
                ("std", actual.std()),
            ):
                np.testing.assert_allclose(stats[key][stat], [expected], atol=1e-8)


def test_reader_rejects_mismatched_camera_sequence(source_dataset):
    """快速读取继续验证活动夹爪与其来源相机属于同一帧。"""
    path = source_dataset / "data/chunk-000/file-001.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["observation.camera.right_wrist.sequence"] += 100
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = runtime.EpisodeReader(
        source_dataset, json.loads((source_dataset / "meta/info.json").read_text())
    )
    with pytest.raises(ValueError, match="不是同一帧"):
        reader.read(2, 3)


def test_reader_rejects_missing_or_wrong_frame_indices(source_dataset):
    """源帧缺失或帧号重排均不得悄悄导出。"""
    path = source_dataset / "data/chunk-000/file-001.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["frame_index"] = 2
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = runtime.EpisodeReader(
        source_dataset, json.loads((source_dataset / "meta/info.json").read_text())
    )
    with pytest.raises(ValueError, match="帧编号"):
        reader.read(2, 3)
    with pytest.raises(ValueError, match="帧数"):
        reader.read(3, 4)


def test_reader_rejects_non_numeric_raw_state(source_dataset):
    """原始状态仍须能转换为 float32，不能只检查向量形状。"""
    path = source_dataset / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["observation.state"] = ["invalid"] * 16
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = runtime.EpisodeReader(
        source_dataset, json.loads((source_dataset / "meta/info.json").read_text())
    )
    with pytest.raises(ValueError, match="could not convert string to float"):
        reader.read(0, 1)


def test_offline_queue_waits_and_delivers_every_frame():
    """慢消费者导致队列满时，生产者等待并完整交付所有帧。"""
    encoder = runtime.OfflineStreamingEncoder(
        fps=30, rgb_encoder=runtime.encoder_config("h264"), queue_maxsize=1
    )
    frames = queue.Queue(maxsize=1)
    received = []

    def consume():
        """模拟慢编码线程。"""
        for _ in range(5):
            time.sleep(0.12)
            received.append(int(frames.get()[0, 0, 0]))

    thread = threading.Thread(target=consume)
    encoder._episode_active = True
    encoder._first_frames = {}
    encoder._threads = {"video": thread}
    encoder._frame_queues = {"video": frames}
    encoder._result_queues = {"video": queue.Queue()}
    thread.start()
    for value in range(5):
        encoder.feed_frame("video", np.full((4, 4, 3), value, dtype=np.uint8))
    thread.join(timeout=3)
    assert received == list(range(5))


def test_offline_queue_reports_dead_encoder():
    """编码线程退出时即便队列已满也应及时失败。"""
    encoder = runtime.OfflineStreamingEncoder(
        fps=30, rgb_encoder=runtime.encoder_config("h264"), queue_maxsize=1
    )
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    encoder._episode_active = True
    encoder._first_frames = {}
    encoder._threads = {"video": thread}
    encoder._frame_queues = {"video": queue.Queue(maxsize=1)}
    encoder._frame_queues["video"].put(np.zeros((4, 4, 3), dtype=np.uint8))
    encoder._result_queues = {"video": queue.Queue()}
    with pytest.raises(RuntimeError, match="编码线程提前退出"):
        encoder.feed_frame("video", np.zeros((4, 4, 3), dtype=np.uint8))


def test_encoder_auto_fallback_and_gpu_failure(monkeypatch):
    """硬件探测失败时自动策略回退，显式 GPU 策略失败。"""
    import handumi.scripts._record_encoding as encoding

    monkeypatch.setattr(encoding, "_available_hardware_vcodecs", lambda: ["h264_nvenc"])
    monkeypatch.setattr(
        runtime,
        "probe_encoder",
        lambda *args: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    assert runtime.select_encoder("auto", (240, 134), 4) == "h264"
    with pytest.raises(RuntimeError, match="GPU 编码探测失败"):
        runtime.select_encoder("gpu", (240, 134), 4)


def test_shard_aggregation_uses_source_order_after_out_of_order_completion(
    monkeypatch, tmp_path
):
    """子进程乱序上报完成时，聚合仍按源回合分片顺序进行。"""
    import lerobot.datasets.aggregate as aggregate

    events = queue.Queue()
    for event in (("episode", 2, 1, {}), ("done",), ("episode", 0, 1, {}), ("done",)):
        events.put(event)

    class EventQueue:
        """模拟已预先排列结果的跨进程队列。"""

        def get(self, timeout):
            """返回下一条预排结果。"""
            return events.get(timeout=timeout)

        def close(self):
            """模拟关闭队列。"""

        def join_thread(self):
            """模拟等待队列线程。"""

    class Process:
        """模拟成功结束的工作进程。"""

        pid = 1
        exitcode = 0

        def __init__(self, **kwargs):
            """接收进程参数。"""

        def start(self):
            """模拟进程启动。"""

        def join(self):
            """模拟进程回收。"""

        def is_alive(self):
            """返回已退出状态。"""
            return False

    class Context:
        """提供可预测的进程和结果队列。"""

        def Process(self, **kwargs):
            """返回模拟工作进程。"""
            return Process(**kwargs)

        def Queue(self):
            """返回预排结果队列。"""
            return EventQueue()

    calls = []
    monkeypatch.setattr(export_tcp.mp, "get_context", lambda _: Context())
    monkeypatch.setattr(
        aggregate, "aggregate_datasets", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(export_tcp, "_repair_aggregated_stats", lambda _: None)
    shards = [
        {"indices": [0], "repo_id": "local/first", "root": tmp_path / "first"},
        {"indices": [2], "repo_id": "local/second", "root": tmp_path / "second"},
    ]
    export_tcp._export_shards(shards, tmp_path / "output", "local/output", 2)
    assert calls[0]["repo_ids"] == ["local/first", "local/second"]
    assert calls[0]["roots"] == [tmp_path / "first", tmp_path / "second"]
    assert calls[0]["concatenate_videos"] is False


def test_parallel_worker_failure_removes_partial_output(source_dataset, tmp_path):
    """某工作进程数据错误时，全部分片停止且不发布残缺结果。"""
    path = source_dataset / "data/chunk-000/file-001.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["observation.camera.right_wrist.sequence"] += 100
    pq.write_table(pa.Table.from_pylist(rows), path)
    output = tmp_path / "failure"
    process = subprocess.Popen(
        _export_command(source_dataset, output, 2),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=60)
        assert process.returncode != 0
        assert "导出" in stdout and ("同一帧" in stdout or "子进程异常" in stdout)
        assert not output.exists()
        assert not list(tmp_path.glob(".failure-*"))
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def test_gpu_export_240_by_134_when_available(source_dataset, tmp_path):
    """硬件可用时真实编码、聚合和解码全部回合。"""
    try:
        runtime.probe_encoder("h264_nvenc", (240, 134), 2)
    except Exception as exc:
        pytest.skip(f"NVENC 不可用: {exc}")
    output = tmp_path / "gpu"
    command = _export_command(source_dataset, output, 2)
    command[command.index("cpu")] = "gpu"
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.images.front"]["shape"] == [134, 240, 3]
    import av

    frames = []
    for path in sorted((output / "videos").rglob("*.mp4")):
        with av.open(str(path)) as container:
            frames.extend(
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            )
    assert len(frames) == 7
    expected = [[100, 0, 0]] + [
        [100, frame * 50, episode * 40] for episode in (2, 3) for frame in range(3)
    ]
    for frame, color in zip(frames, expected, strict=True):
        np.testing.assert_allclose(frame[67, 120], color, atol=15)


@pytest.mark.parametrize(
    ("encoder", "workers"),
    [("bad", None), ("cpu", 0), ("auto", -1), ("gpu", True), ("cpu", 1.5)],
)
def test_export_rejects_bad_performance_options_before_io(
    tmp_path, monkeypatch, encoder, workers
):
    """性能参数无效时不读取源数据或创建输出路径。"""
    monkeypatch.setattr(
        export_tcp,
        "resolve_dataset_selection",
        lambda *_: pytest.fail("意外访问源数据"),
    )
    with pytest.raises(ValueError):
        export_tcp.export(
            tmp_path / "source",
            tmp_path / "output",
            "a/b",
            encoder=encoder,
            workers=workers,
        )
    assert not (tmp_path / "output").exists()
