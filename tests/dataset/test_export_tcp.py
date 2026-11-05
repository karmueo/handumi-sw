"""验证 TCP 导出的坐标变换、图像处理和质量筛选。"""

import json
import sys
from types import SimpleNamespace

import cv2
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


@pytest.mark.parametrize("shape", [(108, 192), (191, 107), (13, 13), (1, 500)])
def test_letterbox_default_matches_previous_output(shape):
    """默认参数与原实现的缩放、取整和像素内容一致。"""
    image = np.random.default_rng(123).integers(0, 256, (*shape, 3), dtype=np.uint8)
    height, width = shape
    scale = min(224 / width, 224 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    expected = np.zeros((224, 224, 3), dtype=np.uint8)
    x, y = (224 - size[0]) // 2, (224 - size[1]) // 2
    expected[y : y + size[1], x : x + size[0]] = resized
    np.testing.assert_array_equal(letterbox(image), expected)


@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_letterbox_centers_odd_border_remainder(shape):
    """横图和竖图的奇数补边余量分别位于底部或右侧。"""
    image = np.full((*shape, 3), 255, dtype=np.uint8)
    output = export_tcp.resize_image(image, (8, 8), "letterbox")
    if shape[0] < shape[1]:
        assert not output[0].any()
        assert output[1:6].min() == 255
        assert not output[6:].any()
    else:
        assert not output[:, 0].any()
        assert output[:, 1:6].min() == 255
        assert not output[:, 6:].any()


@pytest.mark.parametrize("vertical", [False, True])
def test_center_crop_removes_odd_remainder_from_bottom_or_right(vertical):
    """中心裁剪保留居中像素，多余的一行或一列放在底部或右侧。"""
    image = np.broadcast_to(np.arange(7, dtype=np.uint8)[None, :, None], (4, 7, 3))
    if vertical:
        image = image.transpose(1, 0, 2)
    output = export_tcp.resize_image(image, (4, 4), "center-crop")
    expected = image[1:5, :] if vertical else image[:, 1:5]
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("vertical", [False, True])
def test_center_crop_scales_and_discards_outer_bands(vertical):
    """中心裁剪在缩放后丢弃横图或竖图两端的内容。"""
    image = np.full((4, 8, 3), (50, 100, 200), dtype=np.uint8)
    image[:, :2] = (255, 0, 0)
    image[:, -2:] = (0, 255, 0)
    if vertical:
        image = image.transpose(1, 0, 2)
    output = export_tcp.resize_image(image, (6, 6), "center-crop")
    np.testing.assert_array_equal(output, np.full((6, 6, 3), (50, 100, 200)))


@pytest.mark.parametrize("shape", [(4, 8), (8, 4), (4, 4)])
def test_stretch_preserves_edges_at_non_square_size(shape):
    """直接拉伸保留横图、竖图和方图边缘以及指定目标宽高。"""
    image = np.full((*shape, 3), (32, 128, 240), dtype=np.uint8)
    image[:, 0] = (200, 100, 50)
    image[:, -1] = (50, 100, 200)
    output = export_tcp.resize_image(image, (16, 12), "stretch")
    assert output.shape == (12, 16, 3)
    np.testing.assert_array_equal(output[6, 0], image[0, 0])
    np.testing.assert_array_equal(output[6, -1], image[0, -1])


@pytest.mark.parametrize(
    (
        "extra_args",
        "expected_size",
        "expected_mode",
        "expected_encoder",
        "expected_workers",
    ),
    [
        ([], (224, 224), "letterbox", "cpu", None),
        (
            ["--image-size", "320", "240", "--resize-mode", "center-crop"],
            (320, 240),
            "center-crop",
            "cpu",
            None,
        ),
        (["--encoder", "auto", "--workers", "2"], (224, 224), "letterbox", "auto", 2),
    ],
)
def test_cli_passes_image_options(
    monkeypatch,
    extra_args,
    expected_size,
    expected_mode,
    expected_encoder,
    expected_workers,
):
    """命令行默认值和自定义值按宽、高顺序传给导出接口。"""
    calls = []
    monkeypatch.setattr(
        export_tcp, "export", lambda *args, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export-tcp", "source", "--output", "out", "--repo-id", "a/b", *extra_args],
    )
    export_tcp.main()
    assert calls == [
        {
            "image_size": expected_size,
            "resize_mode": expected_mode,
            "encoder": expected_encoder,
            "workers": expected_workers,
        }
    ]


@pytest.mark.parametrize(
    ("cpu_count", "episode_count", "options", "expected_workers"),
    [
        (16, 12, {}, 8),
        (3, 12, {}, 3),
        (16, 3, {}, 3),
        (None, 12, {}, 1),
        (3, 12, {"workers": 11}, 11),
    ],
)
def test_export_default_cpu_and_worker_limit(
    tmp_path, monkeypatch, cpu_count, episode_count, options, expected_workers
):
    """验证默认 CPU 和进程数限制，并保留显式进程数的行为。"""
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    rows = [{"episode_index": index, "length": 2} for index in range(episode_count)]
    info = {
        "fps": 30,
        "total_episodes": episode_count,
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
    calls = []
    monkeypatch.setattr(export_tcp.os, "cpu_count", lambda: cpu_count)
    monkeypatch.setattr(export_tcp, "load_info", lambda _: info)
    monkeypatch.setattr(
        export_tcp.pq,
        "read_table",
        lambda *args, **kwargs: SimpleNamespace(to_pylist=lambda: rows),
    )
    monkeypatch.setattr(
        export_tcp,
        "quality_selection",
        lambda *args: (list(range(episode_count)), []),
    )
    monkeypatch.setattr(
        export_tcp,
        "select_encoder",
        lambda *args: calls.append(args) or "h264",
    )

    def check_shards(shards, *args):
        """验证配置已落实到分片，无需启动进程或写入实际数据。"""
        assert len(shards) == expected_workers
        assert [index for shard in shards for index in shard["indices"]] == list(
            range(episode_count)
        )
        assert all(shard["codec"] == "h264" for shard in shards)
        raise RuntimeError("已验证配置")

    monkeypatch.setattr(export_tcp, "_export_shards", check_shards)
    with pytest.raises(RuntimeError, match="已验证配置"):
        export_tcp.export(source, output, "local/output", **options)
    assert calls == [("cpu", (224, 224), expected_workers)]
    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--resize-mode", "bad"],
        ["--image-size", "224"],
        ["--image-size", "wide", "224"],
        ["--image-size", "0", "224"],
        ["--image-size", "-2", "224"],
        ["--image-size", "225", "224"],
        ["--image-size", "224", "225"],
        ["--encoder", "invalid"],
        ["--workers", "0"],
        ["--workers", "-1"],
        ["--workers", "x"],
    ],
)
def test_cli_rejects_invalid_image_options(monkeypatch, extra_args):
    """参数不合法时命令行以错误状态退出且不开始导出。"""
    calls = []
    monkeypatch.setattr(
        export_tcp, "export", lambda *args, **kwargs: calls.append(args)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export-tcp", "source", "--output", "out", "--repo-id", "a/b", *extra_args],
    )
    with pytest.raises(SystemExit) as exc:
        export_tcp.main()
    assert exc.value.code == 2
    assert not calls


@pytest.mark.parametrize(
    ("image_size", "resize_mode"),
    [
        ((0, 224), "letterbox"),
        ((224, 225), "letterbox"),
        ((224.0, 224), "letterbox"),
        ((True, 224), "letterbox"),
        ((224,), "letterbox"),
        (None, "letterbox"),
        ((224, 224), "bad"),
    ],
)
def test_export_rejects_invalid_options_before_io(
    tmp_path, monkeypatch, image_size, resize_mode
):
    """Python 导出入口在访问源数据和创建输出目录之前拒绝非法选项。"""
    source = tmp_path / "missing-source"
    output = tmp_path / "missing-parent" / "output"

    def fail_if_read(*args, **kwargs):
        """在参数检查前尝试读取源数据时使测试失败。"""
        pytest.fail("非法参数触发了源数据读取")

    monkeypatch.setattr(export_tcp, "resolve_dataset_selection", fail_if_read)
    with pytest.raises(ValueError):
        export_tcp.export(
            source,
            output,
            "local/output",
            image_size=image_size,
            resize_mode=resize_mode,
        )
    assert not output.parent.exists()


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
        export_tcp.pq,
        "read_table",
        lambda *args, **kwargs: SimpleNamespace(to_pylist=lambda: [row]),
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
    monkeypatch.setattr(
        export_tcp,
        "EpisodeReader",
        lambda *args: SimpleNamespace(read=lambda *args: (poses, [0.8, 0.7])),
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
    writer.writer = SimpleNamespace()
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
        export_tcp.export(source, output, "local/output", encoder="cpu", workers=1)
    assert writer._is_finalized
    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize(
    ("image_size", "resize_mode", "preprocessing"),
    [
        ((224, 224), "letterbox", "letterbox_inter_area_black_border"),
        ((80, 40), "letterbox", "letterbox_inter_area_black_border"),
        ((80, 40), "center-crop", "center_crop_inter_area"),
        ((80, 40), "stretch", "stretch_inter_area"),
    ],
)
def test_export_shared_video_and_skipped_episode(
    tmp_path, monkeypatch, image_size, resize_mode, preprocessing
):
    """真实写入器保留两个回合末帧，视频读取跨过被拒绝回合。"""
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    video_dir = source / "videos" / "observation.images.right_wrist" / "chunk-000"
    video_dir.mkdir(parents=True)
    path = video_dir / "file-000.mp4"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert video.isOpened()
    for intensity in [20, 40, 60, 80, 120, 160]:
        frame = np.full(
            (48, 64, 3), (intensity // 4, intensity // 2, intensity), dtype=np.uint8
        )
        frame[:6, :] = (0, 220, 0)
        frame[-6:, :] = (220, 0, 0)
        video.write(frame)
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
    monkeypatch.setattr(
        export_tcp,
        "EpisodeReader",
        lambda *args: SimpleNamespace(
            read=lambda index, length: (
                raw(episode=index).signals["observation.tracking.right_tcp_pose"],
                raw(episode=index).signals["observation.gripper.right.normalized"],
            )
        ),
    )
    output = tmp_path / "output"
    export_tcp.export(
        source,
        output,
        "local/output",
        image_size=image_size,
        resize_mode=resize_mode,
        encoder="cpu",
        workers=1,
    )

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
        intensity = [20, 40, 120, 160][index]
        image = item["observation.images.front"]
        assert tuple(image.shape) == (3, image_size[1], image_size[0])
        expected_rgb = np.array([intensity, intensity // 2, intensity // 4]) / 255
        np.testing.assert_allclose(
            image[:, image_size[1] // 2, image_size[0] // 2], expected_rgb, atol=0.05
        )
        if image_size == (224, 224):
            # 方形补黑位于顶部和底部。
            np.testing.assert_allclose(image[:, 2, 112], 0, atol=0.06)
            np.testing.assert_allclose(image[:, -3, 112], 0, atol=0.06)
        else:
            # 80×40 的补黑位于左右两侧；裁剪和拉伸都填满画面。
            expected_side = 0 if resize_mode == "letterbox" else expected_rgb
            np.testing.assert_allclose(image[:, 20, 2], expected_side, atol=0.06)
            np.testing.assert_allclose(image[:, 20, -3], expected_side, atol=0.06)
            # 中心裁剪去掉上下色带，补黑和拉伸均保留色带。
            expected_top = (
                expected_rgb if resize_mode == "center-crop" else [0, 220 / 255, 0]
            )
            expected_bottom = (
                expected_rgb if resize_mode == "center-crop" else [0, 0, 220 / 255]
            )
            np.testing.assert_allclose(image[:, 2, 40], expected_top, atol=0.08)
            np.testing.assert_allclose(image[:, -3, 40], expected_bottom, atol=0.08)
    contract = json.loads((output / "meta" / "fastumi.json").read_text())
    output_info = json.loads((output / "meta" / "info.json").read_text())
    assert output_info["features"]["observation.images.front"]["shape"] == [
        image_size[1],
        image_size[0],
        3,
    ]
    assert contract["image_size"] == [image_size[1], image_size[0]]
    assert contract["image_preprocessing"] == preprocessing
    assert [row["source_episode_index"] for row in contract["episodes"]] == [0, 2]
    assert contract["rejected"][0]["source_episode_index"] == 1
