"""验证新版 LeRobot 的严格视频录制与扩展元数据保留。"""

import json

import numpy as np
import pytest
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from handumi.dataset.lerobot_compat import preserve_dataset_info
from handumi.scripts.record import (
    StreamingEncodingError,
    _install_strict_streaming_encoder,
    _prepare_streaming_episode,
)


def test_streaming_resume_preserves_custom_metadata_and_frame_counts(tmp_path):
    """实际编码、丢帧取消与续录期间，自定义字段和总数保持正确。"""
    root = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["value"]},
        "observation.images.test": {
            "dtype": "video", "shape": (32, 32, 3),
            "names": ["height", "width", "channel"],
        },
    }
    config = RGBEncoderConfig(vcodec="h264")
    dataset = LeRobotDataset.create(
        "local/streaming", fps=10, features=features, root=root,
        rgb_encoder=config, streaming_encoding=True, encoder_queue_maxsize=30,
    )
    preserve_dataset_info(dataset, {"handumi": {"source": "测试"}, "vendor_extra": {"id": 42}})
    _install_strict_streaming_encoder(dataset)

    def append_frames(target, count):
        """写入指定数量的合成观测。"""
        for index in range(count):
            target.add_frame({
                "observation.state": np.array([index], dtype=np.float32),
                "observation.images.test": np.full((32, 32, 3), index * 20, dtype=np.uint8),
                "task": "test",
            })

    def assert_info(episodes, frames):
        """检查扩展字段与更新后的标准计数。"""
        info = json.loads((root / "meta/info.json").read_text())
        assert info["handumi"] == {"source": "测试"}
        assert info["vendor_extra"] == {"id": 42}
        assert (info["total_episodes"], info["total_frames"]) == (episodes, frames)

    append_frames(dataset, 3)
    _prepare_streaming_episode(dataset, 3)
    dataset.save_episode()
    assert_info(1, 3)
    append_frames(dataset, 2)
    with pytest.raises(StreamingEncodingError, match="frame counts"):
        _prepare_streaming_episode(dataset, 3)
    dataset.clear_episode_buffer()
    assert_info(1, 3)
    dataset.finalize()
    assert_info(1, 3)

    dataset = LeRobotDataset.resume(
        "local/streaming", root=root, rgb_encoder=config,
        streaming_encoding=True, encoder_queue_maxsize=30,
    )
    preserve_dataset_info(dataset)
    _install_strict_streaming_encoder(dataset)
    append_frames(dataset, 2)
    _prepare_streaming_episode(dataset, 2)
    dataset.save_episode()
    assert_info(2, 5)
    dataset.finalize()
    assert_info(2, 5)
    loaded = LeRobotDataset("local/streaming", root=root)
    assert loaded.num_episodes == 2
    assert len(loaded) == 5
    assert loaded[4]["observation.images.test"].shape == (3, 32, 32)
