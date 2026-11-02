"""验证 VIVE 流式编码策略、逐帧校验和安全提交。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from handumi.scripts import _record_encoding as encoding
from handumi.scripts import umi_capture


def test_auto_h264_probes_hardware_before_software(monkeypatch):
    """自动模式仅尝试 H.264 硬编码，并按实际探测结果回退。"""
    attempted = []
    monkeypatch.setattr(encoding, "_available_hardware_vcodecs", lambda: ["hevc_nvenc", "h264_nvenc"])

    def probe(codec, **kwargs):
        """模拟硬编码失败、软件编码通过。"""
        attempted.append((codec, kwargs))
        return (codec == "h264", "unavailable")

    monkeypatch.setattr(encoding, "_probe_video_encoder", probe)
    selected = encoding._select_video_encoder(
        policy="auto", requested_vcodec=None, width=1920, height=1080, fps=30,
        camera_count=1, requested_threads=None, h264_only=True,
    )
    assert [item[0] for item in attempted] == ["h264_nvenc", "h264"]
    assert selected.vcodec == "h264" and not selected.hardware
    assert all(item[1]["width"] == 1920 and item[1]["height"] == 1080 for item in attempted)


def test_gpu_policy_reports_unavailable(monkeypatch):
    """显式 GPU 模式不回退到软件编码。"""
    monkeypatch.setattr(encoding, "_available_hardware_vcodecs", lambda: ["h264_nvenc"])
    monkeypatch.setattr(encoding, "_probe_video_encoder", lambda *a, **k: (False, "probe failed"))
    with pytest.raises(SystemExit, match="probe failed"):
        encoding._select_video_encoder(
            policy="gpu", requested_vcodec=None, width=64, height=48, fps=30,
            camera_count=1, requested_threads=None, h264_only=True,
        )


def test_hardware_success_and_complete_probe_failure(monkeypatch):
    """优先选择探测成功硬编码；所有候选失败时明确退出。"""
    monkeypatch.setattr(encoding, "_available_hardware_vcodecs", lambda: ["h264_nvenc"])
    monkeypatch.setattr(encoding, "_probe_video_encoder", lambda *a, **k: (True, None))
    kwargs = dict(policy="auto", requested_vcodec=None, width=64, height=48, fps=30,
                  camera_count=1, requested_threads=None, h264_only=True)
    assert encoding._select_video_encoder(**kwargs).vcodec == "h264_nvenc"
    monkeypatch.setattr(encoding, "_probe_video_encoder", lambda *a, **k: (False, "bad codec"))
    with pytest.raises(SystemExit, match="bad codec"):
        encoding._select_video_encoder(**kwargs)


def test_vive_probe_uses_streaming_path_and_nvenc_low_latency(monkeypatch):
    """VIVE 探测走实际流式编码路径；NVENC 配置显式禁用 B 帧。"""
    from lerobot.configs.video import RGBEncoderConfig

    attempts = []
    monkeypatch.setattr(encoding, "_available_hardware_vcodecs", lambda: ["h264_nvenc"])
    monkeypatch.setattr(
        encoding, "_probe_video_encoder", lambda *a, **k: pytest.fail("不能调用旧探测路径"),
    )

    def probe(codec, **kwargs):
        """模拟硬件流式失败、软件流式探测成功。"""
        attempts.append(codec)
        return codec == "h264", "streaming failed"

    monkeypatch.setattr(encoding, "_probe_vive_streaming_encoder", probe)
    selected = encoding._select_video_encoder(
        policy="auto", requested_vcodec=None, width=1920, height=1080, fps=30,
        camera_count=1, requested_threads=None, h264_only=True, streaming_probe=True,
    )
    assert selected.vcodec == "h264" and attempts == ["h264_nvenc", "h264"]
    monkeypatch.setattr(RGBEncoderConfig, "detect_available_encoders", lambda *a: ["h264_nvenc"])
    config = encoding._vive_rgb_encoder_config("h264_nvenc")
    assert config.g == 2 and config.crf == 30
    assert config.get_codec_options(as_strings=True)["bf"] == "0"


@pytest.mark.parametrize("resume", [False, True])
def test_vive_create_and_resume_enable_strict_streaming(tmp_path, monkeypatch, resume):
    """新建和续录均使用相同的流式参数并安装解码帧数检查。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from handumi.dataset import lerobot_compat
    from handumi.scripts import _vive_encoding_process

    dataset = SimpleNamespace(writer=SimpleNamespace(_streaming_encoder=Mock()))
    factory = Mock(return_value=dataset)
    monkeypatch.setattr(LeRobotDataset, "resume" if resume else "create", factory)
    monkeypatch.setattr(lerobot_compat, "preserve_dataset_info", lambda *_: None)
    monkeypatch.setattr(umi_capture, "features", lambda _: {})
    monkeypatch.setattr(umi_capture, "_select_video_encoder", lambda **_: encoding._VideoEncoderSelection("h264", False, 4))
    isolated = Mock(return_value=Mock())
    monkeypatch.setattr(_vive_encoding_process, "ViveProcessEncoder", isolated)
    args = SimpleNamespace(resume=resume, repo_id="local/test", output_dir=tmp_path, encoder="cpu", fps=30)
    assert umi_capture._open_recording_dataset(args, {"resolution": [64, 48]}, {}) is dataset
    kwargs = factory.call_args.kwargs
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_queue_maxsize"] == 30
    assert kwargs["image_writer_threads"] == 0
    assert kwargs["rgb_encoder"].vcodec == "h264"
    assert dataset.writer._streaming_encoder._verify_video_frames
    isolated.assert_called_once()


class FakeEncoder:
    """可注入故障、丢帧和输出结果的编码器。"""

    def __init__(self, path):
        """初始化临时输出与丢帧统计。"""
        self.path = path
        self._dropped_frames = {}
        self.feed_error = None
        self.finish_error = None

    def start_episode(self, **kwargs):
        """接受片段配置。"""

    def feed_frame(self, video_key, image):
        """按测试设置模拟丢帧或编码错误。"""
        if self.feed_error:
            raise self.feed_error

    def finish_episode(self):
        """返回临时文件或抛出预设错误。"""
        if self.finish_error:
            raise self.finish_error
        return {"camera": (self.path, {"count": 2})}

    def cancel_episode(self):
        """模拟停止编码。"""

    def close(self):
        """模拟释放编码器。"""


def _strict(tmp_path):
    """构造含两帧有效输入的严格编码器。"""
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    base = FakeEncoder(path)
    strict = encoding._StrictStreamingEncoder(base, verify_video_frames=True)
    strict.start_episode(["camera"], tmp_path)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    strict.feed_frame("camera", image)
    strict.feed_frame("camera", image)
    return strict, base


def test_actual_video_frame_mismatch_is_not_committed(tmp_path, monkeypatch):
    """实际解码帧数不同会拒绝保存并清理暂存视频。"""
    strict, base = _strict(tmp_path)
    monkeypatch.setattr(encoding, "_decoded_video_frame_count", lambda _, **kwargs: 1)
    with pytest.raises(encoding.StreamingEncodingError, match="decoded video frame count"):
        strict.prepare_episode(2)
    assert not Path(base.path).exists()


def test_queue_drop_and_encoder_thread_failure(tmp_path):
    """队列丢帧与编码线程失败均转换为严格录制错误。"""
    strict, base = _strict(tmp_path)
    base._dropped_frames["camera"] = 1
    with pytest.raises(encoding.StreamingEncodingError, match="dropped"):
        strict.prepare_episode(2)
    base.feed_error = RuntimeError("线程崩溃")
    with pytest.raises(encoding.StreamingEncodingError, match="线程崩溃"):
        strict.feed_frame("camera", np.zeros((4, 4, 3), dtype=np.uint8))


def test_queue_overflow_is_rejected_at_feed(tmp_path, monkeypatch):
    """当前入队导致丢帧时立刻报错，不等待保存阶段。"""
    strict, base = _strict(tmp_path)

    def overflow(video_key, image):
        """模拟底层满队列时只记录丢帧并返回的行为。"""
        base._dropped_frames[video_key] = 1

    monkeypatch.setattr(base, "feed_frame", overflow)
    with pytest.raises(encoding.StreamingEncodingError, match="queue dropped"):
        strict.feed_frame("camera", np.zeros((4, 4, 3), dtype=np.uint8))


def test_finish_failure_does_not_commit_dataset(tmp_path):
    """编码结束失败时不提交数据集，也不计为成功片段。"""
    strict, base = _strict(tmp_path)
    base.finish_error = RuntimeError("编码失败")
    dataset = SimpleNamespace(writer=SimpleNamespace(_streaming_encoder=strict), save_episode=Mock())
    with pytest.raises(encoding.StreamingEncodingError, match="编码失败"):
        umi_capture._save_streaming_episode(dataset, 2)
    dataset.save_episode.assert_not_called()


def test_decode_failure_does_not_commit_dataset(tmp_path, monkeypatch):
    """解码异常必须阻止数据提交，并清理临时视频。"""
    strict, base = _strict(tmp_path)
    dataset = SimpleNamespace(writer=SimpleNamespace(_streaming_encoder=strict), save_episode=Mock())

    def fail(*args, **kwargs):
        """模拟损坏视频的解码失败。"""
        raise encoding.StreamingEncodingError("unable to decode")

    monkeypatch.setattr(encoding, "_decoded_video_frame_count", fail)
    with pytest.raises(encoding.StreamingEncodingError, match="unable to decode"):
        umi_capture._save_streaming_episode(dataset, 2)
    dataset.save_episode.assert_not_called()
    assert not base.path.exists()


@pytest.mark.parametrize("codec,depth,expected", [
    ("h264_nvenc", False, 4), ("h264", False, None),
    ("h264_nvenc", True, None), ("h264_qsv", False, None),
])
def test_decode_policy_uses_encoder_and_stream_type(tmp_path, monkeypatch, codec, depth, expected):
    """仅 NVENC RGB 视频启用有界帧并行，深度及其他编码保持原策略。"""
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    strict = encoding._StrictStreamingEncoder(FakeEncoder(path), verify_video_frames=True, rgb_vcodec=codec)
    strict.start_episode(["camera"], tmp_path, depth_video_keys=["camera"] if depth else [])
    for _ in range(2):
        strict.feed_frame("camera", np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr(encoding.os, "cpu_count", lambda: 12)
    decode = Mock(return_value=2)
    monkeypatch.setattr(encoding, "_decoded_video_frame_count", decode)
    strict.prepare_episode(2)
    decode.assert_called_once_with(path, frame_threads=expected)


def test_frame_decoder_config_and_drains_frames(monkeypatch, tmp_path):
    """配置帧线程并迭代全部输出，包括解码器收尾帧。"""
    import av

    stream = SimpleNamespace(thread_type="SLICE", codec_context=SimpleNamespace(thread_count=0))
    container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=Mock(return_value=iter([1, 2, 3])))
    context = Mock()
    context.__enter__ = Mock(return_value=container)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(av, "open", lambda _: context)
    assert encoding._decoded_video_frame_count(tmp_path / "video.mp4", frame_threads=4) == 3
    assert stream.thread_type == "FRAME" and stream.codec_context.thread_count == 4
    container.decode.assert_called_once_with(stream)


def test_frame_decoder_counts_real_b_frame_tail(tmp_path):
    """真实含 B 帧视频经跨帧并行解码时仍须计入结尾延迟帧。"""
    import av

    path = tmp_path / "b-frames.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.max_b_frames = 2
        for index in range(12):
            image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 12 and any(frame.pict_type == 3 for frame in frames)
    assert encoding._decoded_video_frame_count(path, frame_threads=4) == 12


def test_real_invalid_video_reports_decode_error(tmp_path):
    """无法打开或解码的视频转换为明确的流式编码异常。"""
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(encoding.StreamingEncodingError, match="unable to decode"):
        encoding._decoded_video_frame_count(path, frame_threads=4)


@pytest.mark.parametrize("cores,expected", [(None, 1), (1, 1), (2, 1), (3, 2), (32, 4)])
def test_decoder_thread_budget(monkeypatch, cores, expected):
    """校验线程数限制在 1–4 并为采集预留处理器。"""
    monkeypatch.setattr(encoding.os, "cpu_count", lambda: cores)
    assert encoding._recommended_decoder_threads() == expected


def test_measured_fps_and_save_stage_timings(monkeypatch):
    """帧率来自实际接收间隔，保存总耗时包含编码准备和提交。"""
    moments = iter([10.0, 10.4, 10.7])
    monkeypatch.setattr(umi_capture.time, "monotonic", lambda: next(moments))
    prepare = Mock(return_value={"encoder_flush_s": 0.1, "video_verify_s": 0.25})
    monkeypatch.setattr(umi_capture, "_prepare_streaming_episode", prepare)
    dataset = SimpleNamespace(save_episode=Mock())
    durations = umi_capture._save_streaming_episode(dataset, 3)
    metrics = umi_capture._capture_metrics(
        [1.0, 1.05, 1.1], [2.0, 2.04, 2.08], [0.001, 0.002, 0.003], durations, 0.9,
    )
    assert metrics["capture_fps"] == pytest.approx(20)
    assert metrics["camera_interval_ms_p95"] == pytest.approx(40)
    assert metrics["add_frame_ms_p95"] == pytest.approx(2.9)
    assert metrics["encode_finish_s"] == pytest.approx(0.4)
    assert metrics["encoder_flush_s"] == pytest.approx(0.1)
    assert metrics["video_verify_s"] == pytest.approx(0.25)
    assert metrics["commit_s"] == pytest.approx(0.3)
    assert metrics["save_total_s"] == pytest.approx(0.9)
    assert metrics["save_total_s"] >= metrics["encode_finish_s"] + metrics["commit_s"]
    prepare.assert_called_once_with(dataset, 3)
    dataset.save_episode.assert_called_once()
