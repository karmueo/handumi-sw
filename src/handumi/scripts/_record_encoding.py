"""共享录制编码器探测和流式视频完整性检查。"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol, cast

import numpy as np

log = logging.getLogger("handumi.record")

_SOFTWARE_VIDEO_CODEC = "h264"
_HARDWARE_VIDEO_CODECS = (
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
    "h264_vaapi",
    "h264_qsv",
)
_VALID_VIDEO_CODECS = {
    _SOFTWARE_VIDEO_CODEC,
    "hevc",
    "libsvtav1",
    *_HARDWARE_VIDEO_CODECS,
}


class StreamingEncodingError(RuntimeError):
    """流式编码无法产生与观测逐帧对应的视频。"""


@dataclass(frozen=True)
class _VideoEncoderSelection:
    """记录探测通过的编码器、硬件类型与线程预算。"""
    vcodec: str
    hardware: bool
    threads: int | None


class _StreamingEncoderProtocol(Protocol):
    """约束 LeRobot 流式编码器需要实现的接口。"""
    def start_episode(
        self, video_keys: list[str], temp_dir: Path, depth_video_keys: list[str] | None = None,
    ) -> None:
        """开始片段，按视频键创建临时视频和编码线程。"""
        ...

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """将一帧送入指定视频的编码队列。"""
        ...

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """完成编码并返回各视频文件与统计信息。"""
        ...

    def cancel_episode(self) -> None:
        """取消当前片段并删除临时结果。"""
        ...

    def close(self) -> None:
        """释放编码器资源。"""
        ...


class _StrictStreamingEncoder:
    """先验证编码结果再提交片段，避免编码失败后留下部分数据行。"""

    def __init__(self, encoder: _StreamingEncoderProtocol, *, verify_video_frames: bool = False,
                 rgb_vcodec: str | None = None) -> None:
        """包装编码器；按需在提交数据前检查实际视频解码帧数。"""
        self._encoder = encoder
        self._verify_video_frames = verify_video_frames
        self._rgb_vcodec = rgb_vcodec
        self._depth_video_keys: frozenset[str] = frozenset()
        self._video_keys: tuple[str, ...] = ()
        self._frame_counts: dict[str, int] = {}
        self._prepared_results: dict[str, tuple[Path, dict | None]] | None = None

    def start_episode(
        self, video_keys: list[str], temp_dir: Path, depth_video_keys: list[str] | None = None,
    ) -> None:
        """开始编码新片段并初始化每路帧数。"""
        self._prepared_results = None
        self._video_keys = tuple(video_keys)
        self._depth_video_keys = frozenset(depth_video_keys or ())
        self._frame_counts = dict.fromkeys(self._video_keys, 0)
        self._encoder.start_episode(
            video_keys=video_keys, temp_dir=temp_dir, depth_video_keys=depth_video_keys,
        )

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """向编码器送帧，检测队列丢帧与编码线程故障。"""
        before = self._drop_count(video_key)
        try:
            self._encoder.feed_frame(video_key, image)
        except RuntimeError as exc:
            raise StreamingEncodingError(str(exc)) from exc
        after = self._drop_count(video_key)
        if after > before:
            raise StreamingEncodingError(
                f"encoder queue dropped a frame for {video_key}"
            )
        self._frame_counts[video_key] = self._frame_counts.get(video_key, 0) + 1

    def prepare_episode(self, expected_frames: int) -> dict[str, float]:
        """结束编码并检查接收帧、文件及可选的实际解码帧数。"""
        if self._prepared_results is not None:
            return {"encoder_flush_s": 0.0, "video_verify_s": 0.0}
        dropped = {
            key: self._drop_count(key)
            for key in self._video_keys
            if self._drop_count(key) > 0
        }
        if dropped:
            raise StreamingEncodingError(f"encoder dropped video frames: {dropped}")
        mismatched = {
            key: count
            for key, count in self._frame_counts.items()
            if count != expected_frames
        }
        if mismatched:
            raise StreamingEncodingError(
                f"streamed frame counts do not match {expected_frames}: {mismatched}"
            )

        flush_start = time.monotonic()
        try:
            results = self._encoder.finish_episode()
        except Exception as exc:
            raise StreamingEncodingError(f"encoder failed while finishing: {exc}") from exc

        flush_s = time.monotonic() - flush_start
        verify_start = time.monotonic()
        try:
            for key in self._video_keys:
                if key not in results:
                    raise StreamingEncodingError(f"encoder returned no video for {key}")
                path, stats = results[key]
                path = Path(path)
                if not path.is_file() or path.stat().st_size <= 0:
                    raise StreamingEncodingError(f"encoder produced an empty video for {key}")
                if expected_frames >= 2 and stats is None:
                    raise StreamingEncodingError(f"encoder returned no statistics for {key}")
                if self._verify_video_frames:
                    frame_threads = (
                        _recommended_decoder_threads()
                        if self._rgb_vcodec == "h264_nvenc" and key not in self._depth_video_keys
                        else None
                    )
                    actual = _decoded_video_frame_count(path, frame_threads=frame_threads)
                    if actual != expected_frames:
                        raise StreamingEncodingError(
                            f"decoded video frame count for {key} is {actual}, expected {expected_frames}"
                        )
            self._prepared_results = results
            return {"encoder_flush_s": flush_s, "video_verify_s": time.monotonic() - verify_start}
        except Exception:
            self._remove_results(results)
            raise

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """返回已通过检查的视频给数据集写入器。"""
        if self._prepared_results is None:
            raise StreamingEncodingError(
                "streaming video was not prepared before saving the episode"
            )
        results = self._prepared_results
        self._prepared_results = None
        return results

    def cancel_episode(self) -> None:
        """取消当前片段并清理临时视频。"""
        if self._prepared_results is not None:
            self._remove_results(self._prepared_results)
            self._prepared_results = None
        self._encoder.cancel_episode()
        self._video_keys = ()
        self._frame_counts.clear()

    def close(self) -> None:
        """关闭编码器并清理未提交的视频。"""
        if self._prepared_results is not None:
            self._remove_results(self._prepared_results)
            self._prepared_results = None
        self._encoder.close()

    def _drop_count(self, video_key: str) -> int:
        """读取底层编码器报告的丢帧数。"""
        dropped = getattr(self._encoder, "_dropped_frames", {})
        return int(dropped.get(video_key, 0)) if isinstance(dropped, dict) else 0

    @staticmethod
    def _remove_results(results: dict[str, tuple[Path, dict | None]]) -> None:
        """删除已生成但不再使用的临时视频。"""
        for path, _ in results.values():
            video_path = Path(path)
            try:
                video_path.unlink(missing_ok=True)
                video_path.parent.rmdir()
            except OSError:
                pass


def _decoded_video_frame_count(path: Path, *, frame_threads: int | None = None) -> int:
    """解码视频并返回真实帧数，解码异常视为编码失败。"""
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if frame_threads is not None:
                stream.thread_type = "FRAME"
                stream.codec_context.thread_count = frame_threads
            return sum(1 for _ in container.decode(stream))
    except Exception as exc:
        raise StreamingEncodingError(f"unable to decode {path}: {exc}") from exc


def _recommended_decoder_threads() -> int:
    """限制校验的解码线程，为采集与预览预留处理器。"""
    return max(1, min(4, (os.cpu_count() or 1) - 1))


def _recommended_encoder_threads(camera_count: int) -> int:
    """为采集预留处理器，并限制单相机的软件编码线程数。"""
    usable_cpus = max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(4, usable_cpus // max(1, camera_count)))


def _available_hardware_vcodecs() -> list[str]:
    """返回 PyAV 支持的硬件编码器候选。"""
    from av.codec import Codec

    available: list[str] = []
    for vcodec in _HARDWARE_VIDEO_CODECS:
        try:
            Codec(vcodec, "w")
        except Exception:
            continue
        available.append(vcodec)
    return available


def _probe_codec_options(vcodec: str, encoder_threads: int | None) -> dict[str, str]:
    """生成与实际流式编码一致的编码器探测参数。"""
    options: dict[str, str] = {}
    if vcodec in ("h264", "hevc"):
        options.update({"g": "2", "crf": "30"})
    elif vcodec == "libsvtav1":
        options.update({"g": "2", "crf": "30", "preset": "12"})
    elif vcodec in ("h264_videotoolbox", "hevc_videotoolbox"):
        options.update({"g": "2", "q:v": "40"})
    elif vcodec in ("h264_nvenc", "hevc_nvenc"):
        options.update({"rc": "constqp", "qp": "30"})
    elif vcodec == "h264_vaapi":
        options["qp"] = "30"
    elif vcodec == "h264_qsv":
        options["global_quality"] = "30"

    if encoder_threads is not None:
        if vcodec == "libsvtav1":
            options["svtav1-params"] = f"lp={encoder_threads}"
        elif vcodec not in _HARDWARE_VIDEO_CODECS:
            options["threads"] = str(encoder_threads)
    return options


def _probe_video_encoder(
    vcodec: str,
    *,
    width: int,
    height: int,
    fps: int,
    encoder_threads: int | None,
) -> tuple[bool, str | None]:
    """通过与 LeRobot 相同的 PyAV 后端实际编码并封装探测视频。"""
    import av
    from av.video.stream import VideoStream

    container = None
    try:
        with tempfile.TemporaryDirectory(prefix="handumi-encoder-probe-") as tmp:
            path = Path(tmp) / "probe.mp4"
            container = av.open(str(path), "w")
            stream = cast(
                VideoStream,
                container.add_stream(
                    vcodec,
                    fps,
                    options=_probe_codec_options(vcodec, encoder_threads),
                ),
            )
            stream.pix_fmt = "yuv420p"
            stream.width = width
            stream.height = height
            stream.time_base = Fraction(1, fps)
            for value in (0, 64, 128):
                frame = np.full((height, width, 3), value, dtype=np.uint8)
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = value // 64
                video_frame.time_base = Fraction(1, fps)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError("probe produced an empty MP4")
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _vive_rgb_encoder_config(vcodec: str):
    """创建 VIVE 使用的 H.264 配置；NVENC 禁用 B 帧以兼容短 GOP。"""
    from lerobot.configs.video import RGBEncoderConfig

    extra_options = {"bf": "0"} if vcodec == "h264_nvenc" else {}
    return RGBEncoderConfig(vcodec=vcodec, extra_options=extra_options)


def _probe_vive_streaming_encoder(
    vcodec: str, *, width: int, height: int, fps: int, encoder_threads: int | None,
) -> tuple[bool, str | None]:
    """使用生产流式编码器、配置和真实尺寸编码并完整解码探测帧。"""
    from handumi.scripts._vive_encoding_process import ViveProcessEncoder

    strict = None
    try:
        with tempfile.TemporaryDirectory(prefix="handumi-stream-probe-") as tmp:
            encoder = ViveProcessEncoder(
                fps=fps, rgb_encoder=_vive_rgb_encoder_config(vcodec),
                queue_size=30, encoder_threads=encoder_threads,
            )
            strict = _StrictStreamingEncoder(
                encoder, verify_video_frames=True, rgb_vcodec=vcodec,
            )
            strict.start_episode(["probe"], Path(tmp))
            for value in (0, 64, 128):
                strict.feed_frame("probe", np.full((height, width, 3), value, dtype=np.uint8))
            strict.prepare_episode(3)
            strict.close()
            strict = None
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if strict is not None:
            strict.close()


def _select_video_encoder(
    *,
    policy: str,
    requested_vcodec: str | None,
    width: int,
    height: int,
    fps: int,
    camera_count: int,
    requested_threads: int | None,
    h264_only: bool = False,
    streaming_probe: bool = False,
) -> _VideoEncoderSelection:
    """根据策略选择实际编码探测通过的编码器，不可用时抛出错误。"""
    hardware_codecs = _available_hardware_vcodecs()
    if h264_only:
        hardware_codecs = [codec for codec in hardware_codecs if codec.startswith("h264_")]
    if policy == "auto" and not hardware_codecs:
        log.warning("No supported hardware encoder is exposed; falling back to CPU encoding.")
    explicit_codec = requested_vcodec not in (None, "auto")
    if explicit_codec:
        if requested_vcodec not in _VALID_VIDEO_CODECS:
            choices = ", ".join(sorted(_VALID_VIDEO_CODECS))
            raise SystemExit(
                f"Unsupported --vcodec {requested_vcodec!r}; choose one of: {choices}"
            )
        candidates = [str(requested_vcodec)]
    elif policy == "cpu":
        candidates = [_SOFTWARE_VIDEO_CODEC]
    elif policy == "gpu":
        candidates = hardware_codecs
    else:
        candidates = [*hardware_codecs, _SOFTWARE_VIDEO_CODEC]

    if not candidates:
        raise SystemExit(
            "--encoder gpu requested, but PyAV/FFmpeg exposes no supported hardware encoder."
        )

    failures: list[str] = []
    for candidate in candidates:
        hardware = candidate in _HARDWARE_VIDEO_CODECS
        threads = (
            requested_threads
            if requested_threads is not None
            else None if hardware else _recommended_encoder_threads(camera_count)
        )
        probe = _probe_vive_streaming_encoder if streaming_probe else _probe_video_encoder
        ok, error = probe(
            candidate,
            width=width,
            height=height,
            fps=fps,
            encoder_threads=threads,
        )
        if ok:
            return _VideoEncoderSelection(
                vcodec=candidate,
                hardware=hardware,
                threads=threads,
            )
        failures.append(f"{candidate}: {error or 'unknown error'}")
        if hardware:
            log.warning("Hardware encoder %s failed its startup probe: %s", candidate, error)

    requested = requested_vcodec if explicit_codec else policy
    details = "; ".join(failures)
    raise SystemExit(f"No usable video encoder for {requested!r}: {details}")


def _install_strict_streaming_encoder(
    dataset: object, *, verify_video_frames: bool = False, isolate_process: bool = False,
    rgb_vcodec: str | None = None,
) -> None:
    """安装严格编码包装器，并可启用实际视频帧数检查。"""
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    if writer is None or encoder is None:
        raise RuntimeError("LeRobot did not create the requested streaming encoder.")
    if isolate_process:
        from handumi.scripts._vive_encoding_process import ViveProcessEncoder

        encoder = ViveProcessEncoder(
            fps=encoder.fps, rgb_encoder=encoder._rgb_encoder,
            depth_encoder=encoder._depth_encoder, queue_size=encoder.queue_maxsize,
            encoder_threads=encoder._encoder_threads,
        )
    writer._streaming_encoder = _StrictStreamingEncoder(
        encoder, verify_video_frames=verify_video_frames, rgb_vcodec=rgb_vcodec
    )


def _prepare_streaming_episode(dataset: object, expected_frames: int) -> dict[str, float]:
    """编码、验证当前片段；成功后可由数据集安全提交。"""
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    if not isinstance(encoder, _StrictStreamingEncoder):
        # RuntimeError on purpose: this guards an internal setup invariant
        # (install_strict_streaming_encoder ran), not a caller-supplied type.
        raise RuntimeError(  # noqa: TRY004
            "HandUMI strict streaming encoder is not installed."
        )
    return encoder.prepare_episode(expected_frames)


