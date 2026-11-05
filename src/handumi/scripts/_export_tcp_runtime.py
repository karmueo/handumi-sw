"""TCP 离线导出的源数据读取、无丢帧编码和并行回合调度。"""

from __future__ import annotations

import os
import queue
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from lerobot.datasets.video_utils import StreamingVideoEncoder

from handumi.dataset.gripper import read_normalized
from handumi.dataset.raw import TRACKING_VALIDITY_NAMES
from handumi.dataset.reader import validate_raw_state_metadata


class EpisodeReader:
    """复用 Arrow 扫描器，读取转换和夹爪校验需要的源列。"""

    def __init__(self, root: Path, info: dict) -> None:
        """检查原始布局并记录目标列。"""
        validate_raw_state_metadata(info)
        self.metadata = info["handumi"]
        files = sorted((root / "data").glob("chunk-*/*.parquet"))
        if not files:
            raise ValueError("源数据缺少 Parquet 文件")
        self.dataset = ds.dataset(files, format="parquet")
        self.columns = [
            "episode_index",
            "frame_index",
            "observation.state",
            "observation.valid",
            "observation.tracking.right_tcp_pose",
        ]
        for side in ("left", "right"):
            self.columns.extend(
                f"observation.gripper.{side}.{suffix}"
                for suffix in ("normalized", "valid", "sample_time_ns", "sequence")
            )
        camera = self.metadata["sources"]["gripper"]["right"].get("camera")
        self.columns.extend(
            f"observation.camera.{camera}.{suffix}"
            for suffix in ("sample_time_ns", "sequence")
        )
        missing = set(self.columns) - set(self.dataset.schema.names)
        if missing:
            raise ValueError(f"源数据缺少列: {', '.join(sorted(missing))}")

    def read(self, index: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """按源回合读取 TCP 和归一化夹爪，核验帧序和源相机。"""
        table = self.dataset.to_table(
            columns=self.columns,
            filter=ds.field("episode_index") == index,
            use_threads=False,
        )
        values = {name: np.asarray(table[name].to_pylist()) for name in self.columns}
        if table.num_rows != length or not length:
            raise ValueError("TCP 帧数与回合元数据不一致")
        if not np.array_equal(values["frame_index"].reshape(-1), np.arange(length)):
            raise ValueError("源回合帧编号不连续")
        # 旧读取器会强制转为 float32；保留对不可转换原始状态的拒绝。
        states = np.asarray(values["observation.state"], dtype=np.float32)
        if states.shape != (length, 16):
            raise ValueError("原始状态维度错误")
        if values["observation.valid"].shape != (length, len(TRACKING_VALIDITY_NAMES)):
            raise ValueError("原始跟踪有效性维度错误")
        read_normalized(values, self.metadata)
        return (
            values["observation.tracking.right_tcp_pose"],
            values["observation.gripper.right.normalized"],
        )


class OfflineStreamingEncoder(StreamingVideoEncoder):
    """编码队列满时施加背压，编码异常和退出均向调用者报告。"""

    def start_episode(self, video_keys, temp_dir, depth_video_keys=None) -> None:
        """开始新回合并缓存首帧，用于单帧视频的图像统计。"""
        self._first_frames = {}
        super().start_episode(video_keys, temp_dir, depth_video_keys)

    def _put(self, video_key: str, item: np.ndarray | None) -> None:
        """等待队列空位，同时检查编码线程是否异常退出。"""
        deadline = time.monotonic() + 120
        thread = self._threads[video_key]
        while True:
            if not thread.is_alive():
                try:
                    status, message = self._result_queues[video_key].get_nowait()
                except queue.Empty:
                    raise RuntimeError(f"编码线程提前退出: {video_key}") from None
                raise RuntimeError(f"编码线程提前退出: {status}: {message}")
            try:
                self._frame_queues[video_key].put(item, timeout=0.1)
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"编码队列等待超时: {video_key}") from None

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """安全送帧；队列拥塞时等待，不丢弃图像。"""
        if not self._episode_active:
            raise RuntimeError("编码回合尚未开始")
        if video_key not in self._first_frames:
            self._first_frames[video_key] = image.copy()
        self._put(video_key, image.copy())

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """结束编码，拒绝超时、异常和缺失结果。"""
        if not self._episode_active:
            raise RuntimeError("编码回合尚未开始")
        results = {}
        for key in self._frame_queues:
            self._put(key, None)
        for key, thread in self._threads.items():
            thread.join(timeout=120)
            if thread.is_alive():
                raise RuntimeError(f"编码线程结束超时: {key}")
            try:
                status, data = self._result_queues[key].get(timeout=5)
            except queue.Empty:
                raise RuntimeError(f"编码线程无结果: {key}") from None
            if status != "ok":
                raise RuntimeError(f"编码线程失败: {data}")
            if data is None and key in self._first_frames:
                from lerobot.datasets.compute_stats import (
                    RunningQuantileStats,
                    auto_downsample_height_width,
                )

                image = self._first_frames[key].transpose(2, 0, 1)
                image = auto_downsample_height_width(image).transpose(1, 2, 0)
                stats = RunningQuantileStats()
                stats.update(image.reshape(-1, image.shape[-1]))
                data = stats.get_statistics()
            results[key] = (self._video_paths[key], data)
        self._cleanup()
        self._episode_active = False
        return results


def encoder_config(codec: str):
    """为 H.264 软件和硬件编码创建固定画质配置。"""
    from lerobot.configs.video import RGBEncoderConfig

    if codec == "h264":
        return RGBEncoderConfig(vcodec=codec, preset="fast", crf=23)
    if codec == "h264_nvenc":
        # FFmpeg 的 NVENC 枚举中 15 对应 p4；LeRobot 参数校验要求整数。
        return RGBEncoderConfig(
            vcodec=codec, preset=15, crf=23, extra_options={"bf": "0"}
        )
    return RGBEncoderConfig(vcodec=codec, crf=23)


def probe_encoder(codec: str, image_size: tuple[int, int], workers: int) -> None:
    """同时启动所需数量编码器，并验证生产尺寸的实际视频可完整解码。"""
    import tempfile

    from handumi.scripts._record_encoding import _decoded_video_frame_count

    config = encoder_config(codec)
    encoders = []
    with tempfile.TemporaryDirectory(prefix="handumi-tcp-probe-") as tmp:
        try:
            for index in range(workers):
                encoder = OfflineStreamingEncoder(
                    fps=30, rgb_encoder=config, queue_maxsize=1, encoder_threads=1
                )
                encoders.append(encoder)
                encoder.start_episode([f"probe{index}"], Path(tmp))
            width, height = image_size
            # 一帧队列和三次送帧使每个线程都已建立视频会话，之后再统一结束。
            for index, encoder in enumerate(encoders):
                for value in (0, 64, 128):
                    image = np.full((height, width, 3), value, dtype=np.uint8)
                    encoder.feed_frame(f"probe{index}", image)
            for index, encoder in enumerate(encoders):
                result = encoder.finish_episode()
                path, _ = result[f"probe{index}"]
                if _decoded_video_frame_count(path, frame_threads=1) != 3:
                    raise RuntimeError("编码探测帧数不匹配")
        finally:
            for encoder in encoders:
                encoder.close()


def select_encoder(policy: str, image_size: tuple[int, int], workers: int) -> str:
    """以生产配置探测所有并发硬件会话，按策略选择编码器。"""
    from handumi.scripts._record_encoding import _available_hardware_vcodecs

    if policy == "cpu":
        return "h264"
    failures = []
    for codec in _available_hardware_vcodecs():
        if not codec.startswith("h264_"):
            continue
        try:
            probe_encoder(codec, image_size, workers)
        except Exception as exc:
            failures.append(f"{codec}: {exc}")
        else:
            return codec
    detail = "; ".join(failures) or "没有可用的 H.264 硬件编码器"
    if policy == "gpu":
        raise RuntimeError(f"GPU 编码探测失败: {detail}")
    print(f"GPU 编码不可用，回退 CPU：{detail}", flush=True)
    return "h264"


def split_episodes(kept: list[int], rows: list[dict], workers: int) -> list[list[int]]:
    """按帧数近似均分连续回合，输出顺序与源数据相同。"""
    groups = []
    start = 0
    remaining = sum(int(rows[i]["length"]) for i in kept)
    for group_index in range(workers):
        groups_left = workers - group_index
        if groups_left == 1:
            groups.append(kept[start:])
            break
        target = remaining / groups_left
        end, count = start, 0
        max_end = len(kept) - (groups_left - 1)
        while end < max_end:
            next_count = count + int(rows[kept[end]]["length"])
            if end > start and abs(target - count) <= abs(target - next_count):
                break
            count = next_count
            end += 1
        groups.append(kept[start:end])
        start, remaining = end, remaining - count
    return groups


def worker_threads(workers: int) -> int:
    """为每个进程分配有限的软件编码线程。"""
    return max(1, min(4, (os.cpu_count() or 1) // workers))
