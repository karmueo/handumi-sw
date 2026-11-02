"""对比 VIVE 使用的三种 LeRobot 编码方式，输出逐段及汇总性能报告。"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from handumi.scripts._record_encoding import (
    _install_strict_streaming_encoder,
    _select_video_encoder,
    _vive_rgb_encoder_config,
)
from handumi.scripts.umi_capture import _capture_metrics, _save_streaming_episode


def _frames(width, height):
    """创建种子固定的动态画面，所有测试进程使用同一组输入。"""
    generator = np.random.default_rng(20261102)
    # 大尺度纹理和移动图案使内容随帧变化，同时避免白噪声造成异常压缩负载。
    low = generator.integers(0, 256, (height // 16 + 1, width // 16 + 1, 3), dtype=np.uint8)
    base = np.repeat(np.repeat(low, 16, axis=0), 16, axis=1)[:height, :width]
    return [np.ascontiguousarray(np.roll(base, index * 8, axis=1)) for index in range(12)]


def _worker(args):
    """在隔离进程中运行一个真实 LeRobot 写入及保存场景。"""
    import av
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args.run_dir.mkdir(parents=True, exist_ok=False)
    frame_images = _frames(args.width, args.height)
    streaming = args.worker_mode != "legacy"
    selection = _select_video_encoder(
        policy="auto" if args.worker_mode == "auto" else "cpu", requested_vcodec=None,
        width=args.width, height=args.height, fps=args.fps, camera_count=1,
        requested_threads=None, h264_only=True, streaming_probe=streaming,
    )
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": None},
        "observation.images.right_wrist": {
            "dtype": "video", "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        "local/vive-encoding-benchmark", fps=args.fps, features=features,
        root=args.run_dir / "dataset", use_videos=True,
        rgb_encoder=(
            _vive_rgb_encoder_config(selection.vcodec)
            if streaming else RGBEncoderConfig(vcodec=selection.vcodec)
        ),
        streaming_encoding=streaming, encoder_queue_maxsize=30,
        encoder_threads=selection.threads if streaming else None,
        image_writer_threads=0 if streaming else 4,
    )
    if streaming:
        _install_strict_streaming_encoder(
            dataset, verify_video_frames=True, isolate_process=True, rgb_vcodec=selection.vcodec,
        )
    times, camera_times, add_times = [], [], []
    deadline = time.monotonic()
    try:
        for index in range(args.frames):
            delay = deadline + index / args.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            stamp = time.monotonic()
            frame = {
                "observation.state": np.full(16, index, dtype=np.float32),
                "observation.images.right_wrist": frame_images[index % len(frame_images)],
                "task": "synthetic benchmark",
            }
            dataset.add_frame(frame)
            finished = time.monotonic()
            times.append(stamp)
            camera_times.append(stamp)
            add_times.append(finished - stamp)
        save_start = time.monotonic()
        if streaming:
            durations = _save_streaming_episode(dataset, args.frames)
        else:
            dataset.save_episode()
            durations = {"encode_finish_s": None, "encoder_flush_s": None,
                         "video_verify_s": None, "commit_s": time.monotonic() - save_start}
        save_total = time.monotonic() - save_start
        result = _capture_metrics(times, camera_times, add_times, durations, save_total)
        dataset.finalize()
        videos = list((args.run_dir / "dataset" / "videos").rglob("*.mp4"))
        if len(videos) != 1:
            raise RuntimeError(f"期望一个视频文件，实际为 {len(videos)}")
        with av.open(str(videos[0])) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        if decoded != args.frames:
            raise RuntimeError(f"视频实际帧数 {decoded} 与数据帧数 {args.frames} 不一致")
        result.update(mode=args.worker_mode, codec=selection.vcodec,
                      hardware=selection.hardware, decoded_frames=decoded,
                      encoder_threads=selection.threads if streaming else None,
                      width=args.width, height=args.height, target_fps=args.fps)
        (args.run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        dataset.finalize()


def _run(args, mode, name, frames):
    """在独立进程和独立目录运行一段，返回结果或错误信息。"""
    run_dir = args.output_dir / f"{mode}-{name}"
    command = [
        sys.executable, "-m", "handumi.scripts.benchmark_vive_encoding",
        "--worker-mode", mode, "--run-dir", str(run_dir), "--frames", str(frames),
        "--width", str(args.width), "--height", str(args.height), "--fps", str(args.fps),
    ]
    print(f"运行 {mode} {name}：{frames} 帧", flush=True)
    # 包含预定输入时长及编码、提交和进程启动余量，卡死不能阻塞后续测试。
    timeout_s = frames / args.fps + 180
    try:
        process = subprocess.run(
            command, text=True, capture_output=True, check=False, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        message = f"工作进程在 {timeout_s:.1f} 秒后超时：{exc}"
        (run_dir / "worker.log").write_text(message + "\n")
        return {"mode": mode, "error": message, "returncode": None}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "worker.log").write_text(process.stdout + "\n" + process.stderr)
    if process.returncode:
        return {"mode": mode, "error": process.stderr[-4000:], "returncode": process.returncode}
    return json.loads((run_dir / "result.json").read_text())


def main():
    """运行隔离基准，生成包含失败原因、逐段数据及中位数的报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--modes", nargs="+", choices=("legacy", "cpu", "auto"),
        default=["legacy", "cpu", "auto"], help="选择要运行的基准模式，默认运行全部模式",
    )
    parser.add_argument("--worker-mode", choices=("legacy", "cpu", "auto"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.frames, args.warmup_frames, args.runs) < 1:
        parser.error("尺寸、帧率、帧数和次数必须为正整数")
    if args.worker_mode:
        if args.run_dir is None:
            parser.error("内部工作进程需要 --run-dir")
        _worker(args)
        return
    if args.output_dir is None:
        parser.error("需要指定新的 --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"width": args.width, "height": args.height, "target_fps": args.fps,
              "frames_per_run": args.frames, "warmup_frames": args.warmup_frames,
              "runs_per_mode": args.runs, "input": "seeded moving synthetic RGB", "modes": {}}
    failed = False
    for mode in dict.fromkeys(args.modes):
        warmup = _run(args, mode, "warmup", args.warmup_frames)
        runs = [_run(args, mode, str(index + 1), args.frames) for index in range(args.runs)]
        errors = [result for result in [warmup, *runs] if "error" in result]
        failed |= bool(errors)
        valid = [result for result in runs if "error" not in result]
        saves = [result["save_total_s"] for result in valid]
        flushes = [result["encoder_flush_s"] for result in valid
                   if result["encoder_flush_s"] is not None]
        verifications = [result["video_verify_s"] for result in valid
                         if result["video_verify_s"] is not None]
        report["modes"][mode] = {
            "warmup": warmup, "runs": runs,
            "save_median_s": statistics.median(saves) if saves else None,
            "save_p95_s": float(np.percentile(saves, 95)) if saves else None,
            "encoder_flush_median_s": statistics.median(flushes) if flushes else None,
            "video_verify_median_s": statistics.median(verifications) if verifications else None,
            "errors": errors,
        }
    baseline = report["modes"].get("legacy", {}).get("save_median_s")
    for mode in ("cpu", "auto"):
        if mode not in report["modes"]:
            continue
        current = report["modes"][mode]["save_median_s"]
        report["modes"][mode]["save_reduction"] = (
            1 - current / baseline if baseline and current is not None else None
        )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["VIVE 编码基准", f"分辨率：{args.width}×{args.height} @ {args.fps} FPS"]
    for mode, data in report["modes"].items():
        lines.append(f"{mode}: 保存中位数 {data['save_median_s']} 秒，P95 {data['save_p95_s']} 秒，失败 {len(data['errors'])} 次")
        if data["video_verify_median_s"] is not None:
            lines.append(f"  编码收尾中位数 {data['encoder_flush_median_s']} 秒，解码校验中位数 {data['video_verify_median_s']} 秒")
    text = "\n".join(lines) + "\n"
    (args.output_dir / "report.txt").write_text(text)
    print(text, end="")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
