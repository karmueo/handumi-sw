"""在真实数据子集上比较旧版、CPU 和 GPU TCP 导出的耗时与内存。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def benchmark_source(source: Path, destination: Path, count: int) -> int:
    """创建只读引用源数据的临时数据集，返回保留回合的帧数。"""
    destination.mkdir()
    for name in ("data", "videos"):
        (destination / name).symlink_to(
            (source / name).resolve(), target_is_directory=True
        )
    shutil.copytree(source / "meta", destination / "meta")
    path = destination / "meta" / "handumi_quality.json"
    report = json.loads(path.read_text())
    report["dataset"] = f"local/{destination.name}"
    accepted = [row for row in report["episodes"] if row["status"] == "accepted"]
    selected = {int(row["episode_index"]) for row in accepted[:count]}
    if not selected:
        raise ValueError("源数据没有通过质量检查的回合")
    frames = sum(int(row["frame_count"]) for row in accepted[:count])
    for row in report["episodes"]:
        if int(row["episode_index"]) not in selected:
            row["status"] = "rejected"
            row["findings"] = []
    path.write_text(json.dumps(report))
    return frames


def peak_tree_rss(process) -> int | None:
    """采样整个进程树的当前常驻内存；无 psutil 时返回空值。"""
    try:
        import psutil

        root = psutil.Process(process.pid)
        values = [root, *root.children(recursive=True)]
        return sum(item.memory_info().rss for item in values if item.is_running())
    except (ImportError, OSError, ProcessLookupError):
        return None
    except Exception as exc:
        # 进程在枚举时退出属于正常竞争，其余异常仍应暴露。
        if type(exc).__name__ in ("NoSuchProcess", "AccessDenied", "ZombieProcess"):
            return None
        raise


def _stop_process_group(process) -> None:
    """终止基准进程组并回收进程，必要时强制结束。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    # 主进程可能已经退出但工作进程仍存活，确保进程组全部结束。
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_case(
    command: list[str], log: Path, frames: int, *, timeout: float | None = None
) -> dict:
    """运行一次导出并测量墙钟时间和采样峰值内存。"""
    started = time.monotonic()
    peak = None
    timed_out = False
    timeout = frames / 30 + 180 if timeout is None else timeout
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    stream.write(f"导出超过 {timeout:.1f} 秒，已终止进程组。\n")
                    break
                sample = peak_tree_rss(process)
                if sample is not None:
                    peak = max(peak or 0, sample)
                time.sleep(0.1)
        finally:
            _stop_process_group(process)
    elapsed = time.monotonic() - started
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "seconds": elapsed,
        "frames_per_second": frames / elapsed,
        "peak_rss_bytes_sampled": peak,
        "log": log.name,
    }


def main() -> None:
    """解析基准参数，运行预热与重复采样，输出可比较的结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="已完成质量检查的本地数据集")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="不存在的结果目录"
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="采样通过的回合数（默认 20）"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="每种方案的测量次数（默认 3）"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="每种方案的预热次数（默认 1）"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(240, 134),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--resize-mode",
        choices=("letterbox", "center-crop", "stretch"),
        default="stretch",
    )
    parser.add_argument(
        "--baseline-ref", help="优化前代码 Git 版本，测量 legacy 时必填"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("legacy", "cpu1", "cpu4", "gpu1", "gpu4"),
        default=("legacy", "cpu1", "cpu4", "gpu1", "gpu4"),
        help="需要测量的方案，默认全部测量",
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.runs <= 0 or args.warmup < 0:
        parser.error("episodes 和 runs 必须为正整数；warmup 必须非负")
    if "legacy" in args.modes and not args.baseline_ref:
        parser.error("测量 legacy 时必须指定 --baseline-ref")
    if len(set(args.modes)) != len(args.modes):
        parser.error("基准方案不能重复")
    source = args.source.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=args.output_dir) as tmp:
        scratch = Path(tmp)
        sample = scratch / source.name
        frames = benchmark_source(source, sample, args.episodes)
        legacy = scratch / "legacy_export_tcp.py"
        repo_root = Path(__file__).resolve().parents[3]
        if "legacy" in args.modes:
            code = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{args.baseline_ref}:src/handumi/scripts/export_tcp.py",
                ],
                cwd=repo_root,
            )
            legacy.write_bytes(code)
        modes = {
            "legacy": ([sys.executable, str(legacy)], []),
            "cpu1": (
                [sys.executable, "-m", "handumi.scripts.export_tcp"],
                ["--encoder", "cpu", "--workers", "1"],
            ),
            "cpu4": (
                [sys.executable, "-m", "handumi.scripts.export_tcp"],
                ["--encoder", "cpu", "--workers", "4"],
            ),
            "gpu1": (
                [sys.executable, "-m", "handumi.scripts.export_tcp"],
                ["--encoder", "gpu", "--workers", "1"],
            ),
            "gpu4": (
                [sys.executable, "-m", "handumi.scripts.export_tcp"],
                ["--encoder", "gpu", "--workers", "4"],
            ),
        }
        # 子进程运行旧版脚本时仍须导入当前本地包。
        package_root = str(repo_root / "src")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(None, (package_root, os.environ.get("PYTHONPATH")))
        )
        for mode in args.modes:
            entry, options = modes[mode]
            results[mode] = []
            for run in range(args.warmup + args.runs):
                output = scratch / f"{mode}-{run}"
                command = [
                    *entry,
                    str(sample),
                    "--output",
                    str(output),
                    "--repo-id",
                    f"local/{mode}-{run}",
                    "--resize-mode",
                    args.resize_mode,
                    "--image-size",
                    *(str(n) for n in args.image_size),
                    *options,
                ]
                print(
                    f"{mode} {'预热' if run < args.warmup else '测量'} {run + 1}",
                    flush=True,
                )
                log = args.output_dir / f"{mode}-{run}.log"
                result = run_case(command, log, frames)
                shutil.rmtree(output, ignore_errors=True)
                if run >= args.warmup:
                    results[mode].append(result)
        summary = {}
        for mode, runs in results.items():
            passed = [
                run for run in runs if run["returncode"] == 0 and not run["timed_out"]
            ]
            summary[mode] = {
                "successful_runs": len(passed),
                "median_seconds": statistics.median(run["seconds"] for run in passed)
                if passed
                else None,
                "median_frames_per_second": statistics.median(
                    run["frames_per_second"] for run in passed
                )
                if passed
                else None,
                "peak_rss_bytes_sampled": max(
                    (run["peak_rss_bytes_sampled"] or 0 for run in passed), default=0
                )
                or None,
            }
        report = {
            "source": str(source),
            "frames": frames,
            "episodes_requested": args.episodes,
            "image_size": list(args.image_size),
            "resize_mode": args.resize_mode,
            "baseline_ref": args.baseline_ref,
            "warmup": args.warmup,
            "runs": results,
            "summary": summary,
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        (args.output_dir / "report.txt").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
