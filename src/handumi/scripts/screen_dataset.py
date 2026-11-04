#!/usr/bin/env python3
"""根据目标机器人筛选数据集，并显示筛选和报告写入进度。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.screening import (
    RetargetScreeningConfig,
    render_screening_markdown,
    screen_dataset,
    screening_report_path,
    write_screening_report,
)
from handumi.dataset.screening_progress import ScreenProgress
from handumi.dataset.selection import resolve_dataset_selection
from handumi.robots.registry import EMBODIMENT_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget every episode through the conversion solver and grade the "
            "result, so unusable episodes are removed before joint conversion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Local dataset path or Hugging Face repo id.")
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, required=True)
    parser.add_argument("--retarget-mode", choices=("auto", "local-relative", "anchored", "absolute-table", "absolute-tcp"), default="auto")
    parser.add_argument("--base-placement", choices=("auto", "fixed", "episode-xy", "dataset-xy-yaw"), default="auto")
    parser.add_argument("--robot-tcp-calibration", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--no-cache", action="store_true")
    cache_mode.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--deployment-calibration", type=Path, default=None)
    parser.add_argument("--revision", default="main", help="Hub dataset revision.")
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated episode indices (default: every episode).",
    )
    parser.add_argument(
        "--deployment-profile",
        choices=("auto", "local", "sim"),
        default="auto",
        help="Table placement source, matching replay and convert.",
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Processes solving episodes at once. Episodes are independent, so "
        "this changes wall-clock only. The default leaves room for the threads "
        "the solver already uses inside one episode; 1 solves them in order.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to DATASET/meta/handumi_screening_<robot>.json.",
    )
    parser.add_argument(
        "--progress", choices=("auto", "on", "off"), default="auto",
        help="Show screening progress in a terminal or as periodic text.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Human-readable report; defaults beside the JSON report.",
    )
    parser.add_argument(
        "--max-position-error-m",
        type=float,
        default=None,
    )
    parser.add_argument("--initial-position-tolerance-m", type=float, default=None)
    parser.add_argument(
        "--max-rotation-error-deg",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=(
            "Explicit controller->TCP YAML instead of the capture's snapshot; "
            "pass the same file to `handumi convert` so it reuses these solves."
        ),
    )
    parser.add_argument(
        "--max-base-rotation-deg",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Reject episodes whose arm base (first joint) swings farther than "
            "this from home. Off by default; 60 matches what XHUMAN's real "
            "Piper teleoperation stays under."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the screening plan without solving episodes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        selection = resolve_dataset_selection(args.dataset, revision=args.revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    from handumi.scripts.replay.replay_deployment import select_saved_dataset_placement

    args.dataset_root = selection.root
    select_saved_dataset_placement(args, allow_explicit=True)

    episodes = None
    if args.episodes:
        try:
            episodes = [int(part) for part in args.episodes.split(",") if part.strip()]
        except ValueError as exc:
            raise SystemExit(f"Invalid --episodes list: {exc}") from exc

    report_path = args.report or screening_report_path(selection.root, args.robot)
    print(
        "Screening plan\n"
        f"  Dataset: {selection.root}\n"
        f"  Robot: {args.robot}\n"
        f"  Retargeting: {args.retarget_mode}\n"
        f"  Base placement: {args.base_placement}\n"
        f"  Placement file: {getattr(getattr(args, 'deployment_selection', None), 'path', None) or args.deployment_calibration or 'automatic'}\n"
        f"  Episodes: {'all' if episodes is None else episodes}\n"
        f"  Controller->TCP: {args.controller_tcp_calibration or 'dataset snapshot / robot default'}\n"
        f"  Report: {report_path}"
    )
    if args.dry_run:
        return

    config = RetargetScreeningConfig(
        max_position_error_m=args.max_position_error_m if args.max_position_error_m is not None else (0.005 if args.retarget_mode == "absolute-tcp" else 0.03),
        max_rotation_error_deg=args.max_rotation_error_deg if args.max_rotation_error_deg is not None else (5.0 if args.retarget_mode == "absolute-tcp" else 45.0),
        max_base_rotation_deg=args.max_base_rotation_deg,
    )
    with ScreenProgress(mode=args.progress) as display:
        payload = screen_dataset(
            selection.root,
            robot=args.robot,
            repo_id=selection.repo_id,
            revision=args.revision,
            episodes=episodes,
            deployment_profile=args.deployment_profile,
            rig_config=args.rig_config,
            jobs=args.jobs,
            config=config,
            controller_tcp_calibration=args.controller_tcp_calibration,
            retarget_mode=args.retarget_mode,
            base_placement=args.base_placement,
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
            refresh_cache=args.refresh_cache,
            robot_tcp_calibration=args.robot_tcp_calibration,
            deployment_calibration=args.deployment_calibration,
            deployment_selection=getattr(args, "deployment_selection", None),
            initial_position_tolerance_m=args.initial_position_tolerance_m,
            progress=args.progress,
            _progress_display=display,
        )
        display.phase("保存报告")
        writing_started = time.monotonic()
        write_screening_report(report_path, payload)
        markdown_path = args.markdown or report_path.with_suffix(".md")
        markdown_path.write_text(render_screening_markdown(payload), encoding="utf-8")
        if "performance" in payload:
            # 先写轨迹和报告，再补记写入耗时；第二次只更新小型 JSON。
            performance = payload["performance"]
            performance["report_writing_seconds"] = time.monotonic() - writing_started
            performance["total_seconds"] += performance["report_writing_seconds"]
            write_screening_report(report_path, {k: v for k, v in payload.items() if k != "_solves"})
        display.phase("报告已保存")

    summary = payload["summary"]
    print(
        f"\n[screen] {summary['total']} episodes: {summary['accepted']} accepted, "
        f"{summary['rejected']} rejected"
    )
    print(f"[screen] report: {report_path}")
    print(f"[screen] markdown: {markdown_path}")
    if summary["review_episode_indices"]:
        print(
            "[screen] warnings needing your review: "
            + ",".join(str(i) for i in summary["review_episode_indices"])
        )
    if summary["reject_episode_indices"]:
        excluded = ",".join(str(i) for i in summary["reject_episode_indices"])
        print(
            f"[screen] rejected: {excluded}\n"
            "[screen] to remove them, feed this report to analyze and curate:\n"
            f"  handumi dataset analyze {selection.root} "
            f"--quality-report {report_path}\n"
            f"  handumi dataset curate {selection.root} "
            f"--output <new_root> --exclude {excluded}"
        )
    else:
        print("[screen] no episode is unusable for this embodiment.")


if __name__ == "__main__":
    sys.exit(main())
