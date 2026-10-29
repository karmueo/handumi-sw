#!/usr/bin/env python3
"""Small command router for the common HandUMI workflow."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Command:
    module: str
    description: str


COMMANDS = {
    ("calibrate", "umi"): Command("handumi.scripts.calibrate_umi", "导入和验证原生 UMI 标定"),
    ("doctor",): Command("handumi.scripts.doctor", "Check recording readiness"),
    ("setup",): Command("handumi.scripts.setup.setup_hardware", "Run guided setup"),
    ("setup", "ports"): Command(
        "handumi.scripts.setup.setup_ports", "Identify cameras and serial ports"
    ),
    ("record",): Command("handumi.scripts.record", "Record demonstrations"),
    ("visualize",): Command(
        "handumi.scripts.visualize", "View cameras and gripper openings in Rerun"
    ),
    ("validate",): Command("handumi.scripts.validate", "Validate recorded episodes"),
    ("dataset", "analyze"): Command(
        "handumi.scripts.analyze_dataset",
        "Analyze statistics and episode outliers",
    ),
    ("dataset", "initial-distribution"): Command(
        "handumi.scripts.initial_distribution",
        "Extract initial joint statistics from a robot dataset",
    ),
    ("dataset", "qa"): Command(
        "handumi.scripts.qa_dataset",
        "Run every episode review a conversion depends on",
    ),
    ("dataset", "screen"): Command(
        "handumi.scripts.screen_dataset",
        "Grade episodes by how well a robot can retarget them",
    ),
    ("dataset", "inpaint"): Command(
        "handumi.scripts.inpaint_context",
        "Repaint the context camera to show the target robot",
    ),
    ("dataset", "curate"): Command(
        "handumi.scripts.curate_dataset",
        "Create a validated dataset from an analysis report",
    ),
    ("dataset", "merge"): Command(
        "handumi.scripts.merge_datasets",
        "Join recording sessions into one dataset",
    ),
    ("dataset", "direction"): Command(
        "handumi.scripts.direction_dataset",
        "Flag episodes that demonstrate the task backwards",
    ),
    ("dataset", "export"): Command(
        "handumi.scripts.export_dataset",
        "Rewrite a joints dataset into an external stack's layout",
    ),
    ("replay",): Command(
        "handumi.scripts.replay.replay_in_sim", "Replay a dataset in simulation"
    ),
    ("replay-joints",): Command(
        "handumi.scripts.replay.replay_in_sim_joints",
        "Replay a converted joint-level dataset",
    ),
    ("replay-real",): Command(
        "handumi.scripts.replay.replay_in_real",
        "Replay a converted joints dataset on the physical robot",
    ),
    ("convert",): Command("handumi.scripts.conversion", "Convert data to a robot"),
    ("completion",): Command(
        "handumi.scripts.completion", "Enable Bash, Zsh, or Fish completion"
    ),
    ("teleop",): Command(
        "handumi.scripts.teleop_sim",
        "Teleoperate in simulation",
    ),
    ("teleop-real",): Command(
        "handumi.scripts.teleop_real",
        "Teleoperate a physical robot",
    ),
    ("teleop-record",): Command(
        "handumi.scripts.teleop_record",
        "Record real-robot teleoperation demonstrations",
    ),
    ("camera", "pico"): Command("handumi.scripts.pico_camera", "Stream cameras to PICO"),
    ("calibrate", "grippers"): Command(
        "handumi.scripts.setup.calibrate_grippers", "Calibrate Feetech grippers"
    ),
    ("calibrate", "openarm-grippers"): Command(
        "handumi.scripts.setup.calibrate_openarm_grippers",
        "Calibrate OpenArm grippers",
    ),
    ("calibrate", "piper-grippers"): Command(
        "handumi.real.piper.calibrate_grippers",
        "Set the physical Piper gripper zero points",
    ),
    ("calibrate", "spatial"): Command(
        "handumi.scripts.setup.calibrate_spatial", "Calibrate cameras and workspace"
    ),
    ("calibrate", "workspace-from-video"): Command(
        "handumi.scripts.recover_workspace_camera",
        "Solve the context camera pose from a recording",
    ),
    ("calibrate", "tcp"): Command(
        "handumi.scripts.setup.calibrate_tcp_offset", "Calibrate controller-to-TCP"
    ),
    ("calibrate", "verify"): Command(
        "handumi.scripts.setup.verify_calibrations",
        "Verify the complete calibration chain",
    ),
    ("servo", "home"): Command("handumi.scripts.setup.home_servos", "Home Feetech servos"),
    ("servo", "set-id"): Command(
        "handumi.scripts.setup.set_servo_id", "Assign a Feetech servo ID"
    ),
    ("tracking", "pose"): Command(
        "handumi.scripts.setup.print_controller_pose", "Print live controller poses"
    ),
}


def _program_name() -> str:
    name = Path(sys.argv[0]).name
    return name if name in {"handumi", "hu"} else "handumi"


def _print_help(
    prefix: tuple[str, ...] = (), *, program: str = "handumi"
) -> None:
    label_prefix = f"{program} {' '.join(prefix)}" if prefix else program
    print(f"usage: {label_prefix} <command> [options]\n")
    print("Commands:")
    commands = {
        path: command
        for path, command in COMMANDS.items()
        if path[: len(prefix)] == prefix and len(path) > len(prefix)
    }
    labels = [" ".join(path) for path in commands]
    width = max(len(label) for label in labels)
    for (path, command), label in zip(commands.items(), labels, strict=True):
        print(f"  {program} {label:<{width}}  {command.description}")
    print(f"\nRun '{program} <command> --help' for command-specific options.")


def _resume_recording_device(arguments: list[str]) -> str | None:
    """读取续录数据集的设备类型；元数据不可用时交由录制入口校验。"""
    if "--resume" not in arguments:
        return None
    output_dir = None
    for index, value in enumerate(arguments):
        if value.startswith("--output-dir="):
            output_dir = value.split("=", 1)[1]
        elif value == "--output-dir" and index + 1 < len(arguments):
            output_dir = arguments[index + 1]
    if output_dir is None:
        return None
    try:
        info_path = Path(output_dir) / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metadata = info.get("handumi") if isinstance(info, dict) else None
    device = metadata.get("recording_device") if isinstance(metadata, dict) else None
    return device if device in ("pico", "meta", "vive") else None


def main(argv: list[str] | None = None) -> None:
    program = _program_name()
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        _print_help(program=program)
        return
    group = (values[0],)
    group_exists = any(
        path[:1] == group and len(path) > 1 for path in COMMANDS
    ) and group not in COMMANDS
    if group_exists and (len(values) == 1 or values[1] in {"-h", "--help"}):
        _print_help(group, program=program)
        return
    match = next(
        (
            (path, command)
            for path, command in sorted(
                COMMANDS.items(), key=lambda item: len(item[0]), reverse=True
            )
            if tuple(values[: len(path)]) == path
        ),
        None,
    )
    if match is None:
        requested = " ".join(values[:2])
        raise SystemExit(
            f"Unknown HandUMI command {requested!r}. Run '{program} --help'."
        )
    path, command = match
    rest = values[len(path) :]
    if path in {("teleop-real",), ("teleop-record",), ("replay-real",)}:
        # This IK is a tiny latency-sensitive dense solve. On the supported
        # NVIDIA setup, CPU execution has far lower tail latency than CUDA.
        # Respect an explicit user override for benchmarking/debugging.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if path in {("record",), ("visualize",), ("doctor",)}:
        # 显式设备优先，其次是续录数据集，最后使用本机 rig。
        selected = None
        for index, value in enumerate(rest):
            if value.startswith("--device="):
                selected = value.split("=", 1)[1]
            elif value == "--device" and index + 1 < len(rest):
                selected = rest[index + 1]
        if selected is None and path == ("record",):
            selected = _resume_recording_device(rest)
        if selected is None:
            import yaml
            rig = Path("configs/rig.yaml")
            for index, value in enumerate(rest):
                if value == "--rig-config" and index + 1 < len(rest):
                    rig = Path(rest[index + 1])
                elif value.startswith("--rig-config="):
                    rig = Path(value.split("=", 1)[1])
            if rig.exists():
                selected = (yaml.safe_load(rig.read_text()) or {}).get("recording", {}).get("device")
            else:
                selected = "vive"
        if selected == "vive":
            from handumi.scripts.umi_capture import main as umi_main
            umi_main(rest, mode=path[0])
            return
    if path in {("replay",), ("convert",)} and not any(value == "--robot" or value.startswith("--robot=") for value in rest):
        import yaml
        rig = Path("configs/rig.yaml")
        preferred = (yaml.safe_load(rig.read_text()) or {}).get("recording", {}).get("robot") if rig.exists() else "rm75b"
        if preferred:
            rest.extend(["--robot", str(preferred)])
    module = importlib.import_module(command.module)
    previous_argv = sys.argv
    try:
        sys.argv = [f"{program} {' '.join(path)}", *rest]
        module.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
