"""共享录制输出目录的覆盖参数、只读检查和清理逻辑。"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)  # 目录清理日志。


def _overwrite_arg(value: str) -> bool:
    """解析覆盖开关；仅接受 true/false，其他值报告参数错误。"""
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _check_output_directory(args: argparse.Namespace) -> None:
    """检查新录制的输出路径；追加模式保持原校验，检查不修改文件。"""
    if args.resume:
        return
    # 保留原始路径用于符号链接检查，规范路径用于危险目录检查。
    output_dir = Path(args.output_dir)
    if output_dir.is_symlink():
        raise SystemExit(f"Output directory must not be a symbolic link: {output_dir}")
    resolved = output_dir.resolve()
    protected = {Path.home().resolve(), Path.cwd().resolve(), *Path.cwd().resolve().parents}
    if resolved == Path(resolved.anchor) or resolved in protected:
        raise SystemExit(f"Refusing unsafe output directory: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SystemExit(f"Output path is not a directory: {output_dir}")
        if not args.overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite true to replace it or --resume to append."
            )


def _prepare_output_directory(args: argparse.Namespace) -> None:
    """重新检查并清理待覆盖目录；预演和追加不删除，清理失败终止录制。"""
    _check_output_directory(args)
    if args.resume or args.dry_run:
        return
    if Path(args.output_dir).exists():
        log.info("Removing existing dataset directory: %s", args.output_dir)
        try:
            shutil.rmtree(args.output_dir)
        except OSError as exc:
            raise SystemExit(f"Cannot overwrite {args.output_dir}: {exc}") from exc


