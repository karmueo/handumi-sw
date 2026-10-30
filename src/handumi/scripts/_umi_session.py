"""解析 UMI 标定路径并原子读写独立桌面会话，供命令入口共用。"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from handumi.calibration.umi_table import validate_session

DEFAULT_CALIBRATION = Path("configs/calibration/umi/local/calibration.yaml")
DEFAULT_SESSION = Path("configs/calibration/umi/local/session.yaml")


def calibration_paths(args):
    """按命令行、rig、默认路径的优先级返回几何与桌面标定路径。"""
    rig = yaml.safe_load(args.rig_config.read_text()) if args.rig_config.exists() else {}
    config = (rig or {}).get("umi") or {}
    return (
        args.calibration or Path(config.get("calibration", DEFAULT_CALIBRATION)),
        args.session_calibration or Path(config.get("session_calibration", DEFAULT_SESSION)),
    )


def load_session(path, snapshot, *, required=True, explicit=False, require_verified=True):
    """加载并校验会话；仅诊断/预览的隐式缺失文件可返回 None。"""
    if not path.exists() and not required and not explicit:
        return None
    try:
        session = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取桌面标定 {path}；请运行 calibrate umi table 和 verify-table") from exc
    return validate_session(session, snapshot, require_verified=require_verified)


def save_session(path, session):
    """同目录写入临时文件并替换会话，写入失败时保留原文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(yaml.safe_dump(session, sort_keys=False))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
