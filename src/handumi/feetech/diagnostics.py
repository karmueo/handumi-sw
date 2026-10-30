"""按需追加夹爪视觉、通信和开度换算 JSONL 日志，供静止采样诊断。"""

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _diagnostic_logger() -> logging.Logger | None:
    """读取启动环境变量并打开独立日志文件，失败时警告并禁用诊断。"""
    # 诊断路径在启动进程前设置。
    destination = (os.environ.get("HANDUMI_GRIPPER_DIAGNOSTICS")
                   or os.environ.get("HANDUMI_FEETECH_DIAGNOSTICS"))
    if not destination:
        return None
    logger = logging.getLogger("handumi.feetech.diagnostics")
    try:
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        logging.getLogger("handumi.record").warning("Cannot open Gripper diagnostics: %s", exc)
        return None
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logging.getLogger("handumi.record").info("Gripper diagnostics: %s", path.resolve())
    return logger


def log_diagnostic(event: str, **fields: object) -> None:
    """追加带时间戳的事件；未启用时直接返回。"""
    logger = _diagnostic_logger()
    if logger is None:
        return
    # 单调时钟用于关联采样，墙钟及进程号用于区分复跑记录。
    record = dict(event=event, time_ns=time.time_ns(),
                  monotonic_ns=time.monotonic_ns(), pid=os.getpid(), **fields)
    logger.info(json.dumps(record, ensure_ascii=False, default=str))
