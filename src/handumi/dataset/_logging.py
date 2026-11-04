"""过滤 LeRobot 对 HandUMI 已知扩展字段的重复警告。"""

import logging


def _keep_lerobot_message(record: logging.LogRecord) -> bool:
    """仅隐藏针对单个 handumi 扩展字段的警告，保留其他日志。"""
    return not (
        record.levelno == logging.WARNING
        and record.getMessage()
        == "Unknown fields in DatasetInfo: ['handumi']. These will be ignored."
    )


def configure_lerobot_logging() -> None:
    """为当前进程安装一次精确过滤，不加载 LeRobot 或改变日志级别。"""
    logger = logging.getLogger("lerobot.datasets.utils")  # 上游元数据解析日志。
    if _keep_lerobot_message not in logger.filters:
        logger.addFilter(_keep_lerobot_message)
