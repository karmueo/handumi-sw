"""验证扩展元数据日志过滤的范围及独立进程中的自动安装。"""

import logging
import subprocess
import sys

from lerobot.datasets.utils import DatasetInfo

from handumi.dataset._logging import configure_lerobot_logging


def test_handumi_extension_warning_is_suppressed(caplog):
    """读取已知扩展时保持标准元数据解析成功且不输出警告。"""
    info = DatasetInfo.from_dict({
        "codebase_version": "v3.0", "fps": 30, "features": {}, "handumi": {},
    })
    assert info.fps == 30
    assert not caplog.records


def test_other_metadata_warnings_and_errors_remain_visible(caplog):
    """未知扩展、混合字段、其他警告及错误仍可供用户诊断。"""
    for extras in ({"vendor_extra": {}}, {"handumi": {}, "vendor_extra": {}}):
        DatasetInfo.from_dict({
            "codebase_version": "v3.0", "fps": 30, "features": {}, **extras,
        })
    logger = logging.getLogger("lerobot.datasets.utils")  # 被过滤的上游日志。
    logger.warning("Other metadata warning")
    logger.error("Unknown fields in DatasetInfo: ['handumi']. These will be ignored.")
    logging.getLogger("another.module").warning(
        "Unknown fields in DatasetInfo: ['handumi']. These will be ignored."
    )
    assert len(caplog.records) == 5
    assert "vendor_extra" in caplog.records[0].message
    assert "vendor_extra" in caplog.records[1].message
    assert caplog.records[2].message == "Other metadata warning"
    assert caplog.records[3].levelno == logging.ERROR
    assert caplog.records[4].name == "another.module"


def test_logging_configuration_is_idempotent():
    """重复配置不会累积过滤器或改变上游日志级别。"""
    logger = logging.getLogger("lerobot.datasets.utils")  # 上游日志配置。
    original_filters = list(logger.filters)  # 首次导入后已安装的过滤器。
    original_level = logger.level  # 调用方设置的日志级别。
    configure_lerobot_logging()
    configure_lerobot_logging()
    assert logger.filters == original_filters
    assert logger.level == original_level


def test_dataset_import_installs_filter_in_fresh_process():
    """新解释器导入数据集后自动过滤，覆盖 spawn 工作进程的启动方式。"""
    result = subprocess.run(
        [sys.executable, "-c", "\n".join([
            "import logging",
            "import handumi.dataset.reader",
            "logger = logging.getLogger('lerobot.datasets.utils')",
            'logger.warning("Unknown fields in DatasetInfo: [\'handumi\']. These will be ignored.")',
            "logger.warning('Other metadata warning')",
        ])],
        capture_output=True, text=True, check=True,
    )
    assert "Unknown fields" not in result.stderr
    assert "Other metadata warning" in result.stderr
