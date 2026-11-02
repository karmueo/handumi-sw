"""适配 LeRobot 的数据集元数据类型并保留 HandUMI 扩展字段。"""

import json
from dataclasses import fields
from pathlib import Path

from lerobot.datasets.utils import DatasetInfo


class DatasetInfoWithExtras(DatasetInfo):
    """写回标准字段时同时保留 LeRobot 不认识的数据集扩展。"""

    def to_dict(self):
        """返回标准信息和原有扩展字段的 JSON 内容。"""
        return {**self._extra_fields, **super().to_dict()}


def preserve_dataset_info(dataset, updates=None):
    """保留已有自定义字段，并可覆盖新增或更新的字段。"""
    info = dataset.meta.info
    path = Path(dataset.root) / "meta" / "info.json"
    original = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    names = {field.name for field in fields(DatasetInfo)}
    extras = {key: value for key, value in original.items() if key not in names}
    if isinstance(info, DatasetInfoWithExtras):
        extras.update(info._extra_fields)
    if updates:
        extras.update({key: value for key, value in updates.items() if key not in names})
    wrapped = DatasetInfoWithExtras(**{name: getattr(info, name) for name in names})
    wrapped._extra_fields = extras
    dataset.meta.info = wrapped


class DatasetCardInfo:
    """向新版 LeRobot 数据卡片接口提供包含扩展字段的完整信息。"""

    def __init__(self, info):
        """保存待写入卡片的信息字典。"""
        self._info = info

    def to_dict(self):
        """返回未经丢弃字段处理的信息字典。"""
        return self._info
