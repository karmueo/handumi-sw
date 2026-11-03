"""安全持久化逐段求解结果，检测损坏并以原子替换避免半成品。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 1
META_FIELD = "__handumi_cache_meta__"


def digest_bytes(data: bytes) -> str:
    """返回字节内容的 SHA-256 摘要。"""
    return hashlib.sha256(data).hexdigest()


def digest_file(path: str | Path | None) -> str | None:
    """返回文件内容摘要；缺失时返回 None。"""
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(value: np.ndarray | None) -> dict[str, Any] | None:
    """摘要数组的形状、类型与内容，区分缺失值。"""
    if value is None:
        return None
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ValueError("缓存输入不支持 object 数组")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": digest_bytes(array.tobytes()),
    }


def fingerprint(payload: dict[str, Any]) -> str:
    """对已解析的求解语义生成与键顺序无关的摘要。"""
    return digest_bytes(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


class EpisodeSolveCache:
    """按内容键读写完整 rollout 或确定性失败，不使用 pickle。"""

    def __init__(self, directory: Path):
        """配置缓存目录；目录在首次写入时创建。"""
        self.directory = Path(directory)

    def path(self, key: str) -> Path:
        """返回摘要对应的文件路径。"""
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("无效的缓存摘要")
        return self.directory / key[:2] / f"{key}.npz"

    def load(self, key: str) -> tuple[dict[str, np.ndarray] | None, str | None]:
        """读取并验证结果；缺失或损坏均返回未命中。"""
        try:
            with np.load(self.path(key), allow_pickle=False) as archive:
                meta = json.loads(str(archive[META_FIELD].item()))
                if not isinstance(meta, dict):
                    return None, None
                if (
                    meta.get("version") != CACHE_SCHEMA_VERSION
                    or meta.get("key") != key
                ):
                    return None, None
                failure = meta.get("failure")
                if failure is not None:
                    return (None, failure) if isinstance(failure, str) else (None, None)
                manifest = meta["arrays"]
                if not isinstance(manifest, dict) or set(archive.files) != set(
                    manifest
                ) | {META_FIELD}:
                    return None, None
                values = {name: np.asarray(archive[name]) for name in manifest}
                if any(
                    array_digest(values[name]) != expected
                    for name, expected in manifest.items()
                ):
                    return None, None
                return values, None
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            EOFError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
            zlib.error,
        ):
            return None, None

    def store(
        self,
        key: str,
        rollout: dict[str, np.ndarray] | None = None,
        *,
        failure: str | None = None,
    ) -> None:
        """完整写入临时文件并原子替换；失败时不留下目标半文件。"""
        if (rollout is None) == (failure is None):
            raise ValueError("必须且只能指定 rollout 或确定性失败")
        arrays = (
            {}
            if rollout is None
            else {name: np.asarray(value) for name, value in rollout.items()}
        )
        if META_FIELD in arrays:
            raise ValueError("rollout 使用了缓存保留字段")
        manifest = {name: array_digest(value) for name, value in arrays.items()}
        metadata = {
            "version": CACHE_SCHEMA_VERSION,
            "key": key,
            "arrays": manifest,
            "failure": failure,
        }
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".pending-", suffix=".npz", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    **arrays,
                    **{META_FIELD: np.asarray(json.dumps(metadata, sort_keys=True))},
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
