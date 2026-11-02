"""验证 UMI 续录在打开设备前拒绝空数据集和缺失的本地元数据。"""

import json

import pytest

from handumi.scripts import umi_capture


def test_empty_resume_stops_before_devices(tmp_path, monkeypatch):
    """空数据集给出可执行提示，且不初始化设备或预览。"""
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"total_episodes": 0}))
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})

    def unexpected_device(*args, **kwargs):
        """打开设备意味着启动前校验失效。"""
        pytest.fail("不应打开设备")

    monkeypatch.setattr(umi_capture, "NativeUmi", unexpected_device)
    with pytest.raises(SystemExit) as error:
        umi_capture.main([
            "--resume", "--output-dir", str(tmp_path),
            "--calibration", str(calibration), "--no-preview",
        ])
    assert error.value.code == 2
    assert list((tmp_path / "meta").iterdir()) == [tmp_path / "meta/info.json"]


def test_resume_metadata_validation(tmp_path):
    """缺失的元数据和标定不一致应报错，完整且匹配时允许继续。"""
    with pytest.raises(ValueError, match="无法读取"):
        umi_capture.validate_resume_target(tmp_path, {})
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({
        "total_episodes": 1, "handumi": {"calibration": "original"},
    }))
    with pytest.raises(ValueError, match="tasks.parquet"):
        umi_capture.validate_resume_target(tmp_path, {})
    (tmp_path / "meta/tasks.parquet").touch()
    with pytest.raises(ValueError, match="meta/episodes"):
        umi_capture.validate_resume_target(tmp_path, {})
    (tmp_path / "meta/episodes").mkdir()
    (tmp_path / "meta/episodes/file-000.parquet").touch()
    with pytest.raises(ValueError, match="旧数据集未保存 TCP"):
        umi_capture.validate_resume_target(tmp_path, {})
    (tmp_path / "meta/info.json").write_text(json.dumps({
        "total_episodes": 1, "handumi": {"calibration": "original"},
        "features": {"observation.tracking.right_tcp_pose": umi_capture.pose7_feature()},
    }))
    with pytest.raises(ValueError, match="标定不一致"):
        umi_capture.validate_resume_target(tmp_path, {})
    umi_capture.validate_resume_target(tmp_path, {"calibration": "original"})


@pytest.mark.parametrize("feature", [
    {},
    {"dtype": "float64", "shape": [7], "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
    {"dtype": "float32", "shape": [8], "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
    {"dtype": "float32", "shape": [7], "names": ["bad"]},
    "invalid",
])
def test_resume_rejects_invalid_tcp_feature(tmp_path, feature):
    """字段缺少属性或类型、形状、名称错误时禁止续录。"""
    (tmp_path / "meta/episodes").mkdir(parents=True)
    (tmp_path / "meta/tasks.parquet").touch()
    (tmp_path / "meta/episodes/file-000.parquet").touch()
    (tmp_path / "meta/info.json").write_text(json.dumps({
        "total_episodes": 1, "handumi": {},
        "features": {"observation.tracking.right_tcp_pose": feature},
    }))
    with pytest.raises(ValueError, match="TCP 位姿字段"):
        umi_capture.validate_resume_target(tmp_path, {})


def test_legacy_resume_fails_before_device_preview_and_directory_changes(tmp_path, monkeypatch):
    """旧数据缺少 TCP 列时，启动前保留原数据并拒绝创建设备和预览。"""
    output = tmp_path / "dataset"
    (output / "meta/episodes").mkdir(parents=True)
    (output / "meta/tasks.parquet").touch()
    (output / "meta/episodes/file-000.parquet").touch()
    info_path = output / "meta/info.json"
    original = json.dumps({"total_episodes": 1, "handumi": {}})
    info_path.write_text(original)
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "metadata", lambda *a: {})
    monkeypatch.setattr(umi_capture, "load_session", lambda *a, **k: {})

    def unexpected(*args, **kwargs):
        """续录预检失败时不得触发后续副作用。"""
        pytest.fail("续录预检之后的设备或目录操作不应执行")

    monkeypatch.setattr(umi_capture, "NativeUmi", unexpected)
    monkeypatch.setattr(umi_capture, "_start_camera_viewer", unexpected)
    monkeypatch.setattr(umi_capture, "_prepare_output_directory", unexpected)
    with pytest.raises(SystemExit) as error:
        umi_capture.main([
            "--resume", "--output-dir", str(output), "--calibration", str(calibration),
        ])
    assert error.value.code == 2
    assert info_path.read_text() == original
