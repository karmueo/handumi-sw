"""验证录制命令的覆盖参数、目录保护和启动顺序，无需实际硬件。"""

from pathlib import Path

import pytest

from handumi.scripts import _record_output, record


def _args(root, *extra):
    """构造录制参数，省略无关的机器本地配置解析。"""
    return record.parse_args(["--output-dir", str(root), *extra])


@pytest.mark.parametrize("extra, expected", [([], True), (["true"], True), (["false"], False)])
def test_overwrite_argument(tmp_path, extra, expected):
    """默认开启覆盖，显式布尔值按约定解析。"""
    flags = ["--overwrite", *extra] if extra else []
    assert _args(tmp_path / "dataset", *flags).overwrite is expected


@pytest.mark.parametrize("flags", [["--overwrite", "yes"], ["--overwrite"], ["--no-overwrite"]])
def test_invalid_overwrite_argument(tmp_path, flags):
    """拒绝非法值、缺失值和未提供的否定开关。"""
    with pytest.raises(SystemExit) as error:
        _args(tmp_path / "dataset", *flags)
    assert error.value.code == 2


def test_overwrite_removes_entire_directory(tmp_path):
    """覆盖清理全部嵌套内容，保留兄弟目录和链接指向的外部文件。"""
    root = tmp_path / "dataset"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "old.txt").write_text("old")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep")
    (root / "link").symlink_to(sibling)
    record._prepare_output_directory(_args(root))
    assert not root.exists()
    assert sibling.read_text() == "keep"


def test_no_overwrite_preserves_directory(tmp_path):
    """关闭覆盖后报错并保留原始数据。"""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep").write_text("keep")
    with pytest.raises(SystemExit, match="already exists"):
        record._prepare_output_directory(_args(root, "--overwrite", "false"))
    assert (root / "keep").read_text() == "keep"


@pytest.mark.parametrize("overwrite", ["true", "false"])
def test_missing_directory_is_left_for_dataset_creation(tmp_path, overwrite):
    """缺失目录由数据集创建接口负责创建。"""
    root = tmp_path / "dataset"
    record._prepare_output_directory(_args(root, "--overwrite", overwrite))
    assert not root.exists()


@pytest.mark.parametrize("overwrite", ["true", "false"])
def test_resume_preserves_directory(tmp_path, overwrite):
    """追加模式忽略覆盖开关并保留已有内容。"""
    (tmp_path / "keep").write_text("keep")
    record._prepare_output_directory(_args(tmp_path, "--resume", "--overwrite", overwrite))
    assert (tmp_path / "keep").read_text() == "keep"


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling", "root", "home", "cwd", "parent"])
def test_unsafe_output_is_rejected(tmp_path, monkeypatch, kind):
    """危险目录、文件及符号链接均在清理前拒绝。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    target = tmp_path / "target"
    if kind == "file":
        target.write_text("keep")
    elif kind in {"symlink", "dangling"}:
        target.symlink_to(home if kind == "symlink" else tmp_path / "missing")
    else:
        target = {"root": Path("/"), "home": home, "cwd": workspace, "parent": tmp_path}[kind]
    with pytest.raises(SystemExit):
        record._prepare_output_directory(_args(target))
    assert target.exists() or target.is_symlink()


def _mock_startup(monkeypatch, args):
    """替换机器配置解析，并阻止测试意外连接硬件。"""
    args.no_video = True
    args.fps = 30
    args.cam_width = 640
    args.cam_height = 480
    args.active_sides = ("left", "right")
    args.cam_ids = []
    args._resume_handumi = None
    monkeypatch.setattr(record, "parse_args", lambda: args)
    monkeypatch.setattr(record, "_resolve_recording_args", lambda value: value)
    monkeypatch.setattr(record, "_validate_args", lambda _: None)
    monkeypatch.setattr(record, "_selected_camera_names", lambda _: [])
    monkeypatch.setattr(record, "_robot_metadata", lambda _: {})
    monkeypatch.setattr(record, "_recording_tcp_calibration_metadata", lambda **_: ({}, "test"))
    monkeypatch.setattr(record, "_resolve_spatial_session_metadata", lambda _: None)
    monkeypatch.setattr(record, "resolve_camera_ids", lambda *a, **k: [])
    monkeypatch.setattr(record, "build_camera_specs", lambda *a, **k: ([], None))
    monkeypatch.setattr(record, "build_features", lambda *a, **k: {})
    monkeypatch.setattr(record, "_validate_resume_target", lambda *a, **k: None)
    monkeypatch.setattr(record, "_resume_handumi_metadata", lambda **k: {})

    def hardware_forbidden(*args, **kwargs):
        """硬件连接发生时立即令测试失败。"""
        pytest.fail("Hardware must not be opened")

    monkeypatch.setattr(record, "build_tracker", hardware_forbidden)


@pytest.mark.parametrize("exists, resume, mode", [(False, False, "new dataset"), (True, False, "overwrite entire directory"), (True, True, "resume")])
def test_dry_run_is_read_only(tmp_path, monkeypatch, capsys, exists, resume, mode):
    """预演输出目录计划，既不修改目录也不打开硬件。"""
    root = tmp_path / "dataset"
    if exists:
        root.mkdir()
        (root / "keep").write_text("keep")
    flags = ["--dry-run", *(["--resume"] if resume else [])]
    args = _args(root, *flags)
    _mock_startup(monkeypatch, args)
    record.main()
    assert mode in capsys.readouterr().out
    assert root.exists() is exists
    if exists:
        assert (root / "keep").read_text() == "keep"


def test_cleanup_failure_stops_startup(tmp_path, monkeypatch):
    """删除失败时立即终止，避免继续连接硬件或创建数据集。"""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep").write_text("keep")
    _mock_startup(monkeypatch, _args(root))

    def fail_cleanup(path):
        """模拟权限错误，保留原始目录。"""
        raise PermissionError("denied")

    monkeypatch.setattr(_record_output.shutil, "rmtree", fail_cleanup)
    with pytest.raises(SystemExit, match="Cannot overwrite"):
        record.main()
    assert (root / "keep").read_text() == "keep"


@pytest.mark.parametrize("dry_run, overwrite, resume", [(True, "true", False), (False, "true", False), (False, "false", False), (False, "true", True), (False, "false", True)])
def test_vive_record_entrypoint(tmp_path, monkeypatch, dry_run, overwrite, resume):
    """Vive 实际入口遵循覆盖、追加及预演约定，硬件启动前完成目录处理。"""
    from handumi.scripts import umi_capture

    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep").write_text("keep")
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text("{}")
    monkeypatch.setattr(umi_capture, "verify", lambda *a, **k: {})
    monkeypatch.setattr(umi_capture, "metadata", lambda _: {})
    monkeypatch.setattr(umi_capture, "validate_resume_target", lambda *a: None)

    class HardwareReached(Exception):
        """标记目录处理完成后的硬件启动位置。"""

    def stop_at_hardware(*args, **kwargs):
        """在硬件连接前停止测试，验证目录已经按要求处理。"""
        raise HardwareReached

    monkeypatch.setattr(umi_capture, "NativeUmi", stop_at_hardware)
    flags = ["--device", "vive", "--output-dir", str(root), "--repo-id", "local/umi-demo",
             "--calibration", str(calibration), "--overwrite", overwrite, "--no-preview"]
    if resume:
        flags.append("--resume")
    if dry_run:
        umi_capture.main([*flags, "--dry-run"])
    elif overwrite == "false" and not resume:
        with pytest.raises(SystemExit, match="already exists"):
            umi_capture.main(flags)
    else:
        with pytest.raises(HardwareReached):
            umi_capture.main(flags)
    preserved = dry_run or resume or overwrite == "false"
    assert root.exists() is preserved
    if preserved:
        assert (root / "keep").read_text() == "keep"
