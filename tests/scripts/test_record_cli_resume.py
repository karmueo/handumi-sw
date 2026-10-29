"""验证录制入口按续录数据集和显式参数选择追踪设备。"""

import json
import sys

import pytest

from handumi.scripts import cli, record, umi_capture


@pytest.mark.parametrize(
    ("saved_device", "rig_device", "expected"),
    [
        ("meta", "vive", "legacy"),
        ("pico", "vive", "legacy"),
        ("vive", "meta", "vive"),
        ("vive", "pico", "vive"),
    ],
)
@pytest.mark.parametrize("equals_syntax", [False, True])
def test_resume_routes_using_dataset_device(
    tmp_path, monkeypatch, saved_device, rig_device, expected, equals_syntax
):
    """续录时已保存的设备优先于当前 rig，支持两种输出路径参数形式。"""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"handumi": {"recording_device": saved_device}})
    )
    rig = tmp_path / "rig.yaml"
    rig.write_text(f"recording:\n  device: {rig_device}\n")
    calls = []
    monkeypatch.setattr(record, "main", lambda: calls.append(("legacy", sys.argv[1:])))

    def fake_umi_main(argv, *, mode):
        """记录 Vive 入口收到的参数和调用模式。"""
        assert mode == "record"
        calls.append(("vive", argv))

    monkeypatch.setattr(umi_capture, "main", fake_umi_main)
    output_arg = (
        [f"--output-dir={root}"] if equals_syntax else ["--output-dir", str(root)]
    )
    arguments = ["--resume", *output_arg, "--rig-config", str(rig)]

    cli.main(["record", *arguments])

    assert calls == [(expected, arguments)]


@pytest.mark.parametrize(
    ("saved_device", "selected_device", "expected"),
    [("meta", "vive", "vive"), ("vive", "meta", "legacy")],
)
@pytest.mark.parametrize("equals_syntax", [False, True])
def test_resume_explicit_device_has_priority(
    tmp_path, monkeypatch, saved_device, selected_device, expected, equals_syntax
):
    """显式选择设备时仍交由对应入口检查与数据集的兼容性。"""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"handumi": {"recording_device": saved_device}})
    )
    calls = []
    monkeypatch.setattr(record, "main", lambda: calls.append("legacy"))
    monkeypatch.setattr(
        umi_capture, "main", lambda argv, *, mode: calls.append("vive")
    )
    device_arg = (
        [f"--device={selected_device}"]
        if equals_syntax
        else ["--device", selected_device]
    )

    cli.main([
        "record", "--resume", "--output-dir", str(root), *device_arg,
    ])

    assert calls == [expected]


@pytest.mark.parametrize("contents", [None, "{", "{}", '{"handumi": []}'])
def test_unreadable_resume_metadata_is_left_to_selected_entrypoint(
    tmp_path, monkeypatch, contents
):
    """元数据不可读取时仍将参数交给录制入口，以便其报告具体错误。"""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    if contents is not None:
        (root / "meta/info.json").write_text(contents)
    rig = tmp_path / "rig.yaml"
    rig.write_text("recording:\n  device: meta\n")
    calls = []
    monkeypatch.setattr(record, "main", lambda: calls.append("legacy"))

    cli.main(["record", "--resume", "--output-dir", str(root), "--rig-config", str(rig)])

    assert calls == ["legacy"]


def test_new_recording_uses_rig_even_when_output_contains_metadata(tmp_path, monkeypatch):
    """新录制按 rig 选设备，输出目录中旧数据的设备不会参与路由。"""
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"handumi": {"recording_device": "vive"}})
    )
    rig = tmp_path / "rig.yaml"
    rig.write_text("recording:\n  device: meta\n")
    calls = []
    monkeypatch.setattr(record, "main", lambda: calls.append("legacy"))

    cli.main(["record", "--output-dir", str(root), "--rig-config", str(rig)])

    assert calls == ["legacy"]
