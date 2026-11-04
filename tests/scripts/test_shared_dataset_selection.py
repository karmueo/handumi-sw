"""验证 QA、回放和转换对数据集共享摆放的选择及缓存身份。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from handumi.dataset.screening import _replay_args, solver_signature
from handumi.scripts import conversion
from handumi.scripts import screen_dataset as screen_cli
from handumi.scripts.qa_dataset import build_parser as qa_parser
from handumi.scripts.qa_dataset import review_steps
from handumi.scripts.replay import replay_deployment, replay_in_sim, replay_prepare


@pytest.fixture
def shared_root(tmp_path: Path) -> Path:
    """建立与共享搜索产物同格式、仅用于测试的数据集目录。"""
    path = tmp_path / ".handumi" / "base_placement" / "rm75b.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "kind": "handumi_robot_table_calibration",
        "robot": "rm75b", "scope": "simulation", "verified": False,
        "source": "dataset_xy_yaw_search",
        "calibration": {"robot_from_table": {
            "position": [0.5, 0.2, 0.0], "quaternion": [0, 0, 0, 1],
        }},
    }), encoding="utf-8")
    return tmp_path


def _replay(root: Path, *options: str):
    """构造回放参数，保持入口的解析顺序。"""
    args = replay_in_sim.build_parser().parse_args([str(root), "--robot", "rm75b", *options])
    args.dataset_root = root
    args.repo_id = "local/test"
    replay_deployment.select_saved_dataset_placement(args, allow_explicit=True)
    replay_in_sim.resolve_replay_options(args)
    return args


def test_auto_uses_same_shared_layout_and_strict_defaults(shared_root):
    """自动模式采用搜索布局，严格门槛与重复解析后均保持一致。"""
    args = _replay(shared_root)
    assert (args.retarget_mode, args.base_placement) == ("absolute-tcp", "dataset-xy-yaw")
    assert args.deployment_selection.profile == "dataset-xy-yaw"
    assert args.deployment_selection.path == shared_root / ".handumi/base_placement/rm75b.yaml"
    assert (args.initial_position_tolerance_m, args.max_ik_position_error_m,
            args.max_ik_rotation_error_deg, args.absolute_orientation) == (0.005, 0.005, 5, "table-absolute")
    replay_in_sim.resolve_replay_options(args)
    assert (args.initial_position_tolerance_m, args.max_ik_position_error_m,
            args.max_ik_rotation_error_deg) == (0.005, 0.005, 5)


def test_explicit_thresholds_survive_auto_selection_and_repeated_resolution(shared_root):
    """显式阈值即使等于另一模式默认值，也不能被自动识别为可覆盖默认值。"""
    args = _replay(shared_root, "--initial-position-tolerance-m", "0.01",
                   "--max-ik-position-error-m", "0.03", "--max-ik-rotation-error-deg", "45")
    replay_in_sim.resolve_replay_options(args)
    assert (args.initial_position_tolerance_m, args.max_ik_position_error_m,
            args.max_ik_rotation_error_deg) == (0.01, 0.03, 45)


@pytest.mark.parametrize("options", [
    ["--base-placement", "fixed"],
    ["--retarget-mode", "absolute-table"],
    ["--retarget-mode", "local-relative"],
    ["--deployment-calibration", "custom.yaml"],
    ["--deployment-profile", "local"],
])
def test_explicit_options_take_precedence(shared_root, options):
    """显式模式、布局和实体配置不会被共享仿真布局覆盖。"""
    args = _replay(shared_root, *options)
    assert args.base_placement != "dataset-xy-yaw"
    assert not hasattr(args, "deployment_selection")


def test_auto_preserves_local_physical_priority_and_sim_ignores_it(shared_root):
    """auto 选择本地实体标定，显式 sim 使用已保存仿真布局。"""
    local = shared_root / "physical.yaml"
    local.write_text("scope: physical\n")
    rig = shared_root / "rig.yaml"
    rig.write_text(yaml.safe_dump({"deployment": {"table_calibrations": {"rm75b": str(local)}}}))
    auto = _replay(shared_root, "--rig-config", str(rig))
    assert auto.base_placement == "auto" and auto.retarget_mode == "auto"
    sim = _replay(shared_root, "--rig-config", str(rig), "--deployment-profile", "sim")
    assert sim.base_placement == "dataset-xy-yaw"


def test_absent_or_invalid_shared_layout(shared_root):
    """缺失时维持原有默认；布局声明错误时明确报错，不退回粗估。"""
    path = shared_root / ".handumi/base_placement/rm75b.yaml"
    original = path.read_text()
    path.unlink()
    assert _replay(shared_root).base_placement == "auto"
    with pytest.raises(SystemExit, match="has not been resolved"):
        _replay(shared_root, "--retarget-mode", "absolute-tcp", "--base-placement", "dataset-xy-yaw")
    payload = yaml.safe_load(original)
    for key, invalid in (("robot", "piper"), ("scope", "physical"), ("source", "other")):
        broken = dict(payload)
        broken[key] = invalid
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(SystemExit):
            _replay(shared_root)


def test_qa_passes_explicit_shared_settings_to_screen():
    """QA 原样传递模式、布局、标定文件及三个精度参数。"""
    args = qa_parser().parse_args([
        "ds", "--robot", "rm75b", "--retarget-mode", "absolute-tcp",
        "--base-placement", "dataset-xy-yaw", "--controller-tcp-calibration", "controller.yaml",
        "--robot-tcp-calibration", "robot.yaml", "--initial-position-tolerance-m", "0.01",
        "--max-position-error-m", "0.02", "--max-rotation-error-deg", "9",
    ])
    command = next(command for _, command in review_steps(args, "ds") if "screen" in command)
    for flag, expected in (
        ("--retarget-mode", "absolute-tcp"), ("--base-placement", "dataset-xy-yaw"),
        ("--controller-tcp-calibration", "controller.yaml"), ("--robot-tcp-calibration", "robot.yaml"),
        ("--initial-position-tolerance-m", "0.01"), ("--max-position-error-m", "0.02"),
        ("--max-rotation-error-deg", "9.0"),
    ):
        assert command[command.index(flag) + 1] == expected


def test_four_entrypoints_share_identical_cache_identity(shared_root, monkeypatch):
    """默认回放、显式回放、筛选和转换按同一参数及标定生成相同缓存键。"""
    from handumi.calibration import robot_tcp
    from handumi.robots import registry

    tcp_path = shared_root / "robot-tcp.yaml"
    tcp_path.write_text("test")
    tool = SimpleNamespace(path=tcp_path, sha256="test", verified=False,
                           matrices={"right": np.eye(4)}, ee_links={"right": "Link7"})
    cfg = SimpleNamespace(arms={"right": SimpleNamespace(ee_link="Link7", joint_names=("joint1",))},
                          ik_weights=SimpleNamespace(collision_enabled=False),
                          urdf=shared_root / "robot.urdf", gripper_max_width_m=0.036)
    monkeypatch.setattr(robot_tcp, "load_robot_tcp_calibration", lambda *args, **kwargs: tool)
    monkeypatch.setattr(registry, "load_robot_config", lambda *args: cfg)
    monkeypatch.setattr(replay_prepare, "_source_calibration", lambda *args: {"source": "test"})
    monkeypatch.setattr(replay_in_sim, "_resolved_controller_device", lambda *args: "vive")
    monkeypatch.setattr(conversion, "load_robot_config", lambda *args: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setattr(conversion, "calibration_path_for_device", lambda *args: tcp_path)
    monkeypatch.setattr(conversion, "controller_tcp_calibration_metadata", lambda *args, **kwargs: {"sha256": "test"})
    monkeypatch.setattr(conversion, "load_controller_tcp_calibration", lambda *args: object())
    info = {"handumi": {"tracking_workspace": "table", "recording_device": "vive"}}
    loaded = (np.zeros((2, 16)), 30.0, info, None)
    default = _replay(shared_root)
    explicit = _replay(shared_root, "--retarget-mode", "absolute-tcp", "--base-placement", "dataset-xy-yaw")
    screened = _replay_args(root=shared_root, repo_id="local/test", revision="main", robot="rm75b",
                            episode=0, deployment_profile="sim", rig_config=shared_root / "missing-rig.yaml")
    parser = conversion.build_parser()
    converted = parser.parse_args([str(shared_root), "--robot", "rm75b"])
    converted.root, converted.repo_id = shared_root, "local/test"
    replay_deployment.select_saved_dataset_placement(converted, allow_explicit=True)
    conversion._resolve_cli_profile(parser, converted)
    conversion._resolve_retarget_mode(parser, converted, info)
    conversion._resolve_conversion_tcp_calibration(converted, info)
    assert converted.controller_device == "vive"
    converted = conversion._replay_args_for_episode(converted, 0)
    entries = [default, explicit, screened, converted]
    identities = [replay_prepare.episode_identity(args, loaded) for args in entries]
    assert all(item == identities[0] for item in identities)
    # 筛选门禁和转换均在签名前规范化实际设备，回放也应采用相同身份。
    for args in entries:
        args.controller_device = replay_in_sim._resolved_controller_device(args, info)
    signatures = [solver_signature(args, deployment_path=args.deployment_selection.path, source_info=info)
                  for args in entries]
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("options,position,rotation,initial", [
    ([], 0.005, 5.0, None),
    (["--max-position-error-m", "0.03", "--max-rotation-error-deg", "45",
      "--initial-position-tolerance-m", "0.01"], 0.03, 45.0, 0.01),
])
def test_screen_cli_builds_defaults_after_shared_selection(
    shared_root, monkeypatch, options, position, rotation, initial,
):
    """筛选 CLI 在选定实际模式后确定默认门槛，并保留显式阈值。"""
    (shared_root / "meta").mkdir()
    monkeypatch.setattr(screen_cli, "resolve_dataset_selection", lambda *args, **kwargs:
                        SimpleNamespace(root=shared_root, repo_id="local/test"))
    monkeypatch.setattr("sys.argv", ["screen", str(shared_root), "--robot", "rm75b", *options])
    seen = {}  # 传给实际筛选入口的参数。

    def run(root, **kwargs):
        """记录最终参数并提供可写出的空报告。"""
        seen.update(kwargs)
        return {"summary": {"total": 0, "accepted": 0, "rejected": 0,
                            "review_episode_indices": [], "reject_episode_indices": []}}

    monkeypatch.setattr(screen_cli, "screen_dataset", run)
    monkeypatch.setattr(screen_cli, "write_screening_report", lambda *args: None)
    monkeypatch.setattr(screen_cli, "render_screening_markdown", lambda payload: "report")
    screen_cli.main()
    assert seen["retarget_mode"] == "absolute-tcp"
    assert seen["base_placement"] == "dataset-xy-yaw"
    assert seen["deployment_selection"].source == "dataset_xy_yaw_search"
    assert seen["config"].max_position_error_m == position
    assert seen["config"].max_rotation_error_deg == rotation
    assert seen["initial_position_tolerance_m"] == initial


@pytest.mark.parametrize("configured", [False, True])
def test_screen_passes_one_shared_selection_to_all_workers(shared_root, monkeypatch, configured):
    """筛选按默认或显式 API 配置设置门槛，所有 worker 接收同一共享布局。"""
    from handumi.calibration import robot_tcp
    from handumi.dataset import screening
    from handumi.robots import registry

    (shared_root / "meta").mkdir()
    (shared_root / "meta/info.json").write_text(json.dumps({"total_episodes": 2, "fps": 30}))
    cfg = screening.RetargetScreeningConfig(max_position_error_m=0.02, max_rotation_error_deg=8)
    tool_path = shared_root / "robot-tcp.yaml"
    tool_path.write_text("test")
    tool = SimpleNamespace(path=tool_path, metadata=lambda: {})
    monkeypatch.setattr(robot_tcp, "load_robot_tcp_calibration", lambda *args, **kwargs: tool)
    monkeypatch.setattr(registry, "load_robot_config", lambda *args: SimpleNamespace(arms={}))
    monkeypatch.setattr(screening, "solver_signature", lambda *args, **kwargs: "same-signature")
    monkeypatch.setattr(screening, "robot_fingerprint", lambda *args, **kwargs: {})
    monkeypatch.setattr(screening, "dataset_payload_manifest", lambda root: {})
    seen = {}  # 提供给每个工作进程的共同参数。

    def map_tasks(episodes, **kwargs):
        """保留 worker 初始化参数，并返回两段可诊断的求解失败。"""
        seen.update(kwargs["setup_kwargs"])
        return [{"episode_index": index, "unreachable": "test", "frame_count": 0,
                 "metrics": {}, "resolved": {}, "solves": {}} for index in episodes]

    monkeypatch.setattr(screening, "map_episodes", map_tasks)
    report = screening.screen_dataset(
        shared_root, robot="rm75b", deployment_profile="sim", progress=False,
        jobs=2, config=cfg if configured else None,
    )
    expected = (0.02, 8) if configured else (0.005, 5)
    assert (report["config"]["max_position_error_m"], report["config"]["max_rotation_error_deg"]) == expected
    assert (seen["retarget_mode"], seen["base_placement"]) == ("absolute-tcp", "dataset-xy-yaw")
    assert seen["deployment_selection"].path == shared_root / ".handumi/base_placement/rm75b.yaml"
    assert seen["deployment_calibration"] is None
    assert (seen["max_position_error_m"], seen["max_rotation_error_deg"]) == expected
