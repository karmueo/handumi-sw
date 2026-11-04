"""验证缺少仿真布局时的数据估计、持久化以及标定优先级。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from handumi.calibration import deployment
from handumi.calibration.simulation_placement import fit_simulation_placement
from handumi.scripts.replay import replay_deployment, replay_in_sim, replay_prepare


def test_single_arm_alignment_estimates_yaw_and_xy_preserving_table_height():
    """已知刚性关系能恢复水平变换，轨迹高度和距离不被缩放。"""
    poses = np.array([[0.4, 0.2, 0.3, 0, 0, 0, 1], [0.6, 0.4, 0.5, 0, 0, 0, 1]])
    rotation = Rotation.from_euler("z", 0.7)  # 预期的桌面到机器人朝向。
    offset = np.array([0.2, -0.3, 0.0])  # 预期的水平平移。
    center = rotation.apply(np.median(poses[:, :3], axis=0)) + offset
    home = np.r_[center, rotation.as_quat()]
    fitted = fit_simulation_placement({"right": poses}, {"right": home})
    np.testing.assert_allclose(fitted[:3], offset, atol=1e-6)
    np.testing.assert_allclose(Rotation.from_quat(fitted[3:]).as_matrix(), rotation.as_matrix(), atol=1e-6)
    mapped = Rotation.from_quat(fitted[3:]).apply(poses[:, :3]) + fitted[:3]
    np.testing.assert_allclose(mapped[:, 2], poses[:, 2])
    assert np.linalg.norm(mapped[1] - mapped[0]) == pytest.approx(np.linalg.norm(poses[1, :3] - poses[0, :3]))
    np.testing.assert_array_equal(fitted, fit_simulation_placement({"right": poses}, {"right": home}))


def test_dual_arm_alignment_uses_arm_separation():
    """双臂按左右布局恢复朝向，保持两侧共享同一刚体变换。"""
    poses = {side: np.array([[x, 0, 0.3, 0, 0, 0, 1]])
             for side, x in (("left", -0.2), ("right", 0.2))}
    home = {"left": np.array([0.5, -0.2, 0.5, 0, 0, 0, 1]),
            "right": np.array([0.5, 0.2, 0.5, 0, 0, 0, 1])}
    fitted = fit_simulation_placement(poses, home)
    np.testing.assert_allclose(fitted[:3], [0.5, 0, 0], atol=1e-6)
    assert Rotation.from_quat(fitted[3:]).as_euler("xyz")[2] == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("poses", [np.empty((0, 7)), np.full((2, 7), np.nan), np.zeros((2, 7))])
def test_alignment_rejects_missing_or_invalid_pose_data(poses):
    """没有有效 TCP 时保留数据错误，不能写入看似有效的随机布局。"""
    with pytest.raises(ValueError, match="No valid"):
        fit_simulation_placement({"right": poses}, {"right": np.array([0, 0, 0, 0, 0, 0, 1])})


@pytest.mark.parametrize("profile", ["auto", "sim"])
def test_missing_simulation_uses_callback(tmp_path, monkeypatch, profile):
    """默认布局缺失时调用估计，并保持标准标定接口返回类型。"""
    monkeypatch.setattr(deployment, "SIM_CALIBRATION_DIR", tmp_path)
    expected = object()  # 回调返回值可直接传给调用方。
    selected = deployment.resolve_deployment_calibration(
        "rm75b", profile=profile, rig_config=tmp_path / "rig.yaml",
        simulation_fallback=lambda: expected,
    )
    assert selected is expected


def test_explicit_local_and_malformed_calibrations_do_not_use_estimation(tmp_path, monkeypatch):
    """显式路径、真实布局要求和配置内容错误均保留原有诊断。"""
    monkeypatch.setattr(deployment, "SIM_CALIBRATION_DIR", tmp_path)

    def unexpected():
        """这些错误不能触发估计。"""
        pytest.fail("Estimation must not replace explicit or malformed calibration")

    with pytest.raises(SystemExit, match="Missing deployment calibration"):
        deployment.resolve_deployment_calibration("rm75b", explicit_path=tmp_path / "missing.yaml", simulation_fallback=unexpected)
    with pytest.raises(SystemExit, match="No lab-local"):
        deployment.resolve_deployment_calibration("rm75b", profile="local", rig_config=tmp_path / "rig.yaml", simulation_fallback=unexpected)
    (tmp_path / "rm75b.yaml").write_text("kind: wrong\n")
    with pytest.raises(SystemExit, match="expected kind"):
        deployment.resolve_deployment_calibration("rm75b", profile="sim", simulation_fallback=unexpected)


def test_dataset_estimate_is_saved_and_reused(tmp_path, monkeypatch, capsys):
    """首次按真实输入标定估计，各段共享并复用同一仿真文件。"""
    args = replay_in_sim.build_parser().parse_args([str(tmp_path), "--robot", "rm75b"])
    args.dataset_root = tmp_path
    args.repo_id = "local/test"
    info = {"total_episodes": 2, "handumi": {"tracking_workspace": "table"}}
    states = np.zeros((3, 16))  # 合成有效控制器位姿。
    states[:, 6] = states[:, 13] = 1
    states[:, 7] = [0.1, 0.2, 0.3]
    reads = []  # 每个 episode 仅供首次估计读取。

    def load(trial):
        """提供每段相同的合成采集输入。"""
        reads.append(trial.episode)
        return states.copy(), 30, info, None

    def model_transform(link):
        """HOME 的目标工具位置位于机器人前方。"""
        matrix = np.eye(4)
        matrix[0, 3] = 0.5
        return matrix

    model = SimpleNamespace(update_cfg=lambda values: None, get_transform=model_transform)
    runtime = SimpleNamespace(
        load_urdf=lambda **kwargs: model, joint_names=("joint1",),
        config=SimpleNamespace(home_q=np.zeros(1), arms={"right": SimpleNamespace(ee_link="tool")}),
    )
    monkeypatch.setattr("handumi.dataset.ensure_metadata", lambda ref: info)
    monkeypatch.setattr("handumi.robots.registry.load_embodiment", lambda robot: runtime)
    monkeypatch.setattr(replay_in_sim, "load_episode_states", load)
    monkeypatch.setattr(replay_in_sim, "_resolved_controller_device", lambda *a: "vive")
    monkeypatch.setattr(replay_prepare, "_source_calibration", lambda *a: {
        "left": [0, 0, 0, 0, 0, 0, 1], "right": [0.1, 0, 0, 0, 0, 0, 1], "source": "test",
    })
    selected = replay_deployment.estimate_dataset_deployment(args)
    # 原始中心为 0.2，经 TCP 偏移后为 0.3，放到 HOME x=0.5 需平移 0.2。
    assert selected.pose7[0] == pytest.approx(0.2)
    assert selected.scope == "simulation" and not selected.verified
    assert selected.profile == "estimated"
    original = selected.path.read_bytes()  # 固定布局复用时不会重写。
    again = replay_deployment.estimate_dataset_deployment(args)
    assert reads == [0, 1]
    assert again.path.read_bytes() == original
    payload = yaml.safe_load(original)
    assert payload["estimation"]["ik_validated"] is False
    assert "only horizontal position and yaw" in capsys.readouterr().out


def test_screening_propagates_estimated_layout_to_all_workers(tmp_path, monkeypatch):
    """QA 先估计一次，再将文件传给 worker，并记录未实测的来源。"""
    import json

    from handumi.dataset import screening

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"total_episodes": 2, "fps": 30}))
    path = tmp_path / "estimated.yaml"  # 模拟已持久化的估计结果。
    path.write_text("estimated layout")
    selected = deployment.DeploymentCalibration(
        robot="rm75b", pose7=np.array([0.2, 0, 0, 0, 0, 0, 1]), path=path,
        scope="simulation", source="dataset_home_alignment_v1", verified=False,
        profile="estimated", lab=None,
    )
    calls = []  # 父进程估计次数及接收到的完整数据集路径。

    def estimate(args):
        """返回共享布局供并行流程测试。"""
        calls.append(args.dataset_root)
        return selected

    def solve(episodes, **kwargs):
        """验证 worker 全部收到同一显式文件；可达性失败仍逐段生成报告。"""
        assert kwargs["setup_kwargs"]["deployment_calibration"] == path
        return [{"episode_index": index, "unreachable": "unreachable test target",
                 "frame_count": 0, "metrics": {}, "resolved": {}, "solves": {}}
                for index in episodes]

    monkeypatch.setattr(deployment, "SIM_CALIBRATION_DIR", tmp_path / "sim")
    monkeypatch.setattr(replay_deployment, "estimate_dataset_deployment", estimate)
    monkeypatch.setattr(screening, "map_episodes", solve)
    report = screening.screen_dataset(tmp_path, robot="rm75b", deployment_profile="sim", progress=False, jobs=2)
    assert calls == [tmp_path]
    assert report["deployment_calibration"]["source"] == selected.source
    assert report["deployment_calibration"]["scope"] == "simulation"
    assert report["deployment_calibration"]["verified"] is False
    assert report["summary"]["rejected"] == 2
    assert "not physically verified" in screening.render_screening_markdown(report)
