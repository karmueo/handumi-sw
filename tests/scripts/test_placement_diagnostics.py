"""测试 RM75B 腕心保守包络及其 URDF 几何读取。"""

from pathlib import Path

import numpy as np
import pytest

from handumi.scripts.replay.diagnose_placement import arm_geometry, wrist_envelope


def test_rm75b_wrist_envelope_includes_rotation_tolerance():
    """工具偏置较长时，朝向误差对腕心位置的影响必须计入。"""
    urdf = Path(__file__).resolve().parents[2] / "assets/rm75b/rm75b.urdf"
    geometry = arm_geometry(urdf)
    assert geometry[0] == pytest.approx(0.2405)
    assert geometry[1] == pytest.approx(0.466)
    assert geometry[2] == pytest.approx(0.18357, abs=0.0001)
    tool = np.eye(4)
    tool[2, 3] = 0.226
    target = np.eye(4)[None, :, :]
    wrist, height, outer, inner, tolerance = wrist_envelope(target, tool, geometry)
    np.testing.assert_allclose(wrist[0], [0, 0, -0.370])
    assert tolerance > 0.037
    assert outer > 0.466
    assert inner < 0.18357
    assert height == pytest.approx(0.2405)


def test_diagnostics_continues_when_search_fails(monkeypatch, tmp_path):
    """完整搜索失败时仍分析源轨迹并记录有限搜索不确定。"""
    import json
    import sys
    from types import SimpleNamespace

    from handumi.scripts.replay import diagnose_placement as diag

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["diagnose", "dataset", "--episode", "7", "--output", str(output)]
    )
    monkeypatch.setattr(
        diag.replay,
        "resolve_dataset_selection",
        lambda *a, **k: SimpleNamespace(repo_id="x", root=tmp_path),
    )
    monkeypatch.setattr(
        diag,
        "resolve_deployment_calibration",
        lambda *a, **k: SimpleNamespace(
            pose7=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        ),
    )
    monkeypatch.setattr(
        diag.replay, "load_episode_states", lambda *a: (np.zeros((2, 16)), 30, {}, None)
    )
    monkeypatch.setattr(
        diag.replay, "_metadata_tcp_snapshot", lambda *a: {"trusted": True}
    )
    monkeypatch.setattr(
        diag.replay, "_metadata_tcp_calibration", lambda *a: ("calibration", "source")
    )
    monkeypatch.setattr(diag.replay, "_is_trusted_tcp_snapshot", lambda *a: True)
    monkeypatch.setattr(
        diag.replay, "handumi_metadata", lambda *a: {"tracking_workspace": "table"}
    )
    poses = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (2, 1))
    monkeypatch.setattr(
        diag.replay,
        "apply_tcp_calibration_to_states",
        lambda *a: (None, None, None, None, poses, None),
    )

    def failed(*args):
        """模拟有限预算搜索失败。"""
        raise SystemExit("bounded search failed")

    monkeypatch.setattr(diag, "prepare_episode", failed)
    runtime = SimpleNamespace(
        config=SimpleNamespace(arms={}), urdf_path="unused", load_urdf=lambda **kw: None
    )
    monkeypatch.setattr(diag, "load_embodiment", lambda *a: runtime)
    monkeypatch.setattr(
        diag,
        "load_robot_tcp_calibration",
        lambda *a: SimpleNamespace(matrices={"right": np.eye(4)}),
    )
    monkeypatch.setattr(diag, "arm_geometry", lambda *a: (0.2405, 0.466, 0.183, 0.144))
    monkeypatch.setattr(
        diag, "independent_ik", lambda *a: [{"frame": 0, "passed": True}]
    )
    diag.main()
    report = json.loads(output.read_text())
    assert report["classification"] == "bounded_search_inconclusive"
    assert report["search_failure"] == "bounded search failed"
    assert report["cases"][0]["independent_ik"][0]["passed"]
    assert not output.with_suffix(".npz").exists()

    def cached_failure(*args):
        """模拟只有摘要的旧失败缓存。"""
        raise SystemExit("Unable to find strict episode XY placement: cached")

    rollout = {
        "table_from_robot_pose7": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "left_pos_error_m": np.zeros(2),
        "right_pos_error_m": np.zeros(2),
        "left_rot_error_deg": np.zeros(2),
        "right_rot_error_deg": np.zeros(2),
        "base_placement_diagnostics_json": np.array(["{}"]),
    }
    monkeypatch.setattr(diag, "prepare_episode", cached_failure)
    monkeypatch.setattr(diag.replay, "solve_episode", lambda *a: rollout)
    diag.main()
    report = json.loads(output.read_text())
    assert report["classification"] == "continuous_strictly_feasible"
    assert report["search_failure"] is None
    assert output.with_suffix(".npz").exists()


def test_outer_geometry_certificate_is_a_sufficient_disproof():
    """仅垂直超限或互不相交的外圆盘输出几何反证。"""
    from handumi.scripts.replay.diagnose_placement import (
        outer_impossibility_certificate,
    )

    assert (
        outer_impossibility_certificate(
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), 0.0, 1.0
        )
        is None
    )
    vertical = outer_impossibility_certificate(np.array([[0.0, 0.0, 1.1]]), 0.0, 1.0)
    assert vertical["kind"] == "vertical"
    disjoint = outer_impossibility_certificate(
        np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]]), 0.0, 1.0
    )
    assert disjoint["kind"] == "disjoint_outer_disks"
