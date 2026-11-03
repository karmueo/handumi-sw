"""验证桌面世界中的动态底座标注和关节数据的逐段摆放。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import viser

from handumi.scripts.replay import replay_in_sim as replay
from handumi.scripts.replay import replay_in_sim_joints as joints


def test_table_measurements_follow_base_and_replace_old_handles(monkeypatch):
    """坐标侧栏和基座标记使用桌面坐标，重建时删除旧节点与控件。"""
    server = viser.ViserServer(port=0, verbose=False)
    runtime = SimpleNamespace(
        active_sides=("right",), arm_joint_names=lambda side: ["joint1"]
    )
    urdf = SimpleNamespace(
        scene=None,
        joint_map={"joint1": SimpleNamespace(parent="base")},
        get_transform=lambda link: np.eye(4),
    )
    nodes, texts, folders = {}, [], []
    old_node, old_markdown, old_folder = (
        server.scene.add_icosphere,
        server.gui.add_markdown,
        server.gui.add_folder,
    )

    def capture_node(path, **kwargs):
        """记录基座参考点节点。"""
        node = old_node(path, **kwargs)
        nodes[path] = node
        return node

    def capture_text(text, **kwargs):
        """记录实际创建的侧栏文本。"""
        texts.append(text)
        return old_markdown(text, **kwargs)

    def capture_folder(*args, **kwargs):
        """记录可移除的坐标文件夹。"""
        folder = old_folder(*args, **kwargs)
        folders.append(folder)
        return folder

    monkeypatch.setattr(server.scene, "add_icosphere", capture_node)
    monkeypatch.setattr(server.gui, "add_markdown", capture_text)
    monkeypatch.setattr(server.gui, "add_folder", capture_folder)
    try:
        rollout = {"robot_from_table_pose7": np.array([[0, 0, 0, 0, 0, 0, 1]])}
        first = replay._render_table(
            server,
            runtime,
            urdf,
            rollout,
            base_pose7=np.array([0.2, -0.1, 0, 0, 0, 0, 1]),
        )
        np.testing.assert_allclose(nodes["/measure/reference"].position, [0.2, -0.1, 0])
        assert "基座原点（桌面坐标）" in texts[-1] and "X +0.200" in texts[-1]
        for handle in first:
            handle.remove()
        assert folders[0]._impl.removed
        replay._render_table(
            server,
            runtime,
            urdf,
            rollout,
            base_pose7=np.array([-0.1, 0.3, 0, 0, 0, 0, 1]),
        )
        np.testing.assert_allclose(nodes["/measure/reference"].position, [-0.1, 0.3, 0])
        assert "X -0.100" in texts[-1] and not folders[-1]._impl.removed
    finally:
        server.stop()


def test_joint_replay_reads_matching_source_episode_placement():
    """转换输出的 source episode 决定 base 摆放，不能使用全局参考覆盖。"""
    meta = {
        "deployment_calibration": {"per_episode": True, "path": None},
        "ik_fidelity": [
            {
                "source_episode_index": 1,
                "base_placement": {"robot_from_table_pose7": [0.1, 0, 0, 0, 0, 0, 1]},
            },
            {
                "source_episode_index": 5,
                "base_placement": {"robot_from_table_pose7": [0, 0.2, 0, 0, 0, 0, 1]},
            },
        ],
    }
    np.testing.assert_allclose(
        joints._deployment_pose7(meta, source_index=5), [0, 0.2, 0, 0, 0, 0, 1]
    )
    with pytest.raises(SystemExit, match="Missing per-episode"):
        joints._deployment_pose7(meta, source_index=3)


def test_joint_verification_preserves_requested_base_policy(monkeypatch, tmp_path):
    """重新求解时透传转换记录的策略，不使用新的默认设置。"""
    captured = []
    monkeypatch.setattr(
        joints,
        "resolve_dataset_selection",
        lambda *args, **kwargs: SimpleNamespace(repo_id="local/source", root=tmp_path),
    )
    monkeypatch.setattr(
        replay, "solve_episode", lambda args: captured.append(args) or {}
    )
    args = SimpleNamespace(
        revision="main", robot="rm75b", rig_config=tmp_path / "rig.yaml"
    )
    joints._solve_source_episode(
        args,
        {
            "retarget_mode": "absolute-tcp",
            "absolute_orientation": "table-absolute",
            "base_placement": "episode-xy",
        },
        tmp_path,
        4,
    )
    assert captured[0].base_placement == "episode-xy" and captured[0].episode == 4
