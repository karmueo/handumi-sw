"""验证 RM75B 夹爪资源及其对七轴运动学的兼容性。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
import yourdfpy

from handumi.robots.registry import load_embodiment, yourdfpy_handler


@pytest.fixture(scope="module")
def runtime():
    """加载 RM75B 配置，供模型行为测试复用。"""
    return load_embodiment("rm75b")


def test_default_self_collision_margin_is_one_millimeter(runtime):
    """RM75B 默认保持一毫米自碰撞间距。"""
    assert runtime.config.ik_weights.collision_enabled
    assert runtime.config.ik_weights.self_collision_margin == 0.001


def test_gripper_meshes_are_loaded_with_meter_scale_and_follow_flange(runtime):
    """三个实际网格按米加载，并随法兰保持来源装配关系。"""
    urdf = runtime.load_urdf(load_meshes=True)
    mesh_links = {
        "gripper_link": "rm75_gripper_base.stl",
        "gripper_left_finger_link": "rm75_gripper_left.stl",
        "gripper_right_finger_link": "rm75_gripper_right.stl",
    }
    flange_to_gripper = np.eye(4)
    flange_to_gripper[2, 3] = 0.006
    for link, filename in mesh_links.items():
        # 获取该 link 在实际场景中加载的几何，避免只检查 XML 路径。
        nodes = urdf.scene.graph.transforms.children[link]
        geometry_nodes = [node for node in nodes if node in urdf.scene.geometry]
        assert len(geometry_nodes) == 1
        loaded = urdf.scene.geometry[geometry_nodes[0]]
        original = trimesh.load_mesh(runtime.urdf_path.parent / "meshes" / filename)
        assert len(loaded.vertices) > 0
        np.testing.assert_allclose(loaded.bounds, original.bounds * 0.001, atol=1e-8)
        np.testing.assert_allclose(
            urdf.get_transform(link, "Link7"), flange_to_gripper, atol=1e-9
        )

    # 用非零关节姿态检查父子关系，确保夹爪随机械臂运动。
    before = urdf.get_transform("gripper_link").copy()
    urdf.update_cfg(runtime.home_q())
    after = urdf.get_transform("gripper_link")
    assert not np.allclose(before, after)
    np.testing.assert_allclose(
        after, urdf.get_transform("Link7") @ flange_to_gripper, atol=1e-9
    )


def test_fixed_gripper_preserves_seven_axis_and_opening_independence(runtime):
    """夹指保持固定，开度输入不会改变七轴命令。"""
    assert runtime.joint_names == tuple(f"joint{i}" for i in range(1, 8))
    assert runtime.command_size == 7
    assert runtime.arm_joint_names("right") == list(runtime.joint_names)
    assert runtime.arms["right"].ee_link == "Link7"
    assert not runtime.finger_joints["right"]
    urdf = runtime.load_urdf()
    for name in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
        assert urdf.joint_map[name].type == "fixed"
    q = runtime.home_q()
    original = q.copy()
    for opening in (0.0, 0.5, 1.0):
        runtime.set_finger_positions(q, {"right": opening})
        np.testing.assert_array_equal(q, original)


def test_gripper_does_not_change_flange_fk(runtime, tmp_path):
    """移除夹爪前后，不同七轴姿态的 Link7 正运动学一致。"""
    tree = ET.parse(runtime.urdf_path)
    root = tree.getroot()
    for element in list(root):
        if element.get("name", "").startswith("gripper_"):
            root.remove(element)
    arm_path = tmp_path / "arm_only.urdf"
    tree.write(arm_path, encoding="unicode")
    arm = yourdfpy.URDF.load(
        str(arm_path),
        filename_handler=yourdfpy_handler(runtime.config.pkg_root),
        load_meshes=False,
    )
    full = runtime.load_urdf()
    for q in (np.zeros(7), runtime.home_q(), np.array([0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8])):
        arm.update_cfg(q)
        full.update_cfg(q)
        np.testing.assert_allclose(
            full.get_transform("Link7"), arm.get_transform("Link7"), atol=1e-9
        )
