"""验证 RM75B 双指开合资源及其对七轴运动学的兼容性。"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
import yourdfpy

from handumi.robots.registry import load_embodiment, yourdfpy_handler

GRIPPER_TRAVEL_M = 0.018134


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


def test_gripper_joints_use_bounded_prismatic_travel(runtime):
    """两指为有界平移关节，七轴接口和法兰保持不变。"""
    assert runtime.joint_names == tuple(f"joint{i}" for i in range(1, 8)) + (
        "gripper_left_finger_joint",
        "gripper_right_finger_joint",
    )
    assert runtime.home_q().shape == (9,)
    assert runtime.command_size == 7
    assert runtime.config.canonical_gripper_width_column is False
    assert runtime.arm_joint_names("right") == [f"joint{i}" for i in range(1, 8)]
    assert runtime.arms["right"].ee_link == "Link7"
    assert len(runtime.finger_joints["right"]) == 2
    urdf = runtime.load_urdf()
    for name, direction in (
        ("gripper_left_finger_joint", (-1.0, 0.0, 0.0)),
        ("gripper_right_finger_joint", (1.0, 0.0, 0.0)),
    ):
        joint = urdf.joint_map[name]
        assert joint.type == "prismatic"
        np.testing.assert_allclose(joint.axis, direction)
        assert joint.limit.lower == 0.0
        assert joint.limit.upper == pytest.approx(GRIPPER_TRAVEL_M)


@pytest.mark.parametrize(
    ("opening", "travel"),
    [(0.0, GRIPPER_TRAVEL_M), (0.5, GRIPPER_TRAVEL_M / 2), (1.0, 0.0)],
)
def test_gripper_mapping_is_linear_and_bidirectional(runtime, opening, travel):
    """开度 0 为最大内移、1 为 CAD 姿态，左右指沿相反 X 方向线性移动。"""
    q = runtime.home_q()
    runtime.set_finger_positions(q, {"right": opening})
    np.testing.assert_allclose(q[-2:], [travel, travel], atol=1e-7)
    urdf = runtime.load_urdf()
    urdf.update_cfg(q)
    left = urdf.get_transform("gripper_left_finger_link", "gripper_link")
    right = urdf.get_transform("gripper_right_finger_link", "gripper_link")
    np.testing.assert_allclose(left[:3, 3], [-travel, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(right[:3, 3], [travel, 0.0, 0.0], atol=1e-7)


def test_gripper_opening_is_clipped(runtime):
    """越界开度被裁剪到 0 和 1，不改变七轴命令。"""
    arm = runtime.home_q()[:7].copy()
    for opening, expected in ((-0.5, GRIPPER_TRAVEL_M), (1.5, 0.0)):
        q = runtime.home_q()
        runtime.set_finger_positions(q, {"right": opening})
        np.testing.assert_allclose(q[-2:], [expected, expected], atol=1e-7)
        np.testing.assert_array_equal(q[:7], arm)


def test_viewer_joint_order_uses_full_state(runtime):
    """查看器按九维实际关节构建，单臂命令仍保持七轴。"""
    sim = runtime.make_sim(port=0)
    assert sim._joint_names == list(runtime.joint_names)
    assert len(sim._joint_names) == 9
    assert sim._command_size == 7


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
    for arm_q in (
        np.zeros(7),
        runtime.home_q()[:7],
        np.array([0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8]),
    ):
        full_q = np.r_[arm_q, GRIPPER_TRAVEL_M, GRIPPER_TRAVEL_M]
        arm.update_cfg(arm_q)
        full.update_cfg(full_q)
        np.testing.assert_allclose(
            full.get_transform("Link7"), arm.get_transform("Link7"), atol=1e-9
        )
