"""验证筛选碰撞评分的局部坐标优化和模型上下文复用。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from handumi.dataset.screening import _mesh_separation
from handumi.scripts.replay.replay_prepare import EpisodePreparationContext


@pytest.mark.parametrize("shift", [2.0, 1.000001, 1.0, 0.999999, 0.3])
@pytest.mark.parametrize("angle", [0.0, 0.3])
def test_local_mesh_distance_matches_world_space(shift, angle):
    """相隔、近接触和穿透时，局部坐标距离与原世界坐标算法一致。"""
    mesh_a = trimesh.creation.box(extents=[1, 1, 1])
    mesh_b = trimesh.creation.box(extents=[1, 1, 1])
    transform_a = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    transform_a[:3, 3] = [0.3, -0.2, 0.4]
    relative = trimesh.transformations.rotation_matrix(-angle / 2, [0, 1, 0])
    relative[:3, 3] = [shift, 0, 0]
    transform_b = transform_a @ relative
    a, b = mesh_a.copy(), mesh_b.copy()
    a.apply_transform(transform_a)
    b.apply_transform(transform_b)
    expected = -max(
        trimesh.proximity.signed_distance(a, b.vertices).max(),
        trimesh.proximity.signed_distance(b, a.vertices).max(),
    )
    actual = _mesh_separation(mesh_a, transform_a, mesh_b, transform_b)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert (actual <= 0) == (expected <= 0)


def test_preparation_context_rebuilds_on_geometry_or_settings_change(monkeypatch):
    """机器人、网格或求解配置改变时，复用状态被丢弃。"""
    from handumi.robots import registry

    loads = []
    def load(name, **kwargs):
        """提供无碰撞模型，记录建立次数。"""
        runtime = SimpleNamespace(config=SimpleNamespace(ik_weights=SimpleNamespace(collision_enabled=False)))
        loads.append((name, kwargs, runtime))
        return runtime
    monkeypatch.setattr(registry, "load_embodiment", load)
    args = SimpleNamespace(robot="rm75b", retarget_mode="auto")
    identity = {"mode": "absolute-tcp", "robot_config_sha256": "config", "urdf_sha256": "urdf", "collision_meshes": {"a": "1"}, "settings": {"x": 1}}
    context = EpisodePreparationContext()
    first, _ = context.resources(args, identity)
    assert context.resources(args, identity)[0] is first
    assert loads[0][1] == {"collision_kind": "gripper-box"}
    second, _ = context.resources(args, {**identity, "collision_meshes": {"a": "2"}})
    assert second is not first
    third, _ = context.resources(args, {**identity, "settings": {"x": 2}})
    assert third is not second
    assert len(loads) == 3


def test_collision_scoring_reuses_frame_fk_between_pairs(monkeypatch):
    """多个重叠连杆对共享同一帧变换，完整保留原精查帧集合。"""
    from handumi.dataset import screening

    calls = []
    frame_count = 3
    self_fn, world_fn = object(), object()
    monkeypatch.setattr(
        screening, "_chunked",
        lambda fn, q: np.full((frame_count, 2), -0.1) if fn is self_fn
        else np.ones((frame_count, 1)),
    )
    def fk(_runtime, q):
        """记录当前帧并提供所有相关连杆位姿。"""
        calls.append(int(q[0]))
        return {name: np.eye(4) for name in ("a", "b", "c")}
    monkeypatch.setattr(screening, "_link_transforms", fk)
    monkeypatch.setattr(screening, "_mesh_separation", lambda *a, **kw: 0.1)
    collision = SimpleNamespace(link_names=["a", "b", "c"],
                                active_idx_i=[0, 0], active_idx_j=[1, 2])
    qpos = np.arange(frame_count).reshape(-1, 1)
    result = screening._collision_metrics(
        qpos, runtime=None, coll=collision, plane=None, pedestal=set(),
        world_fn=world_fn, self_fn=self_fn, meshes={name: object() for name in "abc"})
    assert sorted(calls) == [0, 1, 2]
    assert result["self_collision_frames"] == 0
    assert result["self_collision_capsule_frames"] == frame_count
