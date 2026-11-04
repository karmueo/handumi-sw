"""验证关节回放的 episode 切换、失败回退与显示同步。"""

from types import SimpleNamespace

import numpy as np
import pytest
import viser
import viser.extras

from handumi.scripts.replay import replay_in_sim_joints as joints
from handumi.scripts.replay import replay_scheduler
from handumi.visualize import gripper_curve


class StopViewer(Exception):
    """结束测试中的查看器循环。"""


def rollout(episode, frames, base=None):
    """生成具有可区分帧数、轨迹、夹爪和摆放的轨迹。"""
    poses = np.zeros((frames, 7))
    poses[:, 0] = episode
    poses[:, 6] = 1
    return {
        "qpos": np.full((frames, 1), episode),
        "approach_qpos": np.empty((0, 1)),
        "gripper_normalized": np.full((frames, 2), (episode + 1) / 10),
        "fps": np.array([30 + episode]),
        "retarget_mode": np.array(["absolute-tcp"]),
        "table_from_robot_pose7": np.array([base]) if base else np.empty((0, 7)),
        "robot_from_table_pose7": np.empty((0, 7)),
        "fk_left_pose7_robot_world": poses.copy(),
        "fk_right_pose7_robot_world": poses.copy(),
    }


@pytest.mark.parametrize("render_failure", [False, True])
def test_switch_updates_view_and_failure_keeps_current(monkeypatch, render_failure):
    """暂停时切换单帧数据，验证坐标、曲线和失败回退。"""
    server = viser.ViserServer(port=0, verbose=False)
    controls, nodes, configs, sleeps, scene_bases = {}, {}, [], [], []
    for name in ("add_dropdown", "add_slider", "add_checkbox", "add_text", "add_markdown", "add_image"):
        original = getattr(server.gui, name)

        def capture(*args, _name=name, _original=original, **kwargs):
            """记录真实控件。"""
            handle = _original(*args, **kwargs)
            label = _name if _name == "add_markdown" else kwargs.get("label", args[0] if args else _name)
            controls["curve" if _name == "add_image" else label] = handle
            return handle

        monkeypatch.setattr(server.gui, name, capture)
    for name in ("add_frame", "add_spline_catmull_rom", "add_icosphere"):
        original = getattr(server.scene, name)

        def capture_node(path, *args, _original=original, **kwargs):
            """记录真实场景节点。"""
            handle = _original(path, *args, **kwargs)
            nodes[path] = handle
            return handle

        monkeypatch.setattr(server.scene, name, capture_node)
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    monkeypatch.setattr(viser.extras, "ViserUrdf", lambda *a, **k: SimpleNamespace(update_cfg=lambda q: configs.append(q.copy())))
    monkeypatch.setattr(joints, "load_embodiment", lambda name: SimpleNamespace(load_urdf=lambda **kwargs: None))
    monkeypatch.setattr(joints, "_render_task_scene", lambda *a: scene_bases.append(a[2]["robot_from_table_pose7"].copy()))
    def render_table(proxy, runtime, urdf, data, **kwargs):
        """模拟新环境创建部分内容后发生渲染错误。"""
        proxy.scene.add_frame("/table", show_axes=False)
        if render_failure and len(data["robot_from_table_pose7"]) == 0:
            raise ValueError("table render failed")
        return []

    monkeypatch.setattr(joints, "_render_table", render_table)
    monkeypatch.setattr(gripper_curve, "gripper_curve_image", lambda openings, fps: np.full((4, 4, 3), round(openings[0, 0] * 100), dtype=np.uint8))
    initial = rollout(0, 4, [0.2, 0, 0, 0, 0, 0, 1])
    second = rollout(1, 1)
    reference = {side: initial[f"fk_{side}_pose7_robot_world"] for side in ("left", "right")}
    args = SimpleNamespace(robot="rm75b", episode=0, dataset_info={"total_episodes": 3}, port=0,
                           scene=None, clean_view=False, hide_trajectories=False, verify_reference=reference)
    schedulers = []

    class FakeScheduler:
        """以两次轮询模拟异步准备并检查关闭行为。"""

        def __init__(self, args, episode_count, *, current, worker):
            """记录任务状态。"""
            assert worker is joints._prepare_joint_worker and episode_count == 3
            self.queue, self.running, self.results = [], {}, {}
            self.completed, self.failures, self.closed = {current}, {}, False
            schedulers.append(self)

        def request(self, episode):
            """将请求放入队列。"""
            if episode not in self.queue and episode not in self.running.values():
                self.queue.insert(0, episode)

        def poll(self):
            """完成上一任务并启动下一任务。"""
            if self.running:
                episode = next(iter(self.running.values()))
                self.running.clear()
                if episode == 2:
                    self.failures[episode] = "bad episode"
                else:
                    self.results[episode] = second
                self.completed.add(episode)
            if self.queue:
                self.running[object()] = self.queue.pop(0)

        def take(self, episode):
            """取走结果，或返回错误。"""
            if episode in self.failures:
                raise RuntimeError(self.failures[episode])
            return self.results.pop(episode, None)

        def close(self):
            """记录资源已关闭。"""
            self.closed = True

    monkeypatch.setattr(replay_scheduler, "EpisodeScheduler", FakeScheduler)

    def choose():
        """暂停并请求下一段。"""
        controls["Play"].value = False
        controls["Episode"].value = "1"

    def pending():
        """准备期间继续显示当前段。"""
        assert "继续播放" in controls["add_markdown"].content
        np.testing.assert_allclose(nodes["/robot"].position, [0.2, 0, 0])

    def switched():
        """成功切换后同步所有可观察内容。"""
        assert controls["Frame"].max == 0 and controls["Frame"].value == 0
        assert not controls["Play"].value
        np.testing.assert_allclose(nodes["/robot"].position, [0, 0, 0])
        np.testing.assert_allclose(nodes["/tcp/right"].position, [1, 0, 0])
        assert len(nodes["/traj/tcp_right"].points) == 2
        assert not nodes["/traj/replay_right"].visible
        np.testing.assert_array_equal(controls["curve"].image, 20)
        np.testing.assert_array_equal(configs[-1], [1])
        assert sleeps[-1] == pytest.approx(1 / 31)
        controls["Episode"].value = "2"

    def pending_failure():
        """失败段加载期间旧段仍然可见。"""
        assert "继续播放" in controls["add_markdown"].content

    def failed():
        """失败后恢复选择器并保留旧轨迹。"""
        assert controls["Episode"].value == "1"
        assert "bad episode" in controls["add_markdown"].content
        np.testing.assert_allclose(nodes["/tcp/right"].position, [1, 0, 0])
        assert len(scene_bases) == 2
        raise StopViewer

    def render_failed():
        """场景准备失败后旧环境、姿态及控件均保持可用。"""
        assert controls["Episode"].value == "0"
        assert "table render failed" in controls["add_markdown"].content
        assert controls["Frame"].max == 3 and not controls["Play"].value
        np.testing.assert_allclose(nodes["/robot"].position, [0.2, 0, 0])
        np.testing.assert_array_equal(configs[-1], [0])
        assert nodes["/joint_environment_0"].visible
        assert not schedulers[0].closed
        raise StopViewer

    steps = iter((choose, pending, render_failed) if render_failure
                 else (choose, pending, switched, pending_failure, failed))

    def sleep(duration):
        """执行下一次界面操作。"""
        sleeps.append(duration)
        next(steps)()

    monkeypatch.setattr(joints, "time", SimpleNamespace(sleep=sleep))
    try:
        with pytest.raises(StopViewer):
            joints.show_viewer(args, initial)
        assert schedulers[0].closed
    finally:
        server.stop()


def test_joint_scheduler_prioritizes_and_bounds_results():
    """使用关节 worker 时仍有界后台加载，选择优先且淘汰后可重载。"""
    from concurrent.futures import Future

    class Executor:
        """通过手动完成 future 检查真实调度行为。"""

        def __init__(self):
            """保存任务及关闭状态。"""
            self.jobs = []
            self.closed = False

        def submit(self, worker, values, episode, refresh):
            """记录实际提交的 worker 和任务编号。"""
            assert worker is joints._prepare_joint_worker
            future = Future()
            self.jobs.append((episode, future))
            return future

        def shutdown(self, **kwargs):
            """记录线程池关闭。"""
            self.closed = True

    executor = Executor()
    scheduler = replay_scheduler.EpisodeScheduler(
        SimpleNamespace(), 6, current=0, workers=1, executor=executor,
        worker=joints._prepare_joint_worker,
    )
    try:
        scheduler.poll()
        assert executor.jobs[0][0] == 1
        scheduler.request(5)
        executor.jobs[0][1].set_result({"rollout": {"episode": 1}})
        scheduler.poll()
        assert executor.jobs[1][0] == 5
        while scheduler.running or scheduler.queue:
            for future, episode in list(scheduler.running.items()):
                future.set_result({"rollout": {"episode": episode}})
            scheduler.poll()
        assert scheduler.completed == set(range(6))
        assert len(scheduler.results) <= scheduler.result_capacity
        assert scheduler.take(1) is None
        scheduler.request(1)
        scheduler.poll()
        assert executor.jobs[-1][0] == 1
    finally:
        scheduler.close()
    assert executor.closed
