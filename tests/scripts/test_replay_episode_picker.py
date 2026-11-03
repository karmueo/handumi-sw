"""验证窗口切换 episode 时同步显示，以及失败时保留现有回放。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import viser
import viser.extras

from handumi.scripts.replay import replay_in_sim as replay
from handumi.scripts.replay import replay_scheduler
from handumi.visualize import gripper_curve


class StopViewer(Exception):
    """在检查完成后结束回放循环。"""


def _rollout(episode: int, frames: int, approach_frames: int = 0):
    """构造帧数、开度、轨迹和误差均可区分的回放结果。"""
    data = {
        "qpos": np.full((frames, 1), 10.0 * episode),
        "approach_qpos": np.full((approach_frames, 1), -1.0 - episode),
        "gripper_normalized": np.full((frames, 2), (episode + 1) / 10),
        "fps": np.array([30 + episode]),
    }
    for side in ("left", "right"):
        data[f"{side}_pos_error_m"] = np.full(frames, 0.001 * episode)
        data[f"{side}_rot_error_deg"] = np.full(frames, float(episode))
        for kind in ("target", "achieved"):
            poses = np.zeros((frames, 7))
            poses[:, 0] = episode + np.arange(frames) * 0.01
            poses[:, 6] = 1
            data[f"{kind}_{side}_pose7_robot_world"] = poses
    return data


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    """使用真实可视化控件，替换模型和数据以隔离窗口切换行为。"""
    server = viser.ViserServer(port=0, verbose=False)
    controls, nodes, configurations = {}, {}, []
    stopped = False
    stop_server = server.stop

    def stop():
        """只关闭一次临时服务，支持测试提前失败时清理。"""
        nonlocal stopped
        if not stopped:
            stopped = True
            stop_server()

    monkeypatch.setattr(server, "stop", stop)
    for name in (
        "add_dropdown",
        "add_slider",
        "add_checkbox",
        "add_text",
        "add_markdown",
        "add_image",
    ):
        original = getattr(server.gui, name)

        def capture(*args, _name=name, _original=original, **kwargs):
            """记录实际控件，以检查更新后状态。"""
            handle = _original(*args, **kwargs)
            label = (
                _name
                if _name == "add_markdown"
                else kwargs.get("label", args[0] if args else _name)
            )
            controls[label] = handle
            return handle

        monkeypatch.setattr(server.gui, name, capture)
    for name in ("add_spline_catmull_rom", "add_icosphere", "add_frame"):
        original = getattr(server.scene, name)

        def capture_node(path, *args, _original=original, **kwargs):
            """记录实际场景节点，避免只检查文本而遗漏旧轨迹。"""
            handle = _original(path, *args, **kwargs)
            nodes[path] = handle
            return handle

        monkeypatch.setattr(server.scene, name, capture_node)
    runtime = SimpleNamespace(
        active_sides=("right",),
        load_urdf=lambda **kwargs: None,
    )
    monkeypatch.setattr(replay, "load_embodiment", lambda name: runtime)
    monkeypatch.setattr(replay, "ensure_metadata", lambda ref: {"total_episodes": 3})
    monkeypatch.setattr(replay, "_render_task_scene", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay, "_render_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: server)
    monkeypatch.setattr(
        viser.extras,
        "ViserUrdf",
        lambda *args, **kwargs: SimpleNamespace(
            update_cfg=lambda q: configurations.append(np.asarray(q).copy())
        ),
    )
    monkeypatch.setattr(
        gripper_curve,
        "gripper_curve_image",
        lambda openings, fps: np.full(
            (4, 4, 3), round(float(openings[0, 0]) * 100), dtype=np.uint8
        ),
    )
    args = replay.build_parser().parse_args(
        [
            "unused",
            "--robot",
            "rm75b",
            "--retarget-mode",
            "absolute-tcp",
            "--strict-ik",
        ]
    )
    args.repo_id, args.dataset_root = "local/test", tmp_path
    args.prepare_mode = "off"

    class FakeScheduler:
        """两轮轮询完成一条任务，模拟播放期间的后台准备。"""

        def __init__(self, args, episode_count, *, current, **kwargs):
            """初始化队列和可检查的任务状态。"""
            self.args = args
            self.queue = []
            self.running = {}
            self.completed = {current}
            self.failures = {}
            self.results = {}

        def request(self, episode):
            """只为尚未准备的段排队。"""
            if episode not in self.queue and episode not in self.running.values() and episode not in self.results:
                self.queue.insert(0, episode)

        def poll(self):
            """先完成上一轮任务，再启动队首的一条。"""
            if self.running:
                episode = next(iter(self.running.values()))
                self.running.clear()
                next_args = SimpleNamespace(**vars(self.args))
                next_args.episode = episode
                try:
                    self.results[episode] = replay.solve_episode(next_args)
                except (Exception, SystemExit) as exc:
                    self.failures[episode] = str(exc)
                self.completed.add(episode)
            if self.queue:
                self.running[object()] = self.queue.pop(0)

        def take(self, episode):
            """取得准备结果或报告失败。"""
            if episode in self.failures:
                raise RuntimeError(self.failures[episode])
            return self.results.pop(episode, None)

        def close(self):
            """测试中无需回收真实 worker。"""

    monkeypatch.setattr(replay_scheduler, "EpisodeScheduler", FakeScheduler)
    yield SimpleNamespace(
        args=args,
        controls=controls,
        nodes=nodes,
        configurations=configurations,
    )
    stop()


@pytest.mark.parametrize(
    "approach_frames,hide_trajectories", [(0, False), (2, False), (0, True)]
)
def test_episode_switch_refreshes_view_and_recovers_failure(
    viewer, monkeypatch, approach_frames, hide_trajectories
):
    """缩短帧范围并更新所有显示，下一次加载失败后仍能继续当前回放。"""
    controls, nodes = viewer.controls, viewer.nodes
    viewer.args.hide_trajectories = hide_trajectories
    next_rollout = _rollout(1, 1, approach_frames)
    requested = []

    def solve(args):
        """先返回新轨迹，再模拟 strict IK 拒绝另一个 episode。"""
        requested.append(args.episode)
        assert args is not viewer.args
        assert args.robot == "rm75b" and args.strict_ik
        assert args.retarget_mode == "absolute-tcp"
        assert not controls["Play"].disabled
        assert not controls["Episode"].disabled and not controls["Frame"].disabled
        if args.episode == 2:
            raise SystemExit("IK fidelity threshold exceeded: frame 8")
        return next_rollout

    monkeypatch.setattr(replay, "solve_episode", solve)

    def select_next():
        """暂停后选择下一条轨迹。"""
        controls["Play"].value = False
        controls["Episode"].value = "1"

    def check_success():
        """检查所有动态数据已切换，并请求一条无法求解的轨迹。"""
        assert controls["Episode"].value == "1"
        assert controls["Frame"].value == 0
        assert controls["Frame"].max == approach_frames
        assert not controls["Play"].value
        assert not controls["Episode"].disabled
        np.testing.assert_array_equal(controls["Gripper opening"].image, 20)
        assert "20.0%" in controls["Gripper opening (Link7 flange replay)"].value
        assert "Episode **1**" in controls["add_markdown"].content
        np.testing.assert_array_equal(
            viewer.configurations[-1], [-2.0 if approach_frames else 10.0]
        )
        if not hide_trajectories:
            np.testing.assert_allclose(nodes["/target/right"].position, [1, 0, 0])
            assert len(nodes["/traj/target_right"].points) == 2
            np.testing.assert_allclose(nodes["/traj/target_right"].points[:, 0], 1)
        if approach_frames:
            assert "approach" in controls["EE error (cm/deg)"].value
        else:
            assert "0.1cm/1.0deg" in controls["EE error (cm/deg)"].value
        controls["Episode"].value = "2"

    def check_pending():
        """后台尚未完成时仍展示旧段，播放控件可使用。"""
        assert controls["Episode"].value == "1"
        assert not controls["Play"].disabled
        assert not controls["Episode"].disabled and not controls["Frame"].disabled
        assert "继续播放" in controls["add_markdown"].content

    def check_pending_failure():
        """第二段正在准备时继续保留已成功加载的内容。"""
        assert controls["Episode"].value == "2"
        assert not controls["Play"].disabled

    def check_failure():
        """加载失败后恢复选择器和控制，不改变已经显示的数据。"""
        assert controls["Episode"].value == "1"
        assert controls["Frame"].max == approach_frames
        assert not controls["Play"].disabled
        assert not controls["Frame"].disabled
        assert not controls["Episode"].disabled
        np.testing.assert_array_equal(controls["Gripper opening"].image, 20)
        assert "加载失败" in controls["add_markdown"].content
        assert "frame 8" in controls["add_markdown"].content
        raise StopViewer

    steps = iter((select_next, check_pending, check_success, check_pending_failure, check_failure))
    monkeypatch.setattr(replay, "time", SimpleNamespace(sleep=lambda _: next(steps)()))
    with pytest.raises(StopViewer):
        replay.show_viewer(viewer.args, _rollout(0, 4))
    assert requested == [1, 2]
    assert viewer.args.episode == 0


def test_rapid_episode_choices_only_load_latest(viewer, monkeypatch):
    """控件快速连续更新时只求解最终选择，成功后恢复原播放状态。"""
    requested = []
    monkeypatch.setattr(
        replay,
        "solve_episode",
        lambda args: requested.append(args.episode) or _rollout(args.episode, 3),
    )

    def choose_twice():
        """在主循环处理之前连续更改选择。"""
        viewer.controls["Episode"].value = "1"
        viewer.controls["Episode"].value = "2"

    def check_latest():
        """确认未加载已被替换的选择，也没有重置相机或播放意图。"""
        assert requested == [2]
        assert viewer.controls["Episode"].value == "2"
        assert viewer.controls["Play"].value
        assert viewer.controls["Frame"].max == 2
        raise StopViewer

    def check_pending():
        """最终选择尚未完成时保持原播放状态。"""
        assert viewer.controls["Episode"].value == "2"
        assert viewer.controls["Play"].value

    steps = iter((choose_twice, check_pending, check_latest))
    monkeypatch.setattr(replay, "time", SimpleNamespace(sleep=lambda _: next(steps)()))
    with pytest.raises(StopViewer):
        replay.show_viewer(viewer.args, _rollout(0, 4))


def test_episode_switch_keeps_table_world_and_moves_base(viewer, monkeypatch):
    """不同摆放下机器人根节点与轨迹一起变换，失败后保留当前摆放。"""
    first = _rollout(0, 2)
    second = _rollout(1, 2)
    for data, xy in ((first, [0.0, 0.0]), (second, [0.2, -0.1])):
        data["retarget_mode"] = np.array(["absolute-tcp"])
        data["table_from_robot_pose7"] = np.array([[*xy, 0, 0, 0, 0, 1]], dtype=np.float32)
        data["robot_from_table_pose7"] = np.array([[-xy[0], -xy[1], 0, 0, 0, 0, 1]], dtype=np.float32)
    draws = []

    def draw_table(server, runtime, urdf, rollout, *, clean=False, base_pose7=None):
        """检查桌面恒为世界原点并记录机器人 base。"""
        np.testing.assert_allclose(rollout["robot_from_table_pose7"][0], [0, 0, 0, 0, 0, 0, 1])
        draws.append(np.asarray(base_pose7).copy())
        return []

    monkeypatch.setattr(replay, "_render_table", draw_table)

    def solve(args):
        """先加载带新 base 的回放，再制造求解失败。"""
        if args.episode == 2:
            raise SystemExit("cannot load")
        return second

    monkeypatch.setattr(replay, "solve_episode", solve)

    def choose():
        """选择下一条 episode。"""
        viewer.controls["Episode"].value = "1"

    def check_move():
        """机器人和目标同步移动，而桌面保持不动。"""
        np.testing.assert_allclose(viewer.nodes["/robot"].position, [0.2, -0.1, 0], atol=1e-6)
        np.testing.assert_allclose(viewer.nodes["/target/right"].position,
                                   [1.2 + viewer.controls["Frame"].value * 0.01, -0.1, 0], atol=1e-6)
        np.testing.assert_allclose(draws[-1][:3], [0.2, -0.1, 0], atol=1e-6)
        viewer.controls["Episode"].value = "2"

    def pending():
        """等待后台一轮，不改变当前 base。"""

    def check_rollback():
        """加载失败时不移动已显示的机器人和桌面。"""
        np.testing.assert_allclose(viewer.nodes["/robot"].position, [0.2, -0.1, 0], atol=1e-6)
        assert len(draws) == 2
        raise StopViewer

    steps = iter((choose, pending, check_move, pending, check_rollback))
    monkeypatch.setattr(replay, "time", SimpleNamespace(sleep=lambda _: next(steps)()))
    with pytest.raises(StopViewer):
        replay.show_viewer(viewer.args, first)


def test_selecting_current_episode_cancels_pending_switch(viewer, monkeypatch):
    """在准备其他段期间重新选择当前段，旧任务完成后不得切换。"""
    monkeypatch.setattr(replay, "solve_episode", lambda args: _rollout(args.episode, 2))

    def choose_other():
        """开始准备第 1 段。"""
        viewer.controls["Episode"].value = "1"

    def return_current():
        """任务未完成时选择回第 0 段。"""
        assert "继续播放" in viewer.controls["add_markdown"].content
        viewer.controls["Episode"].value = "0"

    def check_cancelled():
        """即使第 1 段已完成，仍显示第 0 段。"""
        assert viewer.controls["Episode"].value == "0"
        assert "Episode **0**" in viewer.controls["add_markdown"].content
        np.testing.assert_array_equal(viewer.configurations[-1], [0.0])
        raise StopViewer

    steps = iter((choose_other, return_current, check_cancelled))
    monkeypatch.setattr(replay, "time", SimpleNamespace(sleep=lambda _: next(steps)()))
    with pytest.raises(StopViewer):
        replay.show_viewer(viewer.args, _rollout(0, 4))
