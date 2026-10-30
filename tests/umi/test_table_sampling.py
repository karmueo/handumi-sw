"""验证桌面停稳窗口、定位原始时间线与单帧质量检查。"""
import threading
from types import SimpleNamespace

import numpy as np

from handumi.scripts._umi_table_sampling import (
    CAPTURING,
    CURRENT_VIEW_COMPLETE,
    TABLE_TARGET_VIEWS,
    StationaryViewCapture,
    TrackerJournal,
    image_quality,
    stationary,
)


def samples():
    """创建覆盖半秒的 100 Hz 静止定位样本。"""
    return [SimpleNamespace(pc_monotonic_ns=i*10_000_000,
            right_tracked=True,
            right_device_controller_pose=np.array([0.,0,0,0,0,0,1])) for i in range(56)]


def test_stability_rejects_motion_invalid_and_gaps():
    """停稳需要完整连续窗口；运动、丢定位和采样缺口都应拒绝。"""
    history = samples()
    assert stationary(history, 550_000_000)
    assert not stationary(history[-10:], 550_000_000)
    history[30].right_tracked = False
    assert not stationary(history, 550_000_000)
    history[30].right_tracked = True
    history[30].right_device_controller_pose[0] = .01
    assert not stationary(history, 550_000_000)
    history = samples()
    assert not stationary(history[:20]+history[30:], 550_000_000)
    history[-1].right_device_controller_pose[3:] = [0,0,np.sin(.02),np.cos(.02)]
    assert not stationary(history, 550_000_000)


def test_journal_preserves_invalid_samples_and_deduplicates(tmp_path):
    """逐批收集不重复写入，失效样本和原始时间戳仍保留。"""
    history = samples()
    history[10].right_tracked = False
    tracker = SimpleNamespace(_history=history, _lock=threading.Lock())
    data = {}
    journal = TrackerJournal(data, SimpleNamespace(directory=tmp_path))
    journal.drain(tracker)
    journal.drain(tracker)
    assert len(data['raw_tracker']['times_ns']) == 56
    assert data['raw_tracker']['valid'][10] is False
    assert len((tmp_path/'tracker.jsonl').read_text().splitlines()) == 56


def test_quality_detects_bad_pixels(table_setup):
    """已知投影通过质量检查，明显错配的像素会超限。"""
    _, data, _, _ = table_setup
    observation = tuple(np.array(part) for part in data['observations'][0])
    matrix = np.array([[50.,0,32],[0,50,24],[0,0,1]])
    assert max(image_quality(observation,matrix,np.zeros(4))) < 1e-6
    observation[1][:] += 10
    assert min(image_quality(observation,matrix,np.zeros(4))) > 2


def test_stationary_view_progress_and_duplicate_rejection():
    """仅合格停稳帧推进进度，完成后必须换独立视角才能继续。"""
    from handumi.scripts._umi_table_sampling import StationaryViewCapture

    capture = StationaryViewCapture(frames_per_view=2)
    pose = np.array([0., 0, 0, 0, 0, 0, 1])
    assert not capture.update(pose, False, [])
    assert not capture.update(pose, True, ["warming_up"])
    assert capture.count == 0
    assert capture.update(pose, True, [])
    assert "1/2" in capture.message
    assert not capture.update(pose, True, ["single_frame_quality_failed"])
    assert capture.count == 1
    assert capture.update(pose, True, [])
    assert "VIEW COMPLETE" in capture.message
    assert len(capture.completed) == 1
    assert not capture.update(pose, True, [])
    pose[0] = 0.06
    assert capture.update(pose, True, [])
    assert capture.update(pose, True, [])
    assert len(capture.completed) == 2
    pose[0] = 0
    assert not capture.update(pose, True, [])
    assert "2 / at least 4" in capture.markdown(10)


def test_stationary_view_rotation_and_incomplete_view_switch():
    """未完成时换视角重新计数，旋转基线也可建立独立视角。"""
    from scipy.spatial.transform import Rotation

    from handumi.scripts._umi_table_sampling import StationaryViewCapture

    capture = StationaryViewCapture(frames_per_view=2)
    pose = np.array([0., 0, 0, 0, 0, 0, 1])
    assert capture.update(pose, True, [])
    pose[0] = 0.1
    assert capture.update(pose, True, [])
    assert capture.count == 1
    assert capture.update(pose, True, [])
    pose[3:] = Rotation.from_euler('z', 15, degrees=True).as_quat()
    assert capture.update(pose, True, [])
    assert capture.count == 1


def test_handheld_stability_accepts_small_tremor():
    """短窗口内允许轻微手抖，明显平移、转动和过期定位仍拒绝。"""
    from scipy.spatial.transform import Rotation

    history = samples()[-31:]
    for index, sample in enumerate(history):
        sample.right_device_controller_pose[0] = 0.004 * np.sin(index * 0.3)
        sample.right_device_controller_pose[3:] = Rotation.from_euler(
            'z', np.sin(index * 0.3), degrees=True
        ).as_quat()
    assert stationary(history, 550_000_000)
    assert not stationary(history[1:], 550_000_000)
    assert not stationary(history, 610_000_000)
    history[-1].right_device_controller_pose[0] = 0.006
    assert not stationary(history, 550_000_000)
    history[-1].right_device_controller_pose[0] = 0
    history[-1].right_device_controller_pose[3:] = Rotation.from_euler(
        'z', 2, degrees=True
    ).as_quat()
    assert not stationary(history, 550_000_000)


def test_target_view_capture_completes_exactly_fifteen_views():
    """第 14 个视角后仍继续，第 15 个完成才标记目标达成。"""
    capture = StationaryViewCapture(
        frames_per_view=15, target_views=TABLE_TARGET_VIEWS,
    )
    pose = np.array([0.0, 0, 0, 0, 0, 0, 1])
    for view in range(TABLE_TARGET_VIEWS - 1):
        pose[0] = view * 0.06
        for _ in range(capture.frames_per_view):
            assert capture.update(pose, True, [])
        assert len(capture.completed) == view + 1
        assert not capture.target_reached
        assert capture.status == CURRENT_VIEW_COMPLETE
        assert capture.progress_text == f"Completed views: {view + 1}/15"

    duplicate = np.array([0.0, 0, 0, 0, 0, 0, 1])
    assert not capture.update(duplicate, True, [])
    assert len(capture.completed) == 14
    assert capture.status == CURRENT_VIEW_COMPLETE
    assert capture.progress_text == "Completed views: 14/15"

    final = np.array([14 * 0.06, 0, 0, 0, 0, 0, 1])
    assert not capture.update(final, True, ["single_frame_quality_failed"])
    assert len(capture.completed) == 14
    assert capture.status == CAPTURING
    for _ in range(capture.frames_per_view):
        assert capture.update(final, True, [])
    assert len(capture.completed) == TABLE_TARGET_VIEWS
    assert capture.target_reached
    assert capture.status == CURRENT_VIEW_COMPLETE
    assert capture.progress_text == "Completed views: 15/15"
    markdown = capture.markdown()
    assert "# CURRENT VIEW COMPLETE" in markdown
    assert "Completed views: 15/15" in markdown
    assert "Remaining" not in markdown and "frames" not in markdown


def test_incomplete_view_switch_does_not_increment_completed_views():
    """未收满 15 帧就换视角时重新计数，旧视角不计入完成数量。"""
    capture = StationaryViewCapture(
        frames_per_view=15, target_views=TABLE_TARGET_VIEWS,
    )
    pose = np.array([0.0, 0, 0, 0, 0, 0, 1])
    for _ in range(14):
        assert capture.update(pose, True, [])
    assert capture.count == 14 and not capture.completed

    pose[0] = 0.1
    assert capture.update(pose, True, [])
    assert capture.count == 1
    assert not capture.completed
    for _ in range(13):
        assert capture.update(pose, True, [])
    assert capture.count == 14 and not capture.completed
    assert capture.update(pose, True, [])
    assert len(capture.completed) == 1
    assert not capture.target_reached
