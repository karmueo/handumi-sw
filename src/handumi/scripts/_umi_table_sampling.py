"""桌面采集的连续停稳判定、视角进度、单帧质量检查和原始定位时间线保存。"""

import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

STATIONARY_WINDOW_S = 0.3  # 手持大致稳定所需的连续观察时长。
STATIONARY_POSITION_M = 0.005  # 相对窗口起始位置允许的轻微手抖。
STATIONARY_ROTATION_DEG = 1.5  # 相对窗口起始姿态允许的轻微转动。
TABLE_TARGET_VIEWS = 15  # 桌面标定固定收集的独立视角数。
CAPTURING = "CAPTURING"  # 预热、移动或观测不合格时的简化状态。
CURRENT_VIEW_COMPLETE = "CURRENT VIEW COMPLETE"  # 当前视角完成或停留在已完成视角。


def image_quality(observation, matrix, distortion):
    """返回原图重投影中位数和 P95；检测缺失返回 None。"""
    if observation is None:
        return None
    points, pixels, pose = observation
    predicted, _ = cv2.fisheye.projectPoints(
        points.reshape(-1, 1, 3),
        Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3],
        matrix, distortion,
    )
    errors = np.linalg.norm(predicted.reshape(-1, 2) - pixels, axis=1)
    return np.percentile(errors, [50, 95]).tolist()


def stationary(history, timestamp):
    """要求最近约 0.3 秒连续有效，相对窗口起点位移≤5 mm、旋转≤1.5°。"""
    window_ns = round(STATIONARY_WINDOW_S * 1e9)  # 允许额外 50 ms 的历史边界余量。
    samples = [s for s in history if timestamp - window_ns - 50_000_000 <= s.pc_monotonic_ns <= timestamp]
    if len(samples) < 2 or timestamp - samples[0].pc_monotonic_ns < window_ns:
        return False
    stamps = np.array([s.pc_monotonic_ns for s in samples])
    if timestamp - stamps[-1] > 50_000_000 or np.any(np.diff(stamps) > 50_000_000):
        return False
    if not all(s.right_tracked for s in samples):
        return False
    poses = np.array([s.right_device_controller_pose for s in samples])
    if not np.isfinite(poses).all():
        return False
    rotations = Rotation.from_quat(poses[:, 3:])
    return bool(
        np.max(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1)) <= STATIONARY_POSITION_M
        and np.max((rotations[0].inv() * rotations).magnitude()) <= np.deg2rad(STATIONARY_ROTATION_DEG)
    )


class TrackerJournal:
    """按原始时间戳去重保存全部定位样本，包含失效样本。"""

    def __init__(self, data, diagnostics):
        """初始化独立高频数组和可选逐批落盘文件。"""
        self.data = data
        self.path = diagnostics.directory / 'tracker.jsonl' if diagnostics else None
        self.last = -1
        data['raw_tracker'] = {'times_ns': [], 'poses': [], 'valid': []}

    def drain(self, tracker):
        """复制设备历史后释放锁，再写入新增样本；返回历史用于停稳判定。"""
        with tracker._lock:
            history = list(tracker._history)
        rows = []
        for sample in history:
            if sample.pc_monotonic_ns <= self.last:
                continue
            row = {'time_ns': int(sample.pc_monotonic_ns),
                   'pose': sample.right_device_controller_pose.tolist(),
                   'valid': bool(sample.right_tracked)}
            self.last = row['time_ns']
            self.data['raw_tracker']['times_ns'].append(row['time_ns'])
            self.data['raw_tracker']['poses'].append(row['pose'])
            self.data['raw_tracker']['valid'].append(row['valid'])
            rows.append(row)
        if self.path and rows:
            with self.path.open('a', encoding='utf-8') as stream:
                stream.writelines(json.dumps(row) + '\n' for row in rows)
        return history


class StationaryViewCapture:
    """记录停稳视角进度，抑制已完成视角的重复采样。"""

    def __init__(self, frames_per_view=15, target_views=None):
        """设置每视角有效帧数和可选目标视角数，并初始化已完成视角。"""
        self.frames_per_view = frames_per_view  # 每个视角的采集配额。
        self.target_views = target_views  # None 表示沿用原有计时采集。
        self.completed = []  # 已完成视角的参考位姿。
        self.anchor = None  # 当前视角首次有效观测的位姿。
        self.count = 0  # 当前视角已接收的合格帧数。
        self.status = CAPTURING  # 桌面标定顶部和图像叠字使用的简化状态。
        self.message = "WAITING FOR STEADY HOLD"

    @property
    def target_reached(self):
        """返回是否已收满目标独立视角；未设置目标时始终为 False。"""
        return self.target_views is not None and len(self.completed) >= self.target_views

    @property
    def progress_text(self):
        """返回英文视角进度；有目标时显示 N/15，否则只显示完成数量。"""
        if self.target_views is None:
            return f"Completed views: {len(self.completed)}"
        return f"Completed views: {len(self.completed)}/{self.target_views}"

    @staticmethod
    def different(pose, reference):
        """判断两个视角是否达到 5 cm 或 10° 的独立视角基线。"""
        return (
            np.linalg.norm(pose[:3] - reference[:3]) >= 0.05
            or (Rotation.from_quat(reference[3:]).inv()
                * Rotation.from_quat(pose[3:])).magnitude() >= np.deg2rad(10)
        )

    def update(self, pose, stable, reasons):
        """依据停稳和质量门决定是否收帧，并更新英文界面提示；返回收帧标志。"""
        if reasons:
            self.status = CAPTURING
            self.message = "WAITING FOR VALID OBSERVATION: " + ", ".join(reasons)
            return False
        if not stable:
            self.status = CAPTURING
            self.message = "MOVING / WAITING FOR STEADY HOLD - Hold roughly steady for about 0.3 s"
            return False
        pose = np.asarray(pose)
        if any(not self.different(pose, ref) for ref in self.completed):
            self.status = CURRENT_VIEW_COMPLETE
            self.message = "VIEW COMPLETE - Move at least 5 cm or rotate at least 10 deg to a new view"
            return False
        if self.anchor is None or self.different(pose, self.anchor):
            self.anchor = pose.copy()
            self.count = 0
        self.count += 1
        self.status = CAPTURING
        self.message = f"CAPTURING: {self.count}/{self.frames_per_view} frames - Keep roughly steady"
        if self.count >= self.frames_per_view:
            self.completed.append(self.anchor)
            self.anchor = None
            self.status = CURRENT_VIEW_COMPLETE
            self.message = "VIEW COMPLETE - Move to a new view; final quality checks run after capture"
        return True

    def markdown(self, remaining=None):
        """返回 Rerun 英文提示；有目标视角时只保留状态和完成进度。"""
        if self.target_views is not None:
            return f"# {self.status}\n\n{self.progress_text}"
        return (
            f"# {self.message}\n\n"
            f"Completed views: **{len(self.completed)} / at least 4** | "
            f"Remaining: **{max(0, remaining):.1f} s**\n\n"
            "Collect at least 4 views. More views are welcome. Quality checks run when the timer ends."
        )
