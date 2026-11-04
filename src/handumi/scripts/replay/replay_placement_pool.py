"""以有限 spawn 进程并行执行数据集底座搜索中的独立任务。"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing
import queue
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
from scipy.spatial.transform import Rotation

from handumi.scripts.replay.replay_scheduler import _spawn_environment, _worker_init

# 每个 worker 独有运行时；不将 JAX、URDF 或可变碰撞检查器跨进程序列化。
_ARGS = None
_CONTEXTS = None
_REFERENCE = None
_RUNTIME = None
_CHECKER = None
_SAMPLE_EVALUATORS = {}
_EVENTS = None
_LAST_PROGRESS = 0.0
_LAST_STAGE = None


def _initialize(values, contexts, reference, events):
    """安装只包含原始数据的搜索快照，并限制 worker 线程和 CPU 亲和性。"""
    global _ARGS, _CONTEXTS, _REFERENCE, _RUNTIME, _CHECKER, _SAMPLE_EVALUATORS, _EVENTS
    _worker_init(events)
    _ARGS = argparse.Namespace(**values)
    _CONTEXTS = contexts
    _REFERENCE = reference
    _RUNTIME = None
    _CHECKER = None
    _SAMPLE_EVALUATORS = {}
    _EVENTS = events


def _emit(episode, stage, current=None, total=None):
    """限频、非阻塞地报告 worker 进度；队列繁忙时直接丢弃。"""
    global _LAST_PROGRESS, _LAST_STAGE
    if _EVENTS is None:
        return
    now = time.monotonic()
    if (stage == _LAST_STAGE and now - _LAST_PROGRESS < 0.2
            and not (current is not None and current == total)):
        return
    _LAST_PROGRESS, _LAST_STAGE = now, stage
    try:
        _EVENTS.put_nowait((episode, stage, current, total))
    except (queue.Full, BrokenPipeError, OSError, ValueError):
        pass


def _context(index):
    """在进程内惰性建立模型与检查器，返回指定段的独立只读输入。"""
    global _RUNTIME, _CHECKER
    if _RUNTIME is None:
        from handumi.robots.registry import load_embodiment
        from handumi.robots.self_collision import GripperBoxCollisionChecker

        _RUNTIME = load_embodiment(_ARGS.robot, collision_kind="gripper-box")
        if _RUNTIME.config.ik_weights.collision_enabled:
            solver = _RUNTIME.solver_cls()
            _CHECKER = GripperBoxCollisionChecker(
                _RUNTIME.robot,
                solver.robot_collision,
                _RUNTIME.load_urdf(),
                _RUNTIME.config.ik_weights.self_collision_margin,
            )
    return {**_CONTEXTS[index], "runtime": _RUNTIME, "collision_checker": _CHECKER}


def _task(item):
    """执行独立首帧、代表帧或整段评分任务，返回确定性结果。"""
    from handumi.retargeting.base_placement import first_frame_candidates
    from handumi.scripts.replay.replay_dataset_placement import (
        _candidate_calibration,
        _full_episode,
        _sample_evaluator,
    )

    global _LAST_PROGRESS, _LAST_STAGE
    kind, episode, candidate = item
    _LAST_PROGRESS, _LAST_STAGE = 0.0, None
    if kind not in {"first", "sample", "full"}:
        raise ValueError(f"Unknown placement task: {kind}")
    with contextlib.redirect_stdout(io.StringIO()):
        if _RUNTIME is None:
            _emit(episode, "模型初始化")
        context = _context(episode)
        if kind == "first":
            return [value.xy for value in first_frame_candidates(
                context["runtime"], context["tool"], context["targets"], candidate,
                context["locked"], max_seeds=1,
            )]
        if kind == "sample":
            if episode not in _SAMPLE_EVALUATORS:
                def sample_progress(_index, current, total):
                    """将代表帧进度发送到父进程的有限队列。"""
                    _emit(episode, "代表帧评分", current, total)

                _SAMPLE_EVALUATORS[episode] = _sample_evaluator(
                    _ARGS, [context], _REFERENCE, frame_progress=sample_progress,
                )
            return _SAMPLE_EVALUATORS[episode](candidate)
        if kind == "full":
            selection = _candidate_calibration(_REFERENCE, candidate.pose)
            def solve_progress(stage, **details):
                """将整段求解阶段和帧数发送到父进程。"""
                _emit(episode, stage, details.get("current"), details.get("total"))

            return _full_episode(_ARGS, context, episode, selection,
                                 progress_callback=solve_progress)


class PlacementPool:
    """复用受限进程池并按输入顺序归并独立候选与 episode 的评分。"""

    _FIELDS = (
        "loaded", "targets", "states_for_retarget", "gripper_openings",
        "targets_for_search", "samples", "locked", "tool", "input_identity",
    )
    # 复用方案的私有临时目录必须显式列入白名单，避免传递回调和 JAX 对象。
    _PRIVATE_ARGS = ("_placement_scratch_dir",)

    def __init__(self, args, contexts, reference, workers, progress=None):
        """保存输入快照和任务限制；模型仅在 worker 首次运行时创建。"""
        if workers < 1:
            raise ValueError("placement workers 必须至少为 1")
        self.values = {
            key: value for key, value in vars(args).items()
            if not key.startswith("_") or key in self._PRIVATE_ARGS
        }
        self.contexts = [{key: context[key] for key in self._FIELDS} for context in contexts]
        self.reference = reference
        self.workers = workers
        self.progress = progress
        self.executor = None
        self.events = None

    def __enter__(self):
        """创建 spawn 池，实际进程在首次提交任务时启动。"""
        context = multiprocessing.get_context("spawn")
        self.events = (context.Queue(maxsize=max(16, self.workers * 8))
                       if self.progress and getattr(self.progress, "mode", None) != "off"
                       else None)
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialize,
            initargs=(self.values, self.contexts, self.reference, self.events),
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        """关闭进程池；异常时取消尚未执行的任务。"""
        try:
            if exc_type is not None:
                # ProcessPoolExecutor 取消 future 不会停止运行中的耗时 IK。
                processes = list((self.executor._processes or {}).values())
                for process in processes:
                    try:
                        if process.is_alive():
                            process.terminate()
                    except (OSError, ValueError):
                        pass
                for process in processes:
                    process.join(timeout=2)
                    if process.is_alive():
                        try:
                            process.kill()
                        except (OSError, ValueError):
                            pass
                        process.join(timeout=2)
            self.executor.shutdown(wait=True, cancel_futures=exc_type is not None)
        finally:
            self.executor = None
            if self.events is not None:
                self.events.close()
                self.events.join_thread()
                self.events = None

    def _drain_progress(self, stage, completed, total):
        """转发目前可读的 worker 帧进度，保持主阶段完成数不变。"""
        if self.events is None or self.progress is None:
            return
        while True:
            try:
                episode, worker_stage, current, frame_total = self.events.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            self.progress.update(stage, completed, total,
                                 detail=f"episode {episode + 1}/{len(self.contexts)} · {worker_stage}",
                                 frame_current=current, frame_total=frame_total)

    def _map(self, tasks, total, stage):
        """限制在途任务并按原序返回结果，完成事件由父进程汇总。"""
        if self.executor is None:
            raise RuntimeError("PlacementPool 必须在 with 范围内使用")
        pending = {}
        results = [None] * total
        source = iter(enumerate(tasks))
        completed = 0
        with _spawn_environment():
            while True:
                while len(pending) < self.workers * 2:
                    try:
                        index, item = next(source)
                    except StopIteration:
                        break
                    pending[self.executor.submit(_task, item)] = index
                if not pending:
                    break
                ready, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                self._drain_progress(stage, completed, total)
                for future in ready:
                    index = pending.pop(future)
                    results[index] = future.result()
                    completed += 1
                    if self.progress is not None:
                        self.progress.update(stage, completed, total,
                                             detail=f"已处理 {completed}/{total}")
            self._drain_progress(stage, completed, total)
        return results

    def first_frames(self, reference_pose):
        """对八个朝向及每段首帧求 XY 候选，返回二维顺序的列表。"""
        reference_pose = np.asarray(reference_pose, dtype=float)
        reference_rotation = Rotation.from_quat(reference_pose[3:])
        poses = []
        for yaw in range(8):
            pose = reference_pose.copy()
            pose[3:] = (Rotation.from_euler("z", yaw * np.pi / 4)
                        * reference_rotation).as_quat()
            poses.append(pose)
        count = len(self.contexts)
        tasks = (("first", episode, pose) for pose in poses for episode in range(count))
        results = self._map(tasks, 8 * count, "生成候选")
        return [results[yaw * count:(yaw + 1) * count] for yaw in range(8)]

    def sample_scores(self, candidates):
        """对每个候选和每段的代表帧评分，返回每候选的汇总分。"""
        count = len(self.contexts)
        tasks = (("sample", episode, candidate)
                 for candidate in candidates for episode in range(count))
        results = self._map(tasks, len(candidates) * count, "代表帧预筛")
        return [tuple(map(sum, zip(*results[i * count:(i + 1) * count], strict=True)))
                for i in range(len(candidates))]

    def full_scores(self, candidates):
        """逐段验证全部候选，并按候选与段索引顺序返回评分。"""
        count = len(self.contexts)
        tasks = (("full", episode, candidate)
                 for candidate in candidates for episode in range(count))
        results = self._map(tasks, len(candidates) * count, "完整轨迹验证")
        return [results[i * count:(i + 1) * count] for i in range(len(candidates))]
