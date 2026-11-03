"""收集回放阶段耗时，并在主进程显示有界的批量准备进度。"""

from __future__ import annotations

import queue
import sys
import time
from collections import defaultdict

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.text import Text


def report(args, stage: str, **details) -> None:
    """向可选回调报告阶段及进度；普通单段求解无需创建显示器。"""
    callback = getattr(args, "_replay_progress", None)
    if callback is not None:
        callback(stage, **details)


class WorkerProgress:
    """按阶段累计耗时，限频且非阻塞地发送少量进度事件。"""

    def __init__(self, episode: int, events=None, interval: float = 0.2):
        """记录开始时刻与事件发送间隔。"""
        self.episode = episode
        self.events = events
        self.interval = interval
        self.started = self.changed = time.monotonic()
        self.last_sent = 0.0
        self.stage = None
        self.stage_seconds = defaultdict(float)
        self.cache_hit = False

    def __call__(self, stage: str, **details) -> None:
        """更新阶段计时，队列满时丢弃中间事件而不阻塞计算。"""
        now = time.monotonic()
        changed = stage != self.stage
        if changed:
            if self.stage is not None:
                self.stage_seconds[self.stage] += now - self.changed
            self.stage = stage
            self.changed = now
        if stage == "cache_hit":
            self.cache_hit = True
        if self.events is None or (not changed and now - self.last_sent < self.interval):
            return
        self.last_sent = now
        try:
            self.events.put_nowait({"episode": self.episode, "stage": stage, **details})
        except (queue.Full, OSError, ValueError):
            pass

    def finish(self) -> dict:
        """返回累计阶段秒数与墙钟秒数，供最终结果可靠传回。"""
        now = time.monotonic()
        if self.stage is not None:
            self.stage_seconds[self.stage] += now - self.changed
            self.stage = None
        return {"cache_hit": self.cache_hit, "elapsed": now - self.started,
                "stage_seconds": dict(self.stage_seconds)}


class BatchProgress:
    """在终端统一显示进度；重定向输出时定期打印纯文本状态。"""

    def __init__(self, total: int, mode: str = "auto", *, console=None):
        """初始化总体计数和最多一个状态项对应每个活动任务。"""
        self.total = total
        self.mode = mode
        self.console = console or Console(file=sys.stderr)
        self.terminal = self.console.is_terminal and mode != "off"
        self.started = time.monotonic()
        self.last_print = 0.0
        self.active = {}
        self.progress = Progress(TextColumn("[progress.description]{task.description}"),
                                 BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                                 console=self.console)
        self.task = self.progress.add_task("准备 episode", total=total)
        self.live = Live(console=self.console, refresh_per_second=5) if self.terminal else None

    def __enter__(self):
        """开始交互终端刷新。"""
        if self.live:
            self.live.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        """任何退出路径都关闭终端动态显示。"""
        if self.live:
            self.live.stop()

    def update(self, scheduler, *, force: bool = False) -> None:
        """读取调度器的最新事件和完成数，并限频刷新文本或终端。"""
        if self.mode == "off":
            return
        self.active.update(scheduler.drain_events())
        running = set(scheduler.running.values())
        self.active = {ep: event for ep, event in self.active.items() if ep in running}
        done = len(scheduler.completed)
        failures = len(scheduler.failures)
        summary = (f"{done}/{self.total} 段 成功 {done - failures} 失败 {failures} "
                   f"缓存命中 {scheduler.cache_hits} 用时 {time.monotonic() - self.started:.1f}s")
        lines = []
        for ep in sorted(running):
            event = self.active.get(ep, {})
            phase = event.get("stage", "等待工作进程")
            if phase == "cache_hit":
                phase = "缓存命中"
            frame = (f" 已完成 {event['current']}/{event['total']} 帧"
                     if "current" in event and "total" in event else "")
            candidate = f" 候选 {event['candidate']}" if event.get("candidate") else ""
            lines.append(f"episode {ep}: {phase}{candidate}{frame}")
        now = time.monotonic()
        if self.live:
            self.progress.update(self.task, completed=done)
            self.live.update(Group(self.progress, Text(summary), *(Text(line) for line in lines)))
        elif force or now - self.last_print >= 5:
            self.console.print("[replay] " + summary + ("; " + "; ".join(lines) if lines else ""),
                               markup=False, highlight=False)
            self.last_print = now
