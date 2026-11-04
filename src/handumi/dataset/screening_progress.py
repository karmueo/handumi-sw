"""在主进程展示数据筛选总体进度、活动任务和求解阶段。"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.text import Text


def progress_mode(value: bool | str) -> str:
    """将原有布尔开关映射为进度模式，拒绝无效设置。"""
    if value is True:
        return "auto"
    if value is False:
        return "off"
    if value in {"auto", "on", "off"}:
        return value
    raise ValueError(f"Unsupported progress mode: {value!r}")


class ScreenProgress:
    """只在主进程绘制进度；工作进程事件可限频丢弃。"""

    def __init__(self, total: int | None = None, mode: bool | str = "auto", *, console=None):
        """创建进度显示器，total 可在完成数据集解析后再设置。"""
        self.mode = progress_mode(mode)
        self.console = console or Console(file=sys.stderr)
        self.total = total
        self.done = 0
        self.failures = 0
        self.cache_hits = 0
        self.active: dict[int, dict[str, Any]] = {}
        self.finished: set[int] = set()
        self.stage = "初始化"
        self.stage_detail = ""
        self.started = time.monotonic()
        self.last_print = 0.0
        self.last_render = 0.0
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=self.console,
        )
        self.task = self.progress.add_task("筛选 episode", total=total)
        self.live = (
            Live(console=self.console, auto_refresh=False)
            if (self.mode != "off" and self.console.is_terminal
                and not self.console.is_dumb_terminal) else None
        )

    def __enter__(self):
        """启动交互显示，并立即报告初始化。"""
        if self.live is not None:
            self.live.start()
        self.tick(force=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        """成功、错误或中断均关闭显示，不更改实际完成数。"""
        if self.live is not None:
            self.live.stop()
        elif self.mode != "off":
            self.tick(force=True)

    def set_total(self, total: int) -> None:
        """设置总段数并刷新；不得使用预估段数冒充实际完成数。"""
        self.total = total
        self.progress.update(self.task, total=total)
        self.tick(force=True)

    def phase(self, stage: str, current: int | None = None, total: int | None = None) -> None:
        """设置全局阶段；可附加布局估计等独立阶段的数量。"""
        changed = self.stage != stage
        self.stage = stage
        self.stage_detail = f" {current}/{total}" if current is not None and total is not None else ""
        self.tick(force=changed)

    def on_progress(self, event: dict[str, Any]) -> None:
        """更新一段的最新事件；忽略完成后迟到的队列事件。"""
        episode = int(event["episode"])
        if episode not in self.finished:
            self.active[episode] = event
        self.tick()

    def on_result(self, episode: int, item: dict[str, Any]) -> None:
        """按可靠的任务结果更新计数，转发工作进程保留的警告。"""
        self.done += 1
        self.failures += int(item.get("unreachable") is not None)
        self.cache_hits += int(bool(item.get("performance", {}).get("cache_hit")))
        self.finished.add(episode)
        self.active.pop(episode, None)
        for warning in item.get("warnings", ()):
            self.console.print(str(warning), markup=False, highlight=False)
        self.tick()

    def tick(self, *, force: bool = False) -> None:
        """限频刷新终端，非终端每五秒最多输出一条状态。"""
        if self.mode == "off":
            return
        now = time.monotonic()
        elapsed = now - self.started
        total = "?" if self.total is None else str(self.total)
        eta = "--"
        if self.total is not None and self.done >= 2 and self.done < self.total:
            eta = f"{elapsed / self.done * (self.total - self.done):.0f}s"
        summary = (
            f"{self.stage}{self.stage_detail} · {self.done}/{total} 段 · "
            f"求解失败 {self.failures} · 缓存命中 {self.cache_hits} · "
            f"用时 {elapsed:.1f}s · 估算剩余 {eta}"
        )
        lines = []
        for episode, event in sorted(self.active.items()):
            stage = "缓存命中" if event.get("stage") == "cache_hit" else event.get("stage", "处理中")
            candidate = f" 候选 {event['candidate']}" if event.get("candidate") is not None else ""
            frame = (f" 帧 {event['current']}/{event['total']}"
                     if event.get("current") is not None and event.get("total") is not None else "")
            lines.append(f"episode {episode}: {stage}{candidate}{frame}")
        if self.live is not None:
            if not force and now - self.last_render < 0.2:
                return
            self.progress.update(self.task, completed=self.done)
            self.live.update(
                Group(self.progress, Text(summary), *(Text(line) for line in lines)),
                refresh=True,
            )
            self.last_render = now
        elif force or now - self.last_print >= 5:
            suffix = "; " + "; ".join(lines) if lines else ""
            self.console.print(f"[screen] {summary}{suffix}", markup=False, highlight=False)
            self.last_print = now
