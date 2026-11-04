"""验证筛选进度在交互终端、重定向和关闭模式下的行为。"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from handumi.dataset import screening_progress


def test_terminal_updates_are_visible_before_exit(monkeypatch):
    """动态显示应在计算期间刷新，而非等关闭后才输出。"""
    monkeypatch.setenv("TERM", "xterm")
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None)
    with screening_progress.ScreenProgress(total=2, console=console) as display:
        display.phase("逐段筛选")
        display.on_progress({"episode": 1, "stage": "逐帧求解", "candidate": 3,
                             "current": 4, "total": 8})
        display.tick(force=True)
        assert "逐帧求解" in stream.getvalue()
        assert "候选 3" in stream.getvalue()
        assert "帧 4/8" in stream.getvalue()
    assert display.live is not None and not display.live.is_started


def test_non_terminal_throttles_same_stage_and_ignores_late_events(monkeypatch):
    """相同阶段限频，完成后迟到的 worker 状态不重新显示。"""
    clock = [100.0]
    monkeypatch.setattr(screening_progress.time, "monotonic", lambda: clock[0])
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)
    with screening_progress.ScreenProgress(total=2, console=console) as display:
        display.phase("仿真布局估计", 0, 200)
        first = stream.getvalue()
        clock[0] += 1
        display.phase("仿真布局估计", 1, 200)
        assert stream.getvalue() == first
        clock[0] += 5
        display.phase("仿真布局估计", 2, 200)
        assert "2/200" in stream.getvalue()
        display.on_progress({"episode": 0, "stage": "逐帧求解"})
        display.on_result(0, {"unreachable": None, "performance": {"cache_hit": True}})
        display.on_progress({"episode": 0, "stage": "迟到事件"})
        assert 0 not in display.active
        assert display.done == 1 and display.cache_hits == 1
    assert "\x1b" not in stream.getvalue()


def test_off_mode_preserves_warning_without_progress():
    """关闭进度只抑制状态，不能吞掉缓存和严格验收警告。"""
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None)
    with screening_progress.ScreenProgress(total=1, mode=False, console=console) as display:
        display.phase("初始化")
        display.on_result(0, {"unreachable": None, "warnings": ["[replay] warning: test"]})
    assert "[replay] warning: test" in stream.getvalue()
    assert "筛选 episode" not in stream.getvalue()


def test_dumb_terminal_uses_plain_status(monkeypatch):
    """不支持动态控制符的终端仍能看到实时纯文本进度。"""
    monkeypatch.setenv("TERM", "dumb")
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None)
    with screening_progress.ScreenProgress(total=1, console=console) as display:
        display.phase("准备数据")
        assert display.live is None
        assert "准备数据" in stream.getvalue()


@pytest.mark.parametrize("value,expected", [(True, "auto"), (False, "off"), ("on", "on")])
def test_legacy_progress_mode(value, expected):
    """旧布尔参数和新增字符串模式均可解析。"""
    assert screening_progress.progress_mode(value) == expected
