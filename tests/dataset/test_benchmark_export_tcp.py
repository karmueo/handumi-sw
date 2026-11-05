"""验证 TCP 导出基准的超时与中断清理。"""

import signal
import subprocess

import pytest

from handumi.scripts import benchmark_export_tcp as benchmark


@pytest.mark.parametrize("interrupt", [False, True])
def test_benchmark_reaps_process_group_on_timeout_or_interrupt(
    tmp_path, monkeypatch, interrupt
):
    """采样超时或中断时终止整个子进程组并回收主进程。"""
    calls = []

    class Process:
        """记录基准控制动作的进程替身。"""

        pid = 1234
        returncode = None

        def poll(self):
            """模拟持续运行。"""
            return self.returncode

        def wait(self, timeout=None):
            """模拟退出并记录回收。"""
            calls.append(("wait", timeout))
            self.returncode = -signal.SIGTERM
            return self.returncode

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        benchmark.os, "killpg", lambda pid, sig: calls.append((pid, sig))
    )
    if interrupt:
        monkeypatch.setattr(
            benchmark,
            "peak_tree_rss",
            lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            benchmark.run_case(["command"], tmp_path / "log", 10, timeout=10)
    else:
        result = benchmark.run_case(["command"], tmp_path / "log", 10, timeout=0)
        assert result["timed_out"]
        assert result["returncode"] == -signal.SIGTERM
    assert (1234, signal.SIGTERM) in calls
    assert any(item[0] == "wait" for item in calls)
