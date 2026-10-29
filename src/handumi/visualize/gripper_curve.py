"""为仿真回放生成独立于机器人夹爪几何的归一化开度曲线。"""

import numpy as np


def add_gripper_curve(server, openings, fps):
    """绘制左右开度随时间的曲线；当前播放帧由回放器数值组件指示。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6, 2), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    values = np.asarray(openings)
    times = np.arange(len(values)) / fps
    axis.plot(times, values[:, 0], label="Left")
    axis.plot(times, values[:, 1], label="Right")
    axis.set(xlabel="Time (s)", ylabel="Opening", ylim=(-0.05, 1.05))
    axis.legend(loc="upper right")
    figure.tight_layout()
    canvas.draw()
    return server.gui.add_image(
        np.asarray(canvas.buffer_rgba())[:, :, :3].copy(), label="Gripper opening"
    )
