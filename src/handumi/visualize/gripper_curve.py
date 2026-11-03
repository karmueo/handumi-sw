"""为仿真回放生成独立于机器人夹爪几何的归一化开度曲线。"""

import numpy as np


def gripper_curve_image(openings, fps):
    """生成左右夹爪开度曲线图，供新建或更新回放图表使用。"""
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
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def add_gripper_curve(server, openings, fps):
    """在侧栏创建开度曲线，返回可原位更新图像的控件。"""
    return server.gui.add_image(
        gripper_curve_image(openings, fps), label="Gripper opening"
    )
