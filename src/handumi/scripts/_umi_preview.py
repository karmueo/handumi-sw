"""为 Vive 采集、诊断和预览提供共享的 Rerun 布局。"""

from handumi.visualization import BACKGROUND_COLOR


def capture_blueprint(rrb, *, table_frame):
    """创建相机、追踪、夹爪与健康状态的四面板布局。"""
    return rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(origin="/camera/right_wrist", name="Camera"),
            rrb.Spatial3DView(
                origin="/tracking",
                background=rrb.Background(color=[*BACKGROUND_COLOR, 255]),
                line_grid=False,
                name="Table: Tracker, Camera and TCP" if table_frame
                else "OpenVR raw: Tracker, Camera and TCP (no table)",
            ),
            rrb.TimeSeriesView(
                origin="/gripper",
                name="Gripper openness",
                axis_y=rrb.ScalarAxis(range=(0.0, 1.0)),
            ),
            rrb.TextDocumentView(origin="/capture_status", name="采集状态"),
            grid_columns=2,
        ),
        rrb.BlueprintPanel(expanded=True),
    )
