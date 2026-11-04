# Robot/table deployment calibrations

These transforms place the robot-agnostic HandUMI table frame in a target
robot's world. They are target deployment configuration, not part of the raw
recording and not Controller-to-TCP offsets.

```text
table/
├── sim/                 committed canonical simulation layouts
│   └── <robot>.yaml
└── local/               one physical laboratory's private measurements
    ├── example.yaml     committed template
    └── <robot>.yaml     ignored by Git
```

Use `--deployment-profile sim` for reproducible public simulation results and
`--deployment-profile local` when validating a measured physical installation.
The default `auto` profile selects a conventional local file when it exists and
otherwise falls back to the canonical simulation file. Replay always prints
the resolved profile and path.

在 QA、筛选、回放和转换中，如果 `auto` 或 `sim` 找不到标准仿真布局，
程序会提示并根据数据集的 TCP 轨迹中心和机器人 HOME 位姿估计水平位置与朝向，
保存到 `DATASET/.handumi/deployment/<robot>.yaml`，后续复用同一文件。
估计假定底座竖直且与桌面等高，不改变采集轨迹的高度或尺度；
文件标记为 `scope: simulation`、`verified: false`，可达性仍由后续 IK 检查判断。
该估计适用于在桌面坐标系采集的数据；保留旧估计直至移除该文件后重新计算。
已有显式或本地标定保持优先；显式文件缺失、文件内容无效和 `local` 缺失仍会报错。
`absolute-tcp` 的 `auto` 配置继续保留已有单位变换和逐段摆放搜索规则。

对 RM75B，当数据集已有共享摆放搜索结果
`DATASET/.handumi/base_placement/rm75b.yaml` 时，QA、筛选、回放和转换的
默认选项会优先使用这份仿真布局，并采用 `absolute-tcp` 和 `dataset-xy-yaw`。
自动选择保留文件内的仿真来源及未实测状态，逐段重新检查或复用对应缓存。
显式指定其他重定向模式、固定布局、标定文件或 `local` 部署时按用户设置执行；
`auto` 模式下已有本地实体标定也保持优先。显式 `sim` 选择共享仿真布局。
默认首帧及逐帧位置误差上限为 5 mm、逐帧姿态误差上限为 5°；
用户显式给出的阈值会继续生效。初次搜索与重估仍由回放的
`--base-placement dataset-xy-yaw` 和 `--refresh-base-placement` 控制。

Each local calibration must use `scope: physical`, declare the same stable
`lab` identifier configured under `deployment.lab` in `configs/rig.yaml`, and
remain `verified: false` until the hardware checks pass. A laboratory may use
`deployment.table_calibrations.<robot>` in `configs/rig.yaml` to override the
conventional local path.

Dataset-specific placements (for example a scene recorded with the board away
from the demonstrated workspace) are visualization artifacts rather than
canonical configuration. Keep them under `outputs/calibration/table/` and pass
them explicitly with `--deployment-calibration`.
