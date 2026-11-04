# 在仿真中回放本地录制数据

`handumi replay` 将一段原始 HandUMI episode 重定向到配置的双臂机器人，并在 Viser 中显示目标 TCP 轨迹和实际达到的 TCP 轨迹。源录制数据不绑定机器人；启动回放时选择机器人模型、逆运动学（IK）配置和桌面摆放位置。

## 安装回放依赖

首次回放前，安装 Python 仿真扩展依赖：

```bash
uv sync --extra sim
```

在 PICO 工作站上，使用 `bash install.sh --sim` 替代最后的 `uv sync` 命令。该脚本会在同步依赖后重新安装本地构建的 XRoboToolkit 包，避免它被覆盖。如果工作站只回放已有数据集，则不需要 XRoboToolkit。

轨迹回放直接读取 Parquet 状态列，不解码数据集中的 MP4 相机数据，因此 IK 回放不需要 FFmpeg。检查或整理相机视频时，请安装并验证 FFmpeg：

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
ffprobe -version
uv run python -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec OK')"
```

如果出现 `Could not load libtorchcodec`，请参阅 [TorchCodec 加载数据集视频失败](../troubleshooting.md#dataset-video-loading-fails-in-torchcodec)。

## 使用本地数据集

将录制目录作为 `DATASET` 传入。本地回放不会下载数据：

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0
```

修改 `--episode` 可选择其他 episode。如果只需要 IK 结果和保存的 NPZ 文件，可添加 `--headless`。否则，请打开 Viser 输出的地址，通常是 <http://localhost:8080>。

### 逐段缓存与后台准备

回放先准备当前 episode，再打开窗口；默认使用最多八个独立进程，在后台准备其余 episode。播放不会因后台求解暂停。选择尚未准备的 episode 时，选择项会排在尚未启动的后台任务之前；已有任务继续运行，结果准备完成后才切换画面。失败时界面保留当前 episode，并显示失败原因和后台进度。

```bash
# 默认：显示当前段，并在后台准备其他段
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --strict-ik

# 只预计算全部段，不打开窗口；默认最多使用 8 个进程
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp \
  --strict-ik --headless --prepare-mode all

# 只在选择时准备；或临时关闭缓存；或强制重新计算
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --prepare-mode off
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --no-cache
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --refresh-cache
```

普通 `--headless` 只处理所选 episode；显式 `--prepare-mode all` 才准备全部段。全部预计算会逐段报告失败，初始段失败时仍继续其他段；headless 全量运行存在失败时返回非零状态。关闭窗口时不再提交任务，停止仍在执行的后台进程；缓存采用临时文件和原子替换，未完成的结果不会成为可用缓存。

无窗口全量准备会从选中段开始并行处理，最多同时运行 `--prepare-workers` 个任务，默认值为 8；并行数应结合 CPU 和内存调整。进度显示已完成、成功、失败、缓存命中数及每个活动任务的当前阶段和已完成帧数。基座搜索的候选数是已尝试的数量，搜索长度不固定。模型初始化及首帧求解可能包含 JAX 编译；汇总中的阶段时间是各工作进程耗时之和，墙钟时间单独列出。非交互终端定期输出纯文本，`--progress off` 可关闭进度显示。

首次准备或使用 `--refresh-cache` 时会计算轨迹；后续在输入和求解配置不变时可读取缓存，并重新执行严格误差验收。求解相关源码变化也会使已有缓存失效。比较计算速度时应分别记录冷缓存和热缓存，避免将缓存读取时间当作求解加速。

默认缓存目录为 `DATASET/.handumi/solve_cache/`，可用 `--cache-dir PATH` 指定。缓存文件不需要 pickle，也不进入 Git 或发布包。每段的内容键覆盖该段状态和开度、相关采集元数据、实际控制器/TCP/桌面标定、标定路径与内容、机器人配置和 URDF、求解参数、算法及数值库版本。追加其他 episode 不会使旧段失效；该段输入、实际标定、模型或参数改变时会重新求解。同内容的标定文件换了路径也会重新求解，以保证结果中记录的来源正确。

单段处理时，控制台的 `cache hit` / `cache miss` 显示 episode 编号和内容键前 12 位；无窗口全量准备显示缓存命中总数。完整内容键是缓存 NPZ 的文件名。缓存损坏会自动视为未命中并重新写入。

回放、筛选和转换在实际输入与参数一致时可复用完整结果；筛选仍执行碰撞检查，转换仍执行筛选门禁与精度检查。`--strict-ik` 即使命中缓存也重新检查误差；普通固定摆放回放命中超差缓存时保留警告。可确定的首帧不可达或摆放搜索失败可被缓存，`--refresh-cache` 可重新尝试；临时 I/O 和 worker 崩溃不作为永久不可达缓存。

`--no-cache` 关闭磁盘读写，后台只保留少量最近结果，被淘汰后再选择可能重新求解；它不能与 `--prepare-mode all` 同时使用。打开窗口后修改机器人模型或配置时，后台生成缓存会按新模型刷新，当前窗口显示的机器人模型则需重启回放后更新。

对于安装了 JAX CUDA 插件但没有可用 CUPTI 的工作站，建议设置 `JAX_PLATFORMS=cpu`。否则，JAX 可能输出 `Unable to load cuPTI` 堆栈信息，然后继续使用 CPU；该信息不代表回放或 IK 失败。

### RM75B 夹爪开合

RM75B 的内部关节状态由七轴和两个平移夹指组成，初始 home 为夹指全开。回放把录制的归一化开度线性映射到两指：1 为 CAD 零位全开，0 为每指向内最大平移 18.134 mm；左指沿 −X、右指沿 +X 运动。夹指为弯曲形状，模型按 STL 整体 X 边界在闭合后保留约 0.2 mm 间隙，实际指尖间隙可能更大，该数值不代表实物净宽或抓取精度。

机械臂命令和 Link7 法兰接口仍为七轴，转换数据也保持既有七轴布局，不新增米制夹爪宽度列。夹爪开度保存在独立的 `observation.gripper.<side>.normalized` 辅助列；`handumi replay-joints` 遇到该列时逐帧恢复双指位置，旧七列数据仍可加载并保持 home 的夹指姿态。查看器使用七轴加双指的完整内部状态，但该状态不改变训练数据契约。模型、URDF 或夹爪配置变化会改变回放缓存键，使旧缓存失效。绝对 TCP 模式的碰撞盒覆盖夹爪主体和两指的整个开合行程；连续碰撞检查对夹指平移按实际位移计，旋转专用认证只用于旋转关节。

### RM75B 摆放与关键帧诊断

`absolute-tcp` 的自适应摆放先搜索靠近桌面原点的候选；近端搜索失败后，再用有限的宽半径候选寻找整段严格达标的摆放。实际结果和搜索预算记录在每段缓存中。搜索只检查有限个数值候选，未找到结果表示搜索未确定可行解；机器人模型、工具标定或输入改变后会生成新缓存键。

### 为整个数据集固定同一底座

RM75B 的 `dataset-xy-yaw` 模式先联合评估数据集全部 episode，搜索一组共享底座 XY 和水平转角；高度和倾斜沿用参考仿真摆放。完整严格通过的段数优先，其次比较严格通过的连续前缀帧比例及与参考摆放的距离。搜索使用有限候选，不保证全局最优或保证超过半数通过。默认先筛选代表帧，再对最多 8 个候选完整求解整个数据集，因此首次运行可能需要较长时间。

`absolute-tcp` 默认的首帧位置容差和逐帧位置误差上限均为 5 mm，逐帧姿态误差上限为 5°；这组默认值也适用于 RM75B 的共享底座搜索。可通过原有容差参数显式调整。首次估计会依次显示输入准备、候选生成、代表帧预筛和完整轨迹验证的进度；完整验证显示当前候选、episode 和已求解帧数。终端输出重定向时定期记录纯文本状态，`--progress off` 关闭进度显示。

```bash
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp \
  --base-placement dataset-xy-yaw --strict-ik --headless --prepare-mode all

# 显式重新估计底座；--refresh-cache 仅重算逐段轨迹
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp \
  --base-placement dataset-xy-yaw --refresh-base-placement \
  --strict-ik --headless --prepare-mode all
```

至少一段完整通过后，位姿保存至 `DATASET/.handumi/base_placement/rm75b.yaml`，估计详情和本次评估结果保存在同目录 JSON；成功轨迹进入现有逐段缓存。后续运行沿用已保存位姿，即使追加数据或更换模型、标定和精度参数也不会自动重估。重估没有找到完整通过段时保留旧位姿并返回失败。全部段都会尝试；任何段失败时全量命令非零退出，报告保留已完成的成功和失败信息。最终结果只为选中段另行导出 NPZ。

新模式始终验证完整原始帧，不支持 `--start-frame`、`--max-frames`、`--stride` 的裁剪设置或 `--only-manipulation`。首次搜索覆盖全部 episode；`--prepare-mode off` 和 `background` 只影响之后的逐段准备。高级选项 `--placement-search-candidates` 可调整完整验证候选数量。已有物理部署标定时应选择 `--deployment-profile sim`；搜索得到的是未经过实体验证的仿真标定。要在其他回放中使用保存的位姿，可传入 `--base-placement fixed --deployment-calibration DATASET/.handumi/base_placement/rm75b.yaml`。本模式沿用当前精度、关节连续性和自碰撞检查，未增加桌面或物体碰撞检查。

下面的诊断命令使用可信的桌面控制器标定快照，检查腕心内外半径必要条件，并对指定关键帧执行不受帧间步长约束的独立多初值 IK。它还复用正式回放完整轨迹，保存 JSON 报告与 NPZ。该诊断当前限定 RM75B 的 base 高度为零，且 base 朝向与桌面相同：

```bash
.venv/bin/python -m handumi.scripts.replay.diagnose_placement outputs/umi-demo \
  --episode 7 --frames 0,91,135 --seeds 24 --max-nfev 200 \
  --output outputs/replay_diagnostics/umi-demo/episode_7.json
```

环域检查考虑了 TCP 的位置和旋转误差对腕心的保守影响。关键帧在固定底座下独立求解，完整轨迹还必须通过原有的关节限位、帧间步长与 TCP 误差检查。独立 IK 搜索失败或整个有限摆放搜索失败，均不能单独证明几何不可达。

## 基于绝对桌面坐标的重定向

在已标定桌面工作空间中采集的录制数据通常会自动选择 `absolute-table`。检查新的机器人形态时，可以显式指定：

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-profile sim
```

`robot_from_table` 用于确定示教桌面坐标系在机器人世界坐标系中的位置，不会移动机器人底座。例如，OpenArm v1 的 URDF 基座固定在世界坐标系的 `Z=0`，肩部安装点位于 `Z=0.698 m`，仿真标定中的 `Z=0.28755 m` 是暂定的桌面高度。

各部署配置的用途和归属如下：

| 配置 | 位置 | 含义 |
| --- | --- | --- |
| `sim` | `configs/calibration/table/sim/<robot>.yaml` | 可移植的标准仿真布局；纳入版本控制，`scope: simulation`，不代表实验室实际布局 |
| `local` | 被 Git 忽略的 `configs/calibration/table/local/<robot>.yaml` | 某个实验室中实体机器人和桌面的实测摆放位置；`scope: physical` |
| `auto` | 已配置时使用 local，否则使用 sim | 便捷的默认选项；始终输出并保存最终选用的配置及文件 |

显式指定 `--deployment-calibration FILE` 会覆盖所有配置。请勿修改标准仿真文件以匹配实体实验室，也不要使用 `robot_from_table` 补偿错误的控制器到 TCP 标定。

### 配置实验室摆放位置

为每台已安装的机器人创建一份私有标定文件。目标目录已被 Git 忽略：

```bash
cp configs/calibration/table/local/example.yaml \
  configs/calibration/table/local/piper.yaml
```

将复制文件中的 `lab` 设置为下方使用的相同稳定标识，测量 `T_robot_world_table`，并在实体接触检查通过之前保持 `verified: false`。在 `configs/rig.yaml` 中选择该实验室：

```yaml
deployment:
  lab: my_research_lab
```

回放会自动发现 `local/piper.yaml`。只有当实验室将私有文件存放在此惯用目录之外时，才需要使用可选的 `deployment.table_calibrations.piper` 设置。

随后在实验室检查时显式要求使用该配置：

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/datasets/handumi-demo \
  --robot piper \
  --episode 0 \
  --deployment-profile local
```

`handumi calibrate verify --robot piper --device pico` 不接受仿真文件作为实体标定，并会报告尚未验证的本地文件。

### 使用数据集专用摆放位置进行可视化

标准摆放位置假定采集时标定板位于任务场景的近侧边缘，因此示教发生在桌面坐标系的 `+Y` 侧。如果录制时标定板位于任务区域远侧，示教位置就会落在桌面坐标系的负 `Y` 侧，映射后的目标会落到机器人底座上，导致严格的起始检查失败。可通过输出的 `source TCP workspace bounds` 诊断：如果 `Y` 范围大部分为负，问题出在标定板摆放位置。

对于此类数据集，编写专用的桌面 YAML 文件，将场景移回机械臂前方，并显式传入：

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/datasets/handumi-demo \
  --robot piper \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-calibration outputs/calibration/table/handumi-demo_piper_viz.yaml \
  --initial-position-tolerance-m 0.05
```

这样可以保留示教在世界坐标系中的运动，供检查使用，但不表示回放精度得到保证。在推导此修正的采集数据中，episode 0 的跟踪误差中位数为 `2.4 cm`，末段仍超过 `10 cm`，因为无论采用何种刚性摆放，示教都会超出 BiPiper 的双臂工作空间。请将此类 YAML 文件保存在 `outputs/calibration/table/` 下。采集时应摆放好 ChArUco 板，使操作发生在板的前方（`+Y` 方向，远离操作者），从源头避免此问题。

推导修正的过程完全属于后处理，无需修改原始数据集。请注意坐标系：桌面坐标系中 `+Z` 向上，`+Y` 沿水平方向远离操作者；头显世界坐标系则以 `+Y` 向上。因此，`Y` 范围大部分为负表示水平偏移。先从输出的 `source TCP workspace bounds` 读取示教范围，再平移标准摆放位置，使该范围居中位于机械臂前方，四元数保持不变。上述示例中的范围为 `y = [-0.273, 0.045]`（中心约为 `-0.11 m`），将标准位置 `[0.30, 0.0, 0.0]` 调整为 `[0.5422, -0.1106, 0.0446]`：向前移动约等于示教位于原点后方的深度，并在横向重新居中。对其他机器人以相同方式调整其摆放位置后，即可回放同一批录制数据，无需修改数据。

## OpenArm v1

当前 OpenArm 配置在离线回放中使用比实时遥操作更大的关节步长：

```yaml
replay:
  max_joint_delta: 0.35
```

这不会改变实体 OpenArm 的指令频率、速度限制、看门狗或跟随误差检查。仿真 URDF 在夹爪闭合至 `0 mm` 时，手指碰撞网格之间还保留约 `0.48 mm` 的间隙。实体后端保留其原有的电机闭合和张开标定。

对于 `outputs/20260714_224135`，暂定的桌面刚体变换得到以下结果：

| Episode | 最大 TCP 位置误差 | 以 3 cm 为阈值的结果 |
| --- | ---: | --- |
| 0 | 2.92 cm | 通过 |
| 1 | 4.71 cm | 检查不可达片段 |
| 2 | 4.34 cm | 检查不可达片段 |

请勿简单减小桌面在 X 方向的平移量来掩盖这些峰值。在同一录制数据中，`X` 小于或等于 `0.168 m` 会使 episode 0 跨越奇异分支，产生 17–20 cm 的误差。后续更适合采用可达范围限制或工作空间缩放策略，并保持实测桌面变换的准确性。

## TRLC-DK1

TRLC-DK1 目前支持在仿真中进行双臂运动学回放，暂未提供 HandUMI 实体硬件后端。

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot trlc_dk1 \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-profile sim
```

双臂 URDF 使用两个带命名空间的 DK1 从臂，底座间距暂定为 `0.60 m`，桌面变换也为暂定值。对于上述录制数据的 episode 0，当前配置得到的最大位置误差为 `0.22 cm`，最大姿态误差为 `22.19 度`。

TRLC 网格使用类似 `meshes/visual/base_link.glb` 的路径，相对于 `assets/trlc-dk1` 解析。如果 Viser 输出 `Can't find meshes/...` 且只显示轨迹线，请更新仓库并重启回放进程，以重新加载 URDF。

## Axol

Axol 支持使用相同的自动 absolute-table 流程，在仿真中进行双臂运动学回放：

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/datasets/handumi-screws \
  --robot axol \
  --episode 0
```

Axol URDF 的 `+X` 指向左臂，`+Y` 指向操作者，`+Z` 向上。因此，其暂定仿真标定将 HandUMI 桌面坐标系绕 Z 轴旋转 180 度，并将示教工作空间放在 Axol 世界坐标系中的 `[0.05714, 0.12376, 0.25022]` m。该位置根据包含三个 episode 的完整验证录制数据拟合，仍为 `verified: false`，并非实体桌面的实测结果。

使用配置的离线回放关节步长时，三个 episode 均通过默认的严格 IK 阈值检查：

| Episode | 平均位置误差 | 最大位置误差 | 最大姿态误差 |
| --- | ---: | ---: | ---: |
| 0 | 0.04 cm | 2.72 cm | 9.30 度 |
| 1 | 0.03 cm | 1.52 cm | 5.26 度 |
| 2 | 0.03 cm | 0.38 cm | 7.05 度 |

提供的 Axol 模型将 `left_gripper` 和 `right_gripper` 表示为固定连杆。录制的夹爪开度仍保留在轨迹元数据中，但需要带可驱动手指关节的 Axol URDF 才能在网格上显示开合。Axol 目前没有实体硬件后端。

## 回放关节级数据集

`handumi replay` 从不绑定机器人的采集数据求解 IK。`handumi replay-joints` 则使用已经包含关节数据的数据集，无需运行 IK：查看器准确显示数据集中存储的数值，正向运动学仅用于报告这些关节对应的 TCP 位置。通过它可以查看策略实际使用的训练数据。

```bash
JAX_PLATFORMS=cpu handumi replay-joints \
  outputs/datasets/handumi-demo-clean-piper-joints \
  --episode 0
```

查看器默认先播放 `--episode` 指定的段（默认 0），并在后台准备其他 episode；使用侧栏的 **Episode** 下拉框切换，当前段会循环播放。切换失败时保留当前画面。`--headless`、输出文件和 `--verify` 仅处理初始指定的段。

该命令读取数据集自身的编码。标准转换结果（`handumi.state_layout` 使用弧度和米）会直接展开为 URDF 关节；采用 LeRobot 从臂布局的数据集，无论由 HandUMI 导出还是由插件录制（`robot_type: bi_piper_follower`、`bi_openarm_follower`），都通过导出器使用的相同布局定义解码，因此参考采集数据和 HandUMI 导出数据可以在同一查看器中回放。如果元数据未指定布局，请传入 `--state-layout`；如果插件录制时使用了角度选项，请传入 `--use-degrees`。缺少此检测时，插件向量与标准向量同为 14 列，`-99.8` 会被直接当作弧度渲染，且不会报错。

`--verify` 通过两种不同方式，将标准转换结果与源采集数据进行比较：

- **完整性**要求精确一致。从相互重叠的 `observation.state` 和 `action` 视图重建求解轨迹后，其结果必须与转换时记录的 `ik_fidelity.output_state_sha256` 逐字节匹配。
- **求解器一致性**会重新求解采集数据并在任务空间比较（默认容差为 `1 mm`、`0.5 deg`）。这不是逐位精确检查，因为 IK 求解器的结果在不同运行之间无法完全复现：在同一台机器上，使用相同参数连续求解同一个 episode，两次结果相差约 `2e-3 rad`。因此转换会复用轨迹回放已经求解的结果，`qpos_sha256` 也只与存储的轨迹匹配。

```bash
JAX_PLATFORMS=cpu handumi replay-joints \
  outputs/datasets/handumi-demo-clean-piper-joints \
  --episode 0 --headless --verify --strict-verify
```

如需在实体机器人上回放同一数据集，请使用 `handumi replay-real`；参阅[实体机器人遥操作](../physical_robots/real_teleoperation.md)。

## 阅读诊断信息

回放在求解前会输出源工具标识和标定哈希值。如果录制时使用的实体工具是 Piper，那么回放到 OpenArm、TRLC 或 Axol 时看到 `source tool: robot=piper` 属于正常情况。与该标识绑定的控制器到 TCP 标定快照用于重建示教时的 Piper TCP，随后再应用目标机器人形态。

重要输出字段包括：

- `deployment calibration`：最终选用的配置、范围、验证状态和源文件；
- `source TCP workspace bounds`：可达性检查使用的、与机器人无关的桌面坐标系空间范围；
- `start prepared`：初始求解迭代次数和首帧误差；
- `approach`：回放前的过渡过程，使机器人从初始姿态逐渐移动到准备好的起始姿态，查看器因此与仿真遥操作一样从初始姿态打开。该过程仅用于渲染，单独存储为 `approach_qpos`，不进入 `qpos` 或误差指标。可通过 `--approach-seconds` 调整（`0` 表示直接从第 0 帧开始回放）；
- `IK EE error`：双臂位置和姿态的平均及最大误差；
- `max_joint_delta`：为机器人形态选用的离线关节步长限制；
- `saved`：包含目标、实际达到的 TCP 位姿、误差和 qpos 的 NPZ 文件。

自动验证时请使用 `--strict-ik`。当最大位置或姿态误差超过所选阈值时，程序会退出：

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot trlc_dk1 \
  --episode 0 \
  --headless \
  --strict-ik
```
