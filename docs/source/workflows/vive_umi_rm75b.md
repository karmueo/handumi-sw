# Vive UMI 采集与 RM75B 仿真

本文适用于一台 Vive Tracker、一台鱼眼 USB 相机及双 ArUco 夹爪。设备映射为逻辑右侧，录制 30 FPS；运行时使用 SteamVR 和 Python OpenVR，不需要 ROS。RM75B 回放使用七轴模型的 Link7 法兰，不包含真实夹爪接触。

## 1. 安装

在项目根目录执行：

```bash
bash install.sh --device vive --sim
source .venv/bin/activate
```

安装器保留已有 `configs/rig.yaml`。若该文件仍选择旧设备，后续命令显式添加 `--device vive`。新配置默认选择 Vive 和 RM75B。旧设备可以通过 `--device meta` 或 `--device pico` 安装；模板见 `configs/rig.legacy.example.yaml`。

启动 SteamVR，确认 Tracker 已连接且定位有效。为相机使用 `/dev/v4l/by-path/...-video-index0` 稳定路径，避免摄像头编号变化后绑定错误。

## 2. 导入现有几何标定

确认相机、Tracker、标签及夹爪尖端安装均未改变。以下文件路径和序列号对应参考项目；相机的 `by-path` 路径取决于 USB 端口，导入前请用 `ls -l /dev/v4l/by-path/` 确认当前的 `video-index0` 路径，并在需要时替换示例中的 `--camera-device`。更换相机或改变安装关系后，应先重新标定几何参数。

```bash
handumi calibrate umi import \
  --camera /home/scl/work/UMI/FastUMI_Data/config/calibration/T_cam_tracker.yaml \
  --tcp /home/scl/work/UMI/FastUMI_Data/config/calibration/T_u_cam_tcp.yaml \
  --gripper /home/scl/work/UMI/FastUMI_Data/ros2_ws/src/gripper_openness/config/gripper_calibration.yaml \
  --serial LHR-B77A06A7 \
  --camera-device /dev/v4l/by-path/pci-0000:c2:00.4-usb-0:1.4.4:1.0-video-index0
```

若外参旁存在同名 `.binding.yaml`，导入会校验 Tracker 序列号及文件摘要。输出默认位于 `configs/calibration/umi/local/calibration.yaml`，该目录不提交、不打包。相机 USB 端口变化时，使用新路径重新导入；导入会替换现有快照并清除时间标定，需要重新执行下一节的时间标定。原 ROS/V4L2 时间偏移不会被采用。

可先运行 `handumi visualize --device vive` 检查图像、双标签开度和定位。窗口使用 Rerun；键盘命令在启动程序的终端中输入。真实净宽尚未测量，界面显示 0～1 开度。

## 3. 标定原生时间偏移

固定 6×6 AprilGrid：tag36h11、标签边长 55 mm、间距/边长比 0.3。确保实物与参考板一致。将该标定板放入相机画面并至少保持 6 个标签可见，再移动 UMI，进行变速平移、多轴旋转和起停；覆盖不同距离和视角。房间内的零散标签和夹爪标签不能替代这块标定板。

时间标定会自动打开 Rerun 窗口，显示采集画面和本帧检测到的标定板标签、角点及 ID。角点定位在标签黑色方块的外边界，无法确认外边界的标签不会用于标定。绿色的 `OBSERVATION ACCEPTED` 表示该帧已作为标定观测采纳；黄色的 `NO OBSERVATION` 表示本帧未采纳，通常需要确保至少 6 个标签可见且姿态求解成功。画面还显示识别的标签数、已采纳帧数和剩余时间。预览使用参与检测的帧，约为相机帧率的一半；帧被采纳并不代表最终标定已通过质量检查。

```bash
handumi calibrate umi time --duration 30 \
  --capture-output configs/calibration/umi/local/time-observations.json
handumi calibrate umi verify
```

无图形界面时，可在采集命令后添加 `--no-preview`。时间标定与预览共享同一相机和 Tracker，无需另起 `handumi visualize`。预览图像不会写入下方的观测 JSON。

采集文件保存角点和定位时间线，便于离线重算：

```bash
handumi calibrate umi time \
  --input configs/calibration/umi/local/time-observations.json
```

求解固定安装外参，仅优化固定板位姿和时间偏移。质量门包括至少 30 帧、时间块验证集、重投影中位数 ≤1 px/P95 ≤2 px、闭环 ≤5 mm/1°。静止或退化运动、跟踪缺失、偏移触边、几何不符都会拒绝；Tracker 的实际位移或旋转需要明显超过静止时的定位噪声。失败时不会覆盖原标定；修改设备、分辨率、帧率或几何配置后必须重新标定。

## 4. 桌面坐标系标定

继续使用时间标定的 6×6 AprilGrid，将板平放并固定在桌面上：IDs 0–5
位于靠近操作者的一行，从左到右递增；ID 6 位于 ID 0 的远侧，IDs 30–35
位于最远的一行。坐标原点为标签区域中心投影到真实桌面的位置，`+X` 向右、
`+Y` 远离操作者、`+Z` 向上。纸张白边不计入标签区域中心。

测量标签印刷平面距真实桌面的高度，包括硬板和垫片。下面的 `0.0082` 仅为
8.2 mm 示例，必须替换为实际高度；直接贴在桌面上可显式填写 `0`。
`--board-height-m` 必须提供非负有限值，不能省略。

```bash
handumi calibrate umi table --board-height-m 0.0082
handumi calibrate umi verify-table
```

桌面 `table` 默认启用停稳采集；`--stationary` 仍保留以兼容旧命令，但无需再
显式添加。每次采集先预热 5 秒，随后持续采集且不受采集时长限制：`table`
不读取也不校验 `--duration`。每换到一个新视角，保持大致稳定约 0.3 秒后开始
收集 15 帧合格观测；允许相对观察窗口起始位姿不超过 5 mm、1.5° 的轻微手抖。
明显移动时暂停收帧，已收集的有效帧保留。

每个视角必须收满 15 帧；新视角相对所有已完成视角至少移动 5 cm 或旋转 10°。
重复视角、无效帧和未收满就切换的视角不会增加完成数量。完成第 15 个独立视角的
最后一帧后，先保存该帧和逐帧诊断，把预览刷新为 15/15，再自动结束采集并执行
原有求解和质量验收。`table` 采满视角仍需通过最终质量验收。

Rerun 顶部状态区和图像叠字只使用两种英文状态：预热、移动或观测不合格时显示
`CAPTURING`；完成当前视角或停留在已完成视角时显示 `CURRENT VIEW COMPLETE`。
第二行显示 `Completed views: N/15`。该模式不显示倒计时、帧数或详细拒绝原因；
相机画面、标定板轮廓和标签保留，详细信息继续写入逐帧诊断。`--no-preview`
仍可关闭界面。

`verify-table` 默认行为保持不变：预热 5 秒后按 `--duration`（默认 30 秒）
连续采集。需要停稳收帧时可显式添加 `--stationary`，此时仍受 `--duration`
计时约束并沿用每视角 15 帧的原有要求；例如：

```bash
handumi calibrate umi verify-table --stationary --duration 60
```

`verify-table` 完成采集后直接检查已保存的变换，不重新拟合。`table` 和
`verify-table` 是两次独立采集；`table` 完成后必须按原流程运行
`handumi calibrate umi verify-table`，通过后才能录制。

启动时打印实际标定文件、设备、分辨率、帧率及时间偏移。`table` 预热 5 秒后
自动收满 15 个视角；`verify-table` 预热 5 秒后按 `--duration` 采集，默认
30 秒。保持板固定并缓慢移动相机，覆盖不同位置和倾角。单帧原图重投影中位数须
≤1 px、P95 ≤2 px。逐帧诊断保留详细拒绝原因，供离线分析。
时间偏移标定仍使用变速运动。最终桌面验收阈值保持不变。

`tracker.jsonl` 逐批保存预热及采集期间的原始高频定位，包括失效样本；
`observations.json` 的 `raw_tracker` 同时保存时间戳（纳秒）、原始设备位姿和
有效性，`sampling_policy` 保存采样门限、目标视角数和结束方式
（`target_views_reached` 或 `duration_elapsed`）。失败或 Ctrl+C 后保留已取得的
数据。

`table` 自动完成 15 个独立视角，每个完成视角各 15 帧，因此最终至少保留 225 帧
有效观测；未完成视角中已接收的合格帧也会保留，切换未完成视角后总数可能更多。
`verify-table` 默认重新采集约 30 秒，保持板固定，缓慢移动 UMI，覆盖至少四个
不同视角。两个命令选中的视角之间均需相距至少 5 cm 或旋转至少 10°，每帧至少
识别六个标签；`verify-table` 总计至少 30 帧有效观测。采集使用现有相机内参、
Tracker→Camera 外参及原生时间偏移，无须夹爪上的双标签同时可见。

桌面求解和独立验证均要求平移闭环 RMS ≤8 mm、旋转 RMS ≤1°，原图重投影
中位数 ≤3 px/P95 ≤10 px。`table` 创建新的会话身份，之后必须通过
`verify-table` 才能录制。独立验证直接检查已保存的变换，不重新拟合；验证
失败或中断后禁止使用该会话录制，重新验证通过可恢复使用。求解失败保留旧文件。

两个命令会自动在桌面会话文件旁的 `table-diagnostics/` 下，为每次采集创建独立
诊断目录，启动和结束时均打印完整路径。默认位于
`configs/calibration/umi/local/table-diagnostics/`，不提交到 Git。成功、失败或
按 Ctrl+C 中断都会保留已采集的数据：

- `report.json`：本次参数、成功/失败状态、有效帧数、目标视角数、结束方式、拒绝
  原因统计、检测耗时和检测前后帧龄；完成误差计算时包含全部验收指标及阈值。
- `frames.jsonl`：逐帧立即保存标签 ID、ArUco 原始检出数和被拒候选数、接收/拒绝
  原因、同步误差、检测耗时、Tracker 位姿和检测到的板角点，便于分析未被接收的帧。
- `images/`：标签少于 20 个的帧以 JPEG（质量 95）保存原图，间隔至少 250 ms，
  最多 150 张；`frames.jsonl` 的 `image` 字段指向对应文件。
- `observations.json`：参与求解的角点、板位姿和同步 Tracker 位姿；中断时保存
  已采纳的部分观测。
- `calibration.yaml`、`session-input.yaml`（独立验证时）：本次使用的标定快照。
- `error.txt`（失败时）、`session-result.yaml`（成功时）：异常堆栈或结果快照。

除低标签帧外，诊断不保存相机图像。若进程被强制结束，汇总文件可能仍标记为 `collecting`，
可先检查已经写入的 `frames.jsonl`。`--capture-output <路径>` 也可额外导出
桌面观测；诊断数据用于离线分析，正式 `table` / `verify-table` 仍要求重新采集。
日志中的 `passed` 表示本次命令通过，`table` 后仍须完成独立验证。

桌面文件默认保存到 `configs/calibration/umi/local/session.yaml`，保持独立于
几何和时间标定。两个命令和录制命令都支持 `--calibration` 与
`--session-calibration`，路径优先级为命令行、`configs/rig.yaml` 的
`umi.calibration` / `umi.session_calibration`、上述默认本地路径。

验证通过后可移走板，保持桌面和定位设置不变。SteamVR 重定位、重启、定位
基准变化后，应把板放回原有位置并运行 `verify-table`；失败时重新求解并验证，
随后使用新的数据集目录。板移走后，单个移动 Tracker 无法可靠判断全局原点
是否漂移。通过软件质量门仍需人工确认板的尺寸、方向和真实桌面高度。

## 5. 诊断和录制

```bash
handumi doctor --device vive --no-preview
handumi record --device vive --output-dir outputs/umi-demo \
  --repo-id local/umi-demo --task 'Pick and place'
```

在终端中操作：

- 空格：开始录制；录制中再次按下则保存。
- Esc：录制中取消当前段；待录时退出。
- Ctrl+C：取消当前段并退出。

每段从有效观测开始。ArUco 双标签短暂丢失时，默认在 500 毫秒内沿用最近一次有效开度；丢标超过保持期限、定位失效、图像重复或同步失败，都会丢弃当前段并回到待录；再次空格才开始新段。保存操作不会越过失效检查。退出不会删除已保存片段。

正式录制在打开设备和处理输出目录前检查桌面文件及独立验证结果。
没有桌面标定时，`doctor` 和 `visualize` 仍可用于设备检查，并明确显示 OpenVR
原始参考系；显式指定了错误的桌面路径时会报错。

Rerun 显示固定参考坐标轴与 TCP 的完整 XYZ 坐标轴，红/绿/蓝分别表示 X/Y/Z。
TCP 默认轴长 8 cm，可用 `--tcp-axis-length-m 0.12` 调整；只允许正的有限数值。
TCP 轴随位置和旋转更新，定位失效时隐藏、恢复后重显；仅夹爪标签丢失时仍可
观察有效的 TCP 姿态。

夹爪漏检保持期限由 `--gripper-hold-ms` 控制，单位为整数毫秒，默认 `500`；
`0` 彻底关闭保持，任何漏检帧都按无效处理。该参数在 `handumi doctor`、
`handumi record` 和 `handumi visualize` 的 Vive UMI 模式中统一可用；负数或
非整数会在加载标定、处理输出目录和打开设备前报错。

使用相同参数添加 `--resume` 可续录；设备、几何/时间标定、桌面会话身份和
数据契约必须一致。重复运行 `verify-table` 不改变桌面身份，可以继续原数据集，
数据集保留初次录制的标定快照。重新运行 `table` 会生成新身份，必须使用新的
`--output-dir`，旧的未标定数据集也不能续录。`--no-preview` 可关闭 Rerun，
录制仍需要交互终端。

## 6. 数据检查和仿真

```bash
handumi validate outputs/umi-demo
handumi replay outputs/umi-demo --robot rm75b --retarget-mode local-relative --strict-ik
handumi dataset qa outputs/umi-demo --robot rm75b --deployment-profile sim
handumi convert outputs/umi-demo --robot rm75b --retarget-mode local-relative --output outputs/umi-joints
handumi replay-joints outputs/umi-joints --robot rm75b
```

质量检查不通过时，先按报告重录或使用现有筛选流程。默认从 `[0,20,0,70,0,90,90]°` 开始，以 1∶1 相对运动映射至 Link7。查看目标/实际轨迹、IK 误差、开度百分比和曲线。目标机器人没有夹爪关节，转换结果仅有七轴，开度作为独立辅助列保存。

原始数据仍是 16 维，未采集左侧中立填充，净宽槽位为零占位。新版 `synchronized_sources_v2` 和 `width_available=false` 表明没有米制净宽。读取器必须使用 `observation.gripper.right.normalized`，禁止将零占位解释为实测闭合；要求米制净宽的转换会拒绝该数据。

`observation.state` 保存桌面坐标系中的 Tracker 锚点位姿，工具 TCP 偏移仍由
后续流程应用一次；原始 OpenVR 位姿与桌面变换保存在 tracking 列。数据集的
`tracking_workspace=table` 和 `umi_table_session` 保存参考系及完整桌面快照。
回放/转换的 `auto` 模式因此选择 `absolute-table`，需要目标机器人的
`robot_from_table` 配置；上方命令显式使用 `local-relative` 演示相对运动回放。
桌面标定不包含机器人与桌面的安装关系。

## 7. 实机验收

先完成时间和桌面标定，再分别录制静止、平移旋转、夹爪开合片段。检查图像与
开度同帧、TCP 轴旋转正确且定位稳定，将 TCP 接触桌面时高度应接近零。
录制中遮挡标签、遮挡 Tracker，应立即丢弃该段；遮挡 Tracker 时 TCP 三轴应
隐藏，恢复后需重新按空格。最后保存有效片段并完成上述质量检查、转换及七轴回放。

自动测试不代替实机验收。需要设备操作者完成上述动作，并确认实物标定板尺寸、SteamVR 定位和相机视野。
