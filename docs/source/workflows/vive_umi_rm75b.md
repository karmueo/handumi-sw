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

## 4. 诊断和录制

```bash
handumi doctor --device vive --no-preview
handumi record --device vive --output-dir outputs/umi-demo \
  --repo-id local/umi-demo --task 'Pick and place'
```

在终端中操作：

- 空格：开始录制；录制中再次按下则保存。
- Esc：录制中取消当前段；待录时退出。
- Ctrl+C：取消当前段并退出。

每段从有效观测开始。任何待写入帧丢标、定位失效、图像重复或同步失败，都会丢弃当前段并回到待录；再次空格才开始新段。保存操作不会越过失效检查。退出不会删除已保存片段。

使用相同参数添加 `--resume` 可续录；设备、标定和数据契约必须一致。`--no-preview` 可关闭 Rerun，录制仍需要交互终端。

## 5. 数据检查和仿真

```bash
handumi validate outputs/umi-demo
handumi replay outputs/umi-demo --robot rm75b --retarget-mode local-relative --strict-ik
handumi dataset qa outputs/umi-demo --robot rm75b --deployment-profile sim
handumi convert outputs/umi-demo --robot rm75b --output outputs/umi-joints
handumi replay-joints outputs/umi-joints --robot rm75b
```

质量检查不通过时，先按报告重录或使用现有筛选流程。默认从 `[0,20,0,70,0,90,90]°` 开始，以 1∶1 相对运动映射至 Link7。查看目标/实际轨迹、IK 误差、开度百分比和曲线。目标机器人没有夹爪关节，转换结果仅有七轴，开度作为独立辅助列保存。

原始数据仍是 16 维，未采集左侧中立填充，净宽槽位为零占位。新版 `synchronized_sources_v2` 和 `width_available=false` 表明没有米制净宽。读取器必须使用 `observation.gripper.right.normalized`，禁止将零占位解释为实测闭合；要求米制净宽的转换会拒绝该数据。

## 6. 实机验收

先完成时间标定，再分别录制静止、平移旋转、夹爪开合片段。检查图像与开度同帧、方向正确且定位稳定。录制中遮挡标签、遮挡 Tracker，应立即丢弃该段，恢复后需重新按空格。最后保存有效片段并完成上述质量检查、转换及七轴回放。

自动测试不代替实机验收。需要设备操作者完成上述动作，并确认实物标定板尺寸、SteamVR 定位和相机视野。
