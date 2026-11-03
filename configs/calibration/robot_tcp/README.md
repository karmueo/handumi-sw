# 机器人 TCP 标定

`absolute-tcp` 使用示教 TCP 的桌面绝对位姿求机器人 TCP 目标，再通过本标定求 URDF 末端链接目标。快照的 `ee_to_tcp` 为末端链接到 TCP 的米制齐次矩阵，使用列向量。

将 `example.yaml` 复制为 `local/<robot>.yaml` 并填入实际外参；也可使用 `--robot-tcp-calibration PATH` 指定快照。`local/` 中的机器专用文件不进入 Git 和发布包。请记录标定来源及摘要，实测后更新 `verified`。

RM75B 的本地快照可以由 FastUMI 的相机到法兰外参 `T_cam_hand` 和相机到 TCP 外参 `T_d_cam_tcp` 组合：`inverse(T_cam_hand) @ T_d_cam_tcp`。当前来源的部分工具尺寸尚待实测，标注 `verified: false`。

桌面到 base 的参考变换依次使用 `--deployment-calibration` 指定的文件、本地已登记标定；两者都没有时以坐标系重合作为初始参考。`--deployment-profile sim` 显式选择仿真摆放。显式文件不存在或格式无效时会报错。

`absolute-tcp` 的 `--base-placement auto` 在只有重合假设时，为每条 episode 单独选择固定的虚拟底座位置：若原位置整段严格达标则保持不动，否则只调整桌面 X/Y、固定 Z 和朝向，联合首帧姿态与关节限制生成候选，并用原有关节步长限制逐帧验证；最后对通过的结果作有界的径向和邻近方向细化。结果是搜索预算与实际分辨率内找到的较小数值解，非凸问题不保证全局最优。每条 episode 的实际摆放和搜索诊断记录在输出中；搜索结果属于 simulation，未经过物理测量。

`auto` 尊重显式、本地物理标定和显式仿真摆放，保持它们固定。`--base-placement fixed` 固定当前参考，用于复现原位置的成功或失败；`--base-placement episode-xy` 明确启用虚拟搜索，不能覆盖实测物理标定。切换 episode 时界面在桌面坐标中移动机器人，桌面、目标和场景保持一致。

示教和机器人 TCP 使用相同轴约定；两端相机的安装外参差异仍会产生视差。默认 IK 门槛为位置 5 mm、朝向 5°，不包含标定本身的物理误差。首帧的位置和朝向必须同时达标。固定摆放下后续超差时，普通回放给出帧号及误差，`--strict-ik` 回放退出；自适应摆放始终要求整段通过，搜索失败时退出。筛选会拒绝超差 episode，转换不会将其写成成功结果。

输入沿用当前 `controller_raw_compact` 数据布局：记录的控制器位姿须有明确的 controller→TCP 标定，回放时先换算为桌面 TCP。该布局声明与 `applied_to_state: true` 相互矛盾，程序会拒绝这类数据；未知布局不自动猜测其位姿含义。

```bash
handumi replay outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --strict-ik

handumi dataset screen outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --episodes 0

handumi convert outputs/umi-demo --robot rm75b --retarget-mode absolute-tcp --episodes 0 --output outputs/umi-demo-rm75b
```

回放窗口侧栏的 `Episode` 下拉框可直接切换采集片段，编号从 0 开始；`--episode` 指定初始片段。切换会沿用当前机器人、标定、帧筛选及 IK 参数，加载期间暂停播放，成功后同步更新轨迹、误差、帧范围和夹爪曲线。加载或严格 IK 检查失败时保留当前回放，并在侧栏显示原因；切换操作不会额外保存求解文件。

三个入口应使用相同的工具、桌面标定和 `--base-placement` 策略。若显式提供工具快照或桌面标定，请在筛选和转换时传入相同的 `--robot-tcp-calibration PATH` 和 `--deployment-calibration PATH`；转换会检查筛选报告的求解配置，配置不匹配时需重新筛选。筛选使用 `--max-position-error-m`、`--max-rotation-error-deg` 修改门槛；回放和转换分别使用 `--max-ik-position-error-m`、`--max-ik-rotation-error-deg`，两边应设为相同值。
