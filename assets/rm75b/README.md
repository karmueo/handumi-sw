# RM75B 仿真模型

七轴机械臂 URDF 来自用户提供的 FastUMI_Data/assets/rm_75.urdf；机械臂网格来自 RealManRobot ROS 2 仓库的 rm_description/meshes/rm_75_arm，供应商仓库提交为 `08820fe5ae8c840c12d646106d27632128cde0f3`。机械臂资源的许可证见 LICENSE（Apache-2.0）。

夹爪定义和三个 STL 网格来自用户提供的 FastUMI_Data 模型，源 CAD 为 `夹爪.stp`；供应商机械臂许可证不代表夹爪资源的许可证。夹爪包含主体、底板、相机和左右夹指，移除了手持握柄。STL 使用毫米和原装配坐标，加载时缩放为米。

原始模型名称为 `rm_75_description`，未进一步区分 RM75B；本项目使用 `rm75b` 作为设备配置名。模型保留七个旋转关节，以 `Link7` 为法兰。夹爪主体相对法兰沿 +Z 偏移 6 mm 安装，左右夹指固定在 CAD 姿态，开度数据不会驱动其运动。该偏移用于模型装配，不能作为经过实测的抓取 TCP；TCP 应单独标定。夹爪没有实测质量和惯量，当前回放也不模拟夹爪接触动力学。
