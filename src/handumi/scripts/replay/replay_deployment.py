"""统一选择数据集共享摆放，并在缺少标准仿真标定时估计数据集专用布局。"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml

from handumi.calibration.deployment import load_deployment_calibration
from handumi.calibration.simulation_placement import fit_simulation_placement


def select_saved_dataset_placement(args, *, allow_explicit: bool = False) -> bool:
    """为自动配置加载已验证共享仿真布局，并统一为绝对 TCP 求解模式。

    显式其他模式、摆放、标定和本地实体部署优先；不存在共享文件时返回 False。
    筛选和转换可通过 allow_explicit 加载用户显式指定的 dataset-xy-yaw；
    回放原有的显式搜索与刷新流程由其入口自行执行。布局无效时抛出错误。
    """
    from handumi.calibration.deployment import local_calibration_path
    from handumi.config import DEFAULT_RIG_CONFIG

    robot = getattr(args, "robot", None) or getattr(args, "embodiment", None)
    root = getattr(args, "dataset_root", None) or getattr(args, "root", None)
    mode = getattr(args, "retarget_mode", "auto")
    policy = getattr(args, "base_placement", "auto")
    explicit_shared = allow_explicit and policy == "dataset-xy-yaw"
    if not robot or root is None or (policy != "auto" and not explicit_shared):
        return False
    if robot != "rm75b":
        if explicit_shared:
            raise SystemExit("dataset-xy-yaw currently requires --robot rm75b.")
        return False
    if mode not in {"auto", "absolute-tcp"} or getattr(args, "deployment_calibration", None) is not None:
        return False
    profile = getattr(args, "deployment_profile", "auto")
    rig = getattr(args, "rig_config", None) or DEFAULT_RIG_CONFIG
    local = profile == "local" or (profile == "auto" and local_calibration_path(robot, rig_config=rig) is not None)
    if local:
        if explicit_shared:
            raise SystemExit("dataset-xy-yaw requires simulation deployment; local physical calibration takes precedence.")
        return False
    path = Path(root).resolve() / ".handumi" / "base_placement" / f"{robot}.yaml"
    if not path.exists():
        if explicit_shared:
            raise SystemExit(f"Dataset-wide base placement has not been resolved: {path}")
        return False
    selection = load_deployment_calibration(path, expected_robot=robot, profile="dataset-xy-yaw")
    if selection.scope != "simulation" or selection.source != "dataset_xy_yaw_search":
        raise SystemExit(f"Dataset-wide base placement {path} must be a dataset_xy_yaw_search simulation calibration.")
    args.retarget_mode = "absolute-tcp"
    args.base_placement = "dataset-xy-yaw"
    args.deployment_selection = selection
    return True


def estimate_dataset_deployment(args):
    """复用数据集的估计布局，或按实际 TCP 输入生成未实测的仿真布局。

    参数与回放入口相同；只写数据集私有目录，不改原始轨迹或标准配置。
    读取、输入标定或模型异常继续向上传递；估计结果不保证 IK 可达。
    """
    from handumi.calibration.control_tcp import ControllerTcpCalibration
    from handumi.calibration.robot_tcp import load_robot_tcp_calibration
    from handumi.dataset import DatasetRef, ensure_metadata
    from handumi.robots.registry import load_embodiment
    from handumi.robots.utils import mat_to_pose7
    from handumi.scripts.replay import replay_in_sim as replay
    from handumi.scripts.replay.replay_prepare import _source_calibration

    path = Path(args.dataset_root) / ".handumi" / "deployment" / f"{args.robot}.yaml"
    print("[deployment] warning: no robot/table placement is defined; "
          "using a data-derived simulation estimate, not physically verified.", flush=True)
    print("[deployment] assumption: robot base and tabletop have the same height; "
          "only horizontal position and yaw are estimated.", flush=True)
    if path.exists():
        selection = load_deployment_calibration(path, expected_robot=args.robot, profile="estimated")
        if selection.scope != "simulation":
            raise SystemExit(f"Estimated deployment {path} must have simulation scope.")
        print(f"[deployment] reusing estimated simulation placement: {path}", flush=True)
        return selection

    info = ensure_metadata(DatasetRef.from_repo_id(args.repo_id, root=args.dataset_root, revision=args.revision))
    if info.get("handumi", {}).get("tracking_workspace") != "table":
        raise ValueError("Estimating robot/table placement requires a dataset recorded in the table workspace.")
    runtime = load_embodiment(args.robot)  # 活动侧、HOME 关节和模型几何。
    model = runtime.load_urdf(load_meshes=False)
    model.update_cfg(dict(zip(runtime.joint_names, runtime.config.home_q, strict=True)))
    home_poses = {side: mat_to_pose7(model.get_transform(arm.ee_link))
                  for side, arm in runtime.config.arms.items()}
    if args.retarget_mode == "absolute-tcp":
        tool = load_robot_tcp_calibration(args.robot, runtime.config.arms,
                                         explicit_path=args.robot_tcp_calibration)
        home_poses = {side: tool.achieved_tcp(side, pose) for side, pose in home_poses.items()}
    device = replay._resolved_controller_device(args, info)  # 与正式求解相同的设备和 TCP 来源。
    source = _source_calibration(args, info, device)
    calibration = None if source is None else ControllerTcpCalibration(
        left=np.asarray(source["left"]), right=np.asarray(source["right"]),
    )
    trajectories = {side: [] for side in home_poses}  # 每段等量抽样，避免长片段独占估计。
    count = int(info.get("total_episodes", 0))
    for episode in range(count):
        trial = argparse.Namespace(**vars(args))
        trial.episode = episode
        states, _, _, _ = replay.load_episode_states(trial)
        indices = np.unique(np.linspace(0, len(states) - 1, min(len(states), 64), dtype=int))
        states = states[indices]
        if calibration is not None:
            states = replay.apply_tcp_calibration_to_states(states, calibration, source["source"])[0]
        for side in home_poses:
            start = 0 if side == "left" else 7
            trajectories[side].append(states[:, start:start + 7])
    pose = fit_simulation_placement(
        {side: np.concatenate(parts) if parts else np.empty((0, 7))
         for side, parts in trajectories.items()}, home_poses,
    )
    data = {
        "schema_version": 1, "kind": "handumi_robot_table_calibration",
        "robot": args.robot, "scope": "simulation", "verified": False,
        "source": "dataset_home_alignment_v1",
        "estimation": {"episodes": count, "assumptions": ["base_at_table_height", "upright_base"],
                       "method": "trajectory_centers_and_home_poses", "ik_validated": False},
        "calibration": {"robot_from_table": {
            "position": pose[:3].tolist(), "quaternion": pose[3:].tolist(),
        }},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"[deployment] estimated simulation placement saved: {path}", flush=True)
    return load_deployment_calibration(path, expected_robot=args.robot, profile="estimated")
