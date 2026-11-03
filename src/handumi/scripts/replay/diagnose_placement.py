"""诊断 RM75B 桌面底座摆放、独立关键帧 IK 与完整连续轨迹。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from handumi.calibration.deployment import resolve_deployment_calibration
from handumi.calibration.robot_tcp import load_robot_tcp_calibration
from handumi.robots.registry import load_embodiment
from handumi.robots.utils import pose7_to_mat, pose_inv
from handumi.scripts.replay import replay_in_sim as replay
from handumi.scripts.replay.replay_prepare import prepare_episode


def arm_geometry(path):
    """从已核对结构的 RM75B URDF 读取肩高、臂长、腕部偏移和肘限位。"""
    root = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    xyz = [
        np.fromstring(joints[f"joint{i}"].find("origin").get("xyz"), sep=" ")
        for i in range(1, 8)
    ]
    expected_axes = (
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 1],
    )
    expected_rpy = (
        [-1.5708, 0, 0],
        [1.5708, 0, 0],
        [-1.5708, 0, 0],
        [1.5708, 0, 0],
        [-1.5708, 0, 0],
        [1.5708, 0, 0],
        [0, 0, 0],
    )
    for i in range(7):
        joint = joints[f"joint{i + 1}"]
        if (
            joint.get("type") != "revolute"
            or joint.find("parent").get("link")
            != ("base_link" if i == 0 else f"Link{i}")
            or joint.find("child").get("link") != f"Link{i + 1}"
            or not np.allclose(
                np.fromstring(joint.find("axis").get("xyz"), sep=" "), expected_axes[i]
            )
            or not np.allclose(
                np.fromstring(joint.find("origin").get("rpy"), sep=" "), expected_rpy[i]
            )
            or not np.allclose(xyz[i][:2], 0)
            or (i in (1, 3, 5) and not np.allclose(xyz[i], 0))
        ):
            raise ValueError("URDF 结构与本诊断的 RM75B 几何假设不符")
    limit = joints["joint4"].find("limit")
    # 保守地扩大可折叠角度，覆盖 URDF 的小数 RPY 近似。
    elbow = min(
        np.pi,
        max(abs(float(limit.get("lower"))), abs(float(limit.get("upper")))) + 1e-4,
    )
    upper, lower = abs(xyz[2][2]), abs(xyz[4][2])
    minimum = np.sqrt(upper**2 + lower**2 + 2 * upper * lower * np.cos(elbow))
    return float(xyz[0][2]), float(upper + lower), float(minimum), float(xyz[6][2])


def wrist_envelope(
    tcp, tool, geometry, position_tolerance=0.005, rotation_tolerance_deg=5.0
):
    """计算误差门槛下的腕心，返回宽松外半径与收缩内半径必要条件。"""
    shoulder_z, outer, inner, wrist_offset = geometry
    wrist_tcp = np.linalg.inv(tool) @ np.array([0.0, 0.0, -wrist_offset, 1.0])
    tolerance = (
        position_tolerance
        + 2
        * np.linalg.norm(wrist_tcp[:3])
        * np.sin(np.deg2rad(rotation_tolerance_deg) / 2)
        + 1e-5
    )
    wrist = np.einsum("nij,j->ni", tcp, wrist_tcp)[:, :3]
    return wrist, shoulder_z, outer + tolerance, max(0.0, inner - tolerance), tolerance


def outer_impossibility_certificate(wrist, shoulder_z, outer):
    """给出垂直超限或两帧水平圆盘不相交的严格必要条件反证。"""
    vertical = np.abs(wrist[:, 2] - shoulder_z)
    invalid = np.flatnonzero(vertical > outer)
    if len(invalid):
        index = int(invalid[0])
        return {
            "kind": "vertical",
            "frame": index,
            "excess_m": float(vertical[index] - outer),
        }
    radii = np.sqrt(np.maximum(outer**2 - vertical**2, 0))
    for first in range(len(wrist)):
        distance = np.linalg.norm(wrist[first + 1 :, :2] - wrist[first, :2], axis=1)
        gaps = distance - radii[first] - radii[first + 1 :]
        if len(gaps) and np.max(gaps) > 1e-8:
            offset = int(np.argmax(gaps))
            return {
                "kind": "disjoint_outer_disks",
                "frames": [first, first + 1 + offset],
                "gap_m": float(gaps[offset]),
            }
    return None


def independent_ik(model, runtime, tool, target, xy, frames, seed_count, max_nfev):
    """对固定摆放和各关键帧执行独立多初值 IK，并以模型 FK 校验。"""
    lower = np.asarray(runtime.robot.joints.lower_limits, dtype=float)
    upper = np.asarray(runtime.robot.joints.upper_limits, dtype=float)
    home = np.asarray(runtime.config.home_q, dtype=float)
    rng = np.random.default_rng(2026)
    output = []
    for frame in frames:
        target_base = target[frame].copy()
        target_base[:2, 3] -= xy
        for seed_no in range(seed_count):
            seed = home.copy() if seed_no == 0 else rng.uniform(lower, upper)

            def residual(q):
                """以 TCP 平移和旋转误差组成无量纲残差。"""
                model.update_cfg(dict(zip(runtime.joint_names, q, strict=True)))
                actual = model.get_transform("Link7") @ tool
                return np.r_[
                    (actual[:3, 3] - target_base[:3, 3]) / 0.005,
                    Rotation.from_matrix(
                        target_base[:3, :3].T @ actual[:3, :3]
                    ).as_rotvec()
                    / np.deg2rad(5),
                ]

            result = least_squares(
                residual, seed, bounds=(lower, upper), max_nfev=max_nfev
            )
            err = residual(result.x)
            position = float(np.linalg.norm(err[:3]) * 0.005)
            rotation = float(np.linalg.norm(err[3:]) * 5)
            output.append(
                {
                    "frame": int(frame),
                    "base_xy_m": xy.tolist(),
                    "seed_index": seed_no,
                    "seed_qpos": seed.tolist(),
                    "qpos": result.x.tolist(),
                    "position_error_m": position,
                    "rotation_error_deg": rotation,
                    "passed": position <= 0.005 and rotation <= 5.0,
                    "nfev": result.nfev,
                }
            )
            if position <= 0.005 and rotation <= 5.0:
                # 一次独立 FK 验证已足以证明该固定摆放下的此帧可达。
                break
    return output


def main():
    """准备正式轨迹并输出几何必要条件、独立 IK 和逐候选连续求解记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--frames", default="0,91,135,292,375")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.max_nfev < 1:
        parser.error("数值预算必须为正整数")
    replay_args = replay.build_parser().parse_args(
        [
            args.dataset,
            "--robot",
            "rm75b",
            "--episode",
            str(args.episode),
            "--retarget-mode",
            "absolute-tcp",
            "--strict-ik",
            "--headless",
        ]
    )
    selection = replay.resolve_dataset_selection(
        args.dataset, revision=replay_args.revision
    )
    replay_args.repo_id, replay_args.dataset_root = selection.repo_id, selection.root
    deployment = resolve_deployment_calibration("rm75b", identity_fallback=True)
    reference = pose_inv(deployment.pose7)
    if not np.allclose(reference[2], 0, atol=1e-8) or not np.allclose(
        reference[3:], [0, 0, 0, 1], atol=1e-8
    ):
        raise SystemExit("此诊断仅支持 base 高度为零且姿态与桌面相同的配置。")
    states, _, info, _ = replay.load_episode_states(replay_args)
    snapshot = replay._metadata_tcp_snapshot(info)
    selected = replay._metadata_tcp_calibration(info)
    if (
        replay.handumi_metadata(info).get("tracking_workspace") != "table"
        or snapshot is None
        or not replay._is_trusted_tcp_snapshot(snapshot)
        or selected is None
    ):
        raise SystemExit("此诊断需要桌面坐标及可信的原始控制器到 TCP 标定快照。")
    _, _, _, _, tcp_poses, _ = replay.apply_tcp_calibration_to_states(states, *selected)
    targets = pose7_to_mat(tcp_poses).astype(float)
    rollout = None
    failure = None
    failure_attempts = None
    try:
        rollout = prepare_episode(replay_args)
    except SystemExit as exc:
        failure = str(exc)
        failure_attempts = getattr(exc, "attempts", None)
        if failure_attempts is None and failure.startswith(
            "Unable to find strict episode XY placement:"
        ):
            # 负缓存只保存摘要，诊断时重新求解以取得完整候选明细。
            try:
                rollout = replay.solve_episode(replay_args)
                failure = None
            except SystemExit as retry_exc:
                failure_attempts = getattr(retry_exc, "attempts", None)
    runtime = load_embodiment("rm75b")
    tool = load_robot_tcp_calibration("rm75b", runtime.config.arms).matrices["right"]
    wrist, shoulder_z, outer, inner, tolerance = wrist_envelope(
        targets, tool, arm_geometry(runtime.urdf_path)
    )
    certificate = outer_impossibility_certificate(wrist, shoulder_z, outer)

    def max_distance(xy):
        """计算给定底座到所有目标腕心的最大距离。"""
        return float(np.linalg.norm(wrist - np.array([*xy, shoulder_z]), axis=1).max())

    fit = minimize(
        max_distance,
        wrist[:, :2].mean(axis=0),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8},
    )
    final_xy = (
        np.asarray(rollout["table_from_robot_pose7"][0, :2], dtype=float)
        if rollout is not None
        else None
    )
    if rollout is not None and (
        not np.allclose(rollout["table_from_robot_pose7"][0, 2], 0, atol=1e-8)
        or not np.allclose(
            rollout["table_from_robot_pose7"][0, 3:], [0, 0, 0, 1], atol=1e-8
        )
    ):
        raise SystemExit("此诊断仅支持 base 高度为零且姿态与桌面相同的配置。")
    frames = [
        int(x) for x in args.frames.split(",") if x and 0 <= int(x) < len(targets)
    ]
    model = runtime.load_urdf(load_meshes=False)
    cases = []
    placements = [("identity", np.zeros(2)), ("outer_minimax", fit.x)]
    if final_xy is not None:
        placements.append(("continuous_solution", final_xy))
    for name, xy in placements:
        distances = np.linalg.norm(wrist - np.array([*xy, shoulder_z]), axis=1)
        cases.append(
            {
                "name": name,
                "base_xy_m": xy.tolist(),
                "min_wrist_distance_m": float(distances.min()),
                "max_wrist_distance_m": float(distances.max()),
                "outside_outer_frames": np.flatnonzero(distances > outer).tolist(),
                "inside_inner_frames": np.flatnonzero(distances < inner).tolist(),
                "independent_ik": independent_ik(
                    model, runtime, tool, targets, xy, frames, args.seeds, args.max_nfev
                ),
            }
        )
    result = {
        "episode": args.episode,
        "classification": (
            "continuous_strictly_feasible"
            if rollout is not None
            else "geometry_impossible"
            if certificate is not None
            else "bounded_search_inconclusive"
        ),
        "search_failure": failure,
        "search_failure_attempts": failure_attempts,
        "geometry_certificate": certificate,
        "independent_seed_budget": args.seeds,
        "independent_max_nfev": args.max_nfev,
        "geometry_note": "环域是保守必要条件；外半径最小化不保证内半径、姿态或关节可行。数值搜索失败不能证明不可达。",
        "wrist_tolerance_m": tolerance,
        "inner_m": inner,
        "outer_m": outer,
        "outer_minimax_optimiser_success": bool(fit.success),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rollout is not None:
        pos = np.maximum(rollout["left_pos_error_m"], rollout["right_pos_error_m"])
        rot = np.maximum(rollout["left_rot_error_deg"], rollout["right_rot_error_deg"])
        result["continuous"] = {
            "base_xy_m": final_xy.tolist(),
            "max_position_error_m": float(pos.max()),
            "max_rotation_error_deg": float(rot.max()),
            "search": json.loads(str(rollout["base_placement_diagnostics_json"][0])),
        }
        np.savez_compressed(args.output.with_suffix(".npz"), **rollout)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"诊断报告: {args.output}")
    if rollout is not None:
        print(f"连续轨迹: {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
