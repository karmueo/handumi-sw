"""验证真实 LeRobot 写入读取与 RM75B 七轴回放的完整数据边界。"""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.calibration.umi import digest, pose7
from handumi.cameras.base import CameraSample
from handumi.dataset.reader import load_raw_episode
from handumi.grippers.base import GripperObservation
from handumi.scripts.umi_capture import assemble_frame, features, metadata
from handumi.tracking.base import ControllerPairSample
from handumi.tracking.vive import ViveTrackingProvider


def snapshot():
    """生成完全合成的标定快照，测试不使用私人安装参数。"""
    value = {
        "schema": "umi_calibration_v1",
        "serial": "synthetic",
        "camera_device": "synthetic",
        "resolution": [64, 48],
        "fps": 30,
        "fourcc": "MJPG",
        "intrinsics": [50, 50, 32, 24],
        "distortion": [0] * 4,
        "tracker_camera": np.eye(4).tolist(),
        "tracker_tcp": np.eye(4).tolist(),
        "gripper": {
            "min_marker_dist_mm": 40,
            "max_marker_dist_mm": 120,
            "marker_size_mm": 16,
            "dictionary_name": "DICT_4X4_50",
            "left_finger_tag_id": 0,
            "right_finger_tag_id": 1,
        },
        "tracker_clock": "host_query_midpoint",
        "camera_clock": "opencv_host_capture",
        "time": None,
    }
    value["geometry_digest"] = digest({k: v for k, v in value.items() if k != "time"})
    value["time"] = {
        "accepted": True,
        "offset_s": 0,
        "geometry_digest": value["geometry_digest"],
    }
    return value


@pytest.fixture
def capture(tmp_path, table_setup):
    """写入可被项目真实 reader 打开的 32 帧原始数据集。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    profile, _, session, _ = table_setup
    provider = ViveTrackingProvider("synthetic", ControllerTcpCalibration(
        np.array([0, 0, 0, 0, 0, 0, 1]), pose7(profile["tracker_tcp"]),
    ))
    provider.set_workspace_from_device_pose(pose7(session["table_from_device"]))
    root = tmp_path / "capture"
    dataset = LeRobotDataset.create(
        repo_id="local/umi-test",
        root=root,
        fps=30,
        robot_type="handumi_raw",
        features=features(profile),
        use_videos=True,
        vcodec="h264",
        image_writer_threads=2,
    )
    dataset.meta.info["handumi"] = metadata(profile, session)
    for i in range(32):
        stamp = 1_000_000_000 + round(i * 1e9 / 30)
        anchor = np.r_[
            i * 0.0001, 0.0, 0.0, Rotation.from_rotvec([0, 0, i * 0.0002]).as_quat()
        ]
        tracking = replace(
            ControllerPairSample.empty("vive"),
            right_tracked=True,
            right_device_tracked=True,
            right_pose_valid=True,
            connected=True,
            streaming=True,
            clock_synced=True,
            right_controller_pose=anchor.astype(np.float32),
            right_device_controller_pose=anchor.astype(np.float32),
            pc_monotonic_ns=stamp,
            aligned_time_ns=stamp,
            sequence=i + 1,
        )
        tracking = provider._in_workspace(tracking)
        image = CameraSample(np.full((48, 64, 3), i, dtype=np.uint8), stamp, i + 1)
        gripper = GripperObservation(i / 31, True, stamp, i + 1)
        dataset.add_frame(
            assemble_frame(image, tracking, gripper, stamp, stamp + 150_000_000, "test")
        )
    dataset.save_episode()
    dataset.finalize()
    return root


def test_record_reader_quality_roundtrip(capture):
    """源时间、夹爪开度和未知净宽经实际存储后保持语义。"""
    from handumi.dataset.quality import validate_episode

    episode = load_raw_episode(repo_id="local/umi-test", root=capture, episode=0)
    assert episode.states.shape == (32, 16)
    assert np.all(episode.states[:, 14:] == 0)
    np.testing.assert_allclose(
        episode.signals["observation.gripper.right.normalized"],
        np.linspace(0, 1, 32),
        atol=1e-6,
    )
    assert validate_episode(episode.states, fps=30, signals=episode.signals).accepted


def test_rm75_replay_and_conversion(capture, tmp_path):
    """同一片段通过实际 IK 转换后仍保持七轴与独立开度。"""
    from handumi.dataset.canonical import canonicalize_joint_trajectory
    from handumi.dataset.writer import EpisodeResult, write_dataset
    from handumi.robots.registry import load_embodiment
    from handumi.scripts.replay import replay_in_sim as replay
    from handumi.scripts.replay import replay_in_sim_joints as joints

    runtime = load_embodiment("rm75b")
    assert runtime.active_sides == ("right",)
    assert len(runtime.joint_names) == 7
    assert len(runtime.load_urdf(load_meshes=True).scene.geometry) == 8
    args = replay.build_parser().parse_args(
        [str(capture), "--robot", "rm75b", "--retarget-mode", "local-relative"]
    )
    args.repo_id, args.dataset_root = "local/umi-test", capture
    rollout = replay.solve_episode(args)
    assert rollout["qpos"].shape == (32, 7)
    np.testing.assert_allclose(
        rollout["gripper_normalized"][:, 1], np.linspace(0, 1, 32), atol=1e-6
    )
    canonical = canonicalize_joint_trajectory(rollout["qpos"], runtime=runtime)
    output = tmp_path / "converted"
    info = json.loads((capture / "meta/info.json").read_text())
    write_dataset(
        output_root=output,
        source_root=capture,
        source_info=info,
        episodes=[
            EpisodeResult(
                0,
                canonical[:-1],
                canonical[1:],
                "test",
                gripper_normalized=rollout["gripper_normalized"][:-1],
                action_gripper_normalized=rollout["gripper_normalized"][1:],
            )
        ],
        robot_type="rm75b",
        joint_names=list(runtime.joint_names),
        fps=30,
        handumi_metadata={
            "state_layout": "yaml_arm_joints_plus_logical_gripper_width_m",
            "target_robot": {"name": "rm75b"},
        },
    )
    joint_args = joints.build_parser().parse_args([str(output), "--robot", "rm75b"])
    joint_args.repo_id, joint_args.dataset_root = "local/converted", output
    joint_args.dataset_info = json.loads((output / "meta/info.json").read_text())
    joint_args.external_layout = None
    result = joints.build_rollout(joint_args)
    np.testing.assert_allclose(
        result["gripper_normalized"][:, 1], np.linspace(0, 1, 32)[:-1], atol=1e-6
    )


def test_convert_command_uses_source_tool_calibration(capture, tmp_path):
    """真实转换入口读取 UMI 工具标定并保留相邻帧的夹爪开度。"""
    import pandas as pd

    from handumi.scripts import cli

    output = tmp_path / "converted-command"
    cli.main(
        [
            "convert", str(capture), "--robot", "rm75b", "--output", str(output),
            "--allow-flagged-episodes", "--no-solve-cache", "--jobs", "1",
            "--retarget-mode", "local-relative",
        ]
    )

    converted_info = json.loads((output / "meta/info.json").read_text())
    calibration = converted_info["handumi"]["controller_tcp_calibration"]
    assert calibration["source_tool"] == "fastumi"
    assert calibration["tracking_device"] == "vive"
    assert calibration["applied_to_state"] is True
    rows = pd.read_parquet(output / "data/chunk-000/file-000.parquet")
    np.testing.assert_allclose(
        rows["observation.gripper.right.normalized"],
        np.linspace(0, 1, 32)[:-1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rows["action.gripper.right.normalized"],
        np.linspace(0, 1, 32)[1:],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("identity_key", "identity"),
    [("source_tool", "fastumi"), ("source_robot", "piper")],
)
def test_convert_resolves_bound_tool_and_robot_calibrations(identity_key, identity):
    """工具或机器人身份均能保留记录的 TCP 外参及追踪设备。"""
    from handumi.scripts import conversion

    profile = snapshot()
    profile["tracker_tcp"][0][3] = 0.125
    profile["geometry_digest"] = digest(
        {k: v for k, v in profile.items() if k not in ("time", "geometry_digest")}
    )
    profile["time"]["geometry_digest"] = profile["geometry_digest"]
    source_info = {"handumi": metadata(profile)}
    source_calibration = source_info["handumi"]["controller_tcp_calibration"]
    source_calibration.pop("source_tool")
    source_calibration[identity_key] = identity
    args = conversion.build_parser().parse_args(["input", "--robot", "rm75b"])

    selection = conversion._resolve_conversion_tcp_calibration(args, source_info)

    assert f"{identity}/vive" in selection.source
    assert selection.metadata[identity_key] == identity
    assert selection.metadata["applied_to_state"] is True
    assert source_calibration["applied_to_state"] is False
    assert args.controller_device == "vive"
    np.testing.assert_allclose(selection.calibration.right[:3], [0.125, 0, 0])


def test_table_record_preserves_raw_transform_and_auto_mode(capture):
    """真实存储保留设备原始位姿及桌面变换，自动回放识别桌面语义。"""
    import pandas as pd

    from handumi.robots.utils import pose7_to_mat
    from handumi.scripts.replay import replay_in_sim as replay

    info = json.loads((capture / "meta/info.json").read_text())
    assert info["handumi"]["tracking_workspace"] == "table"
    assert info["handumi"]["controller_tcp_calibration"]["applied_to_state"] is False
    rows = pd.read_parquet(capture / "data/chunk-000/file-000.parquet")
    first = rows.iloc[0]
    raw = first["observation.tracking.right_device_controller_pose"]
    workspace = first["observation.tracking.workspace_from_device_pose"]
    state = first["observation.state"]
    np.testing.assert_allclose(raw, [0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(pose7_to_mat(state[7:14]), pose7_to_mat(workspace) @ pose7_to_mat(raw), atol=1e-6)
    args = replay.build_parser().parse_args([str(capture), "--robot", "rm75b"])
    assert replay._resolved_retarget_mode(args, info) == "absolute-table"
    args.retarget_mode = "local-relative"
    assert replay._resolved_retarget_mode(args, info) == "local-relative"
