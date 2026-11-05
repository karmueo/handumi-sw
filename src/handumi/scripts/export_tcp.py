"""将 Vive 原始示教导出为回合起点坐标系下的 LeRobot TCP 数据集。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from handumi.dataset.reader import load_raw_episode
from handumi.dataset.selection import resolve_dataset_selection
from handumi.dataset.writer import load_info


def relative_tcp(poses: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """将桌面系 TCP 及夹爪开度转换为回合首帧坐标系中的七维数组。"""
    poses = np.asarray(poses, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) != len(gripper):
        raise ValueError("TCP 或夹爪数据形状错误")
    if not len(poses) or not np.isfinite(poses).all() or not np.isfinite(gripper).all():
        raise ValueError("TCP 或夹爪数据为空或包含非有限值")
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    if np.any(np.abs(norms - 1) > 1e-3) or np.any((gripper < 0) | (gripper > 1)):
        raise ValueError("TCP 四元数或归一化夹爪数值无效")
    rotations = Rotation.from_quat(poses[:, 3:])
    initial = rotations[0].inv()
    positions = initial.apply(poses[:, :3] - poses[0, :3])
    angles = (initial * rotations).as_rotvec()
    return np.column_stack((positions, angles, gripper)).astype(np.float32)


def letterbox(image: np.ndarray) -> np.ndarray:
    """将 RGB 图像等比缩放并居中补黑至 224×224。"""
    height, width = image.shape[:2]
    scale = min(224 / width, 224 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    output = np.zeros((224, 224, 3), dtype=np.uint8)
    x, y = (224 - size[0]) // 2, (224 - size[1]) // 2
    output[y : y + size[1], x : x + size[0]] = resized
    return output


def quality_selection(
    root: Path, repo_id: str, episodes: list[dict]
) -> tuple[list[int], list[dict]]:
    """检查录制质量报告覆盖全部原始回合并返回保留和拒绝的编号。"""
    path = root / "meta" / "handumi_quality.json"
    if not path.is_file():
        raise ValueError("缺少录制质量报告，请先运行 handumi validate")
    try:
        report = json.loads(path.read_text())
        rows = report["episodes"]
        by_index = {int(row["episode_index"]): row for row in rows}
        for row in rows:
            int(row["frame_count"])
            if row["status"] not in ("accepted", "rejected"):
                raise ValueError("invalid status")
            if row["status"] == "rejected" and not isinstance(row["findings"], list):
                raise ValueError("invalid findings")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "录制质量报告损坏或缺字段，请重新运行 handumi validate"
        ) from exc
    if report.get("dataset") != repo_id or len(rows) != len(episodes):
        raise ValueError(
            "质量报告的数据集或回合数量不匹配，请重新运行 handumi validate"
        )
    if set(by_index) != set(range(len(episodes))) or len(by_index) != len(rows):
        raise ValueError("质量报告回合编号不完整，请重新运行 handumi validate")
    kept, rejected = [], []
    for episode in episodes:
        index = int(episode["episode_index"])
        row = by_index[index]
        if int(row["frame_count"]) != int(episode["length"]):
            raise ValueError(
                f"回合 {index} 的质量报告帧数不匹配，请重新运行 handumi validate"
            )
        if row["status"] == "accepted":
            kept.append(index)
        elif row["status"] == "rejected":
            rejected.append(
                {"source_episode_index": index, "findings": row["findings"]}
            )
        else:
            raise ValueError(f"回合 {index} 的质量状态无效")
    if not kept:
        raise ValueError("所有回合均被质量检查拒绝，请处理报告中的问题并重新检查")
    return kept, rejected


def write_episode(
    dataset, source: Path, info: dict, row: dict, repo_id: str, index: int
) -> None:
    """读取一个回合的同步位姿和视频，向输出写入同帧示教数据。"""
    raw = load_raw_episode(root=source, repo_id=repo_id, episode=index)
    state = relative_tcp(
        raw.signals["observation.tracking.right_tcp_pose"],
        raw.signals["observation.gripper.right.normalized"],
    )
    if len(state) != int(row["length"]):
        raise ValueError("TCP 帧数与回合元数据不一致")
    video_key = "observation.images.right_wrist"
    prefix = f"videos/{video_key}/"
    video = source / info["video_path"].format(
        video_key=video_key,
        chunk_index=row[prefix + "chunk_index"],
        file_index=row[prefix + "file_index"],
    )
    start = round(float(row[prefix + "from_timestamp"]) * 30)
    end = round(float(row[prefix + "to_timestamp"]) * 30)
    if start < 0 or end - start != len(state):
        raise ValueError("视频时间区间与回合帧数不匹配")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"无法打开视频 {video}")
    try:
        if end > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError("视频帧数不足")
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError("无法定位视频回合起点")
        for value in state:
            ok, bgr = cap.read()
            if not ok:
                raise ValueError("视频帧数不足")
            rgb = cv2.cvtColor(letterbox(bgr), cv2.COLOR_BGR2RGB)
            dataset.add_frame(
                {
                    "observation.images.front": rgb,
                    "observation.state": value,
                    "action": value.copy(),
                    "task": str(row["tasks"][0]),
                }
            )
    finally:
        cap.release()
    dataset.save_episode()


def export(source: Path, output: Path, repo_id: str) -> None:
    """导出数据到临时目录，完成完整性校验后发布到目标目录。"""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from handumi.dataset.curation import validate_dataset_integrity

    selection = resolve_dataset_selection(source)
    info = load_info(source)
    meta = info.get("handumi", {})
    if (
        meta.get("recording_device") != "vive"
        or meta.get("active_sides") != ["right"]
        or meta.get("capture_schema") != "synchronized_sources_v2"
        or meta.get("tcp_pose", {}).get("frame") != "table"
        or meta.get("tcp_pose", {}).get("feature")
        != "observation.tracking.right_tcp_pose"
        or int(info.get("fps", 0)) != 30
    ):
        raise ValueError("需要 30 FPS、单右侧、桌面系 TCP 的 Vive 录制")
    episode_files = sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    episodes = sorted(
        pq.read_table(episode_files).to_pylist(), key=lambda row: row["episode_index"]
    )
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("回合元数据数量不匹配")
    if [int(row["episode_index"]) for row in episodes] != list(range(len(episodes))):
        raise ValueError("原始回合编号不连续或重复")
    kept, rejected = quality_selection(source, selection.repo_id, episodes)
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise FileExistsError(f"输出路径已存在: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    build = temp / "dataset"
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    }
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id,
            fps=30,
            features=features,
            root=build,
            robot_type="umi",
            image_writer_threads=4,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast", crf=23),
        )
        mapping = []
        for new_index, index in enumerate(kept):
            try:
                write_episode(
                    dataset, source, info, episodes[index], selection.repo_id, index
                )
                mapping.append(
                    {"episode_index": new_index, "source_episode_index": index}
                )
                print(f"已导出 {new_index + 1}/{len(kept)}，源回合 {index}", flush=True)
            except Exception as exc:
                raise RuntimeError(f"导出源回合 {index} 失败: {exc}") from exc
        dataset.finalize()
        contract = {
            "pose_frame": "episode_start_tcp",
            "end_frame": "tcp",
            "tcp_definition": "gripper_grasp_center",
            "state_layout": "xyz_rotvec_gripper",
            "action_layout": "xyz_rotvec_gripper",
            "action_source": "demonstration_trajectory",
            "state_time_reference": "image_timestamp",
            "action_time_reference": "image_timestamp",
            "position_unit": "m",
            "rotation_unit": "rad",
            "gripper_representation": "normalized_0_1",
            "image_size": [224, 224],
            "image_preprocessing": "letterbox_inter_area_black_border",
            "source_dataset": source.name,
            "episodes": mapping,
            "rejected": rejected,
        }
        (build / "meta" / "fastumi.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
        )
        validate_dataset_integrity(build, repo_id=repo_id)
        if output.exists():
            raise FileExistsError(f"输出路径已存在: {output}")
        os.rename(build, output)
    finally:
        try:
            if dataset is not None and not dataset._is_finalized:
                dataset.finalize()
        finally:
            shutil.rmtree(temp)


def main() -> None:
    """解析命令行参数并导出数据集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    export(args.dataset, args.output, args.repo_id)


if __name__ == "__main__":
    main()
