"""Robot registry backed only by ``configs/robots/*.yaml``."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yaml
import yourdfpy

from handumi.robots.kinematics import BimanualKinematicsSolver, KinematicsConfig

if TYPE_CHECKING:
    from handumi.sim.viser_sim import ViserSim

SOURCE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = (
    SOURCE_ROOT if (SOURCE_ROOT / "configs" / "robots").exists() else PACKAGE_ROOT
)
REPO_ROOT = RESOURCE_ROOT  # Backward-compatible name for callers/tests.
CONFIG_DIR = RESOURCE_ROOT / "configs" / "robots"
SIDES: tuple[str, str] = ("left", "right")


def robot_config_metadata(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Identify one robot profile: name, path, content hash and the parsed YAML.

    Datasets record this so a reader knows exactly which embodiment
    definition produced them. A capture stores the HandUMI tool's robot; a
    conversion must store the robot it converted *to*, which is why the
    helper lives here and not in the recorder.
    """
    path = Path(config_dir) / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in Path(config_dir).glob("*.yaml")))
        raise SystemExit(
            f"Unknown robot {name!r}; expected {path}. Available: {available or 'none'}."
        )
    raw = path.read_bytes()
    return {
        "name": name,
        "config_path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "configuration": yaml.safe_load(raw) or {},
    }


def available_robot_names() -> tuple[str, ...]:
    """Return robot names discovered from ``configs/robots/*.yaml``."""

    if not CONFIG_DIR.exists():
        return ()
    return tuple(sorted(path.stem for path in CONFIG_DIR.glob("*.yaml")))


EMBODIMENT_NAMES: tuple[str, ...] = available_robot_names()


@dataclass(frozen=True)
class GripperJointConfig:
    """One robot joint driven by a normalized HandUMI gripper opening."""

    name: str
    closed_value: float = 0.0
    open_value: float | None = None


@dataclass(frozen=True)
class GripperJointRuntime:
    """Resolved gripper joint mapping for normalized HandUMI openings."""

    index: int
    closed_value: float
    open_value: float


@dataclass(frozen=True)
class RobotArmConfig:
    """YAML-declared logical arm mapping."""

    ee_link: str
    joint_names: tuple[str, ...] = ()
    gripper_joints: tuple[GripperJointConfig, ...] = ()


@dataclass(frozen=True)
class RobotRealConfig:
    """Robot defaults for real-hardware teleop.

    Machine-local connection details (CAN ports, camera IDs, Feetech ports)
    stay in ``configs/rig.yaml``; these values describe how this robot should
    be commanded once the local rig has supplied the transport.
    """

    backend: str | None = None
    command_rate_hz: float = 100.0
    max_joint_speed_deg_s: float = 180.0
    max_joint_acceleration_deg_s2: float = 720.0
    home_max_joint_speed_deg_s: float = 20.0
    home_timeout_s: float = 30.0
    home_tolerance_deg: float = 3.0
    startup_speed_percent: int = 10
    speed_percent: int = 80
    gripper_effort: int = 1000


@dataclass(frozen=True)
class RobotConfig:
    kind: str
    urdf: Path
    pkg_root: Path
    mjcf: Path | None
    mjcf_joint_map: dict[str, str]
    mjcf_joint_prefix_map: dict[str, str]
    arms: dict[str, RobotArmConfig]
    ee_links: dict[str, str]
    home_q: np.ndarray
    home_poses: dict[str, np.ndarray]
    default_home_pose: str
    # Nominal working posture the IK posture cost pulls toward (radians, full
    # actuated vector; NaN = no preference for that joint). Defaults to home_q;
    # only matters with ik_weights.posture > 0.
    posture_q: np.ndarray
    ik_weights: KinematicsConfig
    replay_max_joint_delta: float | None
    # Offline replay-only IK weight overrides ("pos"/"ori"/"rest"). Teleop
    # profiles are tuned for a human in the loop and may hold the arm near
    # home harder than following a recorded trajectory allows.
    replay_ik_weights: dict[str, float]
    replay_gripper_mode: str
    gripper_max_width_m: float
    controller_tcp_calibrations: dict[str, Path]
    handumi_gripper: str | None
    handumi_controller_mount: str | None
    real: RobotRealConfig
    real_options: dict[str, Any]
    manipulation_lock_joints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmRuntime:
    """Validated arm metadata resolved against the loaded kinematic model."""

    side: str
    ee_link: str
    ee_index: int
    joint_names: tuple[str, ...]
    joint_indices: tuple[int, ...]


@dataclass(frozen=True)
class RobotRuntime:
    """Resolved robot config plus constructors used by scripts."""

    name: str
    config: RobotConfig
    urdf_path: Path
    robot: pk.Robot
    arms: dict[str, ArmRuntime]
    solver_cls: type
    config_cls: type = KinematicsConfig
    command_size: int = 0
    default_port: int = 8003
    default_axis_map: str = "x,z,y"
    default_compare_axis_maps: tuple[str, ...] = ("x,z,y",)
    default_workspace: str = "rest"
    wrist_forward: float = 0.34
    wrist_height: float = 0.24
    wrist_lateral: float = 0.23
    # Per side: resolved finger joints. The joint value for a given HandUMI
    # opening is interpolated from ``closed_value`` to ``open_value``.
    finger_joints: dict[str, tuple[GripperJointRuntime, ...]] = None  # type: ignore[assignment]
    manipulation_lock_indices: tuple[int, ...] = ()

    @property
    def ee_indices(self) -> tuple[int, int]:
        # The legacy pose-pair interface uses the fixed root as a placeholder
        # for an absent side. It is never an IK target or an output joint.
        return (
            self.arms["left"].ee_index if "left" in self.arms else 0,
            self.arms["right"].ee_index if "right" in self.arms else 0,
        )

    @property
    def active_sides(self) -> tuple[str, ...]:
        return tuple(side for side in SIDES if side in self.arms)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.robot.joints.actuated_names)

    def arm_joint_names(self, side: str) -> list[str]:
        return list(self.arms[side].joint_names) if side in self.arms else []

    def home_q(self, name: str | None = None) -> np.ndarray:
        """Return a copy of a named safe starting pose."""
        pose_name = name or self.config.default_home_pose
        try:
            return self.config.home_poses[pose_name].astype(np.float32).copy()
        except KeyError as exc:
            available = ", ".join(sorted(self.config.home_poses))
            raise ValueError(
                f"Unknown home pose {pose_name!r} for {self.name}; use {available}."
            ) from exc

    def arm_joint_indices(self, side: str) -> list[int]:
        return list(self.arms[side].joint_indices) if side in self.arms else []

    def set_finger_positions(
        self, q: np.ndarray, normalized: Mapping[str, float]
    ) -> np.ndarray:
        """Write the gripper-finger joint values for a 0-1 opening per side
        into ``q`` (in place) and return it."""
        for side, fingers in (self.finger_joints or {}).items():
            fraction = float(np.clip(normalized.get(side, 0.0), 0.0, 1.0))
            for finger in fingers:
                q[finger.index] = finger.closed_value + (
                    fraction * (finger.open_value - finger.closed_value)
                )
        return q

    def urdf_arm_joint_names(self, *, is_left: bool) -> list[str]:
        """Compatibility accessor for older callers."""
        side = "left" if is_left else "right"
        return self.arm_joint_names(side)

    def command_to_arm_q(self, command: np.ndarray) -> np.ndarray:
        names = self.arm_joint_names("left")
        return np.asarray(command[: len(names)], dtype=float)

    def mjcf_actuator_name(self, urdf_joint_name: str) -> str:
        """Map a URDF joint name to the configured MJCF actuator/joint name."""

        exact = self.config.mjcf_joint_map.get(urdf_joint_name)
        if exact is not None:
            return exact
        for source_prefix, target_prefix in self.config.mjcf_joint_prefix_map.items():
            if urdf_joint_name.startswith(source_prefix):
                return target_prefix + urdf_joint_name[len(source_prefix) :]
        return urdf_joint_name

    def load_urdf(self, *, load_meshes: bool = False) -> yourdfpy.URDF:
        return yourdfpy.URDF.load(
            str(self.urdf_path),
            filename_handler=yourdfpy_handler(self.config.pkg_root),
            mesh_dir=str(self.urdf_path.parent),
            load_meshes=load_meshes,
        )

    def make_sim(
        self,
        *,
        port: int | None = None,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        scene_bodies: list | None = None,
    ) -> ViserSim:
        from handumi.sim.viser_sim import ViserSim

        return ViserSim(
            urdf_path=self.urdf_path,
            filename_handler=yourdfpy_handler(self.config.pkg_root),
            left_joint_names=self.arm_joint_names("left"),
            right_joint_names=self.arm_joint_names("right"),
            command_size=self.command_size,
            arm_q_fn=lambda command: np.asarray(command, dtype=float),
            joint_names=joint_names or list(self.joint_names),
            default_q=default_q,
            scene_bodies=scene_bodies,
            port=self.default_port if port is None else port,
        )

    def make_physics(self, *, scene_config=None):
        del scene_config


def yourdfpy_handler(pkg_root: str | Path):
    """Resolve package and relative mesh paths from a configured asset root."""

    root = Path(pkg_root)

    def h(fname: str) -> str:
        if fname.startswith("package://"):
            rest = fname.split("package://", 1)[1]
            direct = root / rest
            if direct.exists():
                return str(direct)
            parts = Path(rest).parts
            if len(parts) >= 2:
                fallback = root / Path(*parts[1:])
                if fallback.exists():
                    return str(fallback)
            return str(direct)
        path = Path(fname)
        if not path.is_absolute():
            relative = root / path
            if relative.exists():
                return str(relative)
        return fname

    return h


def load_robot_config(name: str) -> RobotConfig:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(
            f"Unsupported robot {name!r}. Expected one of {available_robot_names()}."
        )
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    weights = data.get("ik_weights") or {}
    replay = data.get("replay") or {}
    replay_gripper_mode = str(replay.get("gripper_retarget", "normalized"))
    if replay_gripper_mode not in {"normalized", "physical-width"}:
        raise ValueError(
            "replay.gripper_retarget must be 'normalized' or 'physical-width'."
        )
    replay_weights_raw = replay.get("ik_weights") or {}
    if not isinstance(replay_weights_raw, dict):
        raise TypeError("replay.ik_weights must be a mapping.")
    unknown = set(replay_weights_raw) - {"pos", "ori", "rest", "posture", "limit_margin"}
    if unknown:
        raise ValueError(
            "replay.ik_weights only overrides pos/ori/rest/posture/limit_margin, "
            f"got {sorted(unknown)}."
        )
    replay_ik_weights = {
        key: float(value) for key, value in replay_weights_raw.items()
    }
    real = data.get("real") or {}
    urdf = _resolve_path(data["urdf"])
    pkg_root = _resolve_path(data["pkg_root"])
    mjcf = _resolve_path(data["mjcf"]) if data.get("mjcf") else None
    home_q = np.asarray(data.get("home_q") or [], dtype=np.float32)
    default_home_pose = str(data.get("default_home_pose") or "home_q")
    raw_home_poses = data.get("home_poses") or {}
    home_poses = {
        str(pose_name): np.asarray(values, dtype=np.float32)
        for pose_name, values in raw_home_poses.items()
    }
    if not home_poses:
        home_poses[default_home_pose] = home_q
    elif default_home_pose not in home_poses:
        raise ValueError(
            f"default_home_pose {default_home_pose!r} is not present in home_poses."
        )
    home_q = home_poses[default_home_pose]
    # null entries mean "no preference": the posture cost skips that joint
    # and the start-pose seed takes it from home_q.
    posture_q = np.asarray(
        [np.nan if value is None else float(value) for value in (data.get("posture_q") or [])],
        dtype=np.float32,
    )
    if posture_q.size == 0:
        posture_q = home_q
    arms = _parse_arms(data)
    controller_tcp_calibrations = {
        str(device): _resolve_path(value)
        for device, value in (data.get("controller_tcp_calibrations") or {}).items()
    }
    handumi_tool = data.get("handumi_tool") or {}
    manipulation = data.get("manipulation") or {}
    manipulation_lock_joints = tuple(
        str(name) for name in (manipulation.get("lock_joints") or ())
    )
    return RobotConfig(
        kind=str(data.get("kind") or name),
        urdf=urdf,
        pkg_root=pkg_root,
        mjcf=mjcf,
        mjcf_joint_map={
            str(key): str(value)
            for key, value in (data.get("mjcf_joint_map") or {}).items()
        },
        mjcf_joint_prefix_map={
            str(key): str(value)
            for key, value in (data.get("mjcf_joint_prefix_map") or {}).items()
        },
        arms=arms,
        ee_links={side: arm.ee_link for side, arm in arms.items()},
        home_q=home_q,
        home_poses=home_poses,
        default_home_pose=default_home_pose,
        posture_q=posture_q,
        gripper_max_width_m=float(data.get("gripper_max_width_m", 0.08)),
        controller_tcp_calibrations=controller_tcp_calibrations,
        handumi_gripper=(
            str(handumi_tool["gripper"]) if handumi_tool.get("gripper") else None
        ),
        handumi_controller_mount=(
            str(handumi_tool["controller_mount"])
            if handumi_tool.get("controller_mount")
            else None
        ),
        ik_weights=KinematicsConfig(
            pos_weight=float(weights.get("pos", 100.0)),
            ori_weight=float(weights.get("ori", 15.0)),
            rest_weight=float(weights.get("rest", 2.0)),
            posture_weight=float(weights.get("posture", 0.0)),
            limit_margin_weight=float(weights.get("limit_margin", 0.0)),
            limit_margin_rad=float(weights.get("limit_margin_rad", 0.17453292)),
            manipulability_weight=float(weights.get("manipulability", 0.0)),
            max_joint_delta=(
                None
                if weights.get("max_joint_delta") is None
                else float(weights["max_joint_delta"])
            ),
            self_collision_weight=float(weights.get("self_collision", 0.0)),
            self_collision_margin=float(weights.get("self_collision_margin", 0.01)),
            self_collision_pairs=str(weights.get("self_collision_pairs", "all")),
            self_collision_always=bool(weights.get("self_collision_always", False)),
            collision_activation_distance=float(
                weights.get("collision_activation_distance", 0.10)
            ),
            max_reach=(
                None
                if weights.get("max_reach") is None
                else float(weights["max_reach"])
            ),
        ),
        replay_max_joint_delta=(
            None
            if replay.get("max_joint_delta") is None
            else float(replay["max_joint_delta"])
        ),
        replay_ik_weights=replay_ik_weights,
        replay_gripper_mode=replay_gripper_mode,
        real=RobotRealConfig(
            backend=(None if real.get("backend") is None else str(real["backend"])),
            command_rate_hz=float(real.get("command_rate_hz", 100.0)),
            max_joint_speed_deg_s=float(real.get("max_joint_speed_deg_s", 180.0)),
            max_joint_acceleration_deg_s2=float(
                real.get("max_joint_acceleration_deg_s2", 720.0)
            ),
            home_max_joint_speed_deg_s=float(
                real.get("home_max_joint_speed_deg_s", 20.0)
            ),
            home_timeout_s=float(real.get("home_timeout_s", 30.0)),
            home_tolerance_deg=float(real.get("home_tolerance_deg", 3.0)),
            startup_speed_percent=int(real.get("startup_speed_percent", 10)),
            speed_percent=int(real.get("speed_percent", 80)),
            gripper_effort=int(real.get("gripper_effort", 1000)),
        ),
        real_options={str(key): value for key, value in real.items()},
        manipulation_lock_joints=manipulation_lock_joints,
    )


def build_pruned_collision_model(
    *,
    urdf,
    robot,
    home_q: np.ndarray,
    arms: Mapping[str, Any],
    gripper_joints: Mapping[str, tuple[str, ...]],
    margin: float,
    pairs_mode: str = "all",
):
    """Capsulize the URDF and drop pairs that can never signal a real fault.

    Capsulized links overlap their structural neighbours (the fit is
    deliberately coarse), so pairs already violating ``margin`` at the rest
    pose are ignored -- the same idea as MoveIt's generated disable list --
    leaving only genuinely avoidable contacts such as arm-vs-arm.

    Shared by the solver (which uses it for the opt-in collision cost) and by
    offline auditing, which needs the same pruning for embodiments that never
    enable that cost.
    """
    if pairs_mode not in {"all", "inter-arm"}:
        raise ValueError("self_collision_pairs must be 'all' or 'inter-arm'.")
    candidate = pk.collision.RobotCollision.from_urdf(urdf)
    distances = np.asarray(
        candidate.compute_self_collision_distance(robot, jnp.asarray(home_q))
    )
    idx_i = np.asarray(candidate.active_idx_i)
    idx_j = np.asarray(candidate.active_idx_j)
    ignore = {
        (candidate.link_names[idx_i[k]], candidate.link_names[idx_j[k]])
        for k in np.flatnonzero(distances < margin)
    }
    # Pairs inside one gripper assembly are never meaningful: fingers must be
    # free to close on each other (that is grasping, not a fault) and the TCP
    # is a virtual grasp-point link between them.
    joint_children = {j.name: j.child for j in urdf.robot.joints}
    for side, arm in arms.items():
        gripper_names = set(gripper_joints.get(side, ()))
        assembly = {arm.ee_link}
        assembly.update(
            joint_children[name] for name in gripper_names if name in joint_children
        )
        # Mimic followers (e.g. the mirrored second finger) belong to the same
        # assembly even though only the driven joint is declared.
        assembly.update(
            j.child
            for j in urdf.robot.joints
            if j.mimic is not None and j.mimic.joint in gripper_names
        )
        for k in range(len(idx_i)):
            a = candidate.link_names[idx_i[k]]
            b = candidate.link_names[idx_j[k]]
            if a in assembly and b in assembly:
                ignore.add((a, b))
    if pairs_mode == "inter-arm":
        # Keep only left_*-vs-right_* pairs: intra-arm interference is already
        # prevented by the vendor's joint limits, and each dropped pair removes
        # a residual (plus its FK jacobian) from every solve.
        def _side(link_name: str) -> str | None:
            for side in SIDES:
                if link_name.startswith(f"{side}_"):
                    return side
            return None

        for k in range(len(idx_i)):
            a = candidate.link_names[idx_i[k]]
            b = candidate.link_names[idx_j[k]]
            if _side(a) is None or _side(a) == _side(b):
                ignore.add((a, b))
    if not ignore:
        return candidate
    return pk.collision.RobotCollision.from_urdf(
        urdf, user_ignore_pairs=tuple(sorted(ignore))
    )


def load_embodiment(name: str, *, collision_kind: str = "full") -> RobotRuntime:
    """加载机器人模型；回放可选用固定夹爪盒碰撞模型。"""
    if collision_kind not in {"full", "gripper-box"} or (collision_kind == "gripper-box" and name != "rm75b"):
        raise ValueError("Unsupported collision model selection")
    cfg = load_robot_config(name)
    urdf = yourdfpy.URDF.load(
        str(cfg.urdf),
        filename_handler=yourdfpy_handler(cfg.pkg_root),
        mesh_dir=str(cfg.urdf.parent),
        load_meshes=False,
    )
    robot = pk.Robot.from_urdf(urdf)
    arms = _resolve_arms(name, cfg, robot)
    ee_indices = (
        arms["left"].ee_index if "left" in arms else 0,
        arms["right"].ee_index if "right" in arms else 0,
    )
    arm_joint_indices = {
        side: list(arms[side].joint_indices) if side in arms else [] for side in SIDES
    }
    locked_joint_indices = _resolve_lock_joint_indices(
        name, cfg.manipulation_lock_joints, robot
    )
    home_q = cfg.home_q
    if home_q.size == 0:
        home_q = np.zeros(robot.joints.num_actuated_joints, dtype=np.float32)
    if len(home_q) != robot.joints.num_actuated_joints:
        raise ValueError(
            f"{name} home_q has {len(home_q)} values, expected "
            f"{robot.joints.num_actuated_joints}."
        )
    posture_q = cfg.posture_q if cfg.posture_q.size else home_q
    if len(posture_q) != robot.joints.num_actuated_joints:
        raise ValueError(
            f"{name} posture_q has {len(posture_q)} values, expected "
            f"{robot.joints.num_actuated_joints}."
        )
    for pose_name, pose_q in cfg.home_poses.items():
        if len(pose_q) != robot.joints.num_actuated_joints:
            raise ValueError(
                f"{name} home pose {pose_name!r} has {len(pose_q)} values, expected "
                f"{robot.joints.num_actuated_joints}."
            )

    robot_collision = None
    if cfg.ik_weights.collision_enabled:
        if name == "rm75b":
            from handumi.robots.self_collision import build_rm75b_collision_model

            if collision_kind == "gripper-box":
                from handumi.robots.self_collision import build_rm75b_gripper_box_model
                robot_collision = build_rm75b_gripper_box_model(urdf)
            else:
                robot_collision = build_rm75b_collision_model(urdf)
        else:
            robot_collision = build_pruned_collision_model(
                urdf=urdf,
                robot=robot,
                home_q=home_q,
                arms=arms,
                gripper_joints={
                    side: tuple(g.name for g in cfg.arms[side].gripper_joints)
                    for side in arms
                },
                margin=cfg.ik_weights.self_collision_margin,
                pairs_mode=cfg.ik_weights.self_collision_pairs,
            )
        if (
            cfg.ik_weights.self_collision_weight > 0.0
            and len(np.asarray(robot_collision.active_idx_i)) == 0
        ):
            # A self-collision cost with no pairs would silently do nothing.
            # inter-arm mode requires the repo's left_/right_ link prefixes;
            # e.g. openarm's `openarm_left_*` names do not match it.
            raise ValueError(
                f"{name}: self_collision is enabled but no active capsule pairs "
                "remain after pruning. With self_collision_pairs: inter-arm the "
                "URDF links must be prefixed left_/right_; otherwise use 'all'."
            )

    class _Solver(BimanualKinematicsSolver):
        def __init__(
            self,
            config: KinematicsConfig | None = None,
            locked_joint_indices: tuple[int, ...] | None = None,
        ) -> None:
            resolved = config or cfg.ik_weights
            super().__init__(
                robot=robot,
                ee_indices=ee_indices,
                arm_joint_indices=arm_joint_indices,
                home_q=home_q,
                posture_q=posture_q,
                config=resolved,
                locked_joint_indices=(
                    locked_joint_indices
                    if locked_joint_indices is not None
                    else ()
                ),
                robot_collision=(
                    robot_collision if resolved.collision_enabled else None
                ),
            )

    command_size = max(len(arm.joint_names) for arm in arms.values())
    finger_joints = _resolve_finger_joints(urdf, robot, cfg, arms)
    return RobotRuntime(
        name=name,
        config=cfg,
        urdf_path=cfg.urdf,
        robot=robot,
        arms=arms,
        solver_cls=_Solver,
        command_size=command_size,
        default_port=8002 if name == "axol" else 8003,
        finger_joints=finger_joints,
        manipulation_lock_indices=locked_joint_indices,
    )


def resolve_home_q(
    runtime: RobotRuntime,
    *,
    rig_config: Path | None = None,
    explicit_name: str | None = None,
) -> tuple[str, np.ndarray]:
    """Resolve CLI, machine-local, then embodiment default home selection."""
    name = explicit_name
    if name is None and rig_config is not None and rig_config.exists():
        with rig_config.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        local = ((data.get("robots") or {}).get(runtime.name) or {}).get("home_pose")
        if local:
            name = str(local)
    name = name or runtime.config.default_home_pose
    return name, runtime.home_q(name)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else RESOURCE_ROOT / path


def _parse_arms(data: dict[str, Any]) -> dict[str, RobotArmConfig]:
    arms_data = data.get("arms")
    if arms_data is None:
        legacy_ee_links = data.get("ee_links")
        if not isinstance(legacy_ee_links, dict):
            raise ValueError("Robot config must define arms or legacy ee_links.")
        arms_data = {side: {"ee_link": legacy_ee_links[side]} for side in SIDES}
    if not isinstance(arms_data, dict):
        raise TypeError("arms must be a mapping.")
    if not arms_data or set(arms_data) - set(SIDES):
        raise ValueError("arms must declare left, right, or both.")

    arms: dict[str, RobotArmConfig] = {}
    for side in SIDES:
        if side not in arms_data:
            continue
        raw_arm = arms_data.get(side)
        if not isinstance(raw_arm, dict):
            raise TypeError(f"arms.{side} must be a mapping.")
        ee_link = raw_arm.get("ee_link")
        if not ee_link:
            raise ValueError(f"arms.{side}.ee_link is required.")
        joint_names = tuple(str(name) for name in (raw_arm.get("joint_names") or ()))
        arms[side] = RobotArmConfig(
            ee_link=str(ee_link),
            joint_names=joint_names,
            gripper_joints=_parse_gripper_joints(raw_arm.get("gripper_joints")),
        )
    return arms


def _parse_gripper_joints(value: Any) -> tuple[GripperJointConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("gripper_joints must be a list.")
    joints: list[GripperJointConfig] = []
    for item in value:
        if isinstance(item, str):
            joints.append(GripperJointConfig(name=item))
            continue
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("Each gripper_joints entry must be a name or mapping.")
        open_value = item.get("open")
        closed_value = item.get("closed", 0.0)
        joints.append(
            GripperJointConfig(
                name=str(item["name"]),
                closed_value=float(closed_value),
                open_value=None if open_value is None else float(open_value),
            )
        )
    return tuple(joints)


def _resolve_lock_joint_indices(
    name: str,
    joint_names: tuple[str, ...],
    robot: pk.Robot,
) -> tuple[int, ...]:
    if not joint_names:
        return ()
    actuated_names = list(robot.joints.actuated_names)
    missing = [joint for joint in joint_names if joint not in actuated_names]
    if missing:
        raise ValueError(
            f"{name}: manipulation.lock_joints not in URDF actuated joints: "
            f"{missing}"
        )
    return tuple(actuated_names.index(joint) for joint in joint_names)


def _resolve_arms(
    name: str, cfg: RobotConfig, robot: pk.Robot
) -> dict[str, ArmRuntime]:
    actuated_names = list(robot.joints.actuated_names)
    link_names = list(robot.links.names)
    arms: dict[str, ArmRuntime] = {}
    for side, arm in cfg.arms.items():
        joint_names = arm.joint_names or tuple(
            joint_name
            for joint_name in actuated_names
            if joint_name.startswith(f"{side}_")
        )
        if not joint_names:
            raise ValueError(
                f"{name}: arms.{side}.joint_names is required because no "
                f"actuated joints start with {side!r}."
            )
        missing_joints = [joint for joint in joint_names if joint not in actuated_names]
        if missing_joints:
            raise ValueError(
                f"{name}: arms.{side}.joint_names not in URDF actuated joints: "
                f"{missing_joints}"
            )
        if arm.ee_link not in link_names:
            raise ValueError(
                f"{name}: arms.{side}.ee_link {arm.ee_link!r} is not a URDF link."
            )
        arms[side] = ArmRuntime(
            side=side,
            ee_link=arm.ee_link,
            ee_index=link_names.index(arm.ee_link),
            joint_names=tuple(joint_names),
            joint_indices=tuple(actuated_names.index(joint) for joint in joint_names),
        )
    return arms


def _resolve_finger_joints(
    urdf: yourdfpy.URDF,
    robot: pk.Robot,
    cfg: RobotConfig,
    arms: dict[str, ArmRuntime],
) -> dict[str, tuple[GripperJointRuntime, ...]]:
    actuated_names = list(robot.joints.actuated_names)
    fingers_by_side: dict[str, tuple[GripperJointRuntime, ...]] = {}
    for side in SIDES:
        if side not in arms:
            fingers_by_side[side] = ()
            continue
        configured = cfg.arms[side].gripper_joints
        fingers: list[GripperJointRuntime] = []
        if configured:
            for gripper_joint in configured:
                if gripper_joint.name not in actuated_names:
                    raise ValueError(
                        f"arms.{side}.gripper_joints contains non-actuated joint "
                        f"{gripper_joint.name!r}."
                    )
                open_value = (
                    gripper_joint.open_value
                    if gripper_joint.open_value is not None
                    else _joint_open_value(urdf, gripper_joint.name)
                )
                fingers.append(
                    GripperJointRuntime(
                        index=actuated_names.index(gripper_joint.name),
                        closed_value=gripper_joint.closed_value,
                        open_value=open_value,
                    )
                )
        else:
            for joint_name in arms[side].joint_names:
                joint = urdf.joint_map.get(joint_name)
                if joint is None or joint.type != "prismatic" or joint.limit is None:
                    continue
                fingers.append(
                    GripperJointRuntime(
                        index=actuated_names.index(joint_name),
                        closed_value=0.0,
                        open_value=_joint_open_value(urdf, joint_name),
                    )
                )
        fingers_by_side[side] = tuple(fingers)
    return fingers_by_side


def _joint_open_value(urdf: yourdfpy.URDF, joint_name: str) -> float:
    joint = urdf.joint_map.get(joint_name)
    if joint is None or joint.limit is None:
        raise ValueError(
            f"Cannot infer open value for {joint_name!r}; set open in YAML."
        )
    lower, upper = float(joint.limit.lower), float(joint.limit.upper)
    return upper if abs(upper) >= abs(lower) else lower


__all__ = [
    "ArmRuntime",
    "EMBODIMENT_NAMES",
    "GripperJointConfig",
    "GripperJointRuntime",
    "RobotArmConfig",
    "RobotConfig",
    "RobotRealConfig",
    "RobotRuntime",
    "available_robot_names",
    "load_embodiment",
    "robot_config_metadata",
    "load_robot_config",
    "yourdfpy_handler",
]
