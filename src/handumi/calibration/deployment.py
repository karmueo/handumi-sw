"""解析机器人与桌面标定，并允许调用方在缺少仿真布局时提供估计结果。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from handumi.config import DEFAULT_RIG_CONFIG, load_optional_rig_section
from handumi.robots.registry import RESOURCE_ROOT
from handumi.robots.utils import quat_normalize

SIM_CALIBRATION_DIR = RESOURCE_ROOT / "configs" / "calibration" / "table" / "sim"


@dataclass(frozen=True)
class DeploymentCalibration:
    """One robot-world-from-table transform and its deployment provenance."""

    robot: str
    pose7: np.ndarray
    path: Path | None
    scope: str
    source: str
    verified: bool
    profile: str
    lab: str | None


def simulation_calibration_path(robot: str) -> Path:
    return SIM_CALIBRATION_DIR / f"{robot}.yaml"


def local_calibration_path(
    robot: str,
    *,
    rig_config: Path = DEFAULT_RIG_CONFIG,
) -> Path | None:
    """Return an override or conventional lab-local ``<robot>.yaml`` path."""
    if rig_config.exists():
        deployment = load_optional_rig_section(rig_config, "deployment")
        calibrations = deployment.get("table_calibrations") or {}
        if not isinstance(calibrations, dict):
            raise SystemExit(
                f"Invalid 'deployment.table_calibrations' in {rig_config}; "
                "expected a mapping."
            )
        value = calibrations.get(robot)
        if value is not None:
            path = Path(str(value)).expanduser()
            if path.is_absolute() or path.exists():
                return path
            relative_to_rig = rig_config.parent / path
            return relative_to_rig if relative_to_rig.exists() else path

    conventional = (
        rig_config.parent / "calibration" / "table" / "local" / f"{robot}.yaml"
    )
    return conventional if conventional.exists() else None


def load_deployment_calibration(
    path: Path,
    *,
    expected_robot: str | None = None,
    profile: str = "explicit",
) -> DeploymentCalibration:
    """Load a versioned robot/table YAML, including legacy schema-v1 files."""
    if not path.exists():
        raise SystemExit(
            f"Missing deployment calibration: {path}\n"
            "Create a lab-local file and select it in configs/rig.yaml, or use "
            "--deployment-profile sim."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid deployment calibration {path}; expected a mapping.")
    if data.get("kind") != "handumi_robot_table_calibration":
        raise SystemExit(
            f"Invalid deployment calibration {path}: expected "
            "kind: handumi_robot_table_calibration."
        )
    robot = str(data.get("robot") or "")
    if expected_robot is not None and robot != expected_robot:
        raise SystemExit(
            f"Deployment calibration {path} declares robot {robot!r}; "
            f"expected {expected_robot!r}."
        )
    scope = data.get("scope")
    if scope not in {"simulation", "physical"}:
        raise SystemExit(
            f"Invalid deployment calibration {path}: missing or invalid scope. "
            "Declare scope: physical for a measured lab installation or "
            "scope: simulation for a portable simulation layout."
        )
    verified = data.get("verified") is True
    if scope == "simulation" and verified:
        raise SystemExit(
            f"Invalid deployment calibration {path}: a simulation placement cannot "
            "be physically verified. Create a separate scope: physical lab-local file."
        )
    root = data.get("calibration", data)
    try:
        raw_pose = root["robot_from_table"]
        position = np.asarray(raw_pose["position"], dtype=np.float32).reshape(3)
        quaternion = np.asarray(raw_pose["quaternion"], dtype=np.float32).reshape(4)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid deployment calibration {path}: robot_from_table must contain "
            "position[3] and quaternion[4]."
        ) from exc
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise SystemExit(f"Invalid deployment calibration {path}: pose is not finite.")
    if not np.isclose(float(np.linalg.norm(quaternion)), 1.0, atol=1e-3):
        raise SystemExit(
            f"Invalid deployment calibration {path}: robot_from_table quaternion "
            "is not normalized."
        )
    pose7 = np.concatenate([position, quat_normalize(quaternion)]).astype(np.float32)
    return DeploymentCalibration(
        robot=robot,
        pose7=pose7,
        path=path,
        scope=scope,
        source=str(data.get("source") or "unknown"),
        verified=verified,
        profile=profile,
        lab=str(data["lab"]).strip() if data.get("lab") else None,
    )


def resolve_deployment_calibration(
    robot: str,
    *,
    explicit_path: Path | None = None,
    profile: str = "auto",
    rig_config: Path = DEFAULT_RIG_CONFIG,
    identity_fallback: bool = False,
    simulation_fallback: Callable[[], DeploymentCalibration] | None = None,
) -> DeploymentCalibration:
    """优先显式和本地标定；标准仿真文件缺失时可调用估计回调。

    回调仅处理未定义的仿真布局；显式路径、local 要求及无效文件仍严格校验。
    """
    if profile not in {"auto", "local", "sim"}:
        raise ValueError(f"Unsupported deployment profile {profile!r}.")
    if explicit_path is not None:
        return load_deployment_calibration(
            explicit_path,
            expected_robot=robot,
            profile="explicit",
        )

    if profile in {"auto", "local"}:
        local_path = local_calibration_path(robot, rig_config=rig_config)
        if local_path is not None:
            calibration = load_deployment_calibration(
                local_path,
                expected_robot=robot,
                profile="local",
            )
            if calibration.scope != "physical":
                raise SystemExit(
                    f"Lab-local calibration {local_path} has "
                    f"scope={calibration.scope!r}; use scope: physical or select "
                    "--deployment-profile sim."
                )
            if not rig_config.exists():
                raise SystemExit(
                    f"Found lab-local calibration {local_path} but {rig_config} "
                    "does not exist. Create it (cp configs/rig.example.yaml "
                    "configs/rig.yaml) and set deployment.lab, or select "
                    "--deployment-profile sim."
                )
            deployment = load_optional_rig_section(rig_config, "deployment")
            rig_lab = str(deployment.get("lab") or "").strip() or None
            if rig_lab is None:
                raise SystemExit(
                    f"Missing deployment.lab in {rig_config}. Give this physical "
                    "installation a stable research-lab identifier."
                )
            if calibration.lab is not None and calibration.lab != rig_lab:
                raise SystemExit(
                    f"Lab mismatch: {local_path} declares {calibration.lab!r}, but "
                    f"{rig_config} selects {rig_lab!r}."
                )
            return replace(calibration, lab=rig_lab)
        if identity_fallback and profile == "auto":
            return DeploymentCalibration(
                robot=robot,
                pose7=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                path=None,
                scope="assumed",
                source="identity_assumed_table_equals_base",
                verified=False,
                profile="identity",
                lab=None,
            )
        if profile == "local":
            raise SystemExit(
                f"No lab-local table calibration was found for {robot!r}. Create:\n"
                f"  {rig_config.parent / 'calibration' / 'table' / 'local' / f'{robot}.yaml'}\n"
                "from configs/calibration/table/local/example.yaml, or configure "
                f"deployment.table_calibrations.{robot} in {rig_config}."
            )
    simulation_path = simulation_calibration_path(robot)  # 标准仿真布局的位置。
    if not simulation_path.exists() and simulation_fallback is not None:
        return simulation_fallback()
    return load_deployment_calibration(
        simulation_path,
        expected_robot=robot,
        profile="sim",
    )


def print_deployment_calibration(
    calibration: DeploymentCalibration,
    *,
    prefix: str = "[replay]",
) -> None:
    """Print the resolved placement provenance and its safety caveat."""
    print(
        f"{prefix} deployment calibration: "
        f"profile={calibration.profile} scope={calibration.scope} "
        f"verified={str(calibration.verified).lower()} source={calibration.source} "
        f"lab={calibration.lab or '-'} path={calibration.path}"
    )
    if calibration.scope == "physical" and not calibration.verified:
        print(
            f"{prefix} warning: lab-local physical calibration is not verified; "
            "measure the real robot/table installation before hardware use."
        )
    elif calibration.scope == "assumed":
        print(f"{prefix} assumption: table and robot base frames coincide.")
    elif calibration.scope == "simulation":
        print(
            f"{prefix} note: simulation placement is not a physical lab calibration."
        )


def deployment_calibration_metadata(calibration: DeploymentCalibration) -> dict[str, Any]:
    """Serializable provenance for rollouts and converted datasets."""
    from handumi.dataset.paths import portable_path

    return {
        "robot": calibration.robot,
        "path": portable_path(calibration.path) if calibration.path is not None else None,
        "profile": calibration.profile,
        "scope": calibration.scope,
        "source": calibration.source,
        "verified": calibration.verified,
        "lab": calibration.lab,
    }


__all__ = [
    "DeploymentCalibration",
    "SIM_CALIBRATION_DIR",
    "deployment_calibration_metadata",
    "load_deployment_calibration",
    "local_calibration_path",
    "print_deployment_calibration",
    "resolve_deployment_calibration",
    "simulation_calibration_path",
]
