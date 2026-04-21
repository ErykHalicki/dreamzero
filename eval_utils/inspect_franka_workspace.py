import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_python_paths() -> None:
    candidate_paths = [
        REPO_ROOT,
        REPO_ROOT / "sim_evals" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_python_paths()


FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FRANKA_ARM_JOINT_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}

DEFAULT_END_EFFECTOR_BODY_CANDIDATES = [
    "base_link",
    "robotiq_base_link",
    "panda_hand",
    "panda_link8",
]


def _resolve_joint_indices(robot, joint_names: list[str]) -> list[int]:
    joint_index_by_name = {name: i for i, name in enumerate(robot.data.joint_names)}
    missing = [name for name in joint_names if name not in joint_index_by_name]
    if missing:
        raise ValueError(
            f"Failed to find arm joints {missing}. "
            f"Available joint names: {list(robot.data.joint_names)}"
        )
    return [joint_index_by_name[name] for name in joint_names]


def _extract_limits_from_tensor(limit_tensor, joint_indices: list[int]) -> torch.Tensor | None:
    if limit_tensor is None:
        return None
    tensor = limit_tensor
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[-1] != 2:
        return None
    if max(joint_indices) >= tensor.shape[0]:
        return None
    return tensor[joint_indices]


def _resolve_arm_joint_limits(robot, joint_names: list[str], joint_indices: list[int]) -> tuple[torch.Tensor, str]:
    for attr_name in ("soft_joint_pos_limits", "joint_pos_limits", "joint_limits"):
        limits = _extract_limits_from_tensor(getattr(robot.data, attr_name, None), joint_indices)
        if limits is not None:
            return limits.to(device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype), f"robot.data.{attr_name}"

    root_view = getattr(robot, "root_physx_view", None)
    if root_view is not None and hasattr(root_view, "get_dof_limits"):
        try:
            limits = _extract_limits_from_tensor(root_view.get_dof_limits(), joint_indices)
            if limits is not None:
                return limits.to(device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype), "robot.root_physx_view.get_dof_limits()"
        except Exception:
            pass

    limits = torch.tensor(
        [FRANKA_ARM_JOINT_LIMITS[name] for name in joint_names],
        device=robot.data.joint_pos.device,
        dtype=robot.data.joint_pos.dtype,
    )
    return limits, "built-in Franka Panda fallback"


def _resolve_end_effector_body_name(robot, explicit_name: str | None) -> str:
    body_names = list(robot.data.body_names)
    if explicit_name is not None:
        if explicit_name not in body_names:
            raise ValueError(
                f"Body '{explicit_name}' not found. Available body names: {body_names}"
            )
        return explicit_name

    for candidate in DEFAULT_END_EFFECTOR_BODY_CANDIDATES:
        if candidate in body_names:
            return candidate

    for name in body_names:
        lowered = name.lower()
        if "gripper" in lowered and "base_link" in lowered:
            return name

    for name in body_names:
        lowered = name.lower()
        if "hand" in lowered:
            return name

    raise ValueError(
        "Failed to auto-resolve an end-effector body. "
        f"Available body names: {body_names}"
    )


def _resolve_root_position(robot) -> np.ndarray | None:
    for attr_name in ("root_pos_w", "root_state_w"):
        value = getattr(robot.data, attr_name, None)
        if value is None:
            continue
        if attr_name == "root_state_w":
            return value[0, :3].detach().cpu().numpy().astype(np.float32)
        return value[0].detach().cpu().numpy().astype(np.float32)
    return None


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{float(point[0])} {float(point[1])} {float(point[2])}\n")


def _sample_workspace_points(
    base_env,
    robot_name: str,
    body_name: str,
    joint_names: list[str],
    num_samples: int,
    settle_steps: int,
    seed: int,
    gripper_joint: float,
) -> tuple[np.ndarray, dict]:
    robot = base_env.scene[robot_name]
    joint_indices = _resolve_joint_indices(robot, joint_names)
    arm_limits, limit_source = _resolve_arm_joint_limits(robot, joint_names, joint_indices)
    body_names = list(robot.data.body_names)
    body_index = body_names.index(body_name)
    joint_index_by_name = {name: i for i, name in enumerate(robot.data.joint_names)}
    gripper_index = joint_index_by_name.get("finger_joint")

    env_ids = torch.tensor([0], device=robot.data.joint_pos.device, dtype=torch.long)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    rng = np.random.default_rng(seed)

    lo = arm_limits[:, 0]
    hi = arm_limits[:, 1]
    samples = []

    for _ in range(num_samples):
        sampled_arm_np = rng.uniform(
            low=lo.detach().cpu().numpy(),
            high=hi.detach().cpu().numpy(),
        ).astype(np.float32)
        sampled_arm = torch.from_numpy(sampled_arm_np).to(
            device=robot.data.joint_pos.device,
            dtype=robot.data.joint_pos.dtype,
        )
        joint_pos = default_joint_pos.clone()
        joint_vel = default_joint_vel.clone()
        joint_pos[0, joint_indices] = sampled_arm
        joint_vel[0, joint_indices] = 0.0
        if gripper_index is not None:
            joint_pos[0, gripper_index] = gripper_joint
            joint_vel[0, gripper_index] = 0.0

        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        for _ in range(max(1, settle_steps)):
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)

        point = robot.data.body_pos_w[0, body_index, :3].detach().cpu().numpy().astype(np.float32)
        samples.append(point)

    points = np.stack(samples, axis=0)
    root_position = _resolve_root_position(robot)
    relative_points = None if root_position is None else points - root_position[None, :]

    metadata = {
        "body_name": body_name,
        "robot_name": robot_name,
        "joint_names": joint_names,
        "joint_limit_source": limit_source,
        "joint_limits": {
            name: [float(arm_limits[i, 0].item()), float(arm_limits[i, 1].item())]
            for i, name in enumerate(joint_names)
        },
        "root_position_world": None if root_position is None else root_position.tolist(),
        "num_samples": int(num_samples),
        "settle_steps": int(settle_steps),
        "gripper_joint": float(gripper_joint),
        "seed": int(seed),
        "world_min_xyz": points.min(axis=0).tolist(),
        "world_max_xyz": points.max(axis=0).tolist(),
        "world_span_xyz": (points.max(axis=0) - points.min(axis=0)).tolist(),
        "world_mean_xyz": points.mean(axis=0).tolist(),
    }
    if relative_points is not None:
        metadata["relative_min_xyz"] = relative_points.min(axis=0).tolist()
        metadata["relative_max_xyz"] = relative_points.max(axis=0).tolist()
        metadata["relative_span_xyz"] = (relative_points.max(axis=0) - relative_points.min(axis=0)).tolist()
        metadata["max_radius_from_base_m"] = float(np.linalg.norm(relative_points, axis=1).max())

    return points, metadata


def main(
    env_id: str = "DROID_BIMANUAL_RIGHT_ONLY",
    robot_name: str = "left_robot",
    scene: int = 1,
    body_name: str | None = None,
    samples: int = 2000,
    settle_steps: int = 1,
    gripper_joint: float = 0.0,
    headless: bool = True,
    use_fabric: bool = False,
    seed: int = 0,
    output_dir: str | None = None,
):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Sample Franka end-effector workspace in Isaac.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = False
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if output_dir is None:
        output_path = REPO_ROOT / "runs" / "workspace_samples" / timestamp
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        env_id,
        device=args_cli.device,
        num_envs=1,
        use_fabric=use_fabric,
    )
    if hasattr(env_cfg, "set_scene"):
        env_cfg.set_scene(scene)

    env = None

    try:
        env = gym.make(env_id, cfg=env_cfg)
        env.reset()
        env.reset()
        base_env = getattr(env, "unwrapped", env)

        for _ in range(4):
            base_env.sim.step(render=False)
            base_env.scene.update(dt=base_env.physics_dt)

        robot = base_env.scene[robot_name]
        resolved_body_name = _resolve_end_effector_body_name(robot, body_name)
        print(f"[workspace] env_id={env_id}")
        print(f"[workspace] robot_name={robot_name}")
        print(f"[workspace] body_name={resolved_body_name}")
        print(f"[workspace] body_names={list(robot.data.body_names)}")

        points, metadata = _sample_workspace_points(
            base_env=base_env,
            robot_name=robot_name,
            body_name=resolved_body_name,
            joint_names=FRANKA_ARM_JOINT_NAMES,
            num_samples=samples,
            settle_steps=settle_steps,
            seed=seed,
            gripper_joint=gripper_joint,
        )

        npz_path = output_path / f"{robot_name}_workspace_points.npz"
        ply_path = output_path / f"{robot_name}_workspace_points.ply"
        json_path = output_path / f"{robot_name}_workspace_stats.json"

        np.savez_compressed(npz_path, points=points)
        _write_ascii_ply(ply_path, points)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[workspace] saved npz: {npz_path}")
        print(f"[workspace] saved ply: {ply_path}")
        print(f"[workspace] saved json: {json_path}")
        print(f"[workspace] world min xyz: {np.round(points.min(axis=0), 4).tolist()}")
        print(f"[workspace] world max xyz: {np.round(points.max(axis=0), 4).tolist()}")
        print(f"[workspace] world span xyz: {np.round(points.max(axis=0) - points.min(axis=0), 4).tolist()}")
        if "max_radius_from_base_m" in metadata:
            print(f"[workspace] max radius from base: {metadata['max_radius_from_base_m']:.4f} m")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
