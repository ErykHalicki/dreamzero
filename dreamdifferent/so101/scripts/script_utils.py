import json
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTS_PATH = PROJECT_ROOT / "config" / "so101_ports.json"
DEFAULT_CALIBRATION_ROOT = PROJECT_ROOT / "config" / "calibration"
DEFAULT_FOLLOWER_CALIBRATION_DIR = DEFAULT_CALIBRATION_ROOT / "robots" / "so_follower"
DEFAULT_LEADER_CALIBRATION_DIR = DEFAULT_CALIBRATION_ROOT / "teleoperators" / "so_leader"
DEFAULT_HOME_POSE_PATH = PROJECT_ROOT / "config" / "so101_home_pose.json"
DEFAULT_FINAL_POSE_PATH = PROJECT_ROOT / "config" / "so101_final_pose.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot"


def load_ports(config_path: Path) -> dict[str, str]:
    with config_path.open("r", encoding="utf-8") as f:
        ports = json.load(f)

    missing_roles = {"leader", "follower"} - ports.keys()
    if missing_roles:
        missing = ", ".join(sorted(missing_roles))
        raise ValueError(f"Missing port entries in {config_path}: {missing}")

    return {
        "leader": str(ports["leader"]),
        "follower": str(ports["follower"]),
    }


def follower_config_kwargs(port: str) -> dict[str, object]:
    return {
        "port": port,
        "id": "follower",
        "calibration_dir": DEFAULT_FOLLOWER_CALIBRATION_DIR,
    }


def leader_config_kwargs(port: str) -> dict[str, object]:
    return {
        "port": port,
        "id": "leader",
        "calibration_dir": DEFAULT_LEADER_CALIBRATION_DIR,
    }


def load_home_pose(home_pose_path: Path) -> dict[str, float]:
    with home_pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)

    if not isinstance(pose, dict) or not pose:
        raise ValueError(f"Home pose file is empty or invalid: {home_pose_path}")

    return {str(key): float(value) for key, value in pose.items()}


def save_home_pose(home_pose_path: Path, pose: dict[str, float]) -> None:
    home_pose_path.parent.mkdir(parents=True, exist_ok=True)
    home_pose_path.write_text(json.dumps(pose, indent=2) + "\n", encoding="utf-8")


def load_final_pose(final_pose_path: Path) -> dict[str, float]:
    return load_home_pose(final_pose_path)


def save_final_pose(final_pose_path: Path, pose: dict[str, float]) -> None:
    save_home_pose(final_pose_path, pose)


def extract_joint_pose(observation: dict[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in observation.items()
        if key.endswith(".pos")
    }


def move_robot_to_pose(
    robot,
    target_pose: dict[str, float],
    duration_s: float,
    fps: int,
) -> None:
    from lerobot.utils.robot_utils import precise_sleep

    current_pose = extract_joint_pose(robot.get_observation())
    common_keys = [key for key in target_pose if key in current_pose]
    if not common_keys:
        return

    steps = max(int(duration_s * fps), 1)
    for step_idx in range(1, steps + 1):
        t0 = time.perf_counter()
        alpha = step_idx / steps
        action = {
            key: (1.0 - alpha) * current_pose[key] + alpha * target_pose[key]
            for key in common_keys
        }
        robot.send_action(action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def return_to_pose_if_enabled(args, robot, return_pose) -> None:
    if not args.return_to_initial_pose:
        return
    from lerobot.utils.robot_utils import precise_sleep

    print(
        "Returning robot to home pose"
        if args.return_pose_source == "home"
        else "Returning robot to initial pose"
    )
    move_robot_to_pose(
        robot=robot,
        target_pose=return_pose,
        duration_s=args.return_move_time_sec,
        fps=args.fps,
    )
    precise_sleep(0.2)
