import argparse
import time
from pathlib import Path

from script_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_FINAL_POSE_PATH,
    DEFAULT_PORTS_PATH,
    follower_config_kwargs,
    load_final_pose,
    load_ports,
)


DEFAULT_DATASET_REPO_ID = "local/so101_teleop"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded LeRobot episode on the SO-101 follower arm."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo identifier used for local dataset lookup.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base directory where the dataset is stored locally.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Optional FPS override for replay. Defaults to the dataset FPS.",
    )
    parser.add_argument(
        "--final-pose-path",
        type=Path,
        default=DEFAULT_FINAL_POSE_PATH,
        help="Path to a saved final pose JSON file used when replay exits.",
    )
    parser.add_argument(
        "--return-move-time-sec",
        type=float,
        default=1.5,
        help="Duration of the smooth motion used when returning to the final pose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor import make_default_robot_action_processor
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.constants import ACTION
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import log_say

    ports = load_ports(args.config)
    dataset_root = args.dataset_root / args.dataset_repo_id

    robot = SO101Follower(SO101FollowerConfig(**follower_config_kwargs(ports["follower"])))

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        episodes=[args.episode],
    )
    actions = dataset.select_columns(ACTION)
    robot_action_processor = make_default_robot_action_processor()
    replay_fps = args.fps if args.fps is not None else dataset.fps
    final_pose = load_final_pose(args.final_pose_path) if args.final_pose_path.exists() else None

    robot.connect()

    try:
        log_say(f"Replaying episode {args.episode}")
        for idx in range(dataset.num_frames):
            t0 = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {
                name: action_array[i] for i, name in enumerate(dataset.features[ACTION]["names"])
            }

            robot_obs = robot.get_observation()
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)

            precise_sleep(max(1.0 / replay_fps - (time.perf_counter() - t0), 0.0))
    finally:
        if final_pose is not None:
            log_say("Returning robot to final pose")
            move_robot_to_pose(
                robot=robot,
                target_pose=final_pose,
                duration_s=args.return_move_time_sec,
                fps=replay_fps,
            )
            precise_sleep(0.2)
        robot.disconnect()


if __name__ == "__main__":
    main()
