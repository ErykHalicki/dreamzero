import argparse
from pathlib import Path

from script_utils import (
    DEFAULT_HOME_POSE_PATH,
    DEFAULT_PORTS_PATH,
    extract_joint_pose,
    follower_config_kwargs,
    load_ports,
    save_home_pose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the current SO-101 follower joint pose as a reusable home pose JSON file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_HOME_POSE_PATH,
        help="Path to write the saved home pose JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    ports = load_ports(args.config)
    robot = SO101Follower(SO101FollowerConfig(**follower_config_kwargs(ports["follower"])))
    robot.connect()

    try:
        observation = robot.get_observation()
        pose = extract_joint_pose(observation)
    finally:
        robot.disconnect()

    save_home_pose(args.output, pose)
    print(f"Saved home pose to: {args.output}")


if __name__ == "__main__":
    main()
