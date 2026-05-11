import argparse
from pathlib import Path

from script_utils import DEFAULT_PORTS_PATH, follower_config_kwargs, leader_config_kwargs, load_ports


def setup_motor(role: str, port: str) -> None:
    if role == "follower":
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        motor = SO101Follower(SO101FollowerConfig(**follower_config_kwargs(port)))
    else:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        motor = SO101Leader(SO101LeaderConfig(**leader_config_kwargs(port)))

    motor.setup_motors()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("leader", "follower"))
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ports = load_ports(args.config)
    setup_motor(args.role, ports[args.role])


if __name__ == "__main__":
    main()
