"""Move the SO-101 follower to the starting pose of episode 88 and hold.

Pose derivation
---------------
Source:  robot-learning-team43/so101_teleop_private, episode 88
Method:  mean of observation.state over the first 5 frames (~167 ms).
Computed from local parquet cache via pandas.

Torque is left enabled after the script exits so the arm holds position.
"""
import argparse

from script_utils import DEFAULT_PORTS_PATH, follower_config_kwargs, load_ports, move_robot_to_pose

DATASET_START_POSE = {
    "shoulder_pan.pos": 7.1209,
    "shoulder_lift.pos": -72.6154,
    "elbow_flex.pos": 29.5385,
    "wrist_flex.pos": 88.5714,
    "wrist_roll.pos": -89.7143,
    "gripper.pos": 1.1612,
}

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=None, help="Follower serial port, e.g. /dev/ttyACM0")
args = parser.parse_args()

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

port = args.port or load_ports(DEFAULT_PORTS_PATH)["follower"]
robot = SO101Follower(SO101FollowerConfig(**follower_config_kwargs(port), disable_torque_on_disconnect=False))
robot.connect()

try:
    move_robot_to_pose(robot, DATASET_START_POSE, duration_s=3.0, fps=50)
finally:
    robot.disconnect()
