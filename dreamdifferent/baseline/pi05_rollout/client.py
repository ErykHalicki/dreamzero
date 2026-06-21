#!/usr/bin/env python
"""One-shot LeRobot async robot client wrapper for the persistent Pi05 server."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys


DEFAULT_CHECKPOINT = (
    "/cluster/scratch/dohkim/pi05/experiments/pi05_homogeneous_lora_a100/"
    "outputs/checkpoints/025000/pretrained_model"
)
DEFAULT_TASK = "pick up the bottle and place it into the container"
DEFAULT_START_POSE_SCRIPT = (
    "/Users/dhkim/workspace/ETH/3DVision/dreamdifferent/dreamzero/dreamdifferent/"
    "so101/scripts/goto_start_pose2.py"
)
DEFAULT_CALIBRATION_DIR = (
    "/Users/dhkim/workspace/ETH/3DVision/dreamdifferent/dreamzero/dreamdifferent/"
    "so101/config/calibration/robots/so_follower/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--server", default="127.0.0.1:8000")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B140333651")
    parser.add_argument("--calibration-dir", default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--robot-id", default="follower")
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--front-camera-index", default="3")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.3)
    parser.add_argument("--aggregate-fn-name", default="weighted_average")
    parser.add_argument("--start-pose-script", default=DEFAULT_START_POSE_SCRIPT)
    parser.add_argument("--skip-start-pose", action="store_true")
    parser.add_argument(
        "--return-to-start-on-ctrl-c",
        dest="return_to_start_on_ctrl_c",
        action="store_true",
        help="Run the start-pose script after Ctrl-C stops the rollout.",
    )
    parser.add_argument(
        "--no-return-to-start-on-ctrl-c",
        dest="return_to_start_on_ctrl_c",
        action="store_false",
        help="Do not run the start-pose script after Ctrl-C.",
    )
    parser.add_argument("--debug-visualize-queue-size", dest="debug_visualize_queue_size", action="store_true")
    parser.add_argument("--no-debug-visualize-queue-size", dest="debug_visualize_queue_size", action="store_false")
    parser.set_defaults(debug_visualize_queue_size=True, return_to_start_on_ctrl_c=True)
    return parser.parse_args()


def run_start_pose(script: str) -> None:
    subprocess.run([sys.executable, script], check=True)


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> None:
    args = parse_args()

    if not args.skip_start_pose:
        run_start_pose(args.start_pose_script)

    cameras = (
        "{ "
        f"wrist: {{type: opencv, index_or_path: {args.wrist_camera_index}, "
        f"width: {args.camera_width}, height: {args.camera_height}, fps: {args.camera_fps}}}, "
        f"front: {{type: opencv, index_or_path: {args.front_camera_index}, "
        f"width: {args.camera_width}, height: {args.camera_height}, fps: {args.camera_fps}}}"
        "}"
    )

    command = [
        sys.executable,
        "-m",
        "lerobot.async_inference.robot_client",
        f"--server_address={args.server}",
        "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}",
        f"--robot.calibration_dir={args.calibration_dir}",
        f"--robot.id={args.robot_id}",
        f"--robot.cameras={cameras}",
        f"--task={args.task}",
        "--policy_type=pi05",
        f"--pretrained_name_or_path={args.checkpoint}",
        f"--policy_device={args.policy_device}",
        f"--actions_per_chunk={args.actions_per_chunk}",
        f"--chunk_size_threshold={args.chunk_size_threshold}",
        f"--aggregate_fn_name={args.aggregate_fn_name}",
        f"--debug_visualize_queue_size={args.debug_visualize_queue_size}",
    ]

    process = subprocess.Popen(command)
    try:
        raise SystemExit(process.wait())
    except KeyboardInterrupt:
        stop_child(process)
        if args.return_to_start_on_ctrl_c:
            print("Ctrl-C received. Returning robot to start pose...", file=sys.stderr)
            run_start_pose(args.start_pose_script)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
