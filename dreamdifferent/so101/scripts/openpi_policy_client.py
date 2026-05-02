#!/usr/bin/env python3
"""Control the SO101 follower with an OpenPI websocket policy server.

The SO101 OpenPI adapter server returns absolute action chunks.  This client
validates the server chunk shape, executes only the configurable prefix of each
chunk, and can prefetch new chunks asynchronously while the robot is moving.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import functools
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client

from script_utils import (
    DEFAULT_FINAL_POSE_PATH,
    DEFAULT_HOME_POSE_PATH,
    DEFAULT_PORTS_PATH,
    follower_config_kwargs,
    load_final_pose,
    load_home_pose,
    load_ports,
    move_robot_to_pose,
)


JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
ACTION_DIM = len(JOINT_NAMES)
DEFAULT_DEBUG_OBSERVATION_DIR = Path(__file__).resolve().parents[1] / "outputs" / "policy_client_debug"


@dataclasses.dataclass(frozen=True)
class PolicyResult:
    actions: np.ndarray
    elapsed_ms: float
    server_timing: dict[str, Any]


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class OpenPIWebsocketClient:
    def __init__(self, host: str, port: int) -> None:
        self.uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        logging.info("Connecting to OpenPI policy server at %s", self.uri)
        self.websocket = websockets.sync.client.connect(self.uri, compression=None, max_size=None)
        self.metadata = _unpackb(self.websocket.recv())

    def infer(self, observation: dict[str, Any], server_action_horizon: int) -> PolicyResult:
        start = time.perf_counter()
        self.websocket.send(_packb(observation))
        response = self.websocket.recv()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server returned an error:\n{response}")

        result = _unpackb(response)
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        expected_shape = (server_action_horizon, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected server actions shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Server returned non-finite action values")

        return PolicyResult(
            actions=actions,
            elapsed_ms=elapsed_ms,
            server_timing=dict(result.get("server_timing", {})),
        )

    def close(self) -> None:
        self.websocket.close()


def build_camera_configs(args: argparse.Namespace) -> dict[str, object]:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    camera_config: dict[str, object] = {}
    if args.camera:
        camera_config[args.camera_name] = OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    if args.secondary_camera:
        camera_config[args.secondary_camera_name] = OpenCVCameraConfig(
            index_or_path=args.secondary_camera_index,
            width=args.secondary_camera_width,
            height=args.secondary_camera_height,
            fps=args.camera_fps,
        )
    return camera_config


def build_robot_config(args: argparse.Namespace, follower_port: str):
    from lerobot.robots.so_follower import SO101FollowerConfig

    config_kwargs = follower_config_kwargs(follower_port)
    camera_config = build_camera_configs(args)
    if camera_config:
        config_kwargs["cameras"] = camera_config
    return SO101FollowerConfig(**config_kwargs)


def make_robot_action_processor():
    try:
        from lerobot.processor import make_default_robot_action_processor
    except ImportError:
        from lerobot.processor.factory import make_default_robot_action_processor

    return make_default_robot_action_processor()


def precise_sleep(duration_s: float) -> None:
    try:
        from lerobot.utils.robot_utils import precise_sleep as lerobot_precise_sleep
    except ImportError:
        if duration_s > 0:
            time.sleep(duration_s)
    else:
        lerobot_precise_sleep(max(duration_s, 0.0))


def resize_with_pad_uint8(image: Any, target_size: int) -> np.ndarray:
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = arr.astype(np.uint8, copy=False)

    height, width = arr.shape[:2]
    if (height, width) == (target_size, target_size):
        return arr

    pil_image = Image.fromarray(arr)
    ratio = max(width / target_size, height / target_size)
    resized_width = max(int(width / ratio), 1)
    resized_height = max(int(height / ratio), 1)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    padded = Image.new(resized.mode, (target_size, target_size), 0)
    pad_width = max((target_size - resized_width) // 2, 0)
    pad_height = max((target_size - resized_height) // 2, 0)
    padded.paste(resized, (pad_width, pad_height))
    return np.asarray(padded, dtype=np.uint8)


def extract_state(robot_observation: dict[str, Any]) -> np.ndarray:
    missing = [name for name in JOINT_NAMES if name not in robot_observation]
    if missing:
        raise KeyError(f"Robot observation is missing joint keys: {missing}")
    return np.asarray([float(robot_observation[name]) for name in JOINT_NAMES], dtype=np.float32)


def build_policy_observation(
    robot_observation: dict[str, Any],
    *,
    prompt: str,
    front_camera_name: str,
    wrist_camera_name: str,
    image_size: int,
) -> dict[str, Any]:
    missing_cameras = [name for name in (front_camera_name, wrist_camera_name) if name not in robot_observation]
    if missing_cameras:
        raise KeyError(
            "Robot observation is missing camera keys "
            f"{missing_cameras}. Available keys: {sorted(robot_observation.keys())}"
        )

    return {
        "observation/images/front": resize_with_pad_uint8(robot_observation[front_camera_name], image_size),
        "observation/images/wrist": resize_with_pad_uint8(robot_observation[wrist_camera_name], image_size),
        "observation/state": extract_state(robot_observation),
        "prompt": prompt,
    }


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def maybe_dump_first_policy_observation(observation: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.debug_dump_observation or getattr(args, "_debug_observation_dumped", False):
        return

    from PIL import Image

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.debug_observation_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    front_image = np.asarray(observation["observation/images/front"])
    wrist_image = np.asarray(observation["observation/images/wrist"])
    state = np.asarray(observation["observation/state"])

    Image.fromarray(front_image).save(output_dir / "front.jpg")
    Image.fromarray(wrist_image).save(output_dir / "wrist.jpg")

    summary = {
        "prompt": observation["prompt"],
        "state_joint_names": list(JOINT_NAMES),
        "state": state.astype(float).tolist(),
        "front_image": _array_summary(front_image),
        "wrist_image": _array_summary(wrist_image),
    }
    (output_dir / "observation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    logging.info(
        "Dumped first policy observation to %s (state=%s)",
        output_dir,
        np.array2string(state, precision=3, suppress_small=True),
    )
    setattr(args, "_debug_observation_dumped", True)


def actions_to_execute(result: PolicyResult, execute_actions_per_chunk: int) -> np.ndarray:
    return result.actions[:execute_actions_per_chunk].copy()


def action_row_to_dict(action_row: np.ndarray) -> dict[str, float]:
    return {name: float(action_row[idx]) for idx, name in enumerate(JOINT_NAMES)}


def clamp_action_to_observation(
    action: dict[str, float],
    robot_observation: dict[str, Any],
    max_relative_target: float,
) -> dict[str, float]:
    if max_relative_target <= 0:
        return action

    clamped: dict[str, float] = {}
    for name, target in action.items():
        current = float(robot_observation[name])
        lo = current - max_relative_target
        hi = current + max_relative_target
        clamped[name] = min(max(target, lo), hi)
    return clamped


def apply_action(
    *,
    robot,
    robot_action_processor,
    robot_observation: dict[str, Any],
    action_row: np.ndarray,
    max_relative_target: float,
    dry_run: bool,
) -> dict[str, float]:
    action = action_row_to_dict(action_row)
    action = clamp_action_to_observation(action, robot_observation, max_relative_target)
    processed_action = robot_action_processor((action, robot_observation))
    if not dry_run:
        robot.send_action(processed_action)
    return processed_action


def should_stop(start_time: float, steps: int, args: argparse.Namespace) -> bool:
    if args.max_steps > 0 and steps >= args.max_steps:
        return True
    if args.duration_sec > 0 and (time.perf_counter() - start_time) >= args.duration_sec:
        return True
    return False


def infer_from_robot_observation(
    *,
    client: OpenPIWebsocketClient,
    robot_observation: dict[str, Any],
    args: argparse.Namespace,
) -> PolicyResult:
    policy_observation = build_policy_observation(
        robot_observation,
        prompt=args.prompt,
        front_camera_name=args.secondary_camera_name,
        wrist_camera_name=args.camera_name,
        image_size=args.image_size,
    )
    maybe_dump_first_policy_observation(policy_observation, args)
    return client.infer(policy_observation, args.server_action_horizon)


def log_policy_result(label: str, result: PolicyResult, queue_len: int | None = None) -> None:
    server_ms = result.server_timing.get("infer_ms")
    server_msg = f", server_infer_ms={server_ms:.1f}" if isinstance(server_ms, (int, float)) else ""
    queue_msg = f", queue={queue_len}" if queue_len is not None else ""
    logging.info("%s: client_elapsed_ms=%.1f%s%s", label, result.elapsed_ms, server_msg, queue_msg)


def run_dry_run(*, client: OpenPIWebsocketClient, robot, args: argparse.Namespace) -> None:
    requests = max(args.max_steps, 1)
    for idx in range(requests):
        robot_observation = robot.get_observation()
        result = infer_from_robot_observation(client=client, robot_observation=robot_observation, args=args)
        selected = actions_to_execute(result, args.execute_actions_per_chunk)
        log_policy_result(f"dry_run_request={idx}", result, len(selected))
        logging.info("first_action=%s", np.array2string(selected[0], precision=4, suppress_small=True))


def run_sync_control(*, client: OpenPIWebsocketClient, robot, robot_action_processor, args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    steps = 0
    step_dt = 1.0 / args.fps

    while not should_stop(start_time, steps, args):
        chunk_observation = robot.get_observation()
        result = infer_from_robot_observation(client=client, robot_observation=chunk_observation, args=args)
        selected_actions = actions_to_execute(result, args.execute_actions_per_chunk)
        log_policy_result("sync_chunk", result, len(selected_actions))

        for action_row in selected_actions:
            if should_stop(start_time, steps, args):
                break
            loop_start = time.perf_counter()
            robot_observation = robot.get_observation()
            apply_action(
                robot=robot,
                robot_action_processor=robot_action_processor,
                robot_observation=robot_observation,
                action_row=action_row,
                max_relative_target=args.max_relative_target,
                dry_run=False,
            )
            steps += 1
            precise_sleep(step_dt - (time.perf_counter() - loop_start))


def run_async_control(*, client: OpenPIWebsocketClient, robot, robot_action_processor, args: argparse.Namespace) -> None:
    step_dt = 1.0 / args.fps
    queue: collections.deque[np.ndarray] = collections.deque()
    steps = 0
    empty_since: float | None = None
    last_status_t = 0.0

    initial_observation = robot.get_observation()
    initial_result = infer_from_robot_observation(client=client, robot_observation=initial_observation, args=args)
    queue.extend(actions_to_execute(initial_result, args.execute_actions_per_chunk))
    log_policy_result("initial_chunk", initial_result, len(queue))

    if args.enter_gate:
        input("Initial action chunk is ready. Press Enter to start sending actions to the robot...")

    start_time = time.perf_counter()
    threshold_count = max(1, math.ceil(args.chunk_size_threshold * args.execute_actions_per_chunk))
    logging.info("Starting async control loop")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future: concurrent.futures.Future[PolicyResult] | None = None
        while not should_stop(start_time, steps, args):
            loop_start = time.perf_counter()
            robot_observation = robot.get_observation()

            if future is not None and future.done():
                result = future.result()
                new_actions = list(actions_to_execute(result, args.execute_actions_per_chunk))
                if args.queue_update_mode == "replace":
                    queue.clear()
                queue.extend(new_actions)
                log_policy_result("async_chunk", result, len(queue))
                future = None

            if future is None and len(queue) <= threshold_count:
                inference_observation = robot_observation.copy()
                future = executor.submit(
                    infer_from_robot_observation,
                    client=client,
                    robot_observation=inference_observation,
                    args=args,
                )

            if queue:
                empty_since = None
                action_row = queue.popleft()
                apply_action(
                    robot=robot,
                    robot_action_processor=robot_action_processor,
                    robot_observation=robot_observation,
                    action_row=action_row,
                    max_relative_target=args.max_relative_target,
                    dry_run=False,
                )
                steps += 1
            else:
                if empty_since is None:
                    empty_since = time.perf_counter()
                    logging.warning("Action queue is empty; holding current pose while waiting for inference")
                elif time.perf_counter() - empty_since > args.empty_queue_timeout_sec:
                    raise TimeoutError(
                        f"Action queue stayed empty for > {args.empty_queue_timeout_sec:.1f}s; stopping for safety"
                    )
                hold_action = extract_state(robot_observation)
                apply_action(
                    robot=robot,
                    robot_action_processor=robot_action_processor,
                    robot_observation=robot_observation,
                    action_row=hold_action,
                    max_relative_target=args.max_relative_target,
                    dry_run=False,
                )

            now = time.perf_counter()
            if now - last_status_t >= args.status_interval_sec:
                last_status_t = now
                logging.info("control_status: steps=%d, queue=%d, future_pending=%s", steps, len(queue), future is not None)

            precise_sleep(step_dt - (time.perf_counter() - loop_start))


def validate_args(args: argparse.Namespace) -> None:
    if args.server_action_horizon != 24:
        logging.warning(
            "The current SO101 adapter server is expected to return horizon 24; validating against %d as requested",
            args.server_action_horizon,
        )
    if args.server_action_horizon <= 0:
        raise ValueError("--server-action-horizon must be positive")
    if not 1 <= args.execute_actions_per_chunk <= args.server_action_horizon:
        raise ValueError("--execute-actions-per-chunk must satisfy 1 <= value <= --server-action-horizon")
    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise ValueError("--chunk-size-threshold must be in [0, 1]")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not (args.camera and args.secondary_camera):
        raise ValueError("Both --camera and --secondary-camera are required for the SO101 OpenPI policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23261)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--config", type=Path, default=DEFAULT_PORTS_PATH)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--duration-sec", type=float, default=10.0, help="0 means no duration limit")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--mode", choices=("async", "sync"), default="async")
    parser.add_argument("--server-action-horizon", type=int, default=24)
    parser.add_argument("--execute-actions-per-chunk", type=int, default=12)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.9)
    parser.add_argument("--queue-update-mode", choices=("replace", "append"), default="replace")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-relative-target", type=float, default=10.0, help="<=0 disables client-side clamping")
    parser.add_argument("--dry-run", action="store_true", help="Run inference but do not send actions to the robot")
    parser.add_argument("--enter-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--start-home-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the robot to --home-pose-path after connecting and before inference/control.",
    )
    parser.add_argument("--home-pose-path", type=Path, default=DEFAULT_HOME_POSE_PATH)
    parser.add_argument(
        "--home-move-time-sec",
        type=float,
        default=1.5,
        help="Duration of the smooth motion to the home pose at startup.",
    )
    parser.add_argument("--return-final-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-pose-path", type=Path, default=DEFAULT_FINAL_POSE_PATH)
    parser.add_argument("--return-move-time-sec", type=float, default=1.5)
    parser.add_argument("--empty-queue-timeout-sec", type=float, default=2.0)
    parser.add_argument("--status-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--debug-dump-observation",
        action="store_true",
        help="Save the first policy observation summary and resized camera images for debugging.",
    )
    parser.add_argument("--debug-observation-dir", type=Path, default=DEFAULT_DEBUG_OBSERVATION_DIR)

    parser.add_argument("--camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-name", default="wrist", help="Primary camera name; sent as observation/images/wrist.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--secondary-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--secondary-camera-name",
        default="front",
        help="Secondary camera name; sent as observation/images/front.",
    )
    parser.add_argument("--secondary-camera-index", type=int, default=2)
    parser.add_argument("--secondary-camera-width", type=int, default=640)
    parser.add_argument("--secondary-camera-height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    validate_args(args)

    from lerobot.robots.so_follower import SO101Follower

    ports = load_ports(args.config)
    robot_config = build_robot_config(args, ports["follower"])
    robot = SO101Follower(robot_config)
    robot_action_processor = make_robot_action_processor()
    home_pose = load_home_pose(args.home_pose_path) if args.home_pose_path.exists() else None
    final_pose = load_final_pose(args.final_pose_path) if args.final_pose_path.exists() else None
    client: OpenPIWebsocketClient | None = None
    robot_connected = False

    try:
        client = OpenPIWebsocketClient(args.host, args.port)
        logging.info("Server metadata: %s", client.metadata)

        robot.connect()
        robot_connected = True

        if args.start_home_pose:
            if home_pose is None:
                raise FileNotFoundError(f"Home pose file does not exist: {args.home_pose_path}")
            logging.info("Moving robot to home pose")
            move_robot_to_pose(
                robot=robot,
                target_pose=home_pose,
                duration_s=args.home_move_time_sec,
                fps=max(int(args.fps), 1),
            )
            precise_sleep(0.2)

        if args.dry_run:
            run_dry_run(client=client, robot=robot, args=args)
        elif args.mode == "sync":
            run_sync_control(client=client, robot=robot, robot_action_processor=robot_action_processor, args=args)
        else:
            run_async_control(client=client, robot=robot, robot_action_processor=robot_action_processor, args=args)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        if robot_connected:
            if final_pose is not None and args.return_final_pose:
                logging.info("Returning robot to final pose")
                move_robot_to_pose(
                    robot=robot,
                    target_pose=final_pose,
                    duration_s=args.return_move_time_sec,
                    fps=max(int(args.fps), 1),
                )
                precise_sleep(0.2)
            robot.disconnect()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
