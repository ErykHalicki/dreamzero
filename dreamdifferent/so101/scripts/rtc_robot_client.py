#!/usr/bin/env python
"""Custom SO-101 robot client for the RTC policy server."""

from __future__ import annotations

import argparse
import logging
import pickle  # nosec: local trusted robotics process payloads
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("rtc_robot_client")
_REFILL_IDLE_SLEEP_S = 0.01
_REFILL_ERROR_SLEEP_S = 0.5


@dataclass
class RTCClientSetup:
    task: str
    fps: float
    robot_type: str
    hw_features: dict[str, Any]
    dataset_features: dict[str, Any]
    action_feature_names: list[str]
    rename_map: dict[str, str]


@dataclass
class TimedObservationPayload:
    timestamp: float
    timestep: int
    observation: dict[str, Any]


@dataclass
class ActionPayload:
    action: Any | None = None
    actions: list[Any] | None = None
    ordered_action_keys: list[str] | None = None
    queue_size: int = 0
    server_timestep: int = 0


def _coerce_camera_index(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _extract_joint_pose(observation: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in observation.items() if key.endswith(".pos")}


def _return_to_initial_pose(robot, target_pose: dict[str, float], duration_s: float, fps: int) -> None:
    if not target_pose:
        return
    LOGGER.info("Returning robot to initial position before shutdown")
    current_pose = _extract_joint_pose(robot.get_observation())
    common_keys = [key for key in target_pose if key in current_pose]
    if not common_keys:
        LOGGER.warning("No common joint keys found for return-to-initial-position")
        return

    steps = max(int(duration_s * fps), 1)
    for step in range(1, steps + 1):
        start_t = time.perf_counter()
        alpha = step / steps
        action = {
            key: (1.0 - alpha) * current_pose[key] + alpha * target_pose[key]
            for key in common_keys
        }
        robot.send_action(action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - start_t), 0.0))


def _build_robot_config(args: argparse.Namespace):
    cameras = {
        args.camera_name: OpenCVCameraConfig(
            index_or_path=_coerce_camera_index(str(args.camera_index)),
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps or args.fps,
        )
    }
    if args.robot_type != "so101_follower":
        raise ValueError("This RTC client currently implements the requested so101_follower setup only.")
    return SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=Path(args.calibration_dir),
        cameras=cameras,
    )


def _build_setup(robot, args: argparse.Namespace) -> tuple[RTCClientSetup, list[str]]:
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    observation_features_hw = {
        key: value
        for key, value in robot.observation_features.items()
        if isinstance(value, tuple) or (value is float and key.endswith(".pos"))
    }
    action_features_hw = {
        key: value for key, value in robot.action_features.items() if key.endswith(".pos")
    }

    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=True,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")

    setup = RTCClientSetup(
        task=args.task,
        fps=args.fps,
        robot_type=args.robot_type,
        hw_features=hw_features,
        dataset_features=dataset_features,
        action_feature_names=list(action_features_hw.keys()),
        rename_map={},
    )
    return setup, list(action_features_hw.keys())


def _send_observation(stub, payload: TimedObservationPayload) -> None:
    data = pickle.dumps(payload)
    iterator = send_bytes_in_chunks(data, services_pb2.Observation, log_prefix="[RTC CLIENT] Observation", silent=True)
    stub.SendObservations(iterator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--robot-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="follower")
    parser.add_argument("--calibration-dir", default="config/calibration/robots/so_follower")
    parser.add_argument("--camera-name", default="camera1")
    parser.add_argument("--camera-index", default="0")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--task", default="SO101 teleoperation task")
    parser.add_argument("--refill-threshold", type=int, default=5)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means infinite.")
    parser.add_argument("--return-duration", type=float, default=3.0)
    parser.add_argument("--return-fps", type=int, default=50)
    parser.add_argument("--no-return-to-initial-position", action="store_true")
    return parser.parse_args()


def _load_runtime_imports() -> None:
    global grpc
    global OpenCVCameraConfig, aggregate_pipeline_dataset_features, create_initial_features
    global make_default_processors, make_robot_from_config, SO101FollowerConfig
    global services_pb2, services_pb2_grpc, grpc_channel_options, send_bytes_in_chunks
    global combine_feature_dicts, hw_to_dataset_features
    global register_third_party_plugins, precise_sleep, init_logging

    import grpc as _grpc

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig as _OpenCVCameraConfig
    from lerobot.datasets import aggregate_pipeline_dataset_features as _aggregate_pipeline_dataset_features, create_initial_features as _create_initial_features
    from lerobot.processor import make_default_processors as _make_default_processors
    from lerobot.robots import make_robot_from_config as _make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig as _SO101FollowerConfig
    from lerobot.transport import services_pb2 as _services_pb2, services_pb2_grpc as _services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options as _grpc_channel_options, send_bytes_in_chunks as _send_bytes_in_chunks
    from lerobot.utils.feature_utils import combine_feature_dicts as _combine_feature_dicts, hw_to_dataset_features as _hw_to_dataset_features
    from lerobot.utils.import_utils import register_third_party_plugins as _register_third_party_plugins
    from lerobot.utils.robot_utils import precise_sleep as _precise_sleep
    from lerobot.utils.utils import init_logging as _init_logging

    grpc = _grpc
    OpenCVCameraConfig = _OpenCVCameraConfig
    aggregate_pipeline_dataset_features = _aggregate_pipeline_dataset_features
    create_initial_features = _create_initial_features
    make_default_processors = _make_default_processors
    make_robot_from_config = _make_robot_from_config
    SO101FollowerConfig = _SO101FollowerConfig
    services_pb2 = _services_pb2
    services_pb2_grpc = _services_pb2_grpc
    grpc_channel_options = _grpc_channel_options
    send_bytes_in_chunks = _send_bytes_in_chunks
    combine_feature_dicts = _combine_feature_dicts
    hw_to_dataset_features = _hw_to_dataset_features
    register_third_party_plugins = _register_third_party_plugins
    precise_sleep = _precise_sleep
    init_logging = _init_logging



class RTCRefillWorker:
    """Background network/refill worker so the robot control loop stays non-blocking."""

    def __init__(self, stub, args: argparse.Namespace):
        self.stub = stub
        self.args = args
        self.shutdown = threading.Event()
        self.latest_obs_lock = threading.Lock()
        self.latest_obs: TimedObservationPayload | None = None
        self.queue_lock = threading.Lock()
        self.action_queue: deque[Any] = deque()
        self.ordered_keys: list[str] = []
        self.thread = threading.Thread(target=self._loop, daemon=True, name="RTCActionRefill")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.shutdown.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)

    def set_latest_observation(self, payload: TimedObservationPayload) -> None:
        with self.latest_obs_lock:
            self.latest_obs = payload

    def pop_action(self) -> tuple[Any, list[str]] | None:
        with self.queue_lock:
            if not self.action_queue:
                return None
            return self.action_queue.popleft(), list(self.ordered_keys)

    def qsize(self) -> int:
        with self.queue_lock:
            return len(self.action_queue)

    def _loop(self) -> None:
        while not self.shutdown.is_set():
            if self.qsize() > self.args.refill_threshold:
                time.sleep(_REFILL_IDLE_SLEEP_S)
                continue

            with self.latest_obs_lock:
                obs_payload = self.latest_obs
            if obs_payload is None:
                time.sleep(_REFILL_IDLE_SLEEP_S)
                continue

            try:
                _send_observation(self.stub, obs_payload)
                response = self.stub.GetActions(services_pb2.Empty())
                payload = pickle.loads(response.data)  # nosec: trusted local robotics process

                if isinstance(payload, ActionPayload):
                    actions = payload.actions if payload.actions is not None else []
                    if payload.action is not None:
                        actions = [payload.action, *actions]
                    ordered_keys = payload.ordered_action_keys or []
                    queue_size = payload.queue_size
                else:
                    actions = payload.get("actions", [])
                    single_action = payload.get("action")
                    if single_action is not None:
                        actions = [single_action, *actions]
                    ordered_keys = payload.get("ordered_action_keys", [])
                    queue_size = payload.get("queue_size", 0)

                if not actions:
                    time.sleep(_REFILL_IDLE_SLEEP_S)
                    continue

                with self.queue_lock:
                    self.ordered_keys = list(ordered_keys)
                    self.action_queue.extend(actions)
                    local_size = len(self.action_queue)

                LOGGER.debug(
                    "Received RTC actions: len=%d local_queue=%d server_queue=%s",
                    len(actions),
                    local_size,
                    queue_size,
                )
            except grpc.RpcError as exc:
                LOGGER.error("gRPC refill error: %s", exc)
                time.sleep(_REFILL_ERROR_SLEEP_S)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("RTC refill error: %s", exc)
                time.sleep(_REFILL_ERROR_SLEEP_S)


def main() -> None:
    args = parse_args()
    _load_runtime_imports()
    register_third_party_plugins()
    init_logging()

    robot = make_robot_from_config(_build_robot_config(args))
    channel = None
    refill_worker: RTCRefillWorker | None = None
    initial_pose: dict[str, float] = {}
    robot_action_processor = make_default_processors()[1]

    try:
        LOGGER.info("Connecting robot %s on %s", args.robot_type, args.robot_port)
        robot.connect()
        initial_pose = _extract_joint_pose(robot.get_observation())
        LOGGER.info("Captured initial robot position (%d joints)", len(initial_pose))

        setup, fallback_action_keys = _build_setup(robot, args)
        channel = grpc.insecure_channel(
            args.server_address,
            grpc_channel_options(initial_backoff=f"{1.0 / args.fps:.4f}s"),
        )
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        stub.Ready(services_pb2.Empty())
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(setup)))
        LOGGER.info("Connected to RTC policy server at %s", args.server_address)

        refill_worker = RTCRefillWorker(stub, args)
        refill_worker.start()

        start_time = time.perf_counter()
        timestep = 0
        control_dt = 1.0 / args.fps
        while True:
            loop_start = time.perf_counter()
            if args.duration > 0 and (loop_start - start_time) >= args.duration:
                LOGGER.info("Duration limit reached (%.1fs)", args.duration)
                break

            obs = robot.get_observation()
            refill_worker.set_latest_observation(
                TimedObservationPayload(
                    timestamp=time.time(),
                    timestep=timestep,
                    observation=obs,
                )
            )

            next_item = refill_worker.pop_action()
            if next_item is not None:
                action, ordered_keys = next_item
                if not ordered_keys:
                    ordered_keys = fallback_action_keys
                if len(action) != len(ordered_keys):
                    raise ValueError(f"Action dim {len(action)} does not match action keys {len(ordered_keys)}")
                action_dict = {key: action[i].item() for i, key in enumerate(ordered_keys)}
                processed = robot_action_processor((action_dict, obs))
                robot.send_action(processed)

            timestep += 1
            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(control_dt - elapsed, 0.0))
            if elapsed > control_dt:
                LOGGER.warning("Control loop slower than target FPS: %.1f Hz < %.1f Hz", 1.0 / elapsed, args.fps)

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        if refill_worker is not None:
            refill_worker.stop()
        if robot.is_connected:
            if not args.no_return_to_initial_position:
                try:
                    _return_to_initial_pose(robot, initial_pose, args.return_duration, args.return_fps)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Could not return robot to initial pose: %s", exc)
            LOGGER.info("Disconnecting robot")
            robot.disconnect()
        if channel is not None:
            channel.close()
        LOGGER.info("RTC robot client stopped")


if __name__ == "__main__":
    main()
