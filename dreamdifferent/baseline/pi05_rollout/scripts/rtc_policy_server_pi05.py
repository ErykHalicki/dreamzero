#!/usr/bin/env python
"""Custom LeRobot RTC policy server for pi05 real robot rollouts.

This intentionally mirrors the `lerobot-rollout --inference.type=rtc` action
production path rather than the official `lerobot.async_inference` chunking
client/server.  The gRPC service reuses LeRobot's existing AsyncInference proto
messages, but the pickled payloads are specific to this lightweight RTC server.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle  # nosec: local trusted robotics process payloads
import signal
import threading
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("rtc_policy_server")
_IDLE_SLEEP_S = 0.01
_MAX_CONSECUTIVE_ERRORS = 10
_ERROR_RETRY_DELAY_S = 0.5


@dataclass
class RTCClientSetup:
    """Runtime metadata sent by the robot client after it connects hardware."""

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


def _normalize_prev_actions_length(prev_actions: Any, target_steps: int) -> Any:
    """Pad/truncate RTC prefix actions exactly as LeRobot's RTC engine does."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded




def _receive_stream_bytes(request_iterator, log_prefix: str) -> bytes:
    """Receive a chunked LeRobot transport stream without version-specific kwargs."""
    chunks: list[bytes] = []
    for item in request_iterator:
        chunks.append(item.data)
        if item.transfer_state == services_pb2.TransferState.TRANSFER_END:
            return b"".join(chunks)
    if chunks:
        return b"".join(chunks)
    raise RuntimeError(f"{log_prefix} received an empty byte stream")

def _resolve_action_key_order(
    policy_action_names: list[str] | None,
    dataset_action_names: list[str],
) -> list[str]:
    """Same policy-vs-hardware action order resolution as LeRobot rollout."""
    if not policy_action_names:
        return dataset_action_names
    policy_action_names = list(policy_action_names)
    if len(policy_action_names) != len(dataset_action_names):
        LOGGER.warning(
            "policy.action_feature_names length (%d) != robot action dim (%d); using robot order",
            len(policy_action_names),
            len(dataset_action_names),
        )
        return dataset_action_names
    if set(dataset_action_names) != set(policy_action_names):
        LOGGER.warning("policy.action_feature_names keys do not match robot action keys; using robot order")
        return dataset_action_names
    return policy_action_names


class RTCPolicyServer:
    """gRPC service that owns policy loading, RTC inference, and the action queue."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
            prefix_attention_schedule=args.prefix_attention_schedule,
            debug=args.rtc_debug,
        )

        self.policy_path = args.policy_path
        self.policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        if hasattr(self.policy_cfg, "rtc_config"):
            self.policy_cfg.rtc_config = self.rtc_config
        policy_class = get_policy_class(self.policy_cfg.type)

        LOGGER.info("Loading policy: %s", self.policy_path)
        self.policy = policy_class.from_pretrained(self.policy_path, config=self.policy_cfg)
        self.policy.config.rtc_config = self.rtc_config
        if hasattr(self.policy, "init_rtc_processor"):
            self.policy.init_rtc_processor()
        self.policy.to(self.device)
        self.policy.eval()
        LOGGER.info("Policy loaded: type=%s device=%s", self.policy_cfg.type, self.device)

        self.setup: RTCClientSetup | None = None
        self.preprocessor = None
        self.postprocessor = None
        self.relative_step = None
        self.normalizer_step = None
        self.ordered_action_keys: list[str] = []

        self.action_queue = ActionQueue(self.rtc_config)
        self.obs_lock = threading.Lock()
        self.latest_obs: TimedObservationPayload | None = None
        self.ready_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.server_timestep = 0

    # ------------------------------------------------------------------
    # gRPC API
    # ------------------------------------------------------------------

    def Ready(self, request, context):  # noqa: N802
        LOGGER.info("Client connected: %s", context.peer())
        self.action_queue.clear()
        with self.obs_lock:
            self.latest_obs = None
        self.server_timestep = 0
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        setup = pickle.loads(request.data)  # nosec: trusted local robotics process
        if isinstance(setup, dict):
            setup = RTCClientSetup(**setup)
        if not isinstance(setup, RTCClientSetup):
            raise TypeError(f"Expected RTCClientSetup, got {type(setup)}")

        self.setup = setup
        self.ordered_action_keys = _resolve_action_key_order(
            getattr(self.policy.config, "action_feature_names", None),
            setup.action_feature_names,
        )

        LOGGER.info(
            "Received client setup: robot=%s fps=%.1f task=%r action_dim=%d",
            setup.robot_type,
            setup.fps,
            setup.task,
            len(self.ordered_action_keys),
        )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.policy_path,
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": setup.rename_map},
            },
        )

        self.relative_step = next(
            (s for s in self.preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self.normalizer_step = next(
            (s for s in self.preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self.relative_step is not None and self.relative_step.action_names is None:
            cfg_names = getattr(self.policy.config, "action_feature_names", None)
            self.relative_step.action_names = list(cfg_names) if cfg_names else list(setup.action_feature_names)
            LOGGER.info("Relative actions enabled; RTC prefix will be re-anchored")

        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.action_queue.clear()
        self.ready_event.set()
        self._ensure_thread()
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = _receive_stream_bytes(request_iterator, "[RTC SERVER] Observation")
        payload = pickle.loads(received_bytes)  # nosec: trusted local robotics process
        if isinstance(payload, dict):
            payload = TimedObservationPayload(**payload)
        if not isinstance(payload, TimedObservationPayload):
            raise TypeError(f"Expected TimedObservationPayload, got {type(payload)}")
        with self.obs_lock:
            self.latest_obs = payload
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        if not self.ready_event.is_set():
            payload = ActionPayload(actions=[], ordered_action_keys=[], queue_size=0, server_timestep=self.server_timestep)
            return services_pb2.Actions(data=pickle.dumps(payload))

        actions = []
        for _ in range(max(self.args.max_actions_per_response, 1)):
            action = self.action_queue.get()
            if action is None:
                break
            actions.append(action)
        payload = ActionPayload(
            actions=actions,
            ordered_action_keys=self.ordered_action_keys,
            queue_size=self.action_queue.qsize(),
            server_timestep=self.server_timestep,
        )
        return services_pb2.Actions(data=pickle.dumps(payload))

    # ------------------------------------------------------------------
    # RTC inference thread
    # ------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._rtc_loop, daemon=True, name="RTCPolicyServerLoop")
        self.thread.start()
        LOGGER.info("RTC policy server inference thread started")

    def _rtc_loop(self) -> None:
        latency_tracker = LatencyTracker()
        consecutive_errors = 0
        while not self.shutdown_event.is_set():
            if not self.ready_event.is_set() or self.setup is None:
                time.sleep(_IDLE_SLEEP_S)
                continue

            with self.obs_lock:
                obs_payload = self.latest_obs
            if obs_payload is None:
                time.sleep(_IDLE_SLEEP_S)
                continue

            if self.action_queue.qsize() > self.args.queue_threshold:
                time.sleep(_IDLE_SLEEP_S)
                continue

            try:
                self._run_one_rtc_inference(obs_payload, latency_tracker)
                consecutive_errors = 0
            except Exception as exc:  # noqa: BLE001 - robotics loop should log and retry transient failures
                consecutive_errors += 1
                LOGGER.error("RTC inference error (%d/%d): %s", consecutive_errors, _MAX_CONSECUTIVE_ERRORS, exc)
                LOGGER.debug(traceback.format_exc())
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    LOGGER.error("Stopping RTC server after persistent inference failures")
                    self.shutdown_event.set()
                    break
                time.sleep(_ERROR_RETRY_DELAY_S)

    def _run_one_rtc_inference(
        self,
        obs_payload: TimedObservationPayload,
        latency_tracker: LatencyTracker,
    ) -> None:
        assert self.setup is not None
        assert self.preprocessor is not None
        assert self.postprocessor is not None

        current_time = time.perf_counter()
        idx_before = self.action_queue.get_action_index()
        prev_actions = self.action_queue.get_left_over()

        time_per_step = 1.0 / self.setup.fps
        latency = latency_tracker.max()
        delay = math.ceil(latency / time_per_step) if latency else 0

        obs_batch = build_dataset_frame(self.setup.hw_features, obs_payload.observation, prefix="observation")
        obs_batch = prepare_observation_for_inference(
            obs_batch,
            self.device,
            self.setup.task,
            self.setup.robot_type,
        )
        obs_batch["task"] = [self.setup.task]
        preprocessed = self.preprocessor(obs_batch)

        if prev_actions is not None and self.relative_step is not None:
            raw_state = self.relative_step.get_cached_state()
            if raw_state is not None:
                prev_abs = self.action_queue.get_processed_left_over()
                if prev_abs is not None and prev_abs.numel() > 0:
                    prev_actions = reanchor_relative_rtc_prefix(
                        prev_actions_absolute=prev_abs,
                        current_state=raw_state,
                        relative_step=self.relative_step,
                        normalizer_step=self.normalizer_step,
                        policy_device=self.device,
                    )

        if prev_actions is not None:
            prev_actions = _normalize_prev_actions_length(prev_actions, self.rtc_config.execution_horizon)

        # Do not wrap RTC inference in torch.no_grad()/inference_mode():
        # LeRobot RTC computes prefix guidance with torch.autograd.grad inside
        # RTCProcessor.denoise_step. The policy method itself may be decorated
        # with no_grad(), so we explicitly re-enable grad around the call.
        with torch.enable_grad():
            actions = self.policy.predict_action_chunk(
                preprocessed,
                inference_delay=delay,
                prev_chunk_left_over=prev_actions,
            )
        original = actions.squeeze(0).detach().clone()
        processed = self.postprocessor(actions.detach()).squeeze(0)

        new_latency = time.perf_counter() - current_time
        real_delay = math.ceil(new_latency / time_per_step)
        latency_tracker.add(new_latency)

        self.action_queue.merge(original, processed, real_delay, idx_before)
        self.server_timestep += 1
        LOGGER.debug(
            "RTC chunk ready: obs_ts=%d latency=%.3fs delay=%d queue=%d",
            obs_payload.timestep,
            new_latency,
            real_delay,
            self.action_queue.qsize(),
        )

    def stop(self) -> None:
        LOGGER.info("Stopping RTC policy server")
        self.shutdown_event.set()
        self.ready_event.clear()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policy-path", required=True, help="Path or Hub id of the trained pi05 checkpoint.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS before client setup arrives.")
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--prefix-attention-schedule", default="linear")
    parser.add_argument("--queue-threshold", type=int, default=30)
    parser.add_argument("--max-actions-per-response", type=int, default=10)
    parser.add_argument("--rtc-debug", action="store_true")
    return parser.parse_args()


def _load_runtime_imports() -> None:
    global grpc, torch
    global PreTrainedConfig, RTCAttentionSchedule
    global get_policy_class, make_pre_post_processors
    global ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix, RTCConfig
    global prepare_observation_for_inference
    global NormalizerProcessorStep, RelativeActionsProcessorStep
    global services_pb2, services_pb2_grpc, receive_bytes_in_chunks
    global build_dataset_frame, register_third_party_plugins, init_logging

    import grpc as _grpc
    import torch as _torch

    from lerobot.configs import PreTrainedConfig as _PreTrainedConfig, RTCAttentionSchedule as _RTCAttentionSchedule
    from lerobot.policies import get_policy_class as _get_policy_class, make_pre_post_processors as _make_pre_post_processors
    from lerobot.policies.rtc import ActionQueue as _ActionQueue, LatencyTracker as _LatencyTracker, reanchor_relative_rtc_prefix as _reanchor_relative_rtc_prefix
    from lerobot.policies.rtc.configuration_rtc import RTCConfig as _RTCConfig
    from lerobot.policies.utils import prepare_observation_for_inference as _prepare_observation_for_inference
    from lerobot.processor import NormalizerProcessorStep as _NormalizerProcessorStep, RelativeActionsProcessorStep as _RelativeActionsProcessorStep
    from lerobot.transport import services_pb2 as _services_pb2, services_pb2_grpc as _services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks as _receive_bytes_in_chunks
    from lerobot.utils.feature_utils import build_dataset_frame as _build_dataset_frame
    from lerobot.utils.import_utils import register_third_party_plugins as _register_third_party_plugins
    from lerobot.utils.utils import init_logging as _init_logging

    grpc = _grpc
    torch = _torch
    PreTrainedConfig = _PreTrainedConfig
    RTCAttentionSchedule = _RTCAttentionSchedule
    get_policy_class = _get_policy_class
    make_pre_post_processors = _make_pre_post_processors
    ActionQueue = _ActionQueue
    LatencyTracker = _LatencyTracker
    reanchor_relative_rtc_prefix = _reanchor_relative_rtc_prefix
    RTCConfig = _RTCConfig
    prepare_observation_for_inference = _prepare_observation_for_inference
    NormalizerProcessorStep = _NormalizerProcessorStep
    RelativeActionsProcessorStep = _RelativeActionsProcessorStep
    services_pb2 = _services_pb2
    services_pb2_grpc = _services_pb2_grpc
    receive_bytes_in_chunks = _receive_bytes_in_chunks
    build_dataset_frame = _build_dataset_frame
    register_third_party_plugins = _register_third_party_plugins
    init_logging = _init_logging


def _parse_schedule_after_import(value: str):
    normalized = value.upper()
    try:
        return RTCAttentionSchedule[normalized]
    except KeyError as exc:
        choices = ", ".join(member.name.lower() for member in RTCAttentionSchedule)
        raise ValueError(f"Unknown RTC schedule '{value}'. Choices: {choices}") from exc


def main() -> None:
    args = parse_args()
    _load_runtime_imports()
    register_third_party_plugins()
    init_logging()
    args.prefix_attention_schedule = _parse_schedule_after_import(args.prefix_attention_schedule)

    service = RTCPolicyServer(args)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    server.add_insecure_port(f"{args.host}:{args.port}")

    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: ARG001
        LOGGER.info("Received signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    server.start()
    LOGGER.info(
        "RTC policy server listening on %s:%d (policy=%s, horizon=%d)",
        args.host,
        args.port,
        args.policy_path,
        args.execution_horizon,
    )
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        service.stop()
        server.stop(grace=1.0)
        LOGGER.info("RTC policy server stopped")


if __name__ == "__main__":
    main()
