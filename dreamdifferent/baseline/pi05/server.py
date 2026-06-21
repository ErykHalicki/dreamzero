#!/usr/bin/env python
"""Persistent LeRobot async policy server for Pi05 LoRA checkpoints."""

from __future__ import annotations

import argparse
import logging
import pickle  # nosec - LeRobot async transport uses pickled local objects.
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2, services_pb2_grpc


DEFAULT_CHECKPOINT = (
    "/cluster/scratch/dohkim/pi05/experiments/pi05_homogeneous_lora_a100/"
    "outputs/checkpoints/025000/pretrained_model"
)


def _normalize_checkpoint(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _load_policy(policy_type: str, checkpoint: str, device: str):
    policy_class = get_policy_class(policy_type)
    adapter_config_path = Path(checkpoint) / "adapter_config.json"

    if adapter_config_path.exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(checkpoint)
        base_model_path = peft_config.base_model_name_or_path
        if not base_model_path:
            raise ValueError(f"No base_model_name_or_path in PEFT adapter config: {adapter_config_path}")

        logging.info("Loading PEFT adapter from %s on base %s", checkpoint, base_model_path)
        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        policy = policy_class.from_pretrained(base_model_path, config=policy_config)
        policy = PeftModel.from_pretrained(policy, checkpoint, config=peft_config, is_trainable=False)
    else:
        logging.info("Loading full policy checkpoint from %s", checkpoint)
        policy = policy_class.from_pretrained(checkpoint)

    policy.to(device)
    policy.eval()
    return policy


class PersistentPolicyServer(PolicyServer):
    """PolicyServer variant that keeps one loaded policy resident across clients."""

    def __init__(
        self,
        config: PolicyServerConfig,
        checkpoint: str,
        policy_type: str,
        device: str,
        actions_per_chunk: int,
    ):
        super().__init__(config)
        self.loaded_checkpoint = _normalize_checkpoint(checkpoint)
        self.loaded_policy_type = policy_type
        self.loaded_device = device
        self.loaded_actions_per_chunk = actions_per_chunk
        self._processor_rename_map: dict[str, str] = {}

        start = time.perf_counter()
        self.policy = _load_policy(policy_type, self.loaded_checkpoint, device)
        self.policy_type = policy_type
        self.device = device
        self.actions_per_chunk = actions_per_chunk
        self.preprocessor, self.postprocessor = self._make_processors(rename_map={})
        self.logger.info("Persistent policy loaded in %.4f seconds", time.perf_counter() - start)

    def _make_processors(self, rename_map: dict[str, str]):
        device_override = {"device": self.loaded_device}
        return make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.loaded_checkpoint,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

    def _validate_policy_specs(self, policy_specs: RemotePolicyConfig) -> None:
        requested_checkpoint = _normalize_checkpoint(policy_specs.pretrained_name_or_path)
        if requested_checkpoint != self.loaded_checkpoint:
            raise ValueError(
                "Persistent server is already running a different checkpoint. "
                f"loaded={self.loaded_checkpoint} requested={requested_checkpoint}. "
                "Restart the server to use another checkpoint."
            )

        if policy_specs.policy_type != self.loaded_policy_type:
            raise ValueError(
                f"Policy type mismatch: loaded={self.loaded_policy_type} requested={policy_specs.policy_type}"
            )

        if policy_specs.device != self.loaded_device:
            raise ValueError(f"Policy device mismatch: loaded={self.loaded_device} requested={policy_specs.device}")

        if policy_specs.actions_per_chunk != self.loaded_actions_per_chunk:
            raise ValueError(
                "actions_per_chunk mismatch: "
                f"loaded={self.loaded_actions_per_chunk} requested={policy_specs.actions_per_chunk}"
            )

        required_features = set(self.policy.config.input_features)
        client_features = set(policy_specs.lerobot_features)
        missing = sorted(required_features - client_features)
        if missing:
            raise ValueError(f"Client observation features are missing policy inputs: {missing}")

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            "Receiving policy instructions from %s | Policy type: %s | Pretrained name or path: %s | "
            "Actions per chunk: %s | Device: %s",
            client_id,
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            policy_specs.actions_per_chunk,
            policy_specs.device,
        )
        self._validate_policy_specs(policy_specs)

        self._reset_server()
        self.shutdown_event.clear()
        self.fps_tracker.reset()
        self.observation_queue = Queue(maxsize=1)
        self.last_processed_obs = None
        self.lerobot_features = policy_specs.lerobot_features

        if policy_specs.rename_map != self._processor_rename_map:
            self.logger.info("Rebuilding processors for rename_map=%s", policy_specs.rename_map)
            self.preprocessor, self.postprocessor = self._make_processors(policy_specs.rename_map)
            self._processor_rename_map = dict(policy_specs.rename_map)
        else:
            self.logger.info("Reusing loaded policy and processors")

        return services_pb2.Empty()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--policy-type", default="pi05")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--inference-latency", type=float, default=1 / 30)
    parser.add_argument("--obs-queue-timeout", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s:%(lineno)d %(message)s")

    checkpoint = _normalize_checkpoint(args.checkpoint)
    if not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    logging.info(pformat(asdict(cfg)))

    policy_server = PersistentPolicyServer(
        cfg,
        checkpoint=checkpoint,
        policy_type=args.policy_type,
        device=args.policy_device,
        actions_per_chunk=args.actions_per_chunk,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info("Persistent PolicyServer started on %s:%s", cfg.host, cfg.port)
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    main()
