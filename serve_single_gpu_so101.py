"""
Single-GPU inference server for DreamZero SO101 direct robot control.

This copy speaks the msgpack websocket protocol used by
dreamdifferent/so101/scripts/openpi_policy_client.py. The original DROID/Isaac
serving path stays in serve_single_gpu.py.
"""

import dataclasses
import logging
import os
import datetime
import asyncio
import functools
import time
import traceback
from typing import Any

import imageio
import msgpack
import numpy as np
import torch
# Force early CUDA initialization and pre-import CUDA extension modules that
# use TLS (Thread Local Storage). On ARM64 (GB10/DGX Spark), these libraries
# are loaded lazily via dlopen during model instantiation, which overflows the
# TLS slot table and triggers: ld.so assertion 'idx == 0' failed.
# Pre-importing here registers their TLS slots before the table fills up.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
try:
    import flash_attn  # noqa: F401
except ImportError:
    pass
try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    pass
try:
    import transformer_engine  # noqa: F401
except (ImportError, OSError):
    pass
import groot.vla.model.dreamzero.modules.wan2_1_attention  # noqa: F401
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import tyro
from einops import rearrange
from tianshou.data import Batch
import websockets.asyncio.server
import websockets.frames

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag

logger = logging.getLogger(__name__)

ACTION_DIM = 6
DEFAULT_ACTION_HORIZON = 24
SO101_METADATA = {
    "protocol": "dreamzero_so101_v1",
    "embodiment_tag": "so101",
    "action_dim": ACTION_DIM,
    "action_horizon": DEFAULT_ACTION_HORIZON,
}


@dataclasses.dataclass
class Args:
    port: int = 23261
    model_path: str = "./checkpoints/dreamzero_so101"
    tokenizer_path: str = "/tmp/pretrained_checkpoints/umt5-xxl"
    enable_dit_cache: bool = False
    index: int = 0
    action_horizon: int = DEFAULT_ACTION_HORIZON
    host: str = "0.0.0.0"


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


def _dict_get(mapping: dict, key: str, default=None):
    if key in mapping:
        return mapping[key]
    key_bytes = key.encode("utf-8")
    if key_bytes in mapping:
        return mapping[key_bytes]
    return default


def _init_single_gpu() -> None:
    """Initialize torch.distributed for a single GPU (required by GrootSimPolicy)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _strip_batch_dim(value: np.ndarray) -> np.ndarray:
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    return value


def _batch_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        items = value.items()
    elif hasattr(value, "__dict__"):
        items = value.__dict__.items()
    else:
        return {}

    flattened: dict[str, Any] = {}
    for key, item in items:
        key = str(key)
        if isinstance(item, (dict, Batch)) and key == "action":
            for subkey, subitem in _batch_to_dict(item).items():
                flattened[f"action.{subkey}"] = subitem
        else:
            flattened[key] = item
    return flattened


class SingleGPUSO101Policy:
    """Wraps GrootSimPolicy for SO101 robot inference."""

    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        output_dir: str | None = None,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
    ) -> None:
        self._policy = groot_policy
        self._output_dir = output_dir
        self._action_horizon = action_horizon
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.front": [],
            "video.wrist": [],
        }
        self._is_first_call = True
        self.video_across_time: list[torch.Tensor] = []

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _convert_observation(self, obs: dict) -> dict:
        image_key_mapping = {
            "observation/images/front": "video.front",
            "observation/images/wrist": "video.wrist",
        }

        for client_key, model_key in image_key_mapping.items():
            data = _dict_get(obs, client_key)
            if isinstance(data, np.ndarray):
                if data.ndim == 4:
                    self._frame_buffers[model_key].extend(list(data))
                else:
                    self._frame_buffers[model_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        converted: dict = {}
        for model_key, buffer in self._frame_buffers.items():
            if not buffer:
                raise KeyError(f"Missing required SO101 camera observation for {model_key}")
            if len(buffer) >= num_frames:
                frames_to_use = buffer[-num_frames:]
            else:
                frames_to_use = buffer.copy()
                while len(frames_to_use) < num_frames:
                    frames_to_use.insert(0, buffer[0])
            converted[model_key] = np.stack(frames_to_use, axis=0)

        state = _dict_get(obs, "observation/state")
        if state is None:
            raise KeyError("Missing required SO101 observation/state")
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape[0] != ACTION_DIM:
            raise ValueError(f"Expected SO101 state shape ({ACTION_DIM},), got {state.shape}")

        converted["state.joint_pos"] = state[:5].reshape(1, 5)
        converted["state.gripper_pos"] = state[5:].reshape(1, 1)
        converted["annotation.task"] = _dict_get(obs, "prompt", "")
        return converted

    def _convert_action(self, action_batch: Any) -> np.ndarray:
        action_dict = _batch_to_dict(action_batch)
        joint_action = action_dict.get("action.joint_pos")
        gripper_action = action_dict.get("action.gripper_pos")
        if joint_action is None:
            raise KeyError(f"Model output missing action.joint_pos; available keys={sorted(action_dict.keys())}")
        if gripper_action is None:
            raise KeyError(f"Model output missing action.gripper_pos; available keys={sorted(action_dict.keys())}")

        joint_action = _strip_batch_dim(_as_numpy(joint_action))
        gripper_action = _strip_batch_dim(_as_numpy(gripper_action))
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        if gripper_action.ndim == 1:
            gripper_action = gripper_action.reshape(-1, 1)
        elif gripper_action.ndim == 0:
            gripper_action = gripper_action.reshape(1, 1)

        if joint_action.shape[-1] != 5:
            raise ValueError(f"Expected action.joint_pos last dim 5, got {joint_action.shape}")
        if gripper_action.shape[-1] != 1:
            raise ValueError(f"Expected action.gripper_pos last dim 1, got {gripper_action.shape}")
        if joint_action.shape[0] != gripper_action.shape[0]:
            raise ValueError(f"Action horizon mismatch: joint={joint_action.shape}, gripper={gripper_action.shape}")

        actions = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        expected_shape = (self._action_horizon, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected SO101 actions shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Model returned non-finite SO101 action values")
        return actions

    def infer(self, obs: dict) -> dict[str, Any]:
        start = time.perf_counter()
        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)

        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)

        self.video_across_time.append(video_pred)

        if self._is_first_call:
            self._is_first_call = False

        actions = self._convert_action(result_batch.act)
        return {
            "actions": actions,
            "server_timing": {"infer_ms": (time.perf_counter() - start) * 1000.0},
        }

    def _reset_state(self, save_video: bool = True) -> None:
        if save_video and self.video_across_time and self._output_dir:
            try:
                video_cat = torch.cat(self.video_across_time, dim=2)
                frames = self._policy.trained_model.action_head.vae.decode(
                    video_cat,
                    tiled=self._policy.trained_model.action_head.tiled,
                    tile_size=(
                        self._policy.trained_model.action_head.tile_size_height,
                        self._policy.trained_model.action_head.tile_size_width,
                    ),
                    tile_stride=(
                        self._policy.trained_model.action_head.tile_stride_height,
                        self._policy.trained_model.action_head.tile_stride_width,
                    ),
                )
                frames = rearrange(frames, "B C T H W -> B T H W C")[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                frame_list = list(frames)
                if frame_list and len(frame_list[0].shape) == 3 and frame_list[0].shape[2] in [1, 3, 4]:
                    os.makedirs(self._output_dir, exist_ok=True)
                    existing = [f for f in os.listdir(self._output_dir) if f.endswith(".mp4")]
                    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                    n = (len(frame_list) - 1) // 8
                    output_path = os.path.join(self._output_dir, f"{len(existing):06}_{timestamp}_n{n}.mp4")
                    imageio.mimsave(output_path, frame_list, fps=5, codec="libx264")
                    logger.info("Saved video to: %s", output_path)
            except Exception as e:
                logger.warning("Failed to save video on reset: %s", e)

        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self.video_across_time = []

    def reset(self) -> None:
        self._reset_state(save_video=True)


class SO101WebsocketServer:
    def __init__(self, policy: SingleGPUSO101Policy, host: str, port: int, action_horizon: int) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = dict(SO101_METADATA)
        self._metadata["action_horizon"] = action_horizon
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logger.info("SO101 connection from %s opened", websocket.remote_address)
        await websocket.send(_packb(self._metadata))

        while True:
            try:
                obs = _unpackb(await websocket.recv())
                endpoint = _dict_get(obs, "endpoint")
                if endpoint == "reset":
                    self._policy.reset()
                    await websocket.send("reset successful")
                    continue

                await websocket.send(_packb(self._policy.infer(obs)))
            except websockets.ConnectionClosed:
                logger.info("SO101 connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    # PyTorch native SDPA — FlashAttention/TE may not support GB10 (cc 12.1)
    os.environ["ATTENTION_BACKEND"] = "torch"

    torch._dynamo.config.recompile_limit = 800

    _init_single_gpu()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    model_path = args.model_path

    logger.info("Loading DreamZero SO101 policy from %s", model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("so101"),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        tokenizer_path_override=args.tokenizer_path,
    )

    parent_dir = os.path.dirname(model_path)
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_name = os.path.basename(model_path.rstrip("/"))
    output_dir = os.path.join(parent_dir, f"real_world_eval_gen_so101_{date_suffix}_{args.index}", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Videos will be saved to: %s", output_dir)

    wrapper_policy = SingleGPUSO101Policy(
        groot_policy=policy,
        output_dir=output_dir,
        action_horizon=args.action_horizon,
    )
    server = SO101WebsocketServer(
        policy=wrapper_policy,
        host=args.host,
        port=args.port,
        action_horizon=args.action_horizon,
    )
    logger.info("SO101 server listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
