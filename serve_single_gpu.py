"""
Single-GPU inference server for DreamZero (GB10 DGX Spark).

The original socket_test_optimized_AR.py uses two GPUs to run classifier-free guidance
(CFG) in parallel via torch.distributed. This file is a simplified version that loads
the whole model on a single GPU and runs everything sequentially — no distributed
coordination, no worker loops.

Usage:
    python serve_single_gpu.py --model-path <path/to/checkpoint> --port 8000

Or with torchrun (single process):
    torchrun --nproc_per_node=1 serve_single_gpu.py --model-path <path/to/checkpoint> --port 8000
"""

import dataclasses
import logging
import os
import datetime
import asyncio

import imageio
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

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    port: int = 8000
    model_path: str = "./checkpoints/dreamzero"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None


def _init_single_gpu() -> None:
    """Initialize torch.distributed for a single GPU (required by GrootSimPolicy)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


class SingleGPUDroidPolicy:
    """
    Wraps GrootSimPolicy for single-GPU inference.

    Handles observation format conversion (roboarena -> AR_droid), frame accumulation,
    and action format conversion. All distributed coordination from the original
    ARDroidRoboarenaPolicy is removed.
    """

    FRAMES_PER_CHUNK = 4

    def __init__(self, groot_policy: GrootSimPolicy, output_dir: str | None = None) -> None:
        self._policy = groot_policy
        self._output_dir = output_dir

        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": [],
        }
        self._call_count = 0
        self._is_first_call = True
        self._current_session_id: str | None = None
        self.video_across_time: list[torch.Tensor] = []
        self._msg_index = 0

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _convert_observation(self, obs: dict) -> dict:
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }

        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        self._frame_buffers[droid_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        converted: dict = {}
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                converted[droid_key] = np.stack(frames_to_use, axis=0)

        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)

        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

        converted["annotation.language.action_text"] = obs.get("prompt", "")
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        joint_action = None
        gripper_action = None

        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value

        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)

        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)

        N = joint_action.shape[0]

        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)

        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

    def infer(self, obs: dict) -> np.ndarray:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info("Session changed '%s' -> '%s', resetting state",
                            self._current_session_id, session_id)
                self._reset_state()
            else:
                logger.info("New session: '%s'", session_id)
            self._current_session_id = session_id

        self._msg_index += 1
        self._call_count += 1

        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)

        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)

        self.video_across_time.append(video_pred)

        action_dict = {}
        for k in dir(result_batch.act):
            if k.startswith("action."):
                action_dict[k] = getattr(result_batch.act, k)

        if self._is_first_call:
            self._is_first_call = False

        return self._convert_action(action_dict)

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
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    def reset(self, reset_info: dict) -> None:
        self._reset_state(save_video=True)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    # PyTorch native SDPA — FlashAttention/TE may not support GB10 (cc 12.1)
    os.environ["ATTENTION_BACKEND"] = "torch"

    torch._dynamo.config.recompile_limit = 800

    _init_single_gpu()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    embodiment_tag = "oxe_droid"
    model_path = args.model_path

    logger.info("Loading DreamZero policy from %s", model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    parent_dir = os.path.dirname(model_path)
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_name = os.path.basename(model_path)
    output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Videos will be saved to: %s", output_dir)

    wrapper_policy = SingleGPUDroidPolicy(groot_policy=policy, output_dir=output_dir)

    server_config = PolicyServerConfig(
        image_resolution=(180, 320),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    server = WebsocketPolicyServer(
        policy=wrapper_policy,
        server_config=server_config,
        host="0.0.0.0",
        port=args.port,
    )
    logger.info("Server listening on 0.0.0.0:%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
