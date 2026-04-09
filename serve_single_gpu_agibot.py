"""
Single-GPU inference server for DreamZero — AgiBot G1 embodiment.

AgiBot G1 is a bimanual robot (7 DOF per arm) with 3 cameras:
  top_head, hand_left, hand_right.

Client observation keys (sent by AgiBot client):
  observation/top_head                 -> video.top_head
  observation/hand_left                -> video.hand_left
  observation/hand_right               -> video.hand_right
  observation/left_arm_joint_position  -> state.left_arm_joint_position  (7,)
  observation/right_arm_joint_position -> state.right_arm_joint_position (7,)
  observation/left_effector_position   -> state.left_effector_position   (1,)
  observation/right_effector_position  -> state.right_effector_position  (1,)
  observation/head_position            -> state.head_position            (2,)
  observation/waist_pitch              -> state.waist_pitch              (1,)
  observation/waist_lift               -> state.waist_lift               (1,)
  prompt                               -> annotation.language.action_text
  session_id                           -> session tracking

Server action output: np.ndarray (N, 22) in order:
  left_arm(7), right_arm(7), left_effector(1), right_effector(1),
  head(2), waist(2), robot_velocity(2)

Usage:
    bash scripts/inference/serve_single_gpu_agibot.sh --model-path <path/to/checkpoint>
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


def _get_policy_input_resolution(policy: GrootSimPolicy) -> tuple[int, int] | None:
    """Return the raw video input resolution expected by the policy as (H, W)."""
    eval_transform = getattr(policy, "eval_transform", None)
    transforms = getattr(eval_transform, "transforms", []) if eval_transform is not None else []
    for transform in transforms:
        original_resolutions = getattr(transform, "original_resolutions", None)
        if not original_resolutions:
            continue
        first_resolution = next(iter(original_resolutions.values()), None)
        if first_resolution is None:
            continue
        width, height = first_resolution
        return int(height), int(width)
    return None


# Action keys in the order they are concatenated in the output array.
# Dimensions: left_arm(7), right_arm(7), left_effector(1), right_effector(1),
#             head(2), waist(2), robot_velocity(2) = 22 total.
_ACTION_KEY_ORDER = [
    "action.left_arm_joint_position",
    "action.right_arm_joint_position",
    "action.left_effector_position",
    "action.right_effector_position",
    "action.head_position",
    "action.waist_pitch",
    "action.waist_lift",
    "action.robot_velocity",
]

_STATE_KEY_MAPPING = {
    "observation/left_arm_joint_position":  "state.left_arm_joint_position",
    "observation/right_arm_joint_position": "state.right_arm_joint_position",
    "observation/left_effector_position":   "state.left_effector_position",
    "observation/right_effector_position":  "state.right_effector_position",
    "observation/head_position":            "state.head_position",
    "observation/waist_pitch":              "state.waist_pitch",
    "observation/waist_lift":               "state.waist_lift",
}


class SingleGPUAgibotPolicy:
    """
    Wraps GrootSimPolicy for single-GPU AgiBot G1 inference.

    Handles observation format conversion (client keys -> model keys),
    frame accumulation, and action format conversion.
    """

    FRAMES_PER_CHUNK = 4

    def __init__(self, groot_policy: GrootSimPolicy, output_dir: str | None = None) -> None:
        self._policy = groot_policy
        self._output_dir = output_dir

        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.top_head":   [],
            "video.hand_left":  [],
            "video.hand_right": [],
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
            "observation/top_head":  "video.top_head",
            "observation/hand_left": "video.hand_left",
            "observation/hand_right": "video.hand_right",
        }

        for client_key, model_key in image_key_mapping.items():
            if client_key in obs:
                data = obs[client_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        self._frame_buffers[model_key].extend(list(data))
                    else:
                        self._frame_buffers[model_key].append(data)

        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK

        converted: dict = {}
        for model_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    frames_to_use = buffer[-num_frames:]
                else:
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                converted[model_key] = np.stack(frames_to_use, axis=0)

        for client_key, model_key in _STATE_KEY_MAPPING.items():
            if client_key in obs:
                val = obs[client_key]
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    val = val.reshape(1, -1)
                converted[model_key] = val.astype(np.float64)

        converted["annotation.language.action_text"] = obs.get("prompt", "")
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        parts = []
        for key in _ACTION_KEY_ORDER:
            val = action_dict.get(key)
            if val is None:
                logger.warning("Action key '%s' missing from model output", key)
                return np.zeros((1, 22), dtype=np.float32)
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            # Normalize to 2D (T, dim): squeeze batch dim if 3D, expand if 0D/1D.
            if val.ndim == 3:
                val = val.squeeze(0)   # (1, T, dim) -> (T, dim)
            elif val.ndim == 1:
                val = val.reshape(1, -1)
            elif val.ndim == 0:
                val = val.reshape(1, 1)
            parts.append(val)

        # Some keys may predict only 1 step while others predict the full horizon.
        # Tile any T=1 parts to match the maximum T found across all parts.
        max_t = max(p.shape[0] for p in parts)
        if max_t > 1:
            parts = [
                np.tile(p, (max_t, 1)) if p.shape[0] == 1 else p
                for p in parts
            ]

        return np.concatenate(parts, axis=-1).astype(np.float32)

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
    os.environ["ATTENTION_BACKEND"] = os.environ.get("ATTENTION_BACKEND", "torch")

    torch._dynamo.config.disable = True

    _init_single_gpu()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    embodiment_tag = "agibot"
    model_path = args.model_path

    logger.info("Loading DreamZero AgiBot policy from %s", model_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        tokenizer_path_override="google/umt5-xxl",
    )

    parent_dir = os.path.dirname(model_path)
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    checkpoint_name = os.path.basename(model_path)
    output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Videos will be saved to: %s", output_dir)

    wrapper_policy = SingleGPUAgibotPolicy(groot_policy=policy, output_dir=output_dir)

    input_resolution = _get_policy_input_resolution(policy)
    logger.info("DreamZero AgiBot raw input resolution: %s", input_resolution)

    server_config = PolicyServerConfig(
        image_resolution=input_resolution,
        needs_wrist_camera=True,   # top_head camera
        n_external_cameras=2,      # hand_left, hand_right
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
