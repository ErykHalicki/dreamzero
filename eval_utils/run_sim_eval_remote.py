"""
Run DROID policy evaluation against a remote policy server (e.g. Cloudflare tunnel).

Usage:

    python run_sim_eval_remote.py --url https://xxx.trycloudflare.com --episodes 10 --scene 1 --headless
"""

import json
import os
import time
import uuid
import logging

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
import websockets.sync.client
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from openpi_client import image_tools, msgpack_numpy
from sim_evals.inference.abstract_client import InferenceClient

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class RemoteWebsocketClient:
    """WebSocket client that supports both ws:// and wss:// (Cloudflare tunnel) URLs."""

    def __init__(self, url: str) -> None:
        self._packer = msgpack_numpy.Packer()

        # Normalize URL to wss:// for https, ws:// for others
        url = url.rstrip("/")
        if url.startswith("https://"):
            self._uri = "wss://" + url[len("https://"):]
        elif url.startswith("http://"):
            self._uri = "ws://" + url[len("http://"):]
        elif not url.startswith("ws://") and not url.startswith("wss://"):
            self._uri = "wss://" + url
        else:
            self._uri = url

        logging.info(f"Connecting to {self._uri}...")
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())

    def get_server_metadata(self):
        return self._server_metadata

    def infer(self, obs: dict) -> dict:
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: dict) -> None:
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        self._ws.send(data)
        self._ws.recv()


class DreamZeroJointPosClient(InferenceClient):
    def __init__(self,
                url: str,
                open_loop_horizon: int = 8,
                debug_dump_dir: str | None = None,
    ) -> None:
        self.client = RemoteWebsocketClient(url)
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())
        self.infer_cnt = 0
        debug_root = debug_dump_dir or os.environ.get("DREAMZERO_DEBUG_DIR")
        self.debug_dump_dir = Path(debug_root) if debug_root else None
        if self.debug_dump_dir is not None:
            self.debug_dump_dir.mkdir(parents=True, exist_ok=True)
            print(f"DreamZero debug dump dir: {self.debug_dump_dir}")

    def _dump_request_images(self, request_data: dict, instruction: str) -> None:
        if self.debug_dump_dir is None:
            return
        idx = self.infer_cnt
        image_key_to_name = {
            "observation/exterior_image_0_left": "right",
            "observation/exterior_image_1_left": "left",
            "observation/wrist_image_left": "wrist",
        }
        file_prefixes = {
            image_name: f"franka_remote_infer_{image_name}_{idx:04d}"
            for image_name in image_key_to_name.values()
        }
        for payload_key, image_name in image_key_to_name.items():
            image = request_data.get(payload_key)
            if image is None:
                continue
            cv2.imwrite(
                str(self.debug_dump_dir / f"{file_prefixes[image_name]}.png"),
                cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
            )
        with open(self.debug_dump_dir / "franka_remote_meta.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "infer_cnt": idx,
                        "wall_time": time.time(),
                        "prompt": instruction,
                        "session_id": self.session_id,
                        "file_prefixes": file_prefixes,
                    }
                )
                + "\n"
            )

    def visualize(self, request: dict):
        curr_obs = self._extract_observation(request)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        combined = np.concatenate([right_img, wrist_img, left_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"], 180, 320),
                "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
                "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }
            for k, v in request_data.items():
                print(f"{k}: {v.shape if not isinstance(v, str) else v}")

            self._dump_request_images(request_data, instruction)
            result = self.client.infer(request_data)
            self.infer_cnt += 1
            actions = result["actions"] if isinstance(result, dict) else result
            assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
            assert actions.shape[-1] == 8, f"Expected 8 action dimensions (7 joints + 1 gripper), got {actions.shape[-1]}"
            self.pred_action_chunk = actions

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        img3 = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        both = np.concatenate([img1, img2, img3], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


def main(
        episodes: int = 10,
        scene: int = 1,
        headless: bool = True,
        url: str = "https://example.trycloudflare.com",
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "pick up the can and put it in the mug"
        case 3:
            instruction = "put the banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    client = DreamZeroJointPosClient(url=url)
    print(f"Server metadata: {client.client.get_server_metadata()}")

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
