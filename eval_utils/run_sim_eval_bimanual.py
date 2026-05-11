"""
Run DreamZero DROID evaluation in a dual-Franka Isaac scene while keeping
policy I/O right-arm only.

This mirrors run_sim_eval.py except it loads the DROID_BIMANUAL_RIGHT_ONLY
environment, which adds a passive left Franka and a left wrist camera.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import numpy as np
import torch
import tyro
from tqdm import tqdm

from openpi_client import image_tools

try:
    from run_sim_eval import DreamZeroJointPosClient, _show_viz_frame
except ImportError:
    from .run_sim_eval import DreamZeroJointPosClient, _show_viz_frame


class DreamZeroJointPosBimanualClient(DreamZeroJointPosClient):
    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        curr_obs = super()._extract_observation(obs_dict, save_to_disk=save_to_disk)
        # In the Isaac scene, external_cam sits at +Y (left-arm side) and
        # external_cam_2 sits at -Y (right-arm side). The base client inherited
        # legacy right/left_image names, so we add explicit aliases here to avoid
        # confusion in debug dumps.
        curr_obs["left_exterior_image"] = curr_obs["right_image"]
        curr_obs["right_exterior_image"] = curr_obs["left_image"]
        curr_obs["left_wrist_image"] = (
            obs_dict["policy"]["left_wrist_cam"][0].clone().detach().cpu().numpy()
        )
        return curr_obs

    def _dump_request_images(self, request_data: dict, instruction: str) -> None:
        if self.debug_dump_dir is None:
            return

        idx = self.infer_cnt
        curr_obs = getattr(self, "_debug_curr_obs", None)
        if curr_obs is None:
            return

        image_map = {
            "exterior_left_arm_side": curr_obs["left_exterior_image"],
            "exterior_right_arm_side": curr_obs["right_exterior_image"],
            "wrist_right_arm": curr_obs["wrist_image"],
            "wrist_left_arm": curr_obs["left_wrist_image"],
        }
        file_prefixes = {
            name: f"franka_bimanual_infer_{name}_{idx:04d}"
            for name in image_map
        }

        for name, image in image_map.items():
            cv2.imwrite(
                str(self.debug_dump_dir / f"{file_prefixes[name]}.png"),
                cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
            )

        with open(self.debug_dump_dir / "franka_bimanual_meta.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "infer_cnt": idx,
                        "wall_time": time.time(),
                        "prompt": instruction,
                        "session_id": self.session_id,
                        "camera_semantics": {
                            "exterior_left_arm_side": "Y+ exterior camera near left arm",
                            "exterior_right_arm_side": "Y- exterior camera near right arm",
                            "wrist_left_arm": "left_robot wrist camera",
                            "wrist_right_arm": "robot wrist camera",
                        },
                        "file_prefixes": file_prefixes,
                    }
                )
                + "\n"
            )

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
            self._debug_curr_obs = curr_obs
            self._dump_request_images(request_data, instruction)
            self._debug_curr_obs = None
            result = self.client.infer(request_data)
            self.infer_cnt += 1
            actions = result["actions"] if isinstance(result, dict) else result
            assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
            assert actions.shape[-1] == 8, f"Expected 8 action dimensions (7 joints + 1 gripper), got {actions.shape[-1]}"
            self.pred_action_chunk = actions

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        img3 = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        both = np.concatenate([img1, img2, img3], axis=1)

        return {"action": action, "viz": both}


def main(
    episodes: int = 10,
    scene: int = 1,
    headless: bool = True,
    host: str = "localhost",
    port: int = 6000,
):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run DROID bimanual sim eval.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "DROID_BIMANUAL_RIGHT_ONLY",
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
    env = gym.make("DROID_BIMANUAL_RIGHT_ONLY", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # need second render cycle to get correctly loaded materials
    client = DreamZeroJointPosBimanualClient(remote_host=host, remote_port=port)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    max_steps = env.env.max_episode_length
    show_viz = not headless

    with torch.no_grad():
        for ep in range(episodes):
            for _ in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if show_viz:
                    show_viz = _show_viz_frame("Right Camera", ret["viz"], show_viz)
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
    if show_viz:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    simulation_app.close()


if __name__ == "__main__":
    args = tyro.cli(main)
