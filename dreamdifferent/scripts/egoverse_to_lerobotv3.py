#!/usr/bin/env python3
"""Convert Egoverse h5 episodes to LeRobot v3.0 format.

Usage:
    python egoverse_to_lerobotv3.py \
        --src_dir dreamdifferent/data/ \
        --tgt_path /output/dir \
        --task_type bag_groceries

Requires lerobot >= 0.5 (CODEBASE_VERSION v3.0). Use the smolvla venv:
    source dreamdifferent/baseline/smolvla/.venv/bin/activate

H5 structure:
  bag_groceries (2-arm):
    observations/qpos_arm_left (N,7), qpos_hand_left (N,17),
    observations/qpos_arm_right (N,7), qpos_hand_right (N,17)
    actions_arm_left (N,7), actions_hand_left (N,17),
    actions_arm_right (N,7), actions_hand_right (N,17)
    observations/images/aria_rgb_cam/color (N,480,640,3)
    observations/images/oakd_front_view/color (N,540,960,3)

  object_in_bowl (1-arm):
    observations/qpos_arm (N,7), qpos_hand (N,17)
    actions_arm (N,7), actions_hand (N,17)
    observations/images/aria_rgb_cam/color (N,480,640,3)
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


TASK_CONFIGS = {
    "bag_groceries": {
        "fps": 50,
        "task_instruction": "bag the groceries",
        "state_dim": 48,  # arm_l(7)+hand_l(17)+arm_r(7)+hand_r(17)
        "action_dim": 48,
        "cameras": {
            "observation.images.aria_rgb_cam": {
                "h5_key": "observations/images/aria_rgb_cam/color",
                "shape": (480, 640, 3),
            },
            "observation.images.oakd_front_view": {
                "h5_key": "observations/images/oakd_front_view/color",
                "shape": (540, 960, 3),
            },
        },
    },
    "object_in_bowl": {
        "fps": 50,
        "task_instruction": "put the object in the bowl",
        "state_dim": 24,
        "action_dim": 24,
        "cameras": {
            "observation.images.aria_rgb_cam": {
                "h5_key": "observations/images/aria_rgb_cam/color",
                "shape": (480, 640, 3),
            },
        },
    },
}


def make_features(task_type: str) -> dict:
    cfg = TASK_CONFIGS[task_type]
    features = {}
    for cam_key, cam_cfg in cfg["cameras"].items():
        features[cam_key] = {
            "dtype": "video",
            "shape": cam_cfg["shape"],
            "names": ["height", "width", "channels"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (cfg["state_dim"],),
        "names": None,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (cfg["action_dim"],),
        "names": None,
    }
    return features


def convert_episode(h5_path: Path, task_type: str, dataset: LeRobotDataset) -> None:
    cfg = TASK_CONFIGS[task_type]
    task = cfg["task_instruction"]

    with h5py.File(h5_path, "r") as f:
        if task_type == "bag_groceries":
            state = np.hstack([
                f["observations/qpos_arm_left"][:],
                f["observations/qpos_hand_left"][:],
                f["observations/qpos_arm_right"][:],
                f["observations/qpos_hand_right"][:],
            ]).astype(np.float32)
            action = np.hstack([
                f["actions_arm_left"][:],
                f["actions_hand_left"][:],
                f["actions_arm_right"][:],
                f["actions_hand_right"][:],
            ]).astype(np.float32)
        else:
            state = np.hstack([
                f["observations/qpos_arm"][:],
                f["observations/qpos_hand"][:],
            ]).astype(np.float32)
            action = np.hstack([
                f["actions_arm"][:],
                f["actions_hand"][:],
            ]).astype(np.float32)

        # Egoverse stores images uncompressed; pull all frames into memory once
        # so we can iterate and feed them straight to the streaming encoder.
        images = {
            cam_key: f[cam_cfg["h5_key"]][:]
            for cam_key, cam_cfg in cfg["cameras"].items()
        }

    n_frames = len(state)
    for i in tqdm(range(n_frames), desc=f"  {h5_path.name}", leave=False):
        frame = {
            "observation.state": state[i],
            "action": action[i],
            "task": task,
        }
        for cam_key, cam_array in images.items():
            frame[cam_key] = cam_array[i]
        dataset.add_frame(frame)

    dataset.save_episode()


MANIFEST_FILE = "processed_episodes.json"


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"processed": []}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert Egoverse h5 to LeRobot v3.0")
    parser.add_argument("--src_dir", required=True, help="Directory containing .h5 episode files")
    parser.add_argument("--tgt_path", required=True, help="Root output directory")
    parser.add_argument("--task_type", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--repo_id", default=None, help="e.g. egoverse/bag_groceries")
    parser.add_argument("--vcodec", default="h264",
                        help="Video codec (h264, hevc, libsvtav1, auto). Default h264 matches the v2 script.")
    parser.add_argument("--debug", action="store_true", help="Process only first 2 episodes")
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task_type]
    repo_id = args.repo_id or f"egoverse/{args.task_type}"

    dataset_root = Path(args.tgt_path) / repo_id
    manifest_path = dataset_root / MANIFEST_FILE
    manifest = load_manifest(manifest_path)
    already_processed = set(manifest["processed"])

    h5_files = sorted(Path(args.src_dir).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files in {args.src_dir}")
    h5_files = [f for f in h5_files if f.name not in already_processed]
    if args.debug:
        h5_files = h5_files[:2]
    if not h5_files:
        print("All episodes already processed. Nothing to do.")
        return
    print(f"Found {len(h5_files)} new episode file(s) for '{args.task_type}' "
          f"({len(already_processed)} already processed)")

    if (dataset_root / "meta" / "info.json").exists():
        print(f"Resuming existing dataset at {dataset_root}")
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=str(dataset_root),
            vcodec=args.vcodec,
            streaming_encoding=True,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=str(dataset_root),
            fps=cfg["fps"],
            robot_type="franka",
            features=make_features(args.task_type),
            vcodec=args.vcodec,
            streaming_encoding=True,
        )

    try:
        for h5_path in tqdm(h5_files, desc="Episodes"):
            convert_episode(h5_path, args.task_type, dataset)
            manifest["processed"].append(h5_path.name)
            save_manifest(manifest_path, manifest)
    finally:
        dataset.finalize()

    print(f"\nDone. Dataset at {dataset_root}")


if __name__ == "__main__":
    main()
