#!/usr/bin/env python3
"""Convert Egoverse h5 episodes to LeRobot v2.0 format.

Usage:
    python egoverse_to_lerobotv2.py \
        --src_dir dreamdifferent/data/ \
        --tgt_path /output/dir \
        --task_type bag_groceries

Install compatible lerobot first:
    pip install git+https://github.com/huggingface/lerobot.git@32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3

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
import subprocess

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
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
                "shape": [480, 640, 3],
            },
            "observation.images.oakd_front_view": {
                "h5_key": "observations/images/oakd_front_view/color",
                "shape": [540, 960, 3],
            },
        },
    },
    "object_in_bowl": {
        "fps": 50,
        "task_instruction": "put the object in the bowl",
        "state_dim": 24,  # arm(7)+hand(17)
        "action_dim": 24,
        "cameras": {
            "observation.images.aria_rgb_cam": {
                "h5_key": "observations/images/aria_rgb_cam/color",
                "shape": [480, 640, 3],
            },
        },
    },
}


def make_features(task_type: str) -> dict:
    cfg = TASK_CONFIGS[task_type]
    features = {}
    for cam_key, cam_cfg in cfg["cameras"].items():
        h, w, c = cam_cfg["shape"]
        features[cam_key] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(cfg["fps"]),
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    features["observation.state"] = {"dtype": "float32", "shape": [cfg["state_dim"]]}
    features["action"] = {"dtype": "float32", "shape": [cfg["action_dim"]]}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def generate_modality_json(meta_path: Path, task_type: str) -> None:
    """Write modality.json mapping state/action dims to named DOF groups (needed for GEAR)."""
    cfg = TASK_CONFIGS[task_type]

    def entry(key, start, end):
        return {"original_key": key, "start": start, "end": end,
                "rotation_type": None, "absolute": True, "dtype": "float32", "range": None}

    if task_type == "bag_groceries":
        state_fields = {
            "left_arm_joint_position":   entry("observation.state", 0,  7),
            "left_hand_joint_position":  entry("observation.state", 7,  24),
            "right_arm_joint_position":  entry("observation.state", 24, 31),
            "right_hand_joint_position": entry("observation.state", 31, 48),
        }
        action_fields = {
            "left_arm_joint_position":   entry("action", 0,  7),
            "left_hand_joint_position":  entry("action", 7,  24),
            "right_arm_joint_position":  entry("action", 24, 31),
            "right_hand_joint_position": entry("action", 31, 48),
        }
    else:
        state_fields = {
            "arm_joint_position":  entry("observation.state", 0, 7),
            "hand_joint_position": entry("observation.state", 7, 24),
        }
        action_fields = {
            "arm_joint_position":  entry("action", 0, 7),
            "hand_joint_position": entry("action", 7, 24),
        }

    modality_config = {
        "state": state_fields,
        "action": action_fields,
        "video": {cam_key.split(".")[-1]: {"original_key": cam_key}
                  for cam_key in cfg["cameras"]},
    }
    out = meta_path / "modality.json"
    with open(out, "w") as f:
        json.dump(modality_config, f, indent=4)
    print(f"Written {out}")


def encode_video(h5_cam_dataset, fps: float, output_path: Path) -> None:
    """Pipe raw RGB frames from h5 directly to ffmpeg — no intermediate JPEG files."""
    n, h, w, _ = h5_cam_dataset.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(h5_cam_dataset[:].tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed for {output_path}")


def convert_episode(h5_path: Path, task_type: str, dataset: LeRobotDataset) -> None:
    cfg = TASK_CONFIGS[task_type]
    episode_index = dataset.meta.total_episodes

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

        for cam_key, cam_cfg in cfg["cameras"].items():
            video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, cam_key)
            print(f"  Encoding {cam_key}...")
            encode_video(f[cam_cfg["h5_key"]], cfg["fps"], video_path)

    for i in tqdm(range(len(state)), desc=f"  {h5_path.name}", leave=False):
        dataset.add_frame({"observation.state": state[i], "action": action[i]})

    dataset.save_episode(task=cfg["task_instruction"], encode_videos=False)


MANIFEST_FILE = "processed_episodes.json"


def load_manifest(meta_path: Path) -> dict:
    manifest_path = meta_path / MANIFEST_FILE
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"processed": []}


def save_manifest(meta_path: Path, manifest: dict) -> None:
    meta_path.mkdir(parents=True, exist_ok=True)
    with open(meta_path / MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert Egoverse h5 to LeRobot v2.0")
    parser.add_argument("--src_dir", required=True, help="Directory containing .h5 episode files")
    parser.add_argument("--tgt_path", required=True, help="Root output directory")
    parser.add_argument("--task_type", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--repo_id", default=None, help="e.g. egoverse/bag_groceries")
    parser.add_argument("--debug", action="store_true", help="Process only first 2 episodes")
    args = parser.parse_args()

    cfg = TASK_CONFIGS[args.task_type]
    repo_id = args.repo_id or f"egoverse/{args.task_type}"

    meta_path = Path(args.tgt_path) / repo_id / "meta"
    manifest = load_manifest(meta_path)
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

    dataset_root = Path(args.tgt_path) / repo_id
    if (dataset_root / "meta" / "info.json").exists():
        print(f"Resuming existing dataset at {dataset_root}")
        dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), local_files_only=True)
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=str(dataset_root),
            fps=cfg["fps"],
            robot_type="franka",
            features=make_features(args.task_type),
        )

    for h5_path in tqdm(h5_files, desc="Episodes"):
        convert_episode(h5_path, args.task_type, dataset)
        manifest["processed"].append(h5_path.name)
        save_manifest(meta_path, manifest)

    dataset.consolidate(run_compute_stats=False)

    generate_modality_json(meta_path, args.task_type)

    print(f"\nDone. Dataset at {Path(args.tgt_path) / repo_id}")


if __name__ == "__main__":
    main()
