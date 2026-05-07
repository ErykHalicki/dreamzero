#!/usr/bin/env python3
"""Convert an SO101 LeRobot v3 dataset to the episode-based v2 layout.

This keeps the source dataset untouched and writes a DreamZero-compatible
LeRobot v2.0 dataset with one parquet and one video per episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_SRC = Path("/workspace/datasets/so101_bottle")
DEFAULT_DST = Path("/workspace/datasets/so101_bottle_lerobot_v2")
DEFAULT_VIDEO_KEYS = ("observation.images.front", "observation.images.wrist")
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_TASK = "pick up the bottle and place it into the container"

LOG = logging.getLogger("so101_v3_to_lerobotv2")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=4, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_v3_tasks(src: Path) -> tuple[dict[int, str], list[dict[str, Any]]]:
    jsonl_path = src / "meta" / "tasks.jsonl"
    parquet_path = src / "meta" / "tasks.parquet"
    if jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    elif parquet_path.exists():
        df = pq.read_table(parquet_path).to_pandas().reset_index()
        if "task" not in df.columns or "task_index" not in df.columns:
            raise ValueError(f"Unexpected tasks parquet schema: {df.columns.tolist()}")
        rows = [
            {"task_index": int(row["task_index"]), "task": str(row["task"])}
            for _, row in df.iterrows()
        ]
    else:
        rows = [{"task_index": 0, "task": DEFAULT_TASK}]

    rows = sorted(rows, key=lambda row: int(row["task_index"]))
    return {int(row["task_index"]): str(row["task"]) for row in rows}, rows


def load_v3_episodes(src: Path) -> pd.DataFrame:
    paths = sorted((src / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 episode metadata found under {src / 'meta' / 'episodes'}")
    frames = [pq.read_table(path).to_pandas() for path in paths]
    episodes = pd.concat(frames, ignore_index=True).sort_values("episode_index")
    expected = list(range(len(episodes)))
    actual = episodes["episode_index"].astype(int).tolist()
    if actual != expected:
        raise ValueError("Episode indices are not contiguous from 0")
    return episodes


def format_v3_data_path(info: dict[str, Any], chunk_idx: int, file_idx: int) -> Path:
    return Path(info["data_path"].format(chunk_index=chunk_idx, file_index=file_idx))


def format_v3_video_path(info: dict[str, Any], video_key: str, chunk_idx: int, file_idx: int) -> Path:
    return Path(info["video_path"].format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx))


def v2_episode_chunk(ep_idx: int, chunks_size: int = 1000) -> int:
    return ep_idx // chunks_size


def output_data_path(dst: Path, ep_idx: int) -> Path:
    rel = DEFAULT_DATA_PATH.format(episode_chunk=v2_episode_chunk(ep_idx), episode_index=ep_idx)
    return dst / rel


def output_video_path(dst: Path, ep_idx: int, video_key: str) -> Path:
    rel = DEFAULT_VIDEO_PATH.format(
        episode_chunk=v2_episode_chunk(ep_idx), video_key=video_key, episode_index=ep_idx
    )
    return dst / rel


def as_feature_shape(feature: dict[str, Any]) -> int:
    shape = feature.get("shape", [1])
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"Expected 1D vector feature, got shape={shape}")
    return int(shape[0])


def make_output_info(
    src_info: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    embodiment_tag: str,
    video_codec: str,
) -> dict:
    features = dict(src_info["features"])
    features["annotation.task"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    codec_name = "h264" if video_codec == "h264" else "av1"
    for feature in features.values():
        if feature.get("dtype") == "video" and "info" in feature:
            feature["info"] = dict(feature["info"])
            feature["info"]["video.codec"] = codec_name

    out_info = dict(src_info)
    out_info.update(
        {
            "codebase_version": "v2.0",
            "robot_type": embodiment_tag,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": int(src_info.get("total_tasks", 1)),
            "total_videos": total_episodes * len([k for k, ft in features.items() if ft.get("dtype") == "video"]),
            "total_chunks": max(1, (total_episodes + 999) // 1000),
            "chunks_size": 1000,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": DEFAULT_DATA_PATH,
            "video_path": DEFAULT_VIDEO_PATH,
            "features": features,
        }
    )
    out_info.pop("data_files_size_in_mb", None)
    out_info.pop("video_files_size_in_mb", None)
    return out_info


def make_modality_json() -> dict[str, Any]:
    def entry(original_key: str, start: int, end: int) -> dict[str, Any]:
        return {
            "original_key": original_key,
            "start": start,
            "end": end,
            "rotation_type": None,
            "absolute": True,
            "dtype": "float32",
            "range": None,
        }

    return {
        "state": {
            "joint_pos": entry("observation.state", 0, 5),
            "gripper_pos": entry("observation.state", 5, 6),
        },
        "action": {
            "joint_pos": entry("action", 0, 5),
            "gripper_pos": entry("action", 5, 6),
        },
        "video": {
            "front": {"original_key": "observation.images.front"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "task": {"original_key": "annotation.task"},
        },
    }


def normalize_episode_dataframe(
    episode_df: pd.DataFrame,
    ep_idx: int,
    global_from: int,
    task_index: int,
) -> pd.DataFrame:
    out = episode_df.sort_values("frame_index").reset_index(drop=True).copy()
    length = len(out)
    out["frame_index"] = np.arange(length, dtype=np.int64)
    out["episode_index"] = np.full(length, ep_idx, dtype=np.int64)
    out["index"] = np.arange(global_from, global_from + length, dtype=np.int64)
    out["task_index"] = np.full(length, task_index, dtype=np.int64)
    out["annotation.task"] = np.full(length, task_index, dtype=np.int64)
    return out


def write_episode_parquet(dst: Path, ep_idx: int, episode_df: pd.DataFrame) -> None:
    out_path = output_data_path(dst, ep_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(episode_df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")


def video_codec_args(video_codec: str) -> list[str]:
    if video_codec == "h264":
        return ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    if video_codec == "av1":
        return ["-c:v", "libsvtav1", "-crf", "24", "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unsupported video codec: {video_codec}")


def run_ffmpeg_slice(
    ffmpeg: str,
    src_video: Path,
    dst_video: Path,
    *,
    start_frame: int,
    num_frames: int,
    fps: int,
    video_codec: str,
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    vf = f"trim=start_frame={start_frame}:end_frame={start_frame + num_frames},setpts=PTS-STARTPTS"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_video),
        "-vf",
        vf,
        "-an",
        "-r",
        str(fps),
        "-frames:v",
        str(num_frames),
        *video_codec_args(video_codec),
        str(dst_video),
    ]
    subprocess.run(cmd, check=True)


def ffprobe_frame_count(ffprobe: str, video_path: Path) -> int:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return int(result.stdout.strip())


def ffprobe_video_info(ffprobe: str, video_path: Path) -> dict[str, str]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def compute_stats(parquet_paths: list[Path], columns: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    buffers: dict[str, list[np.ndarray]] = {col: [] for col in columns}
    for path in tqdm(parquet_paths, desc="Computing stats"):
        df = pq.read_table(path, columns=[col for col in columns if col in pq.read_schema(path).names]).to_pandas()
        for col in columns:
            if col not in df.columns:
                continue
            values = df[col].to_numpy()
            if len(values) == 0:
                continue
            if isinstance(values[0], (list, tuple, np.ndarray)):
                arr = np.stack(values).astype(np.float64)
            else:
                arr = values.astype(np.float64).reshape(-1, 1)
            buffers[col].append(arr)

    for col, parts in buffers.items():
        if not parts:
            continue
        data = np.concatenate(parts, axis=0)
        stats[col] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def compute_relative_stats(parquet_paths: list[Path], action_horizon: int) -> dict[str, Any]:
    mappings = {
        "joint_pos": (0, 5),
        "gripper_pos": (5, 6),
    }
    buffers: dict[str, list[np.ndarray]] = {key: [] for key in mappings}

    for path in tqdm(parquet_paths, desc="Computing relative stats"):
        df = pq.read_table(path, columns=["observation.state", "action"]).to_pandas()
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        action = np.stack(df["action"].to_numpy()).astype(np.float64)
        traj_len = len(df)
        usable = max(traj_len - action_horizon, 0)
        for key, (start, end) in mappings.items():
            for frame_idx in range(usable):
                ref_state = state[frame_idx, start:end]
                chunk_end = min(frame_idx + action_horizon, traj_len)
                relative = action[frame_idx:chunk_end, start:end] - ref_state
                buffers[key].append(relative)

    stats: dict[str, Any] = {}
    for key, parts in buffers.items():
        if not parts:
            continue
        data = np.concatenate(parts, axis=0)
        stats[key] = {
            "max": np.max(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats


def validate_output(dst: Path, expected_episodes: int, expected_frames: int, video_keys: tuple[str, ...]) -> None:
    info = read_json(dst / "meta" / "info.json")
    if info["total_episodes"] != expected_episodes:
        raise AssertionError(f"total_episodes={info['total_episodes']} expected {expected_episodes}")
    if info["total_frames"] != expected_frames:
        raise AssertionError(f"total_frames={info['total_frames']} expected {expected_frames}")

    episodes = [json.loads(line) for line in (dst / "meta" / "episodes.jsonl").read_text().splitlines()]
    data_paths = sorted((dst / "data").glob("*/*.parquet"))
    video_paths = sorted((dst / "videos").glob("*/*/*.mp4"))
    if len(data_paths) != expected_episodes:
        raise AssertionError(f"Found {len(data_paths)} parquet files, expected {expected_episodes}")
    if len(video_paths) != expected_episodes * len(video_keys):
        raise AssertionError(
            f"Found {len(video_paths)} video files, expected {expected_episodes * len(video_keys)}"
        )

    for row in episodes:
        ep_idx = int(row["episode_index"])
        length = int(row["length"])
        df = pq.read_table(output_data_path(dst, ep_idx)).to_pandas()
        if len(df) != length:
            raise AssertionError(f"Episode {ep_idx}: parquet length {len(df)} != {length}")
        if set(df["episode_index"].astype(int).tolist()) != {ep_idx}:
            raise AssertionError(f"Episode {ep_idx}: unexpected episode_index values")
        if df["frame_index"].astype(int).tolist() != list(range(length)):
            raise AssertionError(f"Episode {ep_idx}: frame_index is not 0..length-1")
        if "annotation.task" not in df.columns:
            raise AssertionError(f"Episode {ep_idx}: annotation.task missing")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        sample_indices = sorted({0, expected_episodes // 2, expected_episodes - 1})
        for ep_idx in sample_indices:
            length = int(episodes[ep_idx]["length"])
            for video_key in video_keys:
                path = output_video_path(dst, ep_idx, video_key)
                count = ffprobe_frame_count(ffprobe, path)
                if count != length:
                    raise AssertionError(f"{path}: {count} frames, expected {length}")
                stream = ffprobe_video_info(ffprobe, path)
                if stream.get("codec_name") != "h264":
                    raise AssertionError(f"{path}: expected h264, got {stream.get('codec_name')}")
                if stream.get("width") != "640" or stream.get("height") != "480":
                    raise AssertionError(f"{path}: expected 640x480, got {stream}")
                if stream.get("avg_frame_rate") != "30/1":
                    raise AssertionError(f"{path}: expected 30 fps, got {stream.get('avg_frame_rate')}")


def convert(
    src: Path,
    dst: Path,
    *,
    overwrite: bool,
    validate: bool,
    recompute_stats: bool,
    skip_relative_stats: bool,
    action_horizon: int,
    embodiment_tag: str,
    video_codec: str,
    video_keys: tuple[str, ...],
) -> None:
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Source dataset missing meta/info.json: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    src_info = read_json(src / "meta" / "info.json")
    fps = int(src_info["fps"])
    if fps != 30:
        LOG.warning("Expected SO101 fps=30, got %s. Continuing with dataset fps.", fps)

    state_dim = as_feature_shape(src_info["features"]["observation.state"])
    action_dim = as_feature_shape(src_info["features"]["action"])
    if state_dim != 6 or action_dim != 6:
        raise ValueError(f"Expected 6D SO101 state/action, got state={state_dim}, action={action_dim}")

    for video_key in video_keys:
        if video_key not in src_info["features"]:
            raise ValueError(f"Video key {video_key!r} not found in source features")

    tasks_by_index, task_rows = load_v3_tasks(src)
    episodes = load_v3_episodes(src)
    expected_frames = int(episodes["length"].sum())

    dst.mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)
    if (src / "README.md").exists():
        shutil.copy2(src / "README.md", dst / "README.md")

    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    episode_rows: list[dict[str, Any]] = []

    for _, ep in tqdm(episodes.iterrows(), total=len(episodes), desc="Episodes"):
        ep_idx = int(ep["episode_index"])
        length = int(ep["length"])
        data_key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        if data_key not in data_cache:
            data_path = src / format_v3_data_path(src_info, *data_key)
            data_cache[data_key] = pq.read_table(data_path).to_pandas()

        shard_df = data_cache[data_key]
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        if to_idx - from_idx != length:
            raise ValueError(f"Episode {ep_idx}: metadata length mismatch")

        episode_df = shard_df[shard_df["episode_index"].astype(int) == ep_idx].copy()
        if len(episode_df) != length:
            raise ValueError(f"Episode {ep_idx}: sliced dataframe length {len(episode_df)} != {length}")
        task_indices = sorted({int(x) for x in episode_df["task_index"].tolist()})
        if not task_indices:
            task_indices = [0]
        task_index = task_indices[0]
        task_texts = [tasks_by_index.get(task_idx, DEFAULT_TASK) for task_idx in task_indices]
        episode_df = normalize_episode_dataframe(episode_df, ep_idx, from_idx, task_index)
        write_episode_parquet(dst, ep_idx, episode_df)

        for video_key in video_keys:
            video_chunk = int(ep[f"videos/{video_key}/chunk_index"])
            video_file = int(ep[f"videos/{video_key}/file_index"])
            src_video = src / format_v3_video_path(src_info, video_key, video_chunk, video_file)
            start_frame = int(round(float(ep[f"videos/{video_key}/from_timestamp"]) * fps))
            dst_video = output_video_path(dst, ep_idx, video_key)
            run_ffmpeg_slice(
                ffmpeg,
                src_video,
                dst_video,
                start_frame=start_frame,
                num_frames=length,
                fps=fps,
                video_codec=video_codec,
            )

        episode_rows.append({"episode_index": ep_idx, "tasks": task_texts, "length": length})

    out_info = make_output_info(src_info, len(episode_rows), expected_frames, embodiment_tag, video_codec)
    out_info["total_tasks"] = len(task_rows)
    out_info["total_videos"] = len(episode_rows) * len(video_keys)
    write_json(dst / "meta" / "info.json", out_info)
    write_jsonl(dst / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(dst / "meta" / "episodes.jsonl", episode_rows)
    write_json(dst / "meta" / "modality.json", make_modality_json())
    write_json(dst / "meta" / "embodiment.json", {"robot_type": embodiment_tag, "embodiment_tag": embodiment_tag})

    if recompute_stats:
        paths = sorted((dst / "data").glob("*/*.parquet"))
        stats = compute_stats(paths, ["observation.state", "action", "timestamp"])
        write_json(dst / "meta" / "stats.json", stats)
    elif (src / "meta" / "stats.json").exists():
        shutil.copy2(src / "meta" / "stats.json", dst / "meta" / "stats.json")

    if not skip_relative_stats:
        paths = sorted((dst / "data").glob("*/*.parquet"))
        relative_stats = compute_relative_stats(paths, action_horizon)
        write_json(dst / "meta" / "relative_stats_dreamzero.json", relative_stats)

    if validate:
        validate_output(dst, len(episode_rows), expected_frames, video_keys)

    LOG.info("Converted %d episodes (%d frames) to %s", len(episode_rows), expected_frames, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--skip-relative-stats", action="store_true")
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--embodiment-tag", default="so101")
    parser.add_argument("--video-codec", choices=["h264", "av1"], default="h264")
    parser.add_argument("--video-keys", nargs="+", default=list(DEFAULT_VIDEO_KEYS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        convert(
            args.src.resolve(),
            args.dst.resolve(),
            overwrite=args.overwrite,
            validate=args.validate,
            recompute_stats=args.recompute_stats,
            skip_relative_stats=args.skip_relative_stats,
            action_horizon=args.action_horizon,
            embodiment_tag=args.embodiment_tag,
            video_codec=args.video_codec,
            video_keys=tuple(args.video_keys),
        )
    except Exception as exc:
        LOG.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
