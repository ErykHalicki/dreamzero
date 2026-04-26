#!/usr/bin/env python3
"""Compare train/val distributions for the Franka/ORCA bag-groceries split.

This script is intentionally separate from split creation. It can be run in an
environment with pyarrow to inspect raw parquet vector distributions, and
optionally with opencv-python to sample video RGB/temporal statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np


DEFAULT_DATASET_ROOT = Path("/cluster/scratch/eugseo/datasets/bag_groceries_communal")
DEFAULT_SPLIT_PATH = (
    Path(__file__).resolve().parent / "bag_groceries_seed42_90_10.json"
)
ACTION_HORIZON = 24
GROUPS = {
    "left_arm": (0, 7),
    "left_hand": (7, 24),
    "right_arm": (24, 31),
    "right_hand": (31, 48),
}
CAMERAS = (
    "observation.images.aria_rgb_cam",
    "observation.images.oakd_front_view",
)


class OnlineVectorStats:
    def __init__(self, dim: int, *, max_samples: int, rng: random.Random):
        self.dim = dim
        self.max_samples = max_samples
        self.rng = rng
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.samples: list[np.ndarray] = []

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.size == 0:
            return
        if values.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {values.shape[1]}")

        self.sum += values.sum(axis=0)
        self.sumsq += np.square(values).sum(axis=0)
        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))

        for row in values:
            self.count += 1
            if len(self.samples) < self.max_samples:
                self.samples.append(row.astype(np.float32, copy=True))
            else:
                sample_idx = self.rng.randrange(self.count)
                if sample_idx < self.max_samples:
                    self.samples[sample_idx] = row.astype(np.float32, copy=True)

    def finalize(self) -> dict:
        if self.count == 0:
            return {"count": 0}
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        sample_arr = np.asarray(self.samples, dtype=np.float64)
        quantiles = {}
        if len(sample_arr):
            for name, q in [("q01", 0.01), ("q50", 0.50), ("q99", 0.99)]:
                quantiles[name] = np.quantile(sample_arr, q, axis=0).tolist()
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "sample_count_for_quantiles": len(self.samples),
            **quantiles,
        }


class OnlineScalarStats:
    def __init__(self, *, max_samples: int, rng: random.Random):
        self.stats = OnlineVectorStats(1, max_samples=max_samples, rng=rng)

    def add(self, values: np.ndarray) -> None:
        self.stats.add(np.asarray(values, dtype=np.float64).reshape(-1, 1))

    def finalize(self) -> dict:
        result = self.stats.finalize()
        if result.get("count", 0) == 0:
            return result
        out = {"count": result["count"], "sample_count_for_quantiles": result["sample_count_for_quantiles"]}
        for key in ["mean", "std", "min", "max", "q01", "q50", "q99"]:
            if key in result:
                out[key] = result[key][0]
        return out


class ImageStats:
    def __init__(self):
        self.pixel_count = 0
        self.sum = np.zeros(3, dtype=np.float64)
        self.sumsq = np.zeros(3, dtype=np.float64)
        self.temporal_abs_diff = []
        self.frames = 0
        self.videos = 0

    def add_frame(self, frame_rgb: np.ndarray, *, pixel_stride: int) -> None:
        frame = frame_rgb[::pixel_stride, ::pixel_stride].astype(np.float64) / 255.0
        pixels = frame.reshape(-1, 3)
        self.pixel_count += len(pixels)
        self.sum += pixels.sum(axis=0)
        self.sumsq += np.square(pixels).sum(axis=0)
        self.frames += 1

    def add_temporal_diff(self, prev_rgb: np.ndarray, frame_rgb: np.ndarray, *, pixel_stride: int) -> None:
        prev = prev_rgb[::pixel_stride, ::pixel_stride].astype(np.float32) / 255.0
        cur = frame_rgb[::pixel_stride, ::pixel_stride].astype(np.float32) / 255.0
        self.temporal_abs_diff.append(float(np.mean(np.abs(cur - prev))))

    def finalize(self) -> dict:
        if self.pixel_count == 0:
            return {"videos": self.videos, "frames": self.frames, "pixel_count": 0}
        mean = self.sum / self.pixel_count
        var = np.maximum(self.sumsq / self.pixel_count - np.square(mean), 0.0)
        return {
            "videos": self.videos,
            "frames": self.frames,
            "pixel_count": self.pixel_count,
            "rgb_mean_0_1": mean.tolist(),
            "rgb_std_0_1": np.sqrt(var).tolist(),
            "temporal_abs_diff_mean": float(np.mean(self.temporal_abs_diff)) if self.temporal_abs_diff else None,
            "temporal_abs_diff_std": float(np.std(self.temporal_abs_diff)) if self.temporal_abs_diff else None,
        }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _video_path(dataset_root: Path, camera: str, episode_index: int) -> Path:
    return dataset_root / "videos" / "chunk-000" / camera / f"episode_{episode_index:06d}.mp4"


def _array_from_arrow_column(table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def _sample_relative_actions(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    action_horizon: int,
    samples_per_episode: int,
    rng: random.Random,
) -> np.ndarray:
    max_anchor = len(actions) - action_horizon + 1
    if max_anchor <= 0 or samples_per_episode <= 0:
        return np.empty((0, actions.shape[1]), dtype=np.float32)
    anchors = [rng.randrange(max_anchor) for _ in range(min(samples_per_episode, max_anchor))]
    rel_rows = []
    for anchor in anchors:
        rel_rows.append(actions[anchor : anchor + action_horizon] - states[anchor])
    return np.concatenate(rel_rows, axis=0)


def _update_vector_stats(
    split_name: str,
    episode_indices: Iterable[int],
    dataset_root: Path,
    *,
    max_sample_rows: int,
    relative_samples_per_episode: int,
    seed: int,
) -> dict:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return {"available": False, "reason": f"pyarrow unavailable: {type(exc).__name__}: {exc}"}

    rng = random.Random(seed)
    state_stats = OnlineVectorStats(48, max_samples=max_sample_rows, rng=random.Random(seed + 11))
    action_stats = OnlineVectorStats(48, max_samples=max_sample_rows, rng=random.Random(seed + 12))
    action_delta_stats = OnlineVectorStats(48, max_samples=max_sample_rows, rng=random.Random(seed + 13))
    relative_action_stats = OnlineVectorStats(48, max_samples=max_sample_rows, rng=random.Random(seed + 14))
    group_norm_stats = {
        name: {
            "state_l2": OnlineScalarStats(max_samples=max_sample_rows, rng=random.Random(seed + 100 + i)),
            "action_l2": OnlineScalarStats(max_samples=max_sample_rows, rng=random.Random(seed + 200 + i)),
            "action_delta_l2": OnlineScalarStats(max_samples=max_sample_rows, rng=random.Random(seed + 300 + i)),
        }
        for i, name in enumerate(GROUPS)
    }

    missing = []
    for episode_index in episode_indices:
        path = _episode_path(dataset_root, episode_index)
        if not path.exists():
            missing.append(episode_index)
            continue
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = _array_from_arrow_column(table, "observation.state")
        actions = _array_from_arrow_column(table, "action")
        state_stats.add(states)
        action_stats.add(actions)
        if len(actions) > 1:
            action_delta_stats.add(np.diff(actions, axis=0))
        rel_actions = _sample_relative_actions(
            states,
            actions,
            action_horizon=ACTION_HORIZON,
            samples_per_episode=relative_samples_per_episode,
            rng=rng,
        )
        relative_action_stats.add(rel_actions)

        for group_name, (start, end) in GROUPS.items():
            group_norm_stats[group_name]["state_l2"].add(np.linalg.norm(states[:, start:end], axis=1))
            group_norm_stats[group_name]["action_l2"].add(np.linalg.norm(actions[:, start:end], axis=1))
            if len(actions) > 1:
                group_norm_stats[group_name]["action_delta_l2"].add(
                    np.linalg.norm(np.diff(actions[:, start:end], axis=0), axis=1)
                )

    return {
        "available": True,
        "missing_episode_indices": missing,
        "state": state_stats.finalize(),
        "action": action_stats.finalize(),
        "action_delta": action_delta_stats.finalize(),
        "relative_action_sample": relative_action_stats.finalize(),
        "group_l2": {
            group_name: {stat_name: stat.finalize() for stat_name, stat in stats.items()}
            for group_name, stats in group_norm_stats.items()
        },
    }


def _sample_video_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= num_frames:
        return list(range(frame_count))
    return sorted(set(int(round(x)) for x in np.linspace(0, frame_count - 1, num_frames)))


def _compute_image_stats(
    split_indices: list[int],
    dataset_root: Path,
    *,
    seed: int,
    episodes_per_split: int,
    frames_per_episode: int,
    pixel_stride: int,
) -> dict:
    try:
        import cv2
    except Exception as exc:
        return {"available": False, "reason": f"opencv unavailable: {type(exc).__name__}: {exc}"}

    rng = random.Random(seed)
    sampled_episodes = list(split_indices)
    rng.shuffle(sampled_episodes)
    sampled_episodes = sampled_episodes[:episodes_per_split]
    result = {camera: ImageStats() for camera in CAMERAS}
    missing = []

    for episode_index in sampled_episodes:
        for camera in CAMERAS:
            path = _video_path(dataset_root, camera, episode_index)
            if not path.exists():
                missing.append(str(path))
                continue
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    missing.append(str(path))
                    continue
                stats = result[camera]
                stats.videos += 1
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                prev_frame = None
                for frame_idx in _sample_video_frame_indices(frame_count, frames_per_episode):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame_bgr = cap.read()
                    if not ok:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    stats.add_frame(frame_rgb, pixel_stride=pixel_stride)
                    if prev_frame is not None:
                        stats.add_temporal_diff(prev_frame, frame_rgb, pixel_stride=pixel_stride)
                    prev_frame = frame_rgb
            finally:
                cap.release()

    return {
        "available": True,
        "sampled_episode_indices": sampled_episodes,
        "missing_videos": missing,
        "cameras": {camera: stats.finalize() for camera, stats in result.items()},
    }


def _mean_abs_diff(train_values: list[float], val_values: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(train_values) - np.asarray(val_values))))


def _summarize_vector_comparison(train: dict, val: dict) -> dict:
    if not train.get("available") or not val.get("available"):
        return {"available": False}
    out = {}
    for key in ["state", "action", "action_delta", "relative_action_sample"]:
        train_stats = train[key]
        val_stats = val[key]
        out[key] = {
            "train_count": train_stats["count"],
            "val_count": val_stats["count"],
            "mean_abs_diff_avg_dim": _mean_abs_diff(train_stats["mean"], val_stats["mean"]),
            "std_abs_diff_avg_dim": _mean_abs_diff(train_stats["std"], val_stats["std"]),
        }
        for q_key in ["q01", "q50", "q99"]:
            if q_key in train_stats and q_key in val_stats:
                out[key][f"{q_key}_abs_diff_avg_dim"] = _mean_abs_diff(train_stats[q_key], val_stats[q_key])
    return {"available": True, "features": out}


def _write_markdown(path: Path, analysis: dict) -> None:
    lines = [
        "# Train/Val Distribution Comparison",
        "",
        f"- Dataset: `{analysis['dataset_root']}`",
        f"- Split file: `{analysis['split_path']}`",
        f"- Train episodes: {analysis['episode_counts']['train']}",
        f"- Val episodes: {analysis['episode_counts']['val']}",
        "",
        "## Vector Stats",
        "",
    ]
    comparison = analysis["vector_comparison"]
    if not comparison.get("available"):
        lines.append("- Unavailable. Check `vector_stats.*.reason` in JSON.")
    else:
        lines.append("| feature | train rows | val rows | avg abs mean diff | avg abs std diff |")
        lines.append("|---|---:|---:|---:|---:|")
        for feature, stats in comparison["features"].items():
            lines.append(
                f"| `{feature}` | {stats['train_count']} | {stats['val_count']} | "
                f"{stats['mean_abs_diff_avg_dim']:.6f} | {stats['std_abs_diff_avg_dim']:.6f} |"
            )
        lines.extend(
            [
                "",
                "Notes:",
                "- `state` and `action` are raw per-frame vectors from parquet.",
                "- `action_delta` is `action[t+1] - action[t]`, a simple temporal smoothness proxy.",
                "- `relative_action_sample` samples OpenPI/DreamZero-style targets: `action[t:t+24] - state[t]`.",
            ]
        )

    lines.extend(["", "## Group L2 Means", ""])
    if analysis["vector_stats"]["train"].get("available") and analysis["vector_stats"]["val"].get("available"):
        lines.append("| group | stat | train mean | val mean |")
        lines.append("|---|---|---:|---:|")
        for group in GROUPS:
            for stat_name in ["state_l2", "action_l2", "action_delta_l2"]:
                train_mean = analysis["vector_stats"]["train"]["group_l2"][group][stat_name].get("mean")
                val_mean = analysis["vector_stats"]["val"]["group_l2"][group][stat_name].get("mean")
                if train_mean is not None and val_mean is not None:
                    lines.append(f"| `{group}` | `{stat_name}` | {train_mean:.6f} | {val_mean:.6f} |")

    lines.extend(["", "## Image Stats", ""])
    image_stats = analysis.get("image_stats")
    if not image_stats:
        lines.append("- Not requested. Re-run with `--image-stats` to sample video RGB statistics.")
    elif not image_stats["train"].get("available") or not image_stats["val"].get("available"):
        lines.append("- Unavailable. Check `image_stats.*.reason` in JSON.")
    else:
        lines.append("| camera | train RGB mean | val RGB mean | train temporal diff | val temporal diff |")
        lines.append("|---|---|---|---:|---:|")
        for camera in CAMERAS:
            train_cam = image_stats["train"]["cameras"][camera]
            val_cam = image_stats["val"]["cameras"][camera]
            lines.append(
                f"| `{camera}` | "
                f"{[round(x, 4) for x in train_cam.get('rgb_mean_0_1', [])]} | "
                f"{[round(x, 4) for x in val_cam.get('rgb_mean_0_1', [])]} | "
                f"{train_cam.get('temporal_abs_diff_mean')} | {val_cam.get('temporal_abs_diff_mean')} |"
            )
        lines.extend(
            [
                "",
                "Image stats are sampled RGB/channel and frame-difference sanity checks only. They do not replace",
                "video-sequence evaluation or representation-level checks for a video backbone.",
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--max-sample-rows", type=int, default=50_000)
    parser.add_argument("--max-episodes-per-split", type=int, default=None)
    parser.add_argument("--relative-samples-per-episode", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-stats", action="store_true")
    parser.add_argument("--image-episodes-per-split", type=int, default=20)
    parser.add_argument("--image-frames-per-episode", type=int, default=16)
    parser.add_argument("--pixel-stride", type=int, default=8)
    args = parser.parse_args()

    split = json.loads(args.split_path.read_text())
    train_indices = list(split["train_episode_indices"])
    val_indices = list(split["val_episode_indices"])
    if args.max_episodes_per_split is not None:
        train_indices = train_indices[: args.max_episodes_per_split]
        val_indices = val_indices[: args.max_episodes_per_split]
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.split_path.with_name(args.split_path.stem + "_distribution")

    train_vector = _update_vector_stats(
        "train",
        train_indices,
        args.dataset_root,
        max_sample_rows=args.max_sample_rows,
        relative_samples_per_episode=args.relative_samples_per_episode,
        seed=args.seed,
    )
    val_vector = _update_vector_stats(
        "val",
        val_indices,
        args.dataset_root,
        max_sample_rows=args.max_sample_rows,
        relative_samples_per_episode=args.relative_samples_per_episode,
        seed=args.seed + 1,
    )

    analysis = {
        "dataset_root": str(args.dataset_root),
        "split_path": str(args.split_path),
        "episode_counts": {"train": len(train_indices), "val": len(val_indices)},
        "vector_stats": {"train": train_vector, "val": val_vector},
        "vector_comparison": _summarize_vector_comparison(train_vector, val_vector),
    }

    if args.image_stats:
        analysis["image_stats"] = {
            "train": _compute_image_stats(
                train_indices,
                args.dataset_root,
                seed=args.seed + 2,
                episodes_per_split=args.image_episodes_per_split,
                frames_per_episode=args.image_frames_per_episode,
                pixel_stride=args.pixel_stride,
            ),
            "val": _compute_image_stats(
                val_indices,
                args.dataset_root,
                seed=args.seed + 3,
                episodes_per_split=args.image_episodes_per_split,
                frames_per_episode=args.image_frames_per_episode,
                pixel_stride=args.pixel_stride,
            ),
        }

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(analysis, indent=2) + "\n")
    _write_markdown(md_path, analysis)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
