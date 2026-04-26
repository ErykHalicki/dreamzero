#!/usr/bin/env python3
"""Create and analyze a length-stratified Franka/ORCA train/val split.

The output is intentionally stored outside both OpenPI and DreamZero so the same
split can be used by both projects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    values_sorted = sorted(values)

    def quantile(q: float) -> float:
        idx = min(len(values_sorted) - 1, max(0, round(q * (len(values_sorted) - 1))))
        return values_sorted[idx]

    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "q25": quantile(0.25),
        "median": quantile(0.50),
        "q75": quantile(0.75),
        "max": max(values),
    }


def _make_length_bins(episodes: list[dict], num_bins: int) -> list[list[dict]]:
    ordered = sorted(episodes, key=lambda row: (row["length"], row["episode_index"]))
    bins = []
    for bin_idx in range(num_bins):
        start = round(bin_idx * len(ordered) / num_bins)
        end = round((bin_idx + 1) * len(ordered) / num_bins)
        bins.append(ordered[start:end])
    return [b for b in bins if b]


def _bin_by_episode(episodes: list[dict], num_bins: int) -> dict[int, int]:
    mapping = {}
    for bin_idx, rows in enumerate(_make_length_bins(episodes, num_bins)):
        for row in rows:
            mapping[row["episode_index"]] = bin_idx
    return mapping


def _split_episodes(episodes: list[dict], val_ratio: float, num_bins: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    val_indices = []
    for bin_rows in _make_length_bins(episodes, num_bins):
        rows = list(bin_rows)
        rng.shuffle(rows)
        num_val = max(1, round(len(rows) * val_ratio))
        val_indices.extend(row["episode_index"] for row in rows[:num_val])

    val_set = set(val_indices)
    train_indices = [row["episode_index"] for row in episodes if row["episode_index"] not in val_set]
    return sorted(train_indices), sorted(val_set)


def _task_counts(episodes: list[dict], split_indices: set[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in episodes:
        if row["episode_index"] not in split_indices:
            continue
        for task in row.get("tasks", []):
            counts[str(task)] = counts.get(str(task), 0) + 1
    return counts


def _split_summary(episodes: list[dict], indices: list[int], bin_by_episode: dict[int, int] | None = None) -> dict:
    index_set = set(indices)
    lengths = [row["length"] for row in episodes if row["episode_index"] in index_set]
    bin_counts: dict[str, int] = {}
    if bin_by_episode is not None:
        for row in episodes:
            episode_index = row["episode_index"]
            if episode_index in index_set:
                key = str(bin_by_episode[episode_index])
                bin_counts[key] = bin_counts.get(key, 0) + 1
    return {
        "num_episodes": len(indices),
        "num_frames": sum(lengths),
        "length": _summarize(lengths),
        "tasks": _task_counts(episodes, index_set),
        "bin_counts": bin_counts,
    }


def _video_counts(dataset_root: Path) -> dict[str, int]:
    video_root = dataset_root / "videos" / "chunk-000"
    counts = {}
    for camera_dir in sorted(video_root.glob("*")):
        if camera_dir.is_dir():
            counts[camera_dir.name] = len(list(camera_dir.glob("*.mp4")))
    return counts


def _write_episode_csv(path: Path, episodes: list[dict], train_indices: list[int], val_indices: list[int], bin_by_episode: dict[int, int]) -> None:
    split_by_episode = {episode_index: "train" for episode_index in train_indices}
    split_by_episode.update({episode_index: "val" for episode_index in val_indices})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "split", "length", "length_bin", "tasks"])
        writer.writeheader()
        for row in sorted(episodes, key=lambda x: x["episode_index"]):
            episode_index = row["episode_index"]
            writer.writerow(
                {
                    "episode_index": episode_index,
                    "split": split_by_episode[episode_index],
                    "length": row["length"],
                    "length_bin": bin_by_episode[episode_index],
                    "tasks": " ".join(str(t) for t in row.get("tasks", [])),
                }
            )


def _svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_bin_svg(path: Path, episodes: list[dict], train_indices: list[int], val_indices: list[int], bin_by_episode: dict[int, int]) -> None:
    train_set = set(train_indices)
    val_set = set(val_indices)
    bins = sorted(set(bin_by_episode.values()))
    bin_rows = []
    for bin_idx in bins:
        rows = [row for row in episodes if bin_by_episode[row["episode_index"]] == bin_idx]
        lengths = [row["length"] for row in rows]
        train_count = sum(1 for row in rows if row["episode_index"] in train_set)
        val_count = sum(1 for row in rows if row["episode_index"] in val_set)
        bin_rows.append(
            {
                "bin": bin_idx,
                "min": min(lengths),
                "max": max(lengths),
                "train": train_count,
                "val": val_count,
            }
        )

    width, height = 920, 420
    margin_left, margin_bottom, margin_top = 70, 70, 50
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom
    max_count = max(row["train"] + row["val"] for row in bin_rows)
    group_w = plot_w / len(bin_rows)
    bar_w = group_w * 0.28
    scale = plot_h / max_count

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="30" y="30" font-family="sans-serif" font-size="20" font-weight="700">Length-stratified split bins</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 30}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222"/>',
        '<rect x="710" y="20" width="16" height="16" fill="#4c78a8"/><text x="732" y="34" font-family="sans-serif" font-size="13">train</text>',
        '<rect x="790" y="20" width="16" height="16" fill="#f58518"/><text x="812" y="34" font-family="sans-serif" font-size="13">val</text>',
    ]

    for tick in range(0, max_count + 1, max(1, max_count // 5)):
        y = height - margin_bottom - tick * scale
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>')

    for i, row in enumerate(bin_rows):
        x0 = margin_left + i * group_w + group_w * 0.18
        train_h = row["train"] * scale
        val_h = row["val"] * scale
        parts.append(f'<rect x="{x0:.1f}" y="{height - margin_bottom - train_h:.1f}" width="{bar_w:.1f}" height="{train_h:.1f}" fill="#4c78a8"/>')
        parts.append(f'<rect x="{x0 + bar_w + 8:.1f}" y="{height - margin_bottom - val_h:.1f}" width="{bar_w:.1f}" height="{val_h:.1f}" fill="#f58518"/>')
        label = f"bin {row['bin']}\\n{row['min']}-{row['max']}"
        parts.append(
            f'<text x="{x0 + bar_w:.1f}" y="{height - 42}" text-anchor="middle" font-family="sans-serif" font-size="11">'
            f'<tspan x="{x0 + bar_w:.1f}" dy="0">{_svg_escape("bin " + str(row["bin"]))}</tspan>'
            f'<tspan x="{x0 + bar_w:.1f}" dy="14">{row["min"]}-{row["max"]}</tspan></text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def _write_mean_length_svg(path: Path, analysis: dict) -> None:
    train = analysis["splits"]["train"]["length"]
    val = analysis["splits"]["val"]["length"]
    width, height = 620, 360
    margin_left, margin_bottom, margin_top = 80, 60, 50
    plot_h = height - margin_top - margin_bottom
    max_value = max(train["mean"] + train["std"], val["mean"] + val["std"])
    scale = plot_h / max_value
    bars = [("train", train, "#4c78a8", 190), ("val", val, "#f58518", 360)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="30" y="30" font-family="sans-serif" font-size="20" font-weight="700">Mean episode length</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 40}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222"/>',
    ]
    for tick in range(0, int(max_value) + 1, max(1, int(max_value) // 5)):
        y = height - margin_bottom - tick * scale
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}</text>')

    for label, stats, color, x in bars:
        bar_h = stats["mean"] * scale
        y = height - margin_bottom - bar_h
        std_y_top = height - margin_bottom - (stats["mean"] + stats["std"]) * scale
        std_y_bottom = height - margin_bottom - max(0.0, stats["mean"] - stats["std"]) * scale
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="80" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<line x1="{x + 40}" y1="{std_y_top:.1f}" x2="{x + 40}" y2="{std_y_bottom:.1f}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<line x1="{x + 28}" y1="{std_y_top:.1f}" x2="{x + 52}" y2="{std_y_top:.1f}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<line x1="{x + 28}" y1="{std_y_bottom:.1f}" x2="{x + 52}" y2="{std_y_bottom:.1f}" stroke="#222" stroke-width="2"/>')
        parts.append(f'<text x="{x + 40}" y="{height - 35}" text-anchor="middle" font-family="sans-serif" font-size="14">{label}</text>')
        parts.append(f'<text x="{x + 40}" y="{y - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{stats["mean"]:.1f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def _maybe_parquet_stats(dataset_root: Path, split_indices: list[int], *, action_horizon: int, max_episodes: int) -> dict:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on local env
        return {
            "available": False,
            "reason": f"pyarrow unavailable: {type(exc).__name__}: {exc}",
        }

    state_values = []
    action_values = []
    relative_action_values = []
    for episode_index in split_indices[:max_episodes]:
        path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = table["observation.state"].to_pylist()
        actions = table["action"].to_pylist()
        state_values.extend(states)
        action_values.extend(actions)
        for anchor_idx in range(max(0, len(actions) - action_horizon + 1)):
            anchor_state = states[anchor_idx]
            for h in range(action_horizon):
                relative_action_values.append(
                    [a - s for a, s in zip(actions[anchor_idx + h], anchor_state, strict=True)]
                )

    def vector_summary(rows: list[list[float]]) -> dict:
        if not rows:
            return {"count": 0}
        dim = len(rows[0])
        means = [statistics.mean(row[i] for row in rows) for i in range(dim)]
        stds = [statistics.pstdev(row[i] for row in rows) for i in range(dim)]
        return {
            "count": len(rows),
            "dim": dim,
            "mean_abs_mean": statistics.mean(abs(x) for x in means),
            "mean_std": statistics.mean(stds),
            "max_std": max(stds),
        }

    return {
        "available": True,
        "max_episodes": max_episodes,
        "state": vector_summary(state_values),
        "action": vector_summary(action_values),
        "relative_action": vector_summary(relative_action_values),
    }


def _write_markdown(path: Path, split: dict, analysis: dict) -> None:
    train = analysis["splits"]["train"]
    val = analysis["splits"]["val"]
    total_frames = train["num_frames"] + val["num_frames"]
    val_frame_ratio = val["num_frames"] / total_frames if total_frames else math.nan

    lines = [
        "# Franka/ORCA Bag-Groceries Split Analysis",
        "",
        f"- Dataset: `{split['dataset_root']}`",
        f"- Strategy: `{split['strategy']}`",
        f"- Seed: `{split['seed']}`",
        f"- Train episodes: {train['num_episodes']}",
        f"- Val episodes: {val['num_episodes']}",
        f"- Val frame ratio: {val_frame_ratio:.4f}",
        "",
        "## Episode Length",
        "",
        f"- Train mean/std: {train['length']['mean']:.2f} / {train['length']['std']:.2f}",
        f"- Val mean/std: {val['length']['mean']:.2f} / {val['length']['std']:.2f}",
        f"- Train min/median/max: {train['length']['min']} / {train['length']['median']} / {train['length']['max']}",
        f"- Val min/median/max: {val['length']['min']} / {val['length']['median']} / {val['length']['max']}",
        "",
        "## Task Counts",
        "",
        f"- Train: `{train['tasks']}`",
        f"- Val: `{val['tasks']}`",
        "",
        "## Video Counts",
        "",
    ]
    for key, count in analysis["video_counts"].items():
        lines.append(f"- `{key}`: {count}")

    lines.extend(["", "## Parquet Vector Stats", ""])
    parquet_stats = analysis["parquet_stats"]
    if not parquet_stats["train"]["available"] or not parquet_stats["val"]["available"]:
        reason = parquet_stats["train"].get("reason") or parquet_stats["val"].get("reason")
        lines.append(f"- Skipped: {reason}")
    else:
        lines.append(f"- Train relative action mean std: {parquet_stats['train']['relative_action']['mean_std']:.6f}")
        lines.append(f"- Val relative action mean std: {parquet_stats['val']['relative_action']['mean_std']:.6f}")

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "- `*_episode_table.csv`: per-episode split, length, and length-bin assignment.",
            "- `*_length_bins.svg`: train/val counts per length-stratification bin.",
            "- `*_mean_episode_length.svg`: train/val mean episode length with std bars.",
            "",
            "Video mean-frame visualization is not generated by this stdlib-only script. Use an environment with",
            "`opencv-python`/`imageio` or `ffmpeg` if average RGB frame contact sheets are needed.",
        ]
    )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/cluster/scratch/eugseo/datasets/bag_groceries_communal"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--max-parquet-episodes", type=int, default=20)
    args = parser.parse_args()

    episodes = _read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(args.dataset_root / "meta" / "tasks.jsonl")
    train_indices, val_indices = _split_episodes(episodes, args.val_ratio, args.num_bins, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bag_groceries_seed{args.seed}_{int((1 - args.val_ratio) * 100)}_{int(args.val_ratio * 100)}"
    split_path = args.output_dir / f"{stem}.json"
    analysis_path = args.output_dir / f"{stem}_analysis.json"
    markdown_path = args.output_dir / f"{stem}_analysis.md"
    episode_csv_path = args.output_dir / f"{stem}_episode_table.csv"
    bin_svg_path = args.output_dir / f"{stem}_length_bins.svg"
    mean_length_svg_path = args.output_dir / f"{stem}_mean_episode_length.svg"
    bin_by_episode = _bin_by_episode(episodes, args.num_bins)

    split = {
        "dataset": "bag_groceries_communal",
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "unit": "episode",
        "strategy": "length_stratified_random",
        "val_ratio": args.val_ratio,
        "num_bins": args.num_bins,
        "train_episode_indices": train_indices,
        "val_episode_indices": val_indices,
    }

    analysis = {
        "tasks_jsonl": tasks,
        "splits": {
            "train": _split_summary(episodes, train_indices, bin_by_episode),
            "val": _split_summary(episodes, val_indices, bin_by_episode),
        },
        "video_counts": _video_counts(args.dataset_root),
        "parquet_stats": {
            "train": _maybe_parquet_stats(
                args.dataset_root,
                train_indices,
                action_horizon=args.action_horizon,
                max_episodes=args.max_parquet_episodes,
            ),
            "val": _maybe_parquet_stats(
                args.dataset_root,
                val_indices,
                action_horizon=args.action_horizon,
                max_episodes=args.max_parquet_episodes,
            ),
        },
    }

    split_path.write_text(json.dumps(split, indent=2) + "\n")
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    _write_episode_csv(episode_csv_path, episodes, train_indices, val_indices, bin_by_episode)
    _write_bin_svg(bin_svg_path, episodes, train_indices, val_indices, bin_by_episode)
    _write_mean_length_svg(mean_length_svg_path, analysis)
    _write_markdown(markdown_path, split, analysis)

    print(f"Wrote split: {split_path}")
    print(f"Wrote analysis: {analysis_path}")
    print(f"Wrote report: {markdown_path}")
    print(f"Wrote episode table: {episode_csv_path}")
    print(f"Wrote length-bin plot: {bin_svg_path}")
    print(f"Wrote mean-length plot: {mean_length_svg_path}")


if __name__ == "__main__":
    main()
