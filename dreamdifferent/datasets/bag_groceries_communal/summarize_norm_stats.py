#!/usr/bin/env python3
"""Print a compact summary of OpenPI norm_stats.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_STATS_PATH = (
    Path(__file__).resolve().parents[2]
    / "baseline"
    / "openpi"
    / "assets"
    / "pi05_franka_orca_bag_groceries"
    / "local"
    / "bag_groceries_communal"
    / "norm_stats.json"
)


def _as_list(stats: dict, key: str) -> list[float]:
    values = stats[key]
    if isinstance(values, dict) and "value" in values:
        values = values["value"]
    return [float(v) for v in values]


def _format(values: list[float], max_values: int) -> str:
    shown = ", ".join(f"{v:.6g}" for v in values[:max_values])
    if len(values) > max_values:
        shown += ", ..."
    return f"[{shown}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--max-values", type=int, default=8)
    args = parser.parse_args()

    data = json.loads(args.path.read_text())
    norm_stats = data["norm_stats"]

    print(args.path)
    for key, stats in norm_stats.items():
        print(f"\n[{key}]")
        for stat_name in ("mean", "std", "q01", "q99"):
            values = _as_list(stats, stat_name)
            print(f"{stat_name:>4}: shape=({len(values)},) first={_format(values, args.max_values)}")


if __name__ == "__main__":
    main()
