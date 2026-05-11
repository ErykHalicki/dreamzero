import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


GRIPPER_LIMIT = float(math.pi / 4.0)


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _cmd_state(record: dict, side: str) -> str:
    joint = float(record[f"cmd_{side}_gripper_joint"])
    return "open" if joint > (GRIPPER_LIMIT * 0.5) else "close"


def _raw_value(record: dict, side: str) -> float:
    return float(record[f"raw_{side}_gripper"])


def _step_value(record: dict) -> int:
    return int(record["step"])


def _episode_value(record: dict) -> int:
    return int(record["episode"])


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _print_side_summary(
    *,
    episode: int,
    side: str,
    records: list[dict],
    threshold: float,
    near_band: float,
    max_events: int,
) -> None:
    raw_values = [_raw_value(record, side) for record in records]
    cmd_states = [_cmd_state(record, side) for record in records]

    transitions: list[tuple[int, str, str, float]] = []
    prev_state = cmd_states[0]
    for record, state in zip(records[1:], cmd_states[1:], strict=True):
        if state != prev_state:
            transitions.append((_step_value(record), prev_state, state, _raw_value(record, side)))
            prev_state = state

    first_open_step = next(
        (_step_value(record) for record, state in zip(records, cmd_states, strict=True) if state == "open"),
        None,
    )
    first_close_step = next(
        (_step_value(record) for record, state in zip(records, cmd_states, strict=True) if state == "close"),
        None,
    )

    near_threshold_records = sorted(
        records,
        key=lambda record: abs(_raw_value(record, side) - threshold),
    )
    near_threshold_records = [
        record
        for record in near_threshold_records
        if abs(_raw_value(record, side) - threshold) <= near_band
    ]

    open_count = sum(1 for state in cmd_states if state == "open")
    close_count = len(cmd_states) - open_count

    print(f"[episode {episode}] {side}")
    print(
        "  raw stats:"
        f" min={_format_float(min(raw_values))}"
        f" max={_format_float(max(raw_values))}"
        f" mean={_format_float(sum(raw_values) / len(raw_values))}"
    )
    print(
        "  command states:"
        f" open_steps={open_count}"
        f" close_steps={close_count}"
        f" first_open_step={first_open_step}"
        f" first_close_step={first_close_step}"
        f" transitions={len(transitions)}"
    )

    if transitions:
        print("  transitions:")
        for step, prev_state, next_state, raw in transitions[:max_events]:
            print(
                f"    step={step} {prev_state}->{next_state} raw={_format_float(raw)}"
            )
        if len(transitions) > max_events:
            print(f"    ... {len(transitions) - max_events} more")
    else:
        print("  transitions: none")

    if near_threshold_records:
        print(f"  near-threshold samples (|raw-{threshold:.2f}| <= {near_band:.2f}):")
        for record in near_threshold_records[:max_events]:
            print(
                "    "
                f"step={_step_value(record)} "
                f"raw={_format_float(_raw_value(record, side))} "
                f"cmd={_cmd_state(record, side)} "
                f"obs_before={_format_float(float(record[f'obs_{side}_gripper_before']))} "
                f"obs_after={_format_float(float(record[f'obs_{side}_gripper_after']))}"
            )
        if len(near_threshold_records) > max_events:
            print(f"    ... {len(near_threshold_records) - max_events} more")
    else:
        print("  near-threshold samples: none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Agibot gripper debug JSONL logs.")
    parser.add_argument("path", help="Path to gripper_debug.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold applied to raw gripper values.")
    parser.add_argument(
        "--near-band",
        type=float,
        default=0.1,
        help="Report samples whose raw gripper value lies within this distance of the threshold.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10,
        help="Maximum number of transitions / near-threshold rows to print per side.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    records = _load_records(path)
    if not records:
        raise SystemExit(f"No records found in {path}")

    episodes: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        episodes[_episode_value(record)].append(record)

    print(f"path: {path}")
    print(f"records: {len(records)}")
    print(f"episodes: {sorted(episodes)}")
    print(f"binary threshold: {args.threshold:.2f}")
    print(f"near band: {args.near_band:.2f}")
    print()

    for episode in sorted(episodes):
        episode_records = sorted(episodes[episode], key=_step_value)
        print(f"=== Episode {episode} ===")
        _print_side_summary(
            episode=episode,
            side="left",
            records=episode_records,
            threshold=args.threshold,
            near_band=args.near_band,
            max_events=args.max_events,
        )
        _print_side_summary(
            episode=episode,
            side="right",
            records=episode_records,
            threshold=args.threshold,
            near_band=args.near_band,
            max_events=args.max_events,
        )
        print()


if __name__ == "__main__":
    main()
