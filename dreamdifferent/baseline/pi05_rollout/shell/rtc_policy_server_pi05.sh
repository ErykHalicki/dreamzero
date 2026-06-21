#!/usr/bin/env bash
# Start the custom RTC policy server for a trained pi05 checkpoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
: "${POLICY_PATH:?Set POLICY_PATH to your trained pi05 checkpoint, e.g. /path/to/checkpoints/010000/pretrained_model}"
DEVICE="${DEVICE:-cuda}"
FPS="${FPS:-30}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-10}"
MAX_GUIDANCE_WEIGHT="${MAX_GUIDANCE_WEIGHT:-10.0}"
MAX_ACTIONS_PER_RESPONSE="${MAX_ACTIONS_PER_RESPONSE:-10}"

exec python scripts/rtc_policy_server_pi05.py \
    --host "$HOST" \
    --port "$PORT" \
    --policy-path "$POLICY_PATH" \
    --device "$DEVICE" \
    --fps "$FPS" \
    --execution-horizon "$EXECUTION_HORIZON" \
    --max-guidance-weight "$MAX_GUIDANCE_WEIGHT" \
    --max-actions-per-response "$MAX_ACTIONS_PER_RESPONSE" \
    "$@"
