#!/usr/bin/env bash
# Run the SO-101 RTC robot client for pi05 with two cameras: front + wrist.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/tty.usbmodem5B140333651}"
ROBOT_ID="${ROBOT_ID:-follower}"
CALIBRATION_DIR="${CALIBRATION_DIR:-config/calibration/robots/so_follower}"
FPS="${FPS:-30}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-$FPS}"
FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-0}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-1}"
REFILL_THRESHOLD="${REFILL_THRESHOLD:-5}"
RENAME_MAP="${RENAME_MAP:-{}}"
: "${TASK:?Set TASK to the exact task string used in the pi05 training dataset.}"

CAMERAS_JSON="{\"front\":{\"index_or_path\":\"$FRONT_CAMERA_INDEX\",\"width\":$CAMERA_WIDTH,\"height\":$CAMERA_HEIGHT,\"fps\":$CAMERA_FPS},\"wrist\":{\"index_or_path\":\"$WRIST_CAMERA_INDEX\",\"width\":$CAMERA_WIDTH,\"height\":$CAMERA_HEIGHT,\"fps\":$CAMERA_FPS}}"

python scripts/goto_start_pose.py --port "$ROBOT_PORT"

exec python scripts/rtc_robot_client_pi05.py \
    --server-address "$SERVER_ADDRESS" \
    --robot-type so101_follower \
    --robot-port "$ROBOT_PORT" \
    --robot-id "$ROBOT_ID" \
    --calibration-dir "$CALIBRATION_DIR" \
    --cameras-json "$CAMERAS_JSON" \
    --fps "$FPS" \
    --task "$TASK" \
    --rename-map "$RENAME_MAP" \
    --refill-threshold "$REFILL_THRESHOLD" \
    "$@"
