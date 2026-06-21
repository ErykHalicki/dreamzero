#!/usr/bin/env bash
# Run the SO-101 RTC robot client matching rollout_smolvla_dohyung.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower}"
CALIBRATION_DIR="${CALIBRATION_DIR:-config/calibration/robots/so_follower}"
CAMERA_NAME="${CAMERA_NAME:-camera1}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
REFILL_THRESHOLD="${REFILL_THRESHOLD:-5}"
TASK="SO101 teleoperation task"

python scripts/goto_start_pose2.py --port "$ROBOT_PORT"

exec python scripts/rtc_robot_client.py \
    --server-address "$SERVER_ADDRESS" \
    --robot-type so101_follower \
    --robot-port "$ROBOT_PORT" \
    --robot-id "$ROBOT_ID" \
    --calibration-dir "$CALIBRATION_DIR" \
    --camera-name "$CAMERA_NAME" \
    --camera-index "$CAMERA_INDEX" \
    --camera-width "$CAMERA_WIDTH" \
    --camera-height "$CAMERA_HEIGHT" \
    --fps "$FPS" \
    --task "$TASK" \
    --refill-threshold "$REFILL_THRESHOLD" \
    "$@"
