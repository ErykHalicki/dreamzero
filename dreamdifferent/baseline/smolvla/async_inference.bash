#!/bin/bash
set -euo pipefail

# Defaults — override via env vars or CLI flags.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/work/courses/3dv/team21/workspace/dreamzero/dreamdifferent/baseline/smolvla/smolvla_finetune_so101_bottle/checkpoints/last/pretrained_model}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:50001}"
TASK="${TASK:-Pick up the bottle}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_PATH="${1#*=}"
            shift
            ;;
        --server)
            SERVER_ADDRESS="$2"
            shift 2
            ;;
        --server=*)
            SERVER_ADDRESS="${1#*=}"
            shift
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--checkpoint <path>] [--server <host:port>] [--task <prompt>]" >&2
            exit 1
            ;;
    esac
done

python /Users/dhkim/workspace/ETH/3DVision/dreamdifferent/dreamzero/dreamdifferent/so101/scripts/goto_start_pose2.py

python -m lerobot.async_inference.robot_client \
    --server_address=$SERVER_ADDRESS \
    --robot.type=so101_follower \
    --robot.port="/dev/tty.usbmodem5B140333651" \
    --robot.calibration_dir="/Users/dhkim/workspace/ETH/3DVision/dreamdifferent/dreamzero/dreamdifferent/so101/config/calibration/robots/so_follower/" \
    --robot.id=follower \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
    --task="$TASK" \
    --policy_type=smolvla \
    --pretrained_name_or_path=$CHECKPOINT_PATH \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.3 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
