#!/bin/bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-smolvla_finetune_so101_bottle}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-last}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)
            RUN_NAME="$2"
            shift 2
            ;;
        --run=*)
            RUN_NAME="${1#*=}"
            shift
            ;;
        --checkpoint)
            CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_STEP="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--run <run_name>] [--checkpoint <step|last>]" >&2
            exit 1
            ;;
    esac
done

SMOLVLA_ROOT=/work/courses/3dv/team21/workspace/dreamzero/dreamdifferent/baseline/smolvla
VENV=$SMOLVLA_ROOT/.venv
CHECKPOINT_PATH=$SMOLVLA_ROOT/$RUN_NAME/checkpoints/$CHECKPOINT_STEP/pretrained_model

source $VENV/bin/activate
export LD_LIBRARY_PATH=$SMOLVLA_ROOT/.venv/lib/ffmpeg:$LD_LIBRARY_PATH

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi

echo "Checkpoint available at:"
echo "  $CHECKPOINT_PATH"
echo "Pass this path to the client as --pretrained_name_or_path."

python -m lerobot.async_inference.policy_server \
     --host=0.0.0.0 \
     --port=50001
