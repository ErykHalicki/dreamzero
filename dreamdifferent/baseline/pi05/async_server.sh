#!/bin/bash
set -euo pipefail

if [[ -f "/cluster/home/dohkim/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "/cluster/home/dohkim/miniforge3/etc/profile.d/conda.sh"
    conda activate lerobot_pi05
fi

SCRATCH_ROOT="/cluster/scratch/${USER}/pi05"
HF_HOME="${SCRATCH_ROOT}/hf_home"
HF_HUB_CACHE="${HF_HOME}/hub"
HF_DATASETS_CACHE="${HF_HOME}/datasets"
HF_ASSETS_CACHE="${HF_HOME}/assets"
TRANSFORMERS_CACHE="${HF_HOME}/transformers"
HF_LEROBOT_HOME="${SCRATCH_ROOT}/lerobot"

export HF_HOME
export HF_HUB_CACHE
export HF_DATASETS_CACHE
export HF_ASSETS_CACHE
export TRANSFORMERS_CACHE
export HF_LEROBOT_HOME
unset LEROBOT_HOME

EXP_NAME="${EXP_NAME:-pi05_homogeneous_lora_a100}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-025000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
FPS="${FPS:-30}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)
            EXP_NAME="$2"
            shift 2
            ;;
        --exp=*)
            EXP_NAME="${1#*=}"
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
        --host)
            HOST="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --fps=*)
            FPS="${1#*=}"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

EXP_ROOT="${SCRATCH_ROOT}/experiments/${EXP_NAME}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXP_ROOT}/outputs/checkpoints/${CHECKPOINT_STEP}/pretrained_model}"

exec python server.py \
    --checkpoint="${CHECKPOINT_PATH}" \
    --host="${HOST}" \
    --port="${PORT}" \
    --fps="${FPS}" \
    "${EXTRA_ARGS[@]}"
