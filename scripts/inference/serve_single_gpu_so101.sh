#!/usr/bin/env bash
# Single-GPU inference server for DreamZero SO101 (GB10 / any single CUDA device).
#
# Usage:
#   bash scripts/inference/serve_single_gpu_so101.sh --model-path /path/to/checkpoint
#
# Override defaults with env vars or flags:
#   bash scripts/inference/serve_single_gpu_so101.sh \
#       --model-path /path/to/checkpoint \
#       --port 23261 \
#       --cuda-device 0 \
#       --tokenizer-path /tmp/pretrained_checkpoints/umt5-xxl \
#       --enable-dit-cache

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
MODEL_PATH="${MODEL_PATH:-}"
PORT="${PORT:-23261}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EULER_HOST="${EULER_HOST:-dohkim@euler.ethz.ch}"
EULER_CKPT_DIR="${EULER_CKPT_DIR:-/cluster/scratch/dohkim/pretrained_checkpoints}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/tmp/pretrained_checkpoints}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${LOCAL_CKPT_DIR}/umt5-xxl}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-${LOCAL_CKPT_DIR}/Wan2.1-I2V-14B-480P}"
ENABLE_DIT_CACHE=false

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)       MODEL_PATH="$2";  shift 2 ;;
        --port)             PORT="$2";        shift 2 ;;
        --cuda-device)      CUDA_DEVICE="$2"; shift 2 ;;
        --tokenizer-path)   TOKENIZER_PATH="$2"; shift 2 ;;
        --enable-dit-cache) ENABLE_DIT_CACHE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH       Path to DreamZero SO101 checkpoint (or set MODEL_PATH)"
            echo "  --port PORT             WebSocket server port (default: 23261)"
            echo "  --cuda-device ID        CUDA device index (default: 0)"
            echo "  --tokenizer-path PATH   Tokenizer path (default: /tmp/pretrained_checkpoints/umt5-xxl)"
            echo "  --enable-dit-cache      Enable DiT KV-cache (faster but more VRAM)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------- #
# Validate
# --------------------------------------------------------------------------- #
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path PATH or MODEL_PATH is required" >&2
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: checkpoint directory not found: $MODEL_PATH" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Pretrained component checkpoints
# --------------------------------------------------------------------------- #
mkdir -p "$LOCAL_CKPT_DIR"

wan_checkpoint_complete() {
    [[ -f "${WAN_CKPT_DIR}/config.json" ]] || return 1
    [[ -f "${WAN_CKPT_DIR}/diffusion_pytorch_model.safetensors.index.json" ]] || return 1
    [[ -f "${WAN_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth" ]] || return 1
    [[ -f "${WAN_CKPT_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]] || return 1
    [[ -f "${WAN_CKPT_DIR}/Wan2.1_VAE.pth" ]] || return 1
    compgen -G "${WAN_CKPT_DIR}/diffusion_pytorch_model-*.safetensors" >/dev/null
}

if ! wan_checkpoint_complete; then
    echo "=== Syncing Wan2.1-I2V-14B-480P from Euler (required files only, ~51 GB)... ==="
    rsync -avP \
        --include='diffusion_pytorch_model-*.safetensors' \
        --include='diffusion_pytorch_model.safetensors.index.json' \
        --include='models_t5_umt5-xxl-enc-bf16.pth' \
        --include='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth' \
        --include='Wan2.1_VAE.pth' \
        --include='config.json' \
        --exclude='*' \
        "${EULER_HOST}:${EULER_CKPT_DIR}/Wan2.1-I2V-14B-480P/" "${WAN_CKPT_DIR}/"
    if ! wan_checkpoint_complete; then
        echo "Error: Wan2.1-I2V-14B-480P sync did not produce a complete checkpoint at ${WAN_CKPT_DIR}" >&2
        exit 1
    fi
else
    echo "=== Wan2.1-I2V-14B-480P already present, skipping sync ==="
fi

if [[ "$TOKENIZER_PATH" == "${LOCAL_CKPT_DIR}/umt5-xxl" ]]; then
    if [[ ! -f "${TOKENIZER_PATH}/spiece.model" ]]; then
        echo "=== Syncing umt5-xxl tokenizer from Euler (~5 MB)... ==="
        rsync -avP \
            --include='config.json' \
            --include='spiece.model' \
            --include='tokenizer_config.json' \
            --include='special_tokens_map.json' \
            --exclude='*' \
            "${EULER_HOST}:${EULER_CKPT_DIR}/umt5-xxl/" "${TOKENIZER_PATH}/"
    else
        echo "=== umt5-xxl tokenizer already present, skipping sync ==="
    fi
fi
if [[ ! -f "${TOKENIZER_PATH}/spiece.model" && "$TOKENIZER_PATH" != "google/umt5-xxl" ]]; then
    echo "Error: tokenizer not found at $TOKENIZER_PATH" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export ATTENTION_BACKEND="torch"
export HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-/tmp/huggingface_${USER}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER}}"
export DEEPSPEED_AUTOTUNE_CACHE_DIR="${DEEPSPEED_AUTOTUNE_CACHE_DIR:-/tmp/deepspeed_${USER}}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRITON_CACHE_DIR" "$DEEPSPEED_AUTOTUNE_CACHE_DIR"
# Triton's bundled ptxas does not recognise sm_121a (GB10, CUDA 12.1).
# Use the system ptxas (CUDA 13.0) which supports sm_121a.
export TRITON_PTXAS_PATH=$(which ptxas)
# Prevent ld.so TLS assertion crash on ARM64 (GB10) caused by conflicting
# OpenMP runtimes (libgomp vs libiomp5) loaded by conda packages.
export KMP_DUPLICATE_LIB_OK=TRUE
unset LD_PRELOAD

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_ARGS=(
    --model-path "$MODEL_PATH"
    --port       "$PORT"
    --tokenizer-path "$TOKENIZER_PATH"
)
if [[ "$ENABLE_DIT_CACHE" == "true" ]]; then
    PYTHON_ARGS+=(--enable-dit-cache)
fi

echo "=========================================="
echo "DreamZero SO101 Single-GPU Inference Server"
echo "  Checkpoint  : $MODEL_PATH"
echo "  Port        : $PORT"
echo "  CUDA device : $CUDA_DEVICE"
echo "  Tokenizer   : $TOKENIZER_PATH"
echo "  DiT cache   : $ENABLE_DIT_CACHE"
echo "=========================================="

# Set distributed env vars manually (avoids torchrun subprocess spawn which
# triggers an ld.so TLS assertion crash on ARM64 / GB10 nodes).
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python "${REPO_ROOT}/serve_single_gpu_so101.py" "${PYTHON_ARGS[@]}"
