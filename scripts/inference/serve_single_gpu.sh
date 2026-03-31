#!/usr/bin/env bash
# Single-GPU inference server for DreamZero (GB10 / any single CUDA device).
#
# Usage:
#   bash scripts/inference/serve_single_gpu.sh
#
# Override defaults with env vars or flags:
#   bash scripts/inference/serve_single_gpu.sh \
#       --model-path /path/to/checkpoint \
#       --port 8000 \
#       --cuda-device 0 \
#       --enable-dit-cache

set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
MODEL_PATH="${MODEL_PATH:-./checkpoints/DreamZero-DROID}"
PORT="${PORT:-8000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ENABLE_DIT_CACHE=false

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)       MODEL_PATH="$2";  shift 2 ;;
        --port)             PORT="$2";        shift 2 ;;
        --cuda-device)      CUDA_DEVICE="$2"; shift 2 ;;
        --enable-dit-cache) ENABLE_DIT_CACHE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path PATH       Path to DreamZero checkpoint (default: ./checkpoints/DreamZero-DROID)"
            echo "  --port PORT             WebSocket server port (default: 8000)"
            echo "  --cuda-device ID        CUDA device index (default: 0)"
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
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: checkpoint directory not found: $MODEL_PATH" >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export ATTENTION_BACKEND="torch"
export HYDRA_FULL_ERROR=1
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
)
if [[ "$ENABLE_DIT_CACHE" == "true" ]]; then
    PYTHON_ARGS+=(--enable-dit-cache)
fi

echo "=========================================="
echo "DreamZero Single-GPU Inference Server"
echo "  Checkpoint  : $MODEL_PATH"
echo "  Port        : $PORT"
echo "  CUDA device : $CUDA_DEVICE"
echo "  DiT cache   : $ENABLE_DIT_CACHE"
echo "=========================================="

# Set distributed env vars manually (avoids torchrun subprocess spawn which
# triggers an ld.so TLS assertion crash on ARM64 / GB10 nodes).
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python "${REPO_ROOT}/serve_single_gpu.py" "${PYTHON_ARGS[@]}"
