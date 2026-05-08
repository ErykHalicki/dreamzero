#!/bin/bash
set -e
export HYDRA_FULL_ERROR=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6

# ============ CONFIGURATION ============
DATA_ROOT="/work/scratch/dohkim/datasets/so101_bottle_lerobot_v2" 
# OUTPUT_DIR="/work/scratch/dohkim/checkpoints/dreamzero_so101_bottle_lora"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

# Remote checkpoint locations (Euler)
EULER_HOST="dohkim@euler.ethz.ch"
EULER_CKPT_DIR="/cluster/scratch/dohkim/pretrained_checkpoints"

# Local temp directory for checkpoints (wiped after job ends)
LOCAL_CKPT_DIR="/tmp/pretrained_checkpoints"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_OUTPUT_DIR="/tmp/dreamzero_so101_bottle_lora_output_${TIMESTAMP}"

# Optional resume source. Point this at either a previous output directory or a
# specific checkpoint-* directory. The script copies it into the fresh /tmp run
# directory so HuggingFace Trainer can auto-detect and resume.
RESUME_FROM="${RESUME_FROM:-}"

# Where to sync training output back to on Euler
EULER_OUTPUT_DIR="/cluster/scratch/dohkim/training_output/dreamzero_so101_bottle_lora_${TIMESTAMP}"

# Local shared-filesystem copy (persists after job ends)
LOCAL_SHARED_OUTPUT_DIR="/work/scratch/dohkim/training_output/dreamzero_so101_bottle_lora_${TIMESTAMP}"
# =======================================

# ============ RSYNC CHECKPOINTS FROM EULER → /tmp ============
mkdir -p "$LOCAL_CKPT_DIR"

# Only sync Wan2.1 checkpoints if not already present
if [ ! -f "${LOCAL_CKPT_DIR}/Wan2.1-I2V-14B-480P/config.json" ]; then
    echo "=== Syncing Wan2.1-I2V-14B-480P from Euler (required files only, ~51 GB)... ==="
    rsync -avP \
        --include='diffusion_pytorch_model-*.safetensors' \
        --include='diffusion_pytorch_model.safetensors.index.json' \
        --include='models_t5_umt5-xxl-enc-bf16.pth' \
        --include='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth' \
        --include='Wan2.1_VAE.pth' \
        --include='config.json' \
        --exclude='*' \
        "${EULER_HOST}:${EULER_CKPT_DIR}/Wan2.1-I2V-14B-480P/" "${LOCAL_CKPT_DIR}/Wan2.1-I2V-14B-480P/"
else
    echo "=== Wan2.1-I2V-14B-480P already present, skipping sync ==="
fi

# Only sync tokenizer if not already present
if [ ! -f "${LOCAL_CKPT_DIR}/umt5-xxl/spiece.model" ]; then
    echo "=== Syncing umt5-xxl tokenizer from Euler (~5 MB)... ==="
    rsync -avP \
        --include='config.json' \
        --include='spiece.model' \
        --include='tokenizer_config.json' \
        --include='special_tokens_map.json' \
        --exclude='*' \
        "${EULER_HOST}:${EULER_CKPT_DIR}/umt5-xxl/" "${LOCAL_CKPT_DIR}/umt5-xxl/"
else
    echo "=== umt5-xxl tokenizer already present, skipping sync ==="
fi

echo "=== Checkpoint check complete ==="

WAN_CKPT_DIR="${LOCAL_CKPT_DIR}/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="${LOCAL_CKPT_DIR}/umt5-xxl"
PRETRAINED_MODEL="/work/scratch/dohkim/checkpoints/DreamZero-DROID"
OUTPUT_DIR="${LOCAL_OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOCAL_SHARED_OUTPUT_DIR"

# ============ OPTIONAL RESUME ============
if [ -n "$RESUME_FROM" ]; then
    echo "=== Preparing resume from: $RESUME_FROM ==="

    RESUME_BASENAME=$(basename "${RESUME_FROM%/}")
    if [[ "$RESUME_BASENAME" == checkpoint-* ]]; then
        mkdir -p "${OUTPUT_DIR}/${RESUME_BASENAME}"
        rsync -a --update "${RESUME_FROM%/}/" "${OUTPUT_DIR}/${RESUME_BASENAME}/"
    else
        rsync -a --update "${RESUME_FROM%/}/" "${OUTPUT_DIR}/"
    fi

    LATEST_RESUME_CKPT=$(ls -td "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    if [ -z "$LATEST_RESUME_CKPT" ]; then
        echo "ERROR: RESUME_FROM did not provide any checkpoint-* directory"
        exit 1
    fi
    if [ ! -f "${LATEST_RESUME_CKPT}/trainer_state.json" ]; then
        echo "ERROR: ${LATEST_RESUME_CKPT}/trainer_state.json missing; cannot resume Trainer state"
        exit 1
    fi
    echo "=== Resume checkpoint staged: $LATEST_RESUME_CKPT ==="
fi

# Copy this training script into the output dir for reproducibility
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/so101_training.sh"
cp "$SCRIPT_PATH" "$LOCAL_SHARED_OUTPUT_DIR/so101_training.sh"

# ============ VALIDATION ============
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi

# ============ SYNC OUTPUT BACK TO EULER ============
# Interval (seconds) between periodic checkpoint syncs
SYNC_INTERVAL=${SYNC_INTERVAL:-1200}   # default: every 20 minutes

sync_output_to_euler() {
    echo "=== Syncing training output ($(date)) ==="
    # Sync to Euler scratch (all checkpoints)
    rsync -a --update "${OUTPUT_DIR}/" "${EULER_HOST}:${EULER_OUTPUT_DIR}/" && \
        echo "=== Euler sync complete ===" || \
        echo "=== WARNING: Euler sync failed ==="

    # Sync to local shared filesystem — only the latest checkpoint + non-checkpoint files
    # Find the most recent checkpoint-* directory by modification time
    LATEST_CKPT=$(ls -td "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        LATEST_CKPT_NAME=$(basename "$LATEST_CKPT")
        # Remove any old checkpoint dirs from the shared output
        for old_ckpt in "${LOCAL_SHARED_OUTPUT_DIR}"/checkpoint-*; do
            if [ -d "$old_ckpt" ] && [ "$(basename "$old_ckpt")" != "$LATEST_CKPT_NAME" ]; then
                echo "=== Removing old local checkpoint: $(basename "$old_ckpt") ==="
                rm -rf "$old_ckpt"
            fi
        done
        # Sync the latest checkpoint
        rsync -a --update "$LATEST_CKPT/" "${LOCAL_SHARED_OUTPUT_DIR}/${LATEST_CKPT_NAME}/"
    fi
    # Sync non-checkpoint files (logs, script, etc.) — exclude wandb & checkpoint dirs
    rsync -a --update --exclude='checkpoint-*' --exclude='wandb/' --exclude='wandb-*' "${OUTPUT_DIR}/" "${LOCAL_SHARED_OUTPUT_DIR}/" && \
        echo "=== Local shared sync complete ===" || \
        echo "=== WARNING: local shared sync failed ==="
}

# Periodic background sync — ensures checkpoints reach Euler
# even if the node is killed without warning.
periodic_sync() {
    while true; do
        sleep "$SYNC_INTERVAL"
        sync_output_to_euler
    done
}
periodic_sync &
SYNC_PID=$!

# ============ CLEANUP & SIGNAL HANDLING ============
# When the SLURM time limit hits, the scheduler sends SIGTERM first
# (with a short grace period before SIGKILL). We trap it to:
#   1. kill the training process so we reclaim the foreground
#   2. do a final sync within the grace window
CLEANUP_DONE=0
cleanup() {
    # Guard against running twice (SIGTERM → exit → EXIT trap)
    if [ "$CLEANUP_DONE" -eq 1 ]; then return; fi
    CLEANUP_DONE=1
    echo "=== Caught signal / exit — cleaning up ($(date)) ==="
    # Stop background helpers
    kill $SYNC_PID  2>/dev/null || true
    kill $MONITOR_PID 2>/dev/null || true
    kill $TRAIN_PID 2>/dev/null || true
    wait $TRAIN_PID 2>/dev/null || true
    # No final sync — rely on periodic background sync
}
trap cleanup SIGTERM SIGINT EXIT

# ============ MEMORY MONITOR (GB10 unified memory) ============
monitor_memory() {
    while true; do
        echo "===== MEMORY @ $(date +%H:%M:%S) ====="
        echo "-- System RAM --"
        free -h | awk 'NR==1||NR==2{print}'
        echo "================================="
        sleep 20
    done
}
monitor_memory &
MONITOR_PID=$!

# ============ TRAINING ============
# Run in background so traps can fire while training is in progress
torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/so101_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=17 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    gradient_checkpointing=true \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=50 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    global_batch_size=2 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=0 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=2 \
    frame_seqlen=880 \
    save_strategy=steps \
    training_args.logging_steps=1 \
    so101_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL \
    ++train_dataset.dataset_kwargs.num_steps_per_shard=6000 \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true &
TRAIN_PID=$!
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "=== Training exited with code $TRAIN_EXIT ==="
# Final sync handled by the EXIT trap
