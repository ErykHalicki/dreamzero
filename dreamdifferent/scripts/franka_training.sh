#!/bin/bash
set -e
export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
DATA_ROOT="/work/courses/3dv/team21/datasets/bag_groceries"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

# Remote checkpoint locations (Euler)
EULER_HOST="rwalia@euler.ethz.ch"
EULER_CKPT_DIR="/cluster/scratch/rwalia/pretrained_checkpoints"

# Local temp directory for checkpoints (wiped after job ends)
LOCAL_CKPT_DIR="/tmp/pretrained_checkpoints"
LOCAL_OUTPUT_DIR="/tmp/dreamzero_franka_lora_output"

# Where to sync training output back to on Euler
EULER_OUTPUT_DIR="/cluster/scratch/rwalia/training_output/dreamzero_franka_lora"
# =======================================

# ============ RSYNC CHECKPOINTS FROM EULER → /tmp ============
echo "=== Syncing checkpoints from Euler to /tmp ==="
mkdir -p "$LOCAL_CKPT_DIR"

echo "  Syncing Wan2.1-I2V-14B-480P (required files only, ~51 GB)..."
rsync -avP \
    --include='diffusion_pytorch_model-*.safetensors' \
    --include='diffusion_pytorch_model.safetensors.index.json' \
    --include='models_t5_umt5-xxl-enc-bf16.pth' \
    --include='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth' \
    --include='Wan2.1_VAE.pth' \
    --include='config.json' \
    --exclude='*' \
    "${EULER_HOST}:${EULER_CKPT_DIR}/Wan2.1-I2V-14B-480P/" "${LOCAL_CKPT_DIR}/Wan2.1-I2V-14B-480P/"

echo "  Syncing umt5-xxl (tokenizer files only, ~5 MB)..."
rsync -avP \
    --include='config.json' \
    --include='spiece.model' \
    --include='tokenizer_config.json' \
    --include='special_tokens_map.json' \
    --exclude='*' \
    "${EULER_HOST}:${EULER_CKPT_DIR}/umt5-xxl/" "${LOCAL_CKPT_DIR}/umt5-xxl/"

echo "=== Checkpoint sync complete ==="

WAN_CKPT_DIR="${LOCAL_CKPT_DIR}/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="${LOCAL_CKPT_DIR}/umt5-xxl"
PRETRAINED_MODEL="/work/scratch/rwalia/checkpoints/DreamZero-AgiBot"
OUTPUT_DIR="${LOCAL_OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

# ============ VALIDATION ============
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi

# ============ SYNC OUTPUT BACK TO EULER (trap on exit) ============
sync_output_to_euler() {
    echo "=== Syncing training output back to Euler ==="
    rsync -avP "${OUTPUT_DIR}/" "${EULER_HOST}:${EULER_OUTPUT_DIR}/"
    echo "=== Output sync complete ==="
}
trap sync_output_to_euler EXIT

# ============ TRAINING ============
torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/franka_relative \
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
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_steps=5000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=20000 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=1 \
    frame_seqlen=880 \
    save_strategy=steps \
    franka_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_MODEL \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
