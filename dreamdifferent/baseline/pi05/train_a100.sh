#!/bin/bash
#SBATCH --job-name=pi05_lora
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

if ! command -v module >/dev/null 2>&1 && [[ -f "/etc/profile.d/modules.sh" ]]; then
    source "/etc/profile.d/modules.sh"
fi

if command -v module >/dev/null 2>&1; then
    module load eth_proxy
else
    echo "WARNING: environment modules are not available; skipping 'module load eth_proxy'"
fi

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

EXP_NAME="${EXP_NAME:-pi05_homogeneous_lora_a100}"
EXP_ROOT="${EXP_ROOT:-${SCRATCH_ROOT}/experiments/${EXP_NAME}}"

SOURCE_DATASET_REPO_ID="${SOURCE_DATASET_REPO_ID:-dreamdifferent/so101_bottle}"
DATASET_SLUG="${DATASET_SLUG:-${SOURCE_DATASET_REPO_ID##*/}}"
RAW_DATASET_ROOT="${RAW_DATASET_ROOT:-${EXP_ROOT}/datasets/${DATASET_SLUG}/raw}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_DATASET_REPO_ID}_relative_stats}"
DATASET_ROOT="${DATASET_ROOT:-${EXP_ROOT}/datasets/${DATASET_SLUG}/relative_stats}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXP_ROOT}/outputs}"
JOB_NAME="${JOB_NAME:-${EXP_NAME}}"
POLICY_REPO_ID="${POLICY_REPO_ID:-dohkim/${EXP_NAME}}"
STEPS="${STEPS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
RESUME="${RESUME:-false}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-.*\\.paligemma\\.model\\.language_model\\..*\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)|.*\\.gemma_expert\\..*\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)|model\\.(state_proj|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|action_time_mlp_in|action_time_mlp_out)}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-pi05_so101}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
FORCE_CONVERT="${FORCE_CONVERT:-false}"

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    STEPS="$1"
    shift
fi

mkdir -p \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_ASSETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${HF_LEROBOT_HOME}" \
    "${EXP_ROOT}/datasets"

export HF_HOME
export HF_HUB_CACHE
export HF_DATASETS_CACHE
export HF_ASSETS_CACHE
export TRANSFORMERS_CACHE
export HF_LEROBOT_HOME
export SOURCE_DATASET_REPO_ID
export RAW_DATASET_ROOT
unset LEROBOT_HOME

if [[ "${FORCE_CONVERT}" == "true" || ! -f "${DATASET_ROOT}/meta/stats.json" ]]; then
    python -c "import os; from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset(os.environ['SOURCE_DATASET_REPO_ID'], root=os.environ['RAW_DATASET_ROOT'], download_videos=True)"

    lerobot-edit-dataset \
        --repo_id "${SOURCE_DATASET_REPO_ID}" \
        --root "${RAW_DATASET_ROOT}" \
        --new_repo_id "${DATASET_REPO_ID}" \
        --new_root "${DATASET_ROOT}" \
        --operation.type recompute_stats \
        --operation.relative_action true \
        --operation.chunk_size 50 \
        --operation.relative_exclude_joints "['gripper']" \
        --push_to_hub false
fi

lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend="${VIDEO_BACKEND}" \
    --policy.type=pi05 \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints='["gripper"]' \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --policy.repo_id="${POLICY_REPO_ID}" \
    --policy.push_to_hub="${PUSH_TO_HUB}" \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model="${COMPILE_MODEL}" \
    --policy.gradient_checkpointing=true \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project="${WANDB_PROJECT}" \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}" \
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
    --steps="${STEPS}" \
    --resume="${RESUME}" \
    --log_freq="${LOG_FREQ}" \
    --save_freq="${SAVE_FREQ}" \
    --policy.device=cuda \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --peft.method_type=LORA \
    --peft.target_modules="${PEFT_TARGET_MODULES}" \
    --peft.r="${LORA_R}" \
    --peft.lora_alpha="${LORA_ALPHA}" \
    "$@"
