#!/bin/bash
#SBATCH --time=24:00:00

RUN_NAME=smolvla_finetune_so101_multi_object_new_frozen_encoder

SMOLVLA_ROOT=/work/courses/3dv/team21/workspace/dreamzero/dreamdifferent/baseline/smolvla
VENV=$SMOLVLA_ROOT/.venv
DATASET_ROOT=/work/courses/3dv/team21/datasets/so101_multi_object_new

source $VENV/bin/activate
cd $SMOLVLA_ROOT
export LD_LIBRARY_PATH=$SMOLVLA_ROOT/.venv/lib/ffmpeg:$LD_LIBRARY_PATH
uv pip install -r requirements.txt

accelerate launch --mixed_precision=bf16 $SMOLVLA_ROOT/train.py \
  --dataset.repo_id=dreamdifferent/so101_multi_object_new \
  --dataset.root=$DATASET_ROOT \
  --output_dir=/home/ehalicki/$RUN_NAME \
  --job_name=$RUN_NAME \
  --policy.repo_id=ehalicki/$RUN_NAME \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=false \
  --rename_map='{"observation.images.wrist": "observation.images.camera1", "observation.images.front": "observation.images.camera2"}' \
  --tolerance_s=0.04 \
  --num_workers=2 \
  --log_freq=1 \
  --save_freq=5000 \
  --steps=50000 \
  --batch_size=62
