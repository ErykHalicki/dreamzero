#!/bin/bash
#SBATCH --time=24:00:00

RUN_NAME=smolvla_finetune_bag_grocieries

SMOLVLA_ROOT=/work/courses/3dv/team21/workspace/dreamzero/dreamdifferent/baseline/smolvla
VENV=$SMOLVLA_ROOT/.venv
DATASET_ROOT=/work/courses/3dv/team21/datasets/bag_groceries_v3/bag_groceries

source $VENV/bin/activate
cd $SMOLVLA_ROOT
uv pip install -r requirements.txt

accelerate launch --mixed_precision=bf16 $SMOLVLA_ROOT/train.py \
  --dataset.repo_id=ehalicki/eth-3dv-2026-bimanual-franka-grocery-bagging \
  --dataset.root=$DATASET_ROOT \
  --output_dir=./outputs/$RUN_NAME \
  --job_name=$RUN_NAME \
  --policy.repo_id=ehalicki/eth-3dv-2026-bimanual-franka-grocery-bagging-smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.max_action_dim=48 \
  --rename_map='{"observation.images.aria_rgb_cam": "observation.images.camera1", "observation.images.oakd_front_view": "observation.images.camera2"}' \
  --steps=100000 \
  --batch_size=1
