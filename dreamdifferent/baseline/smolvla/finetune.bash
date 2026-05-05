#!/bin/bash
#SBATCH --job-name=smolvla-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096MB
#SBATCH --time=24:00:00
#SBATCH --gpus=1

RUN_NAME=smolvla_finetune_bag_grocieries

SMOLVLA_ROOT=/work/courses/3dv/team21/workspace/dreamzero/dreamdifferent/baseline/smolvla
VENV=$SMOLVLA_ROOT/.venv
DATASET_ROOT=/work/courses/3dv/team21/datasets/bag_groceries_v3/bag_groceries

source $VENV/bin/activate
cd $SMOLVLA_ROOT
uv pip install -r requirements.txt

lerobot-train \
  --dataset.repo_id=ehalicki/eth-3dv-2026-bimanual-franka-grocery-bagging \
  --dataset.root=$DATASET_ROOT \
  --output_dir=./outputs/$RUN_NAME \
  --job_name=$RUN_NAME \
  --policy.repo_id=ehalicki/eth-3dv-2026-bimanual-franka-grocery-bagging-smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=1
