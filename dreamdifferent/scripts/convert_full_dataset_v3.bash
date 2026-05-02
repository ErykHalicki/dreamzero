#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=convert_dataset_v3.out

set -euo pipefail

module load eth_proxy

DREAMZERO_ROOT=/cluster/home/ehalicki/dreamzero
SMOLVLA_VENV="${SMOLVLA_VENV:-$DREAMZERO_ROOT/dreamdifferent/baseline/smolvla/.venv}"

source "$SMOLVLA_VENV/bin/activate"
cd "$DREAMZERO_ROOT"

python dreamdifferent/scripts/egoverse_to_lerobotv3.py \
    --src_dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/ \
    --tgt_path /cluster/scratch/ehalicki \
    --task_type bag_groceries
