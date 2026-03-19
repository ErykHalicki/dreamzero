#!/bin/bash

#SBATCH --time=10:00
#SBATCH --output=convert_dataset_test.out

module load eth_proxy #need to load this to access anything outside the ETH network on a cluster node
cd ~
source .venv/bin/activate
cd dreamzero
python dreamdifferent/scripts/egoverse_to_lerobotv2.py \
    --src_dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/ \
    --tgt_path /cluster/scratch/ehalicki \
    --task_type bag_groceries \
    --debug
