#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048MB
#SBATCH --output=convert_dataset.out

module load eth_proxy #need to load this to access anything outside the ETH network on a cluster node
cd ~
source .venv/bin/activate
cd dreamzero

rm -r /cluster/scratch/ehalicki/egoverse/bag_groceries/images #clear the images directory in case the last run didnt shut down gracefully
python dreamdifferent/scripts/egoverse_to_lerobotv2.py \
    --src_dir /cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/ \
    --tgt_path /cluster/scratch/ehalicki \
    --task_type bag_groceries \
