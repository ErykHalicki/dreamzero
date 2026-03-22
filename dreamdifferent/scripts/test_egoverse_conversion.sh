#!/bin/bash
set -e

python dreamdifferent/scripts/egoverse_to_lerobotv2.py \
    --src_dir dreamdifferent/data/groceries \
    --tgt_path /tmp/egoverse_test \
    --task_type bag_groceries \
