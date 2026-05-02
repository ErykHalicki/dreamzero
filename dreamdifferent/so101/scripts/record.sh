#!/usr/bin/env bash
set -euo pipefail

# Resume/download an existing LeRobot dataset from the Hub main branch, then append new episodes.
# python scripts/teleop_record.py \
#     --camera --camera-index 0 --camera-name wrist \
#     --secondary-camera --secondary-camera-index 2 --secondary-camera-name front \
#     --episode-time-sec 120 \
#     --return-move-time-sec 4 \
#     --num-episodes 10 \
#     --dataset-repo-id dreamdifferent/so101_teleop_test2 \
#     --push-to-hub \
#     --task "pick up the black tape and place it into the container" \
#     --resume \
#     --revision main

python scripts/teleop_record.py \
    --camera --camera-index 0 --camera-name wrist \
    --secondary-camera --secondary-camera-index 2 --secondary-camera-name front \
    --episode-time-sec 120 \
    --return-move-time-sec 4 \
    --num-episodes 10 \
    --dataset-repo-id dreamdifferent/so101_shuttlecock \
    --push-to-hub \
    --task "pick up the shuttlecock and place it into the container" \
    --resume \
    --revision main