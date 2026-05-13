#!/usr/bin/env bash
set -euo pipefail

task="pick up the shuttlecock and place it into the container"
num_episodes=10
dataset_repo_id="dreamdifferent/so101_shuttlecock"

usage() {
    cat <<EOF
Usage: $0 [--task TASK] [--num-episodes N] [--dataset-repo-id REPO_ID]

Options:
  --task, -t           Task instruction to store in the dataset.
  --num-episodes, -n   Number of episodes to record. Alias: --num_episodes.
  --dataset-repo-id    Hugging Face dataset repo ID. Alias: --dataset_repo_id.
  --help, -h           Show this help message.

Defaults:
  task: ${task}
  num_episodes: ${num_episodes}
  dataset_repo_id: ${dataset_repo_id}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task|-t)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires an argument." >&2
                usage >&2
                exit 1
            fi
            task="$2"
            shift 2
            ;;
        --num-episodes|--num_episodes|-n)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires an argument." >&2
                usage >&2
                exit 1
            fi
            num_episodes="$2"
            shift 2
            ;;
        --dataset-repo-id|--dataset_repo_id)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires an argument." >&2
                usage >&2
                exit 1
            fi
            dataset_repo_id="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "${task}" ]]; then
    echo "Error: --task must not be empty." >&2
    exit 1
fi

if [[ -z "${dataset_repo_id}" ]]; then
    echo "Error: --dataset-repo-id must not be empty." >&2
    exit 1
fi

if ! [[ "${num_episodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-episodes must be a positive integer." >&2
    exit 1
fi

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
    --num-episodes "${num_episodes}" \
    --dataset-repo-id "${dataset_repo_id}" \
    --push-to-hub \
    --task "${task}" \
    --resume \
    --revision main
