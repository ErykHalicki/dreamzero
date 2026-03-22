#!/bin/bash
set -e

COMMIT="32eb0cec8f322a7d93a1ec2008dd1a11ae6286b3"
CLONE_DIR="/tmp/lerobot_patched"

if [ -d "$CLONE_DIR" ]; then
    echo "Removing existing $CLONE_DIR"
    rm -rf "$CLONE_DIR"
fi

git clone https://github.com/huggingface/lerobot.git "$CLONE_DIR"
cd "$CLONE_DIR"
git checkout "$COMMIT"

sed -i 's/pyav/av/g' pyproject.toml

uv pip install .

cd /
rm -rf "$CLONE_DIR"

echo "Done. LeRobot installed from commit $COMMIT."
