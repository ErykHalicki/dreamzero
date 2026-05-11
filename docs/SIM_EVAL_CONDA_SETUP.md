# DreamZero Simulation Evaluation - Conda Setup Guide

An alternative to the `uv`-based setup in the README's "Testing Out DreamZero in Simulation with API" section, using `conda` instead.

## 1. Clone sim-evals

```bash
git clone https://github.com/arhanjain/sim-evals.git
cd sim-evals
```

## 2. Create and activate conda environment

```bash
conda create -n sim-evals python=3.11 -y
conda activate sim-evals
```

## 3. Install IsaacSim / IsaacLab

```bash
# Install isaacsim from NVIDIA PyPI
pip install isaacsim==5.0.0.0 --extra-index-url https://pypi.nvidia.com

# Install isaaclab (version required by sim-evals)
pip install "isaaclab[all]==2.2.0" --extra-index-url https://pypi.nvidia.com
```

## 4. Install sim-evals and remaining dependencies

```bash
# Install sim-evals without deps to bypass websockets version conflict
pip install -e . --no-deps

# PyTorch (isaacsim 5.0.0.0 requires torch==2.7.0)
# torch 2.7.0's latest CUDA build is cu126, which is forward-compatible with system CUDA 12.9
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# openpi-client (installed from Physical Intelligence repo)
pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"

# ffmpeg (required by mediapy for video saving)
conda install -c conda-forge ffmpeg -y

# Other dependencies
pip install tyro mediapy

# websockets: isaacsim-kernel requires ==12.0 while sim-evals requires >=16.0.
# The original uv config uses override-dependencies to force >=16.0.
# We do the same here — isaacsim's ==12.0 constraint can be safely ignored.
pip install "websockets>=16.0"
```

## 5. Download simulation assets

```bash
# You may need to set your HuggingFace token
# export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

pip install "huggingface_hub[cli]"
huggingface-cli download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
```

## 6. Run evaluation

```bash
cd ..
python eval_utils/run_sim_eval.py --host <API_HOST> --port <API_PORT>
```

Results are saved in the `runs/` directory.

## Troubleshooting

### websockets version conflict
`sim-evals` requires `websockets>=16.0` while `isaacsim-kernel` requires `websockets==12.0`. The original `uv` configuration uses `override-dependencies` to force `>=16.0`. This guide follows the same approach — installing `>=16.0` and ignoring isaacsim's `==12.0` constraint, which does not cause any runtime issues.

### isaacsim / isaaclab version check
If you encounter installation errors, check compatible versions with:
```bash
pip install "isaaclab[all]==2.2.0" --dry-run --extra-index-url https://pypi.nvidia.com
```
