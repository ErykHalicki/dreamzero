# Pi05 SO-101 Training

This folder contains scripts for fine-tuning LeRobot pi0.5 on SO-101 datasets in a cluster setup.
Large downloads, converted datasets, caches, and training outputs are kept under `/cluster/scratch/$USER/pi05`.
Hugging Face Hub, datasets, assets, Transformers, and LeRobot caches are all pointed into that scratch tree by `train.sh`.

## Main Script

Use `train.sh` for the normal LoRA workflow. It runs:

1. Download the source dataset into scratch, if needed.
2. Convert/recompute dataset stats for relative actions, if needed.
3. Train pi0.5 with LoRA using the converted local dataset.

Use `train_full.sh` for full fine-tuning with the vision encoder frozen. It uses the same dataset/download/conversion layout as `train.sh`, but does not pass any PEFT/LoRA options to `lerobot-train`.

The default experiment is:

```bash
EXP_NAME=pi05_homogeneous_lora
SOURCE_DATASET_REPO_ID=dreamdifferent/so101_bottle
```

Running:

```bash
bash train.sh
```

uses this layout:

```bash
/cluster/scratch/$USER/pi05/experiments/pi05_homogeneous_lora/
  datasets/
    so101_bottle/
      raw/
      relative_stats/
  outputs/
```

The script checks for `datasets/<dataset_slug>/relative_stats/meta/stats.json`.
If it exists, conversion is skipped and training starts directly. If it does not exist, the source dataset is downloaded and converted first.

## Common Runs

Smoke test with the integrated experiment layout:

```bash
EXP_NAME=pi05_smoke_test \
BATCH_SIZE=1 \
WANDB_ENABLE=false \
bash train.sh 1 \
  --policy.compile_model=false \
  --save_checkpoint=false \
  --num_workers=0 \
  --log_freq=1
```

This still downloads and converts the dataset if `datasets/<dataset_slug>/relative_stats/meta/stats.json` does not exist for the smoke experiment.

Smoke test using an already converted local dataset:

```bash
EXP_NAME=pi05_smoke_test \
DATASET_REPO_ID=dreamdifferent/so101_bottle_recomputed_stats \
DATASET_ROOT=/cluster/scratch/$USER/pi05/lerobot/dreamdifferent/so101_bottle_recomputed_stats \
BATCH_SIZE=1 \
WANDB_ENABLE=false \
bash train.sh 1 \
  --policy.compile_model=false \
  --save_checkpoint=false \
  --num_workers=0 \
  --log_freq=1
```

Run the default homogeneous LoRA experiment:

```bash
bash train.sh
```

Run the default homogeneous full fine-tuning experiment:

```bash
bash train_full.sh
```

Full fine-tuning uses:

```bash
EXP_NAME=pi05_homogeneous_full
FREEZE_VISION_ENCODER=true
TRAIN_EXPERT_ONLY=false
```

Run fewer training steps:

```bash
bash train.sh 3000
```

or:

```bash
STEPS=3000 bash train.sh
```

Force dataset conversion again:

```bash
FORCE_CONVERT=true bash train.sh
```

Run the heterogeneous LoRA experiment:

```bash
EXP_NAME=pi05_heterogeneous_lora \
SOURCE_DATASET_REPO_ID=dreamdifferent/so101_multi_object_new \
bash train.sh
```

This writes to:

```bash
/cluster/scratch/$USER/pi05/experiments/pi05_heterogeneous_lora/
  datasets/
    so101_multi_object_new/
      raw/
      relative_stats/
  outputs/
```

## Key Overrides

`EXP_NAME`: Experiment name. Controls the default experiment root, job name, and policy repo id.

`SOURCE_DATASET_REPO_ID`: Hugging Face dataset repo to download before conversion.

`DATASET_SLUG`: Local dataset folder name. Defaults to the final path component of `SOURCE_DATASET_REPO_ID`.

`STEPS`: Number of training steps. Defaults to `25000`.

`BATCH_SIZE`: Training batch size. Defaults to `16`.

`NUM_WORKERS`: Dataloader workers. Defaults to `4`.

`VIDEO_BACKEND`: Video decoder backend. Defaults to `pyav` to avoid cluster FFmpeg/torchcodec ABI issues.

`COMPILE_MODEL`: Enable `torch.compile` for the policy. Defaults to `false`; this is more stable on 24 GB 4090s because Triton autotuning can need extra temporary GPU memory.

`LOG_FREQ`: Logging frequency in optimizer steps. Defaults to `200`.

`SAVE_FREQ`: Checkpoint frequency in optimizer steps. Defaults to `2500`.

`RESUME`: Resume from the existing `OUTPUT_DIR` checkpoint. Defaults to `false`.

`LORA_R`: LoRA rank. Defaults to `16`, matching OpenPI's LoRA configs.

`LORA_ALPHA`: LoRA alpha. Defaults to `16`, matching OpenPI's LoRA configs.

`PEFT_TARGET_MODULES`: Regex for LoRA target modules. By default this follows the OpenPI LoRA scope: PaliGemma language-model attention and MLP projections, action-expert attention and MLP projections, plus action projection layers.

`FREEZE_VISION_ENCODER`: Whether to freeze the vision encoder before PEFT wrapping. Defaults to `true`.

`TRAIN_EXPERT_ONLY`: Whether to freeze the VLM and train only the action expert before PEFT wrapping. Defaults to `false`.

`WANDB_ENABLE`: Enable Weights & Biases logging. Defaults to `true`.

`WANDB_PROJECT`: Weights & Biases project name. Defaults to `pi05_so101`.

`PUSH_TO_HUB`: Push the final policy, preprocessor, and postprocessor to the Hugging Face Hub. Defaults to `false`.

`POLICY_REPO_ID`: Hub model repo id used only when `PUSH_TO_HUB=true`. For shared runs, prefer a repo under `dreamdifferent`, e.g. `dreamdifferent/pi05_homogeneous_lora`.

`FORCE_CONVERT`: Set to `true` to rerun conversion even if relative stats already exist.

Extra LeRobot CLI options can be appended after the script arguments:

```bash
bash train.sh 3000 --wandb.enable=true --policy.optimizer_lr=1e-4
```

Push a completed run to the Hub:

```bash
PUSH_TO_HUB=true \
POLICY_REPO_ID=dreamdifferent/pi05_homogeneous_lora \
bash train.sh
```

## Slurm

Submit the default homogeneous LoRA run:

```bash
sbatch train.sh
```

The Slurm scripts load `eth_proxy` before activating conda so Hugging Face and W&B network calls work from compute nodes.

Resume the same run after preemption or time limit:

```bash
sbatch --export=ALL,RESUME=true train.sh
```

For heterogeneous LoRA:

```bash
sbatch --export=ALL,EXP_NAME=pi05_heterogeneous_lora,SOURCE_DATASET_REPO_ID=dreamdifferent/so101_multi_object_new train.sh
```

Override runtime options:

```bash
sbatch --export=ALL,BATCH_SIZE=16,SAVE_FREQ=2500,WANDB_ENABLE=true train.sh
```

For a safer 4090 run, keep the default batch size and compile setting:

```bash
sbatch --export=ALL,BATCH_SIZE=16,COMPILE_MODEL=false train.sh
```

If you can reserve an A100, batch size 32 should be much safer. Adjust the GPU name to the cluster's exact Slurm resource name:

```bash
sbatch --gpus=a100:1 --export=ALL,BATCH_SIZE=32,COMPILE_MODEL=true train.sh
```

## Notes

`train.sh` unsets the deprecated `LEROBOT_HOME` variable and uses `HF_LEROBOT_HOME` instead.

`train.sh` is the single entrypoint for dataset download, relative-stats conversion, and training. This keeps all paths tied to the same experiment name.
