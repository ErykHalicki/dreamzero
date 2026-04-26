# DreamDifferent Baselines

This directory collects baseline-related materials for the DreamDifferent
project, currently focused on comparing DreamZero against an OpenPI/pi0.5
Franka/ORCA setup on the bag-groceries task.

## Contents

- [`openpi/`](openpi): OpenPI fork submodule used for the current pi0.5
  baseline runs.
- [`training_log/`](training_log): experiment logs, run summaries, and training
  progression notes.
- [`transfer_notes/`](transfer_notes): project notes, transfer notes, and
  comparison writeups.

## Recommended Reading Order

1. Read [`training_log/`](training_log) for current run history and checkpoint
   progression.
2. Read [`transfer_notes/`](transfer_notes) for comparison notes and project
   context.
3. Read `openpi/scripts/train/README.md` inside the OpenPI submodule for the
   Euler-specific training and evaluation workflow.
4. Inspect [`../datasets/bag_groceries_communal/`](../datasets/bag_groceries_communal)
   for the saved train/validation split and distribution checks.

## Notes

- The OpenPI fork used for these experiments lives in a separate repository and
  is linked here as a submodule.
- Dataset-specific metadata and split tooling are kept at the DreamDifferent
  project level under `../datasets/`, not inside the baseline folder.
- This directory is intended to hold baseline-specific code, run logs, and
  comparison notes, while dataset artifacts stay at the DreamDifferent project
  level.
