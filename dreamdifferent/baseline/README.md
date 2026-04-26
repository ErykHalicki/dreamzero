# DreamDifferent Baselines

This directory collects baseline-related materials for the DreamDifferent
project, currently focused on comparing DreamZero against an OpenPI/pi0.5
Franka/ORCA setup on the bag-groceries task.

## Contents

- [`notes/`](notes): project notes and transfer notes.
- [`training_run_log.md`](training_run_log.md): compact record of the main
  training runs that have been launched so far.
- [`openpi/`](openpi): OpenPI fork submodule used for the baseline runs.

## Recommended Reading Order

1. Read [`training_run_log.md`](training_run_log.md) for the current run
   history and checkpoint progression.
2. Read `openpi/scripts/train/README.md` inside the OpenPI submodule for the
   Euler-specific training and evaluation workflow.
3. Inspect [`../datasets/bag_groceries_communal/`](../datasets/bag_groceries_communal)
   for the saved train/validation split and distribution checks.

## Notes

- The OpenPI fork used for these experiments lives in a separate repository and
  is linked here as a submodule.
- Dataset-specific metadata and split tooling are kept at the DreamDifferent
  project level under `../datasets/`, not inside the baseline folder.
- The original workspace-level copies of these files may still exist during the
  transition; this directory is the intended long-term home for team-facing
  baseline materials.
