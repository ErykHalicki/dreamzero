# Train/Val Distribution Comparison

- Dataset: `/cluster/scratch/eugseo/datasets/bag_groceries_communal`
- Split file: `/cluster/project/cvg/students/eugseo/workspace/franka_orca_splits/bag_groceries_seed42_90_10.json`
- Train episodes: 270
- Val episodes: 30

## Vector Stats

| feature | train rows | val rows | avg abs mean diff | avg abs std diff |
|---|---:|---:|---:|---:|
| `state` | 704454 | 78091 | 0.019230 | 0.019018 |
| `action` | 704454 | 78091 | 0.019453 | 0.009060 |
| `action_delta` | 704184 | 78061 | 0.000012 | 0.000341 |
| `relative_action_sample` | 414720 | 46080 | 0.001531 | 0.022907 |

Notes:
- `state` and `action` are raw per-frame vectors from parquet.
- `action_delta` is `action[t+1] - action[t]`, a simple temporal smoothness proxy.
- `relative_action_sample` samples OpenPI/DreamZero-style targets: `action[t:t+24] - state[t]`.

## Group L2 Means

| group | stat | train mean | val mean |
|---|---|---:|---:|
| `left_arm` | `state_l2` | 1.196437 | 1.201918 |
| `left_arm` | `action_l2` | 1.198117 | 1.199033 |
| `left_arm` | `action_delta_l2` | 0.002435 | 0.002296 |
| `left_hand` | `state_l2` | 3.353332 | 3.405455 |
| `left_hand` | `action_l2` | 3.353332 | 3.405455 |
| `left_hand` | `action_delta_l2` | 0.020551 | 0.019893 |
| `right_arm` | `state_l2` | 1.176648 | 1.189222 |
| `right_arm` | `action_l2` | 1.177602 | 1.184826 |
| `right_arm` | `action_delta_l2` | 0.004076 | 0.004161 |
| `right_hand` | `state_l2` | 2.899045 | 2.954695 |
| `right_hand` | `action_l2` | 2.899045 | 2.954695 |
| `right_hand` | `action_delta_l2` | 0.029828 | 0.030419 |

## Image Stats

- Not requested. Re-run with `--image-stats` to sample video RGB statistics.
