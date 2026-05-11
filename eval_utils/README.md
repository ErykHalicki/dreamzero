# DreamZero Checkpoint Evaluation (`eval_utils`)

This folder contains the Isaac Sim / Isaac Lab evaluation entrypoints and websocket helpers used to run DreamZero checkpoints in simulation.

Two nearby folders matter just as much as `eval_utils`:

- `../sim_evals/src/sim_evals/environments/`: Isaac Lab environment definitions, camera wiring, observation terms, action terms, reset logic, and env registration.
- `../sim_evals/src/sim_evals/inference/`: robot-specific client adapters that convert Isaac observations into DreamZero server requests and convert DreamZero action chunks back into robot commands.

In other words: `eval_utils` is the runnable layer, but full robot support is always split across `eval_utils`, `sim_evals/.../environments`, and `sim_evals/.../inference`.

## Current support status

| Variant | Main script | Status | Notes |
| --- | --- | --- | --- |
| Original DROID / single-arm Franka | `run_sim_eval.py` | Working baseline | Uses env id `DROID`, single-arm 7+1 action format, 3 cameras. |
| Original DROID / remote websocket server | `run_sim_eval_remote.py` | Working baseline | Same as above, but connects to a remote `ws://` or `wss://` server. |
| AgiBot G1 | `run_sim_eval_agibot.py` | Main working DreamZero checkpoint eval path | Uses env id `AGIBOT_G1_DREAMZERO`, dedicated AgiBot observation/action adapter, low-level stepping, and AgiBot scene presets. |
| Dual Franka | `run_sim_eval_bimanual.py` | Partial / reference path | Loads `DROID_BIMANUAL_RIGHT_ONLY`, adds a passive left Franka and left wrist camera, but policy I/O is still right-arm-only DROID format. |
| PandaOrca / Orca-hand prototype | `inspect_pandaorca_scene.py` | Inspect only / TODO | USD loading works, PandaOrca arms can now be brought up with calibrated left/right exterior cameras in the inspect path, but there is still no policy/action/observation wiring and no full eval runner yet. TODO: add a proper closed-loop evaluator for this variant. |

## Recommended workflows

### 1. AgiBot G1 checkpoint evaluation

This is the most complete robot-specific path in this folder today.

Start the DreamZero AgiBot websocket server from the repo root:

```bash
bash scripts/inference/serve_single_gpu_agibot.sh --model-path /path/to/DreamZero-AgiBot --port 8855
```

For the student-cluster workflow, use `8855`, since ports below `8800` are typically blocked. The AgiBot scripts default to `8000`, so pass `--port 8855` explicitly.

We currently recommend running the AgiBot evaluator in `--no-headless` mode. There is a minor bug in the current AgiBot setup, and keeping the Isaac Sim window open has been the more reliable path in practice. This recommendation is specific to AgiBot; the original Franka / DROID path is generally fine in headless mode.

If the policy is running on another machine, forward the same port locally:

```bash
ssh -L 8855:localhost:8855 cluster-tunnel
```

Optional scene bring-up / camera check:

```bash
python eval_utils/inspect_agibot_scene.py --scene 1
```

Run one evaluation episode:

```bash
python eval_utils/run_sim_eval_agibot.py --host 127.0.0.1 --port 8855 --episodes 1 --scene 1 --no-headless
```

Useful AgiBot notes:

- Outputs are written under `runs/YYYY-MM-DD/HH-MM-SS_<run_tag>/`.
- `run_meta.json` is saved alongside episode videos.
- `--gripper-debug` writes `gripper_debug.jsonl`, which can be summarized with `analyze_agibot_gripper_debug.py`.
- `DREAMZERO_DEBUG_DIR=/path/to/debug_dir` dumps the images sent to the policy server.
- `--random-action-debug` is useful when validating the robot stack without DreamZero inference.
- We recommend `--no-headless` for AgiBot right now because of a minor bring-up / runtime issue in the current setup.
- This path intentionally uses a low-level stepping loop instead of relying only on `env.step(...)`.
- This path also forces `use_fabric=False` because the USD-mounted AgiBot cameras hit a stability issue in some Isaac versions.

### 2. Original DROID / single-arm Franka baseline

Use this when you want the original simulator baseline rather than the AgiBot embodiment.

Local websocket server path:

```bash
python eval_utils/run_sim_eval.py --host 127.0.0.1 --port 8855 --episodes 1 --scene 1
```

If you are serving the DROID checkpoint on the student cluster, launch the server on the same port:

```bash
bash scripts/inference/serve_single_gpu.sh --model-path /path/to/DreamZero-DROID --port 8855
```

Remote websocket server path:

```bash
python eval_utils/run_sim_eval_remote.py --url https://your-server.example --episodes 1 --scene 1 --headless
```

Related serving helpers:

- `scripts/inference/serve_single_gpu.sh` is the standard repo wrapper for DROID-shaped checkpoints.
- `eval_utils/serve_dreamzero_wan22.py` is a DROID-shaped Wan2.2 websocket serving helper living inside this folder.

### 3. Dual Franka vs PandaOrca / Orca-hand

These are not the same thing.

- `run_sim_eval_bimanual.py` is a dual-Franka scene. It still reuses the original DROID-style right-arm 8D policy interface.
- `inspect_pandaorca_scene.py` loads PandaOrca arms into the DROID scene, verifies articulation bring-up, and now also attaches calibrated `left_cam` / `right_cam` exterior cameras in the inspect path.
- `inspect_franka_workspace.py` is a separate geometry / workspace utility for sampling reachable end-effector points in Isaac.

If you are looking for a full evaluation path for the Franka-bimanual-plus-Orca-hand variant, that does not exist yet in the same sense that `run_sim_eval_agibot.py` exists for AgiBot.

TODO: implement a proper closed-loop evaluator for the Franka-bimanual-plus-Orca-hand variant, instead of only the current inspect scaffold.

Current PandaOrca inspect example:

```bash
python eval_utils/inspect_pandaorca_scene.py --variant dual --camera-resolution full --no-headless
```

Notes for the PandaOrca inspect path:

- `--camera-resolution full` uses the imported ARIA-style `640x480` intrinsics.
- `--camera-resolution half` uses the matching half-resolution `320x240` intrinsics.
- The current camera calibration is applied in the inspect path only.
- This is still for scene/camera validation, not DreamZero closed-loop evaluation.

Expected PandaOrca camera prim paths:

- `/World/envs/env_0/pandaorca_right/right_exterior_camera`
- `/World/envs/env_0/pandaorca_left/left_exterior_camera`

## File map

| File | Purpose |
| --- | --- |
| `run_sim_eval.py` | Original local DROID/single-arm Franka evaluation entrypoint. |
| `run_sim_eval_remote.py` | Same baseline evaluation flow, but for a remote websocket URL. |
| `run_sim_eval_bimanual.py` | Dual-Franka scene variant with passive left arm and extra left wrist camera. |
| `run_sim_eval_agibot.py` | Main AgiBot G1 DreamZero evaluation entrypoint. |
| `inspect_agibot_scene.py` | Bring-up and scene inspection helper for AgiBot without DreamZero inference. |
| `inspect_pandaorca_scene.py` | PandaOrca USD inspection helper inside the DROID scene; now includes calibrated left/right exterior cameras for visual validation, but no closed-loop evaluation yet. |
| `inspect_franka_workspace.py` | Workspace sampler for Franka-like robots. |
| `policy_client.py` | Websocket client used by the evaluation-side inference adapters. |
| `policy_server.py` | Generic websocket policy server wrapper. |
| `serve_dreamzero_wan22.py` | DROID-shaped Wan2.2 DreamZero serving helper. |
| `analyze_agibot_gripper_debug.py` | Summarizes AgiBot `gripper_debug.jsonl` logs. |

## How support is split across the codebase

For a robot to be "fully supported", you usually need all of the following:

1. A registered Isaac environment in `../sim_evals/src/sim_evals/environments/`.
2. A robot-specific observation/action adapter in `../sim_evals/src/sim_evals/inference/`.
3. A runnable evaluation entrypoint in `eval_utils/`.
4. A websocket server path that understands the checkpoint's embodiment and modality keys.

AgiBot has all four pieces.

The PandaOrca / Orca-hand path currently only has the inspection side of step 1.

What is already done in that inspect path:

- PandaOrca left/right arm USD references can be loaded into the DROID scene.
- DROID-like PandaOrca arm pose initialization is available for bring-up.
- Calibrated `left_cam` / `right_cam` exterior cameras are attached for visual inspection.

TODO: complete steps 2-4 for PandaOrca / Orca-hand so it becomes a fully supported evaluation path.

## Guideline: adding another robot USD

This is the part to follow if you want to add another robot asset, such as a new AgiBot-like robot or the missing PandaOrca / Orca-hand evaluator.

### First decide what kind of change it is

There are two very different cases:

1. Same embodiment contract, new USD only.
   Example: you are swapping geometry or improving the asset, but the policy still expects the same camera keys, the same joint-state layout, and the same action format.

2. New embodiment contract.
   Example: different joint names, different number of arms, different gripper representation, different camera layout, or different state/action dimensions.

Case 1 is mostly an environment / asset problem.

Case 2 is an environment problem, an inference-adapter problem, and often a checkpoint / server problem too.

The current PandaOrca / Franka-plus-Orca-hand variant falls into case 2.

### Recommended implementation order

#### Step 1. Start with an inspect script

Do this before you write the evaluator.

Use `inspect_pandaorca_scene.py` as the reference pattern if the robot is not yet in an Isaac Lab env.

Verify all of the following first:

- the USD reference loads successfully
- the articulation initializes successfully
- the prim path is stable
- the joint names are what you expect
- the camera prims exist, if the robot has onboard cameras
- the robot appears in the correct pose relative to the task scene

If this step is shaky, the full evaluator will be painful to debug.

#### Step 2. Add the robot asset and env config

Create or extend files under `../sim_evals/src/sim_evals/environments/`.

For references, look at:

- `nvidia_droid.py` for the default single-arm Franka asset definition
- `droid_environment.py` for a simple single-arm env
- `droid_bimanual_environment.py` for a multi-robot scene
- `agibot_g1_environment.py` for a full custom embodiment with custom reset, cameras, observations, and actions

At minimum, the new env needs:

- an `ArticulationCfg` or equivalent robot asset config pointing to the USD
- stable robot prim paths
- `CameraCfg` entries for every view the policy will consume
- observation terms for the joint and gripper state the policy expects
- action terms that map model output dimensions onto the robot joints
- reset logic that restores the robot to a valid initial state
- an env id registration in `../sim_evals/src/sim_evals/environments/__init__.py`

If the robot has task-specific scene logic, follow the AgiBot pattern and create a preset helper like `sim_evals/agibot_scene_presets.py`.

#### Step 3. Add a robot-specific inference adapter

Create a new file under `../sim_evals/src/sim_evals/inference/`.

Use:

- `droid_jointpos.py` as the template for a DROID-like 7+1 interface
- `agibot_jointpos.py` as the template for a multi-camera, multi-arm embodiment

This adapter is where you:

- read the Isaac observation dictionary
- rename / reshape images and state tensors into the websocket keys the server expects
- translate raw DreamZero action chunks back into robot joint commands
- handle gripper scaling or binarization
- optionally smooth or post-process actions
- build the visualization frames shown during rollout

If the robot has different action dimensionality than the existing clients, this file is mandatory.

#### Step 4. Add the runnable evaluator

Add a new `eval_utils/run_sim_eval_<robot>.py` script.

Copy the structure that matches the robot:

- start from `run_sim_eval.py` if `env.step(...)` works and the embodiment is simple
- start from `run_sim_eval_agibot.py` if you need custom reset logic, low-level stepping, custom scene presets, or extensive debug hooks

This script should own:

- CLI flags
- environment creation
- scene / instruction selection
- client construction
- rollout loop
- video writing
- debug output locations

#### Step 5. Make sure the websocket server matches the embodiment

This step is easy to miss.

If the new robot still talks in the same keys and dimensions as an existing embodiment, you may be able to reuse the existing server.

If it does not, you need a dedicated server path similar to `serve_single_gpu_agibot.py`.

The AgiBot path is a good example:

- the evaluator sends `observation/top_head`, `observation/hand_left`, `observation/hand_right`, and multi-part state keys
- the server converts those into the model's internal modality names
- the server concatenates the model outputs into the expected action array

For a new robot, decide explicitly whether you are:

- adapting the robot to an existing checkpoint interface
- or introducing a new embodiment interface that needs its own serving adapter and probably a matching checkpoint

#### Step 6. Validate in this order

Do not jump straight into closed-loop policy evaluation.

Use this order:

1. USD loads and articulation initializes.
2. Reset pose is correct.
3. Joint names and action indexing are correct.
4. Cameras render correctly and with the expected resolution.
5. Observation tensor shapes match what the websocket server expects.
6. Random actions move the intended joints and grippers.
7. Only then run DreamZero closed-loop evaluation.

## TODO: what is missing today for the PandaOrca / Orca-hand variant

Right now the codebase has an inspection scaffold, but not a full evaluator.

The missing pieces are:

- a registered env config for the PandaOrca / Orca-hand robot
- camera observation wiring for the calibrated views you want DreamZero to consume
- state observation terms for the arm and hand joints
- action mapping from DreamZero output into `fer_joint*` and the Orca-hand joints
- reset logic for the robot and scene
- a robot-specific inference adapter under `sim_evals/src/sim_evals/inference/`
- a closed-loop runner such as `run_sim_eval_pandaorca.py`
- a matching websocket serving path if the embodiment keys differ from the existing DROID or AgiBot servers

If you want to implement that path, the cleanest starting point is:

1. keep `inspect_pandaorca_scene.py` for stage bring-up
2. add a dedicated env file under `sim_evals/.../environments`
3. add a dedicated inference adapter under `sim_evals/.../inference`
4. then add a new `run_sim_eval_<variant>.py` runner

## Practical reference points

When in doubt, use these as the main templates:

- Best full robot integration example: `run_sim_eval_agibot.py` + `agibot_g1_environment.py` + `agibot_jointpos.py`
- Best simple baseline example: `run_sim_eval.py` + `droid_environment.py` + `droid_jointpos.py`
- Best inspection-first example for a not-yet-supported robot: `inspect_pandaorca_scene.py`
