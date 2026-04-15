# GenieSim -> Isaac Eval Migration Notes

## Goal
- Move Agibot G1 DreamZero evaluation away from the patched GenieSim benchmark loop.
- Rebuild the evaluation path on top of `dreamzero/sim_evals` and an Isaac-based eval flow.
- Keep GenieSim's G1 embodiment semantics as much as possible:
  - robot asset
  - joint naming/order
  - camera meaning
  - gripper semantics
  - initial state conventions

## New Components

### Isaac Agibot eval runner
- Added [run_sim_eval_agibot.py](/local/home/teame/workspace/dreamzero/eval_utils/run_sim_eval_agibot.py)
- Responsibilities:
  - launch Isaac env
  - connect to DreamZero Agibot websocket server
  - run open-loop chunk inference
  - apply low-level actions
  - save debug images

### Agibot inference client
- Added [agibot_jointpos.py](/local/home/teame/workspace/dreamzero/sim_evals/src/sim_evals/inference/agibot_jointpos.py)
- Responsibilities:
  - convert Isaac observations into Agibot DreamZero server payload
  - send top/head and hand camera images
  - send arm, effector, head, and waist state
  - receive DreamZero action chunks
  - translate raw server action into env action

### Agibot Isaac environment
- Added [agibot_g1_environment.py](/local/home/teame/workspace/dreamzero/sim_evals/src/sim_evals/environments/agibot_g1_environment.py)
- Responsibilities:
  - spawn GenieSim G1 in Isaac
  - build minimal pick scene
  - expose 3-camera observations
  - expose Agibot state observations
  - define action space and termination logic

### GenieSim G1 wrapper
- Added [genie_g1.py](/local/home/teame/workspace/dreamzero/sim_evals/src/sim_evals/environments/genie_g1.py)
- Responsibilities:
  - reuse GenieSim G1 asset paths and metadata
  - define joint names/order
  - define gripper limit
  - define camera specs
  - define initial joint positions

### Scene inspection script
- Added [inspect_agibot_scene.py](/local/home/teame/workspace/dreamzero/eval_utils/inspect_agibot_scene.py)
- Responsibilities:
  - bring up scene without eval
  - inspect robot pose in GUI
  - run random action debug
  - tune base offsets

## What Was Reused

### From GenieSim
- G1 robot embodiment source of truth
- joint ordering and naming
- gripper limit and semantics
- camera meaning:
  - head
  - left hand
  - right hand
- initial joint state conventions
- robot init pose conventions

### From DreamZero
- websocket observation/action contract for Agibot
- chunked inference pattern
- open-loop horizon behavior

### From the Isaac eval side
- `sim_evals` structure
- Isaac env registration / runner pattern
- minimal scene/eval structure

## Major Design Changes During Bring-up

### Moved away from `env.step()`-centric control
- Initial runner used the standard gym-style `env.reset()` / `env.step(action)` flow.
- Agibot G1 did not behave the same way as the inspect path under that route.
- The runner was reworked to use `base_env = raw_env.unwrapped` and directly drive:
  - `action_manager.process_action`
  - `action_manager.apply_action`
  - `scene.write_data_to_sim()`
  - `sim.step(render=...)`
  - `scene.update(...)`

### Matched low-level stepping to inspect behavior
- A key issue was the difference between:
  - `sim.step(render=False)` plus `sim.render()`
  - versus direct `sim.step(render=...)`
- The low-level helper was updated to match the inspect path more closely.

### Standardized action hold time
- Env timing:
  - `physics_dt = 1/120`
  - `decimation = 8`
- Therefore one control action is held for:
  - `8 * 1/120 = 1/15 s = 66.67 ms`
- Both DreamZero actions and random debug actions were aligned to this hold time.

### Added reset settle behavior
- Reset state was not always visually stable immediately after reset.
- Added inspect-style settle stepping after reset before reading the first observation.

## Debugging Findings

### Initial joint state was not the root problem
- Reset debug showed the desired GenieSim joint state was being applied correctly.
- Joint mismatch at reset was effectively zero.

### Not just a camera problem
- External GUI view showed the robot itself could fail to move, so this was not only stale camera imagery.

### Inspect and eval semantics differed in important ways
- `inspect_agibot_scene.py` worked when stepping physics directly.
- The eval runner initially differed in render/apply/update sequencing and wrapper behavior.
- Matching inspect-style low-level stepping was necessary.

### Server-side Agibot action packing bug was found
- Expected raw Agibot action per horizon step: `22` dims
- Observed raw action before fix: `210` dims
- Cause:
  - scalar horizon outputs shaped like `(48,)` were reshaped to `(1, 48)`
  - instead of `(48, 1)`
- Fix:
  - updated [serve_single_gpu_agibot.py](/local/home/teame/workspace/dreamzero/serve_single_gpu_agibot.py)
  - final server action shape now correctly becomes `(48, 22)`

### DreamZero action should be treated as absolute at inference output
- Training/data config may use relative actions.
- But DreamZero inference unapply restores absolute actions by adding the current state back.
- Therefore applying the returned arm commands as absolute joint targets is the correct interpretation.

## Current Status

### Arm behavior
- The robot now moves in Isaac eval through the low-level control path.
- Right-arm behavior looks progressively more plausible than during the original broken bring-up phase.

### Gripper behavior
- Gripper closing was manually checked and does work.
- So gripper actuation path currently looks okay.
- Earlier concern that gripper mapping might be broken is reduced.

### Remaining uncertainty
- Remaining issues are more likely about policy behavior and task success quality than about basic action transport or simulation control.

## Operational Convenience Changes

### Debug image dumps
- `DREAMZERO_DEBUG_DIR` is supported for storing request/debug images per run.

### Base pose tuning options
- Added CLI options:
  - `--base-x`
  - `--base-y`
  - `--base-z`
  - `--base-yaw-deg`

### Random-action debugging
- Added random-action debug paths to validate low-level motion independently of DreamZero policy output.

### Overnight test helper
- Updated [test_run.sh](/local/home/teame/workspace/dreamzero/test_run.sh) for multi-run experiments and separate logs.

## Bottom Line
- The migration from GenieSim benchmark control to Isaac-based eval is now far enough along that the major infrastructure problems have been addressed.
- The biggest concrete bug found during migration was the Agibot server action packing issue, and that has been fixed.
- Low-level robot stepping in Isaac now works.
- Gripper closing has also been confirmed to work.
- The remaining work is mostly about policy/task performance rather than basic embodiment transport.
