import argparse
import importlib.util
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
ISAACLAB_SOURCE_ROOT = WORKSPACE_ROOT / "IsaacLab" / "source"


def _ensure_python_paths() -> None:
    """Make local sim_evals importable without overriding installed IsaacLab packages."""
    candidate_paths = [
        REPO_ROOT,
        REPO_ROOT / "sim_evals" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_python_paths()

from sim_evals.agibot_scene_presets import get_agibot_default_instruction, get_agibot_scene_preset
from sim_evals.inference.agibot_jointpos import Client as DreamZeroAgibotClient

WORKSPACE_ROOT = REPO_ROOT.parent
GENIESIM_ROOT = WORKSPACE_ROOT / "genie_sim"
GENIE_INIT_STATES_PATH = GENIESIM_ROOT / "source/geniesim/benchmark/config/robot_init_states.py"

GENIE_G1_LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]
GENIE_G1_RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]
GENIE_G1_HEAD_JOINT_NAMES = ["idx11_head_joint1", "idx12_head_joint2"]
GENIE_G1_WAIST_JOINT_NAMES = ["idx02_body_joint2", "idx01_body_joint1"]
GENIE_G1_LEFT_GRIPPER_JOINT_NAME = "idx41_gripper_l_outer_joint1"
GENIE_G1_RIGHT_GRIPPER_JOINT_NAME = "idx81_gripper_r_outer_joint1"
GENIE_G1_GRIPPER_LIMIT = float(math.pi / 4.0)


def _load_geniesim_init_states():
    spec = importlib.util.spec_from_file_location("geniesim_robot_init_states", GENIE_INIT_STATES_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load GenieSim init states from {GENIE_INIT_STATES_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1_DEFAULT_STATES


def _expected_initial_joint_positions() -> dict[str, float]:
    init_states = _load_geniesim_init_states()
    head_init = init_states["body_state"][:2]
    waist_init = init_states["body_state"][2:4]
    expected = {
        name: float(value)
        for name, value in zip(
            GENIE_G1_LEFT_ARM_JOINT_NAMES + GENIE_G1_RIGHT_ARM_JOINT_NAMES,
            init_states["init_arm"],
            strict=True,
        )
    }
    expected.update(
        {
            "idx11_head_joint1": float(head_init[0]),
            "idx12_head_joint2": float(head_init[1]),
            "idx02_body_joint2": float(waist_init[0]),
            "idx01_body_joint1": float(waist_init[1]),
        }
    )
    return expected


def _show_viz_frame(window_name: str, frame, show_viz: bool) -> bool:
    if not show_viz:
        return False
    try:
        cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return True
    except cv2.error as exc:
        print(
            "OpenCV GUI is unavailable; continuing without live visualization. "
            f"Original error: {exc}"
        )
        return False


def _prepare_viz_tile(image, size: int = 224):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _obs_to_viz(obs: dict):
    policy_obs = obs["policy"]
    top = policy_obs["top_head"][0].detach().cpu().numpy()
    left = policy_obs["hand_left"][0].detach().cpu().numpy()
    right = policy_obs["hand_right"][0].detach().cpu().numpy()
    return cv2.hconcat(
        [
            _prepare_viz_tile(top),
            _prepare_viz_tile(left),
            _prepare_viz_tile(right),
        ]
    )


def _log_reset_joint_state(base_env) -> None:
    robot = base_env.scene["robot"]
    expected_joint_positions = _expected_initial_joint_positions()
    ordered_names = (
        GENIE_G1_LEFT_ARM_JOINT_NAMES
        + GENIE_G1_RIGHT_ARM_JOINT_NAMES
        + GENIE_G1_HEAD_JOINT_NAMES
        + GENIE_G1_WAIST_JOINT_NAMES
    )
    joint_index_by_name = {name: i for i, name in enumerate(robot.data.joint_names)}
    actual = []
    expected = []
    for name in ordered_names:
        idx = joint_index_by_name[name]
        actual.append(float(robot.data.joint_pos[0, idx].item()))
        expected.append(float(expected_joint_positions[name]))

    actual_t = torch.tensor(actual, dtype=torch.float32)
    expected_t = torch.tensor(expected, dtype=torch.float32)
    diff_t = actual_t - expected_t
    # print(
    #     "[Agibot reset debug] joint max_abs_diff="
    #     f"{float(diff_t.abs().max().item()):.4f} "
    #     f"left_first3={[round(v, 4) for v in actual[:3]]} "
    #     f"right_first3={[round(v, 4) for v in actual[7:10]]} "
    #     f"head={[round(v, 4) for v in actual[14:16]]} "
    #     f"waist={[round(v, 4) for v in actual[16:18]]}"
    # )


def _log_action(step_idx: int, action: torch.Tensor, obs: dict | None = None) -> None:
    action_cpu = action[0].detach().cpu()
    delta_suffix = ""
    if obs is not None:
        policy_obs = obs["policy"]
        obs_left = policy_obs["left_arm_joint_position"][0].detach().cpu()
        obs_right = policy_obs["right_arm_joint_position"][0].detach().cpu()
        obs_action = torch.cat([obs_left, obs_right], dim=0)
        cmd_action = torch.cat([action_cpu[:7], action_cpu[7:14]], dim=0)
        delta = cmd_action - obs_action
        delta_suffix = (
            f" delta_norm={float(delta.norm().item()):.4f}"
            f" max_abs_delta={float(delta.abs().max().item()):.4f}"
            f" left_g_raw={float(action_cpu[14].item()):.4f}"
            f" right_g_raw={float(action_cpu[15].item()):.4f}"
        )
    # print(
    #     "[Agibot action] "
    #     f"step={step_idx} "
    #     f"norm={float(action_cpu.norm().item()):.4f} "
    #     f"max_abs={float(action_cpu.abs().max().item()):.4f} "
    #     f"left_first3={[round(v, 4) for v in action_cpu[:3].tolist()]} "
    #     f"right_first3={[round(v, 4) for v in action_cpu[7:10].tolist()]} "
    #     f"gripper=({float(action_cpu[14].item()):.4f}, {float(action_cpu[15].item()):.4f})"
    #     f"{delta_suffix}"
    # )


def _log_raw_action(step_idx: int, raw_action) -> None:
    action_arr = torch.as_tensor(raw_action, dtype=torch.float32).flatten().cpu()
    # print(
    #     "[Agibot raw action runner] "
    #     f"step={step_idx} "
    #     f"dim={action_arr.numel()} "
    #     f"values={[round(v, 4) for v in action_arr.tolist()]}"
    # )


def _joint_index(robot, joint_name: str) -> int:
    joint_index_by_name = {name: i for i, name in enumerate(robot.data.joint_names)}
    return int(joint_index_by_name[joint_name])


def _obs_gripper_state(obs: dict) -> tuple[float, float]:
    policy_obs = obs["policy"]
    left = float(policy_obs["left_effector_position"][0, 0].detach().cpu().item())
    right = float(policy_obs["right_effector_position"][0, 0].detach().cpu().item())
    return left, right


def _robot_gripper_joint_state(base_env) -> tuple[float, float]:
    robot = base_env.scene["robot"]
    left_idx = _joint_index(robot, GENIE_G1_LEFT_GRIPPER_JOINT_NAME)
    right_idx = _joint_index(robot, GENIE_G1_RIGHT_GRIPPER_JOINT_NAME)
    left_joint = float(robot.data.joint_pos[0, left_idx].detach().cpu().item())
    right_joint = float(robot.data.joint_pos[0, right_idx].detach().cpu().item())
    return left_joint, right_joint


def _gripper_joint_to_model_space(joint_value: float) -> float:
    return float(max(0.0, min(1.0, 1.0 - joint_value / GENIE_G1_GRIPPER_LIMIT)))


def _gripper_joint_semantic(joint_value: float) -> str:
    return "open" if joint_value > (GENIE_G1_GRIPPER_LIMIT * 0.5) else "close"


def _make_gripper_debug_record(
    *,
    episode_idx: int,
    step_idx: int,
    obs_before: dict,
    obs_after: dict,
    action: torch.Tensor,
    raw_action,
    base_env,
) -> dict:
    action_cpu = action[0].detach().cpu()
    left_joint_after, right_joint_after = _robot_gripper_joint_state(base_env)
    left_model_after = _gripper_joint_to_model_space(left_joint_after)
    right_model_after = _gripper_joint_to_model_space(right_joint_after)
    left_obs_before, right_obs_before = _obs_gripper_state(obs_before)
    left_obs_after, right_obs_after = _obs_gripper_state(obs_after)

    record = {
        "episode": int(episode_idx),
        "step": int(step_idx),
        "raw_left_gripper": float(raw_action[14]) if raw_action is not None and len(raw_action) > 14 else None,
        "raw_right_gripper": float(raw_action[15]) if raw_action is not None and len(raw_action) > 15 else None,
        "cmd_left_gripper_joint": float(action_cpu[14].item()),
        "cmd_right_gripper_joint": float(action_cpu[15].item()),
        "cmd_left_gripper_semantic": _gripper_joint_semantic(float(action_cpu[14].item())),
        "cmd_right_gripper_semantic": _gripper_joint_semantic(float(action_cpu[15].item())),
        "obs_left_gripper_before": left_obs_before,
        "obs_right_gripper_before": right_obs_before,
        "obs_left_gripper_after": left_obs_after,
        "obs_right_gripper_after": right_obs_after,
        "joint_left_gripper_after": left_joint_after,
        "joint_right_gripper_after": right_joint_after,
        "joint_left_gripper_after_semantic": _gripper_joint_semantic(left_joint_after),
        "joint_right_gripper_after_semantic": _gripper_joint_semantic(right_joint_after),
        "joint_left_gripper_after_model_space": left_model_after,
        "joint_right_gripper_after_model_space": right_model_after,
        "obs_joint_left_gap_after": float(left_obs_after - left_model_after),
        "obs_joint_right_gap_after": float(right_obs_after - right_model_after),
    }
    return record


def _emit_gripper_debug(record: dict, debug_path: Path | None) -> None:
    print(
        "[Agibot gripper] "
        f"ep={record['episode']} "
        f"step={record['step']} "
        f"raw=({record['raw_left_gripper']}, {record['raw_right_gripper']}) "
        f"cmd_joint=({record['cmd_left_gripper_joint']:.4f}, {record['cmd_right_gripper_joint']:.4f}) "
        f"cmd_semantic=({record['cmd_left_gripper_semantic']}, {record['cmd_right_gripper_semantic']}) "
        f"obs_before=({record['obs_left_gripper_before']:.4f}, {record['obs_right_gripper_before']:.4f}) "
        f"obs_after=({record['obs_left_gripper_after']:.4f}, {record['obs_right_gripper_after']:.4f}) "
        f"joint_after=({record['joint_left_gripper_after']:.4f}, {record['joint_right_gripper_after']:.4f}) "
        f"joint_after_semantic=({record['joint_left_gripper_after_semantic']}, {record['joint_right_gripper_after_semantic']}) "
        f"joint_after_model=({record['joint_left_gripper_after_model_space']:.4f}, {record['joint_right_gripper_after_model_space']:.4f}) "
        f"obs_joint_gap=({record['obs_joint_left_gap_after']:.4f}, {record['obs_joint_right_gap_after']:.4f})"
    )
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _settle_like_inspect(base_env, settle_steps: int, render: bool, extra_renders: int = 2):
    if settle_steps <= 0:
        return base_env.obs_buf
    needs_render = render and (base_env.sim.has_gui() or base_env.sim.has_rtx_sensors())
    for _ in range(settle_steps):
        base_env.sim.step(render=False)
        if needs_render:
            base_env.sim.render()
        base_env.scene.update(dt=base_env.physics_dt)

    if base_env.sim.has_rtx_sensors():
        for _ in range(extra_renders):
            base_env.sim.render()
    base_env.obs_buf = base_env.observation_manager.compute(update_history=True)
    return base_env.obs_buf


def _reset_like_inspect(base_env, settle_steps: int, render: bool):
    base_env.reset()
    base_env.reset()
    obs = _settle_like_inspect(base_env, settle_steps, render=render)
    _log_reset_joint_state(base_env)
    return obs


def _apply_action_low_level(base_env, action: torch.Tensor, render: bool, hold_steps: int | None = None):
    # Agibot intentionally stays on the low-level stepping path. In this
    # embodiment, direct env.step(...) did not reliably move the robot, while
    # this explicit loop matches the stable inspect/debug behavior.
    action = action.to(base_env.device)
    base_env.action_manager.process_action(action)

    if hold_steps is None:
        hold_steps = int(base_env.cfg.decimation)

    needs_render = render and (base_env.sim.has_gui() or base_env.sim.has_rtx_sensors())
    for _ in range(hold_steps):
        base_env.action_manager.apply_action()
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=needs_render)
        base_env.scene.update(dt=base_env.physics_dt)

    if base_env.sim.has_rtx_sensors() and not needs_render:
        base_env.sim.render()
    base_env.obs_buf = base_env.observation_manager.compute(update_history=True)
    return base_env.obs_buf


def _get_action_rate_hz(base_env) -> int:
    return int(round(1.0 / (float(base_env.physics_dt) * int(base_env.cfg.decimation))))


def _get_episode_step_budget(base_env) -> int:
    if hasattr(base_env, "max_episode_length"):
        return int(base_env.max_episode_length)
    action_dt = float(base_env.physics_dt) * int(base_env.cfg.decimation)
    return max(1, int(round(float(base_env.cfg.episode_length_s) / action_dt)))


def _make_small_random_action(base_env, arm_scale: float) -> torch.Tensor:
    robot = base_env.scene["robot"]
    joint_index_by_name = {name: i for i, name in enumerate(robot.data.joint_names)}
    left_idxs = [joint_index_by_name[name] for name in GENIE_G1_LEFT_ARM_JOINT_NAMES]
    right_idxs = [joint_index_by_name[name] for name in GENIE_G1_RIGHT_ARM_JOINT_NAMES]
    left_gripper_idx = joint_index_by_name[GENIE_G1_LEFT_GRIPPER_JOINT_NAME]
    right_gripper_idx = joint_index_by_name[GENIE_G1_RIGHT_GRIPPER_JOINT_NAME]

    current_joint_pos = robot.data.joint_pos[0]
    left = current_joint_pos[left_idxs].clone()
    right = current_joint_pos[right_idxs].clone()
    left_gripper = current_joint_pos[left_gripper_idx].clone()
    right_gripper = current_joint_pos[right_gripper_idx].clone()

    left = left + (torch.rand_like(left) * 2.0 - 1.0) * arm_scale
    right = right + (torch.rand_like(right) * 2.0 - 1.0) * arm_scale

    action = torch.cat(
        [
            left,
            right,
            left_gripper.view(1),
            right_gripper.view(1),
        ]
    ).to(dtype=torch.float32, device=base_env.device)
    return action.unsqueeze(0)


def _format_base_component(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _default_run_tag(scene: int, open_loop_horizon: int, base_x: float, base_z: float) -> str:
    return (
        f"scene{scene}"
        f"_h{open_loop_horizon}"
        f"_bx{_format_base_component(base_x)}"
        f"_bz{_format_base_component(base_z)}"
    )


def main(
    episodes: int = 10,
    scene: int = 1,
    headless: bool = True,
    host: str = "localhost",
    port: int = 8000,
    open_loop_horizon: int = 16,
    instruction: str | None = None,
    geniesim_episode_path: str = "/local/home/teame/workspace/genie_sim/source/geniesim/benchmark/saved_task/table_task_g1/table_task_g1_0.json",
    reset_warmup_steps: int = 10,
    pre_infer_settle_steps: int = 0,
    random_action_debug: bool = False,
    random_action_scale: float = 0.05,
    random_action_interval: int = 30,
    gripper_debug: bool = False,
    gripper_debug_every: int = 1,
    gripper_debug_path: str | None = None,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    base_yaw_deg: float = 0.0,
    output_root: str = "runs",
    run_tag: str | None = None,
):
    from isaaclab.app import AppLauncher

    scene_preset = get_agibot_scene_preset(scene)
    resolved_instruction = instruction.strip() if instruction and instruction.strip() else get_agibot_default_instruction(scene)

    parser = argparse.ArgumentParser(description="Run Agibot DreamZero evaluation in Isaac sim_evals.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "AGIBOT_G1_DREAMZERO",
        device=args_cli.device,
        num_envs=1,
        # Existing USD-mounted cameras on the GenieSim G1 robot hit a Warp/Fabric
        # pose-query code path in some Isaac versions. Disable Fabric here for the
        # local single-env eval path to keep the camera stack stable.
        use_fabric=False,
    )
    env_cfg.set_scene(scene)
    env_cfg.geniesim_episode_path = geniesim_episode_path
    env_cfg.base_x_offset = base_x
    env_cfg.base_y_offset = base_y
    env_cfg.base_z_offset = base_z
    env_cfg.base_yaw_offset_deg = base_yaw_deg
    raw_env = gym.make("AGIBOT_G1_DREAMZERO", cfg=env_cfg)
    base_env = getattr(raw_env, "unwrapped", raw_env)

    # Keep the original Agibot bring-up warmup. This one-time settle right
    # after scene creation has been important for getting the embodiment into a
    # correct initial runtime state before evaluation starts.
    base_env.reset()
    
    for _ in range(10):
        base_env.sim.step(render=True)
        base_env.scene.update(dt=base_env.physics_dt)

    # print("Does it work?")

    print(
        f"[Agibot eval] scene={scene_preset.scene_id} "
        f"asset={scene_preset.scene_asset_name} "
        f"instruction={resolved_instruction!r}"
    )

    effective_run_tag = (run_tag.strip() if run_tag and run_tag.strip() else None) or _default_run_tag(
        scene=scene,
        open_loop_horizon=open_loop_horizon,
        base_x=base_x,
        base_z=base_z,
    )
    timestamp = datetime.now()
    video_dir = (
        Path(output_root)
        / timestamp.strftime("%Y-%m-%d")
        / f"{timestamp.strftime('%H-%M-%S')}_{effective_run_tag}"
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    gripper_debug_file = Path(gripper_debug_path) if gripper_debug_path else video_dir / "gripper_debug.jsonl"
    max_steps = _get_episode_step_budget(base_env)
    video_fps = _get_action_rate_hz(base_env)
    show_viz = not headless
    client = None
    run_meta = {
        "scene": int(scene_preset.scene_id),
        "scene_asset": scene_preset.scene_asset_name,
        "instruction": resolved_instruction,
        "episodes": int(episodes),
        "open_loop_horizon": int(open_loop_horizon),
        "base_x": float(base_x),
        "base_y": float(base_y),
        "base_z": float(base_z),
        "base_yaw_deg": float(base_yaw_deg),
        "headless": bool(headless),
        "video_fps": int(video_fps),
        "max_steps": int(max_steps),
        "output_dir": str(video_dir),
        "debug_dump_dir": os.environ.get("DREAMZERO_DEBUG_DIR"),
    }
    with (video_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[Agibot eval] output_dir={video_dir}")

    with torch.no_grad():
        for episode_idx in range(episodes):
            if client is None and not random_action_debug:
                client = DreamZeroAgibotClient(
                    remote_host=host,
                    remote_port=port,
                    open_loop_horizon=open_loop_horizon,
                )
            if client is not None:
                client.reset()

            settle_steps = max(int(reset_warmup_steps), int(pre_infer_settle_steps))
            obs = _reset_like_inspect(base_env, settle_steps=settle_steps, render=not headless)
            video = []
            pending_random_action = None
            for step_idx in tqdm(range(max_steps), desc=f"Episode {episode_idx + 1}/{episodes}"):
                if random_action_debug:
                    if pending_random_action is None or step_idx % max(1, random_action_interval) == 0:
                        pending_random_action = _make_small_random_action(base_env, random_action_scale)
                    obs = _apply_action_low_level(
                        base_env,
                        pending_random_action,
                        render=not headless,
                        hold_steps=int(base_env.cfg.decimation),
                    )
                    ret = {"action": pending_random_action[0].detach().cpu().numpy(), "viz": _obs_to_viz(obs)}
                else:
                    ret = client.infer(obs, resolved_instruction)
                    if show_viz:
                        show_viz = _show_viz_frame("Agibot Cameras", ret["viz"], show_viz)
                    video.append(ret["viz"])
                    # if "raw_action" in ret:
                    #     _log_raw_action(step_idx, ret["raw_action"])
                    action = torch.tensor(ret["action"], dtype=torch.float32)[None]
                    _log_action(step_idx, action, obs=obs)
                    obs_before = obs
                    obs = _apply_action_low_level(base_env, action, render=not headless)
                    if gripper_debug and step_idx % max(1, gripper_debug_every) == 0:
                        record = _make_gripper_debug_record(
                            episode_idx=episode_idx,
                            step_idx=step_idx,
                            obs_before=obs_before,
                            obs_after=obs,
                            action=action,
                            raw_action=ret.get("raw_action"),
                            base_env=base_env,
                        )
                        _emit_gripper_debug(record, gripper_debug_file)
                    continue

                if show_viz:
                    show_viz = _show_viz_frame("Agibot Cameras", ret["viz"], show_viz)
                video.append(ret["viz"])

            mediapy.write_video(video_dir / f"episode_{episode_idx}.mp4", video, fps=video_fps)

    base_env.close()
    if show_viz:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
