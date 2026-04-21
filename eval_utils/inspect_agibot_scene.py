import argparse
import importlib.util
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
GENIESIM_ROOT = WORKSPACE_ROOT / "genie_sim"
GENIE_INIT_STATES_PATH = GENIESIM_ROOT / "source/geniesim/benchmark/config/robot_init_states.py"


def _ensure_python_paths() -> None:
    candidate_paths = [
        REPO_ROOT,
        REPO_ROOT / "sim_evals" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_python_paths()

from sim_evals.agibot_scene_presets import get_agibot_scene_preset


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


def _log_reset_joint_state(env) -> None:
    base_env = getattr(env, "unwrapped", env)
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
    print(
        "[Agibot inspect] joint max_abs_diff="
        f"{float(diff_t.abs().max().item()):.4f} "
        f"left_first3={[round(v, 4) for v in actual[:3]]} "
        f"right_first3={[round(v, 4) for v in actual[7:10]]} "
        f"head={[round(v, 4) for v in actual[14:16]]} "
        f"waist={[round(v, 4) for v in actual[16:18]]}"
    )


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

def main(
    scene: int = 1,
    headless: bool = False,
    geniesim_episode_path: str = "/local/home/teame/workspace/genie_sim/source/geniesim/benchmark/saved_task/table_task_g1/table_task_g1_0.json",
    step_physics: bool = False,
    sleep_s: float = 0.02,
    random_action: bool = False,
    random_action_scale: float = 0.05,
    random_action_interval: int = 30,
    base_x: float = 0.0,
    base_y: float = 0.0,
    base_z: float = 0.0,
    base_yaw_deg: float = 0.0,
):
    from isaaclab.app import AppLauncher

    scene_preset = get_agibot_scene_preset(scene)

    parser = argparse.ArgumentParser(description="Inspect the Agibot Isaac scene without running DreamZero eval.")
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
        use_fabric=False,
    )
    env_cfg.set_scene(scene)
    env_cfg.geniesim_episode_path = geniesim_episode_path
    env_cfg.base_x_offset = base_x
    env_cfg.base_y_offset = base_y
    env_cfg.base_z_offset = base_z
    env_cfg.base_yaw_offset_deg = base_yaw_deg
    env = gym.make("AGIBOT_G1_DREAMZERO", cfg=env_cfg)
    env.reset()
    env.reset()
    _log_reset_joint_state(env)

    base_env = getattr(env, "unwrapped", env)
    base_env.reset()

    for _ in range(10):
        base_env.sim.step(render=True)
        base_env.scene.update(dt=base_env.physics_dt)
    
    print(
        f"[Agibot inspect] scene={scene_preset.scene_id} "
        f"asset={scene_preset.scene_asset_name} "
        f"default_instruction={scene_preset.default_instruction!r}"
    )
    print("[Agibot inspect] scene ready")
    print("[Agibot inspect] robot prim: /World/envs/env_0/G1")
    print("[Agibot inspect] cameras:")
    print("  - /World/envs/env_0/G1/head_link2/Head_Camera")
    print("  - /World/envs/env_0/G1/gripper_l_base_link/Left_Camera")
    print("  - /World/envs/env_0/G1/gripper_r_base_link/Right_Camera")
    print("[Agibot inspect] Tip: use the GUI articulation/joint inspector on /World/envs/env_0/G1")
    if random_action:
        print(
            "[Agibot inspect] random action enabled "
            f"(scale={random_action_scale:.3f}, interval={random_action_interval})"
        )

    try:
        step_count = 0
        pending_action = None
        while simulation_app.is_running():
            if step_physics:
                if random_action:
                    if pending_action is None or step_count % max(1, random_action_interval) == 0:
                        pending_action = _make_small_random_action(base_env, random_action_scale)
                        base_env.action_manager.process_action(pending_action)
                    base_env.action_manager.apply_action()
                    base_env.scene.write_data_to_sim()
                base_env.sim.step(render=True)
                base_env.scene.update(dt=base_env.physics_dt)
                step_count += 1
            else:
                base_env.sim.render()
                time.sleep(sleep_s)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
