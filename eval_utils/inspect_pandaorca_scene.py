from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Literal

import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SIM_EVALS_ASSETS_ROOT = REPO_ROOT / "sim_evals" / "assets"
PANDAORCA_USD_ROOT = WORKSPACE_ROOT / "pandaorca_description" / "usd"

ENV_ROOT_PRIM_PATH = "/World/envs/env_0"
RIGHT_ARM_PRIM_PATH = f"{ENV_ROOT_PRIM_PATH}/pandaorca_right"
LEFT_ARM_PRIM_PATH = f"{ENV_ROOT_PRIM_PATH}/pandaorca_left"

RIGHT_ARM_USD_PATH = (
    PANDAORCA_USD_ROOT / "fer_orcahand_right_extended" / "fer_orcahand_right_extended.usd"
)
LEFT_ARM_USD_PATH = (
    PANDAORCA_USD_ROOT / "fer_orcahand_left_extended" / "fer_orcahand_left_extended.usd"
)

IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

DROID_LIKE_ARM_JOINT_POS = {
    "fer_joint1": 0.0,
    "fer_joint2": -math.pi / 5.0,
    "fer_joint3": 0.0,
    "fer_joint4": -4.0 * math.pi / 5.0,
    "fer_joint5": 0.0,
    "fer_joint6": 3.0 * math.pi / 5.0,
    "fer_joint7": 0.0,
}


def _ensure_python_paths() -> None:
    candidate_paths = [
        REPO_ROOT,
        REPO_ROOT / "sim_evals" / "src",
    ]
    for path in candidate_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    if importlib.util.find_spec("isaaclab") is not None:
        return

    isaaclab_source_root = WORKSPACE_ROOT / "IsaacLab" / "source"
    isaaclab_package_dirs = [
        isaaclab_source_root / "isaaclab",
        isaaclab_source_root / "isaaclab_assets",
        isaaclab_source_root / "isaaclab_tasks",
        isaaclab_source_root / "isaaclab_rl",
        isaaclab_source_root / "isaaclab_mimic",
        isaaclab_source_root / "isaaclab_contrib",
    ]
    for path in isaaclab_package_dirs:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_python_paths()


def _open_droid_scene(scene: int) -> Path:
    from pxr import Usd
    import omni.usd

    scene_path = SIM_EVALS_ASSETS_ROOT / f"scene{scene}.usd"
    if not scene_path.exists():
        raise ValueError(f"Scene {scene} is not available. Expected USD at: {scene_path}")
    if not Usd.Stage.IsSupportedFile(str(scene_path)):
        raise ValueError(f"Scene USD is not a supported USD file: {scene_path}")

    usd_context = omni.usd.get_context()
    usd_context.disable_save_to_recent_files()
    try:
        opened = usd_context.open_stage(str(scene_path))
    finally:
        usd_context.enable_save_to_recent_files()

    if not opened:
        raise RuntimeError(f"Failed to open DROID scene stage: {scene_path}")

    _wait_for_stage_load()
    return scene_path


def _wait_for_stage_load() -> None:
    import omni.kit.app
    import omni.usd

    usd_context = omni.usd.get_context()
    while True:
        _, _, loading = usd_context.get_stage_loading_status()
        if loading == 0:
            return
        omni.kit.app.get_app_interface().update()


def _get_stage():
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available.")
    return stage


def _ensure_xform_prim(prim_path: str) -> None:
    stage = _get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        return

    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.IsValid():
        raise RuntimeError(f"Failed to define Xform prim at path: {prim_path}")


def _set_or_create_translate_op(xformable, translate_xyz: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    attr = xformable.GetPrim().GetAttribute("xformOp:translate")
    if attr.IsValid():
        UsdGeom.XformOp(attr).Set(Gf.Vec3d(*translate_xyz))
    else:
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translate_xyz))


def _set_or_create_orient_op(xformable, orient_wxyz: tuple[float, float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    attr = xformable.GetPrim().GetAttribute("xformOp:orient")
    quat = Gf.Quatd(*orient_wxyz)
    if attr.IsValid():
        UsdGeom.XformOp(attr).Set(quat)
    else:
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)


def _add_pandaorca_arm(
    prim_path: str,
    usd_path: Path,
    translate_xyz: tuple[float, float, float],
    orient_wxyz: tuple[float, float, float, float],
) -> None:
    from pxr import UsdGeom

    if not usd_path.exists():
        raise FileNotFoundError(f"PandaOrca arm USD does not exist: {usd_path}")

    stage = _get_stage()
    existing = stage.GetPrimAtPath(prim_path)
    if existing.IsValid():
        raise ValueError(f"Prim path already exists in stage: {prim_path}")

    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.IsValid():
        raise RuntimeError(f"Failed to define prim at path: {prim_path}")

    success = prim.GetReferences().AddReference(str(usd_path))
    if not success:
        raise RuntimeError(f"Failed to add USD reference: {usd_path} -> {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    _set_or_create_translate_op(xformable, translate_xyz)
    _set_or_create_orient_op(xformable, orient_wxyz)


def _create_pandaorca_articulation(prim_path: str):
    from isaaclab.assets import Articulation, ArticulationCfg

    cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=IDENTITY_QUAT_WXYZ,
            joint_pos=dict(DROID_LIKE_ARM_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        actuators={},
    )
    return Articulation(cfg)


def _apply_droid_like_arm_pose(robot, label: str) -> dict[str, float]:
    applied_joint_pos: dict[str, float] = {}
    joint_index_by_name = {name: i for i, name in enumerate(robot.data.joint_names)}

    missing = [name for name in DROID_LIKE_ARM_JOINT_POS if name not in joint_index_by_name]
    if missing:
        raise ValueError(
            f"Failed to find expected PandaOrca arm joints for {label}: {missing}. "
            f"Available joint names: {list(robot.data.joint_names)}"
        )

    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()

    for joint_name, joint_value in DROID_LIKE_ARM_JOINT_POS.items():
        joint_index = joint_index_by_name[joint_name]
        joint_pos[:, joint_index] = float(joint_value)
        joint_vel[:, joint_index] = 0.0
        robot.data.default_joint_pos[:, joint_index] = float(joint_value)
        robot.data.default_joint_vel[:, joint_index] = 0.0
        applied_joint_pos[joint_name] = float(joint_value)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.update(0.0)
    return applied_joint_pos


def _print_scene_summary(
    *,
    scene_path: Path,
    variant: Literal["dual", "single"],
    arm_records: list[dict[str, object]],
    env_cfg,
) -> None:
    print(f"[PandaOrca inspect] base scene USD: {scene_path}")
    print(f"[PandaOrca inspect] variant: {variant}")
    print(f"[PandaOrca inspect] DROID env root: {ENV_ROOT_PRIM_PATH}")
    print(
        "[PandaOrca inspect] DROID timing "
        f"physics_dt={float(env_cfg.sim.dt):.6f} "
        f"decimation={int(env_cfg.decimation)} "
        f"render_interval={int(env_cfg.sim.render_interval)} "
        f"action_dt={float(env_cfg.sim.dt) * int(env_cfg.decimation):.6f}"
    )
    print("[PandaOrca inspect] added PandaOrca arm references:")
    for record in arm_records:
        print(
            "  - "
            f"label={record['label']} "
            f"prim={record['prim_path']} "
            f"usd={record['usd_path']} "
            f"translate={record['translate_xyz']} "
            f"arm_joint_pos={record.get('arm_joint_pos')}"
        )
    print(f"[PandaOrca inspect] future control candidate prim: {RIGHT_ARM_PRIM_PATH}")
    print("[PandaOrca inspect] inspect only, no policy/action/obs wiring")
    print("[PandaOrca inspect] DROID robot is not loaded in this inspect path")
    print("[PandaOrca inspect] TODO next step: articulation handle lookup + joint/action mapping")


def _run_loop(
    *,
    simulation_app,
    sim,
    headless: bool,
    play_physics: bool,
    frames: int,
    sleep_s: float,
) -> None:
    def _tick_once() -> None:
        if play_physics:
            sim.step(render=not headless)
        else:
            sim.render()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    if frames > 0:
        for _ in range(frames):
            if not headless and not simulation_app.is_running():
                break
            _tick_once()
        return

    if headless:
        print("[PandaOrca inspect] headless mode: running until interrupted (Ctrl-C).")
        while True:
            _tick_once()
    else:
        print("[PandaOrca inspect] GUI mode: running until the Isaac Sim window closes.")
        while simulation_app.is_running():
            _tick_once()


def main(
    scene: int = 1,
    variant: Literal["dual", "single"] = "dual",
    headless: bool = False,
    play_physics: bool = False,
    frames: int = 0,
    sleep_s: float = 0.02,
    base_x: float = 0.0,
    base_z: float = 0.0,
    right_y: float = -0.386,
    left_y: float | None = None,
):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Inspect PandaOrca arms inside the DROID Isaac scene without DreamZero communication."
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = False
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import sim_evals.environments  # noqa: F401
        import isaaclab.sim as sim_utils
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            "DROID",
            device=args_cli.device,
            num_envs=1,
            use_fabric=True,
        )
        env_cfg.set_scene(scene)

        scene_path = _open_droid_scene(scene)

        sim = sim_utils.SimulationContext(env_cfg.sim)
        sim.set_camera_view(env_cfg.viewer.eye, env_cfg.viewer.lookat)

        _ensure_xform_prim("/World/envs")
        _ensure_xform_prim(ENV_ROOT_PRIM_PATH)

        arm_records: list[dict[str, object]] = []
        pandaorca_robots: list[tuple[str, object]] = []

        resolved_left_y = -float(right_y) if left_y is None else float(left_y)

        right_translate = (float(base_x), float(right_y), float(base_z))
        _add_pandaorca_arm(
            prim_path=RIGHT_ARM_PRIM_PATH,
            usd_path=RIGHT_ARM_USD_PATH,
            translate_xyz=right_translate,
            orient_wxyz=IDENTITY_QUAT_WXYZ,
        )
        pandaorca_robots.append(("right", _create_pandaorca_articulation(RIGHT_ARM_PRIM_PATH)))
        arm_records.append(
            {
                "label": "right",
                "prim_path": RIGHT_ARM_PRIM_PATH,
                "usd_path": str(RIGHT_ARM_USD_PATH),
                "translate_xyz": right_translate,
            }
        )

        if variant == "dual":
            left_translate = (float(base_x), resolved_left_y, float(base_z))
            _add_pandaorca_arm(
                prim_path=LEFT_ARM_PRIM_PATH,
                usd_path=LEFT_ARM_USD_PATH,
                translate_xyz=left_translate,
                orient_wxyz=IDENTITY_QUAT_WXYZ,
            )
            pandaorca_robots.append(("left", _create_pandaorca_articulation(LEFT_ARM_PRIM_PATH)))
            arm_records.append(
                {
                    "label": "left",
                    "prim_path": LEFT_ARM_PRIM_PATH,
                    "usd_path": str(LEFT_ARM_USD_PATH),
                    "translate_xyz": left_translate,
                }
            )

        _wait_for_stage_load()
        sim.reset()
        for label, robot in pandaorca_robots:
            if not robot.is_initialized:
                raise RuntimeError(f"PandaOrca articulation failed to initialize for {label} arm at {robot.cfg.prim_path}")
            applied_joint_pos = _apply_droid_like_arm_pose(robot, label)
            for record in arm_records:
                if record["label"] == label:
                    record["arm_joint_pos"] = applied_joint_pos
                    break
        for _ in range(4):
            sim.render()

        _print_scene_summary(
            scene_path=scene_path,
            variant=variant,
            arm_records=arm_records,
            env_cfg=env_cfg,
        )

        try:
            _run_loop(
                simulation_app=simulation_app,
                sim=sim,
                headless=headless,
                play_physics=play_physics,
                frames=frames,
                sleep_s=sleep_s,
            )
        except KeyboardInterrupt:
            print("[PandaOrca inspect] interrupted by user, shutting down.")
    except Exception as exc:
        print(f"[PandaOrca inspect] fatal error: {exc}")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)
