from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Literal

import numpy as np
import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
SIM_EVALS_ASSETS_ROOT = REPO_ROOT / "sim_evals" / "assets"
PANDAORCA_USD_ROOT = WORKSPACE_ROOT / "pandaorca_description" / "usd"

ENV_ROOT_PRIM_PATH = "/World/envs/env_0"
RIGHT_ARM_PRIM_PATH = f"{ENV_ROOT_PRIM_PATH}/pandaorca_right"
LEFT_ARM_PRIM_PATH = f"{ENV_ROOT_PRIM_PATH}/pandaorca_left"
RIGHT_CAMERA_PRIM_PATH = f"{RIGHT_ARM_PRIM_PATH}/right_exterior_camera"
LEFT_CAMERA_PRIM_PATH = f"{LEFT_ARM_PRIM_PATH}/left_exterior_camera"

RIGHT_ARM_USD_PATH = (
    PANDAORCA_USD_ROOT / "fer_orcahand_right_extended" / "fer_orcahand_right_extended.usd"
)
LEFT_ARM_USD_PATH = (
    PANDAORCA_USD_ROOT / "fer_orcahand_left_extended" / "fer_orcahand_left_extended.usd"
)

IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)

PANDAORCA_CAMERA_CALIB_CAM_TO_BASE = {
    "left": np.array(
        [
            [-0.02199727, -0.80581615, 0.59175708, 0.20403467],
            [-0.99905014, 0.03998766, 0.01731508, -0.25486327],
            [-0.03761575, -0.59081411, -0.80593036, 0.43379187],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "right": np.array(
        [
            [0.02933941, -0.83227828, 0.55358113, 0.17515134],
            [-0.99642232, 0.01956109, 0.0822187, 0.34649483],
            [-0.07925749, -0.55401284, -0.82872675, 0.46895363],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}

ARIA_INTRINSICS = np.array(
    [
        [133.25430222 * 2.0, 0.0, 320.0, 0.0],
        [0.0, 133.25430222 * 2.0, 240.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

ARIA_INTRINSICS_HALF = np.array(
    [
        [133.25430222, 0.0, 320.0 / 2.0, 0.0],
        [0.0, 133.25430222, 240.0 / 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

ARIA_HORIZONTAL_APERTURE = 5.376
ARIA_VERTICAL_APERTURE = ARIA_HORIZONTAL_APERTURE * 480.0 / 640.0

PANDAORCA_CAMERA_RESOLUTIONS = {
    "full": {
        "width": 640,
        "height": 480,
        "intrinsics": ARIA_INTRINSICS,
    },
    "half": {
        "width": 320,
        "height": 240,
        "intrinsics": ARIA_INTRINSICS_HALF,
    },
}

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


def _ros_to_opengl_rotation(rotation_ros: np.ndarray) -> np.ndarray:
    rotation_gl = rotation_ros.copy()
    rotation_gl[:, 1] *= -1.0
    rotation_gl[:, 2] *= -1.0
    return rotation_gl


def _rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _add_pandaorca_camera(
    *,
    prim_path: str,
    parent_prim_path: str,
    label: str,
    cam_to_base: np.ndarray,
    resolution_mode: Literal["full", "half"],
) -> dict[str, object]:
    from pxr import Gf, UsdGeom

    stage = _get_stage()
    existing = stage.GetPrimAtPath(prim_path)
    if existing.IsValid():
        raise ValueError(f"Camera prim path already exists in stage: {prim_path}")

    # The calibration is provided as camera -> arm-base, which already matches
    # the local USD transform direction we need here: parent(base) <- child(cam).
    # The remaining conversion is only between camera axis conventions:
    # ROS/OpenCV-style (+Z forward, +Y down) -> USD/OpenGL (-Z forward, +Y up).
    translate_xyz = tuple(float(v) for v in cam_to_base[:3, 3])
    orient_wxyz = _rotation_matrix_to_quat_wxyz(_ros_to_opengl_rotation(cam_to_base[:3, :3]))

    camera_cfg = PANDAORCA_CAMERA_RESOLUTIONS[resolution_mode]
    intrinsics = camera_cfg["intrinsics"]
    width = int(camera_cfg["width"])
    height = int(camera_cfg["height"])
    fx = float(intrinsics[0, 0])
    focal_length = fx * ARIA_HORIZONTAL_APERTURE / float(width)

    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera_prim = camera.GetPrim()
    if not camera_prim.IsValid():
        raise RuntimeError(f"Failed to define PandaOrca camera at path: {prim_path}")

    xformable = UsdGeom.Xformable(camera_prim)
    _set_or_create_translate_op(xformable, translate_xyz)
    _set_or_create_orient_op(xformable, orient_wxyz)

    camera.CreateProjectionAttr("perspective")
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(ARIA_HORIZONTAL_APERTURE)
    camera.CreateVerticalApertureAttr(ARIA_VERTICAL_APERTURE)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera.CreateFocusDistanceAttr(1.0)

    return {
        "label": label,
        "prim_path": prim_path,
        "parent_prim_path": parent_prim_path,
        "resolution_mode": resolution_mode,
        "width": width,
        "height": height,
        "focal_length": float(focal_length),
        "horizontal_aperture": float(ARIA_HORIZONTAL_APERTURE),
        "vertical_aperture": float(ARIA_VERTICAL_APERTURE),
        "translate_xyz": translate_xyz,
        "orient_wxyz": orient_wxyz,
    }


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
    camera_records: list[dict[str, object]],
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
    print("[PandaOrca inspect] added calibrated cameras:")
    for record in camera_records:
        print(
            "  - "
            f"label={record['label']} "
            f"prim={record['prim_path']} "
            f"parent={record['parent_prim_path']} "
            f"resolution={record['width']}x{record['height']} ({record['resolution_mode']}) "
            f"translate={record['translate_xyz']} "
            f"orient={record['orient_wxyz']}"
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
    camera_resolution: Literal["full", "half"] = "full",
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
    args_cli.enable_cameras = True
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
        camera_records: list[dict[str, object]] = []
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
        camera_records.append(
            _add_pandaorca_camera(
                prim_path=RIGHT_CAMERA_PRIM_PATH,
                parent_prim_path=RIGHT_ARM_PRIM_PATH,
                label="right_cam",
                cam_to_base=PANDAORCA_CAMERA_CALIB_CAM_TO_BASE["right"],
                resolution_mode=camera_resolution,
            )
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
            camera_records.append(
                _add_pandaorca_camera(
                    prim_path=LEFT_CAMERA_PRIM_PATH,
                    parent_prim_path=LEFT_ARM_PRIM_PATH,
                    label="left_cam",
                    cam_to_base=PANDAORCA_CAMERA_CALIB_CAM_TO_BASE["left"],
                    resolution_mode=camera_resolution,
                )
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
            camera_records=camera_records,
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
