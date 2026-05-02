import argparse
import os
import sys
import threading
import time
from pathlib import Path

from script_utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_FINAL_POSE_PATH,
    DEFAULT_HOME_POSE_PATH,
    DEFAULT_PORTS_PATH,
    follower_config_kwargs,
    leader_config_kwargs,
    load_final_pose,
    load_ports,
    move_robot_to_pose,
)


DEFAULT_DATASET_REPO_ID = "local/so101_teleop"


def build_robot_config(args: argparse.Namespace, follower_port: str):
    from lerobot.robots.so_follower import SO101FollowerConfig

    config_kwargs = follower_config_kwargs(follower_port)

    camera_config = build_camera_configs(args)
    if not camera_config:
        return SO101FollowerConfig(**config_kwargs)

    return SO101FollowerConfig(
        **config_kwargs,
        cameras=camera_config,
    )


def build_camera_configs(args: argparse.Namespace) -> dict[str, object]:
    if not (args.camera or args.secondary_camera):
        return {}

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    camera_config = {}
    if args.camera:
        camera_config[args.camera_name] = OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )

    if args.secondary_camera:
        camera_config[args.secondary_camera_name] = OpenCVCameraConfig(
            index_or_path=args.secondary_camera_index,
            width=args.secondary_camera_width,
            height=args.secondary_camera_height,
            fps=args.fps,
        )

    return camera_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PORTS_PATH,
        help="Path to the JSON file containing leader/follower ports.",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-sec", type=int, default=60)
    parser.add_argument("--reset-time-sec", type=int, default=10)
    parser.add_argument("--task", default="SO101 teleoperation task")
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Dataset repo identifier used for local dataset creation metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base directory where the dataset will be stored locally.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append new episodes to an existing dataset instead of creating a new one.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the recorded dataset to the Hugging Face Hub after recording finishes.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the Hugging Face dataset repo as private when pushing.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional Hugging Face dataset card tags to attach when pushing.",
    )
    parser.add_argument(
        "--license",
        default="apache-2.0",
        help="Dataset license metadata used when pushing to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional Hugging Face branch to push the dataset to.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision/branch to download when resuming from the Hub, e.g. 'main'.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable name that stores the Hugging Face token for upload.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable the primary follower camera and record observations with images.",
    )
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--secondary-camera",
        action="store_true",
        help="Enable a secondary follower camera and record it alongside the primary camera.",
    )
    parser.add_argument("--secondary-camera-name", default="wrist")
    parser.add_argument("--secondary-camera-index", type=int, default=1)
    parser.add_argument("--secondary-camera-width", type=int, default=640)
    parser.add_argument("--secondary-camera-height", type=int, default=480)
    parser.add_argument(
        "--home-pose-path",
        type=Path,
        default=DEFAULT_HOME_POSE_PATH,
        help="Deprecated; paused recording now follows the leader instead of returning to a home pose.",
    )
    parser.add_argument(
        "--final-pose-path",
        type=Path,
        default=DEFAULT_FINAL_POSE_PATH,
        help="Path to a saved final pose JSON file used when exiting recording.",
    )
    parser.add_argument(
        "--return-pose-source",
        choices=("initial", "home"),
        default="home",
        help="Deprecated; accepted for compatibility but ignored.",
    )
    parser.add_argument(
        "--return-to-initial-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated; accepted for compatibility but ignored.",
    )
    parser.add_argument(
        "--return-move-time-sec",
        type=float,
        default=1.5,
        help="Duration of the smooth motion used when returning to the final pose at session exit.",
    )
    return parser.parse_args()


def maybe_login_to_huggingface(token_env_var: str) -> None:
    token = os.getenv(token_env_var) or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def is_local_repo_id(repo_id: str) -> bool:
    return repo_id.split("/", 1)[0] == "local" if "/" in repo_id else True


def has_local_dataset_metadata(dataset_root: Path) -> bool:
    return (dataset_root / "meta" / "info.json").exists()


def has_local_episode_metadata(dataset_root: Path) -> bool:
    episodes_dir = dataset_root / "meta" / "episodes"
    return episodes_dir.exists() and any(episodes_dir.rglob("*.parquet"))


def has_finalized_local_dataset(dataset_root: Path) -> bool:
    return has_local_dataset_metadata(dataset_root) and has_local_episode_metadata(dataset_root)


def has_partial_local_dataset(dataset_root: Path) -> bool:
    return dataset_root.exists() and any(dataset_root.iterdir()) and not has_finalized_local_dataset(dataset_root)


def prepare_dataset_root_for_create(dataset_root: Path) -> None:
    if dataset_root.exists() and not any(dataset_root.iterdir()):
        dataset_root.rmdir()


def print_recording_controls() -> None:
    print("\nKeyboard controls:")
    print("  Space       start recording the current episode")
    print("  Right arrow save the current episode")
    print("  Left arrow  discard the current episode and reset")
    print("  Esc         stop the recording session")
    print()


def print_recording_status(
    *,
    episode_idx: int,
    num_episodes: int,
    recording: bool,
    detail: str | None = None,
) -> None:
    status = "recording in progress" if recording else "recording paused"
    message = f"[episode {episode_idx + 1}/{num_episodes}] {status}"
    if detail:
        message = f"{message} - {detail}"
    print(message, flush=True)


def print_terminal_event(message: str) -> None:
    sys.stdout.write("\r" + " " * 120 + "\r" + message + "\n")
    sys.stdout.flush()


def print_elapsed_recording_status(
    events: dict,
    episode_idx: int,
    num_episodes: int,
    episode_time_sec: int,
) -> tuple[threading.Event, threading.Thread]:
    done = threading.Event()

    def update_status() -> None:
        start_t = time.perf_counter()
        while not done.is_set() and not events["exit_early"] and not events["stop_recording"]:
            elapsed_s = time.perf_counter() - start_t
            remaining_s = max(episode_time_sec - elapsed_s, 0.0)
            sys.stdout.write(
                "\r"
                f"[episode {episode_idx + 1}/{num_episodes}] recording in progress "
                f"- elapsed {elapsed_s:05.1f}s / {episode_time_sec}s "
                f"- remaining {remaining_s:05.1f}s "
                "- Right saves, Left discards, Esc stops"
            )
            sys.stdout.flush()
            done.wait(0.2)

        elapsed_s = time.perf_counter() - start_t
        sys.stdout.write(
            "\r"
            f"[episode {episode_idx + 1}/{num_episodes}] recording paused "
            f"- elapsed {elapsed_s:05.1f}s / {episode_time_sec}s"
            + " " * 40
            + "\n"
        )
        sys.stdout.flush()

    thread = threading.Thread(target=update_status, daemon=True)
    thread.start()
    return done, thread


def init_episode_keyboard_listener():
    try:
        from lerobot.utils.control_utils import is_headless
    except ImportError:
        from lerobot.common.control_utils import is_headless

    if is_headless():
        raise RuntimeError(
            "Keyboard controls are required for manual episode recording, but a headless environment was detected."
        )

    from pynput import keyboard

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "start_episode": False,
        "save_episode": False,
        "episode_decision_enabled": False,
    }

    def on_press(key):
        if key == keyboard.Key.space:
            events["start_episode"] = True
        elif key == keyboard.Key.right:
            if not events["episode_decision_enabled"]:
                return
            print_terminal_event("Right arrow pressed. Saving this episode...")
            events["save_episode"] = True
            events["exit_early"] = True
        elif key == keyboard.Key.left:
            if not events["episode_decision_enabled"]:
                return
            print_terminal_event("Left arrow pressed. Discarding this episode...")
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif key == keyboard.Key.esc:
            print_terminal_event("Escape pressed. Stopping data recording...")
            events["stop_recording"] = True
            events["exit_early"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def follow_leader_until_episode_start(
    *,
    robot,
    teleop,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    episode_idx: int,
    num_episodes: int,
) -> bool:
    from lerobot.utils.robot_utils import precise_sleep

    events["start_episode"] = False
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["save_episode"] = False
    events["episode_decision_enabled"] = False
    print_recording_status(
        episode_idx=episode_idx,
        num_episodes=num_episodes,
        recording=False,
        detail="following leader; press Space to start, or Esc to stop",
    )

    status_t = 0.0
    while not events["stop_recording"]:
        if events["start_episode"]:
            events["start_episode"] = False
            sys.stdout.write("\n")
            sys.stdout.flush()
            return True

        start_loop_t = time.perf_counter()
        obs = robot.get_observation()
        act = teleop.get_action()
        act_processed_teleop = teleop_action_processor((act, obs))
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        robot.send_action(robot_action_to_send)

        now = time.perf_counter()
        if now - status_t >= 0.5:
            status_t = now
            sys.stdout.write(
                "\r"
                f"[episode {episode_idx + 1}/{num_episodes}] recording paused "
                "- following leader; press Space to start, Esc to stop"
            )
            sys.stdout.flush()

        precise_sleep(max(1 / fps - (time.perf_counter() - start_loop_t), 0.0))

    sys.stdout.write("\n")
    sys.stdout.flush()
    return False


def wait_for_episode_decision(events: dict, episode_idx: int, num_episodes: int) -> str:
    from lerobot.utils.robot_utils import precise_sleep

    if events["stop_recording"]:
        return "stop"
    if events["rerecord_episode"]:
        events["episode_decision_enabled"] = False
        return "discard"
    if events["save_episode"]:
        events["episode_decision_enabled"] = False
        return "save"

    events["exit_early"] = False
    print_recording_status(
        episode_idx=episode_idx,
        num_episodes=num_episodes,
        recording=False,
        detail="episode time ended; press Right to save or Left to discard",
    )
    while not events["stop_recording"]:
        if events["rerecord_episode"]:
            events["episode_decision_enabled"] = False
            return "discard"
        if events["save_episode"]:
            events["episode_decision_enabled"] = False
            return "save"
        precise_sleep(0.1)
    events["episode_decision_enabled"] = False
    return "stop"



def validate_push_to_hub_args(args: argparse.Namespace) -> None:
    if not args.push_to_hub:
        return

    if "/" not in args.dataset_repo_id:
        raise ValueError(
            "--push-to-hub requires --dataset-repo-id to look like '<hf_username>/<dataset_name>'."
        )

    namespace, _ = args.dataset_repo_id.split("/", 1)
    if namespace == "local":
        raise ValueError(
            "--push-to-hub cannot be used with the default local repo id. "
            "Please set --dataset-repo-id to your real Hugging Face namespace, e.g. '<hf_username>/so101_teleop'."
        )


def main() -> None:
    args = parse_args()
    validate_push_to_hub_args(args)
    camera_configs = build_camera_configs(args)

    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        from lerobot.datasets.feature_utils import hw_to_dataset_features
    except ImportError:
        from lerobot.utils.feature_utils import hw_to_dataset_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    try:
        from lerobot.processor import make_default_processors
    except ImportError:
        from lerobot.processor.factory import make_default_processors
    from lerobot.robots.so_follower import SO101Follower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import init_rerun

    ports = load_ports(args.config)

    robot_config = build_robot_config(args, ports["follower"])
    teleop_config = SO101LeaderConfig(**leader_config_kwargs(ports["leader"]))

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    dataset_root = args.dataset_root / args.dataset_repo_id

    if args.resume:
        if is_local_repo_id(args.dataset_repo_id) and not has_finalized_local_dataset(dataset_root):
            raise FileNotFoundError(
                f"Cannot resume local dataset because no finalized dataset metadata was found at: {dataset_root}"
            )

        try:
            if not has_finalized_local_dataset(dataset_root) and not is_local_repo_id(args.dataset_repo_id):
                maybe_login_to_huggingface(args.hf_token_env)

            dataset = LeRobotDataset.resume(
                repo_id=args.dataset_repo_id,
                root=dataset_root,
                revision=args.revision,
                image_writer_threads=args.image_writer_threads,
            )
        except RepositoryNotFoundError:
            if has_partial_local_dataset(dataset_root):
                raise FileNotFoundError(
                    f"A partial local dataset directory exists at {dataset_root}, but no matching Hub dataset was found. "
                    "This usually happens when a previous run was interrupted before the dataset was pushed. "
                    "Delete or rename the local directory and try again, or rerun without --resume."
                )

            if dataset_root.exists() and any(dataset_root.iterdir()):
                raise FileNotFoundError(
                    f"Could not resume from the Hub, and the local dataset directory is not a finalized dataset: {dataset_root}. "
                    "Delete or rename the directory and try again, or rerun without --resume."
                )

            prepare_dataset_root_for_create(dataset_root)
            print("No existing Hub dataset was found. Starting a new local dataset instead.")
            dataset = LeRobotDataset.create(
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                root=dataset_root,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=bool(camera_configs),
                image_writer_threads=args.image_writer_threads,
            )
        except FileNotFoundError:
            if has_partial_local_dataset(dataset_root) and not is_local_repo_id(args.dataset_repo_id):
                maybe_login_to_huggingface(args.hf_token_env)
                dataset = LeRobotDataset.resume(
                    repo_id=args.dataset_repo_id,
                    root=dataset_root,
                    revision=args.revision,
                    image_writer_threads=args.image_writer_threads,
                )
            else:
                raise
    else:
        prepare_dataset_root_for_create(dataset_root)
        try:
            dataset = LeRobotDataset.create(
                repo_id=args.dataset_repo_id,
                fps=args.fps,
                root=dataset_root,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=bool(camera_configs),
                image_writer_threads=args.image_writer_threads,
            )
        except FileExistsError as e:
            raise FileExistsError(
                f"Dataset directory already exists: {dataset_root}. "
                "Use --resume to append episodes, or choose a different --dataset-repo-id / --dataset-root."
            ) from e

    listener, events = init_episode_keyboard_listener()
    init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()
    final_pose = load_final_pose(args.final_pose_path) if args.final_pose_path.exists() else None

    try:
        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        episode_idx = 0
        while episode_idx < args.num_episodes and not events["stop_recording"]:
            print_recording_controls()
            if not follow_leader_until_episode_start(
                robot=robot,
                teleop=teleop,
                events=events,
                fps=args.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
            ):
                break

            events["exit_early"] = False
            events["rerecord_episode"] = False
            events["save_episode"] = False
            events["episode_decision_enabled"] = True
            status_done, status_thread = print_elapsed_recording_status(
                events=events,
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
                episode_time_sec=args.episode_time_sec,
            )
            try:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=args.episode_time_sec,
                    single_task=args.task,
                    display_data=True,
                )
            finally:
                status_done.set()
                status_thread.join()

            decision = wait_for_episode_decision(events, episode_idx, args.num_episodes)

            if decision == "discard":
                events["rerecord_episode"] = False
                events["exit_early"] = False
                events["save_episode"] = False
                events["episode_decision_enabled"] = False
                dataset.clear_episode_buffer()
                print_recording_status(
                    episode_idx=episode_idx,
                    num_episodes=args.num_episodes,
                    recording=False,
                    detail="episode discarded; reset the environment, then press Space",
                )
                continue

            if decision == "stop":
                dataset.clear_episode_buffer()
                print_recording_status(
                    episode_idx=episode_idx,
                    num_episodes=args.num_episodes,
                    recording=False,
                    detail="session stopping; unsaved episode discarded",
                )
                break

            events["save_episode"] = False
            events["exit_early"] = False
            events["rerecord_episode"] = False
            events["episode_decision_enabled"] = False
            print_recording_status(
                episode_idx=episode_idx,
                num_episodes=args.num_episodes,
                recording=False,
                detail="saving episode",
            )
            dataset.save_episode()

            episode_idx += 1
            if episode_idx < args.num_episodes:
                print_recording_status(
                    episode_idx=episode_idx,
                    num_episodes=args.num_episodes,
                    recording=False,
                    detail="saved; reset the environment, then press Space",
                )
    finally:
        print("Stop recording")
        if listener is not None:
            listener.stop()
        dataset.finalize()
        if final_pose is not None:
            print("Returning robot to final pose")
            move_robot_to_pose(
                robot=robot,
                target_pose=final_pose,
                duration_s=args.return_move_time_sec,
                fps=args.fps,
            )
            precise_sleep(0.2)
        robot.disconnect()
        teleop.disconnect()

    if args.push_to_hub:
        maybe_login_to_huggingface(args.hf_token_env)
        dataset.push_to_hub(
            branch=args.branch,
            tags=args.tags,
            license=args.license,
            private=args.private,
        )


if __name__ == "__main__":
    main()
