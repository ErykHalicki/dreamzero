import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from script_utils import DEFAULT_DATASET_ROOT


DEFAULT_REPO_ID = "dreamdifferent/so101_teleop_test_filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a LeRobot v3 dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id, e.g. dreamdifferent/so101_teleop_test_filtered.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Base directory for local LeRobot datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact local dataset directory. Defaults to "
            "<dataset-root>/<repo-id>, e.g. data/lerobot/dreamdifferent/so101_teleop_test_filtered."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision/branch/tag/commit to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing local dataset directory if it already exists.",
    )
    parser.add_argument(
        "--backup-on-overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the existing local dataset to .backup before overwriting.",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        help="Ask LeRobot to refresh local files from the Hub even if cache files exist.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Download/load metadata and parquet data without downloading videos.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face token for private datasets.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.dataset_root / args.repo_id


def maybe_login_to_huggingface(token_env_var: str) -> None:
    token = os.getenv(token_env_var) or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def prepare_output_dir(output_dir: Path, overwrite: bool, backup_on_overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if not overwrite:
        raise FileExistsError(
            f"Local dataset already exists: {output_dir}\n"
            "Use --overwrite to replace it, or choose a different --output-dir."
        )

    if backup_on_overwrite:
        backup_root = Path(".backup")
        backup_root.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_root / f"{output_dir.name}_{stamp}"
        shutil.move(str(output_dir), str(backup_dir))
        print(f"Moved existing dataset to backup: {backup_dir}")
    else:
        shutil.rmtree(output_dir)
        print(f"Removed existing dataset: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)

    prepare_output_dir(
        output_dir=output_dir,
        overwrite=args.overwrite,
        backup_on_overwrite=args.backup_on_overwrite,
    )
    maybe_login_to_huggingface(args.hf_token_env)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=output_dir,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=not args.no_videos,
    )

    print("Download/load complete.")
    print(f"repo_id: {args.repo_id}")
    print(f"revision: {dataset.revision}")
    print(f"root: {dataset.root}")
    print(f"frames: {len(dataset)}")
    print(f"episodes: {dataset.num_episodes}")
    print(f"tasks:\n{dataset.meta.tasks}")


if __name__ == "__main__":
    main()
