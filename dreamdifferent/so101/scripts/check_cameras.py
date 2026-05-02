import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "camera_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe OpenCV camera indices before recording. "
            "On macOS this helps distinguish USB cameras from iPhone Continuity Camera."
        )
    )
    parser.add_argument("--max-index", type=int, default=8, help="Probe camera indices 0..max-index.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames to discard after opening each camera before saving a snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where snapshots/contact sheet are saved.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print available indices; do not save snapshots.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window for each detected camera. Press q/esc/space to continue.",
    )
    return parser.parse_args()


def list_macos_camera_names() -> None:
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    lines = [
        line.rstrip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.strip().endswith(":")
    ]
    if not lines:
        return

    print("macOS camera devices reported by system_profiler:")
    for line in lines:
        print(f"  {line.strip()}")
    print(
        "\nNote: OpenCV indices do not always map directly to these names; "
        "use the saved snapshots/contact sheet below to identify them."
    )
    print()


def open_camera(index: int, args: argparse.Namespace):
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def capture_snapshot(cap, warmup_frames: int):
    frame = None
    for _ in range(max(warmup_frames, 1)):
        ok, maybe_frame = cap.read()
        if ok:
            frame = maybe_frame
        time.sleep(0.02)
    return frame


def save_contact_sheet(frames: list[tuple[int, object]], output_path: Path) -> None:
    if not frames:
        return

    import cv2
    import numpy as np

    thumb_w, thumb_h = 320, 240
    tiles = []
    for index, frame in frames:
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        cv2.rectangle(thumb, (0, 0), (150, 42), (0, 0, 0), thickness=-1)
        cv2.putText(
            thumb,
            f"index {index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(thumb)

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    grid = []
    for row_idx in range(rows):
        row = []
        for col_idx in range(cols):
            tile_idx = row_idx * cols + col_idx
            row.append(tiles[tile_idx] if tile_idx < len(tiles) else blank)
        grid.append(np.hstack(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(grid))


def preview_camera(index: int, cap) -> None:
    import cv2

    window_name = f"OpenCV camera index {index} - press q/esc/space to continue"
    print(f"Previewing index {index}. Press q/esc/space in the preview window to continue.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(
            frame,
            f"index {index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord(" ")):
            break
    cv2.destroyWindow(window_name)


def main() -> None:
    args = parse_args()

    try:
        import cv2
    except ImportError as e:
        raise SystemExit(
            "OpenCV is required to probe cameras. Run this from the lerobot environment, e.g.\n"
            "  conda activate lerobot\n"
            "  python scripts/check_cameras.py"
        ) from e

    list_macos_camera_names()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = args.output_dir / timestamp
    detected_frames: list[tuple[int, object]] = []

    print(f"Probing OpenCV camera indices 0..{args.max_index} with AVFoundation backend...\n")
    print("index | status | reported resolution | snapshot")
    print("------+--------+---------------------+---------")

    for index in range(args.max_index + 1):
        cap = open_camera(index, args)
        if cap is None:
            print(f"{index:>5} | closed | -                   | -")
            continue

        try:
            frame = capture_snapshot(cap, args.warmup_frames)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame is None:
                print(f"{index:>5} | opened/no frame | {width}x{height:<12} | -")
                continue

            snapshot_path = "-"
            if not args.no_save:
                run_output_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = str(run_output_dir / f"camera_index_{index}.jpg")
                cv2.imwrite(snapshot_path, frame)

            detected_frames.append((index, frame.copy()))
            print(f"{index:>5} | ok     | {width}x{height:<12} | {snapshot_path}")

            if args.preview:
                preview_camera(index, cap)
        finally:
            cap.release()

    if detected_frames and not args.no_save:
        contact_sheet_path = run_output_dir / "contact_sheet.jpg"
        save_contact_sheet(detected_frames, contact_sheet_path)
        print(f"\nSaved contact sheet: {contact_sheet_path}")

    if detected_frames:
        indices = ", ".join(str(index) for index, _ in detected_frames)
        print(f"\nDetected camera indices: {indices}")
        print("Use these in scripts/record.sh as --camera-index and --secondary-camera-index.")
    else:
        print("\nNo cameras detected. Check macOS camera permissions for Terminal/Python.")


if __name__ == "__main__":
    main()
