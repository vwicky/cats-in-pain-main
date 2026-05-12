"""
scan_videos.py — Pre-scan videos before DeepLabCut inference.

Usage:
    python scan_videos.py /path/to/your/video/folder

Options:
    --delete       Delete unreadable videos (default: just report)
    --move FOLDER  Move unreadable videos to FOLDER instead of deleting
    --fix          Re-encode bad videos in-place using ffmpeg (requires ffmpeg installed)
    --min-frames N Fail videos with fewer than N readable frames (default: 1)

Examples:
    python scan_videos.py ~/videos --move ~/videos/bad
    python scan_videos.py ~/videos --fix
    python scan_videos.py ~/videos --delete
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def check_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        print("ERROR: OpenCV not found. Install it with:\n  pip install opencv-python-headless")
        sys.exit(1)


def scan_video(cv2, path: Path, min_frames: int = 1) -> dict:
    """
    Returns a dict with:
      - readable: bool
      - metadata_frames: int (what the file header claims)
      - actual_frames: int (frames OpenCV could actually read)
      - error: str or None
    """
    result = {"path": path, "readable": False, "metadata_frames": 0, "actual_frames": 0, "error": None}

    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            result["error"] = "cv2 could not open file"
            return result

        result["metadata_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try reading up to the first 10 frames to confirm data is there
        read_ok = 0
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                read_ok += 1
            else:
                break
        cap.release()

        result["actual_frames"] = read_ok
        result["readable"] = read_ok >= min_frames

        if result["metadata_frames"] > 0 and read_ok == 0:
            result["error"] = f"metadata says {result['metadata_frames']} frames but 0 were readable (encoding issue)"

    except Exception as e:
        result["error"] = str(e)

    return result


def re_encode(src: Path) -> bool:
    """Re-encode a video with ffmpeg to fix container/codec issues."""
    if shutil.which("ffmpeg") is None:
        print("  [skip] ffmpeg not found in PATH — cannot re-encode")
        return False

    tmp = src.with_suffix(".tmp_fixed" + src.suffix)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac",
        str(tmp)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and tmp.exists():
        src.unlink()
        tmp.rename(src)
        return True
    else:
        if tmp.exists():
            tmp.unlink()
        print(f"  [ffmpeg error] {result.stderr[-300:]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-scan videos for DeepLabCut compatibility")
    parser.add_argument("folder", help="Folder containing videos to scan")
    parser.add_argument("--delete", action="store_true", help="Delete unreadable videos")
    parser.add_argument("--move", metavar="DEST", help="Move unreadable videos to this folder")
    parser.add_argument("--fix", action="store_true", help="Re-encode unreadable videos with ffmpeg")
    parser.add_argument("--min-frames", type=int, default=1, metavar="N",
                        help="Minimum readable frames to consider a video OK (default: 1)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: '{folder}' is not a directory.")
        sys.exit(1)

    cv2 = check_cv2()

    videos = [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        print(f"No video files found in {folder}")
        sys.exit(0)

    print(f"\nScanning {len(videos)} video(s) in: {folder}\n")
    print(f"  {'STATUS':<8}  {'METADATA':>10}  {'READABLE':>10}  FILE")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*50}")

    bad = []
    ok_count = 0

    for path in sorted(videos):
        info = scan_video(cv2, path, args.min_frames)
        status = "OK" if info["readable"] else "BAD"
        flag = "" if info["readable"] else "  <-- problem"
        rel = path.relative_to(folder) if path.is_relative_to(folder) else path
        print(f"  {status:<8}  {info['metadata_frames']:>10}  {info['actual_frames']:>10}  {rel}{flag}")
        if info["error"]:
            print(f"           error: {info['error']}")
        if info["readable"]:
            ok_count += 1
        else:
            bad.append(path)

    print(f"\nResult: {ok_count} OK, {len(bad)} bad\n")

    if not bad:
        print("All videos look good!")
        return

    # --- Actions on bad videos ---
    if args.fix:
        print("Re-encoding bad videos with ffmpeg...")
        for p in bad:
            print(f"  {p.name} ... ", end="", flush=True)
            success = re_encode(p)
            print("fixed" if success else "FAILED")

    elif args.delete:
        print("Deleting bad videos...")
        for p in bad:
            p.unlink()
            print(f"  Deleted: {p}")

    elif args.move:
        dest = Path(args.move)
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Moving bad videos to: {dest}")
        for p in bad:
            target = dest / p.name
            shutil.move(str(p), target)
            print(f"  Moved: {p.name}")

    else:
        print("No action taken. Use --fix, --delete, or --move FOLDER to handle bad videos.")
        print("Tip: --fix re-encodes them with ffmpeg and keeps filenames unchanged.")


if __name__ == "__main__":
    main()