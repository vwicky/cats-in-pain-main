"""
run_inference.py — Batch DeepLabCut SuperAnimal inference with logging.

Usage:
    python run_inference.py --videos /path/to/videos --output /path/to/results

Options:
    --videos   DIR     Folder of .mp4 videos to process (required)
    --output   DIR     Where DLC saves result CSV/H5 files (required)
    --log      FILE    Path to JSON log file (default: ./dlc_inference_log.json)
    --model    NAME    SuperAnimal model name (default: superanimal_quadruped)
    --config   FILE    Optional path to DLC project config.yaml
    --batch    N       Detector batch size (default: 8)
    --thresh   F       Detector confidence threshold 0–1 (default: 0.1)
    --retry           Retry previously failed videos

Example:
    python run_inference.py \\
        --videos ~/dataset/downsampled_5fps_videos \\
        --output ~/dataset/deeplabcut_labeled \\
        --log ~/dataset/inference_log.json \\
        --model superanimal_quadruped
"""

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


# ── Log helpers ───────────────────────────────────────────────────────────────

def load_log(log_path: Path) -> dict:
    if log_path.exists():
        try:
            with open(log_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            backup = log_path.with_suffix(".corrupted.json")
            log_path.rename(backup)
            print(f"WARNING: Log file was corrupted — moved to {backup.name}, starting fresh.")
    return {"videos": {}, "summary": {"total": 0, "ok": 0, "skipped": 0, "failed": 0}}


def save_log(log_path: Path, log: dict):
    """Atomic write — avoids corrupting the log on Ctrl+C mid-save."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = log_path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    tmp.replace(log_path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Core ──────────────────────────────────────────────────────────────────────

def process_videos(args):
    import deeplabcut

    video_dir = Path(args.videos)
    output_dir = Path(args.output)
    log_path = Path(args.log)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in video_dir.rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    log = load_log(log_path)

    # Filter out already-processed videos unless --retry is set
    pending = []
    skipped = 0
    for v in videos:
        key = str(v)
        entry = log["videos"].get(key, {})
        if entry.get("status") == "ok" and not args.retry:
            skipped += 1
        else:
            pending.append(v)

    print(f"\nVideos total:     {len(videos)}")
    print(f"Already done:     {skipped}")
    print(f"To process:       {len(pending)}")
    print(f"Model:            {args.model}")
    print(f"Output:           {output_dir}")
    print(f"Log:              {log_path}\n")

    ok_count = 0
    fail_count = 0

    bar = tqdm(pending, unit="video", ncols=90, colour="green")

    try:
        for video_path in bar:
            key = str(video_path)
            bar.set_description(video_path.name[:40])

            start = time.perf_counter()
            started_at = utc_now()

            try:
                kwargs = dict(
                    videos=[str(video_path)],
                    superanimal_name=args.model,
                    model_name=args.model,
                    pcutoff=args.pcutoff,
                    bbox_threshold=args.bbox_threshold,
                    create_labeled_video=False,
                    plot_trajectories=False,
                    videotype=video_path.suffix.lstrip("."),
                    dest_folder=str(output_dir),
                    batch_size=args.batch,
                    detector_batch_size=args.batch,
                )
                if args.config:
                    kwargs["config"] = args.config

                deeplabcut.video_inference_superanimal(**kwargs)

                elapsed = time.perf_counter() - start
                log["videos"][key] = {
                    "status": "ok",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "elapsed_s": round(elapsed, 2),
                }
                ok_count += 1
                bar.set_postfix(ok=ok_count, fail=fail_count)

            except ValueError as e:
                elapsed = time.perf_counter() - start
                if "need at least one array to stack" in str(e):
                    msg = "no detections (empty predictions)"
                    tqdm.write(f"  [NO DETECTIONS] {video_path.name}")
                else:
                    msg = str(e)
                    tqdm.write(f"  [VALUE ERROR] {video_path.name}: {msg}")

                log["videos"][key] = {
                    "status": "failed",
                    "reason": msg,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "elapsed_s": round(elapsed, 2),
                }
                fail_count += 1
                bar.set_postfix(ok=ok_count, fail=fail_count)

            except Exception as e:
                elapsed = time.perf_counter() - start
                tb = traceback.format_exc()
                tqdm.write(f"  [ERROR] {video_path.name}: {e}")

                log["videos"][key] = {
                    "status": "failed",
                    "reason": str(e),
                    "traceback": tb,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "elapsed_s": round(elapsed, 2),
                }
                fail_count += 1
                bar.set_postfix(ok=ok_count, fail=fail_count)

            # Save log after every video so progress survives crashes
            save_log(log_path, log)

    except KeyboardInterrupt:
        bar.close()
        print("\n\nInterrupted — saving progress to log...")
        save_log(log_path, log)
        print(f"  Saved. Re-run the same command to continue (already-OK videos will be skipped).\n")

    # Final summary in log
    log["summary"] = {
        "total": len(videos),
        "ok": ok_count + (log["summary"].get("ok", 0) if args.retry else skipped),
        "skipped": skipped,
        "failed": fail_count,
        "last_run": utc_now(),
    }
    save_log(log_path, log)

    print(f"\n{'─'*50}")
    print(f"  Done.   OK: {ok_count}   Failed: {fail_count}   Skipped: {skipped}")
    print(f"  Log saved to: {log_path}")
    print(f"{'─'*50}\n")

    if fail_count:
        print("Failed videos:")
        for key, entry in log["videos"].items():
            if entry.get("status") == "failed":
                print(f"  {Path(key).name}  —  {entry.get('reason', '?')}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch DLC SuperAnimal inference")
    parser.add_argument("--videos",  required=True,  help="Folder of input videos")
    parser.add_argument("--output",  required=True,  help="Folder for DLC output files")
    parser.add_argument("--log",     default="dlc_inference_log.json", help="JSON log file path")
    parser.add_argument("--model",   default="superanimal_quadruped",  help="SuperAnimal model name")
    parser.add_argument("--config",  default=None,   help="Optional DLC config.yaml path")
    parser.add_argument("--batch",   type=int,   default=8,   help="Batch size")
    parser.add_argument("--pcutoff",       type=float, default=0.1,  help="Keypoint confidence cutoff (default: 0.1)")
    parser.add_argument("--bbox-threshold", type=float, default=0.9,  help="Detector bbox confidence threshold (default: 0.9)")
    parser.add_argument("--retry",   action="store_true", help="Re-process previously failed videos")
    args = parser.parse_args()

    process_videos(args)


if __name__ == "__main__":
    main()