"""Step 1: Extract frames + run YOLO detection (run once).

For every clip in the manifest:
  - Sample N frames (config ``frames_per_clip``, default 15) from the uniform_inner
    window (10%–90% of clip duration).
  - Save full-resolution JPEGs to  dataset/full_frames/{clip_id}/frame_{01..NN}.jpg
  - Run YOLOv8 cat detection on each frame.
  - Append per-frame bbox results to  dataset_construction/yolo_bboxes.json (keyed by clip_id).

Each clip is written to disk immediately after extraction (YOLO + JSON flush in the
same step), so interrupting the run still keeps completed clips — unlike a batched
extract-then-save design.

Edge cases:
  - No cat detected → bbox entry has "detected": false; step 2 will fallback to
    the full image.
  - Video unreadable → all N entries get "detected": false with a warning.
  - Partial resume: if clip_id already has a complete entry in yolo_bboxes.json
    AND all frame JPEGs exist, the clip is skipped (re-run safe). Use ``--force``
    to re-process those clips.

Usage (from repo root):
    python scripts/01_extract_and_detect.py --config cnn_finetune/config/v2.yaml
    python scripts/01_extract_and_detect.py --config cnn_finetune/config/v2.yaml --limit 100
    python scripts/01_extract_and_detect.py --config cnn_finetune/config/v2.yaml --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "video" / "cnn-finetuning"))

CAT_CLASS_ID = 15
DEFAULT_FRAMES_PER_CLIP = 15
FULL_FRAMES_DIR = PROJECT_ROOT / "src" / "dataset_construction" / "full_frames"
BBOXES_PATH = PROJECT_ROOT / "src" / "dataset_construction" / "yolo_bboxes.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_detect")


# ── Helpers ─────────────────────────────────────────────────────────────────


def uniform_inner_timestamps(duration_sec: float, n: int = DEFAULT_FRAMES_PER_CLIP) -> list[float]:
    """Sample n timestamps evenly in [10%, 90%] of clip duration."""
    d = max(float(duration_sec), 0.1)
    start, end = 0.10 * d, 0.90 * d
    if n == 1:
        return [(start + end) / 2.0]
    return [start + i * (end - start) / (n - 1) for i in range(n)]


def extract_frames_from_video(
    video_path: str,
    timestamps: list[float],
) -> list[np.ndarray | None]:
    """Extract one frame per timestamp (RGB).  Returns None for missed frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return [None] * len(timestamps)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray | None] = []
    for t in timestamps:
        frame_idx = max(0, int(round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            frames.append(None)
    cap.release()
    return frames


def run_yolo_on_frames(
    frames: list[np.ndarray | None],
    yolo_model,
    yolo_conf: float,
    yolo_imgsz: int,
    yolo_device: str,
) -> list[dict]:
    """Run YOLO on a list of RGB frames; returns one bbox dict per frame."""
    results_list: list[dict] = []
    for frame in frames:
        entry: dict = {
            "detected": False,
            "confidence": 0.0,
            "bbox_xyxy": None,  # raw detection [x0,y0,x1,y1]
            "img_wh": None,     # [W, H] — used as fallback by step 2
        }
        if frame is None:
            results_list.append(entry)
            continue

        H, W = frame.shape[:2]
        entry["img_wh"] = [W, H]

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = yolo_model.predict(
                bgr, imgsz=yolo_imgsz, conf=yolo_conf,
                classes=[CAT_CLASS_ID], verbose=False, device=yolo_device,
            )[0]

        if res.boxes is not None and len(res.boxes) > 0:
            best_i = int(res.boxes.conf.argmax().item())
            xyxy = res.boxes.xyxy[best_i].cpu().numpy().tolist()
            conf_val = float(res.boxes.conf[best_i].item())
            entry["detected"] = True
            entry["confidence"] = conf_val
            entry["bbox_xyxy"] = [float(v) for v in xyxy]

        results_list.append(entry)
    return results_list


def save_frames(
    frames: list[np.ndarray | None],
    clip_dir: Path,
) -> list[str]:
    """Save RGB frames as JPEGs; returns list of relative paths."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    n_out = len(frames)
    for p in clip_dir.glob("frame_*.jpg"):
        try:
            idx = int(p.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        if idx > n_out:
            p.unlink()

    saved: list[str] = []
    for i, frame in enumerate(frames, start=1):
        fname = clip_dir / f"frame_{i:02d}.jpg"
        if frame is not None:
            Image.fromarray(frame).save(str(fname), quality=95)
        else:
            # Save a 1×1 black placeholder so the file always exists
            Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(str(fname))
        saved.append(str(fname.relative_to(PROJECT_ROOT)))
    return saved


def clip_is_done(clip_id: str, existing_bboxes: dict, n_frames: int) -> bool:
    """True if clip already has a complete entry and all frame files exist."""
    if clip_id not in existing_bboxes:
        return False
    entry = existing_bboxes[clip_id]
    if len(entry.get("frames", [])) != n_frames:
        return False
    if not all((PROJECT_ROOT / f["path"]).exists() for f in entry["frames"]):
        return False
    clip_dir = FULL_FRAMES_DIR / clip_id
    if not clip_dir.is_dir():
        return False
    for p in clip_dir.glob("frame_*.jpg"):
        try:
            idx = int(p.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            return False
        if idx > n_frames:
            return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract frames and run YOLO detection.")
    parser.add_argument("--config", default="video/cnn-finetuning/config/v2.yaml",
                        help="Path to YAML config (relative to repo root)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many clips (for testing)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel threads for frame extraction (YOLO still single-threaded)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract clips even if yolo_bboxes + frame files look complete")
    parser.add_argument("--yolo-device", default="auto",
                        help="YOLO device: auto | cpu | cuda | mps")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    frames_per_clip = int(cfg.get("frames_per_clip", DEFAULT_FRAMES_PER_CLIP))
    if frames_per_clip < 1:
        logger.error("frames_per_clip must be >= 1 (got %s)", frames_per_clip)
        sys.exit(1)

    yolo_weights = str(PROJECT_ROOT / cfg.get("yolo_weights", "models/yolo/yolov8x.pt"))
    yolo_conf = float(cfg.get("yolo_conf", 0.40))
    yolo_imgsz = int(cfg.get("yolo_imgsz", 640))

    # Resolve YOLO device
    if args.yolo_device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                yolo_device = "cuda"
            elif torch.backends.mps.is_available():
                yolo_device = "mps"
            else:
                yolo_device = "cpu"
        except ImportError:
            yolo_device = "cpu"
    else:
        yolo_device = args.yolo_device

    logger.info("Loading YOLO from %s on device=%s", yolo_weights, yolo_device)
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_weights)
    except Exception as e:
        logger.error("Failed to load YOLO: %s", e)
        sys.exit(1)

    # Load manifest
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    clips: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("suitable_for_training") and r.get("final_label_5")):
                continue
            vp = r.get("video_path")
            if not vp:
                continue
            p = Path(vp)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if not p.exists():
                continue
            clips.append({
                "clip_id": r["snippet_id"],
                "video_path": str(p),
                "duration_sec": float(r.get("duration_sec") or 1.0),
            })

    if args.limit:
        clips = clips[: args.limit]

    logger.info("Clips to process: %d", len(clips))
    logger.info("frames_per_clip=%d (from config, default %d)",
                frames_per_clip, DEFAULT_FRAMES_PER_CLIP)
    logger.info("Full-resolution frames directory: %s", FULL_FRAMES_DIR.resolve())

    # Load existing bboxes (resume support)
    existing_bboxes: dict = {}
    if BBOXES_PATH.exists():
        try:
            existing_bboxes = json.loads(BBOXES_PATH.read_text(encoding="utf-8"))
            logger.info("Loaded %d existing bbox entries (resume mode)", len(existing_bboxes))
        except Exception:
            logger.warning("Could not parse existing %s — starting fresh", BBOXES_PATH)

    if args.force:
        clips_todo = list(clips)
        logger.info("--force: re-processing all %d clips", len(clips_todo))
    else:
        clips_todo = [
            c for c in clips
            if not clip_is_done(c["clip_id"], existing_bboxes, frames_per_clip)
        ]
        logger.info("Skipping %d already-complete clips; %d remaining",
                    len(clips) - len(clips_todo), len(clips_todo))

    if not clips:
        logger.warning(
            "No clips loaded from manifest (check suitable_for_training, final_label_5, "
            "and that each video_path exists under the repo root).",
        )
    elif not clips_todo:
        logger.info(
            "Nothing to do — every clip is already complete for frames_per_clip=%d. "
            "Delete frame dirs, fix yolo_bboxes.json, or pass --force to re-run.",
            frames_per_clip,
        )

    BBOXES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FULL_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    def _extract_one(clip: dict) -> tuple[str, list[np.ndarray | None], list[float]]:
        ts = uniform_inner_timestamps(clip["duration_sec"], frames_per_clip)
        frames = extract_frames_from_video(clip["video_path"], ts)
        return clip["clip_id"], frames, ts

    def _detect_save_flush(
        cid: str,
        frames: list,
        timestamps: list[float],
    ) -> None:
        clip_dir = FULL_FRAMES_DIR / cid
        saved_paths = save_frames(frames, clip_dir)
        bbox_entries = run_yolo_on_frames(
            frames, yolo_model, yolo_conf, yolo_imgsz, yolo_device,
        )
        n = len(timestamps)
        existing_bboxes[cid] = {
            "frames": [
                {
                    "path": saved_paths[i],
                    "timestamp_sec": round(timestamps[i], 4),
                    **bbox_entries[i],
                }
                for i in range(n)
            ]
        }
        BBOXES_PATH.write_text(
            json.dumps(existing_bboxes, indent=2), encoding="utf-8"
        )

    # Extract → save JPEGs → YOLO → JSON per clip so work survives interrupts / OOM.
    if args.workers <= 1:
        for clip in tqdm(clips_todo, desc="extract+detect+save"):
            _, frames, ts = _extract_one(clip)
            _detect_save_flush(clip["clip_id"], frames, ts)
    else:
        q: Queue = Queue(maxsize=max(2, args.workers * 2))

        def producer() -> None:
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(_extract_one, c): c for c in clips_todo}
                    for fut in as_completed(futures):
                        cid, frames, ts = fut.result()
                        q.put((cid, frames, ts))
            except Exception as exc:
                q.put(("__error__", exc))
                return
            q.put(None)

        th = threading.Thread(target=producer, name="frame-extract", daemon=True)
        th.start()
        with tqdm(total=len(clips_todo), desc="extract+detect+save") as pbar:
            while True:
                item = q.get()
                if item is None:
                    break
                if item[0] == "__error__":
                    th.join()
                    raise item[1]
                cid, frames, ts = item
                _detect_save_flush(cid, frames, ts)
                pbar.update(1)
        th.join()

    total_detected = sum(
        sum(1 for f in v["frames"] if f["detected"])
        for v in existing_bboxes.values()
    )
    total_frames = sum(len(v["frames"]) for v in existing_bboxes.values())
    logger.info(
        "Done. %d clips | %d/%d frames with cat detected (%.1f%%)",
        len(existing_bboxes),
        total_detected,
        total_frames,
        100.0 * total_detected / max(total_frames, 1),
    )
    logger.info("Bboxes saved to: %s", BBOXES_PATH)
    logger.info("Full frames saved to: %s", FULL_FRAMES_DIR)


if __name__ == "__main__":
    main()
