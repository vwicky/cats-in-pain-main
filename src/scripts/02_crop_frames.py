"""Step 2: Crop full-resolution frames to padded YOLO bboxes (fast, re-runnable).

Reads:   dataset_construction/yolo_bboxes.json  (produced by 01_extract_and_detect.py)
         cnn_finetune/config/v2.yaml  (for bbox_padding)

Writes:  dataset/cropped_frames_p{padding_pct}/{clip_id}/frame_{01-15}.jpg
         e.g. dataset/cropped_frames_p20/ for bbox_padding: 0.20
              dataset/cropped_frames_p10/ for bbox_padding: 0.10

Fallback: if a frame has no cat detection, the full image is written unchanged.

Re-run safety:
  - Pass --force to re-crop everything.
  - Otherwise, existing JPEG files are skipped.

Usage (from repo root):
    # Default padding from config
    python scripts/02_crop_frames.py --config cnn_finetune/config/v2.yaml

    # Override padding to try a different value (no need to re-run step 1)
    python scripts/02_crop_frames.py --config cnn_finetune/config/v2.yaml --bbox-padding 0.10

    # Force re-crop everything
    python scripts/02_crop_frames.py --config cnn_finetune/config/v2.yaml --force
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BBOXES_PATH = PROJECT_ROOT / "src" / "src" / "dataset_construction" / "yolo_bboxes.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crop_frames")


# ── Geometry helpers ─────────────────────────────────────────────────────────


def _expand_xyxy_pad_clamp(
    x0: float, y0: float, x1: float, y1: float,
    W: int, H: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    """Expand bbox by pad_frac fraction of its own size, then clamp to image."""
    bw, bh = x1 - x0, y1 - y0
    x0 -= pad_frac * bw
    x1 += pad_frac * bw
    y0 -= pad_frac * bh
    y1 += pad_frac * bh
    ix0 = int(math.floor(max(0.0, x0)))
    iy0 = int(math.floor(max(0.0, y0)))
    ix1 = int(math.ceil(min(float(W), x1)))
    iy1 = int(math.ceil(min(float(H), y1)))
    return ix0, iy0, ix1, iy1


# ── Per-clip crop ────────────────────────────────────────────────────────────


def crop_clip(
    clip_id: str,
    frame_entries: list[dict],
    out_dir: Path,
    pad_frac: float,
    force: bool,
) -> tuple[int, int]:
    """Crop all frames for one clip.  Returns (n_cropped, n_skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cropped = 0
    n_skipped = 0

    for i, entry in enumerate(frame_entries, start=1):
        out_path = out_dir / f"frame_{i:02d}.jpg"

        if not force and out_path.exists():
            n_skipped += 1
            continue

        src_path = PROJECT_ROOT / entry["path"]
        if not src_path.exists():
            # Frame was never saved (e.g. unreadable video); write blank
            Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)).save(str(out_path))
            n_cropped += 1
            continue

        img = Image.open(str(src_path)).convert("RGB")
        W, H = img.size  # PIL: (width, height)

        if entry.get("detected") and entry.get("bbox_xyxy"):
            bx0, by0, bx1, by1 = entry["bbox_xyxy"]
            cx0, cy0, cx1, cy1 = _expand_xyxy_pad_clamp(
                bx0, by0, bx1, by1, W, H, pad_frac,
            )
            if cx1 > cx0 and cy1 > cy0:
                img = img.crop((cx0, cy0, cx1, cy1))
            # else: degenerate box → keep full frame

        img.save(str(out_path), quality=95)
        n_cropped += 1

    return n_cropped, n_skipped


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Crop full frames to padded YOLO bboxes.")
    parser.add_argument("--config", default="video/cnn-finetuning/config/v2.yaml")
    parser.add_argument("--bbox-padding", type=float, default=None,
                        help="Override bbox_padding from config (e.g. 0.10)")
    parser.add_argument("--force", action="store_true",
                        help="Re-crop even if output file already exists")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Parallel worker threads (default: all CPUs)")
    args = parser.parse_args()

    import yaml

    cfg_path = PROJECT_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    pad_frac = args.bbox_padding if args.bbox_padding is not None else float(cfg.get("bbox_padding", 0.20))
    pad_pct = int(round(pad_frac * 100))
    out_root = PROJECT_ROOT / "dataset" / f"cropped_frames_p{pad_pct}"

    logger.info("bbox_padding = %.2f  →  output dir: %s", pad_frac, out_root)

    if not BBOXES_PATH.exists():
        logger.error("yolo_bboxes.json not found at %s — run 01_extract_and_detect.py first", BBOXES_PATH)
        sys.exit(1)

    bboxes: dict = json.loads(BBOXES_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded bbox data for %d clips", len(bboxes))

    out_root.mkdir(parents=True, exist_ok=True)

    total_cropped = 0
    total_skipped = 0

    def _process(item: tuple[str, dict]) -> tuple[int, int]:
        clip_id, data = item
        out_dir = out_root / clip_id
        return crop_clip(clip_id, data["frames"], out_dir, pad_frac, args.force)

    items = list(bboxes.items())
    workers = min(args.workers, len(items))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, item): item[0] for item in items}
        for fut in tqdm(as_completed(futures), total=len(items), desc="crop"):
            nc, ns = fut.result()
            total_cropped += nc
            total_skipped += ns

    logger.info(
        "Done. %d clips | %d frames cropped | %d frames skipped (already existed)",
        len(bboxes), total_cropped, total_skipped,
    )
    logger.info("Cropped frames saved to: %s", out_root)
    logger.info(
        "To use in training, add this to your config:\n"
        "  precomputed_frames_dir: \"data/dataset/cropped_frames_p%d\"", pad_pct,
    )


if __name__ == "__main__":
    main()
