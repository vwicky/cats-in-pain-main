#!/usr/bin/env python3
"""
Append DeepLabCut likelihood QC fields to each manifest row (HDF5-only scan).

Writes a **new** JSONL; does not modify the input manifest (non-destructive).
Requires **PyTables** (``pip install tables``) for ``pandas.read_hdf``.

Example:
  python model_training_v2/scripts/enrich_manifest_dlc_quality.py \\
    --config model_training_v2/config_stgcn_dlc.yaml \\
    --out dataset_construction/manifests/final_dataset_v2_dlc_qc.jsonl

Fields added (when ``dlc_h5_path`` resolves):
  mean_dlc_likelihood, frac_high_conf_frames (mean joint LH > 0.5 per frame),
  n_visible_joints (count of joints with temporal mean LH > 0.4),
  has_occlusion (any frame with >30% of joints below 0.3 LH),
  clip_duration_frames, dlc_qc_enriched_at (ISO timestamp).

Optional mirrors of existing GPT columns: multiple_cats_visible, label_confidence
(from ``gpt_n_cats_visible`` / ``audio_confidence`` when present).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO / "video" / "pose-models"
for _p in (REPO, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_loading import  resolve_dlc_h5_path
from deeplabcut_pose_io import  clip_quality_stats_from_hdf5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "video" / "pose-models" / "config_stgcn_dlc.yaml")
    ap.add_argument("--manifest", type=Path, default=None, help="Override cfg data.manifest (repo-relative)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path (new file)")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows (0 = all)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    man_rel = Path(str(args.manifest or data.get("manifest", "")).strip())
    if not str(man_rel):
        ap.error("manifest path missing: set --manifest or cfg data.manifest")
    manifest_path = man_rel if man_rel.is_absolute() else REPO / man_rel

    dlc_dir = Path(str(data.get("dlc_dir", "")).strip())
    if not dlc_dir.is_absolute():
        dlc_dir = REPO / dlc_dir
    suffix = str(data.get("dlc_h5_suffix", "")).strip()

    out_path = args.out if args.out.is_absolute() else REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_in = n_out = n_skip = 0
    with open(manifest_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.limit and n_out >= args.limit:
                break
            row = json.loads(line)
            sid = row.get("snippet_id")
            h5_rel: str | None = None
            if isinstance(sid, str) and sid:
                try:
                    p = resolve_dlc_h5_path(dlc_dir, sid, suffix)
                except ValueError:
                    p = None
                if p is not None and p.is_file():
                    try:
                        h5_rel = str(p.resolve().relative_to(REPO.resolve())).replace("\\", "/")
                    except ValueError:
                        h5_rel = str(p).replace("\\", "/")

            if h5_rel:
                st = clip_quality_stats_from_hdf5(
                    REPO / h5_rel,
                    clip_mean_likelihood_threshold=0.0,
                    per_frame_threshold=1.0,
                )
                row["mean_dlc_likelihood"] = float(st["mean_dlc_likelihood"])
                row["frac_high_conf_frames"] = float(st["frac_high_conf_frames_0p5"])
                row["n_visible_joints"] = int(st["n_visible_joints_0p4"])
                row["has_occlusion"] = bool(st["has_occlusion"])
                row["clip_duration_frames"] = int(st["clip_duration_frames"])
                row["dlc_qc_enriched_at"] = ts
                gpt_n = row.get("gpt_n_cats_visible")
                if gpt_n is not None:
                    try:
                        row["multiple_cats_visible"] = int(gpt_n) > 1
                    except (TypeError, ValueError):
                        row["multiple_cats_visible"] = None
                ac = row.get("audio_confidence")
                if ac is not None:
                    try:
                        row["label_confidence"] = float(ac)
                    except (TypeError, ValueError):
                        row["label_confidence"] = None
            else:
                n_skip += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"Wrote {n_out} lines to {out_path.relative_to(REPO) if out_path.is_relative_to(REPO) else out_path} "
        f"(input lines {n_in}; rows without resolvable HDF5: {n_skip})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
