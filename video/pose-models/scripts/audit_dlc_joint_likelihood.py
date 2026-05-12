#!/usr/bin/env python3
"""
Per-bodypart mean DeepLabCut likelihood across clips (manifest + HDF5 likelihood columns only).

Does **not** import PyTorch or the ST-GCN graph. Requires **PyTables** (``pip install tables``)
so ``pandas.read_hdf`` can open DeepLabCut HDF5 files.

Example:
  python model_training_v2/scripts/audit_dlc_joint_likelihood.py \\
    --config model_training_v2/config_stgcn_dlc.yaml --limit 800
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO / "video" / "pose-models"
for _p in (REPO, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_loading import  resolve_dlc_h5_path
from quadruped_skeleton_spec import  KEYPOINTS


def _bodypart_mean_lh_hdf5(path: Path) -> dict[str, float]:
    """Map bodypart name -> mean likelihood over time (one scalar per bodypart column)."""
    df = pd.read_hdf(path)
    mi = df.columns
    if not isinstance(mi, pd.MultiIndex):
        return {}
    out: dict[str, float] = {}
    for col in mi:
        if str(col[-1]).lower() != "likelihood":
            continue
        bp = str(col[-2]) if len(col) >= 2 else str(col[0])
        s = pd.to_numeric(df[col], errors="coerce")
        out[bp] = float(np.nanmean(s.to_numpy(dtype=np.float64)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "video" / "pose-models" / "config_stgcn_dlc.yaml")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Max clips to load (0 = all suitable rows)")
    ap.add_argument(
        "--include-unsuitable",
        action="store_true",
        help="Include rows where suitable_for_training is not true (default: suitable rows only).",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    man_rel = Path(str(args.manifest or data.get("manifest", "")).strip())
    manifest_path = man_rel if man_rel.is_absolute() else REPO / man_rel
    dlc_dir = Path(str(data.get("dlc_dir", "")).strip())
    if not dlc_dir.is_absolute():
        dlc_dir = REPO / dlc_dir
    suffix = str(data.get("dlc_h5_suffix", "")).strip()

    acc: dict[str, list[float]] = defaultdict(list)
    n_err = 0
    n_ok = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if args.limit and n_ok >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not args.include_unsuitable and row.get("suitable_for_training") is not True:
                continue
            sid = row.get("snippet_id")
            if not isinstance(sid, str) or not sid:
                continue
            try:
                p = resolve_dlc_h5_path(dlc_dir, sid, suffix)
            except ValueError:
                p = None
            if p is None or not p.is_file():
                continue
            try:
                bm = _bodypart_mean_lh_hdf5(p)
            except Exception:
                n_err += 1
                continue
            for bp, v in bm.items():
                acc[bp].append(v)
            n_ok += 1

    if not acc:
        print("No clips loaded — check manifest, dlc_dir, and dlc_h5_suffix in config.")
        return 1

    overall = {bp: float(np.mean(vals)) for bp, vals in acc.items()}
    # Sort low → high; KEYPOINTS only affects tie-break / known names first in display.
    spec_set = set(KEYPOINTS)
    pairs = sorted(overall.items(), key=lambda x: (x[1], 0 if x[0] in spec_set else 1, x[0]))

    print(f"Loaded {n_ok} clips ({n_err} HDF5 read errors). Mean likelihood per bodypart (low → high):")
    for name, v in pairs:
        print(f"  {name:28s}  {v:.4f}")
    print(
        "\nSuggested: bodyparts with dataset-wide mean < 0.3 → mask or drop from graph; "
        "torso parts low → retrain DLC or frontal bias in clips."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
