#!/usr/bin/env python3
"""
Build **non-destructive** per-snippet VitPose QC labels for 17-kp training:

- See ``model_training_v2.vitpose_qc`` for metric definitions and default gates:
  - ``mean_conf >= 0.30`` (tunable via ``--min-mean-conf``)
  - ``y_range < 10`` (tunable via ``--max-y-range``) to drop coordinate-scale / crop

Writes:

- ``src/dataset_construction/reports/vitpose_qc_enrichment.jsonl`` (one object per
  successfully scored clip) with:

  - ``vitpose_qc_ok`` (bool) and ``vitpose_qc_exclusion_reasons`` (list, empty
    if ok)
  - all scalar metrics (``mean_conf``, ``y_range``, …) for analysis

- ``src/dataset_construction/reports/vitpose_qc_excluded_ids.txt`` — one ``snippet_id``
  per line that failed the gates (for inspection).

**Training:** set ``data.vitpose_qc.filter: true`` in ``model_training_v2/config*.yaml``
and point ``enrichment:`` to this file so :func:`data_loading.load_dataset` drops
excluded rows (see config comments).

Example:
  python model_training_v2/scripts/build_vitpose_qc_enrichment.py
  python model_training_v2/scripts/build_vitpose_qc_enrichment.py --min-mean-conf 0.28 --max-y-range 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO / "video" / "pose-models"
for _p in (REPO, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_loading import  load_pose_extraction_index
from vitpose_qc import  (
    apply_vitpose_qc_gates,
    compute_vitpose_clip_metrics,
    load_pose_and_mask,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "manifests" / "final_dataset_v2.jsonl",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO / "video" / "pose-models" / "config.yaml",
        help="For ``data.pose_extraction_index`` (same as training).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "vitpose_qc_enrichment.jsonl",
    )
    ap.add_argument(
        "--excluded-list",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "vitpose_qc_excluded_ids.txt",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "vitpose_qc_enrichment_summary.txt",
    )
    ap.add_argument(
        "--min-mean-conf",
        type=float,
        default=0.30,
        help="Require mean keypoint conf >= this (on real frames, channel 2).",
    )
    ap.add_argument(
        "--max-y-range",
        type=float,
        default=10.0,
        help="Require max(y)-min(y) over all real (x,y) to be < this (artifact screen).",
    )
    ap.add_argument("--visible-thr", type=float, default=0.30, help="Only affects frac_visible in the JSON.")
    ap.add_argument("--limit", type=int, default=0, help="Max manifest rows to score (0 = all suitable).")
    ap.add_argument(
        "--include-unsuitable",
        action="store_true",
        help="Process rows with suitable_for_training == false (default: only suitable).",
    )
    ap.add_argument("--label-field", default="final_label_5", help="Copied into each row for convenience.")
    args = ap.parse_args()

    man = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    out_jsonl = args.output if args.output.is_absolute() else REPO / args.output
    out_excl = args.excluded_list if args.excluded_list.is_absolute() else REPO / args.excluded_list
    out_sum = args.summary if args.summary.is_absolute() else REPO / args.summary
    for p in (out_jsonl, out_excl, out_sum):
        p.parent.mkdir(parents=True, exist_ok=True)

    cfgp = args.config if args.config.is_absolute() else REPO / args.config
    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    pose_index = load_pose_extraction_index(cfg) or None

    rows: list[dict] = []
    n_skip_no_file = 0
    n_skip_bad = 0
    t0 = time.time()
    n_lines = 0
    with open(man, encoding="utf-8") as f:
        for line in f:
            if args.limit and len(rows) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)
            if not args.include_unsuitable and not row.get("suitable_for_training"):
                continue
            sid = str(row.get("snippet_id", "")).strip()
            if not sid:
                continue
            pose, mask, used = load_pose_and_mask(REPO, row, pose_index)
            if used is None:
                n_skip_no_file += 1
                continue
            m = compute_vitpose_clip_metrics(pose, mask, visible_thr=args.visible_thr)
            if m is None:
                n_skip_bad += 1
                continue
            ok, reasons = apply_vitpose_qc_gates(
                m, min_mean_conf=args.min_mean_conf, max_y_range=args.max_y_range
            )
            rec = {
                "snippet_id": sid,
                "pose_path_scored": used,
                "mean_conf": m["mean_conf"],
                "frac_visible": m["frac_visible"],
                "zero_frame_rate": m["zero_frame_rate"],
                "xy_std": m["xy_std"],
                "head_conf": m["head_conf"],
                "body_conf": m["body_conf"],
                "y_range": m["y_range"],
                "vitpose_qc_ok": ok,
                "vitpose_qc_exclusion_reasons": reasons,
                "vitpose_qc_gates": {
                    "min_mean_conf": float(args.min_mean_conf),
                    "max_y_range": float(args.max_y_range),
                },
            }
            if args.label_field in row:
                rec[args.label_field] = row.get(args.label_field)
            rows.append(rec)

    n_ok = int(sum(1 for r in rows if r.get("vitpose_qc_ok")))
    n_ex = len(rows) - n_ok
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(out_excl, "w", encoding="utf-8") as f:
        for r in rows:
            if not r.get("vitpose_qc_ok"):
                f.write(str(r.get("snippet_id", "")) + "\n")
    # Summary
    lines = [
        f"Built: {datetime.now(timezone.utc).isoformat()}Z",
        f"Manifest: {man}",
        f"Gates: min_mean_conf={args.min_mean_conf}  max_y_range={args.max_y_range}  (real frames, channel 2 conf)",
        f"Manifest lines scanned: {n_lines}  (suitable only unless --include-unsuitable)",
        f"Clips successfully scored: {len(rows)}",
        f"  vitpose_qc_ok=True:  {n_ok}",
        f"  vitpose_qc_ok=False: {n_ex}",
        f"  skipped (no pose file): {n_skip_no_file}",
        f"  skipped (unusable array): {n_skip_bad}",
        f"Wrote: {out_jsonl}",
        f"Wrote: {out_excl}",
        f"Elapsed: {time.time() - t0:.1f}s",
    ]
    if rows and any(r.get(args.label_field) == "Paining" for r in rows):
        paining_total = int(sum(1 for r in rows if r.get(args.label_field) == "Paining"))
        paining_kept = int(
            sum(
                1
                for r in rows
                if r.get(args.label_field) == "Paining" and r.get("vitpose_qc_ok")
            )
        )
        lines.append(f"Paining: {paining_kept} / {paining_total} pass gates ({100.0 * paining_kept / max(paining_total, 1):.1f}%)")

    out_sum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
