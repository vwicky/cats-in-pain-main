#!/usr/bin/env python3
"""
Build a non-destructive per-snippet DeepLabCut QC enrichment JSONL for pairwise training.

Analogous to ``build_vitpose_qc_enrichment.py`` for ViTPose poses.

Gates applied (tunable via CLI):
  - ``mean_dlc_likelihood >= --min-mean-likelihood`` (default 0.35)
    Mean likelihood over all T×J in the raw clip (before temporal padding).
  - ``fraction_high_conf_frames >= --min-frac-high-conf`` (default 0.25)
    Fraction of frames where mean_j(likelihood) >= per-frame threshold
    (defaults to ``--min-mean-likelihood`` if not set separately).

Writes:
  - ``src/dataset_construction/reports/dlc_qc_enrichment.jsonl`` — one object per
    successfully scored clip with:
      ``dlc_qc_ok`` (bool), ``dlc_qc_exclusion_reasons`` (list, empty if ok),
      ``mean_dlc_likelihood``, ``frac_high_conf_frames``, ``frac_high_conf_frames_0p5``,
      ``n_visible_joints_0p4``, ``has_occlusion``, ``clip_duration_frames``.
  - ``src/dataset_construction/reports/dlc_qc_excluded_ids.txt`` — one snippet_id per
    line that failed the gates (for inspection).
  - ``src/dataset_construction/reports/dlc_qc_enrichment_summary.txt`` — human-readable
    summary of how many clips passed/failed.

**Training:** set ``data.dlc_qc.filter: true`` in ``config_stgcn_dlc.yaml`` and
point ``enrichment:`` to the JSONL so ``data_loading.load_dataset_for_deeplabcut``
drops excluded rows before the pair split (see config comments).

Example:
  python model_training_v2/scripts/build_dlc_qc_enrichment.py
  python model_training_v2/scripts/build_dlc_qc_enrichment.py \\
    --min-mean-likelihood 0.30 --min-frac-high-conf 0.20
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

from data_loading import  resolve_dlc_h5_path
from deeplabcut_pose_io import  clip_quality_stats_from_hdf5


def _apply_dlc_qc_gates(
    stats: dict,
    *,
    min_mean_likelihood: float,
    min_frac_high_conf: float,
) -> tuple[bool, list[str]]:
    """Return (ok, exclusion_reasons). Reasons are stable machine strings."""
    reasons: list[str] = []
    m = float(stats.get("mean_dlc_likelihood", 0.0))
    f = float(stats.get("fraction_high_conf_frames", 0.0))
    if m < min_mean_likelihood:
        reasons.append(f"mean_lh_lt_{min_mean_likelihood:.2f}")
    if f < min_frac_high_conf:
        reasons.append(f"frac_high_conf_lt_{min_frac_high_conf:.2f}")
    return (len(reasons) == 0, reasons)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO / "video" / "pose-models" / "config_stgcn_dlc.yaml",
        help="ST-GCN DLC config (for data.manifest, data.dlc_dir, data.dlc_h5_suffix).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override cfg data.manifest (repo-relative or absolute).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "dlc_qc_enrichment.jsonl",
    )
    ap.add_argument(
        "--excluded-list",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "dlc_qc_excluded_ids.txt",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "dlc_qc_enrichment_summary.txt",
    )
    ap.add_argument(
        "--min-mean-likelihood",
        type=float,
        default=0.35,
        help="Require mean DLC likelihood (all T×J) >= this threshold (default: 0.35).",
    )
    ap.add_argument(
        "--min-frac-high-conf",
        type=float,
        default=0.25,
        help="Require fraction of frames with mean_j likelihood >= per-frame-threshold >= this (default: 0.25).",
    )
    ap.add_argument(
        "--per-frame-threshold",
        type=float,
        default=None,
        help="Per-frame mean-joint likelihood threshold for counting 'high-conf' frames. "
             "Defaults to --min-mean-likelihood if unset.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N suitable rows (0 = all).",
    )
    ap.add_argument(
        "--include-unsuitable",
        action="store_true",
        help="Score rows with suitable_for_training == false (default: only suitable rows).",
    )
    ap.add_argument(
        "--label-field",
        default="audio_label_10",
        help="Label field name to copy into each scored record for convenience.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data") or {}

    man_rel = Path(str(args.manifest or data_cfg.get("manifest", "")).strip())
    if not str(man_rel):
        ap.error("manifest path missing — set --manifest or cfg data.manifest")
    manifest_path = man_rel if man_rel.is_absolute() else REPO / man_rel

    dlc_dir_raw = Path(str(data_cfg.get("dlc_dir", "")).strip())
    dlc_dir = dlc_dir_raw if dlc_dir_raw.is_absolute() else REPO / dlc_dir_raw
    h5_suffix = str(data_cfg.get("dlc_h5_suffix", "")).strip()

    per_frame_thr = args.per_frame_threshold if args.per_frame_threshold is not None else args.min_mean_likelihood

    out_jsonl = args.output if args.output.is_absolute() else REPO / args.output
    out_excl = args.excluded_list if args.excluded_list.is_absolute() else REPO / args.excluded_list
    out_sum = args.summary if args.summary.is_absolute() else REPO / args.summary
    for p in (out_jsonl, out_excl, out_sum):
        p.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_lines = 0
    n_skip_no_h5 = 0
    n_skip_error = 0
    records: list[dict] = []

    print(
        f"Scanning DLC HDF5 files from {dlc_dir} ...\n"
        f"  Gates: mean_dlc_likelihood >= {args.min_mean_likelihood}  |  "
        f"frac_high_conf_frames >= {args.min_frac_high_conf}  "
        f"(per-frame threshold: {per_frame_thr:.2f})"
    )

    with open(manifest_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            if args.limit and len(records) + n_skip_no_h5 + n_skip_error >= args.limit:
                break

            row = json.loads(line)
            if not args.include_unsuitable and not row.get("suitable_for_training"):
                continue
            sid = str(row.get("snippet_id", "")).strip()
            if not sid:
                continue

            try:
                h5_path = resolve_dlc_h5_path(dlc_dir, sid, h5_suffix)
            except (ValueError, Exception):
                h5_path = None

            if h5_path is None or not h5_path.is_file():
                n_skip_no_h5 += 1
                continue

            try:
                stats = clip_quality_stats_from_hdf5(
                    h5_path,
                    clip_mean_likelihood_threshold=args.min_mean_likelihood,
                    per_frame_threshold=per_frame_thr,
                )
            except Exception as exc:
                print(f"  WARNING: failed to read {h5_path.name} for {sid!r}: {exc}", file=sys.stderr)
                n_skip_error += 1
                continue

            ok, reasons = _apply_dlc_qc_gates(
                stats,
                min_mean_likelihood=args.min_mean_likelihood,
                min_frac_high_conf=args.min_frac_high_conf,
            )

            rec: dict = {
                "snippet_id": sid,
                "dlc_qc_ok": ok,
                "dlc_qc_exclusion_reasons": reasons,
                "mean_dlc_likelihood": float(stats["mean_dlc_likelihood"]),
                "frac_high_conf_frames": float(stats["fraction_high_conf_frames"]),
                "frac_high_conf_frames_0p5": float(stats["frac_high_conf_frames_0p5"]),
                "n_visible_joints_0p4": int(stats["n_visible_joints_0p4"]),
                "has_occlusion": bool(stats["has_occlusion"]),
                "clip_duration_frames": int(stats["clip_duration_frames"]),
                "dlc_qc_gates": {
                    "min_mean_likelihood": float(args.min_mean_likelihood),
                    "min_frac_high_conf": float(args.min_frac_high_conf),
                    "per_frame_threshold": float(per_frame_thr),
                },
            }
            if args.label_field in row:
                rec[args.label_field] = row.get(args.label_field)

            records.append(rec)

    n_ok = int(sum(1 for r in records if r.get("dlc_qc_ok")))
    n_ex = len(records) - n_ok

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    with open(out_excl, "w", encoding="utf-8") as f:
        for r in records:
            if not r.get("dlc_qc_ok"):
                f.write(str(r.get("snippet_id", "")) + "\n")

    elapsed = time.time() - t0
    summary_lines = [
        f"Built: {datetime.now(timezone.utc).isoformat()}Z",
        f"Manifest: {manifest_path}",
        f"DLC dir: {dlc_dir}",
        f"Gates: min_mean_likelihood={args.min_mean_likelihood}  "
        f"min_frac_high_conf={args.min_frac_high_conf}  "
        f"per_frame_threshold={per_frame_thr:.2f}",
        f"Manifest lines scanned: {n_lines}  (suitable only unless --include-unsuitable)",
        f"Clips successfully scored: {len(records)}",
        f"  dlc_qc_ok=True:  {n_ok}",
        f"  dlc_qc_ok=False: {n_ex}",
        f"  skipped (no HDF5 found): {n_skip_no_h5}",
        f"  skipped (HDF5 read error): {n_skip_error}",
        f"Wrote: {out_jsonl}",
        f"Wrote: {out_excl}",
        f"Elapsed: {elapsed:.1f}s",
    ]

    if records and any(r.get(args.label_field) == "Paining" for r in records):
        paining_total = int(sum(1 for r in records if r.get(args.label_field) == "Paining"))
        paining_kept = int(
            sum(1 for r in records if r.get(args.label_field) == "Paining" and r.get("dlc_qc_ok"))
        )
        summary_lines.append(
            f"Paining: {paining_kept} / {paining_total} pass gates "
            f"({100.0 * paining_kept / max(paining_total, 1):.1f}%)"
        )

    out_sum.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
