#!/usr/bin/env python3
"""Aggregate ST-GCN pairwise run_summary.json files into one CSV.

Walks a p4_pairwise_ensemble_bundle_* directory, finds every
**/training/run_summary.json, and emits one row per unique class pair
(key = sorted(class_names) joined by '|'). When multiple trainings exist for the
same pair (e.g. hyperparameter grids), keeps the row with highest
metrics.val_macro_f1_binary.

Usage:
  python src/dataset_construction/reports/export_pairwise_run_summary_metrics.py \\
    --bundle runs/pose-models/p4_pairwise_ensemble_bundle_20260428 \\
    --output src/dataset_construction/reports/pairwise_run_summary_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def pair_key(class_names: list[str] | None) -> str | None:
    if not class_names or len(class_names) != 2:
        return None
    a, b = class_names[0], class_names[1]
    return "|".join(sorted((a.strip(), b.strip())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to p4_pairwise_ensemble_bundle_* root.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    args = p.parse_args()
    bundle: Path = args.bundle.resolve()
    if not bundle.is_dir():
        raise SystemExit(f"bundle not found or not a directory: {bundle}")

    best: dict[str, dict] = {}
    for run_summary in bundle.rglob("training/run_summary.json"):
        try:
            data = json.loads(run_summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        key = pair_key(data.get("class_names"))
        if not key:
            continue
        metrics = data.get("metrics") or {}
        f1 = float(
            metrics.get("val_macro_f1_binary", metrics.get("val_macro_f1_5", 0.0))
        )
        auc = float(metrics.get("val_auc_roc", 0.0))
        row = {
            "pair": key,
            "pain_inclusive": "Paining" in key.split("|"),
            "n_train": data.get("n_train", ""),
            "n_val": data.get("n_val", ""),
            "best_epoch": data.get("best_epoch", ""),
            "val_macro_f1_binary": round(f1, 6),
            "val_auc_roc": round(auc, 6),
            "run_summary_rel": str(run_summary.relative_to(bundle)),
        }
        prev = best.get(key)
        if prev is None or f1 > float(prev["val_macro_f1_binary"]):
            best[key] = row

    rows = sorted(best.values(), key=lambda r: (not r["pain_inclusive"], r["pair"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "pain_inclusive",
        "n_train",
        "n_val",
        "best_epoch",
        "val_macro_f1_binary",
        "val_auc_roc",
        "run_summary_rel",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
