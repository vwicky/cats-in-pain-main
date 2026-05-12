#!/usr/bin/env python3
"""Print top-K hparam cells by val_macro_f1_binary from hparams_sweep_runs.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO / "video" / "pose-models"
for _p in (REPO, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "jsonl",
        type=Path,
        nargs="?",
        default=None,
        help="Path to hparams_sweep_runs.jsonl (or pass --run-dir instead)",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Sweep run root; reads <run-dir>/hparams_sweep_runs.jsonl",
    )
    ap.add_argument("-k", "--top-k", type=int, default=10, help="Number of rows to show")
    ap.add_argument(
        "--wide",
        action="store_true",
        help="Include subdir, output_dir, and error columns",
    )
    args = ap.parse_args()
    p: Path
    if args.run_dir is not None:
        p = args.run_dir / "hparams_sweep_runs.jsonl"
    elif args.jsonl is not None:
        p = args.jsonl
    else:
        ap.print_help()
        print("error: pass JSONL path or --run-dir", file=sys.stderr)
        return 1
    p = p if p.is_absolute() else (REPO / p).resolve()
    if not p.is_file():
        print(f"error: not a file: {p}", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("no rows in jsonl")
        return 0
    df = pd.DataFrame(rows)
    if "val_macro_f1_binary" not in df.columns:
        print("error: no val_macro_f1_binary column", file=sys.stderr)
        return 1
    sort_col = "val_macro_f1_binary"
    d2 = df.copy()
    d2[sort_col] = pd.to_numeric(d2[sort_col], errors="coerce")
    d2 = d2.sort_values(sort_col, ascending=False).head(int(args.top_k))
    base_cols = [
        "val_macro_f1_binary",
        "run_index",
        "lr",
        "batch_size",
        "weight_decay",
        "focal_gamma",
        "scheduler",
        "best_epoch",
        "n_epochs_ran",
        "ce_ratio",
    ]
    cols = [c for c in base_cols if c in d2.columns]
    if args.wide:
        for c in ("subdir", "error", "output_dir"):
            if c in d2.columns:
                cols.append(c)
    out = d2[cols]
    # compact numeric display
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.6f}".format):
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
