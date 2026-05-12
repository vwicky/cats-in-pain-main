#!/usr/bin/env python3
"""
Build a new pipeline run folder with a random subset of an existing candidates.jsonl,
then start from tag_filter with --resume-run-dir pointing at that folder.

Example:
  python data_pipeline_v2/scripts/sample_candidates_for_new_run.py \\
    data_pipeline_v2/runs/OLD_RUN/stage_1_search/candidates.jsonl \\
    data_pipeline_v2/runs/pipeline_v2_run_20260413_120000 \\
    -n 300 --seed 43
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Sample N candidates into a new run's stage_1_search/")
    p.add_argument(
        "source_candidates",
        type=Path,
        help="Path to the full candidates.jsonl (large pool)",
    )
    p.add_argument(
        "new_run_dir",
        type=Path,
        help="New run directory (created), e.g. data_pipeline_v2/runs/pipeline_v2_run_YYYYMMDD_HHMMSS",
    )
    p.add_argument("-n", type=int, default=300, help="How many rows to sample (default 300)")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible sample; omit for a different sample each run",
    )
    args = p.parse_args()

    text = args.source_candidates.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {args.source_candidates}")

    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    n = min(args.n, len(rows))
    sample = rng.sample(rows, n)

    stage1 = args.new_run_dir / "stage_1_search"
    stage1.mkdir(parents=True, exist_ok=True)
    out = stage1 / "candidates.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample), encoding="utf-8")

    manifest = stage1 / "candidate_resample_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_candidates": str(args.source_candidates.resolve()),
                "rows_in_source": len(rows),
                "rows_sampled": n,
                "random_seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {n} candidates to {out}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
