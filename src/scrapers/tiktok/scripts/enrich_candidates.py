#!/usr/bin/env python3
"""Re-run the metadata enrich stage only (yt-dlp extract_info per URL).

Example (from repo root):

  PYTHONPATH=tiktok_pipeline python tiktok_pipeline/scripts/enrich_candidates.py \\
    --config tiktok_pipeline/config/pipeline.yaml \\
    --resume-run-dir tiktok_pipeline/runs/tiktok_pipeline_run_20260410_215605
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TIKTOK_ROOT = Path(__file__).resolve().parent.parent
if str(_TIKTOK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TIKTOK_ROOT))

from src.metadata_enrich import run_metadata_enrich
from src.utils import load_config, load_jsonl, project_root, resolve_path, save_jsonl, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="TikTok pipeline: metadata enrich only (yt-dlp)")
    parser.add_argument("--config", default="src/scrapers/tiktok/config/pipeline.yaml")
    parser.add_argument("--resume-run-dir", required=True, help="Existing run directory with stage_1_search/candidates.jsonl")
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(resolve_path(root, args.config))
    run_dir = Path(args.resume_run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    src_path = run_dir / "stage_1_search" / "candidates.jsonl"
    if not src_path.is_file():
        print(f"ERROR: missing {src_path}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logger(run_dir)
    candidates = load_jsonl(src_path)
    enriched, stats = run_metadata_enrich(candidates, cfg, logger, run_dir)
    if stats.get("skipped"):
        print("metadata_enrich.enabled is false; nothing to do.")
        sys.exit(0)
    out = run_dir / "stage_1_enrich" / "candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(enriched, out, mode="w")
    print(f"Wrote {out} ({len(enriched)} rows). Summary: {stats}")


if __name__ == "__main__":
    main()
