#!/usr/bin/env python3
"""
Merge dataset/metadata_v2.jsonl and dataset/tiktok_metadata.jsonl into a single
JSONL with a ``source`` field on each record. Run from REPO_ROOT:

  python dataset_construction/00_merge_metadata.py

Output path is ``sources.metadata_merged`` in config.yaml (under manifests/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge metadata_v2 + tiktok_metadata JSONL.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write output.",
    )
    args = parser.parse_args()

    cfg = load_config()
    sources = cfg["sources"]
    repo = REPO_ROOT

    v2_path = repo / sources["metadata_v2"]
    tt_path = repo / sources["tiktok_metadata"]
    out_path = repo / sources["metadata_merged"]

    if not v2_path.is_file():
        print(f"Missing input: {v2_path}", file=sys.stderr)
        return 1
    if not tt_path.is_file():
        print(f"Missing input: {tt_path}", file=sys.stderr)
        return 1

    segments: list[tuple[Path, str]] = [
        (v2_path, "metadata_v2"),
        (tt_path, "tiktok_metadata"),
    ]

    total_records = 0
    total_snippets = 0
    counts: dict[str, int] = {"metadata_v2": 0, "tiktok_metadata": 0}
    snippet_counts: dict[str, int] = {"metadata_v2": 0, "tiktok_metadata": 0}

    out_lines: list[str] = []
    for path, source in segments:
        for rec in _iter_jsonl(path):
            rec = dict(rec)
            rec["source"] = source
            out_lines.append(json.dumps(rec, ensure_ascii=False))
            total_records += 1
            counts[source] = counts.get(source, 0) + 1
            sn = rec.get("snippets") or []
            n_sn = len(sn) if isinstance(sn, list) else 0
            total_snippets += n_sn
            snippet_counts[source] = snippet_counts.get(source, 0) + n_sn

    print(
        f"Merge: {counts['metadata_v2']} records ({snippet_counts['metadata_v2']} snippets) "
        f"from metadata_v2 + {counts['tiktok_metadata']} records "
        f"({snippet_counts['tiktok_metadata']} snippets) from tiktok_metadata "
        f"→ {total_records} records, {total_snippets} snippets total."
    )

    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
