#!/usr/bin/env python3
"""
Rebuild metadata_clean_01.jsonl after a static-detection run that removed too much.

Reads ``metadata_merged.jsonl`` and ``reports/static_scan_results.jsonl``, keeps every
snippet whose scan status is **ok** or **static** (treats “static” as normal content),
and drops only **missing** file entries.

Run from REPO_ROOT:
  python dataset_construction/restore_manifest_keep_static.py
  python dataset_construction/restore_manifest_keep_static.py --dry-run

Overwrites ``manifests/metadata_clean_01.jsonl`` (input for later pipeline steps).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild clean manifest keeping static-flagged snippets.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; do not write.")
    args = parser.parse_args()

    cfg = load_config()
    repo = REPO_ROOT
    merged_path = repo / cfg["sources"]["metadata_merged"]
    scan_path = repo / cfg["output"]["reports_dir"] / "static_scan_results.jsonl"
    out_path = repo / cfg["output"]["manifests_dir"] / "metadata_clean_01.jsonl"

    if not merged_path.is_file():
        print(f"Missing {merged_path}", file=sys.stderr)
        return 1
    if not scan_path.is_file():
        print(f"Missing {scan_path} — run 01_static_detection first.", file=sys.stderr)
        return 1

    result_by_id: dict[str, dict[str, Any]] = {}
    with open(scan_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            result_by_id[r["snippet_id"]] = r

    records_in: list[dict[str, Any]] = []
    with open(merged_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records_in.append(json.loads(line))

    input_records = len(records_in)
    input_snippets = sum(len(r.get("snippets") or []) for r in records_in)

    records_out: list[dict[str, Any]] = []
    removed_missing_total = 0

    for rec in records_in:
        new_snips: list[dict[str, Any]] = []
        rm = 0
        for sn in rec.get("snippets") or []:
            if not isinstance(sn, dict):
                continue
            sid = sn.get("id")
            if not isinstance(sid, str):
                continue
            info = result_by_id.get(sid)
            if info is None:
                new_snips.append(sn)
                continue
            if info["status"] == "missing":
                rm += 1
                continue
            # ok or static — keep
            new_snips.append(sn)

        removed_missing_total += rm
        if not new_snips:
            continue

        out_rec = dict(rec)
        out_rec["snippets"] = new_snips
        out_rec["removed_snippets"] = {"static": 0, "missing": rm}
        out_rec["cleaning_stage"] = "01_missing_only"
        records_out.append(out_rec)

    output_records = len(records_out)
    output_snippets = sum(len(r.get("snippets") or []) for r in records_out)
    retained_static = sum(
        1 for r in result_by_id.values() if r.get("status") == "static"
    )

    print("Restore manifest (keep static-flagged snippets, drop only missing)")
    print(f"  Input:  {merged_path}")
    print(f"  Scan:   {scan_path}")
    print(f"  Output: {out_path}")
    print(f"  Before: {input_records:,} records | {input_snippets:,} snippets")
    print(f"  After:  {output_records:,} records | {output_snippets:,} snippets")
    print(f"  Removed missing only: {removed_missing_total} snippets")
    print(f"  Static-flagged rows in scan (now kept): {retained_static}")

    if args.dry_run:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage": "01_missing_only",
        "note": "static_detection static flags ignored; only missing paths removed",
        "input_records": input_records,
        "output_records": output_records,
        "input_snippets": input_snippets,
        "output_snippets": output_snippets,
        "removed_static": 0,
        "removed_missing": removed_missing_total,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as mf:
        for rec in records_out:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mf.write("# " + json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
