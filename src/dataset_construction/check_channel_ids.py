#!/usr/bin/env python3
"""
Audit metadata JSONL sources for channel/uploader fields and cross-reference
metadata_clean_03 snippets. Optionally write an enriched manifest; always
write missing-channel video list for GPT follow-up.

  python dataset_construction/check_channel_ids.py
  python dataset_construction/check_channel_ids.py --verbose
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

CHANNEL_KEYS = (
    "channel",
    "channel_id",
    "uploader",
    "uploader_id",
    "author",
    "author_id",
    "creator",
    "creator_id",
    # TikTok pipeline search/enrich rows (see tiktok_pipeline/.../stage_1_enrich/candidates.jsonl)
    "channel_title",
    "channel_url",
    "uploader_url",
)

# Prefer stable platform ids (e.g. YouTube UC…) over @handles when both exist.
BEST_ID_PRIORITY = (
    "channel_id",
    "uploader_id",
    "author_id",
    "creator_id",
    "channel_title",
    "channel",
    "uploader",
)

# Relative paths under REPO_ROOT (order used for lookup reporting and merge tie-break).
# Include yt-dlp–enriched manifest so coverage reflects channel_id/uploader_id after
# enrich_channels_ytdlp.py (base metadata_clean_03 is excluded from lookups below).
SOURCE_FILES: list[tuple[str, str]] = [
    ("src/dataset_construction/manifests/metadata_clean_03_ytdlp.jsonl", "metadata_clean_03_ytdlp"),
    ("data/dataset/metadata_v2.jsonl", "metadata_v2"),
    ("data/dataset/tiktok_metadata.jsonl", "tiktok_metadata"),
    ("src/dataset_construction/manifests/metadata_merged.jsonl", "metadata_merged"),
    ("src/dataset_construction/manifests/metadata_clean_03.jsonl", "metadata_clean_03"),
    ("logs/v2_pipeline_log.jsonl", "v2_pipeline_log"),
    ("logs/streaming_pipeline_log.jsonl", "streaming_pipeline_log"),
    ("data/dataset/metadata.jsonl", "metadata"),
]

CLEAN03_PATH = "src/dataset_construction/manifests/metadata_clean_03.jsonl"
ENRICHED_OUT = "src/dataset_construction/manifests/metadata_clean_03_channels.jsonl"
MISSING_OUT = "src/dataset_construction/reports/missing_channel_ids.jsonl"

# TikTok yt-dlp dump often fails; stage-1 enrich candidates carry channel_title + webpage_url.
TIKTOK_STAGE1_CANDIDATES_GLOB = "src/scrapers/tiktok/runs/**/stage_1_enrich/candidates.jsonl"


def is_nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    if bad:
        print(f"  [warn] {path.relative_to(REPO_ROOT)}: skipped {bad} malformed lines", file=sys.stderr)
    return rows


def best_channel_from_record(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (value, key_used) from CHANNEL_KEYS priority, or (None, None)."""
    for k in BEST_ID_PRIORITY:
        if k not in CHANNEL_KEYS:
            continue
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), k
    return None, None


def infer_platform(record: dict[str, Any]) -> str:
    """youtube | tiktok | dailymotion (lowercase). Duplicated from 02_gpt_description logic."""
    src = (record.get("source") or "").strip().lower()
    if src == "tiktok_metadata":
        return "tiktok"
    if src == "metadata_v2":
        pass
    for key in ("platform", "source_platform"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            pl = v.strip().lower()
            if "tiktok" in pl:
                return "tiktok"
            if "dailymotion" in pl or "daily_motion" in pl:
                return "dailymotion"
            if "youtube" in pl:
                return "youtube"

    vid = str(record.get("video_id", ""))
    if vid.isdigit() and len(vid) >= 15:
        return "tiktok"

    return "youtube"


def video_url(platform: str, video_id: str) -> str:
    if platform == "tiktok":
        return f"https://www.tiktok.com/video/{video_id}"
    if platform == "dailymotion":
        return f"https://www.dailymotion.com/video/{video_id}"
    return f"https://www.youtube.com/watch?v={video_id}"


def is_stable_uc(value: str) -> bool:
    return value.startswith("UC") and len(value) >= 22


PRIORITY_RANK = {k: i for i, k in enumerate(BEST_ID_PRIORITY)}


def merge_channel_per_video(
    lookup_sources: list[tuple[str, str, dict[str, tuple[str, str]]]],
) -> dict[str, tuple[str, str, str]]:
    """
    video_id -> (best_value, key_used, source_label).
    Best = lowest BEST_ID_PRIORITY index; tie-break: first source in lookup_sources order.
    """
    all_vids: set[str] = set()
    for _lbl, _rel, lu in lookup_sources:
        all_vids.update(lu.keys())
    merged: dict[str, tuple[str, str, str]] = {}
    for vid in all_vids:
        best_rank = 999
        best_si = 999
        best_val = ""
        best_key = ""
        for si, (_lbl, _rel, lu) in enumerate(lookup_sources):
            if vid not in lu:
                continue
            val, key_used = lu[vid]
            r = PRIORITY_RANK.get(key_used, 99)
            if r < best_rank or (r == best_rank and si < best_si):
                best_rank = r
                best_si = si
                best_val = val
                best_key = key_used
        if best_rank < 999:
            lbl = lookup_sources[best_si][0]
            merged[vid] = (best_val, best_key, lbl)
    return merged


def step1_scan_file(rel: str, path: Path, verbose: bool) -> list[dict[str, Any]]:
    print(f"=== {rel} ===")
    if not path.is_file():
        print("  (file not found — skipped)\n")
        return []

    records = load_jsonl(path)
    n = len(records)
    print(f"  Total records: {n}")

    all_keys: set[str] = set()
    present_count: dict[str, int] = {k: 0 for k in CHANNEL_KEYS}
    nonempty_count: dict[str, int] = {k: 0 for k in CHANNEL_KEYS}
    examples: dict[str, str] = {}

    for rec in records:
        for k in rec:
            all_keys.add(k)
        for k in CHANNEL_KEYS:
            if k in rec:
                present_count[k] += 1
                v = rec[k]
                if is_nonempty(v) and isinstance(v, str):
                    nonempty_count[k] += 1
                    if k not in examples:
                        examples[k] = v.strip()[:120]

    print(f"  All keys present: {sorted(all_keys)}")
    print("")
    print("  Channel-related keys found:")
    any_ch = False
    for k in CHANNEL_KEYS:
        if present_count[k] == 0:
            continue
        any_ch = True
        ex = examples.get(k, "")
        pct = 100.0 * nonempty_count[k] / n if n else 0.0
        print(f"    {k}: present in {present_count[k]}/{n} records | example: {json.dumps(ex)}")
        print(f"      non-null non-empty rate: {pct:.1f}%")
    if not any_ch:
        print("    (none of the channel-related keys appear in this file)")
    print("")

    shown = 0
    print("  First 3 records (channel-related fields only):")
    for rec in records[:3]:
        slim = {k: rec[k] for k in CHANNEL_KEYS if k in rec}
        print(json.dumps(slim, indent=2, ensure_ascii=False))
        shown += 1
    if shown == 0:
        print("    (no records)")
    print("")

    if verbose and records:
        print("  Verbose: first 5 full records (pretty-printed):")
        for i, rec in enumerate(records[:5], start=1):
            print(f"  --- record {i} ---")
            print(json.dumps(rec, indent=2, ensure_ascii=False)[:8000])
            print("")

    return records


def build_lookup(records: list[dict[str, Any]], label: str) -> dict[str, tuple[str, str]]:
    """video_id -> (best_value, provenance_key)."""
    out: dict[str, tuple[str, str]] = {}
    for rec in records:
        vid = rec.get("video_id")
        if vid is None or vid == "":
            continue
        vid_s = str(vid)
        val, key_used = best_channel_from_record(rec)
        if val is not None:
            out[vid_s] = (val, key_used or "")
    return out


def discover_tiktok_stage1_candidate_files() -> list[Path]:
    paths = sorted(
        (p for p in REPO_ROOT.glob(TIKTOK_STAGE1_CANDIDATES_GLOB) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return paths


def build_merged_tiktok_stage1_lookup(paths: list[Path]) -> dict[str, tuple[str, str]]:
    """Merge lookups from all runs; later files (newer mtime) overwrite same video_id."""
    merged: dict[str, tuple[str, str]] = {}
    for p in paths:
        merged.update(build_lookup(load_jsonl(p), "tiktok_stage1"))
    return merged


def step1_scan_tiktok_candidates_merged(verbose: bool) -> None:
    paths = discover_tiktok_stage1_candidate_files()
    print("=== TikTok stage-1 enrich candidates (merged runs) ===")
    if not paths:
        print(f"  (no files matched {TIKTOK_STAGE1_CANDIDATES_GLOB} — skipped)\n")
        return
    print(f"  Matched {len(paths)} file(s) under tiktok_pipeline/runs/")
    all_recs: list[dict[str, Any]] = []
    for p in paths:
        all_recs.extend(load_jsonl(p))
    n = len(all_recs)
    print(f"  Total JSONL rows (duplicates across runs possible): {n}")
    lu = build_lookup(all_recs, "merged")
    print(f"  Distinct video_ids with a channel-related field: {len(lu)}")
    nonempty_ct = sum(1 for rec in all_recs if best_channel_from_record(rec)[0] is not None)
    print(f"  Rows with any channel key (non-merged count): {nonempty_ct}/{n}")
    print("  First row (channel-related fields only):")
    if all_recs:
        rec0 = all_recs[0]
        slim = {k: rec0[k] for k in CHANNEL_KEYS if k in rec0}
        print(json.dumps(slim, indent=2, ensure_ascii=False))
    else:
        print("    (no records)")
    print("")
    if verbose and all_recs:
        print("  Verbose: first 3 full records from first file:")
        first_file = load_jsonl(paths[0])
        for i, rec in enumerate(first_file[:3], start=1):
            print(f"  --- file {paths[0].relative_to(REPO_ROOT)} record {i} ---")
            print(json.dumps(rec, indent=2, ensure_ascii=False)[:8000])
            print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit channel/uploader fields across metadata sources.")
    parser.add_argument("--verbose", action="store_true", help="Print first 5 full records per source in Step 1.")
    args = parser.parse_args()

    clean_path = REPO_ROOT / CLEAN03_PATH
    if not clean_path.is_file():
        print(f"Missing required manifest: {clean_path}", file=sys.stderr)
        return 1

    # Step 1
    all_records_by_label: dict[str, list[dict[str, Any]]] = {}
    for rel, label in SOURCE_FILES:
        p = REPO_ROOT / rel
        recs = step1_scan_file(rel, p, args.verbose)
        if recs:
            all_records_by_label[label] = recs

    step1_scan_tiktok_candidates_merged(args.verbose)

    # Step 2 — lookups (exclude metadata_clean_03 from resolution to avoid circularity)
    lookup_sources: list[tuple[str, str, dict[str, tuple[str, str]]]] = []
    title_from_v2: dict[str, str] = {}

    for rel, label in SOURCE_FILES:
        if label == "metadata_clean_03":
            continue
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        recs = all_records_by_label.get(label) or load_jsonl(p)
        lu = build_lookup(recs, label)
        lookup_sources.append((label, rel, lu))
        if label == "v2_pipeline_log":
            for rec in recs:
                vid = str(rec.get("video_id") or "")
                t = rec.get("title")
                if vid and isinstance(t, str) and t.strip():
                    title_from_v2[vid] = t.strip()

    tt_paths = discover_tiktok_stage1_candidate_files()
    if tt_paths:
        tt_lu = build_merged_tiktok_stage1_lookup(tt_paths)
        tt_rel = "src/scrapers/tiktok/runs/*/stage_1_enrich/candidates.jsonl (merged)"
        lookup_sources.insert(1, ("tiktok_stage1_enrich", tt_rel, tt_lu))

    merged_video = merge_channel_per_video(lookup_sources)

    clean_records = load_jsonl(clean_path)
    snippets_total = 0
    per_source_hits: dict[str, int] = defaultdict(int)
    union_snippets = 0
    stable_uc = 0
    display_only = 0
    video_snippet_count: dict[str, int] = defaultdict(int)
    video_record: dict[str, dict[str, Any]] = {}
    snippet_rows: list[tuple[str, str]] = []

    for rec in clean_records:
        vid = str(rec.get("video_id") or "")
        if not vid:
            continue
        video_record[vid] = rec
        snips = rec.get("snippets") or []
        if not isinstance(snips, list):
            continue
        for sn in snips:
            if not isinstance(sn, dict):
                continue
            sid = sn.get("id")
            if not isinstance(sid, str):
                continue
            snippets_total += 1
            video_snippet_count[vid] += 1
            snippet_rows.append((vid, sid))

    for vid, _sid in snippet_rows:
        for lbl, _rel, lu in lookup_sources:
            if vid in lu:
                per_source_hits[lbl] += 1

    for vid, _sid in snippet_rows:
        m = merged_video.get(vid)
        if m is None:
            continue
        val, _k, _lbl = m
        union_snippets += 1
        if is_stable_uc(val):
            stable_uc += 1
        else:
            display_only += 1

    missing_snippets = snippets_total - union_snippets
    pct = lambda a: (100.0 * a / snippets_total) if snippets_total else 0.0

    print("┌" + "─" * 53 + "┐")
    print("│  CHANNEL ID COVERAGE REPORT                         │")
    print("│                                                     │")
    print(f"│  Total snippets in metadata_clean_03:   {snippets_total:<11} │")
    print("│                                                     │")
    print("│  Found channel id:                                  │")
    for lbl, rel, _lu in lookup_sources:
        line = f"From {rel}:"
        if len(line) > 34:
            line = f"From {lbl}:"
        print(f"│    {line:<34} {per_source_hits[lbl]:>5} ({pct(per_source_hits[lbl]):>4.1f}%) │")
    print(f"│    From any source (union):       {union_snippets:>5} ({pct(union_snippets):>4.1f}%) │")
    print("│                                                     │")
    print(f"│  No channel id found anywhere:    {missing_snippets:>5} ({pct(missing_snippets):>4.1f}%) │")
    print("│    → these need GPT-4o inference                   │")
    print("│                                                     │")
    print("│  Channel id quality:                                │")
    print(f"│    UCxxxxx format (stable):       {stable_uc:>5} ({pct(stable_uc):>4.1f}%) │")
    print(f"│    Display name only:             {display_only:>5} ({pct(display_only):>4.1f}%) │")
    print("└" + "─" * 53 + "┘")
    print("")

    snippet_by_plat_total: dict[str, int] = defaultdict(int)
    snippet_by_plat_union: dict[str, int] = defaultdict(int)
    for vid, _sid in snippet_rows:
        plat = infer_platform(video_record.get(vid, {}))
        snippet_by_plat_total[plat] += 1
    for vid, _sid in snippet_rows:
        if merged_video.get(vid):
            plat = infer_platform(video_record.get(vid, {}))
            snippet_by_plat_union[plat] += 1
    print("By platform (snippets in metadata_clean_03):")
    for plat in ("youtube", "tiktok", "dailymotion"):
        tot = int(snippet_by_plat_total.get(plat, 0))
        if tot == 0:
            continue
        u = int(snippet_by_plat_union.get(plat, 0))
        print(f"  {plat}: {u}/{tot} snippets resolved ({100.0 * u / tot:.1f}%)")
    print(
        "  Note: dataset/tiktok_metadata.jsonl is post-process output and usually has no\n"
        "  uploader/channel columns; TikTok creator handles come from merged stage-1\n"
        "  enrich candidates under tiktok_pipeline/runs/ when those runs exist."
    )
    print("")

    coverage = union_snippets / snippets_total if snippets_total else 0.0

    missing_videos = sorted(v for v in video_snippet_count if v not in merged_video)

    # Step 3
    if coverage >= 0.70:
        out_path = REPO_ROOT / ENRICHED_OUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as outf:
            for rec in clean_records:
                new_rec = copy.deepcopy(rec)
                vid = str(new_rec.get("video_id") or "")
                m = merged_video.get(vid)
                if m:
                    val, _k, src_used = m
                    new_rec["channel_id"] = val
                    new_rec["channel_id_source"] = src_used
                else:
                    new_rec["channel_id"] = None
                    new_rec["channel_id_source"] = "missing"
                snips = new_rec.get("snippets")
                if isinstance(snips, list):
                    for sn in snips:
                        if isinstance(sn, dict):
                            if m:
                                sn["channel_id"] = m[0]
                                sn["channel_id_source"] = m[2]
                            else:
                                sn["channel_id"] = None
                                sn["channel_id_source"] = "missing"
                outf.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        print(
            f"✅ Wrote enriched manifest with {union_snippets}/{snippets_total} channel ids resolved\n"
            f"   Run 03_cat_id.py again with this manifest for better clustering."
        )
    else:
        print(
            f"⚠ Only {coverage * 100:.1f}% of snippets have channel ids from metadata.\n"
            "  GPT-4o inference recommended for the remainder.\n"
            "  See Step 4 output for which video_ids need inference."
        )
    print("")

    # Step 4 — always
    missing_path = REPO_ROOT / MISSING_OUT
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    n_vid = len(missing_videos)
    est_cost = n_vid * 200 * 0.0025 / 1000.0
    with open(missing_path, "w", encoding="utf-8") as mf:
        for vid in missing_videos:
            rec = video_record.get(vid) or {}
            plat = infer_platform(rec)
            title = rec.get("original_video") or title_from_v2.get(vid) or ""
            if isinstance(title, str):
                title = title.strip()
            row = {
                "video_id": vid,
                "platform": plat,
                "title": title,
                "snippet_count": int(video_snippet_count[vid]),
                "url": video_url(plat, vid),
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {missing_path.relative_to(REPO_ROOT)}")
    print(f"Videos needing channel id:  {n_vid}")
    print(
        f"Estimated GPT cost at $0.0025/1k tokens, ~200 tokens/video:\n"
        f"  ~${est_cost:.2f} for {n_vid} videos"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
