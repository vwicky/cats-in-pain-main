#!/usr/bin/env python3
"""
Delete downloaded source video+audio for videos that **already produced snippets**
in the consolidated metadata JSONL — frees disk while leaving unprocessed downloads intact.

Targets (repo-root-relative paths from each pipeline's default config):
  - YouTube (data_pipeline_v2): dataset/videos_v2 + dataset/metadata_v2.jsonl
  - TikTok: dataset/tiktok_videos + dataset/tiktok_metadata.jsonl
  - Dailymotion: dailymotion_scraper/output/videos + dailymotion_scraper/dataset/dailymotion_metadata.jsonl

Safety:
  - Default is **dry-run** (list only). Pass ``--execute`` to delete.
  - Skips media files whose mtime is newer than ``--min-age-minutes`` (avoids deleting
    files an active pipeline run may still be writing).
  - Never touches ``*/runs/*``, ``*/tmp_downloads/*``, or snippet output dirs — only
    ``output_dir``-style folders with ``{video_id}_video.*`` / ``{video_id}_audio.*``.

Usage:
  python scripts/cleanup_processed_source_media.py
  python scripts/cleanup_processed_source_media.py --execute --min-age-minutes 90
  python scripts/cleanup_processed_source_media.py --sources youtube,dailymotion --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(repo: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (repo / path).resolve()


def _video_ids_with_snippets(metadata_path: Path, include_zero_snippets: bool) -> set[str]:
    """video_ids that have been processed (optionally requiring at least one saved snippet)."""
    out: set[str] = set()
    if not metadata_path.is_file():
        return out
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = r.get("video_id")
            if not vid:
                continue
            snippets = r.get("snippets") or []
            saved = r.get("saved_chunks_count")
            n_snip = len(snippets) if isinstance(snippets, list) else 0
            if include_zero_snippets:
                # Ran through process stage (has processing markers) even if 0 snippets saved
                if r.get("processed_at") is not None or r.get("total_chunks_analyzed") is not None:
                    out.add(str(vid))
            else:
                if n_snip > 0 or (isinstance(saved, int) and saved > 0):
                    out.add(str(vid))
    return out


def _media_paths_for_id(video_dir: Path, video_id: str) -> list[Path]:
    files: list[Path] = []
    for pattern in (f"{video_id}_video.*", f"{video_id}_audio.*"):
        files.extend(video_dir.glob(pattern))
    return files


def _should_skip_path(path: Path, repo: Path, min_age_sec: float, now: float) -> tuple[bool, str]:
    rp = path.resolve()
    parts = rp.parts
    s = str(rp)
    if "runs" in parts:
        return True, "under runs/ (skipped)"
    if "tmp_downloads" in parts:
        return True, "under tmp_downloads/ (skipped)"
    try:
        rp.relative_to(repo)
    except ValueError:
        return True, "outside repo (skipped)"
    try:
        mtime = path.stat().st_mtime
    except OSError as e:
        return True, f"stat error: {e}"
    if now - mtime < min_age_sec:
        return True, f"too new (mtime within min-age window)"
    return False, ""


def cleanup_source(
    name: str,
    video_dir: Path,
    metadata_path: Path,
    *,
    execute: bool,
    min_age_sec: float,
    include_zero_snippets: bool,
    repo: Path,
) -> tuple[int, int, list[str]]:
    """
    Returns (n_files_deleted_or_would, n_bytes, log_lines).
    """
    ids = _video_ids_with_snippets(metadata_path, include_zero_snippets=include_zero_snippets)
    lines: list[str] = []
    n_files = 0
    n_bytes = 0
    now = time.time()

    if not video_dir.is_dir():
        lines.append(f"[{name}] skip: video dir missing: {video_dir}")
        return 0, 0, lines

    if not metadata_path.is_file():
        lines.append(f"[{name}] skip: metadata missing: {metadata_path}")
        return 0, 0, lines

    lines.append(f"[{name}] metadata: {metadata_path} ({len(ids)} video_ids with qualifying output)")

    for vid in sorted(ids):
        for path in _media_paths_for_id(video_dir, vid):
            skip, reason = _should_skip_path(path, repo, min_age_sec, now)
            if skip:
                lines.append(f"  keep: {path.name} — {reason}")
                continue
            try:
                sz = path.stat().st_size
            except OSError:
                continue
            n_files += 1
            n_bytes += sz
            if execute:
                try:
                    path.unlink()
                    lines.append(f"  deleted: {path}")
                except OSError as e:
                    lines.append(f"  ERROR unlink {path}: {e}")
            else:
                lines.append(f"  would delete ({sz/1e6:.2f} MB): {path}")

    return n_files, n_bytes, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sources",
        default="youtube,tiktok,dailymotion",
        help="Comma-separated: youtube, tiktok, dailymotion",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry-run only).",
    )
    ap.add_argument(
        "--min-age-minutes",
        type=float,
        default=90.0,
        help="Do not delete files modified more recently than this (default: 90). Protects active runs.",
    )
    ap.add_argument(
        "--include-zero-snippets",
        action="store_true",
        help="Also remove media for videos that were processed but saved 0 snippets (still in metadata).",
    )
    ap.add_argument(
        "--youtube-video-dir",
        type=str,
        default=None,
        help="Override YouTube download dir (default: from data_pipeline_v2/config/pipeline.yaml).",
    )
    ap.add_argument(
        "--youtube-metadata",
        type=str,
        default=None,
        help="Override YouTube metadata JSONL path.",
    )
    ap.add_argument(
        "--tiktok-video-dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--tiktok-metadata",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--dailymotion-video-dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--dailymotion-metadata",
        type=str,
        default=None,
    )
    args = ap.parse_args()
    repo = _repo_root()
    min_age_sec = max(0.0, float(args.min_age_minutes) * 60.0)

    cfg_v2 = _repo_root() / "src" / "scrapers" / "data_pipeline_v2" / "config" / "pipeline.yaml"
    cfg_tt = _repo_root() / "src" / "scrapers" / "tiktok" / "config" / "pipeline.yaml"
    cfg_dm = _repo_root() / "src" / "scrapers" / "dailymotion" / "config" / "pipeline.yaml"

    v2 = _load_yaml(cfg_v2) if cfg_v2.is_file() else {}
    tt = _load_yaml(cfg_tt) if cfg_tt.is_file() else {}
    dm = _load_yaml(cfg_dm) if cfg_dm.is_file() else {}

    yt_dir = args.youtube_video_dir or (v2.get("download") or {}).get("output_dir", "data/dataset/videos_v2")
    yt_meta = args.youtube_metadata or (v2.get("output") or {}).get("metadata_file", "data/dataset/metadata_v2.jsonl")
    tt_dir = args.tiktok_video_dir or (tt.get("download") or {}).get("output_dir", "data/dataset/tiktok_videos")
    tt_meta = args.tiktok_metadata or (tt.get("output") or {}).get("metadata_file", "data/dataset/tiktok_metadata.jsonl")
    dm_dir = args.dailymotion_video_dir or (dm.get("download") or {}).get("output_dir", "src/scrapers/dailymotion/output/videos")
    dm_meta = args.dailymotion_metadata or (dm.get("output") or {}).get("metadata_file", "src/scrapers/dailymotion/dataset/dailymotion_metadata.jsonl")

    presets: dict[str, tuple[Path, Path]] = {
        "youtube": (_resolve(repo, yt_dir), _resolve(repo, yt_meta)),
        "tiktok": (_resolve(repo, tt_dir), _resolve(repo, tt_meta)),
        "dailymotion": (_resolve(repo, dm_dir), _resolve(repo, dm_meta)),
    }

    wanted = {x.strip().lower() for x in args.sources.split(",") if x.strip()}
    unknown = wanted - set(presets.keys())
    if unknown:
        print(f"Unknown sources: {unknown}", file=sys.stderr)
        sys.exit(2)

    total_files = 0
    total_bytes = 0
    all_lines: list[str] = []

    for key in ("youtube", "tiktok", "dailymotion"):
        if key not in wanted:
            continue
        vd, md = presets[key]
        nf, nb, lines = cleanup_source(
            key,
            vd,
            md,
            execute=args.execute,
            min_age_sec=min_age_sec,
            include_zero_snippets=args.include_zero_snippets,
            repo=repo,
        )
        total_files += nf
        total_bytes += nb
        all_lines.extend(lines)

    print("\n".join(all_lines))
    print()
    mode = "DELETE" if args.execute else "DRY-RUN"
    print(f"{mode}: {total_files} file(s), ~{total_bytes / 1e9:.3f} GB")
    if not args.execute and total_files:
        print("Re-run with --execute to delete these files.")
    if not args.execute:
        print("(No files were removed.)")


if __name__ == "__main__":
    main()
