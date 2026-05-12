#!/usr/bin/env python3
"""
Fetch channel / uploader metadata via yt-dlp (no video download) and merge
into manifest records.

Uses:
  yt-dlp --dump-json --no-download "<watch url>"

Writes standard fields when missing (or with --force): ``uploader_id``,
``channel_id`` (from yt-dlp), and ``channel`` (display name from
``uploader``). Results are cached in JSONL for resume.

  python dataset_construction/enrich_channels_ytdlp.py
  python dataset_construction/enrich_channels_ytdlp.py --dry-run --limit 5
  python dataset_construction/enrich_channels_ytdlp.py --force
  python dataset_construction/enrich_channels_ytdlp.py --workers 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = "src/dataset_construction/manifests/metadata_clean_03.jsonl"
DEFAULT_OUTPUT = "src/dataset_construction/manifests/metadata_clean_03_ytdlp.jsonl"
DEFAULT_CACHE = "src/dataset_construction/reports/ytdlp_channel_cache.jsonl"


def infer_platform(record: dict[str, Any]) -> str:
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


def watch_url(platform: str, video_id: str) -> str:
    if platform == "tiktok":
        return f"https://www.tiktok.com/video/{video_id}"
    if platform == "dailymotion":
        return f"https://www.dailymotion.com/video/{video_id}"
    return f"https://www.youtube.com/watch?v={video_id}"


def has_channel_fields(rec: dict[str, Any]) -> bool:
    for k in ("uploader_id", "channel_id", "channel", "uploader"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """video_id -> cached payload (must include video_id)."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and o.get("video_id"):
                out[str(o["video_id"])] = o
    return out


def append_cache_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_ytdlp_meta(url: str, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    """
    Return (fields_dict, error_message).
    fields_dict keys: uploader_id, channel_id, channel (display), uploader (same as channel).
    """
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--no-warnings",
        "--skip-download",
        url,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except FileNotFoundError:
        return None, "yt-dlp not found in PATH"
    except Exception as e:
        return None, repr(e)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        return None, err or f"exit {proc.returncode}"

    raw = (proc.stdout or "").strip()
    if not raw:
        return None, "empty stdout"

    try:
        d = json.loads(raw.split("\n", 1)[0])
    except json.JSONDecodeError as e:
        return None, str(e)

    if not isinstance(d, dict):
        return None, "root not object"

    uid = d.get("uploader_id")
    cid = d.get("channel_id")
    up = d.get("uploader")
    uid_s = uid.strip() if isinstance(uid, str) else None
    cid_s = cid.strip() if isinstance(cid, str) else None
    up_s = up.strip() if isinstance(up, str) else None
    display = up_s or uid_s or cid_s
    out = {
        "uploader_id": uid_s,
        "channel_id": cid_s or uid_s,
        "uploader": up_s,
        "channel": display,
    }
    return out, None


def merge_meta(rec: dict[str, Any], meta: dict[str, Any], force: bool) -> dict[str, Any]:
    r = dict(rec)
    for k, v in meta.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if force or not (isinstance(r.get(k), str) and str(r[k]).strip()):
            r[k] = v
    return r


@dataclass
class RecordStats:
    n_cached: int = 0
    n_calls: int = 0
    n_errors: int = 0
    n_skipped_has_fields: int = 0
    n_skipped_platform: int = 0
    n_would_fetch: int = 0


class FetchSharedState:
    """Thread-safe cache, yt-dlp call limit, and per-video_id single-flight for fetches."""

    def __init__(
        self,
        args: argparse.Namespace,
        cache: dict[str, dict[str, Any]],
        cache_path: Path,
    ) -> None:
        self.args = args
        self.cache = cache
        self.cache_path = cache_path
        self.cache_lock = threading.Lock()
        self.limit_lock = threading.Lock()
        self._vid_locks: dict[str, threading.Lock] = {}
        self._vid_locks_guard = threading.Lock()
        self.n_calls = 0

    def lock_for_video(self, vid: str) -> threading.Lock:
        with self._vid_locks_guard:
            lk = self._vid_locks.get(vid)
            if lk is None:
                lk = threading.Lock()
                self._vid_locks[vid] = lk
            return lk


def enrich_one_record(idx: int, rec: dict[str, Any], st: FetchSharedState) -> tuple[int, dict[str, Any], RecordStats]:
    """Process one manifest row. Returns (input index, output row, per-record stats)."""
    stats = RecordStats()
    vid = str(rec.get("video_id") or "")
    if not vid:
        return idx, rec, stats

    plat = infer_platform(rec)
    if st.args.youtube_only and plat != "youtube":
        stats.n_skipped_platform = 1
        return idx, rec, stats

    if has_channel_fields(rec) and not st.args.force:
        stats.n_skipped_has_fields = 1
        return idx, rec, stats

    meta: dict[str, Any] | None = None
    err: str | None = None

    with st.cache_lock:
        c = st.cache.get(vid) if not st.args.force else None
        if c and not c.get("error"):
            meta = {
                "uploader_id": c.get("uploader_id"),
                "channel_id": c.get("channel_id"),
                "uploader": c.get("uploader"),
                "channel": c.get("channel"),
            }
            stats.n_cached = 1

    if meta is None:
        if st.args.dry_run:
            stats.n_would_fetch = 1
            return idx, rec, stats

        vlk = st.lock_for_video(vid)
        with vlk:
            with st.cache_lock:
                c2 = st.cache.get(vid) if not st.args.force else None
                if c2 and not c2.get("error"):
                    meta = {
                        "uploader_id": c2.get("uploader_id"),
                        "channel_id": c2.get("channel_id"),
                        "uploader": c2.get("uploader"),
                        "channel": c2.get("channel"),
                    }
                    stats.n_cached = 1

            if meta is None:
                with st.limit_lock:
                    if st.args.limit is not None and st.n_calls >= st.args.limit:
                        return idx, rec, stats
                    st.n_calls += 1
                    stats.n_calls = 1
                url = watch_url(plat, vid)
                meta, err = fetch_ytdlp_meta(url, st.args.timeout)
                cache_row = {
                    "video_id": vid,
                    "platform": plat,
                    "url": url,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                if meta:
                    cache_row.update(meta)
                    cache_row["error"] = None
                else:
                    cache_row["error"] = err or "unknown"
                    stats.n_errors = 1
                with st.cache_lock:
                    append_cache_line(st.cache_path, cache_row)
                    st.cache[vid] = cache_row
                if st.args.sleep > 0:
                    time.sleep(float(st.args.sleep))

    if meta:
        merged = merge_meta(rec, meta, st.args.force)
        merged["channel_ytdlp_fetched_at"] = datetime.now(timezone.utc).isoformat()
        return idx, merged, stats

    r = dict(rec)
    if err:
        r.setdefault("channel_ytdlp_error", err[:300])
    return idx, r, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich manifest records with yt-dlp channel metadata.")
    ap.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input JSONL manifest (relative to REPO_ROOT or absolute).",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--cache",
        type=str,
        default=DEFAULT_CACHE,
        help="Append-only cache JSONL for resume.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions only; do not write output/cache.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing uploader_id/channel_id/channel.")
    ap.add_argument("--limit", type=int, default=None, help="Max yt-dlp calls (after cache).")
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between network calls.")
    ap.add_argument("--timeout", type=int, default=120, help="yt-dlp subprocess timeout seconds.")
    ap.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Parallel worker threads (yt-dlp subprocesses). Use 1 for strictly sequential behavior.",
    )
    ap.add_argument(
        "--youtube-only",
        action="store_true",
        help="Only enrich youtube platform rows (skip TikTok/DailyMotion).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    cache_path = Path(args.cache)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path

    if not in_path.is_file():
        print(f"Missing input: {in_path}", file=sys.stderr)
        return 1

    records = load_jsonl(in_path)
    cache = load_cache(cache_path)
    n_cached = 0
    n_skipped_has_fields = 0
    n_skipped_platform = 0
    n_errors = 0
    n_would_fetch = 0
    ordered: list[dict[str, Any] | None] = [None] * len(records)

    st = FetchSharedState(args, cache, cache_path)
    workers = max(1, int(args.workers))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(enrich_one_record, i, records[i], st) for i in range(len(records))]
        pbar = tqdm(
            as_completed(futures),
            desc="yt-dlp channel",
            unit="record",
            total=len(futures),
            dynamic_ncols=True,
        )
        for fut in pbar:
            idx, row, s = fut.result()
            ordered[idx] = row
            n_cached += s.n_cached
            n_skipped_has_fields += s.n_skipped_has_fields
            n_skipped_platform += s.n_skipped_platform
            n_errors += s.n_errors
            n_would_fetch += s.n_would_fetch
            pbar.set_postfix(dlp=st.n_calls, cache=n_cached, err=n_errors, refresh=True)

    out_rows = [r for r in ordered if r is not None]
    if len(out_rows) != len(records):
        print("Internal error: missing output rows.", file=sys.stderr)
        return 1

    n_calls = st.n_calls
    print(
        f"Records: {len(records)} | yt-dlp calls: {n_calls} | cache hits: {n_cached} | "
        f"skipped (already had fields): {n_skipped_has_fields} | "
        f"skipped (youtube-only): {n_skipped_platform} | errors: {n_errors}"
    )
    if args.dry_run:
        print(f"Dry run: would call yt-dlp for ~{n_would_fetch} videos (no network, no writes).")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
