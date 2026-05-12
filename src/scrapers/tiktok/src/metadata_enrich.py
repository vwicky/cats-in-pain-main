"""Fill title, description, tags, duration from yt-dlp metadata (no MP4 download).

Hashtag grid pages only yield URLs; this stage runs after search so tag/GPT filters
have text to analyze.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm
from yt_dlp import YoutubeDL

from src.downloader import build_auth_opts
from src.search import candidate_from_ytdlp_info


def _yt_dlp_cache_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {}
    cache = os.environ.get("YTDLP_CACHE_DIR", "").strip()
    if cache:
        os.makedirs(cache, exist_ok=True)
        opts["cachedir"] = cache
    return opts


def _row_needs_ytdlp(row: dict[str, Any], only_when_empty: bool) -> bool:
    if not only_when_empty:
        return True
    title = str(row.get("title") or "").strip()
    desc = str(row.get("description") or "").strip()
    tags = row.get("tags") or []
    ht = row.get("hashtags") or []
    has_tags = isinstance(tags, list) and any(str(t).strip() for t in tags)
    has_ht = isinstance(ht, list) and any(str(t).strip() for t in ht)
    return not title and not desc and not has_tags and not has_ht


def _video_url(row: dict[str, Any]) -> str:
    w = row.get("webpage_url")
    if isinstance(w, str) and w.strip().startswith("http"):
        return w.strip()
    vid = str(row.get("video_id") or "").strip()
    return f"https://www.tiktok.com/video/{vid}" if vid else ""


def _build_ydl_opts(cfg: dict[str, Any]) -> dict[str, Any]:
    ec = cfg.get("metadata_enrich") or {}
    dc = cfg.get("download") or {}
    sc = cfg.get("search") or {}
    cf = (ec.get("cookie_file") or dc.get("cookie_file") or sc.get("cookie_file") or "").strip() or None
    cfb = (
        ec.get("cookies_from_browser")
        or dc.get("cookies_from_browser")
        or sc.get("cookies_from_browser")
        or ""
    ).strip() or None
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        **build_auth_opts(cf, cfb),
        **_yt_dlp_cache_opts(),
    }
    sto = ec.get("socket_timeout_sec")
    if sto is not None and str(sto).strip() != "":
        opts["socket_timeout"] = float(sto)
    return opts


def _fetch_single(
    row: dict[str, Any],
    ydl_opts: dict[str, Any],
    max_retries: int,
    sleep_interval: float,
) -> tuple[dict[str, Any], bool]:
    """Run yt-dlp extract_info for one row. Returns (row_out, success)."""
    url = _video_url(row)
    if not url:
        r = dict(row)
        r["metadata_enrich_error"] = "missing webpage_url and video_id"
        return r, False

    info = None
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            with YoutubeDL({**ydl_opts}) as ydl:
                info = ydl.extract_info(url, download=False)
            break
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(sleep_interval * (attempt + 1))

    if not info:
        r = dict(row)
        r["metadata_enrich_error"] = last_err or "extract_info failed"
        time.sleep(sleep_interval)
        return r, False

    merged = candidate_from_ytdlp_info(row, info)
    if not merged:
        r = dict(row)
        r["metadata_enrich_error"] = "candidate_from_ytdlp_info returned None"
        time.sleep(sleep_interval)
        return r, False

    merged.pop("metadata_enrich_error", None)
    time.sleep(sleep_interval)
    return merged, True


def run_metadata_enrich(
    candidates: list[dict[str, Any]],
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ecfg = cfg.get("metadata_enrich") or {}
    if not ecfg.get("enabled", True):
        logger.info("Metadata enrich: disabled in config; skipping yt-dlp metadata pass.")
        return candidates, {"skipped": True, "enabled": False}

    only_when_empty = bool(ecfg.get("only_when_empty", True))
    sleep_interval = float(ecfg.get("sleep_interval", 1.5))
    max_retries = int(ecfg.get("max_retries", 2))
    parallel_workers = max(1, int(ecfg.get("parallel_workers", 8)))

    out_dir = run_dir / "stage_1_enrich"
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = _build_ydl_opts(cfg)
    stats: dict[str, Any] = {
        "input": len(candidates),
        "needs_fetch": 0,
        "ok": 0,
        "failed": 0,
        "skipped_already_filled": 0,
        "parallel_workers": parallel_workers,
    }

    enriched_slots: list[dict[str, Any] | None] = [None] * len(candidates)
    needs_indices: list[int] = []
    for i, row in enumerate(candidates):
        if not _row_needs_ytdlp(row, only_when_empty):
            stats["skipped_already_filled"] += 1
            enriched_slots[i] = dict(row)
        else:
            stats["needs_fetch"] += 1
            needs_indices.append(i)

    to_fetch: list[int] = []
    for i in needs_indices:
        row = candidates[i]
        if not _video_url(row):
            stats["failed"] += 1
            r = dict(row)
            r["metadata_enrich_error"] = "missing webpage_url and video_id"
            enriched_slots[i] = r
        else:
            to_fetch.append(i)

    if to_fetch:
        if parallel_workers == 1:
            for i in tqdm(to_fetch, desc="Metadata (yt-dlp)", unit="vid"):
                row_out, ok = _fetch_single(candidates[i], ydl_opts, max_retries, sleep_interval)
                enriched_slots[i] = row_out
                if ok:
                    stats["ok"] += 1
                else:
                    stats["failed"] += 1
        else:
            logger.info(
                "Metadata enrich: using %s parallel workers for yt-dlp (sleep_interval=%s)",
                parallel_workers,
                sleep_interval,
            )
            futs = {}
            ex = ThreadPoolExecutor(max_workers=parallel_workers)
            try:
                for i in to_fetch:
                    fut = ex.submit(_fetch_single, candidates[i], ydl_opts, max_retries, sleep_interval)
                    futs[fut] = i
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="Metadata (yt-dlp)",
                    unit="vid",
                ):
                    i = futs[fut]
                    row_out, ok = fut.result()
                    enriched_slots[i] = row_out
                    if ok:
                        stats["ok"] += 1
                    else:
                        stats["failed"] += 1
            except BaseException:
                # Do not block on shutdown while workers are stuck in HTTP (e.g. Ctrl+C).
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                ex.shutdown(wait=True)

    enriched = [r for r in enriched_slots if r is not None]
    if len(enriched) != len(candidates):
        raise RuntimeError(
            f"metadata enrich internal error: filled {len(enriched)} != {len(candidates)}"
        )

    summary_path = out_dir / "enrich_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(
        "Metadata enrich: input=%s needs_ytdlp=%s ok=%s failed=%s skipped_filled=%s",
        stats["input"],
        stats["needs_fetch"],
        stats["ok"],
        stats["failed"],
        stats["skipped_already_filled"],
    )

    return enriched, stats
