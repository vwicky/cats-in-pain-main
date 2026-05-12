"""yt-dlp download batch (adapted from streaming_video_pipeline)."""

from __future__ import annotations

import glob
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from yt_dlp import YoutubeDL

from typing import TYPE_CHECKING

from src.constants import CATEGORY_DISCLAIMER
from src.utils import append_jsonl, load_jsonl, project_root, resolve_path

if TYPE_CHECKING:
    from src.dedup import VideoRegistry


def _cookie_file_for_ytdlp(cookie_file: str | None) -> str | None:
    if cookie_file and str(cookie_file).strip():
        return str(cookie_file).strip()
    env = os.environ.get("COOKIE_FILE", "").strip()
    return env or None


def build_auth_opts(cookie_file: str | None, cookies_from_browser: str | None) -> dict[str, Any]:
    auth_opts: dict[str, Any] = {}
    cf = _cookie_file_for_ytdlp(cookie_file)
    if cf:
        auth_opts["cookiefile"] = cf
    if cookies_from_browser:
        auth_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return auth_opts


def _yt_dlp_extra_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {}
    cache = os.environ.get("YTDLP_CACHE_DIR", "").strip()
    if cache:
        os.makedirs(cache, exist_ok=True)
        opts["cachedir"] = cache
    return opts


def _remote_components(dc: dict[str, Any]) -> set[str]:
    """Allow yt-dlp to fetch EJS challenge solvers (recommended for YouTube). Empty = none."""
    raw = dc.get("remote_components")
    if raw is None:
        return {"ejs:github"}
    if isinstance(raw, (list, tuple, set)):
        return {str(x) for x in raw}
    return {"ejs:github"}


def resolve_single_file(glob_pattern: str) -> str | None:
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def download_video_assets(
    video_id: str,
    output_folder: str,
    cfg: dict[str, Any],
    cookie_file: str | None = None,
    cookies_from_browser: str | None = None,
) -> dict[str, Any]:
    """Adapted from streaming_video_pipeline.download_video_assets."""
    dc = cfg.get("download", {})
    max_dur = float(dc.get("max_duration_seconds", 300))
    max_size = int(dc.get("max_filesize_bytes", 524288000))
    max_h = int(dc.get("max_height", 1080))
    aq = str(dc.get("audio_quality", "192"))
    retries = int(dc.get("retries", 3))
    frag_retries = int(dc.get("fragment_retries", 3))

    os.makedirs(output_folder, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    base_opts = {
        "quiet": True,
        "js_runtimes": {"node": {}},
        "remote_components": _remote_components(dc),
        **build_auth_opts(cookie_file, cookies_from_browser),
        **_yt_dlp_extra_opts(),
    }
    try:
        with YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        duration = info.get("duration")
        filesize = info.get("filesize") or info.get("filesize_approx")
        if duration is None or duration > max_dur:
            return {
                "video_id": video_id,
                "error": "video too long",
                "duration_seconds": duration,
                "status": "too_long",
            }
        if filesize and filesize > max_size:
            return {
                "video_id": video_id,
                "error": "video too large",
                "duration_seconds": duration,
                "filesize_bytes": filesize,
                "status": "too_large",
            }

        title = info.get("title") or video_id

        video_opts = {
            **base_opts,
            "format": f"bestvideo[height<={max_h}][ext=mp4]/bestvideo[height<={max_h}]/bestvideo/best",
            "outtmpl": os.path.join(output_folder, "%(id)s_video.%(ext)s"),
            "merge_output_format": "mp4",
            "max_filesize": max_size,
            "noplaylist": True,
            "retries": retries,
            "fragment_retries": frag_retries,
        }
        with YoutubeDL(video_opts) as ydl:
            ydl.download([url])

        audio_opts = {
            **base_opts,
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(output_folder, "%(id)s_audio.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": aq,
                }
            ],
            "noplaylist": True,
            "retries": retries,
            "fragment_retries": frag_retries,
        }
        with YoutubeDL(audio_opts) as ydl:
            ydl.download([url])

        video_path = resolve_single_file(os.path.join(output_folder, f"{video_id}_video.*"))
        audio_path = resolve_single_file(os.path.join(output_folder, f"{video_id}_audio.*"))
        if not video_path or not audio_path:
            return {"video_id": video_id, "error": "downloaded files not found", "status": "download_error"}

        return {
            "video_id": video_id,
            "title": title,
            "video_path": video_path,
            "audio_path": audio_path,
            "duration_sec": float(duration or 0),
            "filesize_bytes": int(filesize or 0),
            "status": "success",
        }
    except Exception as exc:
        return {"video_id": video_id, "error": str(exc), "status": "download_error"}


def collect_skip_video_ids(cfg: dict, registry: "VideoRegistry | None" = None) -> set[str]:
    """Union: registry skip set + metadata (any row = already processed)."""
    root = project_root()
    skip: set[str] = set()
    if registry is not None:
        skip.update(registry.build_skip_set())
    out = cfg.get("output", {})
    meta = resolve_path(root, out.get("metadata_file", "data/dataset/metadata.jsonl"))
    for row in load_jsonl(meta):
        vid = row.get("video_id")
        if isinstance(vid, str) and vid:
            skip.add(vid)
    return skip


def download_batch(
    video_records: list[dict[str, Any]],
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
    metadata_path: Path,
    pipeline_log_path: Path | None = None,
    registry: "VideoRegistry | None" = None,
) -> list[dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    root = project_root()
    dc = cfg.get("download", {})
    tmp_dir = resolve_path(root, dc.get("tmp_dir", "tmp_downloads"))
    out_dir = resolve_path(root, dc.get("output_dir", "data/dataset/snippets"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cookie_file = (dc.get("cookie_file") or "").strip() or None
    cookies_from_browser = (dc.get("cookies_from_browser") or "").strip() or None

    skip = collect_skip_video_ids(cfg, registry)
    results: list[dict[str, Any]] = []
    stats = Counter()

    for rec in tqdm(video_records, desc="Download", unit="vid"):
        vid = rec.get("video_id")
        if not vid:
            continue
        # skip if in metadata / legacy
        if vid in skip:
            r = {
                "video_id": vid,
                "title": rec.get("title", ""),
                "status": "already_exists",
                "behavioral_category": rec.get("behavioral_category", ""),
                "download_time_sec": 0.0,
            }
            results.append(r)
            stats["already_exists"] += 1
            append_jsonl(r, run_dir / "stage_4_download" / "download_log.jsonl")
            if pipeline_log_path:
                append_jsonl({**r, "stage": "download"}, pipeline_log_path)
            continue

        t0 = time.monotonic()
        dest = str(tmp_dir / vid)
        os.makedirs(dest, exist_ok=True)
        # also check final video in out_dir
        existing = list(out_dir.glob(f"{vid}_video.*"))
        if existing:
            r = {
                "video_id": vid,
                "title": rec.get("title", ""),
                "status": "already_exists",
                "behavioral_category": rec.get("behavioral_category", ""),
                "download_time_sec": 0.0,
            }
            results.append(r)
            stats["already_exists"] += 1
            append_jsonl(r, run_dir / "stage_4_download" / "download_log.jsonl")
            if pipeline_log_path:
                append_jsonl({**r, "stage": "download"}, pipeline_log_path)
            continue

        try:
            dl = download_video_assets(vid, dest, cfg, cookie_file, cookies_from_browser)
        except Exception as e:
            dl = {"video_id": vid, "error": str(e), "status": "download_error"}

        elapsed = time.monotonic() - t0
        st = dl.get("status", "download_error")
        if st == "success":
            stats["success"] += 1
            # move/copy to persistent output_dir with stable names
            import shutil

            for pat in [f"{vid}_video.*", f"{vid}_audio.*"]:
                for f in glob.glob(os.path.join(dest, pat)):
                    shutil.copy2(f, out_dir / os.path.basename(f))
            vp = resolve_single_file(str(out_dir / f"{vid}_video.*"))
            ap = resolve_single_file(str(out_dir / f"{vid}_audio.*"))
            r = {
                "video_id": vid,
                "title": dl.get("title", rec.get("title", "")),
                "status": "success",
                "duration_sec": dl.get("duration_sec"),
                "filesize_bytes": dl.get("filesize_bytes"),
                "video_path": vp,
                "audio_path": ap,
                "download_time_sec": round(elapsed, 2),
                "behavioral_category": rec.get("behavioral_category", ""),
                "query_language": rec.get("query_language", ""),
            }
        elif st == "too_long":
            stats["too_long"] += 1
            r = {
                "video_id": vid,
                "title": rec.get("title", ""),
                "status": "too_long",
                "duration_seconds": dl.get("duration_seconds"),
                "download_time_sec": round(elapsed, 2),
                "behavioral_category": rec.get("behavioral_category", ""),
            }
        elif st == "too_large":
            stats["too_large"] += 1
            r = {
                "video_id": vid,
                "title": rec.get("title", ""),
                "status": "too_large",
                "filesize_bytes": dl.get("filesize_bytes"),
                "download_time_sec": round(elapsed, 2),
                "behavioral_category": rec.get("behavioral_category", ""),
            }
        else:
            stats["errors"] += 1
            r = {
                "video_id": vid,
                "title": rec.get("title", ""),
                "status": "download_error",
                "error": dl.get("error", "unknown"),
                "download_time_sec": round(elapsed, 2),
                "behavioral_category": rec.get("behavioral_category", ""),
            }

        results.append(r)
        append_jsonl(r, run_dir / "stage_4_download" / "download_log.jsonl")
        if pipeline_log_path:
            append_jsonl({**r, "stage": "download"}, pipeline_log_path)

        # cleanup tmp
        try:
            import shutil

            shutil.rmtree(dest, ignore_errors=True)
        except Exception:
            pass

    # summary box
    att = len(video_records)
    print(
        f"""
┌────────────────────────────────────────────────────────┐
│  DOWNLOAD SUMMARY                                      │
│  Attempted:   {att:>6}                                    │
│  Success:     {stats['success']:>6} ({100*stats['success']/max(att,1):.1f}%)                            │
│  Too long:    {stats['too_long']:>6}                                    │
│  Too large:   {stats['too_large']:>6}                                    │
│  Errors:      {stats['errors']:>6}                                    │
│  Already had: {stats['already_exists']:>6}                                    │
└────────────────────────────────────────────────────────┘"""
    )

    # stacked bar by behavioral_category
    by_cat = defaultdict(lambda: defaultdict(int))
    for r in results:
        cat = r.get("behavioral_category") or "Unknown"
        by_cat[cat][r.get("status", "?")] += 1

    if by_cat:
        cats = sorted(by_cat.keys())
        statuses = ["success", "too_long", "too_large", "download_error", "already_exists"]
        bottom = np.zeros(len(cats))
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#3498db"]
        for i, st in enumerate(statuses):
            vals = [by_cat[c].get(st, 0) for c in cats]
            ax.bar(cats, vals, bottom=bottom, label=st, color=colors[i % len(colors)])
            bottom += np.array(vals)
        ax.set_ylabel("Count")
        ax.legend(title="Status")
        fig.suptitle(f"Download outcomes by behavioral_category\n{CATEGORY_DISCLAIMER}", fontsize=9)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(run_dir / "stage_4_download" / "download_outcomes.png", dpi=150)
        plt.close(fig)

    return results
