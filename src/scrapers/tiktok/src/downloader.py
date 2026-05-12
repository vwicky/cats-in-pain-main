"""yt-dlp download batch for TikTok — rate limits, match_filter, cookies.

Audio is demuxed from the downloaded MP4 with ffmpeg (or pydub fallback), not a second
yt-dlp fetch — avoids TikTok ``FFmpegExtractAudio`` / ``bestaudio`` conversion failures.

Authentication (login walls, rate limits):
  - Set ``download.cookie_file`` in pipeline.yaml to a Netscape cookie file, or
  - Set ``download.cookies_from_browser`` (e.g. ``chrome``, ``firefox``) — see yt-dlp
    ``--cookies-from-browser``, or
  - Environment variable ``COOKIE_FILE`` pointing at a cookie file.

Prefer ``webpage_url`` from the search stage for each video; avoid guessing URLs.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from yt_dlp import YoutubeDL

try:
    from yt_dlp.utils import match_filter_func
except ImportError:
    match_filter_func = None  # type: ignore[misc,assignment]

from src.constants import CATEGORY_DISCLAIMER
from src.utils import append_jsonl, load_jsonl, project_root, resolve_path


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


def _build_match_filter(dc: dict[str, Any]):
    """Reject downloads outside [min_clip_seconds, max_clip_seconds] (capped by absolute_max).

    Uses yt-dlp's match-filter DSL (not Python): ``&`` is AND; passing a list of strings ORs them.
    ``!duration`` matches when duration is missing; some extractors omit it until formats resolve.
    """
    if match_filter_func is None:
        return None
    max_sec = float(dc.get("max_clip_seconds", 30))
    abs_max = float(dc.get("absolute_max_duration_seconds", 180))
    max_sec = min(max_sec, abs_max)
    min_sec = float(dc.get("min_clip_seconds", 1))
    return match_filter_func(
        [
            "!duration",
            f"duration >= {min_sec} & duration <= {max_sec}",
        ]
    )


def resolve_single_file(glob_pattern: str) -> str | None:
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _output_media_paths_ok(out_dir: Path, vid: str) -> tuple[str | None, str | None]:
    """Return (video_path, audio_path) if both files exist under out_dir; else (None, None)."""
    vp = resolve_single_file(str(out_dir / f"{vid}_video.*"))
    ap = resolve_single_file(str(out_dir / f"{vid}_audio.*"))
    if not vp or not ap:
        return None, None
    if not Path(vp).is_file() or not Path(ap).is_file():
        return None, None
    return vp, ap


def _tiktok_video_url(rec: dict[str, Any], video_id: str) -> str:
    w = rec.get("webpage_url")
    if isinstance(w, str) and w.strip().startswith("http"):
        return w.strip()
    return f"https://www.tiktok.com/video/{video_id}"


def _extract_audio_mp3_from_video(video_path: str, mp3_out: str, bitrate_k: str) -> None:
    """Mux audio from the downloaded video to MP3.

    A second yt-dlp pass with ``FFmpegExtractAudio`` on ``bestaudio`` often fails on TikTok
    (\"Postprocessing: audio conversion failed\"). Demuxing from the MP4 we already have is
    more reliable and avoids duplicate network fetches.
    """
    br = "".join(c for c in str(bitrate_k) if c.isdigit()) or "192"
    Path(mp3_out).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    ff_err: str | None = None
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-b:a",
            f"{br}k",
            mp3_out,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(mp3_out) and os.path.getsize(mp3_out) > 0:
            return
        ff_err = (r.stderr or r.stdout or "ffmpeg exit != 0").strip()
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(video_path)
        audio.export(mp3_out, format="mp3", bitrate=f"{br}k")
    except Exception as e:
        parts = ["could not extract audio to mp3"]
        if ff_err:
            parts.append(f"ffmpeg: {ff_err[:800]}")
        parts.append(f"pydub: {e}")
        raise RuntimeError("; ".join(parts)) from e
    if not os.path.isfile(mp3_out) or os.path.getsize(mp3_out) == 0:
        raise RuntimeError("audio extract wrote empty mp3")


def download_video_assets(
    video_id: str,
    output_folder: str,
    cfg: dict[str, Any],
    cookie_file: str | None = None,
    cookies_from_browser: str | None = None,
    webpage_url: str | None = None,
) -> dict[str, Any]:
    dc = cfg.get("download", {})
    max_dur_cfg = float(dc.get("max_duration_seconds", dc.get("absolute_max_duration_seconds", 180)))
    max_size = int(dc.get("max_filesize_bytes", 524288000))
    max_h = int(dc.get("max_height", 1080))
    aq = str(dc.get("audio_quality", "192"))
    retries = int(dc.get("retries", 3))
    frag_retries = int(dc.get("fragment_retries", 3))
    sleep_interval = float(dc.get("sleep_interval", 2))
    max_sleep_interval = float(dc.get("max_sleep_interval", 5))

    url = webpage_url if (webpage_url and str(webpage_url).strip().startswith("http")) else f"https://www.tiktok.com/video/{video_id}"

    os.makedirs(output_folder, exist_ok=True)
    mf = _build_match_filter(dc)
    base_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "sleep_interval": sleep_interval,
        "max_sleep_interval": max_sleep_interval,
        **build_auth_opts(cookie_file, cookies_from_browser),
        **_yt_dlp_extra_opts(),
    }
    if mf is not None:
        base_opts["match_filter"] = mf
    try:
        with YoutubeDL({**base_opts, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        duration = info.get("duration")
        filesize = info.get("filesize") or info.get("filesize_approx")
        max_clip = min(float(dc.get("max_clip_seconds", 30)), float(dc.get("absolute_max_duration_seconds", 180)))
        if duration is not None and duration > max_clip:
            return {
                "video_id": video_id,
                "error": "video too long",
                "duration_seconds": duration,
                "status": "too_long",
            }
        if duration is not None and duration > max_dur_cfg:
            return {
                "video_id": video_id,
                "error": "video too long (config)",
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

        # Use pipeline video_id in filenames (stable; TikTok internal id may differ from search row)
        video_opts = {
            **base_opts,
            "format": f"bestvideo[height<={max_h}][ext=mp4]/bestvideo[height<={max_h}]/bestvideo/best",
            "outtmpl": os.path.join(output_folder, f"{video_id}_video.%(ext)s"),
            "merge_output_format": "mp4",
            "max_filesize": max_size,
            "noplaylist": True,
            "retries": retries,
            "fragment_retries": frag_retries,
        }
        with YoutubeDL(video_opts) as ydl:
            ydl.download([url])

        video_path = resolve_single_file(os.path.join(output_folder, f"{video_id}_video.*"))
        if not video_path or not os.path.isfile(video_path):
            return {"video_id": video_id, "error": "downloaded video file not found", "status": "download_error"}

        mp3_out = os.path.join(output_folder, f"{video_id}_audio.mp3")
        try:
            _extract_audio_mp3_from_video(video_path, mp3_out, aq)
        except Exception as mux_exc:
            return {
                "video_id": video_id,
                "error": str(mux_exc),
                "status": "download_error",
            }
        audio_path = mp3_out if os.path.isfile(mp3_out) else None
        if not audio_path:
            return {"video_id": video_id, "error": "audio mp3 not created", "status": "download_error"}

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
        err_s = str(exc).lower()
        if "match filter" in err_s or "not in a suitable format" in err_s:
            return {"video_id": video_id, "error": str(exc), "status": "too_long"}
        return {"video_id": video_id, "error": str(exc), "status": "download_error"}


def collect_skip_video_ids(cfg: dict) -> set[str]:
    """Union: metadata, pipeline log, legacy tiktok final dataset."""
    root = project_root()
    out = cfg.get("output", {})
    meta = resolve_path(root, out.get("metadata_file", "data/dataset/tiktok_metadata.jsonl"))
    plog = resolve_path(root, out.get("pipeline_log", "src/scrapers/tiktok/logs/pipeline_log.jsonl"))
    legacy = resolve_path(root, cfg.get("download", {}).get("legacy_dataset_jsonl", "data/dataset/tiktok_final_dataset.jsonl"))

    skip: set[str] = set()
    for row in load_jsonl(meta):
        vid = row.get("video_id")
        if isinstance(vid, str) and vid:
            skip.add(vid)
    for row in load_jsonl(plog):
        vid = row.get("video_id")
        st = row.get("status", row.get("download_status"))
        if isinstance(vid, str) and vid and st in ("success", "already_exists"):
            skip.add(vid)
    if legacy.is_file():
        for row in load_jsonl(legacy):
            vid = row.get("video_id")
            if isinstance(vid, str) and vid and not vid.startswith("UNKNOWN_"):
                skip.add(vid)
    else:
        logging.getLogger("tiktok_pipeline").debug("Legacy TikTok dataset file not found (optional): %s", legacy)
    return skip


def download_batch(
    video_records: list[dict[str, Any]],
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
    metadata_path: Path,
    pipeline_log_path: Path,
) -> list[dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    root = project_root()
    dc = cfg.get("download", {})
    tmp_dir = resolve_path(root, dc.get("tmp_dir", "src/scrapers/tiktok/tmp_downloads"))
    out_dir = resolve_path(root, dc.get("output_dir", "data/dataset/tiktok_videos"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cookie_file = (dc.get("cookie_file") or "").strip() or None
    cookies_from_browser = (dc.get("cookies_from_browser") or "").strip() or None

    skip = collect_skip_video_ids(cfg)
    results: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for rec in tqdm(video_records, desc="Download", unit="vid"):
        vid = rec.get("video_id")
        if not vid:
            continue
        vid = str(vid)
        page_url = rec.get("webpage_url")
        if isinstance(page_url, str) and page_url.strip().startswith("http"):
            dl_url = page_url.strip()
        else:
            dl_url = _tiktok_video_url(rec, vid)

        if vid in skip:
            vp0, ap0 = _output_media_paths_ok(out_dir, vid)
            if vp0 and ap0:
                r = {
                    "video_id": vid,
                    "title": rec.get("title", ""),
                    "status": "already_exists",
                    "behavioral_category": rec.get("behavioral_category", ""),
                    "download_time_sec": 0.0,
                    "video_path": vp0,
                    "audio_path": ap0,
                }
                results.append(r)
                stats["already_exists"] += 1
                append_jsonl(r, run_dir / "stage_4_download" / "download_log.jsonl")
                append_jsonl({**r, "stage": "download"}, pipeline_log_path)
                continue
            logging.getLogger("tiktok_pipeline").info(
                "Re-downloading %s: in skip set (metadata/log) but media missing under %s",
                vid,
                out_dir,
            )

        t0 = time.monotonic()
        dest = str(tmp_dir / vid)
        os.makedirs(dest, exist_ok=True)
        vp1, ap1 = _output_media_paths_ok(out_dir, vid)
        if vp1 and ap1:
            r = {
                "video_id": vid,
                "title": rec.get("title", ""),
                "status": "already_exists",
                "behavioral_category": rec.get("behavioral_category", ""),
                "download_time_sec": 0.0,
                "video_path": vp1,
                "audio_path": ap1,
            }
            results.append(r)
            stats["already_exists"] += 1
            append_jsonl(r, run_dir / "stage_4_download" / "download_log.jsonl")
            append_jsonl({**r, "stage": "download"}, pipeline_log_path)
            continue

        try:
            dl = download_video_assets(
                vid,
                dest,
                cfg,
                cookie_file,
                cookies_from_browser,
                webpage_url=dl_url,
            )
        except Exception as e:
            dl = {"video_id": vid, "error": str(e), "status": "download_error"}

        elapsed = time.monotonic() - t0
        st = dl.get("status", "download_error")
        if st == "success":
            stats["success"] += 1
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
                "webpage_url": dl_url,
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
        append_jsonl({**r, "stage": "download"}, pipeline_log_path)

        try:
            import shutil

            shutil.rmtree(dest, ignore_errors=True)
        except Exception:
            pass

    att = len(video_records)
    print(
        f"""
┌────────────────────────────────────────────────────────┐
│  DOWNLOAD SUMMARY (TikTok)                             │
│  Attempted:   {att:>6}                                    │
│  Success:     {stats['success']:>6} ({100*stats['success']/max(att,1):.1f}%)                            │
│  Too long:    {stats['too_long']:>6}                                    │
│  Too large:   {stats['too_large']:>6}                                    │
│  Errors:      {stats['errors']:>6}                                    │
│  Already had: {stats['already_exists']:>6}                                    │
└────────────────────────────────────────────────────────┘"""
    )

    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
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
