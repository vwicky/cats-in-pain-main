"""yt-dlp download + ffmpeg MP3 extraction."""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import time
from pathlib import Path

import yt_dlp
from tqdm import tqdm

from config import AUDIO_DIR, METADATA_DIR, SLEEP_BETWEEN_DOWNLOADS, VIDEO_DIR

from utils import append_jsonl


def resolve_single_file(glob_pattern: str) -> str | None:
    matches = glob.glob(glob_pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _find_existing_video(video_id: str) -> str | None:
    matches = glob.glob(os.path.join(VIDEO_DIR, f"{video_id}_video.*"))
    valid = [p for p in matches if os.path.isfile(p) and os.path.getsize(p) > 0]
    return valid[0] if valid else None


def _extract_audio_mp3(video_path: str, audio_path: str) -> bool:
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            audio_path,
            "-y",
            "-loglevel",
            "error",
        ],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0


def download_video(video: dict, download_cfg: dict | None = None) -> tuple[str | None, str | None]:
    """
    Download a Dailymotion video and extract MP3.
    Returns (video_path, audio_path) on success.
    """
    video_id = video["video_id"]
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}_audio.mp3")

    existing_video = _find_existing_video(video_id)
    if existing_video and os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0:
        return existing_video, audio_path

    outtmpl = os.path.join(VIDEO_DIR, f"{video_id}_video.%(ext)s")
    ydl_opts: dict = {
        "outtmpl": outtmpl,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }
    max_dur = (download_cfg or {}).get("max_duration_seconds")
    if max_dur is not None:
        max_f = float(max_dur)

        def _match_filter(info_dict: dict, *, cap: float = max_f) -> str | None:
            d = info_dict.get("duration")
            if d is not None and float(d) > cap:
                return f"duration>{cap}s"
            return None

        ydl_opts["match_filter"] = _match_filter

    page_url = video.get("url") or video.get("webpage_url")
    if not page_url:
        logging.getLogger("dailymotion_pipeline").error("No URL for %s", video_id)
        return None, None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([page_url])
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)
    except Exception as e:
        logging.getLogger("dailymotion_pipeline").warning("Download failed %s: %s", video_id, e)
        return None, None

    video_path = _find_existing_video(video_id)
    if not video_path:
        logging.getLogger("dailymotion_pipeline").warning("Output file missing for %s", video_id)
        return None, None

    if not _extract_audio_mp3(video_path, audio_path):
        logging.getLogger("dailymotion_pipeline").warning("Audio extract failed for %s", video_id)
        try:
            os.remove(video_path)
        except OSError:
            pass
        return None, None

    return video_path, audio_path


def download_batch(
    kept_rows: list[dict],
    logger: logging.Logger,
    run_dir: Path,
    metadata_dir: str = METADATA_DIR,
    download_cfg: dict | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """
    Download each GPT-kept row; append lines to ``stage_4_download/download_log.jsonl``
    (pipeline v2 style). Returns (log_rows, counts dict).
    """
    os.makedirs(metadata_dir, exist_ok=True)
    log_path = run_dir / "stage_4_download" / "download_log.jsonl"
    if log_path.is_file():
        log_path.unlink()

    log_rows: list[dict] = []
    n_ok = n_fail = n_skip = 0

    logger.info("Stage 4 download: %d rows", len(kept_rows))

    for row in tqdm(kept_rows, desc="download", unit="vid", mininterval=0.5):
        vid = row.get("video_id")
        meta_path = os.path.join(metadata_dir, f"{vid}.json")

        if os.path.isfile(meta_path) and os.path.getsize(meta_path) > 0:
            vp = _find_existing_video(vid) or resolve_single_file(
                os.path.join(VIDEO_DIR, f"{vid}_video.*")
            )
            ap = resolve_single_file(os.path.join(VIDEO_DIR, f"{vid}_audio.*"))
            rec = {
                "video_id": vid,
                "status": "already_exists",
                "video_path": vp,
                "audio_path": ap,
                "error": None,
                "behavioral_category": row.get("behavioral_category", "Unknown"),
                "title": row.get("title", ""),
            }
            append_jsonl(rec, log_path)
            log_rows.append(rec)
            n_skip += 1
            logger.info("Download: skip (metadata exists) %s", vid)
            continue

        vp, ap = download_video(row, download_cfg)
        if vp and ap:
            rec = {
                "video_id": vid,
                "status": "success",
                "video_path": vp,
                "audio_path": ap,
                "error": None,
                "behavioral_category": row.get("behavioral_category", "Unknown"),
                "title": row.get("title", ""),
            }
            n_ok += 1
        else:
            rec = {
                "video_id": vid,
                "status": "download_error",
                "video_path": None,
                "audio_path": None,
                "error": "yt_dlp_or_ffmpeg_failed",
                "behavioral_category": row.get("behavioral_category", "Unknown"),
                "title": row.get("title", ""),
            }
            n_fail += 1
        append_jsonl(rec, log_path)
        log_rows.append(rec)

    box = f"""
┌────────────────────────────────────────────────────────┐
│  DOWNLOAD SUMMARY (Dailymotion)                        │
│  Input:      {len(kept_rows):>6} GPT-kept rows                    │
│  Success:    {n_ok:>6}                                        │
│  Skipped:    {n_skip:>6} (metadata already on disk)              │
│  Failed:     {n_fail:>6}                                        │
└────────────────────────────────────────────────────────┘"""
    print(box)
    logger.info(box)

    counts = {"success": n_ok, "skipped": n_skip, "failed": n_fail}
    return log_rows, counts
