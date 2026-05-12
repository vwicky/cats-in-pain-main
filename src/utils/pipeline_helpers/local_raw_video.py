"""
Load pre-staged video/audio from a local directory (bypass yt-dlp).

Expected files under ``local_dir``:

  - ``{video_id}_video.mp4`` and ``{video_id}_audio.mp3`` (preferred), or
  - ``{video_id}.mp4`` only — audio is extracted with ffmpeg to ``{video_id}_audio.mp3``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any


def _extract_audio_mp3_ffmpeg(video_path: str, audio_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-b:a",
            "192k",
            audio_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def download_video_assets_from_local_dir(
    video_id: str,
    output_folder: str,
    local_dir: str,
) -> dict[str, Any]:
    """
    Copy staged files into ``output_folder`` (same layout as yt-dlp output).
    Title is set to ``video_id`` (no YouTube metadata).
    """
    os.makedirs(output_folder, exist_ok=True)
    src = os.path.abspath(os.path.expanduser(local_dir))
    if not os.path.isdir(src):
        return {"video_id": video_id, "error": f"Local video dir not found: {src}"}

    video_pair = f"{video_id}_video.mp4"
    audio_pair = f"{video_id}_audio.mp3"
    single = f"{video_id}.mp4"

    p_video = os.path.join(src, video_pair)
    p_audio = os.path.join(src, audio_pair)
    p_single = os.path.join(src, single)

    video_dest = os.path.join(output_folder, video_pair)
    audio_dest = os.path.join(output_folder, audio_pair)

    if os.path.isfile(p_video) and os.path.isfile(p_audio):
        shutil.copy2(p_video, video_dest)
        shutil.copy2(p_audio, audio_dest)
        return {
            "video_id": video_id,
            "title": video_id,
            "video_path": video_dest,
            "audio_path": audio_dest,
        }

    if os.path.isfile(p_single):
        shutil.copy2(p_single, video_dest)
        _extract_audio_mp3_ffmpeg(video_dest, audio_dest)
        return {
            "video_id": video_id,
            "title": video_id,
            "video_path": video_dest,
            "audio_path": audio_dest,
        }

    tried = [p_video, p_audio, p_single]
    return {
        "video_id": video_id,
        "error": "Local ingest: no matching files. Tried: " + "; ".join(tried),
    }
