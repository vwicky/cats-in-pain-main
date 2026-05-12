"""
Extract 16 kHz mono audio from ``.mp4`` with ffmpeg and split into fixed-length chunks.

Mirrors the workflow in ``audio_preclassifier_v2/notebooks/inference_verify.ipynb`` (Step 2),
but normalizes to **16 kHz mono** for YAMNet.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def extract_audio_wav_16k_mono(video_path: Path, out_dir: Path) -> Path | None:
    """
    Extract the audio track to a mono **16 kHz** PCM WAV using ffmpeg.

    Returns the written path, or ``None`` if ffmpeg fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_audio.wav"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        str(out_path),
        "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg failed for %s: %s",
            video_path.name,
            (proc.stderr or proc.stdout or "").strip(),
        )
        return None
    return out_path


def chunk_audio(
    audio_path: Path,
    chunk_sec: float,
    overlap_sec: float,
    *,
    source_video_name: str | None = None,
) -> list[dict]:
    """
    Split audio into chunks using pydub (same logic as the v2 verification notebook).

    Each chunk dict includes an in-memory ``AudioSegment`` under ``audio_segment``.
    If audio is shorter than ``chunk_sec``, it is still returned as a single chunk.
    """
    seg = AudioSegment.from_file(str(audio_path))
    chunk_ms = int(chunk_sec * 1000)
    overlap_ms = int(overlap_sec * 1000)
    step_ms = max(1, chunk_ms - overlap_ms)

    video_stem = audio_path.stem
    if video_stem.endswith("_audio"):
        video_stem = video_stem[: -len("_audio")]

    chunks: list[dict] = []
    start_ms = 0
    idx = 0
    total_ms = len(seg)

    while start_ms < total_ms:
        end_ms = min(start_ms + chunk_ms, total_ms)
        piece = seg[start_ms:end_ms]
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        dur_sec = end_sec - start_sec
        chunk_id = f"{video_stem}_chunk_{idx:03d}"
        row = {
            "chunk_id": chunk_id,
            "audio_path": str(audio_path.resolve()),
            "video_stem": video_stem,
            "chunk_index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": dur_sec,
            "audio_segment": piece,
        }
        if source_video_name is not None:
            row["source_video"] = source_video_name
        chunks.append(row)
        idx += 1
        if end_ms >= total_ms:
            break
        start_ms += step_ms

    return chunks


def pydub_segment_to_waveform_16k_mono(seg: AudioSegment) -> torch.Tensor:
    """
    Convert a pydub ``AudioSegment`` to float32 tensor ``[1, T]``, mono, ~16 kHz.

    Values are in ``[-1, 1]``. The segment is re-encoded to mono / 16 kHz if needed.
    """
    seg = seg.set_channels(1).set_frame_rate(16000)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    max_val = float(2 ** (8 * seg.sample_width - 1))
    y = samples / max_val
    y = np.clip(y, -1.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
