"""
Light-weight metrics for an extracted clip WAV (16 kHz mono from ffmpeg).

Separate from YAMNet P(cat): ``likely_silent`` flags very low-level waveforms so
the UI can distinguish "almost no audio energy" from "audio present, not cat-like".
"""

from __future__ import annotations

import logging
import math
import struct
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Normalized RMS below this is treated as likely silent / unusable for listening.
_SILENT_RMS = 0.0012


def probe_wav_clip(wav_path: Path) -> dict[str, Any]:
    """Return duration, RMS (normalized ~[0, 1] for int16), and ``likely_silent``."""
    empty: dict[str, Any] = {
        "duration_sec": 0.0,
        "rms": 0.0,
        "likely_silent": True,
    }
    path = Path(wav_path)
    if not path.is_file():
        return empty

    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames <= 0 or frame_rate <= 0 or sample_width <= 0:
                return empty
            raw = wf.readframes(n_frames)
    except (OSError, wave.Error) as exc:
        logger.warning("probe_wav_clip: could not read %s: %s", path, exc)
        return empty

    duration_sec = n_frames / float(frame_rate)

    if sample_width != 2:
        logger.warning(
            "probe_wav_clip: expected 16-bit wav, got width=%s at %s",
            sample_width,
            path,
        )
        rms_norm = 0.0
    elif n_channels < 1:
        rms_norm = 0.0
    else:
        n_samples = n_frames * n_channels
        if len(raw) < n_samples * sample_width:
            return empty
        fmt = "<" + "h" * n_samples
        try:
            samples = struct.unpack(fmt, raw[: n_samples * sample_width])
        except struct.error:
            logger.warning("probe_wav_clip: unpack failed for %s", path)
            return empty
        acc = 0.0
        for s in samples:
            acc += float(s) * float(s)
        rms = math.sqrt(acc / len(samples)) if samples else 0.0
        rms_norm = min(rms / 32768.0, 1.0)

    likely_silent = rms_norm < _SILENT_RMS
    return {
        "duration_sec": round(duration_sec, 4),
        "rms": round(rms_norm, 6),
        "likely_silent": likely_silent,
    }
