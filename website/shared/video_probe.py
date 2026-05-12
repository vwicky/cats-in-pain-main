from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any


def probe_video_duration_sec(video_path: Path) -> float:
    """Read video duration via ffprobe (same contract as inference pipeline)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe duration failed (code {result.returncode}):\n{result.stderr}"
        )
    raw = result.stdout.strip()
    try:
        dur = float(raw)
    except ValueError as e:
        raise RuntimeError(f"Could not parse ffprobe duration from {raw!r}") from e
    if not math.isfinite(dur) or dur <= 0:
        raise RuntimeError(f"Invalid video duration: {dur}")
    return dur


def build_sliding_windows(
    duration_sec: float,
    window_sec: float,
    step_sec: float,
) -> list[tuple[float, float]]:
    """Match src/inference/pipeline.py _build_sliding_windows."""
    if duration_sec <= 0:
        return []
    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be > 0")

    windows: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_sec - 1e-9:
        end = min(t + window_sec, duration_sec)
        windows.append((t, end))
        if end >= duration_sec - 1e-9:
            break
        t += step_sec
    return windows


def count_split_windows(duration_sec: float, window_sec: float, step_sec: float) -> int:
    return len(build_sliding_windows(duration_sec, window_sec, step_sec))
