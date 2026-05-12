"""
Artifact I/O helpers for the inference pipeline.

Creates a timestamped output directory and provides helpers to copy/save
every artifact produced during a pipeline run.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_run_dir(base_dir: str | Path, video_stem: str) -> Path:
    """Create and return ``<base_dir>/<timestamp>_<video_stem>/``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / f"{ts}_{video_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_original_video(video_path: str | Path, run_dir: Path) -> Path:
    """Copy (or hard-link) the original video into the run directory."""
    src = Path(video_path).resolve()
    dst = run_dir / f"original_video{src.suffix}"
    if dst.resolve() == src:
        return dst
    shutil.copy2(src, dst)
    return dst


def save_json(obj: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    """Fallback serialiser for numpy scalars and similar."""
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)
