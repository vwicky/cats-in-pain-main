"""
Bridge to data_pipeline_v2 process stage (YOLO + audio).

Adds ``data_pipeline_v2`` to ``sys.path`` and exposes ``rows_for_yolo_process``
compatible with ``download_log.jsonl`` rows from this scraper.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_DM_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _DM_ROOT.parents[2]
_V2_ROOT = _REPO_ROOT / "src" / "scrapers" / "data_pipeline_v2"
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from src.downloader import resolve_single_file  # noqa: E402
from src.utils import project_root, resolve_path  # noqa: E402


def rows_for_yolo_process(dl_results: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Same logic as ``data_pipeline_v2.src.pipeline._rows_for_yolo_process``:
    keep ``success`` / ``already_exists`` rows with resolvable video+audio paths under
    ``download.output_dir``.
    """
    root = project_root()
    dc = cfg.get("download", {})
    out_dir = resolve_path(root, dc.get("output_dir", "src/scrapers/dailymotion/output/videos"))
    by_vid: dict[str, dict[str, Any]] = {}
    for d in dl_results:
        vid = d.get("video_id")
        if not vid:
            continue
        st = d.get("status")
        if st not in ("success", "already_exists"):
            continue
        vp = d.get("video_path") or resolve_single_file(str(out_dir / f"{vid}_video.*"))
        ap = d.get("audio_path") or resolve_single_file(str(out_dir / f"{vid}_audio.*"))
        if not vp or not ap:
            continue
        by_vid[str(vid)] = {**d, "video_path": vp, "audio_path": ap}
    return list(by_vid.values())


def import_process_stack():
    """Late import of processor stack (after sys.path is set)."""
    from src.audio_classifier import AudioClassifier
    from src.processor import process_batch

    return AudioClassifier, process_batch
