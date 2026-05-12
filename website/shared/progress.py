from __future__ import annotations

from typing import Any

STAGE_ORDER: list[str] = [
    "queued",
    "extract_audio",
    "yamnet",
    "routing",
    "audio_branch",
    "video_branch",
    "aggregating",
    "done",
]

_STAGE_TO_INDEX: dict[str, int] = {s: i for i, s in enumerate(STAGE_ORDER)}


def stage_index(stage: str) -> int:
    return _STAGE_TO_INDEX.get(stage, -1)


def apply_monotonic_progress(
    current: dict[str, Any],
    *,
    stage: str | None = None,
    window_idx: int | None = None,
    n_windows: int | None = None,
    percent: float | None = None,
) -> dict[str, Any]:
    """Return updated progress; never decrease stage rank or window_idx."""
    out = dict(current) if current else {}
    if stage is not None:
        new_i = stage_index(stage)
        old_i = stage_index(str(out.get("stage", "queued")))
        if new_i >= old_i:
            out["stage"] = stage
    if window_idx is not None:
        old_w = int(out.get("window_idx", 0) or 0)
        out["window_idx"] = max(old_w, int(window_idx))
    if n_windows is not None:
        existing = out.get("n_windows")
        if existing is None:
            out["n_windows"] = n_windows
        else:
            out["n_windows"] = max(int(existing), int(n_windows))
    if percent is not None:
        old_p = float(out.get("percent", 0) or 0)
        out["percent"] = min(100.0, max(old_p, float(percent)))
    return out


def slack_line_event(
    *,
    stage: str,
    window: int | None,
    message: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    import datetime as _dt

    return {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "stage": stage,
        "window": window,
        "message": message,
        "job_id": job_id,
    }
