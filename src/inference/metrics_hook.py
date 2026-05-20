"""Optional Prometheus hooks when website observability is on PYTHONPATH."""

from __future__ import annotations

import os
from typing import Callable


def metrics_active() -> bool:
    return os.environ.get("CATPAIN_METRICS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def step_callback() -> Callable[[str, float, BaseException | None], None] | None:
    if not metrics_active():
        return None
    try:
        from observability.pipeline import make_step_callback

        return make_step_callback()
    except ImportError:
        return None


def record_branch(branch: str) -> None:
    if not metrics_active():
        return
    try:
        from observability.pipeline import record_branch as _record

        _record(branch)
    except ImportError:
        pass


def record_windows(n: int) -> None:
    if not metrics_active():
        return
    try:
        from observability.pipeline import record_windows as _record

        _record(n)
    except ImportError:
        pass


def observe_pipeline_total(seconds: float, exc: BaseException | None = None) -> None:
    if not metrics_active():
        return
    try:
        from observability.pipeline import observe_pipeline_total as _observe

        _observe(seconds, exc)
    except ImportError:
        pass
