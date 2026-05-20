"""ML pipeline metric helpers (safe to import when metrics disabled)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from observability.definitions import (
    BRANCH_ROUTED_TOTAL,
    GPU_OOM_TOTAL,
    INFERENCE_DURATION_SECONDS,
    INFERENCE_FAILURES_TOTAL,
    WINDOWS_PROCESSED_TOTAL,
)
from observability.labels import step_to_stage
from observability.registry import metrics_enabled

if TYPE_CHECKING:
    pass


def is_gpu_oom(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "cuda error" in msg


def observe_step(step_name: str, seconds: float, exc: BaseException | None = None) -> None:
    if not metrics_enabled():
        return
    stage = step_to_stage(step_name)
    if stage is None:
        return
    INFERENCE_DURATION_SECONDS.labels(stage=stage).observe(seconds)
    if exc is not None:
        reason = "oom" if is_gpu_oom(exc) else type(exc).__name__
        INFERENCE_FAILURES_TOTAL.labels(stage=stage, reason=reason).inc()
        if is_gpu_oom(exc):
            GPU_OOM_TOTAL.inc()


def observe_pipeline_total(seconds: float, exc: BaseException | None = None) -> None:
    observe_step("pipeline_total", seconds, exc)


def record_branch(branch: str) -> None:
    if metrics_enabled():
        BRANCH_ROUTED_TOTAL.labels(branch=branch).inc()


def record_windows(n: int) -> None:
    if metrics_enabled() and n > 0:
        WINDOWS_PROCESSED_TOTAL.inc(n)


def make_step_callback():
    """Return on_step callback for StepTimer when metrics enabled."""

    def _cb(name: str, seconds: float, exc: BaseException | None) -> None:
        observe_step(name, seconds, exc)

    return _cb if metrics_enabled() else None


def metrics_env_for_subprocess(website_root: str, src_path: str) -> dict[str, str]:
    """Extra environment variables for pipeline subprocess."""
    env = dict(os.environ)
    sep = os.pathsep
    existing = env.get("PYTHONPATH", "")
    paths = [src_path, website_root]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = sep.join(paths)
    if metrics_enabled():
        from observability.registry import multiproc_dir, setup_multiproc_dir

        setup_multiproc_dir()
        env["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir()
        env["CATPAIN_METRICS_ENABLED"] = "1"
    return env
