"""Worker lifecycle metrics."""

from __future__ import annotations

import re
import time

from observability.definitions import (
    GPU_OOM_TOTAL,
    WORKER_BUSY,
    WORKER_CUDA_ERRORS_TOTAL,
    WORKER_JOBS_FAILED_TOTAL,
    WORKER_JOBS_PROCESSED_TOTAL,
    WORKER_LAST_HEARTBEAT_TIMESTAMP,
)
from observability.registry import metrics_enabled

_CUDA_RE = re.compile(r"cuda|cudart|cudnn", re.I)
_OOM_RE = re.compile(r"out of memory|oom|cuda error", re.I)


def heartbeat() -> None:
    if metrics_enabled():
        WORKER_LAST_HEARTBEAT_TIMESTAMP.set(time.time())


def set_busy(busy: bool) -> None:
    if metrics_enabled():
        WORKER_BUSY.set(1 if busy else 0)


def record_job_success() -> None:
    if metrics_enabled():
        WORKER_JOBS_PROCESSED_TOTAL.labels(outcome="success").inc()


def record_job_failure(error_type: str) -> None:
    if metrics_enabled():
        WORKER_JOBS_PROCESSED_TOTAL.labels(outcome="failed").inc()
        WORKER_JOBS_FAILED_TOTAL.labels(error_type=error_type or "unknown").inc()


def scan_pipeline_output(text: str) -> None:
    """Heuristic CUDA/OOM detection from subprocess stderr/stdout."""
    if not metrics_enabled() or not text:
        return
    if _OOM_RE.search(text):
        GPU_OOM_TOTAL.inc()
    if _CUDA_RE.search(text):
        WORKER_CUDA_ERRORS_TOTAL.inc()
