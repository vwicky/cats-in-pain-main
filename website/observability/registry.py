"""Prometheus registry and multiprocess setup for API + worker + pipeline subprocess."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from prometheus_client import CollectorRegistry, REGISTRY as _DEFAULT_REGISTRY
from prometheus_client import multiprocess

REGISTRY: CollectorRegistry = _DEFAULT_REGISTRY

_DEFAULT_MULTIPROC_DIR = "/tmp/catpain_prom_multiproc"


def metrics_enabled() -> bool:
    return os.environ.get("CATPAIN_METRICS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def multiproc_dir() -> str:
    return os.environ.get("PROMETHEUS_MULTIPROC_DIR", _DEFAULT_MULTIPROC_DIR)


def setup_multiproc_dir(*, clean: bool = False) -> str:
    """Ensure PROMETHEUS_MULTIPROC_DIR exists and is exported."""
    path = multiproc_dir()
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = path
    p = Path(path)
    if clean and p.exists():
        shutil.rmtree(path, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return path


def mark_subprocess_dead(pid: int) -> None:
    if pid > 0:
        multiprocess.mark_process_dead(pid)


def child_registry() -> CollectorRegistry:
    """Registry for pipeline subprocess (multiprocess mode)."""
    return REGISTRY
