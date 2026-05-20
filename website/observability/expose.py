"""Expose Prometheus metrics over HTTP (worker process)."""

from __future__ import annotations

import os

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client import multiprocess

from observability.jobs import JobQueueCollector
from observability.registry import metrics_enabled, setup_multiproc_dir

_server_started = False


def worker_metrics_port() -> int:
    return int(os.environ.get("WORKER_METRICS_PORT", "9101"))


def start_worker_metrics_server() -> int | None:
    """Start Prometheus HTTP server aggregating worker + pipeline subprocess metrics."""
    global _server_started
    if not metrics_enabled() or _server_started:
        return None

    setup_multiproc_dir()
    port = worker_metrics_port()

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    registry.register(JobQueueCollector())
    start_http_server(port, registry=registry)
    _server_started = True
    return port
