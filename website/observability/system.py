"""Process and GPU resource gauges (psutil + optional NVML)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import psutil

from observability.definitions import (
    GPU_MEMORY_USED_BYTES,
    GPU_UTILIZATION_PERCENT,
    PROCESS_CPU_PERCENT,
    PROCESS_MEMORY_BYTES,
)
from observability.registry import metrics_enabled

logger = logging.getLogger(__name__)

_stop_events: dict[str, threading.Event] = {}
_threads: dict[str, threading.Thread] = {}


def _update_process_gauges(service: str) -> None:
    proc = psutil.Process()
    PROCESS_CPU_PERCENT.labels(service=service).set(proc.cpu_percent(interval=None))
    PROCESS_MEMORY_BYTES.labels(service=service).set(proc.memory_info().rss)


def _update_gpu_gauges() -> None:
    try:
        import pynvml
    except ImportError:
        return
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPU_UTILIZATION_PERCENT.labels(gpu_index=str(i)).set(float(util.gpu))
            GPU_MEMORY_USED_BYTES.labels(gpu_index=str(i)).set(float(mem.used))
    except Exception as e:
        logger.debug("NVML update skipped: %s", e)


def _collector_loop(service: str, interval: float, stop: threading.Event) -> None:
    # Prime CPU percent measurement
    try:
        psutil.Process().cpu_percent(interval=None)
    except Exception:
        pass
    while not stop.wait(interval):
        if not metrics_enabled():
            continue
        try:
            _update_process_gauges(service)
            if service == "worker":
                _update_gpu_gauges()
        except Exception as e:
            logger.debug("Resource collector error: %s", e)


def start_resource_collector(
    *,
    service: str,
    interval: float = 15.0,
) -> Callable[[], None]:
    """Start background gauge updates; returns stop function."""
    if not metrics_enabled():
        return lambda: None

    existing = _stop_events.get(service)
    if existing is not None:
        return lambda: _stop_resource_collector(service)

    stop = threading.Event()
    thread = threading.Thread(
        target=_collector_loop,
        args=(service, interval, stop),
        name=f"catpain-metrics-{service}",
        daemon=True,
    )
    _stop_events[service] = stop
    _threads[service] = thread
    thread.start()
    return lambda: _stop_resource_collector(service)


def _stop_resource_collector(service: str) -> None:
    stop = _stop_events.pop(service, None)
    _threads.pop(service, None)
    if stop is not None:
        stop.set()
