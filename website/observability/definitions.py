"""Prometheus metric definitions (single source of truth)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

INFERENCE_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)

HTTP_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# HTTP
HTTP_REQUESTS_TOTAL = Counter(
    "catpain_http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "catpain_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "route"],
    buckets=HTTP_DURATION_BUCKETS,
)

HTTP_REQUESTS_IN_FLIGHT = Gauge(
    "catpain_http_requests_in_flight",
    "HTTP requests currently being handled",
)

# Job queue gauges are exposed only via JobQueueCollector (see jobs.py).

# Worker
WORKER_LAST_HEARTBEAT_TIMESTAMP = Gauge(
    "catpain_worker_last_heartbeat_timestamp",
    "Unix timestamp of last worker poll-loop heartbeat",
)

WORKER_JOBS_PROCESSED_TOTAL = Counter(
    "catpain_worker_jobs_processed_total",
    "Jobs finished by worker",
    ["outcome"],
)

WORKER_JOBS_FAILED_TOTAL = Counter(
    "catpain_worker_jobs_failed_total",
    "Jobs that failed in worker",
    ["error_type"],
)

WORKER_BUSY = Gauge(
    "catpain_worker_busy",
    "1 if worker is processing a job, else 0",
)

WORKER_CUDA_ERRORS_TOTAL = Counter(
    "catpain_worker_cuda_errors_total",
    "CUDA-related failures detected in pipeline output",
)

WORKER_RETRIES_TOTAL = Counter(
    "catpain_worker_retries_total",
    "Job retry attempts (reserved for future use)",
)

# ML inference
INFERENCE_DURATION_SECONDS = Histogram(
    "catpain_inference_duration_seconds",
    "Wall-clock seconds per inference stage",
    ["stage"],
    buckets=INFERENCE_DURATION_BUCKETS,
)

INFERENCE_FAILURES_TOTAL = Counter(
    "catpain_inference_failures_total",
    "Inference step failures",
    ["stage", "reason"],
)

GPU_OOM_TOTAL = Counter(
    "catpain_gpu_oom_total",
    "GPU out-of-memory events",
)

WINDOWS_PROCESSED_TOTAL = Counter(
    "catpain_windows_processed_total",
    "Sliding-window clips processed",
)

BRANCH_ROUTED_TOTAL = Counter(
    "catpain_branch_routed_total",
    "Pipeline branch routing decisions",
    ["branch"],
)

# System
PROCESS_CPU_PERCENT = Gauge(
    "catpain_process_cpu_percent",
    "Process CPU utilization percent",
    ["service"],
)

PROCESS_MEMORY_BYTES = Gauge(
    "catpain_process_memory_bytes",
    "Process resident memory in bytes",
    ["service"],
)

GPU_UTILIZATION_PERCENT = Gauge(
    "catpain_gpu_utilization_percent",
    "GPU utilization percent (NVML)",
    ["gpu_index"],
)

GPU_MEMORY_USED_BYTES = Gauge(
    "catpain_gpu_memory_used_bytes",
    "GPU memory used in bytes (NVML)",
    ["gpu_index"],
)
