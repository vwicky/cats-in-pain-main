"""Prometheus metrics for Cat Pain web API and inference worker."""

from observability.registry import REGISTRY, metrics_enabled, setup_multiproc_dir

__all__ = ["REGISTRY", "metrics_enabled", "setup_multiproc_dir"]
