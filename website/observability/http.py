"""FastAPI HTTP metrics middleware and /metrics mount."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import make_asgi_app
from prometheus_client.registry import CollectorRegistry
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from observability.definitions import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_IN_FLIGHT,
    HTTP_REQUESTS_TOTAL,
)
from observability.jobs import JobQueueCollector
from observability.registry import REGISTRY, metrics_enabled


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    for r in request.app.routes:
        match, _ = r.matches(request.scope)
        if match == Match.FULL and hasattr(r, "path"):
            return r.path
    return request.url.path


def _should_skip_latency(path: str) -> bool:
    return path == "/metrics"


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if path == "/metrics" or not metrics_enabled():
            return await call_next(request)

        method = request.method
        route = _route_template(request)
        HTTP_REQUESTS_IN_FLIGHT.inc()
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.perf_counter() - start
            HTTP_REQUESTS_IN_FLIGHT.dec()
            HTTP_REQUESTS_TOTAL.labels(
                method=method, route=route, status=status
            ).inc()
            if not _should_skip_latency(path):
                HTTP_REQUEST_DURATION_SECONDS.labels(
                    method=method, route=route
                ).observe(elapsed)


def install_metrics(app: FastAPI, *, registry: CollectorRegistry | None = None) -> None:
    """Register middleware, job collector, and /metrics ASGI app."""
    reg = registry or REGISTRY
    reg.register(JobQueueCollector())
    app.add_middleware(PrometheusMiddleware)
    metrics_app = make_asgi_app(registry=reg)
    app.mount("/metrics", metrics_app)
