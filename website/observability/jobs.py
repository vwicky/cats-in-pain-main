"""Job queue metrics collected from PostgreSQL on scrape."""

from __future__ import annotations

import logging
from typing import Iterable

from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from sqlalchemy import func, select

logger = logging.getLogger(__name__)

_JOB_STATUSES = ("queued", "running", "failed", "done")


class JobQueueCollector(Collector):
    """Expose job counts by status when Prometheus scrapes /metrics."""

    def collect(self) -> Iterable[GaugeMetricFamily]:
        by_status = {s: 0 for s in _JOB_STATUSES}
        try:
            from db.models import Job
            from db.session import get_engine

            with get_engine().connect() as conn:
                rows = conn.execute(
                    select(Job.status, func.count()).group_by(Job.status)
                ).all()
            for status, count in rows:
                if status in by_status:
                    by_status[status] = int(count)
        except Exception as e:
            logger.debug("JobQueueCollector: DB unavailable: %s", e)

        fam_status = GaugeMetricFamily(
            "catpain_jobs_by_status",
            "Number of jobs in each status",
            labels=["status"],
        )
        for status, count in by_status.items():
            fam_status.add_metric([status], count)
        yield fam_status

        fam_queue = GaugeMetricFamily(
            "catpain_jobs_queue_size",
            "Number of jobs waiting in queue (status=queued)",
        )
        fam_queue.add_metric([], by_status["queued"])
        yield fam_queue
