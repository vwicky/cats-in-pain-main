from __future__ import annotations

import getpass
import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

_engine = None
_session_maker = None


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    user = getpass.getuser()
    return f"postgresql+psycopg://{user}@localhost:5432/catpain_web"


def configure_engine(url: str | None = None) -> None:
    global _engine, _session_maker
    if url:
        os.environ["DATABASE_URL"] = url
    _engine = create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )
    _session_maker = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_engine():
    global _engine
    if _engine is None:
        configure_engine()
    return _engine


def get_session_maker():
    global _session_maker
    if _session_maker is None:
        configure_engine()
    return _session_maker


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def claim_next_queued_job(session: Session):
    """
    Atomically claim the oldest queued job (PostgreSQL SKIP LOCKED).
    Returns Job instance or None.
    """
    from db.models import Job

    job_id = session.execute(
        text(
            """
            UPDATE jobs
            SET status = 'running', started_at = NOW()
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id
            """
        )
    ).scalar_one_or_none()
    session.commit()
    if job_id is None:
        return None
    return session.get(Job, job_id)
