from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import OperationalError

from backend.routers import jobs as jobs_router
from backend.settings import (
    data_dir_writable,
    ffmpeg_available,
    get_settings,
    torch_device_probe,
    try_import_pipeline,
)
from db.session import configure_engine, get_engine, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_engine(settings.database_url)
    try:
        init_db()
    except OperationalError as e:
        logger.error(
            "Database unavailable (startup needs PostgreSQL). Original error: %s\n\n"
            "Fix:\n"
            "  • Start Postgres (e.g. brew services start postgresql@16).\n"
            "  • If you see role \"postgres\" does not exist: Homebrew uses your macOS "
            "username, not postgres. Remove DATABASE_URL from website/.env to auto-pick "
            "your user, or set:\n"
            "      DATABASE_URL=postgresql+psycopg://$(whoami)@localhost:5432/catpain_web\n"
            "  • Docker DB user: postgresql+psycopg://postgres:postgres@localhost:5432/catpain_web\n"
            "  • Create DB: createdb catpain_web\n"
            "More: website/README.md\n",
            e,
        )
        raise
    yield


app = FastAPI(title="Cat Pain Web API", lifespan=lifespan)
_origins = [
    o.strip()
    for o in os.getenv(
        "FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router.router)


@app.get("/health")
def health():
    settings = get_settings()
    db_ok = False
    try:
        from sqlalchemy import text

        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_err = str(e)
    else:
        db_err = None

    pipe_import = try_import_pipeline(settings.repo_root)
    ff = ffmpeg_available()
    data_ok = data_dir_writable(settings.data_dir)
    devices = torch_device_probe()

    overall = db_ok and pipe_import and ff and data_ok
    return {
        "ok": overall,
        "database": {"ok": db_ok, "error": db_err},
        "pipeline_import": pipe_import,
        "ffmpeg": ff,
        "data_dir_writable": data_ok,
        "torch_devices": devices,
    }


@app.get("/")
def root():
    return {"service": "cat-pain-web", "docs": "/docs"}
