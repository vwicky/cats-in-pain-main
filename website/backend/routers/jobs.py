from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.services.job_paths import (
    job_link_dir,
    resolve_run_dir_for_job,
    safe_resolve_under_run_dir,
    write_run_reference,
)
from backend.settings import Settings, get_settings
from db.models import Job
from db.session import get_session_maker
from shared.multicat_params import VALID_MULTICAT_SUMMARY_STRATEGIES, parse_form_bool
from shared.normalize import categorize_run_dir_artifacts, normalize_pipeline_result
from shared.progress import apply_monotonic_progress, slack_line_event
from shared.video_probe import probe_video_duration_sec

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_db():
    sm = get_session_maker()
    db = sm()
    try:
        yield db
    finally:
        db.close()


def _append_event(progress: dict, ev: dict, limit: int = 100) -> dict:
    events = list(progress.get("events") or [])
    events.append(ev)
    return {**progress, "events": events[-limit:]}


@router.post("")
async def create_job(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    mode: str = Form(...),
    device: str = Form("auto"),
    cat_threshold: float = Form(0.5),
    split_window_sec: float = Form(0.0),
    split_step_sec: float = Form(0.0),
    multicat_video_only: str = Form("false"),
    multicat_max_cats: str | None = Form(None),
    multicat_min_track_coverage: str | None = Form(None),
    multicat_decision_threshold: str | None = Form(None),
    multicat_summary_strategy: str | None = Form(None),
    video_path: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    if mode not in ("upload", "local_path"):
        raise HTTPException(400, "mode must be upload or local_path")

    mvo = parse_form_bool(multicat_video_only)
    logger.info(
        "MULTICAT DEBUG: raw_form=%r parsed=%s", multicat_video_only, mvo
    )
    if mvo and not settings.enable_multicat_video:
        raise HTTPException(
            400,
            "multicat video mode is disabled (set ENABLE_MULTICAT_VIDEO=1)",
        )

    try:
        m_max = (
            int(multicat_max_cats)
            if multicat_max_cats not in (None, "")
            else settings.multicat_max_cats_default
        )
    except ValueError as e:
        raise HTTPException(400, "invalid multicat_max_cats") from e
    if m_max < 1 or m_max > 32:
        raise HTTPException(400, "multicat_max_cats must be between 1 and 32")

    try:
        m_cov = (
            float(multicat_min_track_coverage)
            if multicat_min_track_coverage not in (None, "")
            else settings.multicat_min_track_coverage_default
        )
    except ValueError as e:
        raise HTTPException(400, "invalid multicat_min_track_coverage") from e
    if m_cov <= 0 or m_cov > 1.0:
        raise HTTPException(
            400, "multicat_min_track_coverage must be in (0, 1]"
        )

    try:
        m_pain_th = (
            float(multicat_decision_threshold)
            if multicat_decision_threshold not in (None, "")
            else settings.multicat_decision_threshold_default
        )
    except ValueError as e:
        raise HTTPException(400, "invalid multicat_decision_threshold") from e
    if m_pain_th <= 0 or m_pain_th >= 1.0:
        raise HTTPException(
            400, "multicat_decision_threshold must be in (0, 1)"
        )

    m_strat = (
        (multicat_summary_strategy or "").strip()
        or settings.multicat_summary_strategy_default
    )
    if m_strat not in VALID_MULTICAT_SUMMARY_STRATEGIES:
        raise HTTPException(400, f"invalid multicat_summary_strategy: {m_strat!r}")

    job = Job(
        id=uuid.uuid4(),
        status="queued",
        input_video_path="",
        params={
            "device": device,
            "cat_threshold": cat_threshold,
            "split_window_sec": split_window_sec,
            "split_step_sec": split_step_sec,
            "mode": mode,
            "multicat_video_only": mvo,
            "multicat_max_cats": m_max,
            "multicat_min_track_coverage": m_cov,
            "multicat_decision_threshold": m_pain_th,
            "multicat_summary_strategy": m_strat,
        },
        progress=apply_monotonic_progress(
            {},
            stage="queued",
            window_idx=0,
            n_windows=None,
            percent=0,
        ),
    )
    job.progress = _append_event(
        job.progress,
        slack_line_event(
            stage="queued", window=0, message="job created", job_id=str(job.id)
        ),
    )

    dest_video: Path

    if mode == "upload":
        if not file or not file.filename:
            raise HTTPException(400, "file required for upload mode")
        uploads = settings.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        ext = Path(file.filename).suffix or ".mp4"
        dest_video = uploads / f"{job.id}{ext}"
        size = 0
        max_b = settings.max_upload_mb * 1024 * 1024
        with dest_video.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_b:
                    dest_video.unlink(missing_ok=True)
                    raise HTTPException(413, "upload exceeds MAX_UPLOAD_MB")
                f.write(chunk)
        job.input_video_path = str(dest_video.resolve())

    else:
        if not settings.allow_local_paths:
            raise HTTPException(403, "local_path mode disabled (set ALLOW_LOCAL_PATHS=1)")
        if not video_path:
            raise HTTPException(400, "video_path required for local_path mode")
        p = Path(video_path).expanduser()
        if not p.is_absolute() and settings.local_path_base_dir:
            p = (settings.local_path_base_dir / p).resolve()
        else:
            p = p.resolve()
        if settings.local_path_base_dir:
            base = settings.local_path_base_dir.resolve()
            try:
                p.relative_to(base)
            except ValueError:
                raise HTTPException(403, "video_path must be under LOCAL_PATH_BASE_DIR")
        if not p.is_file():
            raise HTTPException(400, f"video not found: {p}")
        job.input_video_path = str(p)

    # Duration guard
    if settings.max_video_duration_sec is not None:
        try:
            d = probe_video_duration_sec(Path(job.input_video_path))
            if d > settings.max_video_duration_sec:
                if mode == "upload":
                    Path(job.input_video_path).unlink(missing_ok=True)
                raise HTTPException(400, "video exceeds MAX_VIDEO_DURATION_SEC")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"could not read video duration: {e}") from e

    split_enabled = split_window_sec > 0 and split_step_sec > 0
    n_windows_prior: int | None = None
    if split_enabled:
        try:
            d = probe_video_duration_sec(Path(job.input_video_path))
            from shared.video_probe import build_sliding_windows

            n_windows_prior = len(build_sliding_windows(d, split_window_sec, split_step_sec))
        except Exception:
            n_windows_prior = None
        job.progress = apply_monotonic_progress(
            job.progress, n_windows=n_windows_prior
        )

    db.add(job)
    db.commit()
    db.refresh(job)
    return {"id": str(job.id), "status": job.status}


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "invalid job id")
    job = db.get(Job, jid)
    if not job:
        raise HTTPException(404, "job not found")
    prog = job.progress or {}
    return {
        "id": str(job.id),
        "status": job.status,
        "params": job.params or {},
        "progress": {
            "stage": prog.get("stage"),
            "window_idx": prog.get("window_idx"),
            "n_windows": prog.get("n_windows"),
            "percent": prog.get("percent"),
            "events": prog.get("events"),
        },
        "error_type": job.error_type,
        "error_message": job.error_message,
    }


@router.get("/{job_id}/result")
def get_result(
    job_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "invalid job id")
    job = db.get(Job, jid)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "done":
        raise HTTPException(409, "job not completed")
    if not job.result_path:
        raise HTTPException(404, "no result path")
    p = Path(job.result_path)
    if not p.is_file():
        raise HTTPException(404, "pipeline_result.json missing")
    raw = json.loads(p.read_text(encoding="utf-8"))
    normalized = normalize_pipeline_result(
        raw,
        repo_root=settings.repo_root,
        positive_label=settings.positive_label,
    )
    meta_model = normalized.get("meta") or {}
    meta_model["wrapper_version"] = settings.wrapper_version
    normalized["meta"] = meta_model
    return normalized


@router.get("/{job_id}/artifacts")
def list_artifacts(
    job_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "invalid job id")
    job = db.get(Job, jid)
    if not job:
        raise HTTPException(404, "job not found")
    run_dir = resolve_run_dir_for_job(settings.data_dir, job_id)
    if not run_dir or not run_dir.is_dir():
        raise HTTPException(404, "run directory not linked yet")
    return categorize_run_dir_artifacts(run_dir)


@router.get("/{job_id}/artifacts/{file_path:path}")
def stream_artifact(
    job_id: str,
    file_path: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "invalid job id")
    job = db.get(Job, jid)
    if not job:
        raise HTTPException(404, "job not found")
    run_dir = resolve_run_dir_for_job(settings.data_dir, job_id)
    if not run_dir or not run_dir.is_dir():
        raise HTTPException(404, "run directory not found")
    try:
        full = safe_resolve_under_run_dir(run_dir, file_path)
    except (ValueError, PermissionError):
        raise HTTPException(403, "invalid path")
    if not full.is_file():
        raise HTTPException(404, "not a file")
    max_b = settings.max_artifact_stream_mb * 1024 * 1024
    if full.stat().st_size > max_b:
        raise HTTPException(413, "artifact too large for streaming (increase MAX_ARTIFACT_STREAM_MB)")

    return FileResponse(
        path=str(full),
        filename=full.name,
    )


@router.delete("/{job_id}")
def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    delete_run: bool = False,
):
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "invalid job id")
    job = db.get(Job, jid)
    if not job:
        raise HTTPException(404, "job not found")

    if delete_run:
        run_dir = resolve_run_dir_for_job(settings.data_dir, job_id)
        if run_dir and run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
        link = job_link_dir(settings.data_dir, job_id)
        if link.is_dir():
            shutil.rmtree(link, ignore_errors=True)

    mode = (job.params or {}).get("mode")
    if mode == "upload" and job.input_video_path:
        p = Path(job.input_video_path)
        if p.is_file():
            try:
                p.relative_to(settings.data_dir.resolve() / "uploads")
                p.unlink(missing_ok=True)
            except ValueError:
                pass

    db.delete(job)
    db.commit()
    return {"deleted": True}
