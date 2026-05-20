from __future__ import annotations

import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from website root when running as script
_WEBSITE_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(_WEBSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WEBSITE_ROOT))

from backend.services.job_paths import job_link_dir, write_run_reference
from backend.services.pipeline_run import run_pipeline_subprocess
from backend.settings import get_settings
from db.models import Job
from db.session import claim_next_queued_job, configure_engine, get_session_maker
from observability.expose import start_worker_metrics_server
from observability.registry import metrics_enabled
from observability.worker import (
    heartbeat,
    record_job_failure,
    record_job_success,
    scan_pipeline_output,
    set_busy,
)
from observability.system import start_resource_collector
from shared.progress import apply_monotonic_progress, slack_line_event
from shared.multicat_params import parse_form_bool
from shared.video_probe import build_sliding_windows, probe_video_duration_sec

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _append_event(progress: dict, ev: dict, limit: int = 100) -> dict:
    events = list(progress.get("events") or [])
    events.append(ev)
    return {**progress, "events": events[-limit:]}


def _find_run_dir(output_base: Path, stem: str, after_ts: float) -> Path | None:
    best: Path | None = None
    best_mtime = 0.0
    for p in output_base.glob(f"*_{stem}"):
        if not p.is_dir():
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt >= after_ts - 1.0 and mt >= best_mtime:
            best = p
            best_mtime = mt
    return best


def _merge_progress(
    db,
    job_id: uuid.UUID,
    *,
    stage: str | None = None,
    window_idx: int | None = None,
    n_windows: int | None = None,
    percent: float | None = None,
    message: str | None = None,
):
    db.expire_all()
    j = db.get(Job, job_id)
    if not j:
        return
    prog = apply_monotonic_progress(
        j.progress or {},
        stage=stage,
        window_idx=window_idx,
        n_windows=n_windows,
        percent=percent,
    )
    if message:
        prog = _append_event(
            prog,
            slack_line_event(
                stage=str(prog.get("stage", stage or "queued")),
                window=prog.get("window_idx"),
                message=message,
                job_id=str(job_id),
            ),
        )
    j.progress = prog
    db.commit()


def _infer_stage_from_fs(
    run_dir: Path | None,
    *,
    split: bool,
    n_windows: int | None,
) -> tuple[str | None, int | None, float | None]:
    """Return (stage, window_idx, percent) hints from disk (best-effort)."""
    if run_dir is None or not run_dir.is_dir():
        return "extract_audio", 0, 2.0

    audio = run_dir / "extracted_audio.wav"
    has_audio = audio.is_file()

    top_result = run_dir / "pipeline_result.json"

    if split:
        wdirs = sorted(run_dir.glob("window_*"))
        done_count = 0
        for wd in wdirs:
            if (wd / "pipeline_result.json").is_file():
                done_count += 1
        idx = max(0, len(wdirs) - 1) if wdirs else 0
        if top_result.is_file():
            return "aggregating", idx, 92.0
        if not has_audio:
            return "yamnet", idx, 15.0
        if has_audio and wdirs:
            pct = 20.0
            if n_windows and n_windows > 0:
                pct = min(90.0, 20.0 + (done_count / n_windows) * 70.0)
            return "video_branch", idx, pct
        if has_audio:
            return "routing", 0, 18.0
        return "yamnet", 0, 10.0

    # Single clip
    if top_result.is_file():
        return "aggregating", 0, 95.0
    if has_audio:
        # Cannot distinguish yamnet vs branch without parsing; monotonic safe:
        return "video_branch", 0, 40.0
    return "extract_audio", 0, 8.0


def _coerce_job_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return parse_form_bool(str(value))


def process_job(job_id: uuid.UUID) -> None:
    settings = get_settings()
    set_busy(True)
    configure_engine(settings.database_url)
    sm = get_session_maker()
    db = sm()
    job = db.get(Job, job_id)
    if not job or job.status != "running":
        db.close()
        set_busy(False)
        return

    params = job.params or {}
    multicat_video_only = _coerce_job_bool(params.get("multicat_video_only", False))
    if multicat_video_only and not settings.enable_multicat_video:
        job.status = "failed"
        job.finished_at = _utcnow()
        job.error_type = "config_error"
        job.error_message = (
            "multicat mode disabled on server (ENABLE_MULTICAT_VIDEO=0)"
        )
        db.commit()
        db.close()
        record_job_failure("config_error")
        set_busy(False)
        return

    device = str(params.get("device", "auto"))
    cat_threshold = float(params.get("cat_threshold", 0.5))
    split_window_sec = float(params.get("split_window_sec", 0.0))
    split_step_sec = float(params.get("split_step_sec", 0.0))
    multicat_max_cats = int(params.get("multicat_max_cats", 8))
    multicat_min_track_coverage = float(
        params.get("multicat_min_track_coverage", 0.15)
    )
    multicat_decision_threshold = float(
        params.get("multicat_decision_threshold", 0.5)
    )
    multicat_summary_strategy = str(
        params.get("multicat_summary_strategy", "coverage_weighted_mean")
    )
    video_path = Path(job.input_video_path)
    stem = video_path.stem
    split_enabled = split_window_sec > 0 and split_step_sec > 0

    n_windows: int | None = None
    if split_enabled:
        try:
            d = probe_video_duration_sec(video_path)
            n_windows = len(build_sliding_windows(d, split_window_sec, split_step_sec))
        except Exception:
            n_windows = None

    after_ts = time.time() - 2.0
    output_base = settings.repo_root / settings.pipeline_output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    result_box: dict = {}
    error_box: list[Exception] = []

    def run_target():
        try:
            result_box["data"] = run_pipeline_subprocess(
                video_path=video_path,
                repo_root=settings.repo_root,
                device=device,
                cat_threshold=cat_threshold,
                split_window_sec=split_window_sec,
                split_step_sec=split_step_sec,
                output_dir=settings.pipeline_output_dir,
                timeout_sec=settings.job_timeout_sec,
                multicat_video_only=multicat_video_only,
                multicat_max_cats=multicat_max_cats,
                multicat_min_track_coverage=multicat_min_track_coverage,
                multicat_decision_threshold=multicat_decision_threshold,
                multicat_summary_strategy=multicat_summary_strategy,
            )
        except subprocess.TimeoutExpired as e:
            error_box.append(e)
            scan_pipeline_output(getattr(e, "stderr", "") or str(e))
        except Exception as e:
            error_box.append(e)
            scan_pipeline_output(str(e))

    th = threading.Thread(target=run_target, daemon=True)
    th.start()

    _merge_progress(
        db,
        job_id,
        stage="extract_audio",
        window_idx=0,
        n_windows=n_windows,
        percent=3.0,
        message="pipeline subprocess started",
    )

    while th.is_alive():
        run_dir = _find_run_dir(output_base, stem, after_ts)
        st, wi, pct = _infer_stage_from_fs(
            run_dir, split=split_enabled, n_windows=n_windows
        )
        # promote routing -> audio/video only when we can read window result branch — skip for MVP
        _merge_progress(
            db,
            job_id,
            stage=st,
            window_idx=wi,
            n_windows=n_windows,
            percent=pct,
        )
        time.sleep(1.5)

    th.join(timeout=5.0)

    db.expire_all()
    job = db.get(Job, job_id)
    if not job:
        db.close()
        return

    if error_box:
        err = error_box[0]
        if isinstance(err, subprocess.TimeoutExpired):
            et = "timeout"
            msg = f"job exceeded JOB_TIMEOUT_SEC ({settings.job_timeout_sec}s)"
        else:
            et = "pipeline_error"
            msg = str(err)
        job.status = "failed"
        job.finished_at = _utcnow()
        job.error_type = et
        job.error_message = msg
        job.progress = apply_monotonic_progress(
            job.progress or {}, stage="aggregating", percent=100.0
        )
        job.progress = _append_event(
            job.progress,
            slack_line_event(
                stage="aggregating",
                window=job.progress.get("window_idx"),
                message=msg,
                job_id=str(job_id),
            ),
        )
        db.commit()
        db.close()
        record_job_failure(et)
        set_busy(False)
        return

    data = result_box.get("data")
    if not isinstance(data, dict):
        job.status = "failed"
        job.finished_at = _utcnow()
        job.error_type = "pipeline_error"
        job.error_message = "pipeline returned no result"
        db.commit()
        db.close()
        record_job_failure("pipeline_error")
        set_busy(False)
        return

    run_dir = Path(str(data.get("run_dir", ""))).resolve()
    result_json = run_dir / "pipeline_result.json"
    if not result_json.is_file():
        job.status = "failed"
        job.finished_at = _utcnow()
        job.error_type = "pipeline_error"
        job.error_message = f"missing {result_json}"
        db.commit()
        db.close()
        record_job_failure("pipeline_error")
        set_busy(False)
        return

    write_run_reference(job_link_dir(settings.data_dir, str(job_id)), run_dir)
    job.result_path = str(result_json)
    job.status = "done"
    job.finished_at = _utcnow()
    job.error_type = None
    job.error_message = None
    job.progress = apply_monotonic_progress(
        job.progress or {},
        stage="done",
        window_idx=job.progress.get("window_idx", 0) if job.progress else 0,
        n_windows=n_windows,
        percent=100.0,
    )
    job.progress = _append_event(
        job.progress,
        slack_line_event(
            stage="done",
            window=job.progress.get("window_idx"),
            message="pipeline complete",
            job_id=str(job_id),
        ),
    )
    db.commit()
    db.close()
    record_job_success()
    set_busy(False)


def main() -> None:
    settings = get_settings()
    configure_engine(settings.database_url)
    sm = get_session_maker()
    if metrics_enabled():
        port = start_worker_metrics_server()
        start_resource_collector(service="worker", interval=15.0)
        print(f"worker started (metrics :{port})", flush=True)
    else:
        print("worker started", flush=True)
    while True:
        heartbeat()
        db = sm()
        try:
            job = claim_next_queued_job(db)
            if job is None:
                time.sleep(1.0)
                continue
            jid = job.id
        finally:
            db.close()
        process_job(jid)


if __name__ == "__main__":
    main()
