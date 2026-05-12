"""YOLO tracking + chunking + audio (adapted from streaming_video_pipeline)."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pydub import AudioSegment
from scenedetect import ContentDetector, detect
from tqdm import tqdm
from ultralytics import YOLO

from src.audio_filter import YamNetCatGate
from src.constants import CATEGORY_DISCLAIMER
from src.utils import append_jsonl, load_jsonl, project_root, resolve_path


class ProcessingTimeoutError(Exception):
    pass


def select_yolo_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cut_video_ffmpeg(input_path: str, output_path: str, start_time: float, duration: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-i",
        input_path,
        "-c:v",
        "copy",
        "-an",
        "-loglevel",
        "error",
        output_path,
    ]
    subprocess.run(cmd, check=False)


def get_single_cat_segments(
    video_path: str,
    model: YOLO,
    cfg: dict,
    deadline: float | None = None,
) -> list[dict[str, float]]:
    """Scene detect + sequential read + YOLO track every frame_skip frames."""
    ycfg = cfg.get("yolo", {})
    cat_id = int(ycfg.get("cat_class_id", 15))
    conf = float(ycfg.get("conf", 0.4))
    imgsz = int(ycfg.get("imgsz", 640))
    min_seg = float(ycfg.get("min_segment_duration", 3.0))
    frame_skip = max(1, int(ycfg.get("frame_skip", 1)))

    def _deadline_exceeded() -> bool:
        return deadline is not None and time.monotonic() > deadline

    scene_list = detect(video_path, ContentDetector())
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return []

        video_segments: list[dict[str, float]] = []

        for scene_idx, (start_time, end_time) in enumerate(scene_list):
            if _deadline_exceeded():
                raise ProcessingTimeoutError()
            start_frame = start_time.get_frames()
            end_frame = end_time.get_frames()
            track_history: dict[int, dict[str, Any]] = {}

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx < end_frame:
                if _deadline_exceeded():
                    raise ProcessingTimeoutError()
                local_i = frame_idx - start_frame
                # Skip decoding pixels on non-YOLO frames (read() decodes; grab() only advances).
                if frame_skip > 1 and (local_i % frame_skip) != 0:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                results = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=conf,
                    imgsz=imgsz,
                    classes=[cat_id],
                )[0]

                if results.boxes.id is None:
                    pass
                else:
                    ids = results.boxes.id.int().cpu().tolist()
                    if len(ids) > 1:
                        for tid in ids:
                            track_history.setdefault(tid, {"frames": [], "valid": False})
                            track_history[tid]["valid"] = False
                    else:
                        tid = ids[0]
                        track = track_history.setdefault(tid, {"frames": [], "valid": True})
                        if track["valid"]:
                            track["frames"].append(frame_idx / fps)

                frame_idx += 1

            for tid, data in track_history.items():
                if not data["valid"] or not data["frames"]:
                    continue
                seg_start = data["frames"][0]
                seg_end = data["frames"][-1]
                duration = seg_end - seg_start
                if duration >= min_seg:
                    video_segments.append(
                        {
                            "scene": float(scene_idx),
                            "start": round(seg_start, 2),
                            "end": round(seg_end, 2),
                            "duration": round(duration, 2),
                        }
                    )

        return video_segments
    finally:
        cap.release()


def load_existing_snippet_intervals(
    metadata_paths: list[Path],
    snippets_dir: Path,
) -> list[tuple[str, float, float]]:
    """Return (video_id, start, end) for overlap checks."""
    intervals: list[tuple[str, float, float]] = []
    for mp in metadata_paths:
        if not mp.is_file():
            continue
        for row in load_jsonl(mp):
            vid = row.get("video_id")
            for sn in row.get("snippets") or []:
                tr = sn.get("timestamp_range")
                if isinstance(tr, list) and len(tr) >= 2 and vid:
                    intervals.append((str(vid), float(tr[0]), float(tr[1])))
    return intervals


def intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Half-open [a0,a1) vs [b0,b1) — use closed overlap for safety."""
    return a0 < b1 and b0 < a1


# Rough footprint for startup logging (YOLO + audio; actual varies by weights)
_EST_YOLO_RAM_MB = 110
_EST_AUDIO_RAM_MB = 50


def _rows_with_existing_media(rows: list[dict[str, Any]], logger: logging.Logger) -> list[dict[str, Any]]:
    """Drop rows whose video_path/audio_path are missing (e.g. deleted after a prior process run)."""
    out: list[dict[str, Any]] = []
    for d in rows:
        vp, ap = d.get("video_path"), d.get("audio_path")
        if vp and ap and Path(vp).is_file() and Path(ap).is_file():
            out.append(d)
    n = len(rows) - len(out)
    if n:
        logger.warning(
            "Skipping %d video(s): source file(s) not found under download output_dir "
            "(re-download with --start-from download, or set yolo.delete_source_media_after_processing: false).",
            n,
        )
    return out


def _dedupe_rows_by_video_id(rows: list[dict[str, Any]], logger: logging.Logger) -> list[dict[str, Any]]:
    """One row per video_id (first wins). Avoids parallel workers duplicating the same id."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for d in rows:
        vid = d.get("video_id")
        if not vid or vid in seen:
            continue
        seen.add(str(vid))
        out.append(d)
    dropped = len(rows) - len(out)
    if dropped:
        logger.info("Processing: dropped %d duplicate video_id row(s) (keep first per id)", dropped)
    return out


def _run_one_video(
    d: dict[str, Any],
    cfg: dict[str, Any],
    existing_intervals: list[tuple[str, float, float]],
    model: YOLO,
    audio_classifier: YamNetCatGate,
    logger: logging.Logger,
    wpath: Path,
    snippets_dir: Path,
    metadata_file: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Process one video; return stats + outputs for aggregation (no JSONL writes here).

    *existing_intervals* must be the full read-only snapshot from metadata (all videos).
    Overlap checks use intervals for this *video_id* only; duplicate *video_id* rows in the
    batch must be deduplicated before parallel processing.
    """
    ycfg = cfg.get("yolo", {})
    min_chunk = float(ycfg.get("min_chunk_duration", 3.0))
    max_chunk = float(ycfg.get("max_chunk_duration", 7.0))
    max_proc_dur = float(ycfg.get("processing_timeout_sec", 300))

    vid = d["video_id"]
    title = d.get("title", vid)
    video_path = d["video_path"]
    audio_path = d["audio_path"]
    cat = d.get("behavioral_category", "Unknown")
    deadline = time.monotonic() + max_proc_dur if max_proc_dur > 0 else None

    vid_intervals = [t for t in existing_intervals if str(t[0]) == str(vid)]
    snippet_durations: list[float] = []
    audio_filtered_here = 0

    if not Path(video_path).is_file() or not Path(audio_path).is_file():
        return {
            "out_result": {
                "video_id": vid,
                "status": "skipped",
                "reason": "source media missing (deleted after prior run or moved; re-download or keep sources)",
                "behavioral_category": cat,
            },
            "metadata_record": None,
            "snippet_durations": [],
            "snippets_per_video": None,
            "outcome_by_cat_delta": {},
            "no_seg_inc": False,
            "audio_filtered": 0,
        }

    t_start = time.monotonic()
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        dur_v = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
        cap.release()
        if dur_v > 300 or dur_v == 0:
            return {
                "out_result": {
                    "video_id": vid,
                    "status": "skipped",
                    "reason": "duration check failed",
                    "behavioral_category": cat,
                },
                "metadata_record": None,
                "snippet_durations": [],
                "snippets_per_video": None,
                "outcome_by_cat_delta": {},
                "no_seg_inc": False,
                "audio_filtered": 0,
            }

        try:
            raw_segments = get_single_cat_segments(str(video_path), model, cfg, deadline=deadline)
        except ProcessingTimeoutError:
            return {
                "out_result": {"video_id": vid, "status": "error", "reason": "timeout", "behavioral_category": cat},
                "metadata_record": None,
                "snippet_durations": [],
                "snippets_per_video": None,
                "outcome_by_cat_delta": {cat: {"timeout": 1}},
                "no_seg_inc": False,
                "audio_filtered": 0,
            }

        if not raw_segments:
            return {
                "out_result": {"video_id": vid, "status": "no_snippets", "reason": "no valid segments", "behavioral_category": cat},
                "metadata_record": None,
                "snippet_durations": [],
                "snippets_per_video": None,
                "outcome_by_cat_delta": {cat: {"no_segments": 1}},
                "no_seg_inc": True,
                "audio_filtered": 0,
            }

        try:
            full_audio = AudioSegment.from_file(str(audio_path))
        except Exception:
            return {
                "out_result": {"video_id": vid, "status": "error", "reason": "audio load", "behavioral_category": cat},
                "metadata_record": None,
                "snippet_durations": [],
                "snippets_per_video": None,
                "outcome_by_cat_delta": {},
                "no_seg_inc": False,
                "audio_filtered": 0,
            }

        valid_chunks: list[dict[str, float]] = []
        for seg in raw_segments:
            curr_start = seg["start"]
            seg_end = seg["end"]
            while (seg_end - curr_start) >= min_chunk:
                chunk_dur = min(max_chunk, seg_end - curr_start)
                valid_chunks.append({"start": curr_start, "duration": chunk_dur})
                curr_start += chunk_dur

        base_name = vid
        record = {
            "video_id": vid,
            "original_video": f"{title}.mp4",
            "processed_at": __import__("datetime").datetime.now().isoformat(),
            "total_segments_found": len(raw_segments),
            "total_chunks_analyzed": len(valid_chunks),
            "saved_chunks_count": 0,
            "snippets": [],
            "behavioral_category": cat,
            "yolo_model_used": str(wpath),
            "audio_model_used": audio_classifier.model_name,
        }

        chunk_audio_rejected = 0
        for i, chunk in enumerate(valid_chunks):
            start_ms = int(chunk["start"] * 1000)
            end_ms = int((chunk["start"] + chunk["duration"]) * 1000)
            audio_chunk = full_audio[start_ms:end_ms]
            ok, proba = audio_classifier.predict(audio_chunk)
            if not ok:
                chunk_audio_rejected += 1
                if getattr(audio_classifier, "_last_predict_failure", None) == "below_threshold":
                    audio_filtered_here += 1
                continue

            t0s, t1s = chunk["start"], chunk["start"] + chunk["duration"]
            overlap = False
            for _ev_id, a, b in vid_intervals:
                if intervals_overlap(t0s, t1s, a, b):
                    overlap = True
                    break
            if overlap:
                logger.debug("Skip overlapping snippet %s [%.2f, %.2f]", vid, t0s, t1s)
                continue

            snippet_id = f"{base_name}_snip_{i}"
            vp_snip = snippets_dir / f"{snippet_id}.mp4"
            ap_snip = snippets_dir / f"{snippet_id}.mp3"
            cut_video_ffmpeg(str(video_path), str(vp_snip), chunk["start"], chunk["duration"])
            audio_chunk.export(str(ap_snip), format="mp3")
            record["saved_chunks_count"] += 1
            record["snippets"].append(
                {
                    "id": snippet_id,
                    "audio_proba": round(float(proba), 4),
                    "timestamp_range": [round(t0s, 2), round(t1s, 2)],
                    "duration": round(chunk["duration"], 2),
                }
            )
            vid_intervals.append((str(vid), t0s, t1s))
            snippet_durations.append(chunk["duration"])

        proc_time = time.monotonic() - t_start
        record["processing_time_sec"] = round(proc_time, 2)
        record["chunks_audio_rejected"] = chunk_audio_rejected

        if record["snippets"]:
            out_r = {**record, "status": "success"}
            return {
                "out_result": out_r,
                "metadata_record": record,
                "snippet_durations": snippet_durations,
                "snippets_per_video": len(record["snippets"]),
                "outcome_by_cat_delta": {cat: {"with_snippets": 1}},
                "no_seg_inc": False,
                "audio_filtered": audio_filtered_here,
            }
        return {
            "out_result": {
                "video_id": vid,
                "status": "no_snippets",
                "reason": "audio filter or overlap",
                "behavioral_category": cat,
            },
            "metadata_record": None,
            "snippet_durations": [],
            "snippets_per_video": None,
            "outcome_by_cat_delta": {cat: {"audio_or_dup": 1}},
            "no_seg_inc": False,
            "audio_filtered": audio_filtered_here,
        }

    except Exception as e:
        logger.exception("process %s: %s", vid, e)
        return {
            "out_result": {"video_id": vid, "status": "error", "error": str(e), "behavioral_category": cat},
            "metadata_record": None,
            "snippet_durations": [],
            "snippets_per_video": None,
            "outcome_by_cat_delta": {},
            "no_seg_inc": False,
            "audio_filtered": 0,
        }
    finally:
        if cfg.get("yolo", {}).get("delete_source_media_after_processing", True):
            try:
                if video_path and Path(video_path).is_file():
                    Path(video_path).unlink()
                if audio_path and Path(audio_path).is_file():
                    Path(audio_path).unlink()
            except Exception:
                pass


_POOL_CFG: dict[str, Any] | None = None
_POOL_SNAPSHOT: list[tuple[str, float, float]] | None = None
_POOL_MODEL: YOLO | None = None
_POOL_AC: YamNetCatGate | None = None
_POOL_LOG: logging.Logger | None = None
_POOL_WPATH: Path | None = None
_POOL_SNIPPETS: Path | None = None
_POOL_META: Path | None = None
_POOL_RUN: Path | None = None


def _pool_init(
    cfg: dict[str, Any],
    snapshot: list[tuple[str, float, float]],
    device: str,
    weights_abs: str,
    snippets_dir: str,
    metadata_file: str,
    run_dir: str,
    wpath_str: str,
) -> None:
    """Load YOLO + YamNetCatGate once per worker process (spawn), not per task."""
    global _POOL_CFG, _POOL_SNAPSHOT, _POOL_MODEL, _POOL_AC, _POOL_LOG
    global _POOL_WPATH, _POOL_SNIPPETS, _POOL_META, _POOL_RUN
    _POOL_CFG = cfg
    # Read-only overlap intervals (copy made in parent before spawn); workers must not mutate.
    _POOL_SNAPSHOT = snapshot
    _POOL_WPATH = Path(wpath_str)
    _POOL_SNIPPETS = Path(snippets_dir)
    _POOL_META = Path(metadata_file)
    _POOL_RUN = Path(run_dir)
    logging.basicConfig(level=logging.WARNING)
    _POOL_LOG = logging.getLogger(f"proc.{os.getpid()}")
    _POOL_MODEL = YOLO(weights_abs)
    _POOL_MODEL.to(device)
    _POOL_AC = YamNetCatGate(cfg, _POOL_LOG)


def _pool_worker_safe(row: dict[str, Any]) -> dict[str, Any]:
    """Run one video in a pool worker; never raise (survives OOM/segfault class errors where possible)."""
    vid = row.get("video_id", "?")
    cat = row.get("behavioral_category", "Unknown")
    try:
        assert _POOL_CFG is not None and _POOL_SNAPSHOT is not None
        assert _POOL_MODEL is not None and _POOL_AC is not None and _POOL_LOG is not None
        assert _POOL_WPATH is not None and _POOL_SNIPPETS is not None and _POOL_META is not None and _POOL_RUN is not None
        return _run_one_video(
            row,
            _POOL_CFG,
            list(_POOL_SNAPSHOT),
            _POOL_MODEL,
            _POOL_AC,
            _POOL_LOG,
            _POOL_WPATH,
            _POOL_SNIPPETS,
            _POOL_META,
            _POOL_RUN,
        )
    except BaseException as e:
        log = logging.getLogger(f"proc.{os.getpid()}")
        log.exception("worker failed video_id=%s", vid)
        return {
            "out_result": {
                "video_id": vid,
                "status": "error",
                "error": repr(e),
                "behavioral_category": cat,
            },
            "metadata_record": None,
            "snippet_durations": [],
            "snippets_per_video": None,
            "outcome_by_cat_delta": {},
            "no_seg_inc": False,
            "audio_filtered": 0,
        }


def _merge_outcome_delta(obc: dict[str, dict[str, int]], delta: dict[str, dict[str, int]]) -> None:
    if not delta:
        return
    for cat, inner in delta.items():
        for k, v in inner.items():
            obc[cat][k] += v


_yolo_model: YOLO | None = None
_yolo_weights: str | None = None


def get_yolo_model(weights: str, device: str) -> YOLO:
    global _yolo_model, _yolo_weights
    if _yolo_model is None or _yolo_weights != weights:
        _yolo_model = YOLO(weights)
        _yolo_model.to(device)
        _yolo_weights = weights
    return _yolo_model


def process_batch(
    download_results: list[dict[str, Any]],
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
    audio_classifier: YamNetCatGate,
) -> list[dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    root = project_root()
    out_cfg = cfg.get("output", {})
    snippets_dir = resolve_path(root, out_cfg.get("snippets_dir", "data/dataset/snippets"))
    metadata_file = resolve_path(root, out_cfg.get("metadata_file", "data/dataset/metadata.jsonl"))
    ycfg = cfg.get("yolo", {})
    workers = max(1, int(ycfg.get("processing_workers", 1)))
    parallel_dev = str(ycfg.get("parallel_yolo_device", "cpu")).strip().lower()

    snippets_dir.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    weights = ycfg.get("weights", "models/yolo/yolov8l.pt")
    wpath = root / weights if not Path(weights).is_absolute() else Path(weights)
    wpath_abs = str(wpath.resolve())

    existing_intervals = load_existing_snippet_intervals(
        [metadata_file, run_dir / "stage_5_process" / "metadata.jsonl"],
        snippets_dir,
    )

    successes = [d for d in download_results if d.get("video_path") and d.get("audio_path")]
    out_results: list[dict[str, Any]] = []
    snippet_durations: list[float] = []
    snippets_per_video: list[int] = []
    outcome_by_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    audio_filtered = 0
    no_seg = 0

    def _apply_one(res: dict[str, Any], *, extend_intervals: bool) -> None:
        nonlocal audio_filtered, no_seg
        out_results.append(res["out_result"])
        if res.get("metadata_record"):
            # Order of lines matches completion order when processing_workers > 1 (not chronological).
            append_jsonl(res["metadata_record"], metadata_file)
            append_jsonl(res["metadata_record"], run_dir / "stage_5_process" / "metadata.jsonl")
            if extend_intervals:
                rec = res["metadata_record"]
                vid_m = str(rec.get("video_id", ""))
                for sn in rec.get("snippets") or []:
                    tr = sn.get("timestamp_range")
                    if isinstance(tr, list) and len(tr) >= 2 and vid_m:
                        existing_intervals.append((vid_m, float(tr[0]), float(tr[1])))
        snippet_durations.extend(res.get("snippet_durations") or [])
        spv = res.get("snippets_per_video")
        if spv is not None:
            snippets_per_video.append(spv)
        _merge_outcome_delta(outcome_by_cat, res.get("outcome_by_cat_delta") or {})
        if res.get("no_seg_inc"):
            no_seg += 1
        audio_filtered += int(res.get("audio_filtered") or 0)

    successes = _dedupe_rows_by_video_id(successes, logger)
    successes = _rows_with_existing_media(successes, logger)

    per_worker_mb = _EST_YOLO_RAM_MB + _EST_AUDIO_RAM_MB
    if workers > 1:
        logger.info(
            "Estimated RAM for processing pool: ~%d MB (~%d workers × ~%d MB: YOLO ~%d + audio ~%d)",
            per_worker_mb * workers,
            workers,
            per_worker_mb,
            _EST_YOLO_RAM_MB,
            _EST_AUDIO_RAM_MB,
        )
    else:
        logger.info(
            "Estimated RAM (single process): ~%d MB (YOLO ~%d + audio ~%d)",
            per_worker_mb,
            _EST_YOLO_RAM_MB,
            _EST_AUDIO_RAM_MB,
        )

    if workers == 1:
        device = select_yolo_device()
        logger.info(
            "YOLO device: %s | frame_skip=%s | processing_workers=1",
            device,
            ycfg.get("frame_skip", 1),
        )
        model = get_yolo_model(wpath_abs, device)
        pbar = tqdm(successes, desc="Processing", unit="vid")
        for d in pbar:
            res = _run_one_video(
                d,
                cfg,
                existing_intervals,
                model,
                audio_classifier,
                logger,
                wpath,
                snippets_dir,
                metadata_file,
                run_dir,
            )
            _apply_one(res, extend_intervals=True)
            total_snip = len(snippet_durations)
            pbar.set_postfix(snippets=total_snip, no_snip=no_seg)
    else:
        pd = parallel_dev
        if pd == "auto":
            pd = select_yolo_device()
        if workers > 1 and pd == "mps":
            logger.warning(
                "processing_workers=%d with parallel_yolo_device=mps: multiple processes on one Apple GPU "
                "is untested and may OOM or serialize; prefer processing_workers: 1 for a single GPU, or use cpu.",
                workers,
            )
        if workers > 1 and str(pd).startswith("cuda"):
            logger.warning(
                "processing_workers=%d with CUDA: multiple workers may contend on one GPU unless you assign "
                "distinct devices (e.g. cuda:0 / cuda:1) per machine layout; not auto-assigned here.",
                workers,
            )
        logger.info(
            "YOLO parallel: workers=%s device=%s | frame_skip=%s (initializer loads YOLO+audio once per worker)",
            workers,
            pd,
            ycfg.get("frame_skip", 1),
        )
        # Frozen overlap snapshot for all workers: shallow copy of the list so we do not share the
        # parent's mutable `existing_intervals` reference. Tuple rows are immutable. With "spawn",
        # child processes receive a pickled copy anyway; this makes the intent obvious and keeps the
        # main process free to grow `existing_intervals` only on the sequential code path.
        snap = list(existing_intervals)
        initargs = (
            cfg,
            snap,
            pd,
            wpath_abs,
            str(snippets_dir.resolve()),
            str(metadata_file.resolve()),
            str(run_dir.resolve()),
            wpath_abs,
        )
        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_pool_init,
            initargs=initargs,
        ) as ex:
            futures = {ex.submit(_pool_worker_safe, d): d for d in successes}
            pbar = tqdm(total=len(successes), desc="Processing", unit="vid")
            for fut in as_completed(futures):
                row = futures[fut]
                vid = row.get("video_id", "?")
                try:
                    res = fut.result()
                except Exception as e:
                    logger.exception("Processing future failed (video_id=%s): %s", vid, e)
                    res = {
                        "out_result": {
                            "video_id": vid,
                            "status": "error",
                            "error": f"worker_future: {e!s}",
                            "behavioral_category": row.get("behavioral_category", "Unknown"),
                        },
                        "metadata_record": None,
                        "snippet_durations": [],
                        "snippets_per_video": None,
                        "outcome_by_cat_delta": {},
                        "no_seg_inc": False,
                        "audio_filtered": 0,
                    }
                _apply_one(res, extend_intervals=False)
                pbar.update(1)
                pbar.set_postfix(snippets=len(snippet_durations), no_snip=no_seg)
            pbar.close()

    # plots
    if snippets_per_video:
        fig, ax = plt.subplots()
        ax.hist(snippets_per_video, bins=20, color="steelblue")
        ax.set_xlabel("Snippets per video")
        fig.savefig(run_dir / "stage_5_process" / "snippets_per_video.png", dpi=150)
        plt.close(fig)
    if snippet_durations:
        fig, ax = plt.subplots()
        ax.hist(snippet_durations, bins=30, color="purple")
        ax.set_xlabel("Duration (s)")
        fig.savefig(run_dir / "stage_5_process" / "snippet_durations.png", dpi=150)
        plt.close(fig)

    if outcome_by_cat:
        cats = sorted(outcome_by_cat.keys())
        keys = ["with_snippets", "no_segments", "audio_or_dup", "timeout"]
        bottom = np.zeros(len(cats))
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#27ae60", "#e67e22", "#8e44ad", "#c0392b"]
        for i, k in enumerate(keys):
            vals = [outcome_by_cat[c].get(k, 0) for c in cats]
            ax.bar(cats, vals, bottom=bottom, label=k, color=colors[i % len(colors)])
            bottom += np.array(vals)
        ax.set_ylabel("Count")
        ax.legend()
        fig.suptitle(f"Processing outcomes by behavioral_category\n{CATEGORY_DISCLAIMER}", fontsize=9)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(run_dir / "stage_5_process" / "processing_outcomes_by_category.png", dpi=150)
        plt.close(fig)

    return out_results
