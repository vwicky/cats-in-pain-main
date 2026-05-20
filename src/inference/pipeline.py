#!/usr/bin/env python3
"""
Cat-in-Pain inference pipeline.

Given a video file:
  1. Extract audio (ffmpeg).
  2. YAMNet pre-classifier → P(cat).
  3a. If P(cat) >= threshold → AUDIO BRANCH (AudioSep + emotion classification).
  3b. If P(cat) < threshold → VIDEO BRANCH (ViTPose + 9×STGCN + LogReg stacking).
  4. Save original video, extracted audio, branch artifacts, timing, result JSON.

**Split mode** (``--split-window-sec`` / ``--split-step-sec``): for each sliding
window the pipeline extracts a **clip MP4**, then runs steps 1–4 on **that clip
only** (audio and routing are **per clip**, not computed once for the full video).

Usage
-----
  python src/inference/pipeline.py --video path/to/video.mp4

Run from repo root. All output goes under runs/inference/<timestamp>_<stem>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch

# ── Repo root & sys.path bootstrap ───────────────────────────────────────────
# When run as a script (python src/inference/pipeline.py), Python does not
# recognise this file as part of a package, so relative imports fail.
# We add src/ to sys.path so that ``from inference.X import ...`` always works.
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Third-party DEBUG/INFO that would flood the CLI when root is verbose (HF Hub, HTTP, JIT, plotting).
_NOISY_LOGGER_NAMES = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "urllib3",
    "urllib3.connectionpool",
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "PIL.PngImagePlugin",
    "numba",
    "numba.core",
    "numba.core.byteflow",
    "numba.core.interpreter",
    "h5py",
    "h5py._conv",
    "transformers",
    "transformers.modeling_utils",
    "sentence_transformers",
    "torchaudio",
    "lightning",
    "lightning.pytorch",
    "pytorch_lightning",
    "fsspec",
    "filelock",
    "huggingface_hub",
    "hf_xet",
    "datasets",
)


def _silence_noisy_loggers(level: int = logging.WARNING) -> None:
    for name in _NOISY_LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


def _setup_logging(run_dir: Path | None = None, verbose: bool = False) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
    handlers: list[logging.Handler] = []

    # stderr: keeps stdout free for structured output (e.g. final JSON via print).
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    handlers.append(sh)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "pipeline.log"
        try:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG if verbose else logging.INFO)
            fh.setFormatter(fmt)
            handlers.append(fh)
        except OSError as e:
            print(f"WARNING: Could not open log file {log_path}: {e}", file=sys.stderr)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        handlers=handlers,
        force=True,
    )
    _silence_noisy_loggers(logging.WARNING)

    log_pipeline = logging.getLogger("pipeline")
    log_inference = logging.getLogger("inference")
    if verbose:
        log_pipeline.setLevel(logging.DEBUG)
        log_inference.setLevel(logging.DEBUG)
    else:
        log_pipeline.setLevel(logging.INFO)
        log_inference.setLevel(logging.INFO)

    return log_pipeline


logger = logging.getLogger("pipeline")

# ── Audio pre-classifier bootstrap (mirrors data_pipeline_v2/audio_classifier.py) ──
_AP3_PKG = "ap3_internal"


def _load_yamnet() -> Any:
    """
    Dynamically load YamNetRunner from audio/audio-pre-classifier/src/
    using a synthetic package name to avoid name clashes with our own ``src``.
    """
    ap3 = REPO_ROOT / "audio" / "audio-pre-classifier"
    src = ap3 / "src"
    if not src.is_dir():
        raise RuntimeError(f"audio/audio-pre-classifier/src not found: {src}")

    if _AP3_PKG not in sys.modules:
        pkg = types.ModuleType(_AP3_PKG)
        pkg.__path__ = [str(src)]
        sys.modules[_AP3_PKG] = pkg

    load_order = ("config", "audio_utils", "yamnet_runner")
    for name in load_order:
        full = f"{_AP3_PKG}.{name}"
        if full in sys.modules:
            continue
        path = src / f"{name}.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ap3 module: {path}")
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _AP3_PKG
        sys.modules[full] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

    return sys.modules[f"{_AP3_PKG}.yamnet_runner"].YamNetRunner


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if req == "mps":
        return (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    if req == "cpu":
        return torch.device("cpu")
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested!r}")


def extract_audio(video_path: Path, run_dir: Path) -> Path:
    """Use ffmpeg to extract mono 16 kHz WAV from a video file."""
    out = run_dir / "extracted_audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz for YAMNet
        "-vn",               # no video
        str(out),
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (code {result.returncode}):\n{result.stderr}"
        )
    return out


def run_yamnet(
    audio_path: Path,
    device: torch.device,
    YamNetRunner: Any,
) -> float:
    """Return P(cat) ∈ [0, 1] for the given audio file."""
    runner = YamNetRunner(device=device)
    p_cat = runner.predict_p_cat_path(audio_path)
    logger.info("YAMNet P(cat) = %.4f", p_cat)
    return p_cat


def _probe_video_duration_sec(video_path: Path) -> float:
    """Read video duration via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe duration failed (code {result.returncode}):\n{result.stderr}"
        )
    raw = result.stdout.strip()
    try:
        dur = float(raw)
    except ValueError as e:
        raise RuntimeError(f"Could not parse ffprobe duration from {raw!r}") from e
    if not math.isfinite(dur) or dur <= 0:
        raise RuntimeError(f"Invalid video duration: {dur}")
    return dur


def _build_sliding_windows(
    duration_sec: float,
    window_sec: float,
    step_sec: float,
) -> list[tuple[float, float]]:
    """
    Sliding windows over [0, duration], inclusive tail.

    Example: duration=10, window=6, step=3 -> [0,6], [3,9], [6,10].
    """
    if duration_sec <= 0:
        return []
    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be > 0")

    windows: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_sec - 1e-9:
        end = min(t + window_sec, duration_sec)
        windows.append((t, end))
        if end >= duration_sec - 1e-9:
            break
        t += step_sec
    return windows


def _extract_video_window(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
) -> Path:
    """Extract one clip [start_sec, end_sec] with ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip_dur = max(0.0, float(end_sec - start_sec))
    if clip_dur <= 0:
        raise ValueError(f"Non-positive clip duration: {clip_dur} ({start_sec}, {end_sec})")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{clip_dur:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip extraction failed (code {result.returncode}):\n{result.stderr}"
        )
    return out_path


def _to_rel_under_run(run_dir: Path, path: str | Path) -> str:
    run_dir = run_dir.resolve()
    try:
        return str(Path(path).resolve().relative_to(run_dir))
    except ValueError:
        return str(Path(path))


def _relativize_multicat_cats(cats: list[dict[str, Any]], run_dir: Path) -> None:
    for c in cats:
        if c.get("pose_npy"):
            c["pose_npy"] = _to_rel_under_run(run_dir, c["pose_npy"])
        if c.get("pose_mask_path"):
            c["pose_mask_path"] = _to_rel_under_run(run_dir, c["pose_mask_path"])
        if c.get("pose_video"):
            c["pose_video"] = _to_rel_under_run(run_dir, c["pose_video"])


def _window_p_pain(window_result: dict[str, Any]) -> float | None:
    """
    Extract comparable P(pain) from one window result across branches.

    - video branch: meta_result.p_pain
    - audio branch: emotion.softmax["Paining"] (fallbacks to confidence when class is Paining)
    - multicat video: headline from ``multicat_summary_strategy`` on cats[]
    """
    cats = window_result.get("cats")
    if (
        window_result.get("multicat_video_only")
        and isinstance(cats, list)
        and cats
    ):
        from inference.multicat_aggregate import window_headline_from_cats

        params = window_result.get("multicat_params") or {}
        strat = str(
            params.get("multicat_summary_strategy", "coverage_weighted_mean")
        )
        th = float(params.get("multicat_decision_threshold", 0.5))
        wh = window_headline_from_cats(cats, strategy=strat, pain_threshold=th)
        hp = wh.get("headline_p_pain")
        if isinstance(hp, (int, float)):
            return float(hp)
        return None

    branch = str(window_result.get("branch", ""))
    if branch == "video":
        meta = window_result.get("meta_result")
        if isinstance(meta, dict):
            v = meta.get("p_pain")
            if isinstance(v, (int, float)):
                return float(v)
        return None

    if branch == "audio":
        emo = window_result.get("emotion")
        if isinstance(emo, dict):
            sm = emo.get("softmax")
            if isinstance(sm, dict):
                p = sm.get("Paining")
                if isinstance(p, (int, float)):
                    return float(p)
            cls = str(emo.get("predicted_class", ""))
            conf = emo.get("confidence")
            if cls == "Paining" and isinstance(conf, (int, float)):
                return float(conf)
        return None

    return None


def _summarize_split_windows(
    windows: list[dict[str, Any]],
    *,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    scores: list[float] = []
    n_audio = 0
    n_video = 0

    inner_results: list[dict[str, Any]] = []
    for w in windows:
        res = w.get("result")
        if isinstance(res, dict):
            inner_results.append(res)

    for w in windows:
        res = w.get("result")
        if not isinstance(res, dict):
            continue
        b = str(res.get("branch", ""))
        if b == "audio":
            n_audio += 1
        elif b == "video":
            n_video += 1
        p = _window_p_pain(res)
        if p is not None:
            scores.append(max(0.0, min(1.0, float(p))))

    if scores:
        p_max = max(scores)
        p_mean = sum(scores) / len(scores)
    else:
        p_max = 0.0
        p_mean = 0.0

    summary: dict[str, Any] = {
        "window_count_total": len(windows),
        "window_count_audio_branch": n_audio,
        "window_count_video_branch": n_video,
        "window_count_scored": len(scores),
        "video_level_p_pain_max": p_max,
        "video_level_p_pain_mean": p_mean,
        "decision_threshold": float(decision_threshold),
        "video_level_decision_max": "pain" if p_max >= decision_threshold else "non_pain",
        "video_level_decision_mean": "pain" if p_mean >= decision_threshold else "non_pain",
    }

    if inner_results and inner_results[0].get("multicat_video_only"):
        from inference.multicat_aggregate import clip_level_from_window_results

        params = inner_results[0].get("multicat_params") or {}
        l2 = clip_level_from_window_results(
            inner_results,
            strategy=str(
                params.get("multicat_summary_strategy", "coverage_weighted_mean")
            ),
            pain_threshold=float(
                params.get("multicat_decision_threshold", decision_threshold)
            ),
        )
        summary["multicat_clip_headline_p_pain"] = l2.get("headline_p_pain")
        summary["multicat_clip_decision"] = l2.get("decision")
        summary["multicat_clip_prevalence_fraction"] = l2.get(
            "multicat_prevalence_fraction"
        )
        summary["multicat_clip_cats_total"] = l2.get("multicat_cats_total")
        summary["multicat_n_windows_contributing"] = l2.get(
            "multicat_n_windows_contributing"
        )
        summary["multicat_clip_cats_above_threshold"] = l2.get(
            "multicat_cats_above_threshold"
        )

    return summary


def _run_pipeline_single(
    video_path: Path,
    run_dir: Path,
    *,
    device_str: str,
    cat_threshold: float,
    stack_run: str | Path | None,
    vitpose_model: str | None,
    vitpose_dataset: str | None,
    vitpose_arch: str | None,
    yolo_model: str | None,
    audiosep_config: str | None,
    audiosep_ckpt: str | None,
    emotion_ckpt: str | None,
    multicat_video_only: bool = False,
    multicat_max_cats: int = 8,
    multicat_min_track_coverage: float = 0.15,
    multicat_decision_threshold: float = 0.5,
    multicat_summary_strategy: str = "coverage_weighted_mean",
    window_index: int | None = None,
) -> dict[str, Any]:
    from inference.artifact_io import save_json, save_original_video
    from inference.audio_clip_probe import probe_wav_clip
    from inference.audio_branch import run_audio_branch
    from inference.stgcn_loader import DEFAULT_STACK_RUN
    from inference.timer import StepTimer
    from inference.video_branch import run_video_branch, run_video_branch_multicat
    from inference.pose_assembler import (
        DEFAULT_VITPOSE_ARCH,
        DEFAULT_VITPOSE_DATASET,
        DEFAULT_VITPOSE_MODEL,
        DEFAULT_YOLO,
    )

    timer = StepTimer()
    device = resolve_device(device_str)
    logger.info("Device: %s", device)

    resolved_stack = Path(stack_run) if stack_run else DEFAULT_STACK_RUN
    resolved_vitpose = vitpose_model or DEFAULT_VITPOSE_MODEL
    resolved_vitpose_ds = vitpose_dataset or DEFAULT_VITPOSE_DATASET
    resolved_vitpose_arch = vitpose_arch or DEFAULT_VITPOSE_ARCH
    resolved_yolo = yolo_model or DEFAULT_YOLO

    multicat_params: dict[str, Any] = {
        "multicat_video_only": multicat_video_only,
        "multicat_max_cats": int(multicat_max_cats),
        "multicat_min_track_coverage": float(multicat_min_track_coverage),
        "multicat_decision_threshold": float(multicat_decision_threshold),
        "multicat_summary_strategy": str(multicat_summary_strategy),
    }

    # ── 1. Save original video ────────────────────────────────────────────────
    with timer.step("save_original_video"):
        original_dst = save_original_video(video_path, run_dir)
    logger.info("Original video → %s", original_dst)

    # ── 2. Extract audio ──────────────────────────────────────────────────────
    with timer.step("audio_extraction"):
        audio_path = extract_audio(video_path, run_dir)
    logger.info("Extracted audio → %s", audio_path)
    clip_audio_probe = probe_wav_clip(audio_path)

    # ── 3. YAMNet pre-classifier ──────────────────────────────────────────────
    with timer.step("yamnet_preclassifier"):
        YamNetRunner = _load_yamnet()
        p_cat = run_yamnet(audio_path, device, YamNetRunner)

    # ── 4. Branch routing ─────────────────────────────────────────────────────
    branch_result: dict[str, Any]
    if multicat_video_only:
        logger.info("Multicat mode: forcing VIDEO BRANCH (P(cat)=%.4f logged only)", p_cat)
        branch_result = run_video_branch_multicat(
            video_path,
            run_dir,
            device,
            timer,
            vitpose_model=resolved_vitpose,
            vitpose_dataset=resolved_vitpose_ds,
            vitpose_arch=resolved_vitpose_arch,
            yolo_model=resolved_yolo,
            stack_run_dir=resolved_stack,
            min_track_coverage=float(multicat_min_track_coverage),
            max_cats=int(multicat_max_cats),
            window_index=window_index,
        )
        cats = branch_result.get("cats") or []
        if cats:
            _relativize_multicat_cats(cats, run_dir)
        if branch_result.get("pose_video"):
            branch_result["pose_video"] = _to_rel_under_run(
                run_dir, branch_result["pose_video"]
            )
        logger.info(
            "[pipeline] multicat complete: %d cat(s) scored",
            int(branch_result.get("multicat_cat_count", len(cats))),
        )
    elif p_cat >= cat_threshold:
        logger.info(
            "P(cat)=%.4f >= threshold=%.2f → AUDIO BRANCH", p_cat, cat_threshold
        )
        branch_result = run_audio_branch(
            audio_path,
            run_dir,
            device,
            timer,
            text_query="cat sounds",
            emotion_ckpt=emotion_ckpt,
        )
    else:
        logger.info(
            "P(cat)=%.4f < threshold=%.2f → VIDEO BRANCH", p_cat, cat_threshold
        )
        branch_result = run_video_branch(
            video_path,
            run_dir,
            device,
            timer,
            vitpose_model=resolved_vitpose,
            vitpose_dataset=resolved_vitpose_ds,
            vitpose_arch=resolved_vitpose_arch,
            yolo_model=resolved_yolo,
            stack_run_dir=resolved_stack,
        )

    # ── 5. Assemble final result ──────────────────────────────────────────────
    timing_dict = timer.to_dict()
    timing_dict["total"] = timer.total()

    result: dict[str, Any] = {
        "video": str(video_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "cat_threshold": cat_threshold,
        "clip_audio_probe": clip_audio_probe,
        "p_cat": p_cat,
        "branch": branch_result["branch"],
        "multicat_params": multicat_params,
        "artifacts": {
            "original_video": str(original_dst),
            "extracted_audio": str(audio_path),
        },
        **{k: v for k, v in branch_result.items() if k != "branch"},
        "timing_seconds": timing_dict,
    }

    # Record artifact paths from branch
    if branch_result["branch"] == "audio":
        result["artifacts"]["separated_audio"] = branch_result.get("separated_audio", "")
    else:
        if multicat_video_only:
            result["artifacts"]["pose_video"] = ""
            mc = result.get("cats") or []
            if mc:
                result["artifacts"]["multicat_cats"] = [
                    {
                        "local_track_id": c.get("local_track_id"),
                        "pose_video": c.get("pose_video"),
                        "pose_npy": c.get("pose_npy"),
                    }
                    for c in mc
                    if isinstance(c, dict)
                ]
        else:
            result["artifacts"]["pose_video"] = branch_result.get("pose_video", "")
            result["artifacts"]["pose_npy"] = branch_result.get("pose_npy", "")
            mask_npy = branch_result.get("pose_mask_path") or ""
            if mask_npy:
                result["artifacts"]["pose_mask_npy"] = mask_npy

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    save_json(result, run_dir / "pipeline_result.json")
    save_json(timing_dict, run_dir / "timing.json")

    logger.info(timer.summary())
    logger.info("Pipeline complete. Results → %s", run_dir / "pipeline_result.json")
    return result


# ── Main pipeline function ────────────────────────────────────────────────────

def run_pipeline(
    video_path: str | Path,
    *,
    output_base: str | Path = "runs/inference",
    device_str: str = "auto",
    cat_threshold: float = 0.5,
    stack_run: str | Path | None = None,
    vitpose_model: str | None = None,
    vitpose_dataset: str | None = None,
    vitpose_arch: str | None = None,
    yolo_model: str | None = None,
    audiosep_config: str | None = None,
    audiosep_ckpt: str | None = None,
    emotion_ckpt: str | None = None,
    split_window_sec: float = 0.0,
    split_step_sec: float = 0.0,
    verbose: bool = False,
    multicat_video_only: bool = False,
    multicat_max_cats: int = 8,
    multicat_min_track_coverage: float = 0.15,
    multicat_decision_threshold: float = 0.5,
    multicat_summary_strategy: str = "coverage_weighted_mean",
) -> dict[str, Any]:
    """
    Run the full inference pipeline on a single video.

    Returns the complete result dict (also written to pipeline_result.json).
    """
    from inference.artifact_io import make_run_dir, save_json

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_base = REPO_ROOT / output_base if not Path(output_base).is_absolute() else Path(output_base)
    run_dir = make_run_dir(output_base, video_path.stem)
    logger = _setup_logging(run_dir, verbose=verbose)
    logger.info("Run directory: %s", run_dir)
    logger.info("Video: %s", video_path)
    split_enabled = split_window_sec > 0 and split_step_sec > 0
    if not split_enabled and (split_window_sec > 0 or split_step_sec > 0):
        raise ValueError(
            "To enable splitting, set both --split-window-sec and --split-step-sec (> 0)."
        )

    if not split_enabled:
        return _run_pipeline_single(
            video_path,
            run_dir,
            device_str=device_str,
            cat_threshold=cat_threshold,
            stack_run=stack_run,
            vitpose_model=vitpose_model,
            vitpose_dataset=vitpose_dataset,
            vitpose_arch=vitpose_arch,
            yolo_model=yolo_model,
            audiosep_config=audiosep_config,
            audiosep_ckpt=audiosep_ckpt,
            emotion_ckpt=emotion_ckpt,
            multicat_video_only=multicat_video_only,
            multicat_max_cats=multicat_max_cats,
            multicat_min_track_coverage=multicat_min_track_coverage,
            multicat_decision_threshold=multicat_decision_threshold,
            multicat_summary_strategy=multicat_summary_strategy,
            window_index=None,
        )

    duration_sec = _probe_video_duration_sec(video_path)
    windows = _build_sliding_windows(duration_sec, split_window_sec, split_step_sec)
    if not windows:
        raise RuntimeError("No windows were generated for split inference.")

    logger.info(
        "Split inference enabled: %.2fs windows, %.2fs step, duration %.2fs → %d clips",
        split_window_sec,
        split_step_sec,
        duration_sec,
        len(windows),
    )

    clips_dir = run_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    window_results: list[dict[str, Any]] = []

    for i, (start, end) in enumerate(windows):
        clip_name = f"clip_{i:03d}_{start:06.2f}s_{end:06.2f}s.mp4".replace(":", "_")
        clip_path = clips_dir / clip_name
        logger.info("Extracting clip %d/%d: [%.2f, %.2f] → %s", i + 1, len(windows), start, end, clip_path.name)
        _extract_video_window(video_path, start, end, clip_path)

        clip_run_dir = run_dir / f"window_{i:03d}"
        clip_run_dir.mkdir(parents=True, exist_ok=True)
        clip_result = _run_pipeline_single(
            clip_path,
            clip_run_dir,
            device_str=device_str,
            cat_threshold=cat_threshold,
            stack_run=stack_run,
            vitpose_model=vitpose_model,
            vitpose_dataset=vitpose_dataset,
            vitpose_arch=vitpose_arch,
            yolo_model=yolo_model,
            audiosep_config=audiosep_config,
            audiosep_ckpt=audiosep_ckpt,
            emotion_ckpt=emotion_ckpt,
            multicat_video_only=multicat_video_only,
            multicat_max_cats=multicat_max_cats,
            multicat_min_track_coverage=multicat_min_track_coverage,
            multicat_decision_threshold=multicat_decision_threshold,
            multicat_summary_strategy=multicat_summary_strategy,
            window_index=i,
        )
        window_results.append(
            {
                "window_index": i,
                "start_sec": start,
                "end_sec": end,
                "clip_video": str(clip_path),
                "window_run_dir": str(clip_run_dir),
                "result": clip_result,
            }
        )

    aggregate: dict[str, Any] = {
        "video": str(video_path),
        "run_dir": str(run_dir),
        "mode": "split_sliding_windows",
        "multicat_params": {
            "multicat_video_only": multicat_video_only,
            "multicat_max_cats": int(multicat_max_cats),
            "multicat_min_track_coverage": float(multicat_min_track_coverage),
            "multicat_decision_threshold": float(multicat_decision_threshold),
            "multicat_summary_strategy": str(multicat_summary_strategy),
        },
        "split": {
            "window_sec": split_window_sec,
            "step_sec": split_step_sec,
            "video_duration_sec": duration_sec,
            "n_windows": len(window_results),
        },
        "summary": _summarize_split_windows(
            window_results, decision_threshold=multicat_decision_threshold
        ),
        "windows": window_results,
    }
    save_json(aggregate, run_dir / "pipeline_result.json")
    logger.info("Split pipeline complete. Results → %s", run_dir / "pipeline_result.json")
    return aggregate


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cat-in-Pain inference pipeline: audio or video branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input video file.")
    p.add_argument(
        "--output-dir",
        default="runs/inference",
        help="Base directory for output run folders.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Compute device.",
    )
    p.add_argument(
        "--cat-threshold",
        type=float,
        default=0.5,
        help="YAMNet P(cat) threshold: >= routes to audio branch, < to video branch.",
    )
    p.add_argument(
        "--stack-run",
        default=None,
        help=(
            "Path to the stacked meta-learner run directory (e.g. "
            "runs/pose-models/stgcn_dlc_stack_20260504_235843/run_05). "
            "Defaults to the bundled run_05."
        ),
    )
    p.add_argument(
        "--vitpose-model",
        default=None,
        help=(
            "Path to ViTPose .pth checkpoint "
            "(default: models/pose_est/vitpose-h-apt36k.pth)."
        ),
    )
    p.add_argument(
        "--vitpose-dataset",
        default=None,
        help="VitInference dataset config key (default: apt36k).",
    )
    p.add_argument(
        "--vitpose-arch",
        default=None,
        help="VitInference model_name / size letter (default: h).",
    )
    p.add_argument(
        "--yolo-model",
        default=None,
        help="Override path to YOLO .pt checkpoint (default: models/yolo/yolov8x.pt).",
    )
    p.add_argument(
        "--audiosep-config",
        default=None,
        help="Override AudioSep config YAML (default: audio/AudioSep/config/audiosep_base.yaml).",
    )
    p.add_argument(
        "--audiosep-ckpt",
        default=None,
        help="Override AudioSep checkpoint (default: audio/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt).",
    )
    p.add_argument(
        "--emotion-ckpt",
        default=None,
        help="Override CatEmotionModel checkpoint (default: models/audio_emotions/best_model_final.pth).",
    )
    p.add_argument(
        "--split-window-sec",
        type=float,
        default=0.0,
        help="Optional sliding-window size in seconds (set with --split-step-sec to enable split inference).",
    )
    p.add_argument(
        "--split-step-sec",
        type=float,
        default=0.0,
        help="Optional sliding-window step in seconds (set with --split-window-sec to enable split inference).",
    )
    p.add_argument(
        "--multicat-video-only",
        action="store_true",
        help="Force video branch and run SORT multi-track pose + per-cat ST-GCN.",
    )
    p.add_argument(
        "--multicat-max-cats",
        type=int,
        default=8,
        help="Max scored tracks per clip when --multicat-video-only is set.",
    )
    p.add_argument(
        "--multicat-min-track-coverage",
        type=float,
        default=0.15,
        help="Min fraction of sampled frames a track must appear in to be scored.",
    )
    p.add_argument(
        "--multicat-decision-threshold",
        type=float,
        default=0.5,
        help="p_pain threshold for prevalence / headline decisions (video meta only).",
    )
    p.add_argument(
        "--multicat-summary-strategy",
        type=str,
        default="coverage_weighted_mean",
        choices=(
            "max",
            "mean",
            "majority_above_threshold",
            "coverage_weighted_mean",
        ),
        help="How to aggregate per-cat scores into a headline.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = run_pipeline(
            args.video,
            output_base=args.output_dir,
            device_str=args.device,
            cat_threshold=args.cat_threshold,
            stack_run=args.stack_run,
            vitpose_model=args.vitpose_model,
            vitpose_dataset=args.vitpose_dataset,
            vitpose_arch=args.vitpose_arch,
            yolo_model=args.yolo_model,
            audiosep_config=args.audiosep_config,
            audiosep_ckpt=args.audiosep_ckpt,
            emotion_ckpt=args.emotion_ckpt,
            split_window_sec=args.split_window_sec,
            split_step_sec=args.split_step_sec,
            verbose=args.verbose,
            multicat_video_only=args.multicat_video_only,
            multicat_max_cats=args.multicat_max_cats,
            multicat_min_track_coverage=args.multicat_min_track_coverage,
            multicat_decision_threshold=args.multicat_decision_threshold,
            multicat_summary_strategy=args.multicat_summary_strategy,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as e:
        logging.getLogger("pipeline").exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
