from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared.p_pain import enrich_pipeline_result_inplace
from shared.video_probe import probe_video_duration_sec

WRAPPER_VERSION = "0.1.0"


def _git_commit_short(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _collect_model_hints(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    device = raw.get("device")
    if device:
        out["device"] = device
    for key in (
        "pose_video",
        "pairwise_probs",
        "emotion",
        "artifacts",
    ):
        if key in raw:
            out[key] = raw.get(key)
    arts = raw.get("artifacts")
    if isinstance(arts, dict):
        out["artifact_paths_sample"] = {
            k: str(v)[:500] for k, v in list(arts.items())[:20]
        }
    return out


def normalize_pipeline_result(
    raw: dict[str, Any],
    *,
    repo_root: Path,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Stable API shape: summary, windows, artifacts index, raw, convenience fields."""
    data = json.loads(json.dumps(raw, default=str))
    enrich_pipeline_result_inplace(data, positive_label=positive_label)

    pipeline_version = _git_commit_short(repo_root) or "unknown"

    if data.get("mode") == "split_sliding_windows":
        summary = data.get("summary") or {}
        windows_raw = data.get("windows") or []
        norm_windows = []
        for w in windows_raw:
            res = w.get("result") or {}
            norm_windows.append(_normalize_window_row(w, res, positive_label))

        final_decision = summary.get("video_level_decision_max")
        p_max = summary.get("video_level_p_pain_max")
        p_mean = summary.get("video_level_p_pain_mean")

        artifacts = categorize_run_dir_artifacts(Path(str(data.get("run_dir", ""))))
        return {
            "summary": {
                "final_decision": final_decision,
                "p_pain_max": p_max,
                "p_pain_mean": p_mean,
                "video_duration_sec": (data.get("split") or {}).get("video_duration_sec"),
                "n_windows": (data.get("split") or {}).get("n_windows"),
                "split": data.get("split"),
                "aggregate_summary": summary,
            },
            "windows": norm_windows,
            "artifacts": artifacts,
            "raw": data,
            "final_decision": final_decision,
            "p_pain_max": p_max,
            "p_pain_mean": p_mean,
            "meta": {
                "pipeline_version": pipeline_version,
                "model_version": _model_version_string(data),
                "wrapper_version": WRAPPER_VERSION,
            },
        }

    # Single-video run
    summary = _summarize_single(data, positive_label)
    artifacts = categorize_run_dir_artifacts(Path(str(data.get("run_dir", ""))))
    fd = summary.get("final_decision")
    pm = summary.get("p_pain_max")
    pmean = summary.get("p_pain_mean")

    return {
        "summary": summary,
        "windows": [
            {
                "window_index": 0,
                "start_sec": 0.0,
                "end_sec": summary.get("video_duration_sec"),
                "branch": data.get("branch"),
                "p_cat": data.get("p_cat"),
                "p_pain": pm,
                "decision": fd,
                "artifacts": _branch_artifact_hints(data),
                "probabilities": _extract_branch_probabilities(data),
                "clip_audio_probe": _normalize_clip_audio_probe(data),
            }
        ],
        "artifacts": artifacts,
        "raw": data,
        "final_decision": fd,
        "p_pain_max": pm,
        "p_pain_mean": pmean,
        "meta": {
            "pipeline_version": pipeline_version,
            "model_version": _model_version_string(data),
            "wrapper_version": WRAPPER_VERSION,
        },
    }


def _normalize_clip_audio_probe(res: dict[str, Any]) -> dict[str, Any] | None:
    clip_probe = res.get("clip_audio_probe")
    if not isinstance(clip_probe, dict):
        return None
    return {
        k: clip_probe[k]
        for k in ("duration_sec", "rms", "likely_silent")
        if k in clip_probe
    }


def _model_version_string(raw: dict[str, Any]) -> str:
    hints = _collect_model_hints(raw)
    return json.dumps(hints, sort_keys=True, default=str)[:2000]


def _summarize_single(data: dict[str, Any], positive_label: int) -> dict[str, Any]:
    branch = data.get("branch")
    p_cat = data.get("p_cat")
    timing = data.get("timing_seconds") or {}
    video_path = Path(str(data.get("video", "")))
    duration: float | None = None
    try:
        if video_path.is_file():
            duration = probe_video_duration_sec(video_path)
    except Exception:
        duration = None

    p_pain: float | None = None
    decision: str | None = None
    if branch == "video":
        meta = data.get("meta_result") or {}
        p_pain = meta.get("p_pain")
        if isinstance(p_pain, (int, float)):
            p_pain = float(p_pain)
        else:
            p_pain = None
        decision = meta.get("decision")
    elif branch == "audio":
        emo = data.get("emotion") or {}
        sm = emo.get("softmax") or {}
        if isinstance(sm, dict) and "Paining" in sm:
            p_pain = float(sm["Paining"])
        top = emo.get("predicted_class")
        decision = "pain" if top == "Paining" else ("non_pain" if top else None)

    return {
        "final_decision": decision,
        "p_pain_max": p_pain,
        "p_pain_mean": p_pain,
        "branch": branch,
        "p_cat": p_cat,
        "video_duration_sec": duration,
        "timing_seconds": timing,
        "positive_label": positive_label,
    }


def _normalize_window_row(
    w: dict[str, Any],
    res: dict[str, Any],
    positive_label: int,
) -> dict[str, Any]:
    branch = res.get("branch")
    p_cat = res.get("p_cat")
    p_pain: float | None = None
    decision: str | None = None
    audio_emotion_label: str | None = None
    if branch == "video":
        meta = res.get("meta_result") or {}
        v = meta.get("p_pain")
        if isinstance(v, (int, float)):
            p_pain = float(v)
        decision = meta.get("decision")
    elif branch == "audio":
        emo = res.get("emotion") or {}
        sm = emo.get("softmax") or {}
        pred = emo.get("predicted_class")
        if isinstance(pred, str) and pred:
            audio_emotion_label = pred
        if isinstance(sm, dict) and "Paining" in sm:
            p_pain = float(sm["Paining"])
        if audio_emotion_label is None and isinstance(sm, dict) and sm:
            # Fallback: take argmax softmax class when predicted_class is missing.
            best_label, _ = max(
                ((str(k), float(v)) for k, v in sm.items() if isinstance(v, (int, float))),
                key=lambda kv: kv[1],
                default=(None, None),
            )
            audio_emotion_label = best_label
        top = emo.get("predicted_class")
        decision = "pain" if top == "Paining" else ("non_pain" if top else None)

    return {
        "window_index": w.get("window_index"),
        "start_sec": w.get("start_sec"),
        "end_sec": w.get("end_sec"),
        "branch": branch,
        "p_cat": p_cat,
        "p_pain": p_pain,
        "decision": decision,
        "audio_emotion_label": audio_emotion_label,
        "clip_audio_probe": _normalize_clip_audio_probe(res),
        "clip_video": w.get("clip_video"),
        "window_run_dir": w.get("window_run_dir"),
        "artifacts": _branch_artifact_hints(res),
        "probabilities": _extract_branch_probabilities(res),
    }


def _branch_artifact_hints(res: dict[str, Any]) -> dict[str, Any]:
    arts = res.get("artifacts") or {}
    out: dict[str, Any] = {}
    if not isinstance(arts, dict):
        return out
    for k in (
        "original_video",
        "extracted_audio",
        "separated_audio",
        "pose_video",
        "pose_npy",
        "pose_mask_npy",
    ):
        if k in arts and arts[k]:
            out[k] = arts[k]
    if res.get("separated_audio"):
        out["separated_audio"] = res.get("separated_audio")
    if res.get("pose_video"):
        out["pose_video"] = res.get("pose_video")
    return out


def _to_float_dict(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, dict):
        return out
    for k, v in value.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _extract_branch_probabilities(res: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Normalize branch probabilities for UI rendering.
    - audio_softmax: 10-class (or model-provided) emotion probabilities
    - video_pairwise: per-submodel probabilities (pain-resting, pain-angry, ...)
    - video_meta_class_probs: meta learner class probabilities
    """
    branch = res.get("branch")
    probs: dict[str, dict[str, float]] = {
        "audio_softmax": {},
        "video_pairwise": {},
        "video_pairwise_weights": {},
        "video_meta_class_probs": {},
    }
    if branch == "audio":
        emo = res.get("emotion") or {}
        probs["audio_softmax"] = _to_float_dict(emo.get("softmax"))
    elif branch == "video":
        probs["video_pairwise"] = _to_float_dict(res.get("pairwise_probs"))
        meta = res.get("meta_result") or {}
        probs["video_pairwise_weights"] = _to_float_dict(meta.get("meta_feature_weights"))
        probs["video_meta_class_probs"] = _to_float_dict(meta.get("meta_class_probs"))
    return probs


def categorize_run_dir_artifacts(run_dir: Path) -> dict[str, list[str]]:
    """Categorize files under run_dir (recursive, relative paths)."""
    cats: dict[str, list[str]] = {
        "video": [],
        "audio": [],
        "json": [],
        "other": [],
    }
    if not run_dir.is_dir():
        return cats

    vid_ext = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
    aud_ext = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    for p in run_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(run_dir)).replace("\\", "/")
        suf = p.suffix.lower()
        if suf in vid_ext:
            cats["video"].append(rel)
        elif suf in aud_ext:
            cats["audio"].append(rel)
        elif suf == ".json":
            cats["json"].append(rel)
        else:
            cats["other"].append(rel)

    for k in cats:
        cats[k].sort()
    return cats
