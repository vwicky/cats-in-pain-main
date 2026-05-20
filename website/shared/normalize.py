from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared.multicat_params import VALID_MULTICAT_SUMMARY_STRATEGIES
from shared.p_pain import enrich_pipeline_result_inplace
from shared.video_probe import probe_video_duration_sec

WRAPPER_VERSION = "0.1.0"


def _multicat_import(repo_root: Path) -> Any:
    src = repo_root / "src"
    sp = str(src.resolve())
    if sp not in sys.path:
        sys.path.insert(0, sp)
    from inference import multicat_aggregate  # noqa: PLC0415

    return multicat_aggregate


def _normalize_cat_entries(
    res: dict[str, Any],
    *,
    positive_label: int,
) -> list[dict[str, Any]]:
    cats = res.get("cats")
    if not isinstance(cats, list) or not cats:
        return []
    out: list[dict[str, Any]] = []
    for c in cats:
        if not isinstance(c, dict):
            continue
        meta = c.get("meta_result") or {}
        p = meta.get("p_pain")
        p_f: float | None = float(p) if isinstance(p, (int, float)) else None
        probs = _extract_branch_probabilities(
            {
                "branch": "video",
                "pairwise_probs": c.get("pairwise_probs"),
                "meta_result": meta,
            }
        )
        arts: dict[str, Any] = {}
        if c.get("pose_npy"):
            arts["pose_npy"] = c.get("pose_npy")
        if c.get("pose_mask_path"):
            arts["pose_mask_npy"] = c.get("pose_mask_path")
        if c.get("pose_video"):
            arts["pose_video"] = c.get("pose_video")
        out.append(
            {
                "local_track_id": c.get("local_track_id"),
                "window_index": c.get("window_index"),
                "detection_rate_sampled": c.get("detection_rate_sampled"),
                "n_detected_frames": c.get("n_detected_frames"),
                "p_pain": p_f,
                "decision": meta.get("decision"),
                "artifacts": arts,
                "probabilities": probs,
                "pose_video_url": c.get("pose_video"),
            }
        )
    return out


def _normalize_cat_results_full_row(
    raw_cat: dict[str, Any],
    cat_index: int,
) -> dict[str, Any]:
    """Full peer row for API: probabilities from the raw per-cat pipeline dict (multicat B)."""
    meta = raw_cat.get("meta_result") or {}
    p = meta.get("p_pain")
    p_f: float | None = float(p) if isinstance(p, (int, float)) else None
    probs = _extract_branch_probabilities(raw_cat)
    arts: dict[str, Any] = {}
    if raw_cat.get("pose_npy"):
        arts["pose_npy"] = raw_cat["pose_npy"]
    if raw_cat.get("pose_mask_path"):
        arts["pose_mask_npy"] = raw_cat["pose_mask_path"]
    if raw_cat.get("pose_video"):
        arts["pose_video"] = raw_cat["pose_video"]
    return {
        "cat_index": cat_index,
        "track_id": raw_cat.get("local_track_id"),
        "local_track_id": raw_cat.get("local_track_id"),
        "window_index": raw_cat.get("window_index"),
        "detection_rate_sampled": raw_cat.get("detection_rate_sampled"),
        "n_detected_frames": raw_cat.get("n_detected_frames"),
        "p_pain": p_f,
        "decision": meta.get("decision"),
        "artifacts": arts,
        "probabilities": probs,
        "pose_video_url": raw_cat.get("pose_video"),
    }


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
            norm_windows.append(
                _normalize_window_row(w, res, positive_label, repo_root=repo_root)
            )

        mparams = data.get("multicat_params") or {}
        if mparams.get("multicat_video_only"):
            final_decision = summary.get("multicat_clip_decision")
            p_headline = summary.get("multicat_clip_headline_p_pain")
            p_max = summary.get("video_level_p_pain_max")
            p_mean = summary.get("video_level_p_pain_mean")
        else:
            final_decision = summary.get("video_level_decision_max")
            p_headline = summary.get("video_level_p_pain_max")
            p_max = summary.get("video_level_p_pain_max")
            p_mean = summary.get("video_level_p_pain_mean")

        artifacts = categorize_run_dir_artifacts(Path(str(data.get("run_dir", ""))))
        extra_summary: dict[str, Any] = {}
        if mparams.get("multicat_summary_strategy") == "majority_above_threshold":
            xa = summary.get("multicat_clip_cats_above_threshold")
            ya = summary.get("multicat_clip_cats_total")
            if isinstance(xa, (int, float)) and isinstance(ya, (int, float)):
                extra_summary["multicat_clip_prevalence_label"] = (
                    f"{int(xa)} of {int(ya)} cats above pain threshold"
                )

        return {
            "summary": {
                "final_decision": final_decision,
                "p_pain_max": p_max,
                "p_pain_mean": p_mean,
                "p_pain_headline": p_headline,
                "video_duration_sec": (data.get("split") or {}).get("video_duration_sec"),
                "n_windows": (data.get("split") or {}).get("n_windows"),
                "split": data.get("split"),
                "aggregate_summary": summary,
                "multicat_params": mparams,
                **extra_summary,
            },
            "windows": norm_windows,
            "artifacts": artifacts,
            "raw": data,
            "final_decision": final_decision,
            "p_pain_max": p_max,
            "p_pain_mean": p_mean,
            "p_pain_headline": p_headline,
            "meta": {
                "pipeline_version": pipeline_version,
                "model_version": _model_version_string(data),
                "wrapper_version": WRAPPER_VERSION,
            },
        }

    # Single-video run
    summary = _summarize_single(data, positive_label, repo_root=repo_root)
    artifacts = categorize_run_dir_artifacts(Path(str(data.get("run_dir", ""))))
    fd = summary.get("final_decision")
    pm = summary.get("p_pain_max")
    pmean = summary.get("p_pain_mean")
    ph = summary.get("p_pain_headline")

    mparams_top = data.get("multicat_params") or {}
    is_multicat_single = bool(mparams_top.get("multicat_video_only")) and (
        data.get("mode") != "split_sliding_windows"
    )
    empty_probs: dict[str, dict[str, float]] = {
        "audio_softmax": {},
        "video_pairwise": {},
        "video_pairwise_weights": {},
        "video_meta_class_probs": {},
    }
    window_probs = empty_probs if is_multicat_single else _extract_branch_probabilities(data)
    window_cats_norm = (
        None
        if is_multicat_single
        else (
            _normalize_cat_entries(data, positive_label=positive_label) or None
        )
    )
    top_level_cats: list[dict[str, Any]] | None = None
    if is_multicat_single:
        raw_tl = data.get("cats")
        if isinstance(raw_tl, list) and raw_tl:
            top_level_cats = [
                _normalize_cat_results_full_row(c, i)
                for i, c in enumerate(raw_tl)
                if isinstance(c, dict)
            ]

    win_p_pain = None if is_multicat_single else (ph if ph is not None else pm)
    win_decision = None if is_multicat_single else fd

    out: dict[str, Any] = {
        "summary": summary,
        "windows": [
            {
                "window_index": 0,
                "start_sec": 0.0,
                "end_sec": summary.get("video_duration_sec"),
                "branch": data.get("branch"),
                "p_cat": data.get("p_cat"),
                "p_pain": win_p_pain,
                "decision": win_decision,
                "artifacts": _branch_artifact_hints(data),
                "probabilities": window_probs,
                "clip_audio_probe": _normalize_clip_audio_probe(data),
                "cats": window_cats_norm,
                "multicat_headline": None if is_multicat_single else summary.get("multicat_window_headline"),
                "multicat_empty_reason": summary.get("multicat_empty_reason"),
            }
        ],
        "artifacts": artifacts,
        "raw": data,
        "final_decision": fd,
        "p_pain_max": pm,
        "p_pain_mean": pmean,
        "p_pain_headline": ph,
        "meta": {
            "pipeline_version": pipeline_version,
            "model_version": _model_version_string(data),
            "wrapper_version": WRAPPER_VERSION,
        },
    }
    if is_multicat_single:
        out["multicat_video_only"] = True
        out["multicat_cat_count"] = summary.get("multicat_cat_count")
        if out["multicat_cat_count"] is None:
            out["multicat_cat_count"] = len(data.get("cats") or [])
        out["cats"] = top_level_cats or []
    return out


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


def _summarize_single(
    data: dict[str, Any], positive_label: int, *, repo_root: Path
) -> dict[str, Any]:
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

    mparams = data.get("multicat_params") or {}
    cats = data.get("cats")

    if (
        branch == "video"
        and mparams.get("multicat_video_only")
        and isinstance(cats, list)
        and cats
    ):
        # Multicat B: no synthetic clip-level headline; full clip verdict lives per cat only.
        return {
            "final_decision": None,
            "p_pain_max": None,
            "p_pain_mean": None,
            "p_pain_headline": None,
            "multicat_window_headline": None,
            "multicat_cat_count": int(data.get("multicat_cat_count", len(cats))),
            "branch": branch,
            "p_cat": p_cat,
            "video_duration_sec": duration,
            "timing_seconds": timing,
            "positive_label": positive_label,
            "multicat_params": mparams,
        }

    if branch == "video" and mparams.get("multicat_video_only"):
        diag = data.get("multicat_diag") or {}
        nmc_raw = data.get("multicat_cat_count")
        nmc = int(nmc_raw) if isinstance(nmc_raw, int) else 0
        return {
            "final_decision": None,
            "p_pain_max": None,
            "p_pain_mean": None,
            "p_pain_headline": None,
            "multicat_window_headline": None,
            "multicat_empty_reason": diag.get("status", "no_cats"),
            "multicat_cat_count": nmc,
            "branch": branch,
            "p_cat": p_cat,
            "video_duration_sec": duration,
            "timing_seconds": timing,
            "positive_label": positive_label,
            "multicat_params": mparams,
        }

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
        "p_pain_headline": p_pain,
        "branch": branch,
        "p_cat": p_cat,
        "video_duration_sec": duration,
        "timing_seconds": timing,
        "positive_label": positive_label,
        "multicat_params": mparams,
    }


def _normalize_window_row(
    w: dict[str, Any],
    res: dict[str, Any],
    positive_label: int,
    *,
    repo_root: Path,
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

    cats_norm = _normalize_cat_entries(res, positive_label=positive_label)
    mhead: dict[str, Any] | None = None
    if cats_norm and res.get("multicat_video_only"):
        mca = _multicat_import(repo_root)
        params = res.get("multicat_params") or {}
        strat = str(params.get("multicat_summary_strategy", "coverage_weighted_mean"))
        if strat not in VALID_MULTICAT_SUMMARY_STRATEGIES:
            strat = "coverage_weighted_mean"
        th = float(params.get("multicat_decision_threshold", 0.5))
        raw_cats = res.get("cats") or []
        if isinstance(raw_cats, list):
            mhead = mca.window_headline_from_cats(
                raw_cats, strategy=strat, pain_threshold=th
            )
        hp = mhead.get("headline_p_pain")
        if isinstance(hp, (int, float)):
            p_pain = float(hp)
        decision = str(mhead.get("decision")) if mhead.get("decision") else decision

    empty_reason: str | None = None
    if res.get("multicat_video_only") and branch == "video" and not cats_norm:
        diag = res.get("multicat_diag") or {}
        empty_reason = str(diag.get("status", "no_cats"))

    row: dict[str, Any] = {
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
        "cats": cats_norm if cats_norm else None,
        "multicat_headline": mhead,
        "multicat_empty_reason": empty_reason,
    }

    prev_frac = mhead.get("multicat_prevalence_fraction") if mhead else None
    if mhead and isinstance(prev_frac, (int, float)):
        row["multicat_prevalence_fraction"] = float(prev_frac)
        row["multicat_cats_above_threshold"] = mhead.get(
            "multicat_cats_above_threshold"
        )
        row["multicat_cats_total"] = mhead.get("multicat_cats_total")
    strat = (res.get("multicat_params") or {}).get("multicat_summary_strategy")
    if strat == "majority_above_threshold" and mhead:
        x = mhead.get("multicat_cats_above_threshold")
        y = mhead.get("multicat_cats_total")
        if isinstance(x, int) and isinstance(y, int):
            row["multicat_prevalence_label"] = f"{x} of {y} cats above pain threshold"

    return row


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
