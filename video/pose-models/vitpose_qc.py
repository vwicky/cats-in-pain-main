"""
ViTPose clip quality for 17-kp (T, 17, 3) arrays: confidence + mask, shared by
``audit_vitpose_clip_quality`` and :func:`load_vitpose_enrichment`, and by
:func:`apply_vitpose_qc_gates` when building a training filter.

Default gates (tunable in enrichment script / config):

- ``mean_conf >= 0.30`` — low-confidence bulk filter
- ``y_range < max_y_range`` (default 10) — catch coordinate-scale / crop Artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(repo: Path, p: str | None) -> Path | None:
    if not p or not str(p).strip():
        return None
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def load_pose_and_mask(
    repo: Path,
    row: dict,
    index_by_id: dict[str, dict] | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """
    Load (pose, mask, path_used) using the same path priority as
    :func:`model_training_v2.data_loading.load_dataset` (index ``done`` first).
    """
    sid = str(row.get("snippet_id", "")).strip()
    pose_path: Path | None = None
    mask_path: Path | None = None
    if index_by_id and sid in index_by_id:
        ent = index_by_id[sid]
        if str(ent.get("status", "")).lower() == "done":
            pose_path = _resolve_path(repo, ent.get("pose_path"))
            mask_path = _resolve_path(repo, ent.get("pose_mask_path"))
    if pose_path is None or not pose_path.is_file():
        pose_path = _resolve_path(repo, row.get("pose_path"))
        mask_path = _resolve_path(repo, row.get("pose_mask_path"))
    if pose_path is None or not pose_path.is_file():
        return np.array([]), np.array([]), None
    pose = np.load(pose_path)
    if mask_path is not None and mask_path.is_file():
        mask = np.load(mask_path)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.shape[0] != pose.shape[0]:
            mask = np.ones(pose.shape[0], dtype=bool)
    else:
        mask = np.ones(pose.shape[0], dtype=bool)
    try:
        rel = pose_path.resolve().relative_to(repo.resolve())
        used = str(rel).replace("\\", "/")
    except ValueError:
        used = str(pose_path)
    return pose, mask, used


def compute_vitpose_clip_metrics(
    pose: np.ndarray,
    mask: np.ndarray,
    *,
    visible_thr: float = 0.30,
) -> dict[str, float] | None:
    """Per-clip statistics on **real** frames (mask True). Returns None if unusable."""
    if pose.size == 0 or pose.ndim != 3 or pose.shape[-1] < 3:
        return None
    if mask.shape[0] != pose.shape[0]:
        return None
    real = pose[mask]
    if real.shape[0] == 0:
        return None
    conf = real[:, :, 2].astype(np.float64, copy=False)
    xy = real[:, :, :2].astype(np.float64, copy=False)
    mean_conf = float(np.mean(conf))
    frac_visible = float(np.mean(conf > float(visible_thr)))
    per_frame_mconf = np.mean(conf, axis=1)
    zero_frames = float(np.mean(per_frame_mconf < 0.01))
    xy_std = float(np.mean(np.std(xy, axis=0)))
    n_j = min(17, conf.shape[1])
    head_hi = min(5, n_j)
    head_conf = float(np.mean(conf[:, :head_hi])) if head_hi else float("nan")
    b0, b1 = min(5, n_j), min(7, n_j)
    body_conf = float(np.mean(conf[:, b0:b1])) if b1 > b0 else float("nan")
    y = xy[:, :, 1]
    y_range = float(np.max(y) - np.min(y)) if y.size else 0.0
    return {
        "mean_conf": mean_conf,
        "frac_visible": frac_visible,
        "zero_frame_rate": zero_frames,
        "xy_std": xy_std,
        "head_conf": head_conf,
        "body_conf": body_conf,
        "y_range": y_range,
    }


def apply_vitpose_qc_gates(
    metrics: dict[str, float],
    *,
    min_mean_conf: float = 0.30,
    max_y_range: float = 10.0,
) -> tuple[bool, list[str]]:
    """
    Returns (ok, exclusion_reasons). If not ok, reasons are stable machine strings.
    """
    reasons: list[str] = []
    m = float(metrics.get("mean_conf", 0.0))
    yr = float(metrics.get("y_range", 0.0))
    if m < min_mean_conf:
        reasons.append(f"mean_conf_lt_{min_mean_conf:.2f}")
    if yr >= max_y_range:
        reasons.append(f"y_range_geq_{max_y_range:g}")
    return (len(reasons) == 0, reasons)


def load_vitpose_enrichment(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Read JSONL ``{snippet_id, vitpose_qc_ok, ...}`` into a dict.
    If path missing, returns {}.
    """
    p = path if path.is_absolute() else REPO_ROOT / path
    if not p.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sid = o.get("snippet_id")
            if isinstance(sid, str) and sid:
                out[sid] = o
    return out
