"""
Load DeepLabCut / SuperAnimal HDF5 pose tables into fixed (T, 39, 3) arrays.

Channel layout matches ViT pose: x, y, likelihood as confidence for normalization.
All HDF5 MultiIndex quirks stay in this module.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy: importing ``superanimal_quadruped_stgcn_graph`` pulls ``models`` package which
# eagerly imports torch-backed modules. Clip-QC helpers only need HDF5 likelihood columns.
_kp_spec_cache: tuple[list[str], int] | None = None


def _kp_spec() -> tuple[list[str], int]:
    global _kp_spec_cache
    if _kp_spec_cache is None:
        from quadruped_skeleton_spec import  (  # noqa: PLC0415
            KEYPOINTS,
            NUM_KEYPOINTS,
        )

        _kp_spec_cache = (KEYPOINTS, int(NUM_KEYPOINTS))
    return _kp_spec_cache


def dlc_expected_num_keypoints() -> int:
    """Number of keypoints in the SuperAnimal quadruped DLC spec (lazy-loads graph/torch stack)."""
    return _kp_spec()[1]

_UNDERSCORE_RE = re.compile(r"[\s\-]+")
_MULTI_USCORE = re.compile(r"_+")


def normalize_bodypart_name(name: str) -> str:
    """
    Canonical form for matching HDF5 bodypart labels to ``KEYPOINTS``.

    Handles inconsistent casing/spacing from DLC / SuperAnimal versions.
    """
    s = str(name).strip().lower()
    s = _UNDERSCORE_RE.sub("_", s)
    s = _MULTI_USCORE.sub("_", s).strip("_")
    return s


# DLC / SuperAnimal spellings (normalized) → canonical ``KEYPOINTS`` token.
# SuperAnimal HDF5 sometimes uses ``thai`` (typo) for ``thigh``, and alternate spine/belly names.
# Re-run ``--inspect-bodyparts`` after edits; ``belly_top`` may still be absent in some exports.
BODYPART_ALIASES: dict[str, str] = {
    "back_left_thai": "back_left_thigh",
    "back_right_thai": "back_right_thigh",
    "front_left_thai": "front_left_thigh",
    "front_right_thai": "front_right_thigh",
    "body_middle_left": "back_middle_2",
    "body_middle_right": "back_middle",
    "tail_end": "belly_bottom_2",
}


def _alias_to_spec_token(norm: str) -> str:
    return BODYPART_ALIASES.get(norm, norm)


def _maybe_fill_belly_top_from_neighbors(
    pose: np.ndarray,
    keypoint_names: list[str],
    by_bp: dict[str, dict[str, np.ndarray]],
    missing_spec: list[str],
) -> bool:
    """
    SuperAnimal HDF5 often omits ``belly_top``. Approximate with midpoint of ``back_base`` and
    ``belly_bottom`` (same T), using average likelihood as a weak confidence proxy.
    Returns True if synthetic fill was applied.
    """
    if "belly_top" not in keypoint_names or "back_base" not in keypoint_names or "belly_bottom" not in keypoint_names:
        return False
    j_bt = keypoint_names.index("belly_top")
    if np.any(pose[:, j_bt, 2] > 0):
        return False
    k_bb = _alias_to_spec_token(normalize_bodypart_name("back_base"))
    k_bot = _alias_to_spec_token(normalize_bodypart_name("belly_bottom"))
    if k_bb not in by_bp or k_bot not in by_bp:
        return False
    T = pose.shape[0]
    xb = np.asarray(by_bp[k_bb]["x"], dtype=np.float32).reshape(-1)[:T]
    yb = np.asarray(by_bp[k_bb]["y"], dtype=np.float32).reshape(-1)[:T]
    lb = np.asarray(by_bp[k_bb]["likelihood"], dtype=np.float32).reshape(-1)[:T]
    xo = np.asarray(by_bp[k_bot]["x"], dtype=np.float32).reshape(-1)[:T]
    yo = np.asarray(by_bp[k_bot]["y"], dtype=np.float32).reshape(-1)[:T]
    lo = np.asarray(by_bp[k_bot]["likelihood"], dtype=np.float32).reshape(-1)[:T]
    t_use = min(T, xb.size, yb.size, lb.size, xo.size, yo.size, lo.size)
    pose[:t_use, j_bt, 0] = 0.5 * (xb[:t_use] + xo[:t_use])
    pose[:t_use, j_bt, 1] = 0.5 * (yb[:t_use] + yo[:t_use])
    pose[:t_use, j_bt, 2] = 0.5 * (lb[:t_use] + lo[:t_use])
    if "belly_top" in missing_spec:
        missing_spec.remove("belly_top")
    return True


def list_hdf_bodypart_names(h5_path: str | Path) -> list[str]:
    """
    Return sorted unique raw bodypart labels from a DLC HDF5 (as stored in the file).

    Uses the ``bodyparts`` index level when present; otherwise infers from column tuples.
    """
    path = Path(h5_path)
    df = pd.read_hdf(path)
    mi = df.columns
    if not isinstance(mi, pd.MultiIndex):
        raise ValueError(f"Expected MultiIndex columns, got {type(mi)}")
    names = [n for n in (mi.names or []) if n is not None]
    if "bodyparts" in names:
        raw = sorted({str(x) for x in mi.get_level_values("bodyparts").unique() if str(x).strip()})
        return raw
    seen: set[str] = set()
    for col in mi:
        tup = tuple(col)
        if len(tup) < 2:
            continue
        if str(tup[-1]).lower() not in ("x", "y", "likelihood"):
            continue
        seen.add(str(tup[-2]))
    return sorted(seen)


def compare_bodyparts_to_spec(
    h5_path: str | Path,
    *,
    keypoint_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare HDF5 bodypart names (normalized) to ``KEYPOINTS``.

    Returns keys: ``hdf_raw``, ``hdf_norm``, ``spec_norm_keys``, ``matched_count``,
    ``missing_in_hdf`` (spec joints with no column), ``extra_in_hdf`` (HDF only).
    """
    keypoint_names = keypoint_names or _kp_spec()[0]
    raw = list_hdf_bodypart_names(h5_path)
    hdf_tokens = sorted({_alias_to_spec_token(normalize_bodypart_name(r)) for r in raw})
    spec_tokens = {_alias_to_spec_token(normalize_bodypart_name(k)) for k in keypoint_names}
    spec_norm = {normalize_bodypart_name(k): k for k in keypoint_names}
    matched = [k for k in keypoint_names if _alias_to_spec_token(normalize_bodypart_name(k)) in hdf_tokens]
    missing = [k for k in keypoint_names if _alias_to_spec_token(normalize_bodypart_name(k)) not in hdf_tokens]
    extra = [n for n in hdf_tokens if n not in spec_tokens]
    return {
        "hdf_raw": raw,
        "hdf_norm": hdf_tokens,
        "spec_norm_keys": sorted(spec_norm.keys()),
        "matched_count": len(matched),
        "missing_in_hdf": missing,
        "extra_in_hdf": extra,
    }


def _parse_multindex_columns(df: pd.DataFrame) -> dict[tuple[str | None, str], dict[str, np.ndarray]]:
    """
    Group HDF columns into {(individual|None, bodypart_norm): {'x','y','likelihood' -> (T,)}}.
    """
    mi = df.columns
    if not isinstance(mi, pd.MultiIndex):
        raise ValueError(f"Expected MultiIndex columns in DLC HDF5, got {type(mi)}")

    out: dict[tuple[str | None, str], dict[str, np.ndarray]] = defaultdict(dict)
    for col in mi:
        tup = tuple(col)
        last = str(tup[-1]).lower()
        if last not in ("x", "y", "likelihood"):
            continue
        if len(tup) < 2:
            continue
        bodypart_raw = str(tup[-2])
        bp_norm = normalize_bodypart_name(bodypart_raw)
        if len(tup) >= 4:
            individual = str(tup[-3])
        else:
            individual = None
        key = (individual, bp_norm)
        arr = df[col].to_numpy(dtype=np.float64, copy=False)
        if last == "likelihood":
            out[key]["likelihood"] = arr
        elif last == "x":
            out[key]["x"] = arr
        else:
            out[key]["y"] = arr
    return out


def load_dlc_h5_tensor(
    h5_path: str | Path,
    *,
    expected_joints: int | None = None,
    keypoint_names: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load one DLC HDF5 file → (T, J, 3) float32 in ``keypoint_names`` order (default: KEYPOINTS).

    Missing bodyparts are zero-filled; missing x/y use 0 with likelihood 0.

    Returns:
      pose: (T, J, 3) float32
      meta: ``unknown_bodyparts``, ``missing_spec_bodyparts``, ``n_warnings``, ``T_raw``,
      ``chosen_individual``, or ``error`` on read failure.

    For clip QC thresholds, call ``clip_quality_stats`` / ``passes_clip_quality`` on ``pose``.
    """
    spec_kps, spec_nj = _kp_spec()
    if keypoint_names is None:
        keypoint_names = spec_kps
    if expected_joints is None:
        expected_joints = spec_nj
    if len(keypoint_names) != expected_joints:
        raise ValueError("keypoint_names length must match expected_joints")

    path = Path(h5_path)
    try:
        df = pd.read_hdf(path)
    except Exception as e:
        logger.warning("read_hdf failed for %s: %s", path, e)
        z = np.zeros((0, expected_joints, 3), dtype=np.float32)
        return z, {
            "error": str(e),
            "unknown_bodyparts": [],
            "missing_spec_bodyparts": list(keypoint_names),
            "n_warnings": 1,
        }

    T = len(df)
    if T == 0:
        z = np.zeros((0, expected_joints, 3), dtype=np.float32)
        return z, {
            "unknown_bodyparts": [],
            "missing_spec_bodyparts": list(keypoint_names),
            "n_warnings": 1,
        }

    groups = _parse_multindex_columns(df)
    individuals = sorted({k[0] for k in groups if k[0] is not None})
    chosen_ind: str | None = individuals[0] if individuals else None

    by_bp: dict[str, dict[str, np.ndarray]] = {}
    for (ind, bp_norm), chans in groups.items():
        if "x" not in chans or "y" not in chans or "likelihood" not in chans:
            continue
        if chosen_ind is not None and ind is not None and ind != chosen_ind:
            continue
        bp_key = _alias_to_spec_token(bp_norm)
        by_bp[bp_key] = chans

    unknown: list[str] = []

    spec_norm = {normalize_bodypart_name(k): k for k in keypoint_names}
    pose = np.zeros((T, expected_joints, 3), dtype=np.float32)
    missing_spec: list[str] = []
    used_dlc: set[str] = set()

    for j, kp in enumerate(keypoint_names):
        kn = _alias_to_spec_token(normalize_bodypart_name(kp))
        if kn not in by_bp:
            missing_spec.append(kp)
            continue
        ch = by_bp[kn]
        x = np.asarray(ch["x"], dtype=np.float32).reshape(-1)[:T]
        y = np.asarray(ch["y"], dtype=np.float32).reshape(-1)[:T]
        lh = np.asarray(ch["likelihood"], dtype=np.float32).reshape(-1)[:T]
        t_use = min(T, x.shape[0], y.shape[0], lh.shape[0])
        pose[:t_use, j, 0] = x[:t_use]
        pose[:t_use, j, 1] = y[:t_use]
        pose[:t_use, j, 2] = lh[:t_use]
        used_dlc.add(kn)

    synthetic_belly_top = _maybe_fill_belly_top_from_neighbors(pose, keypoint_names, by_bp, missing_spec)

    for kn in by_bp:
        if kn not in spec_norm and kn not in used_dlc:
            unknown.append(kn)

    n_warnings = int(bool(unknown)) + len(missing_spec)
    meta: dict[str, Any] = {
        "unknown_bodyparts": sorted(set(unknown)),
        "missing_spec_bodyparts": missing_spec,
        "n_warnings": n_warnings,
        "T_raw": T,
        "chosen_individual": chosen_ind,
        "synthetic_belly_top": synthetic_belly_top,
    }
    if missing_spec:
        logger.debug(
            "%s: %d spec bodyparts missing in HDF5 (zero-filled): %s%s",
            path.name,
            len(missing_spec),
            missing_spec[:8],
            "..." if len(missing_spec) > 8 else "",
        )
    return pose, meta


def _empty_clip_quality_stats() -> dict[str, float | bool | int]:
    """Defaults when HDF5 is missing or pose is empty (all keys training / enrich scripts may read)."""
    return {
        "clip_mean_likelihood": 0.0,
        "mean_dlc_likelihood": 0.0,
        "fraction_high_conf_frames": 0.0,
        "frac_high_conf_frames_0p5": 0.0,
        "clip_duration_frames": 0,
        "n_visible_joints_0p4": 0,
        "has_occlusion": False,
        "passes_mean": False,
    }


def clip_quality_stats_from_hdf5(
    h5_path: str | Path,
    *,
    clip_mean_likelihood_threshold: float,
    per_frame_threshold: float,
) -> dict[str, float | bool | int]:
    """
    Clip QC using only likelihood columns — avoids full (T,39,3) assembly (faster for manifests).

    Extended fields (same as :func:`clip_quality_stats`): ``mean_dlc_likelihood`` (duplicate of
    clip mean), ``frac_high_conf_frames_0p5``, ``clip_duration_frames``, ``n_visible_joints_0p4``,
    ``has_occlusion`` (any frame where >30% of joints have LH < 0.3).

    When the HDF5 has an ``individuals`` MultiIndex level (multi-animal tracking), only the
    first individual (e.g. ``animal0``) is used — matching the behaviour of
    :func:`load_dlc_h5_tensor`. Undetected animal slots store sentinel ``-1.0`` likelihoods
    and must be excluded to avoid corrupting clip-level statistics.
    """
    path = Path(h5_path)
    try:
        df = pd.read_hdf(path)
    except Exception as e:
        logger.debug("read_hdf failed for clip QC %s: %s", path, e)
        return dict(_empty_clip_quality_stats())
    mi = df.columns
    if not isinstance(mi, pd.MultiIndex):
        return dict(_empty_clip_quality_stats())

    # Filter to the primary individual when multi-animal tracking is present so that
    # sentinel -1.0 likelihood values for unused animal slots do not corrupt the stats.
    if mi.names and "individuals" in mi.names:
        ind_level = list(mi.names).index("individuals")
        individuals_present = sorted({str(tuple(c)[ind_level]) for c in mi})
        chosen_ind = individuals_present[0] if individuals_present else None
        if chosen_ind is not None:
            lh_cols = [
                c for c in mi
                if str(tuple(c)[-1]).lower() == "likelihood"
                and str(tuple(c)[ind_level]) == chosen_ind
            ]
        else:
            lh_cols = [c for c in mi if str(tuple(c)[-1]).lower() == "likelihood"]
    else:
        lh_cols = [c for c in mi if str(tuple(c)[-1]).lower() == "likelihood"]

    if not lh_cols:
        return dict(_empty_clip_quality_stats())
    mat = df.loc[:, lh_cols].to_numpy(dtype=np.float64, copy=False)
    if mat.size == 0:
        return dict(_empty_clip_quality_stats())
    t, n = mat.shape
    fake = np.zeros((t, n, 3), dtype=np.float64)
    fake[..., 2] = mat
    return clip_quality_stats(
        fake,
        clip_mean_likelihood_threshold=clip_mean_likelihood_threshold,
        per_frame_threshold=per_frame_threshold,
    )


def passes_clip_quality_hdf5(
    h5_path: str | Path,
    *,
    clip_mean_likelihood_threshold: float,
    min_fraction_high_conf_frames: float,
    per_frame_threshold: float | None = None,
) -> bool:
    """Same gates as ``passes_clip_quality`` but reads only likelihood columns from HDF5."""
    pft = (
        float(per_frame_threshold)
        if per_frame_threshold is not None
        else float(clip_mean_likelihood_threshold)
    )
    st = clip_quality_stats_from_hdf5(
        h5_path,
        clip_mean_likelihood_threshold=clip_mean_likelihood_threshold,
        per_frame_threshold=pft,
    )
    mean_ok = float(st["clip_mean_likelihood"]) >= float(clip_mean_likelihood_threshold)
    frac_ok = float(st["fraction_high_conf_frames"]) >= float(min_fraction_high_conf_frames)
    return bool(mean_ok and frac_ok)


def clip_quality_stats(
    pose_txj3: np.ndarray,
    *,
    clip_mean_likelihood_threshold: float,
    per_frame_threshold: float,
) -> dict[str, float | bool | int]:
    """
    Compute QC stats on raw (T, J, 3) before temporal padding to n_frames.

    - clip_mean_likelihood: mean over all T×J of channel 2 (ignores NaN → nanmean)
    - fraction_high_conf_frames: fraction of frames where mean_j likelihood >= per_frame_threshold
    - passes_mean: clip_mean >= clip_mean_likelihood_threshold (caller compares both gates)
    """
    if pose_txj3.size == 0:
        return dict(_empty_clip_quality_stats())
    lh = pose_txj3[..., 2].astype(np.float64, copy=False)
    t, _j = lh.shape
    clip_mean = float(np.nanmean(lh))
    frame_means = np.nanmean(lh, axis=1)
    frac = float(np.mean(frame_means >= float(per_frame_threshold))) if t else 0.0
    passes_mean = clip_mean >= float(clip_mean_likelihood_threshold)
    per_joint_mean = np.nanmean(lh, axis=0)
    n_visible = int(np.sum(per_joint_mean > 0.4))
    frac_joint_below_03 = np.mean(lh < 0.3, axis=1) if t else np.array([], dtype=np.float64)
    has_occ = bool(np.any(frac_joint_below_03 > 0.3)) if t else False
    frac_0p5 = float(np.mean(frame_means > 0.5)) if t else 0.0
    return {
        "clip_mean_likelihood": clip_mean,
        "mean_dlc_likelihood": clip_mean,
        "fraction_high_conf_frames": frac,
        "frac_high_conf_frames_0p5": frac_0p5,
        "clip_duration_frames": int(t),
        "n_visible_joints_0p4": n_visible,
        "has_occlusion": has_occ,
        "passes_mean": passes_mean,
    }


def passes_clip_quality(
    pose_txj3: np.ndarray,
    *,
    clip_mean_likelihood_threshold: float,
    min_fraction_high_conf_frames: float,
    per_frame_threshold: float | None = None,
) -> bool:
    """True if clip passes both mean and fraction gates (for exclude filter)."""
    pft = (
        float(per_frame_threshold)
        if per_frame_threshold is not None
        else float(clip_mean_likelihood_threshold)
    )
    st = clip_quality_stats(
        pose_txj3,
        clip_mean_likelihood_threshold=clip_mean_likelihood_threshold,
        per_frame_threshold=pft,
    )
    mean_ok = float(st["clip_mean_likelihood"]) >= float(clip_mean_likelihood_threshold)
    frac_ok = float(st["fraction_high_conf_frames"]) >= float(min_fraction_high_conf_frames)
    return bool(mean_ok and frac_ok)
