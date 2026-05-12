"""
Loads final_dataset_v2.jsonl into a flat pandas DataFrame.
Merges V1 and V2 pose paths. Produces statistics and plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]

LABEL_COLORS = {
    "Paining": "#e74c3c",
    "Positive_Baseline": "#2ecc71",
    "Agonistic": "#e67e22",
    "Vocalizing": "#9b59b6",
    "HuntingMind": "#3498db",
    # Audio 10-way (``audio_label_10``) — distinct from merged 5-class colors where names overlap.
    "Angry": "#c0392b",
    "Defence": "#16a085",
    "Fighting": "#d35400",
    "Happy": "#27ae60",
    "Mating": "#8e44ad",
    "MotherCall": "#2980b9",
    "Resting": "#7f8c8d",
    "Warning": "#f39c12",
}


def get_multiclass_class_names(cfg: dict) -> list[str]:
    """
    Ordered multiclass names for ``label_int`` (same order as model softmax).

    Prefer ``labels.classes`` when non-empty; otherwise ``labels.classes_5``
    for backward compatibility.
    """
    lab = cfg.get("labels") if isinstance(cfg.get("labels"), dict) else {}
    cl = lab.get("classes")
    if isinstance(cl, list) and len(cl) > 0:
        out = [str(x).strip() for x in cl if str(x).strip()]
        if out:
            return out
    c5 = lab.get("classes_5")
    if isinstance(c5, list) and len(c5) > 0:
        out = [str(x).strip() for x in c5 if str(x).strip()]
        if out:
            return out
    raise ValueError("cfg['labels'] must define non-empty ``classes`` or ``classes_5``")


def _overview_color_for_class(class_name: str) -> str:
    if class_name in LABEL_COLORS:
        return LABEL_COLORS[class_name]
    h = abs(hash(class_name))
    return plt.cm.tab20.colors[h % len(plt.cm.tab20.colors)]


def load_pose_extraction_index(cfg: dict) -> dict[str, dict]:
    """
    Load optional JSONL index: one object per line with snippet_id, pose_path,
    pose_mask_path, status (e.g. done). Returns snippet_id -> row dict.
    Missing/unset path in config → empty dict.
    """
    rel = cfg.get("data", {}).get("pose_extraction_index")
    if not rel:
        return {}
    path = REPO_ROOT / str(rel)
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            sid = o.get("snippet_id")
            if isinstance(sid, str) and sid:
                out[sid] = o
    return out


def pose_file_exists(path: str | None) -> bool:
    """Return True if path is not None and file exists on disk."""
    if path is None or not str(path).strip():
        return False
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.is_file()


def resolve_v1_pose(snippet_id: str, pose_v1_dir: Path) -> str | None:
    """
    Check if {pose_v1_dir}/{snippet_id}_pose.npy exists.
    Return relative path string or None.
    """
    rel = pose_v1_dir / f"{snippet_id}_pose.npy"
    full = REPO_ROOT / rel if not rel.is_absolute() else rel
    if full.is_file():
        return str(rel).replace("\\", "/")
    return None


def _to_repo_relative(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    p = Path(path_str)
    try:
        if p.is_absolute():
            rp = Path(p).resolve()
            root = REPO_ROOT.resolve()
            if root in rp.parents or rp == root:
                return str(rp.relative_to(root)).replace("\\", "/")
    except ValueError:
        pass
    return str(path_str).replace("\\", "/")


def load_dataset(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load final_dataset_v2.jsonl → flat DataFrame.

    Columns included (at minimum):
      snippet_id, video_id, platform, cat_id, cat_id_method,
      final_label_5, final_label_binary, audio_confidence,
      audio_high_confidence, suitable_for_training,
      pose_path, pose_mask_path, pose_n_real_frames,
      gpt_breed_guess, gpt_location_type, gpt_vet_clinic_confirmed,
      gpt_pain_indicators_visible, gpt_n_cats_visible,
      behavioral_category, duration_sec

    Filtering applied:
      - suitable_for_training == True
      - final_label_5 is not null
      - pose_path is not null (pose was extracted successfully)
      - pose file actually exists on disk

    For snippets with no pose_path but that exist in V1 manifest
    (dataset/embeddings/pose/labeled/), resolve via snippet_id:
      {pose_v1_dir}/{snippet_id}_pose.npy
    This allows training on V1 data alongside V2.

    If ``data.pose_extraction_index`` points to a JSONL file (e.g.
    dataset_construction/reports/pose_extraction_index.jsonl), paths from
    ``done`` rows are tried **before** manifest ``pose_path`` so training
    uses the 17-keypoint v2 extraction rather than legacy manifest paths
    (e.g. ``superanimal_v2`` with 39 joints). Falls back to manifest, then V1.

    Add column: pose_version ("v1" | "v2")
    Add column: label_int (integer index into the configured multiclass list)
    Add column: binary_label_int (0=No_Pain, 1=Pain)

    Log summary:
      "Loaded N records: N_v1 V1 poses + N_v2 V2 poses"
      "Dropped N: N unsuitable, N no_label, N no_pose, N file_missing"

    Return filtered DataFrame sorted by snippet_id.
    """
    manifest_path = REPO_ROOT / cfg["data"]["manifest"]
    pose_v1_rel = Path(cfg["data"]["pose_v1_dir"])
    pose_v2_rel = Path(cfg["data"]["pose_v2_dir"])
    pose_index = load_pose_extraction_index(cfg)
    classes_mc: list[str] = get_multiclass_class_names(cfg)
    binary_pain = cfg["labels"]["binary_pain_class"]
    label_field = cfg["labels"]["label_field"]

    rows: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    n0 = len(df)

    unsuitable = int((df["suitable_for_training"] != True).sum()) if "suitable_for_training" in df.columns else n0
    mask_suitable = df["suitable_for_training"] == True if "suitable_for_training" in df.columns else pd.Series([False] * len(df))
    df = df.loc[mask_suitable].copy()

    no_label = 0
    if label_field in df.columns:
        nl = df[label_field].isna() | (df[label_field].astype(str).str.strip() == "")
        no_label = int(nl.sum())
        df = df.loc[~nl].copy()
    else:
        df = df.iloc[0:0].copy()

    # Resolve pose paths and pose_version
    pose_paths: list[str | None] = []
    pose_versions: list[str] = []
    pose_mask_paths: list[str | None] = []
    no_pose = 0
    file_missing = 0
    resolved_from_index = 0

    def _mask_rel_if_exists(raw_m) -> str | None:
        if raw_m is None or (isinstance(raw_m, float) and pd.isna(raw_m)):
            return None
        m = _to_repo_relative(str(raw_m).strip()) if str(raw_m).strip() else None
        return m if m and pose_file_exists(m) else None

    def _infer_version(resolved_path: str) -> str:
        pfull = REPO_ROOT / resolved_path if not Path(resolved_path).is_absolute() else Path(resolved_path)
        try:
            rp = pfull.resolve()
            v1root = (REPO_ROOT / pose_v1_rel).resolve()
            v2root = (REPO_ROOT / pose_v2_rel).resolve()
            if v1root in rp.parents or rp == v1root:
                return "v1"
            if v2root in rp.parents or rp == v2root:
                return "v2"
            s = str(rp).replace("\\", "/")
            if "superanimal_v2" in s or "/pose/v2" in s.lower():
                return "v2"
            if "labeled" in s or "/pose/labeled" in s:
                return "v1"
        except ValueError:
            pass
        return "v2"

    for _, row in df.iterrows():
        sid = row.get("snippet_id")
        raw_pose = row.get("pose_path")
        rel_pose = _to_repo_relative(raw_pose) if raw_pose else None

        resolved: str | None = None
        version = "v2"
        mask_rel: str | None = None

        # Prefer pose_extraction_index over manifest paths so we use 17-kp v2 `.npy`
        # (manifest may still point at legacy `superanimal_v2` 39-kp arrays).
        if pose_index and isinstance(sid, str):
            ent = pose_index.get(sid)
            st = str(ent.get("status", "")).lower() if ent else ""
            if ent and st == "done":
                rp = _to_repo_relative(ent.get("pose_path"))
                rm = _to_repo_relative(ent.get("pose_mask_path")) if ent.get("pose_mask_path") else None
                if rp and pose_file_exists(rp):
                    resolved = rp
                    version = _infer_version(resolved)
                    mask_rel = rm if rm and pose_file_exists(rm) else None
                    resolved_from_index += 1

        if resolved is None and rel_pose and pose_file_exists(rel_pose):
            resolved = rel_pose
            version = _infer_version(resolved)
            mask_rel = _mask_rel_if_exists(row.get("pose_mask_path"))
        elif resolved is None:
            if isinstance(sid, str) and sid:
                v1 = resolve_v1_pose(sid, pose_v1_rel)
                if v1 and pose_file_exists(v1):
                    resolved = v1
                    version = "v1"
                    v1m = pose_v1_rel / f"{sid}_pose_mask.npy"
                    v1m_rel = str(v1m).replace("\\", "/")
                    mask_rel = v1m_rel if pose_file_exists(v1m_rel) else _mask_rel_if_exists(row.get("pose_mask_path"))

            if resolved is None:
                if rel_pose and not pose_file_exists(rel_pose):
                    file_missing += 1
                else:
                    no_pose += 1
                pose_paths.append(None)
                pose_versions.append("none")
                pose_mask_paths.append(None)
                continue

        pose_paths.append(resolved)
        pose_versions.append(version)
        pose_mask_paths.append(mask_rel)

    df = df.copy()
    df["pose_path"] = pose_paths
    df["pose_version"] = pose_versions
    df["pose_mask_path"] = pose_mask_paths

    valid = df["pose_path"].notna()
    df = df.loc[valid].copy()

    # label_int / binary_label_int
    name_to_idx = {c: i for i, c in enumerate(classes_mc)}

    def _lab_int(x):
        if x not in name_to_idx:
            return -1
        return name_to_idx[x]

    df["label_int"] = df[label_field].map(_lab_int)
    df = df.loc[df["label_int"] >= 0].copy()

    def _bin_int(row):
        if "final_label_binary" in row.index and pd.notna(row["final_label_binary"]):
            return 1 if str(row["final_label_binary"]).strip() == "Pain" else 0
        return 1 if row[label_field] == binary_pain else 0

    df["binary_label_int"] = df.apply(_bin_int, axis=1)

    df["pose_path"] = df["pose_path"].apply(lambda x: str(x).replace("\\", "/") if x is not None else None)
    if "pose_mask_path" in df.columns:
        df["pose_mask_path"] = df["pose_mask_path"].apply(
            lambda x: str(x).replace("\\", "/") if x is not None and str(x).strip() else None
        )

    if pose_index and "snippet_id" in df.columns:

        def _fill_nrf(row):
            v = row.get("pose_n_real_frames")
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
            sid = row.get("snippet_id")
            if isinstance(sid, str) and sid in pose_index:
                nrf = pose_index[sid].get("n_real_frames")
                if nrf is not None:
                    return int(nrf)
            return v

        if "pose_n_real_frames" in df.columns:
            df["pose_n_real_frames"] = df.apply(_fill_nrf, axis=1)
        else:

            def _nrf_from_idx(sid):
                if not isinstance(sid, str) or sid not in pose_index:
                    return None
                nrf = pose_index[sid].get("n_real_frames")
                return int(nrf) if nrf is not None else None

            df["pose_n_real_frames"] = df["snippet_id"].map(_nrf_from_idx)

    df = df.sort_values("snippet_id").reset_index(drop=True)

    ddata = cfg.get("data")
    vq = ddata.get("vitpose_qc") if isinstance(ddata, dict) else None
    vq = vq if isinstance(vq, dict) else {}
    if vq.get("filter", False) or vq.get("attach_enrichment_columns", False):
        from .vitpose_qc import load_vitpose_enrichment

        rel = str(
            vq.get("enrichment", "src/dataset_construction/reports/vitpose_qc_enrichment.jsonl")
        ).strip()
        enr_path = Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel
        enr = load_vitpose_enrichment(enr_path)
        if vq.get("filter", False) and not enr:
            raise FileNotFoundError(
                f"data.vitpose_qc.filter is true but enrichment is missing or empty: {enr_path} "
                "(build it with: python model_training_v2/scripts/build_vitpose_qc_enrichment.py)"
            )
        default_ok = bool(vq.get("default_ok_if_missing", True))
        n_before = len(df)

        if vq.get("attach_enrichment_columns", False) and enr:
            def _e(sid) -> dict:
                return (enr.get(str(sid)) or {})  # type: ignore[return-value]

            def _mconf(sid):
                return _e(sid).get("mean_conf")

            def _yr(sid):
                return _e(sid).get("y_range")

            def _qok(sid) -> bool:
                if str(sid) in enr:
                    return bool(_e(sid).get("vitpose_qc_ok", True))
                return default_ok

            def _rason(sid) -> str:
                t = _e(sid).get("vitpose_qc_exclusion_reasons")
                if t is None:
                    return ""
                if isinstance(t, (list, tuple)):
                    return ";".join(str(x) for x in t)
                return str(t)

            df["vitpose_mean_conf"] = df["snippet_id"].map(_mconf)
            df["vitpose_y_range"] = df["snippet_id"].map(_yr)
            df["vitpose_qc_ok"] = df["snippet_id"].map(_qok)
            df["vitpose_qc_exclusion_reasons"] = df["snippet_id"].map(_rason)

        if vq.get("filter", False) and enr:
            def _kept(sid) -> bool:
                r = enr.get(str(sid))
                if r is None:
                    return default_ok
                return bool(r.get("vitpose_qc_ok", True))

            m = df["snippet_id"].map(_kept)
            n_drop = int((~m).sum())
            df = df.loc[m].copy()
            logger.info(
                "ViTPose QC: kept %d / %d (dropped %d) using %s; default_ok_if_missing=%s",
                len(df),
                n_before,
                n_drop,
                enr_path,
                default_ok,
            )
            df = df.reset_index(drop=True)

    n_v1 = int((df["pose_version"] == "v1").sum())
    n_v2 = int((df["pose_version"] == "v2").sum())
    n = len(df)

    logger.info("Loaded %d records: %d V1 poses + %d V2 poses", n, n_v1, n_v2)
    if resolved_from_index > 0:
        logger.info("Resolved %d pose paths from pose_extraction_index", resolved_from_index)
    dropped = n0 - n
    logger.info(
        "Dropped %d: %d unsuitable, %d no_label, %d no_pose, %d file_missing",
        dropped,
        unsuitable,
        no_label,
        no_pose,
        file_missing,
    )

    return df


def resolve_dlc_h5_path(
    dlc_dir: Path | str,
    snippet_id: str,
    suffix: str,
) -> Path | None:
    """
    Resolve DeepLabCut HDF5 path for ``snippet_id``.

    1. Try exact ``{snippet_id}{suffix}`` (suffix should include ``.h5``, usually starting with ``_``).
    2. Else glob ``{snippet_id}_*.h5``. If several files exist, keep those whose names **end with**
       ``suffix``, then prefer names **without** ``_labeled_`` (clean export), then without the
       long ``labeled_before_adapt`` duplicate-stem pattern, then shortest basename.
    """
    import warnings

    dlc_dir = Path(dlc_dir)
    if not snippet_id or not dlc_dir.is_dir():
        return None
    suf = str(suffix).strip()
    if suf and not suf.endswith(".h5"):
        suf = suf + ".h5"
    exact = dlc_dir / (f"{snippet_id}{suf}" if suf else f"{snippet_id}.h5")
    if exact.is_file():
        return exact
    matches = sorted(dlc_dir.glob(f"{snippet_id}_*.h5"))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]

    ending = suf if suf.startswith("_") else f"_{suf}" if suf else ""
    if ending and not ending.endswith(".h5"):
        ending = ending + ".h5"
    candidates = [m for m in matches if ending and m.name.endswith(ending)]
    if not candidates:
        candidates = list(matches)
    if len(candidates) == 1:
        return candidates[0]

    def _pick_key(p: Path) -> tuple:
        n = p.name
        # Prefer no "_labeled_" (minimal export) over adapt snapshots.
        has_labeled = 1 if "_labeled_" in n else 0
        # DLC sometimes writes a long composite: ...fpn_labeled_before_adapt_...fpn.h5 — avoid when a shorter labeled variant exists.
        before_adapt_dup = 1 if "labeled_before_adapt" in n else 0
        return (has_labeled, before_adapt_dup, len(n), n)

    candidates.sort(key=_pick_key)
    chosen = candidates[0]
    if len(candidates) > 1:
        warnings.warn(
            f"Multiple HDF5 for {snippet_id!r} ({len(candidates)} with suffix {ending!r}); "
            f"picking: {chosen.name!r} (prefers no _labeled_, avoids labeled_before_adapt dup, then shortest). "
            "Remove extras or set a longer data.dlc_h5_suffix to be strict.",
            stacklevel=2,
        )
    return chosen


def load_dataset_for_deeplabcut(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load manifest rows suitable for training with labels, but require DeepLabCut ``.h5``
    under ``cfg['data']['dlc_dir']`` resolved via :func:`resolve_dlc_h5_path`.

    Adds ``dlc_h5_path`` (repo-relative), ``label_int``, ``binary_label_int``.
    Does not populate ``pose_path`` for training (DLC uses ``dlc_h5_path`` only).
    """
    manifest_path = REPO_ROOT / cfg["data"]["manifest"]
    classes_mc: list[str] = get_multiclass_class_names(cfg)
    binary_pain = cfg["labels"]["binary_pain_class"]
    label_field = cfg["labels"]["label_field"]
    _dd = Path(str(cfg["data"]["dlc_dir"]))
    dlc_dir = _dd if _dd.is_absolute() else REPO_ROOT / _dd
    h5_suffix = str(cfg["data"].get("dlc_h5_suffix", "")).strip()

    rows: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    n0 = len(df)

    unsuitable = int((df["suitable_for_training"] != True).sum()) if "suitable_for_training" in df.columns else n0
    mask_suitable = df["suitable_for_training"] == True if "suitable_for_training" in df.columns else pd.Series([False] * len(df))
    df = df.loc[mask_suitable].copy()

    no_label = 0
    if label_field in df.columns:
        nl = df[label_field].isna() | (df[label_field].astype(str).str.strip() == "")
        no_label = int(nl.sum())
        df = df.loc[~nl].copy()
    else:
        df = df.iloc[0:0].copy()

    n_after_label = len(df)
    dlc_paths: list[str | None] = []
    for _, row in df.iterrows():
        sid = row.get("snippet_id")
        if not isinstance(sid, str) or not sid:
            dlc_paths.append(None)
            continue
        try:
            p = resolve_dlc_h5_path(dlc_dir, sid, h5_suffix)
        except ValueError:
            raise
        if p is None or not p.is_file():
            dlc_paths.append(None)
            continue
        try:
            rel = str(p.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            rel = str(p).replace("\\", "/")
        dlc_paths.append(rel)

    df = df.copy()
    df["dlc_h5_path"] = dlc_paths
    n_missing_h5 = int(df["dlc_h5_path"].isna().sum())
    df = df.loc[df["dlc_h5_path"].notna()].copy()

    name_to_idx = {c: i for i, c in enumerate(classes_mc)}

    def _lab_int(x):
        if x not in name_to_idx:
            return -1
        return name_to_idx[x]

    df["label_int"] = df[label_field].map(_lab_int)
    df = df.loc[df["label_int"] >= 0].copy()

    excl = (cfg.get("labels") or {}).get("exclude_classes_from_training")
    if isinstance(excl, list) and excl and label_field in df.columns:
        skip = {str(x).strip() for x in excl if str(x).strip()}
        if skip:
            n_before = len(df)
            lab_series = df[label_field].astype(str).str.strip()
            df = df.loc[~lab_series.isin(skip)].copy()
            logger.info(
                "Excluded %d rows with %s in exclude_classes_from_training: %s",
                n_before - len(df),
                label_field,
                sorted(skip),
            )

    def _bin_int(row):
        if "final_label_binary" in row.index and pd.notna(row["final_label_binary"]):
            return 1 if str(row["final_label_binary"]).strip() == "Pain" else 0
        return 1 if row[label_field] == binary_pain else 0

    df["binary_label_int"] = df.apply(_bin_int, axis=1)

    if bool(cfg.get("training", {}).get("binary_only", False)):
        df["label_int"] = df["binary_label_int"].astype(int)

    ddata = cfg.get("data") or {}
    dqc = ddata.get("dlc_qc") if isinstance(ddata, dict) else None
    dqc = dqc if isinstance(dqc, dict) else {}
    if dqc.get("filter", False):
        rel = str(
            dqc.get("enrichment", "src/dataset_construction/reports/dlc_qc_enrichment.jsonl")
        ).strip()
        enr_path = Path(rel) if Path(rel).is_absolute() else REPO_ROOT / rel
        enr: dict[str, dict] = {}
        if enr_path.is_file():
            with open(enr_path, encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _obj = json.loads(_line)
                    _sid = _obj.get("snippet_id")
                    if isinstance(_sid, str) and _sid:
                        enr[_sid] = _obj
        if not enr:
            raise FileNotFoundError(
                f"data.dlc_qc.filter is true but enrichment is missing or empty: {enr_path} "
                "(build it with: python model_training_v2/scripts/build_dlc_qc_enrichment.py)"
            )
        default_ok = bool(dqc.get("default_ok_if_missing", True))
        n_before_dqc = len(df)

        def _dlc_qc_kept(sid) -> bool:
            r = enr.get(str(sid))
            if r is None:
                return default_ok
            return bool(r.get("dlc_qc_ok", True))

        m_dqc = df["snippet_id"].map(_dlc_qc_kept)
        n_drop_dqc = int((~m_dqc).sum())
        df = df.loc[m_dqc].copy()
        logger.info(
            "DLC QC: kept %d / %d (dropped %d) using %s; default_ok_if_missing=%s",
            len(df),
            n_before_dqc,
            n_drop_dqc,
            enr_path,
            default_ok,
        )

    df = df.sort_values("snippet_id").reset_index(drop=True)

    n = len(df)
    logger.info(
        "Loaded %d DeepLabCut-ready records (after suitable+label: %d rows; missing .h5 among those: %d).",
        n,
        n_after_label,
        n_missing_h5,
    )
    return df


def print_dataset_statistics(df: pd.DataFrame, logger: logging.Logger, cfg: dict | None = None):
    """
    Print a structured summary table:

    ════════════════════════════════════════════════════════
    DATASET STATISTICS
    ════════════════════════════════════════════════════════
    Total snippets (training-ready with pose):  N
    Unique cat_ids:                             N
    Unique video_ids:                           N
    Pose V1:  N  |  Pose V2:  N

    K-CLASS DISTRIBUTION (from ``cfg`` when passed, else merged 5)
    ...

    BINARY DISTRIBUTION
    Pain:     N (X%)  |  No_Pain: N (X%)  |  Ratio: 1:X

    PLATFORM
    YouTube:     N (X%)
    TikTok:      N (X%)
    DailyMotion: N (X%)

    POSE QUALITY
    Mean real frames per clip:   X.X / 35
    Clips with all 35 real:      N (X%)
    Clips with <20 real frames:  N (X%)
    ════════════════════════════════════════════════════════
    """
    if cfg is not None:
        if bool(cfg.get("training", {}).get("binary_only", False)):
            mc_title = "BINARY TRAINING (label_int: 0=No_Pain, 1=Pain)"
            lf = "label_int"
            class_rows: list[tuple[str, int]] = [("No_Pain", 0), ("Pain", 1)]
        else:
            classes = get_multiclass_class_names(cfg)
            lf = str(cfg.get("labels", {}).get("label_field", "final_label_5"))
            mc_title = f"{len(classes)}-CLASS DISTRIBUTION ({lf})"
            class_rows = [(c, c) for c in classes]
    else:
        classes = ["Paining", "Positive_Baseline", "Agonistic", "Vocalizing", "HuntingMind"]
        lf = "final_label_5"
        mc_title = "5-CLASS DISTRIBUTION"
        class_rows = [(c, c) for c in classes]
    n = len(df)
    n_cats = df["cat_id"].nunique() if "cat_id" in df.columns else 0
    n_vids = df["video_id"].nunique() if "video_id" in df.columns else 0
    n_v1 = int((df["pose_version"] == "v1").sum()) if "pose_version" in df.columns else 0
    n_v2 = int((df["pose_version"] == "v2").sum()) if "pose_version" in df.columns else 0

    lines = [
        "",
        "════════════════════════════════════════════════════════",
        "DATASET STATISTICS",
        "════════════════════════════════════════════════════════",
        f"Total snippets (training-ready with pose):  {n}",
        f"Unique cat_ids:                             {n_cats}",
        f"Unique video_ids:                           {n_vids}",
        f"Pose V1:  {n_v1}  |  Pose V2:  {n_v2}",
        "",
        mc_title,
        "──────────────────────────────────────────────",
        f"{'Class':<20} {'Count':>7} {'%':>6} {'Unique cats':>12}",
    ]

    for disp, val in class_rows:
        sub = df[df[lf] == val] if lf in df.columns else df.iloc[0:0]
        cnt = len(sub)
        pct = 100.0 * cnt / n if n else 0.0
        uc = sub["cat_id"].nunique() if "cat_id" in sub.columns and len(sub) else 0
        lines.append(f"{str(disp):<20} {cnt:>7} {pct:>5.1f}% {uc:>12}")

    lines.extend(["", "BINARY DISTRIBUTION"])
    n_pain = int((df["binary_label_int"] == 1).sum()) if "binary_label_int" in df.columns else 0
    n_nop = n - n_pain
    p_pain = 100.0 * n_pain / n if n else 0.0
    p_nop = 100.0 * n_nop / n if n else 0.0
    ratio = (n_nop / n_pain) if n_pain > 0 else float("inf")
    ratio_s = f"1:{ratio:.2f}" if n_pain > 0 else "n/a"
    lines.append(f"Pain:     {n_pain} ({p_pain:.1f}%)  |  No_Pain: {n_nop} ({p_nop:.1f}%)  |  Ratio: {ratio_s}")

    lines.extend(["", "PLATFORM"])
    for plat, label in [("YouTube", "YouTube"), ("TikTok", "TikTok"), ("DailyMotion", "DailyMotion")]:
        m = df["platform"] == plat if "platform" in df.columns else pd.Series([], dtype=bool)
        cnt = int(m.sum())
        pct = 100.0 * cnt / n if n else 0.0
        lines.append(f"{label}:     {cnt} ({pct:.1f}%)")

    lines.extend(["", "POSE QUALITY"])
    if "pose_n_real_frames" in df.columns:
        prf = df["pose_n_real_frames"].astype(float)
        mean_rf = float(prf.mean()) if len(prf) else 0.0
        all35 = int((prf >= 35).sum())
        lt20 = int((prf < 20).sum())
        p35 = 100.0 * all35 / n if n else 0.0
        p20 = 100.0 * lt20 / n if n else 0.0
        lines.append(f"Mean real frames per clip:   {mean_rf:.1f} / 35")
        lines.append(f"Clips with all 35 real:      {all35} ({p35:.1f}%)")
        lines.append(f"Clips with <20 real frames:  {lt20} ({p20:.1f}%)")
    else:
        lines.append("(pose_n_real_frames not available)")

    lines.append("════════════════════════════════════════════════════════")

    msg = "\n".join(lines)
    logger.info(msg)


def plot_dataset_overview(df: pd.DataFrame, save_dir: Path, cfg: dict | None = None):
    """
    Generate and save 4 plots to save_dir/data_overview.png:

    Panel 1: Bar chart — multiclass distribution (from ``cfg`` when passed)
    Panel 2: Bar chart — binary Pain/No_Pain
    Panel 3: Stacked bar — platform × class distribution
    Panel 4: Histogram — pose_n_real_frames distribution
              with vertical line at 35 (full clip)

    Figure size (16, 10). seaborn whitegrid.
    Saves to save_dir/data_overview.png.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if cfg is not None:
        if bool(cfg.get("training", {}).get("binary_only", False)):
            lf = "label_int"
            plot_labels = ["No_Pain", "Pain"]
            plot_vals = [0, 1]
        else:
            plot_labels = get_multiclass_class_names(cfg)
            lf = str(cfg.get("labels", {}).get("label_field", "final_label_5"))
            plot_vals = list(plot_labels)
    else:
        plot_labels = ["Paining", "Positive_Baseline", "Agonistic", "Vocalizing", "HuntingMind"]
        lf = "final_label_5"
        plot_vals = list(plot_labels)

    ax = axes[0, 0]
    counts = [int((df[lf] == v).sum()) if lf in df.columns else 0 for v in plot_vals]
    cols = [_overview_color_for_class(str(lb)) for lb in plot_labels]
    ax.bar(plot_labels, counts, color=cols, edgecolor="black", linewidth=0.5)
    ax.set_title(f"{len(plot_labels)}-class distribution ({lf})")
    ax.set_ylabel("Count")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax = axes[0, 1]
    if "binary_label_int" in df.columns:
        bp = [int((df["binary_label_int"] == 0).sum()), int((df["binary_label_int"] == 1).sum())]
    else:
        bp = [0, 0]
    ax.bar(["No_Pain", "Pain"], bp, color=["#3498db", "#e74c3c"], edgecolor="black", linewidth=0.5)
    ax.set_title("Binary Pain / No_Pain")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    if "platform" in df.columns and lf in df.columns:
        plat_order = ["YouTube", "TikTok", "DailyMotion"]
        bottom = np.zeros(len(plat_order))
        for lb, v in zip(plot_labels, plot_vals):
            vals = []
            for p in plat_order:
                vals.append(int(((df["platform"] == p) & (df[lf] == v)).sum()))
            vals = np.array(vals, dtype=float)
            ax.bar(
                plat_order,
                vals,
                bottom=bottom,
                label=lb,
                color=_overview_color_for_class(str(lb)),
                edgecolor="white",
                linewidth=0.3,
            )
            bottom += vals
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Platform × class (stacked)")
    ax.set_ylabel("Count")

    ax = axes[1, 1]
    if "pose_n_real_frames" in df.columns:
        prf = df["pose_n_real_frames"].dropna().astype(float)
        finite = prf[np.isfinite(prf)]
        if len(finite):
            vmax = float(finite.max())
            bins = min(35, max(10, int(vmax) + 1))
            ax.hist(finite, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
            ax.axvline(35, color="crimson", linestyle="--", linewidth=2, label="35 frames")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "pose_n_real_frames: all NA",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="dimgray",
            )
    ax.set_title("pose_n_real_frames distribution")
    ax.set_xlabel("Real frames")
    ax.set_ylabel("Clips")

    plt.tight_layout()
    out = save_dir / "data_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
