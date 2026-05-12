"""
Helpers for class-subset training: enumerate Paining-inclusive label subsets and
filter/remap manifest rows to a k-way problem; plus **binary class-pair** helpers
(all unordered pairs from a label list).
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


def iter_paining_inclusive_subsets(
    classes_5: list[str], *, pain_class: str = "Paining"
) -> list[tuple[str, ...]]:
    """
    All 15 combinations: {Paining} union any non-empty subset of the other 4 classes.

    Order: increasing number of extra classes, then lexicographic by
    the ordered list of non-paining class names (config order is respected via index).
    """
    if pain_class not in classes_5:
        raise ValueError(f"{pain_class} is not in classes_5: {classes_5}")
    order = {c: i for i, c in enumerate(classes_5)}
    rest = [c for c in classes_5 if c != pain_class]
    rest = sorted(rest, key=lambda c: order[c])
    out: list[tuple[str, ...]] = []
    for r in range(1, len(rest) + 1):
        for comb in itertools.combinations(rest, r):
            out.append((pain_class,) + comb)
    return out


def filter_remap_dataframe(
    df: pd.DataFrame,
    class_subset: tuple[str, ...],
    *,
    label_field: str = "final_label_5",
    pain_class: str = "Paining",
) -> pd.DataFrame:
    """
    Keep rows whose multi-class label is in ``class_subset``; re-index labels 0..K-1
    in **subset order** (Paining first, then remaining classes in the order they appear
    in ``class_subset``).
    """
    if pain_class not in class_subset:
        raise ValueError("class_subset must include pain_class for this experiment")
    name_to_int = {name: i for i, name in enumerate(class_subset)}
    sub = df[df[label_field].isin(class_subset)].copy()
    if sub.empty:
        return sub
    sub["label_int"] = sub[label_field].map(name_to_int).astype(int)
    sub["binary_label_int"] = (sub[label_field] == pain_class).astype(int)
    return sub


def split_val_fraction_from_cfg(cfg: dict) -> float | None:
    """
    If ``cfg['split']['val_fraction']`` is a float in (0, 1), return it; else ``None``.

    Used by :func:`first_stratified_group_split` to pick the StratifiedGroupKFold fold
    whose validation row count is closest to this fraction.
    """
    raw = cfg.get("split", {}).get("val_fraction")
    if raw is None:
        return None
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    if f <= 0.0 or f >= 1.0:
        return None
    return f


def first_stratified_group_split(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """
    StratifiedGroupKFold by ``cfg['split']['group_field']`` on ``label_int``.

    - If ``cfg['split']['val_fraction']`` is set (strictly between 0 and 1), uses
      ``n_splits ≈ round(1 / val_fraction)`` (capped by sample and group counts),
      evaluates **every** fold, and returns the (train, val) split whose validation
      set size is closest to that fraction of rows (by count).
    - Otherwise returns the **first** fold only, with ``n_folds`` from config (legacy).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if "label_int" not in df.columns:
        return (
            df.iloc[:0].copy(),
            df.iloc[:0].copy(),
            "label_int column missing (call filter_remap_dataframe first).",
        )

    gcol = cfg["split"]["group_field"]
    rs = int(cfg["split"]["random_state"])
    n_folds_cfg = int(cfg["split"].get("n_folds", 5))

    n_samples = len(df)
    n_groups = int(df[gcol].nunique())
    val_fraction = split_val_fraction_from_cfg(cfg)

    if val_fraction is not None:
        n_splits_target = max(2, int(round(1.0 / val_fraction)))
        n_folds = min(n_splits_target, n_samples, n_groups)
    else:
        n_folds = min(n_folds_cfg, n_samples, n_groups)

    if n_folds < 2:
        return (
            df.iloc[:0].copy(),
            df.iloc[:0].copy(),
            f"Cannot split: n_samples={n_samples}, n_groups={n_groups} → n_folds={n_folds}.",
        )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    X = np.zeros(len(df))
    y = df["label_int"].values
    groups = df[gcol].values

    if val_fraction is None:
        for tr_idx, va_idx in sgkf.split(X, y, groups):
            return df.iloc[tr_idx], df.iloc[va_idx], None
        return df.iloc[:0].copy(), df.iloc[:0].copy(), "no split produced"

    best_pair: tuple[np.ndarray, np.ndarray] | None = None
    best_diff = float("inf")
    for tr_idx, va_idx in sgkf.split(X, y, groups):
        frac = float(len(va_idx)) / float(n_samples)
        d = abs(frac - val_fraction)
        if d < best_diff:
            best_diff = d
            best_pair = (tr_idx, va_idx)
    if best_pair is None:
        return df.iloc[:0].copy(), df.iloc[:0].copy(), "no split produced"
    tr_idx, va_idx = best_pair
    return df.iloc[tr_idx], df.iloc[va_idx], None


def subset_to_dirname(class_subset: tuple[str, ...]) -> str:
    """Subfolder name: k2_Paining__Agonistic"""
    k = len(class_subset)
    body = "__".join(str(c).replace("/", "_") for c in class_subset)
    return f"k{k}_{body}"


def iter_sorted_class_pairs(classes: Iterable[str]) -> list[tuple[str, str]]:
    """
    All unordered pairs from distinct class names (lexicographic order on each pair).

    For 10 classes this yields C(10,2) = 45 pairs.
    """
    s = sorted({str(c).strip() for c in classes if str(c).strip()})
    return list(itertools.combinations(s, 2))


def pair_binary_neg_pos(lo: str, hi: str, *, pain_class: str = "Paining") -> tuple[str, str]:
    """
    Given two distinct class names ``lo`` < ``hi`` lexicographically, return
    ``(negative_class, positive_class)`` for ``label_int`` / ``binary_label_int``
    values 0 and 1.

    Convention:
    - If ``pain_class`` is one of the pair, it is **positive** (1) and the other is (0).
    - Otherwise the lexicographically earlier name is (0) and the later is (1).
    """
    a, b = str(lo).strip(), str(hi).strip()
    if a >= b:
        raise ValueError(f"expected lo < hi lexicographically, got {lo!r}, {hi!r}")
    if pain_class == a:
        return b, a
    if pain_class == b:
        return a, b
    return a, b


def pair_to_dirname(class_lo: str, class_hi: str) -> str:
    """Stable subfolder name for a pair, e.g. ``binary__Angry__Happy``."""
    lo, hi = sorted((str(class_lo).strip(), str(class_hi).strip()))

    def safe(x: str) -> str:
        return re.sub(r"[^\w.\-]+", "_", x, flags=re.UNICODE)

    return f"binary__{safe(lo)}__{safe(hi)}"


def filter_remap_binary_pair(
    df: pd.DataFrame,
    class_lo: str,
    class_hi: str,
    *,
    label_field: str,
    pain_class: str = "Paining",
) -> pd.DataFrame:
    """
    Keep rows whose ``label_field`` is one of the two classes; set ``label_int`` and
    ``binary_label_int`` using :func:`pair_binary_neg_pos` (``lo``/``hi`` sorted).
    """
    lo, hi = sorted((str(class_lo).strip(), str(class_hi).strip()))
    if lo == hi:
        raise ValueError("class pair must be two distinct labels")
    neg, pos = pair_binary_neg_pos(lo, hi, pain_class=pain_class)
    sub = df[df[label_field].isin((lo, hi))].copy()
    if sub.empty:
        return sub
    mp = {neg: 0, pos: 1}
    sub["label_int"] = sub[label_field].map(mp).astype(int)
    sub["binary_label_int"] = sub["label_int"].astype(int)
    return sub


def split_viable(
    train_df: pd.DataFrame, val_df: pd.DataFrame, n_classes: int, *, batch_size: int
) -> tuple[bool, str | None]:
    """Check train/val are usable for one training run."""
    if len(train_df) < 1 or len(val_df) < 1:
        return False, "empty train or val"
    tr_uid = int(train_df["label_int"].nunique()) if "label_int" in train_df.columns else 0
    if tr_uid < 2 and n_classes >= 2:
        return False, "train has a single class only"
    b_tr = int(train_df["binary_label_int"].nunique()) if "binary_label_int" in train_df else 0
    b_va = int(val_df["binary_label_int"].nunique()) if "binary_label_int" in val_df else 0
    if b_va < 2:
        return False, "val has only one binary pain label (e.g. no Paining in val)"
    if b_tr < 2:
        return False, "train has only one binary pain label"
    if len(train_df) < batch_size + 1:
        return (
            False,
            f"train n={len(train_df)} is too small for drop_last and batch_size={batch_size} "
            "(need more than batch_size to form at least one full batch with drop_last=True).",
        )
    return True, None


def filter_dataframe_by_audio_confidence(
    df: pd.DataFrame,
    confidence_col: str,
    min_confidence: float,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Keep rows with **strict** confidence above a threshold (e.g. 70% of the
    high-confidence audio model).

    Interpreting ``min_confidence`` and the column scale:

    - If the column looks like 0--1 (max ≤ 1.5), a value of 0.7 means keep if
      value > 0.7.
    - If the column looks like 0--100 (max > 1.5), a value of 0.7 means
      value > 70. A literal ``min_confidence`` in (1, 100] is a 0--100 cut
      (e.g. 70 -> value > 70) when the column is 0--100; if the column is
      0--1, that same number is treated as a percent: value > 70/100.
    """
    if confidence_col not in df.columns:
        if logger:
            logger.warning("Column %r not found; skipping audio confidence filter.", confidence_col)
        return df

    s = pd.to_numeric(df[confidence_col], errors="coerce")
    n0 = len(df)
    n_nan = int(s.isna().sum())
    v = s.dropna()
    if v.empty:
        if logger:
            logger.warning("All %s are NaN after coercion; returning empty DataFrame", confidence_col)
        return df.iloc[0:0].copy()

    hi = float(v.max())
    if min_confidence < 0:
        if logger:
            logger.warning("min_confidence < 0; skipping filter")
        return df

    if 1.0 < min_confidence <= 100.0:
        if hi > 1.5:
            cut = float(min_confidence)
            label = f"raw cut {cut} on 0-100 scale"
        else:
            cut = float(min_confidence) / 100.0
            label = f"{min_confidence}% on 0-1 scale (cut {cut})"
    elif 0.0 <= min_confidence <= 1.0:
        if hi > 1.5:
            cut = float(min_confidence) * 100.0
            label = f"fraction {min_confidence} -> {cut} on 0-100 scale"
        else:
            cut = float(min_confidence)
            label = f"fraction {min_confidence} on 0-1 scale"
    else:
        if logger:
            logger.warning("min_confidence %s invalid; skipping filter", min_confidence)
        return df

    m = s > cut
    out = df.loc[m].copy()
    if logger:
        logger.info(
            "Audio confidence filter: column=%s  %s  keep %d / %d (dropped %d, NaN %d)",
            confidence_col,
            label,
            len(out),
            n0,
            n0 - len(out),
            n_nan,
        )
    if (
        logger
        and n0 == len(out)
        and n0 > 50
        and 0.0 < min_confidence < 1.0
        and confidence_col
        not in (
            "audio_confidence",
            "audio_conf",
        )
    ):
        nuniq = int(v.nunique()) if v.size else 0
        if nuniq <= 2 and float(v.max() or 0) <= 1.0 + 1e-6:
            logger.warning(
                "All %d rows passed the >%.3g cut. Column %r looks binary (0/1) or is coerced to that—"
                "not a continuous 0-1 *score*. For a real probability threshold (YAMNet-style), use "
                "--audio-confidence-field audio_confidence. "
                "The manifest's audio_high_confidence is a precomputed boolean gate, not the score.",
                n0,
                cut,
                confidence_col,
            )
    return out
