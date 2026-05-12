"""
Build a thesis/analysis DataFrame from merged manifest rows (dicts).
Shared extractors support nested gpt_description or flat gpt_* columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _first_defined(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "yes", "1", "y"}:
            return True
        if s in {"false", "no", "0", "n", ""}:
            return False
    return None


def _nested_gpt_bundle(row: dict[str, Any]) -> dict[str, Any]:
    gd = row.get("gpt_description")
    if isinstance(gd, dict):
        return gd
    return row


def extract_suitable_for_training(row: dict[str, Any]) -> bool | None:
    gb = _nested_gpt_bundle(row)
    flags = gb.get("dataset_flags") if isinstance(gb.get("dataset_flags"), dict) else {}
    return as_bool(
        _first_defined(
            flags.get("suitable_for_training"),
            row.get("suitable_for_training"),
            row.get("gpt_suitable_for_training"),
        )
    )


def extract_manifest_platform(row: dict[str, Any]) -> str | None:
    """Canonical host for platform comparison plots: ``youtube`` or ``tiktok``.

    Reads top-level ``platform``, then ``source`` (string or URL). Unknown or empty
    values return None so callers can exclude or footnote those rows.
    """
    raw = _first_defined(row.get("platform"), row.get("source"))
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    if s in {"youtube", "yt"}:
        return "youtube"
    if s in {"tiktok", "tt", "tik tok", "tiktok.com"}:
        return "tiktok"
    if "tiktok" in s or "tiktok.com" in s:
        return "tiktok"
    if "youtube" in s or "youtu.be" in s:
        return "youtube"
    return None


def aggregate_platform_suitability(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, tuple[int, int, int]], int]:
    """Count suitable / not / missing per canonical platform (youtube, tiktok).

    Returns ``(per_platform, n_rows_skipped_unknown_platform)`` where each platform
    maps to ``(n_true, n_false, n_missing_suitability)``.
    """
    acc: dict[str, list[int]] = {
        "youtube": [0, 0, 0],
        "tiktok": [0, 0, 0],
    }
    n_skip = 0
    for row in rows:
        pl = extract_manifest_platform(row)
        if pl not in acc:
            n_skip += 1
            continue
        su = extract_suitable_for_training(row)
        if su is True:
            acc[pl][0] += 1
        elif su is False:
            acc[pl][1] += 1
        else:
            acc[pl][2] += 1
    out: dict[str, tuple[int, int, int]] = {
        k: (v[0], v[1], v[2]) for k, v in acc.items()
    }
    return out, n_skip


def extract_exclusion_reason(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    flags = gb.get("dataset_flags") if isinstance(gb.get("dataset_flags"), dict) else {}
    v = _first_defined(flags.get("exclusion_reason"), row.get("gpt_exclusion_reason"))
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return str(v)


def extract_breed_guess(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    cats = gb.get("cats")
    breed: Any = None
    if isinstance(cats, dict):
        pc = cats.get("primary_cat")
        if isinstance(pc, dict):
            breed = pc.get("breed_guess")
        elif breed is None:
            breed = cats.get("breed_guess")
    elif isinstance(cats, list):
        for c in cats:
            if not isinstance(c, dict):
                continue
            if c.get("primary") is True or c.get("is_primary") is True:
                breed = c.get("breed_guess")
                break
        if breed is None and cats:
            first = cats[0]
            if isinstance(first, dict):
                breed = first.get("breed_guess")
    breed = _first_defined(breed, gb.get("gpt_breed_guess"), row.get("gpt_breed_guess"))
    if breed is None:
        return None
    s = str(breed).strip()
    return s or None


def extract_setting(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    env = gb.get("environment") if isinstance(gb.get("environment"), dict) else {}
    v = _first_defined(env.get("setting"), gb.get("gpt_setting"), row.get("gpt_setting"))
    if v is None:
        return None
    s = str(v).strip().lower()
    return s or None


def extract_location_type(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    env = gb.get("environment") if isinstance(gb.get("environment"), dict) else {}
    v = _first_defined(
        env.get("location_type"),
        gb.get("gpt_location_type"),
        row.get("gpt_location_type"),
    )
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def extract_primary_behavior(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    beh = gb.get("behavior") if isinstance(gb.get("behavior"), dict) else {}
    v = _first_defined(
        beh.get("primary_behavior"),
        gb.get("gpt_primary_behavior"),
        row.get("gpt_primary_behavior"),
    )
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def extract_lighting(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    vq = gb.get("video_quality") if isinstance(gb.get("video_quality"), dict) else {}
    v = _first_defined(vq.get("lighting"), gb.get("gpt_lighting"), row.get("gpt_lighting"))
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def extract_blur(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    vq = gb.get("video_quality") if isinstance(gb.get("video_quality"), dict) else {}
    v = _first_defined(vq.get("blur"), gb.get("gpt_blur"), row.get("gpt_blur"))
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def extract_gpt_resolution(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    vq = gb.get("video_quality") if isinstance(gb.get("video_quality"), dict) else {}
    v = _first_defined(vq.get("resolution"), gb.get("gpt_resolution"), row.get("gpt_resolution"))
    if v is None:
        return None
    s = str(v).strip().lower()
    return s or None


def extract_gpt_occlusion(row: dict[str, Any]) -> str | None:
    gb = _nested_gpt_bundle(row)
    vq = gb.get("video_quality") if isinstance(gb.get("video_quality"), dict) else {}
    v = _first_defined(vq.get("occlusion"), gb.get("gpt_occlusion"), row.get("gpt_occlusion"))
    if v is None:
        return None
    s = str(v).strip().lower()
    return s or None


def extract_face_visible(row: dict[str, Any]) -> bool | None:
    gb = _nested_gpt_bundle(row)
    beh = gb.get("behavior") if isinstance(gb.get("behavior"), dict) else {}
    return as_bool(
        _first_defined(
            beh.get("face_clearly_visible"),
            gb.get("gpt_face_clearly_visible"),
            row.get("gpt_face_clearly_visible"),
        )
    )


LABEL5_ORDER = ["Agonistic", "HuntingMind", "Paining", "Positive_Baseline", "Vocalizing"]
BINARY_ORDER = ["No_Pain", "Pain"]
# Canonical audio classifier 10-class order (see 04_audio_classification / 05_final_dataset)
LABEL10_ORDER = [
    "Angry",
    "Defence",
    "Fighting",
    "Happy",
    "HuntingMind",
    "Mating",
    "MotherCall",
    "Paining",
    "Resting",
    "Warning",
]
LIGHTING_ORDER = ["bright", "dark", "dim", "normal", "overexposed"]
BLUR_ORDER = ["none", "mild", "severe"]
RESOLUTION_ORDER = ["high", "medium", "low"]
OCCLUSION_ORDER = ["none", "partial", "severe"]


def setting_label_for_stacked_bar(row: dict[str, Any]) -> str | None:
    s_raw = extract_setting(row)
    if s_raw is None:
        return None
    if s_raw in {"indoor", "indoors"}:
        return "Indoor"
    if s_raw in {"outdoor", "outdoors"}:
        return "Outdoor"
    return str(s_raw).strip().title()


def setting_io_hue(row: dict[str, Any]) -> str | None:
    s_raw = extract_setting(row)
    if s_raw is None:
        return None
    if s_raw in {"indoor", "indoors"}:
        return "Indoor"
    if s_raw in {"outdoor", "outdoors"}:
        return "Outdoor"
    return None


def setting_tri_emotion(row: dict[str, Any]) -> str | None:
    s_raw = extract_setting(row)
    if s_raw is None:
        return "Mixed/Unclear"
    if s_raw in {"indoor", "indoors"}:
        return "Indoor"
    if s_raw in {"outdoor", "outdoors"}:
        return "Outdoor"
    return "Mixed/Unclear"


def collapse_rare_categories(
    series: pd.Series,
    min_count: int,
    other_label: str = "other",
) -> pd.Series:
    s = series.astype("object")
    vc = s.dropna().astype(str).value_counts()
    rare = {k for k, v in vc.items() if v < min_count}

    def map_one(x: Any) -> Any:
        if pd.api.types.is_scalar(x) and pd.isna(x):
            return pd.NA
        xs = str(x)
        return other_label if xs in rare else xs

    return s.map(map_one)


def top_k_plus_other(
    series: pd.Series,
    k: int,
    other_label: str = "other",
) -> pd.Series:
    s = series.astype("object")
    vc = s.dropna().astype(str).value_counts()
    keep = set(vc.head(k).index)

    def map_one(x: Any) -> Any:
        if pd.api.types.is_scalar(x) and pd.isna(x):
            return pd.NA
        xs = str(x)
        return xs if xs in keep else other_label

    return s.map(map_one)


def collapse_behaviors_for_emotion_heatmap(
    series: pd.Series,
    behavior_min_count: int,
    top_k: int = 8,
    other_label: str = "other",
) -> pd.Series:
    s1 = collapse_rare_categories(series, behavior_min_count, other_label=other_label)
    return top_k_plus_other(s1, k=top_k, other_label=other_label)


def location_collapsed(series: pd.Series, location_other_min: int) -> pd.Series:
    return collapse_rare_categories(series, location_other_min, other_label="other")


def build_thesis_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "final_label_5": row.get("final_label_5"),
                "final_label_binary": row.get("final_label_binary"),
                "audio_label_10": row.get("audio_label_10"),
                "audio_label_binary": row.get("audio_label_binary"),
                "gpt_setting_raw": extract_setting(row),
                "setting_stack": setting_label_for_stacked_bar(row),
                "setting_io": setting_io_hue(row),
                "setting_tri": setting_tri_emotion(row),
                "gpt_location_type": extract_location_type(row),
                "gpt_primary_behavior": extract_primary_behavior(row),
                "gpt_lighting": extract_lighting(row),
                "gpt_blur": extract_blur(row),
                "gpt_resolution": extract_gpt_resolution(row),
                "gpt_occlusion": extract_gpt_occlusion(row),
                "gpt_face_clearly_visible": extract_face_visible(row),
                "gpt_exclusion_reason": extract_exclusion_reason(row),
                "gpt_breed_guess_flat": row.get("gpt_breed_guess"),
                "breed_extracted": extract_breed_guess(row),
            }
        )
    return pd.DataFrame.from_records(records)


def row_norm_matrix(ct: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    row_sums = ct.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = ct.div(row_sums.replace(0, np.nan), axis=0)
    prop_plot = prop.fillna(0.0)
    return prop_plot, row_sums


def heatmap_annotations_row_norm(ct: pd.DataFrame, row_sums: pd.Series) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = ct.div(row_sums.replace(0, np.nan), axis=0)

    ann = np.empty(ct.shape, dtype=object)
    for i in range(ct.shape[0]):
        if row_sums.iloc[i] == 0:
            ann[i, :] = "—"
            continue
        for j in range(ct.shape[1]):
            v = prop.iloc[i, j]
            if pd.isna(v):
                ann[i, j] = "—"
            elif ct.iloc[i, j] == 0:
                ann[i, j] = "0.00"
            else:
                ann[i, j] = f"{float(v):.2f}"
    return ann
