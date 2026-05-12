"""Rule-based metadata filter for TikTok candidates (defensive against sparse fields)."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


def _safe_str(x: Any) -> str:
    """Never raise; empty string for None / non-string."""
    if x is None:
        return ""
    try:
        return str(x).strip()
    except Exception:
        return ""


def _safe_lower(x: Any) -> str:
    return _safe_str(x).lower()


def _iter_text_fields(row: dict[str, Any]) -> list[str]:
    """Collect searchable text from title, description, hashtags, tags."""
    parts: list[str] = []
    parts.append(_safe_lower(row.get("title")))
    parts.append(_safe_lower(row.get("description")))
    tags = row.get("tags")
    if isinstance(tags, list):
        for t in tags:
            parts.append(_safe_lower(t))
    elif tags is not None:
        parts.append(_safe_lower(tags))
    ht = row.get("hashtags")
    if isinstance(ht, list):
        for h in ht:
            parts.append(_safe_lower(h))
    elif ht is not None:
        parts.append(_safe_lower(ht))
    # Sound / track metadata when present
    for key in ("track", "artist", "album", "music_info"):
        v = row.get(key)
        if v is not None:
            parts.append(_safe_lower(v))
    return [p for p in parts if p]


def _combined_haystack(row: dict[str, Any]) -> str:
    return " ".join(_iter_text_fields(row))


def filter_by_tags(candidates: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Returns (kept, discarded). Each discarded row has discard_reason."""
    tf = cfg.get("tag_filter", {})
    title_kw = [_safe_lower(k) for k in tf.get("reject_title_keywords", []) if _safe_str(k)]
    desc_kw = [_safe_lower(k) for k in tf.get("reject_description_keywords", []) if _safe_str(k)]
    track_kw = [_safe_lower(k) for k in tf.get("reject_if_track_title_keywords", []) if _safe_str(k)]
    dmin = float(tf.get("require_min_duration_sec", 0))
    dmax = float(tf.get("require_max_duration_sec", 999999))
    discard_sound_only = bool(tf.get("discard_trending_sound_only_when_no_text", False))

    kept: list[dict] = []
    discarded: list[dict] = []

    for row in candidates:
        title = _safe_lower(row.get("title"))
        desc = _safe_lower(row.get("description"))
        hay = _combined_haystack(row)
        dur_raw = row.get("duration_seconds")
        try:
            dur = float(dur_raw) if dur_raw is not None else 0.0
        except (TypeError, ValueError):
            dur = 0.0

        reason = None

        if dur > 0 and (dur < dmin or dur > dmax):
            reason = "Duration out of range"

        if reason is None and title_kw:
            if any(kw and kw in title for kw in title_kw):
                reason = "Title keyword match"

        if reason is None and desc_kw:
            if any(kw and kw in desc for kw in desc_kw):
                reason = "Description keyword match"

        if reason is None and desc_kw and hay:
            if any(kw and kw in hay for kw in desc_kw):
                reason = "Metadata keyword match"

        if reason is None and track_kw:
            track_blob = _safe_lower(row.get("track")) + " " + _safe_lower(row.get("artist"))
            if track_blob.strip():
                if any(kw and kw in track_blob for kw in track_kw):
                    reason = "Track/sound keyword match"
            # If no track metadata, do not discard on track keywords

        if reason is None and discard_sound_only:
            has_text = bool(_safe_str(row.get("title"))) or bool(_safe_str(row.get("description")))
            has_tags = bool(row.get("hashtags")) or bool(row.get("tags"))
            if not has_text and not has_tags and _safe_str(row.get("track")):
                reason = "Trending sound only (no text)"

        if reason:
            d = dict(row)
            d["discard_reason"] = reason
            discarded.append(d)
        else:
            kept.append(dict(row))

    return kept, discarded


def run_tag_filter(
    candidates: list[dict],
    cfg: dict,
    logger: logging.Logger,
    run_dir: Any,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    kept, discarded = filter_by_tags(candidates, cfg)
    n = len(candidates)
    nk, nd = len(kept), len(discarded)
    pct = 100.0 * nk / n if n else 0.0

    reasons = Counter()
    for d in discarded:
        reasons[_safe_str(d.get("discard_reason")) or "?"] += 1

    box = f"""
┌────────────────────────────────────────────────────────┐
│  TAG FILTER SUMMARY (TikTok)                           │
│  Input:     {n:>6} candidates (metadata only)           │
│  Kept:      {nk:>6} ({pct:.1f}%)                              │
│  Discarded: {nd:>6} ({100.0-pct:.1f}%)                              │
│                                                        │
│  Discard reasons:                                      │
"""
    for r, c in reasons.most_common():
        box += f"│    {r}:    {c:>6}                        \n"
    box += "└────────────────────────────────────────────────────────┘"
    print(box)
    logger.info(box)

    fig, ax = plt.subplots(figsize=(8, max(3, len(reasons) * 0.35)))
    if reasons:
        ax.barh(list(reasons.keys())[::-1], list(reasons.values())[::-1], color="coral")
    ax.set_xlabel("Count")
    fig.suptitle("Tag filter: discard reasons")
    fig.tight_layout()
    p = run_dir / "stage_2_tag_filter" / "tag_filter_breakdown.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    root = Path(__file__).resolve().parent.parent.parent
    alt = root / "src" / "scrapers" / "tiktok" / "reports" / "tag_filter_breakdown.png"
    alt.parent.mkdir(parents=True, exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=(8, max(3, len(reasons) * 0.35)))
    if reasons:
        ax2.barh(list(reasons.keys())[::-1], list(reasons.values())[::-1], color="coral")
    fig2.savefig(alt, dpi=150)
    plt.close(fig2)

    stats = {"input": n, "kept": nk, "discarded": nd, "reasons": dict(reasons)}
    return kept, discarded, stats
