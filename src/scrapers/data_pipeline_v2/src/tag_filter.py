"""Rule-based tag / category / title / duration filter."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import project_root


def _category_allowed(cat_name: str | None, allowed: list[Any]) -> bool:
    allowed_set = set()
    for a in allowed:
        if a is None:
            allowed_set.add(None)
            allowed_set.add("None")
        else:
            allowed_set.add(str(a))
    if cat_name is None or cat_name == "":
        return None in allowed_set or "None" in allowed_set
    return cat_name in allowed_set


def filter_by_tags(candidates: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """
    Returns (kept, discarded).
    For each discarded record add field: discard_reason: str
    """
    tf = cfg.get("tag_filter", {})
    allowed = tf.get("allowed_categories", [])
    keywords = [k.lower() for k in tf.get("reject_title_keywords", [])]
    dmin = float(tf.get("require_min_duration_sec", 0))
    dmax = float(tf.get("require_max_duration_sec", 999999))

    kept: list[dict] = []
    discarded: list[dict] = []

    for row in candidates:
        title = (row.get("title") or "").lower()
        cat_name = row.get("category_name")
        if not cat_name and row.get("category_id"):
            from src.constants import YOUTUBE_CATEGORIES

            cat_name = YOUTUBE_CATEGORIES.get(str(row.get("category_id")), "") or None

        dur = row.get("duration_seconds")
        if dur is None:
            dur = 0.0
        try:
            dur = float(dur)
        except (TypeError, ValueError):
            dur = 0.0

        reason = None
        if not _category_allowed(cat_name, allowed):
            reason = "Category not allowed"
        elif any(kw in title for kw in keywords):
            reason = "Title keyword match"
        elif dur < dmin or dur > dmax:
            reason = "Duration out of range"

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
        reasons[d.get("discard_reason", "?")] += 1

    box = f"""
┌────────────────────────────────────────────────────────┐
│  TAG FILTER SUMMARY                                    │
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
    stage_dir = run_dir / "stage_2_tag_filter"
    stage_dir.mkdir(parents=True, exist_ok=True)
    p = stage_dir / "tag_filter_breakdown.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    root = project_root()
    alt = root / "src" / "scrapers" / "data_pipeline_v2" / "reports" / "tag_filter_breakdown.png"
    alt.parent.mkdir(parents=True, exist_ok=True)
    fig2, ax2 = plt.subplots(figsize=(8, max(3, len(reasons) * 0.35)))
    if reasons:
        ax2.barh(list(reasons.keys())[::-1], list(reasons.values())[::-1], color="coral")
    fig2.savefig(alt, dpi=150)
    plt.close(fig2)

    stats = {"input": n, "kept": nk, "discarded": nd, "reasons": dict(reasons)}
    return kept, discarded, stats
