"""Fast metadata-only filter on Dailymotion API fields (no video decode)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import CAT_SIGNAL_KEYWORDS, REJECT_SUBSTRING_EXTRA, REJECT_TITLE_KEYWORDS

from utils import save_jsonl

# Whole-word / token match for cat-related content (avoids matching unrelated "cat" substrings where possible)
_CAT_SIGNAL_RE = re.compile(
    r"\b(cats?|kittens?|kitties|feline|kitty|meow|purr|gatos?|gatto|chat)\b",
    re.IGNORECASE,
)


def _combined_text(candidate: dict) -> str:
    """Title/description/tags/channel plus **search_query / seed_query** so multilingual
    API hits that lack “cat/gato” in the video title still pass when our query was cat-related
    (GPT filter remains the quality gate)."""
    ch = candidate.get("channel_title") or candidate.get("channel") or ""
    parts = [
        str(candidate.get("title") or ""),
        str(candidate.get("description") or ""),
        " ".join(candidate.get("tags") or []),
        str(ch),
        str(candidate.get("search_query") or ""),
        str(candidate.get("seed_query") or ""),
    ]
    return " \n ".join(parts).lower()


def passes_tag_filter(candidate: dict) -> tuple[bool, dict[str, Any]]:
    """
    Returns (kept, info_dict).
    info_dict includes passed, discard_reason, and optional matched hints.
    """
    blob = _combined_text(candidate)

    for kw in REJECT_TITLE_KEYWORDS:
        if kw.lower() in blob:
            return False, {
                "passed": False,
                "discard_reason": f"reject_keyword:{kw!r}",
            }

    for extra in REJECT_SUBSTRING_EXTRA:
        if extra.lower() in blob:
            return False, {
                "passed": False,
                "discard_reason": f"reject_substring:{extra!r}",
            }

    if _CAT_SIGNAL_RE.search(blob):
        return True, {"passed": True, "cat_signal": "regex_token"}

    # Fallback: substring list from config (e.g. explicit "cat" in languages)
    for kw in CAT_SIGNAL_KEYWORDS:
        if kw.lower() in blob:
            return True, {"passed": True, "cat_signal": f"keyword:{kw!r}"}

    return False, {
        "passed": False,
        "discard_reason": "no_cat_related_signal_in_metadata",
    }


def run_tag_filter(
    candidates: list[dict],
    logger: logging.Logger,
    run_dir: Path,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """
    Metadata-only tag filter (pipeline v2 style).
    Writes ``stage_2_tag_filter/kept.jsonl``, ``discarded.jsonl``, ``tag_filter_summary.txt``.
    """
    kept: list[dict] = []
    discarded: list[dict] = []

    logger.info("Stage 2 tag filter: scanning %d candidates (metadata + search query text)", len(candidates))

    for c in tqdm(candidates, desc="tag_filter", unit="vid", mininterval=0.5):
        ok, info = passes_tag_filter(c)
        row = dict(c)
        row["tag_filter"] = info
        if ok:
            kept.append(row)
        else:
            discarded.append(row)

    n = len(candidates)
    nk, nd = len(kept), len(discarded)
    pct = 100.0 * nk / n if n else 0.0

    save_jsonl(kept, run_dir / "stage_2_tag_filter" / "kept.jsonl", mode="w")
    save_jsonl(discarded, run_dir / "stage_2_tag_filter" / "discarded.jsonl", mode="w")

    stats: dict[str, Any] = {"input": n, "kept": nk, "discarded": nd}
    with open(run_dir / "stage_2_tag_filter" / "tag_filter_summary.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, indent=2))

    box = f"""
┌────────────────────────────────────────────────────────┐
│  TAG FILTER SUMMARY (Dailymotion metadata)             │
│  Input:     {n:>6} candidates                           │
│  Kept:      {nk:>6} ({pct:.1f}%)                              │
│  Discarded: {nd:>6} ({100.0 - pct:.1f}%)                              │
└────────────────────────────────────────────────────────┘"""
    logger.info(box)

    return kept, discarded, stats
