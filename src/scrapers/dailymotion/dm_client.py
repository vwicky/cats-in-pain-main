"""Dailymotion public API — search and pagination."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from tqdm import tqdm

from config import (
    DM_API_BASE,
    DM_FIELDS,
    DM_MAX_PAGES_PER_QUERY,
    DM_RESULTS_PER_PAGE,
    MAX_VIDEO_DURATION_SEC,
    MIN_VIDEO_DURATION_SEC,
    MIN_VIEWS,
    SEARCH_QUERIES,
    SLEEP_BETWEEN_REQUESTS,
)

HEADERS = {"User-Agent": "CatDatasetCollector/1.0"}


def guess_behavioral_category(search_query: str) -> str:
    """Map Dailymotion search string to data_pipeline_v2 behavioral keys (YouTube-shaped)."""
    q = (search_query or "").lower()
    if any(x in q for x in ("hiss", "fight", "attack", "agon")):
        return "Agonistic"
    if any(x in q for x in ("vet", "pain", "injur", "sick", "rescue", "medical")):
        return "Pain/Vet"
    if any(x in q for x in ("purr", "groom", "sleep", "happy", "funny", "rest")):
        return "Positive_Baseline"
    if any(x in q for x in ("chatter", "hunt", "stalk", "bird", "prey")):
        return "HuntingMind"
    return "Vocalizing"


def enrich_youtube_shaped_fields(
    row: dict[str, Any],
    query_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add aliases matching data_pipeline_v2 YouTube candidate rows.

    *query_meta* comes from GPT query generation (behavioral_category, query_language, seed_query).
    When present, it overrides heuristic behavioral_category and sets query_language to the search language.
    """
    c = dict(row)
    c["duration_seconds"] = float(c.get("duration_sec") or 0)
    c["channel_title"] = c.get("channel") or ""
    c["webpage_url"] = c.get("url") or ""
    c["view_count"] = c.get("views_total")
    c["like_count"] = c.get("likes_total")
    if query_meta and query_meta.get("behavioral_category"):
        c["behavioral_category"] = str(query_meta["behavioral_category"])
    else:
        c["behavioral_category"] = guess_behavioral_category(c.get("search_query", ""))
    if query_meta and query_meta.get("query_language"):
        c["query_language"] = str(query_meta["query_language"])
    else:
        lang = (c.get("language") or "").strip()
        c["query_language"] = lang if lang else "unknown"
    if query_meta and query_meta.get("seed_query") is not None:
        c["seed_query"] = query_meta.get("seed_query")
    c["category_id"] = ""
    c["category_name"] = "Dailymotion"
    c["source_platform"] = "dailymotion"
    return c


def _normalize_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for t in raw:
            if isinstance(t, dict):
                s = t.get("name") or t.get("label") or t.get("id")
                if s is not None:
                    out.append(str(s))
            elif t is not None:
                out.append(str(t))
        return out
    return [str(raw)]


def _normalize_channel(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return str(raw.get("name") or raw.get("id") or raw.get("username") or "")
    return str(raw)


def _search_page(
    query: str,
    page: int,
    sort: str = "relevance",
    *,
    min_duration_sec: float = MIN_VIDEO_DURATION_SEC,
    max_duration_sec: float = MAX_VIDEO_DURATION_SEC,
    _fallback_sort: bool = True,
) -> list[dict]:
    params: dict[str, Any] = {
        "search": query,
        "fields": DM_FIELDS,
        "limit": DM_RESULTS_PER_PAGE,
        "page": page,
        "sort": sort,
        "longer_than": float(min_duration_sec) / 60.0,
        "shorter_than": float(max_duration_sec) / 60.0,
    }
    log = logging.getLogger("dailymotion_pipeline")
    try:
        res = requests.get(
            f"{DM_API_BASE}/videos",
            headers=HEADERS,
            params=params,
            timeout=30,
        )
        if res.status_code == 400 and _fallback_sort and sort != "relevance":
            log.debug("Dailymotion sort=%r rejected for query=%r page=%s; retrying relevance", sort, query, page)
            return _search_page(
                query,
                page,
                "relevance",
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                _fallback_sort=False,
            )
        res.raise_for_status()
        data = res.json()
        return data.get("list") or []
    except Exception as e:
        log.warning("Dailymotion API error query=%r page=%s sort=%s: %s", query, page, sort, e)
        return []


def fetch_video_candidates(
    max_candidates: int | None = None,
    logger: logging.Logger | None = None,
    query_rows: list[dict[str, Any]] | None = None,
    search_options: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Run search strings against the Dailymotion API, paginate, apply gates, deduplicate.

    *query_rows*: from :func:`query_generation.load_or_generate_queries` — each dict has
    ``query``, ``behavioral_category``, ``query_language``, ``seed_query``.
    If omitted, uses :data:`SEARCH_QUERIES` from config (legacy static list).

    *search_options*: optional ``search`` subsection from pipeline YAML. Supported keys:
    ``rotate_sort_orders`` (bool), ``sort_orders`` (list of Dailymotion ``sort`` values),
    ``min_video_duration_seconds``, ``max_video_duration_seconds`` (override ``config.py`` gates).
    """
    opts = search_options or {}
    rotate_sort = bool(opts.get("rotate_sort_orders", False))
    sort_orders = opts.get("sort_orders") or ["relevance", "recent", "visited-week", "random"]
    if not isinstance(sort_orders, list) or not sort_orders:
        sort_orders = ["relevance"]
    min_sec = int(opts.get("min_video_duration_seconds", MIN_VIDEO_DURATION_SEC))
    max_sec = int(opts.get("max_video_duration_seconds", MAX_VIDEO_DURATION_SEC))

    log = logger.info if logger else print
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    if query_rows is None:
        query_rows = [
            {
                "query": q,
                "behavioral_category": None,
                "query_language": "English",
                "seed_query": q,
            }
            for q in SEARCH_QUERIES
        ]

    for qi, qrow in enumerate(
        tqdm(query_rows, desc="search_queries", unit="query", mininterval=0.3)
    ):
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

        query = str(qrow.get("query", "")).strip()
        if not query:
            continue

        sort_mode = sort_orders[qi % len(sort_orders)] if rotate_sort else "relevance"

        log(
            f"[SEARCH] {query!r} "
            f"[{qrow.get('behavioral_category', '?')}] [{qrow.get('query_language', '?')}]"
            f"{' sort=' + sort_mode if rotate_sort else ''}"
        )

        for page in range(1, DM_MAX_PAGES_PER_QUERY + 1):
            if max_candidates is not None and len(candidates) >= max_candidates:
                break

            results = _search_page(
                query,
                page,
                sort_mode,
                min_duration_sec=min_sec,
                max_duration_sec=max_sec,
            )
            if not results:
                break

            for item in results:
                if max_candidates is not None and len(candidates) >= max_candidates:
                    break

                dm_id = item.get("id")
                if not dm_id or dm_id in seen_ids:
                    continue

                duration = int(item.get("duration") or 0)
                if duration < min_sec or duration > max_sec:
                    continue

                views = int(item.get("views_total") or 0)
                if views < MIN_VIEWS:
                    continue

                seen_ids.add(dm_id)
                url = item.get("url") or f"https://www.dailymotion.com/video/{dm_id}"
                raw = {
                    "dm_id": dm_id,
                    "video_id": f"dailymotion_{dm_id}",
                    "title": item.get("title") or "",
                    "url": url,
                    "duration_sec": duration,
                    "views_total": views,
                    "likes_total": int(item.get("likes_total") or 0),
                    "tags": _normalize_tags(item.get("tags")),
                    "channel": _normalize_channel(item.get("channel")),
                    "language": item.get("language") or "",
                    "description": item.get("description") or "",
                    "created_time": item.get("created_time"),
                    "search_query": query,
                }
                candidates.append(enrich_youtube_shaped_fields(raw, qrow))

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        log(f"  → {len(candidates)} unique candidates so far")

    return candidates
