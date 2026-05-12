"""Query generation + YouTube search (API or yt-dlp fallback)."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from yt_dlp import YoutubeDL

from src.constants import YOUTUBE_CATEGORIES
from src.downloader import collect_skip_video_ids
from src.utils import (
    append_jsonl,
    hash_config_section,
    load_jsonl,
    project_root,
    resolve_path,
    retry_with_backoff,
    save_jsonl,
)

QUERY_CACHE_META = "v2_generated_queries.meta.json"
QUERY_CACHE_JSONL = "v2_generated_queries.jsonl"

# Keys in search.* that must not affect GPT seed / query cache hashes
SEARCH_HASH_EXCLUDE = frozenset({"youtube_search_workers"})

# Hardcoded English seeds when GPT seed generation fails (cycled to fill per-category limits)
HARDCODED_SEEDS: dict[str, list[str]] = {
    "Pain/Vet": [
        "cat at veterinarian crying",
        "cat injured limping home",
        "cat after surgery recovery",
        "sick cat at vet examination",
        "cat in pain meowing loudly",
        "injured cat rescue vet",
    ],
    "Agonistic": [
        "two cats fighting hissing",
        "cat attacking another cat",
        "feral cat defensive hiss",
        "cat territorial fight",
        "angry cat swatting",
        "cat aggressive toward dog",
    ],
    "Vocalizing": [
        "cat yowling loudly night",
        "cat meowing constantly",
        "cat mating call sounds",
        "Siamese cat loud meow",
        "kitten crying for mother",
        "cat warning growl vocal",
    ],
    "Positive_Baseline": [
        "cat purring relaxed sleeping",
        "happy cat playing owner",
        "cat grooming calmly sunny",
        "peaceful cat resting couch",
        "content cat slow blink",
        "relaxed cat kneading blanket",
    ],
    "HuntingMind": [
        "cat stalking bird window",
        "cat chattering at birds",
        "indoor cat focused prey",
        "cat hunting toy pounce",
        "barn cat hunting mice",
        "cat tail twitch stalking",
    ],
}

# Order used in prompts, dedupe limits, and config overrides.
BEHAVIORAL_CATEGORY_ORDER: tuple[str, ...] = (
    "Pain/Vet",
    "Agonistic",
    "Vocalizing",
    "Positive_Baseline",
    "HuntingMind",
)

BEHAVIORAL_PROMPT = """Behavioral categories (use these exact keys in output).
**Primary research focus: Pain/Vet** — this project studies pain-related cat behavior; give Pain/Vet
queries extra specificity (vet exam, post-op, limping, dental, injury recovery, vocal distress at
clinic, home nursing) while still suitable for real YouTube footage of live cats (not gore).

- Pain/Vet: cats at vets, post-surgery, injured, in discomfort, recovery, nursing
- Agonistic: cats hissing, fighting, defensive, aggressive
- Vocalizing: cats yowling, calling, warning, mating sounds
- Positive_Baseline: cats relaxed, purring, content, resting
- HuntingMind: cats stalking, chattering, predatory focus

Uniqueness goal: queries should surface different videos than generic two-word cat searches
(long-tail phrases, specific contexts, breeds, regions, rescue vs home, procedures, sounds).
"""

PAIN_VET_ONLY_PROMPT = """You are generating English YouTube search queries for **one** behavioral key only:
**Pain/Vet** (exact string for JSON output).

Focus: cats at veterinarians, post-surgery or post-injury care, visible discomfort, limping, dental or
emergency visits, recovery at home, vocal distress at the clinic, nursing a sick cat — real footage
suitable for research (not gore, not stock-only slideshows).

Use long-tail phrases (typically 4–10 words): specific breeds, regions, clinic vs home, kitten vs senior,
rescue vs indoor, named procedures where plausible. Each query must be clearly distinct (no near-duplicates).
"""


def active_behavioral_categories(search_cfg: dict[str, Any]) -> tuple[str, ...]:
    """
    Subset of categories to search. Default: all five.

    Config: ``search.active_behavioral_categories`` (or ``behavioral_categories``) — list of keys
    such as ``Pain/Vet``. Order follows BEHAVIORAL_CATEGORY_ORDER.
    """
    raw = search_cfg.get("active_behavioral_categories")
    if raw is None:
        raw = search_cfg.get("behavioral_categories")
    if not raw:
        return BEHAVIORAL_CATEGORY_ORDER
    if isinstance(raw, str):
        raw = [raw]
    allowed = set(BEHAVIORAL_CATEGORY_ORDER)
    seen: list[str] = []
    for x in raw:
        s = str(x).strip()
        if s in allowed and s not in seen:
            seen.append(s)
    if not seen:
        return BEHAVIORAL_CATEGORY_ORDER
    return tuple(c for c in BEHAVIORAL_CATEGORY_ORDER if c in seen)


def resolve_seed_query_limits(search_cfg: dict[str, Any]) -> dict[str, int]:
    """
    Per-category English seed counts for **active** categories only. Backward-compatible: if
    ``active_behavioral_categories`` is omitted, all five categories are used.

    If only ``seed_queries_per_category`` is set, it applies to each active category (Pain/Vet still
    overridden by ``seed_queries_per_category_pain_vet`` when that key is present). If the only active
    category is Pain/Vet, ``seed_queries_per_category`` applies to it when ``pain_vet`` is not set.
    """
    active = active_behavioral_categories(search_cfg)
    base = int(search_cfg.get("seed_queries_per_category", 6))
    by_cat = search_cfg.get("seed_queries_per_category_by_category")
    out: dict[str, int] = {}
    if isinstance(by_cat, dict) and by_cat:
        for k in active:
            raw = by_cat.get(k, base)
            try:
                out[k] = int(raw)
            except (TypeError, ValueError):
                out[k] = base
        return out
    n_pv = search_cfg.get("seed_queries_per_category_pain_vet")
    try:
        n_pv_i = int(n_pv) if n_pv is not None else None
    except (TypeError, ValueError):
        n_pv_i = None
    for k in active:
        if k == "Pain/Vet" and n_pv_i is not None:
            out[k] = n_pv_i
        else:
            out[k] = base
    return out


def total_seed_queries_from_limits(limits: dict[str, int]) -> int:
    return sum(limits.values())


def _maybe_random_sample_queries(
    queries: list[dict[str, Any]],
    search_cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """
    If ``random_sample_queries`` is a positive integer and smaller than len(queries),
    return ``random.sample`` of that size (optional ``random_sample_seed`` for reproducibility).
    Returns (queries_to_run, queries_before_sample).
    """
    n0 = len(queries)
    raw = search_cfg.get("random_sample_queries", 0)
    try:
        k = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        k = 0
    if k <= 0 or n0 <= k:
        if k > 0 and n0 <= k:
            logger.info(
                "search.random_sample_queries=%s >= total expanded queries (%s) — using all",
                k,
                n0,
            )
        return queries, n0

    seed = search_cfg.get("random_sample_seed")
    rng = random.Random(int(seed)) if seed is not None else random.Random()
    sampled = rng.sample(queries, k)
    logger.info(
        "Random query sample: running %s of %s expanded queries (random_sample_seed=%r)",
        k,
        n0,
        seed,
    )
    sd = run_dir / "stage_1_search"
    sd.mkdir(parents=True, exist_ok=True)
    manifest = {
        "queries_before_sample": n0,
        "queries_after_sample": k,
        "random_sample_queries": k,
        "random_sample_seed": seed,
    }
    (sd / "search_query_sample.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_jsonl(sampled, sd / "sampled_queries.jsonl", mode="w")
    return sampled, n0


def _dedupe_seeds_per_category(
    raw: list[dict[str, Any]],
    limits: dict[str, int],
    default_limit: int,
) -> list[dict[str, Any]]:
    """Case-insensitive dedupe within each behavioral_category; at most per-category cap."""
    counts: dict[str, int] = defaultdict(int)
    seen: dict[str, set[str]] = defaultdict(set)
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "query" not in item:
            continue
        cat = str(item.get("behavioral_category", "Positive_Baseline"))
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        cap = int(limits.get(cat, default_limit))
        key = q.lower()
        if key in seen[cat] or counts[cat] >= cap:
            continue
        seen[cat].add(key)
        counts[cat] += 1
        out.append({"behavioral_category": cat, "query": q})
    return out


def _dedupe_seed_queries_globally(
    rows: list[dict[str, Any]], logger: logging.Logger
) -> list[dict[str, Any]]:
    """Drop duplicate English seed strings across categories (same query → same YouTube SERP)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        q = " ".join(str(row.get("query", "")).strip().lower().split())
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(row)
    dropped = len(rows) - len(out)
    if dropped:
        logger.info(
            "Removed %d duplicate English seed(s) (identical text in multiple categories)",
            dropped,
        )
    return out


def _dedupe_expanded_queries(
    rows: list[dict[str, Any]], logger: logging.Logger
) -> list[dict[str, Any]]:
    """Drop duplicate (language, query) rows so the same search is not executed twice."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        lang = str(row.get("query_language", "")).strip().lower()
        q = " ".join(str(row.get("query", "")).strip().lower().split())
        if not q:
            continue
        key = (lang, q)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    dropped = len(rows) - len(out)
    if dropped:
        logger.info(
            "Removed %d duplicate expanded query row(s) (same language + query text)",
            dropped,
        )
    return out


def _parse_iso8601_duration(iso: str | None) -> float:
    if not iso:
        return 0.0
    m = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        iso,
    )
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return float(h or 0) * 3600 + float(mn or 0) * 60 + float(s or 0)


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _generate_seed_queries_openai(
    cfg: dict[str, Any], logger: logging.Logger
) -> list[dict[str, Any]]:
    """Returns list of {behavioral_category, query}."""
    search_cfg = cfg["search"]
    limits = resolve_seed_query_limits(search_cfg)
    ordered = tuple(c for c in BEHAVIORAL_CATEGORY_ORDER if c in limits)
    if not limits:
        raise RuntimeError("No behavioral categories in search limits (check active_behavioral_categories).")
    default_n = int(search_cfg.get("seed_queries_per_category", 6))
    model = search_cfg.get("query_generation_model", "gpt-4o-mini")
    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("OpenAI key required for GPT seed generation")

    client = _openai_client(api_key)
    seed_temp = float(search_cfg.get("seed_generation_temperature", 0.58))
    keys_str = ", ".join(f'"{c}"' for c in ordered)
    pain_only = ordered == ("Pain/Vet",)
    system = (
        "You are a research assistant helping build a cat behavior dataset. "
        "Generate diverse YouTube search queries that would find videos of "
        "cats exhibiting the requested behavioral categories. "
        "Prefer **long-tail** queries (typically 4–10 words): specific situations, breeds, "
        "regions, procedures, rescue vs indoor, ASMR/purr/focus sounds, POV, day-in-the-life, "
        "and niche wording—NOT the same few viral-generic patterns every time. "
        "CRITICAL: Each query must be clearly distinct—avoid near-duplicates within a category "
        "(no trivial rewordings, plural/singular-only changes, or the same idea repeated). "
        "Across the full JSON output, **no two queries may normalize to the same lowercase string** "
        "(do not repeat the same search in different categories). "
        "Vary angles: vet clinic vs home care, foster, barn, street, breed-specific, seasonal, "
        "multi-cat vs single-cat, kitten vs senior. "
        "Respect the **exact per-category counts** in the user message. "
        'Return ONLY a JSON object with key \"queries\" whose value is an array of objects with '
        f'"behavioral_category" (string, one of: {keys_str}) and "query" (string).'
    )
    lines = [f"- {cat}: {limits[cat]} distinct queries" for cat in ordered]
    total = total_seed_queries_from_limits(limits)
    prompt_body = PAIN_VET_ONLY_PROMPT if pain_only else BEHAVIORAL_PROMPT
    user = (
        f"{prompt_body}\n\nGenerate exactly this many distinct queries per category:\n"
        + "\n".join(lines)
        + f"\n\n({total} total before deduplication). "
        "Maximize how **different** the resulting YouTube result pages would be—avoid short "
        'generic queries like "cute cat" or "funny cat" that all return the same top videos. '
        "Return JSON only."
    )

    def call():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=seed_temp,
            response_format={"type": "json_object"},
        )
        return resp

    resp = retry_with_backoff(call, max_retries=3)
    text = resp.choices[0].message.content or "{}"
    data = json.loads(text)
    if isinstance(data, dict) and "queries" in data:
        arr = data["queries"]
    elif isinstance(data, list):
        arr = data
    elif isinstance(data, dict):
        arr = data.get("items", data.get("results", []))
    else:
        arr = []
    raw_rows: list[dict[str, Any]] = []
    for item in arr:
        if isinstance(item, dict) and "query" in item:
            raw_rows.append(
                {
                    "behavioral_category": item.get("behavioral_category", "Positive_Baseline"),
                    "query": item["query"],
                }
            )
    out = _dedupe_seeds_per_category(raw_rows, limits, default_n)
    out = _dedupe_seed_queries_globally(out, logger)
    min_ok = min(5, max(1, total_seed_queries_from_limits(limits)))
    if len(out) < min_ok:
        logger.warning("GPT returned few seeds; padding with hardcoded seeds")
        out = []
        for cat in ordered:
            seeds = HARDCODED_SEEDS.get(cat, [])
            if not seeds:
                continue
            lim = limits.get(cat, default_n)
            for i in range(lim):
                out.append(
                    {
                        "behavioral_category": cat,
                        "query": seeds[i % len(seeds)],
                    }
                )
        out = _dedupe_seeds_per_category(out, limits, default_n)
        out = _dedupe_seed_queries_globally(out, logger)
    return out


def _expand_seed_multilingual(
    seed: dict[str, Any],
    target_langs: list[str],
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Expand one seed into len(target_langs) language variants."""
    search_cfg = cfg["search"]
    model = search_cfg.get("query_generation_model", "gpt-4o-mini")
    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    base_q = seed["query"]
    cat = seed["behavioral_category"]
    if not api_key:
        return [
            {
                "behavioral_category": cat,
                "query_language": "English",
                "query": base_q,
                "seed_query": base_q,
            }
        ]

    client = _openai_client(api_key)
    trans_temp = float(search_cfg.get("translation_temperature", 0.35))
    system = (
        "Translate the following YouTube search query into the specified languages. "
        "Make translations natural and idiomatic — how a native speaker would actually "
        "search YouTube for this content. "
        "Preserve the **specificity** of the English query: do not collapse everything to "
        "a generic short phrase that would return the same viral hits in every language. "
        'Return a JSON object with key "translations" '
        'containing an array of objects with fields: language, query.'
    )
    user = json.dumps({"query": base_q, "languages": target_langs})

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=trans_temp,
            response_format={"type": "json_object"},
        )

    resp = retry_with_backoff(call, max_retries=3)
    text = resp.choices[0].message.content or "{}"
    data = json.loads(text)
    translations = data.get("translations", data) if isinstance(data, dict) else []
    if isinstance(translations, dict):
        translations = [translations]
    out: list[dict[str, Any]] = []
    if not translations:
        return [
            {
                "behavioral_category": cat,
                "query_language": "English",
                "query": base_q,
                "seed_query": base_q,
            }
        ]
    for t in translations:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "behavioral_category": cat,
                "query_language": t.get("language", "English"),
                "query": t.get("query", base_q),
                "seed_query": base_q,
            }
        )
    return out if out else [
        {
            "behavioral_category": cat,
            "query_language": "English",
            "query": base_q,
            "seed_query": base_q,
        }
    ]


def load_or_generate_queries(
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load cached queries if search config hash matches; else generate.
    Returns list of dicts with behavioral_category, query_language, query, seed_query.
    """
    root = project_root()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = logs_dir / QUERY_CACHE_META
    cache_path = logs_dir / QUERY_CACHE_JSONL

    current_hash = hash_config_section(cfg, "search", exclude_subkeys=SEARCH_HASH_EXCLUDE)

    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("search_config_hash") == current_hash and cache_path.is_file():
                logger.info("Using cached generated queries (search config hash matches).")
                rows = load_jsonl(cache_path)
                if rows:
                    return rows
        except (json.JSONDecodeError, OSError):
            pass
    else:
        if cache_path.is_file():
            logger.warning(
                "Query cache exists but meta file missing or hash mismatch — regenerating queries."
            )

    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                old = json.load(f)
            if old.get("search_config_hash") != current_hash:
                logger.warning(
                    "Search config changed (hash mismatch). Regenerating queries and overwriting cache."
                )
        except OSError:
            pass

    search_cfg = cfg["search"]
    langs_n = int(search_cfg.get("languages_per_query", 5))
    target_langs = (search_cfg.get("target_languages") or ["English"])[:langs_n]
    active_cats = active_behavioral_categories(search_cfg)

    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    all_queries: list[dict[str, Any]] = []

    if api_key:
        try:
            seeds = _generate_seed_queries_openai(cfg, logger)
        except Exception as e:
            logger.warning("GPT seed generation failed: %s — using hardcoded English seeds", e)
            seeds = []
            for cat in active_cats:
                for q in HARDCODED_SEEDS.get(cat, []):
                    seeds.append({"behavioral_category": cat, "query": q})
    else:
        logger.warning("OPENAI_API_KEY not set — using hardcoded English seeds only.")
        seeds = []
        for cat in active_cats:
            for q in HARDCODED_SEEDS.get(cat, []):
                seeds.append({"behavioral_category": cat, "query": q})

    if not seeds:
        for cat in active_cats:
            for q in HARDCODED_SEEDS.get(cat, []):
                seeds.append({"behavioral_category": cat, "query": q})

    for seed in tqdm(seeds, desc="expand_queries", unit="seed"):
        expanded = _expand_seed_multilingual(seed, target_langs, cfg, logger)
        all_queries.extend(expanded)

    all_queries = _dedupe_expanded_queries(all_queries, logger)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_queries, cache_path, mode="w")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "search_config_hash": current_hash,
                "updated_at": __import__("datetime").datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info("Wrote query cache: %s", cache_path)

    if run_dir:
        gq = run_dir / "stage_1_search" / "generated_queries.jsonl"
        save_jsonl(all_queries, gq, mode="w")

    return all_queries


def _youtube_api_search(
    query: str,
    max_results: int,
    api_key: str,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    youtube = build("youtube", "v3", developerKey=api_key)
    out: list[dict[str, Any]] = []

    def search_call():
        return (
            youtube.search()
            .list(q=query, part="snippet", type="video", maxResults=min(max_results, 50))
            .execute()
        )

    try:
        sres = retry_with_backoff(lambda: search_call(), max_retries=3, retry_on=(HttpError,))
    except Exception as e:
        logger.warning("YouTube search API error for %r: %s", query, e)
        return []

    items = sres.get("items", [])
    ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
    if not ids:
        return []

    def videos_call():
        return (
            youtube.videos()
            .list(part="snippet,contentDetails,statistics", id=",".join(ids[:50]))
            .execute()
        )

    try:
        vres = retry_with_backoff(lambda: videos_call(), max_retries=3, retry_on=(HttpError,))
    except Exception as e:
        logger.warning("YouTube videos.list error: %s", e)
        return []

    for v in vres.get("items", []):
        vid = v["id"]
        sn = v.get("snippet", {})
        cd = v.get("contentDetails", {})
        st = v.get("statistics", {})
        cat_id = sn.get("categoryId", "")
        dur_sec = _parse_iso8601_duration(cd.get("duration"))
        out.append(
            {
                "video_id": vid,
                "title": sn.get("title", ""),
                "description": (sn.get("description") or "")[:500],
                "tags": sn.get("tags", []) or [],
                "category_id": cat_id,
                "category_name": YOUTUBE_CATEGORIES.get(cat_id, ""),
                "duration_iso": cd.get("duration", ""),
                "duration_seconds": dur_sec,
                "view_count": int(st.get("viewCount", 0) or 0),
                "like_count": int(st.get("likeCount", 0) or 0) if "likeCount" in st else None,
                "published_at": sn.get("publishedAt", ""),
                "channel_title": sn.get("channelTitle", ""),
            }
        )
    return out


def _ytdlp_search_fallback(query: str, max_results: int, logger: logging.Logger) -> list[dict[str, Any]]:
    url = f"ytsearch{max_results}:{query}"
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "playlistend": max_results,
        "ignoreerrors": True,
    }
    out: list[dict[str, Any]] = []
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.warning("yt-dlp search failed for %r: %s", query, e)
        return []

    entries = info.get("entries") or []
    for e in entries:
        if not e or not isinstance(e, dict):
            continue
        vid = e.get("id") or e.get("display_id")
        if not vid:
            continue
        dur = e.get("duration")
        out.append(
            {
                "video_id": vid,
                "title": e.get("title", ""),
                "description": (e.get("description") or "")[:500],
                "tags": list(e.get("tags") or []) if isinstance(e.get("tags"), list) else [],
                "category_id": str(e.get("categories", [""])[0]) if e.get("categories") else "",
                "category_name": "",
                "duration_iso": "",
                "duration_seconds": float(dur or 0),
                "view_count": e.get("view_count"),
                "like_count": e.get("like_count"),
                "published_at": e.get("upload_date", ""),
                "channel_title": e.get("uploader") or e.get("channel", ""),
            }
        )
    return out


def _query_plan_key(qrow: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(qrow.get("query", "")),
        str(qrow.get("behavioral_category", "")),
        str(qrow.get("query_language", "English")),
    )


def _candidate_row_plan_key(r: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(r.get("search_query", "")),
        str(r.get("behavioral_category", "")),
        str(r.get("query_language", "English")),
    )


def _fetch_youtube_query_batch(
    qrow: dict[str, Any],
    max_results: int,
    api_key: str,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one search.* query; used by parallel and sequential paths."""
    q = qrow["query"]
    if api_key:
        rows = _youtube_api_search(q, max_results, api_key, logger)
    else:
        rows = _ytdlp_search_fallback(q, max_results, logger)
    return qrow, rows


def run_search(
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Generate/load queries, search YouTube, dedupe, write candidates.jsonl.
    Returns (candidates, stats dict for reporting).
    """
    sns.set_theme(style="whitegrid")
    search_cfg = cfg["search"]
    max_results = int(search_cfg.get("max_results_per_query", 50))
    api_key = (search_cfg.get("youtube_api_key") or os.environ.get("YOUTUBE_API_KEY", "")).strip()

    queries = load_or_generate_queries(cfg, logger, run_dir=run_dir)
    queries, queries_before_sample = _maybe_random_sample_queries(
        queries, search_cfg, logger, run_dir
    )
    queries_planned = len(queries)

    candidates_path = run_dir / "stage_1_search" / "candidates.jsonl"
    resume_partial = bool(cfg.get("_resume_partial_search"))
    existing_rows: list[dict[str, Any]] = []
    if resume_partial and candidates_path.is_file():
        try:
            existing_rows = load_jsonl(candidates_path)
        except OSError:
            existing_rows = []

    if not resume_partial or not existing_rows:
        if candidates_path.exists():
            candidates_path.unlink()
        existing_rows = []
        queries_to_run = queries
        n_preskipped = 0
    else:
        done_keys = {_candidate_row_plan_key(r) for r in existing_rows if isinstance(r, dict)}
        queries_to_run = [q for q in queries if _query_plan_key(q) not in done_keys]
        n_preskipped = queries_planned - len(queries_to_run)

    # Pretty print sample
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        by_cat[q.get("behavioral_category", "?")].append(q)
    n_seeds = len({q.get("seed_query", q.get("query")) for q in queries}) or len(queries) // max(
        1, int(search_cfg.get("languages_per_query", 5))
    )
    sub_note = ""
    if queries_before_sample > len(queries):
        sub_note = f" (subsampled from {queries_before_sample} expanded queries)"
    print(
        f"\n📝 Search queries ({len(by_cat)} categories × ~{search_cfg.get('languages_per_query', 5)} languages → "
        f"{len(queries)} to run{sub_note}):\n"
    )
    for cat, items in list(by_cat.items())[:5]:
        for item in items[:4]:
            lang = item.get("query_language", "?")
            qq = item.get("query", "")
            print(f"    [{cat}] {lang}: {qq!r}")

    if n_preskipped:
        print(
            f"\n↻ Resume: skipping {n_preskipped} query/queries already in candidates.jsonl; "
            f"{len(queries_to_run)} left to run (parallelism = search.youtube_search_workers).\n"
        )

    seen_ids: set[str] = set()
    prior_skip = collect_skip_video_ids(cfg)
    if prior_skip:
        seen_ids.update(prior_skip)
        logger.info(
            "Search: excluding %d video_id(s) already in metadata / pipeline_log / legacy dataset "
            "(no duplicate candidates vs prior runs).",
            len(prior_skip),
        )

    dup_hits: dict[str, list[str]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    per_lang_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"queries": 0, "results": 0, "unique": 0})
    lang_unique: dict[str, set[str]] = defaultdict(set)

    for r in existing_rows:
        if not isinstance(r, dict):
            continue
        vid = r.get("video_id")
        if vid:
            seen_ids.add(str(vid))
        candidates.append(r)
        lang = str(r.get("query_language", "English"))
        if vid:
            lang_unique[lang].add(str(vid))
        per_lang_stats[lang]["results"] += 1
    keys_by_lang: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for r in existing_rows:
        if not isinstance(r, dict):
            continue
        lang = str(r.get("query_language", "English"))
        keys_by_lang[lang].add(_candidate_row_plan_key(r))
    for lang, ks in keys_by_lang.items():
        per_lang_stats[lang]["queries"] = len(ks)

    raw_workers = search_cfg.get("youtube_search_workers", 8)
    try:
        n_workers = max(1, int(raw_workers))
    except (TypeError, ValueError):
        n_workers = 8

    if not api_key:
        logger.warning(
            "Using yt-dlp fallback — no YouTube API key set "
            "(parallel workers=%s; reduce youtube_search_workers if overloaded).",
            n_workers,
        )

    def process_query_results(qrow: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        q = qrow["query"]
        behavioral = qrow.get("behavioral_category", "")
        qlang = qrow.get("query_language", "English")
        per_lang_stats[qlang]["queries"] += 1
        n_dup = 0
        for row in rows:
            vid = row["video_id"]
            row["search_query"] = q
            row["behavioral_category"] = behavioral
            row["query_language"] = qlang
            if vid in seen_ids:
                n_dup += 1
                dup_hits[vid].append(q)
                continue
            seen_ids.add(vid)
            per_lang_stats[qlang]["results"] += 1
            lang_unique[qlang].add(vid)
            candidates.append(row)
            append_jsonl(row, candidates_path)

        tqdm.write(
            f'🔍 [{behavioral} {qlang}] "{q[:60]}..." → {len(rows)} results ({n_dup} dupes)'
        )

    if not queries_to_run:
        logger.info("Search: nothing to run (all %s queries already in candidates.jsonl).", queries_planned)
    elif n_workers <= 1 or len(queries_to_run) <= 1:
        for qrow in tqdm(queries_to_run, desc="YouTube search", unit="query"):
            _, rows = _fetch_youtube_query_batch(qrow, max_results, api_key, logger)
            process_query_results(qrow, rows)
    else:
        logger.info(
            "YouTube search: parallel workers=%d (queries this run=%d)",
            n_workers,
            len(queries_to_run),
        )
        results_by_idx: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(
                    _fetch_youtube_query_batch,
                    qrow,
                    max_results,
                    api_key,
                    logger,
                ): i
                for i, qrow in enumerate(queries_to_run)
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(queries_to_run),
                desc="YouTube search",
                unit="query",
            ):
                i = futures[fut]
                try:
                    qrow, rows = fut.result()
                except Exception as e:
                    logger.warning("YouTube search task failed: %s", e)
                    qrow = queries_to_run[i]
                    rows = []
                results_by_idx[i] = (qrow, rows)
        for i in range(len(queries_to_run)):
            qrow, rows = results_by_idx[i]
            process_query_results(qrow, rows)

    total_unique = len(seen_ids)
    skip_note = f" ({n_preskipped} queries skipped from prior partial run)" if n_preskipped else ""
    print(
        f"\n✅ Search complete: {total_unique} unique video_ids in candidates from "
        f"{queries_planned} planned queries{skip_note}\n"
    )

    # Yield table
    summary_lines = [
        "Language    │ Queries │ Results │ Unique  │ % of total",
        "────────────┼─────────┼─────────┼─────────┼───────────",
    ]
    for lang in sorted(per_lang_stats.keys()):
        st = per_lang_stats[lang]
        u = len(lang_unique[lang])
        pct = 100.0 * u / total_unique if total_unique else 0.0
        summary_lines.append(
            f"{lang[:11]:11} │ {st['queries']:7} │ {st['results']:7} │ {u:7} │ {pct:5.1f}%"
        )
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (run_dir / "stage_1_search" / "search_summary.txt").write_text(summary_text, encoding="utf-8")

    # Bar chart: unique per language
    if lang_unique:
        langs = list(lang_unique.keys())
        counts = [len(lang_unique[l]) for l in langs]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(langs, counts, color="teal")
        ax.set_ylabel("Unique candidates")
        ax.set_title("Search yield by language (unique video_ids)")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        plot_path = run_dir / "stage_1_search" / "search_yield_by_language.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        alt = project_root() / "src" / "scrapers" / "data_pipeline_v2" / "reports" / "search_yield_by_language.png"
        alt.parent.mkdir(parents=True, exist_ok=True)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(langs, counts, color="teal")
        ax2.set_ylabel("Unique candidates")
        ax2.set_title("Search yield by language (unique video_ids)")
        plt.xticks(rotation=45, ha="right")
        fig2.tight_layout()
        fig2.savefig(alt, dpi=150)
        plt.close(fig2)

    stats = {
        "total_queries": queries_planned,
        "queries_skipped_resume": n_preskipped,
        "queries_before_sample": queries_before_sample,
        "unique_candidates": total_unique,
        "per_language": {k: len(v) for k, v in lang_unique.items()},
        "summary_text": summary_text,
    }
    return candidates, stats
