"""
GPT-assisted search query generation — aligned with data_pipeline_v2 / TikTok.

Produces rows: behavioral_category, query_language, query, seed_query.
Caches under dailymotion_scraper/logs/ when the search config hash matches.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utils import hash_config_section, load_jsonl, project_root, save_jsonl

# Same five categories as data_pipeline_v2 — phrased for Dailymotion: **home / pet / close-up**
# clips that survive YOLO+audio (avoid vague terms that match TV news or disaster “alert” strings).
HARDCODED_SEEDS: dict[str, list[str]] = {
    "Pain/Vet": [
        "kitten first vet visit owner filmed",
        "cat meowing at vet examination",
        "sick cat resting home recovery",
        "cat carrier trip veterinarian",
        "injured cat rescued home video",
        "cat after surgery cone home",
    ],
    "Agonistic": [
        "two cats fighting living room",
        "angry cat growling bath",
        "cat hissing at vacuum",
        "feral cat hiss close up",
        "cats fighting loud hiss",
        "angry cat swatting hand",
    ],
    "Vocalizing": [
        "Siamese cat loud meow close",
        "cat chattering at window birds",
        "kitten meowing crying litter",
        "cat yowling night home",
        "cat trilling chirping owner",
        "loudest purr cat world record",
    ],
    "Positive_Baseline": [
        "cat purring loud lap asmr",
        "kitten kneading blanket purr",
        "sleepy cat couch snoring",
        "happy cat playing owner home",
        "cat grooming slow blink",
        "mother cat kittens nursing",
    ],
    "HuntingMind": [
        "cat chattering teeth window bird",
        "kitten stalking toy pounce",
        "cat stalking laser pointer",
        "indoor cat hunting feather toy",
        "cat tail twitch bird watching",
        "barn cat mouse catch",
    ],
}

BEHAVIORAL_PROMPT = """Behavioral categories (use these exact keys in output):
- Pain/Vet: vet visits, illness, injury, recovery — prefer **owner-filmed pet** angles, not generic “breaking” news.
- Agonistic: fights, hisses, growls, defense — **home/street** clips, not sport or human fights.
- Vocalizing: meows, yowls, trills, chattering, loud purrs — sounds a **microphone** can pick up.
- Positive_Baseline: purring, play, grooming, sleep, kneading — calm **indoor cat** footage.
- HuntingMind: stalking, chattering, pounce prep — **window/bird/toy** focus.

Dailymotion retrieval tip: short queries that look like **real user uploads** (home, ASMR, funny pet, close up) work better than abstract phrases that match TV or news uploads.
"""

QUERY_CACHE_META = "dm_generated_queries.meta.json"
QUERY_CACHE_JSONL = "dm_generated_queries.jsonl"


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _retry_call(fn, max_retries: int = 3):
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise last


def _dedupe_seeds_per_category(raw: list[dict[str, Any]], n_per_cat: int) -> list[dict[str, Any]]:
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
        key = q.lower()
        if key in seen[cat] or counts[cat] >= n_per_cat:
            continue
        seen[cat].add(key)
        counts[cat] += 1
        out.append({"behavioral_category": cat, "query": q})
    return out


def _generate_seed_queries_openai(cfg: dict[str, Any], logger: logging.Logger) -> list[dict[str, Any]]:
    search_cfg = cfg["search"]
    n = int(search_cfg.get("seed_queries_per_category", 6))
    model = search_cfg.get("query_generation_model", "gpt-4o-mini")
    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("OpenAI key required for GPT seed generation")

    client = _openai_client(api_key)
    system = (
        "You are a research assistant helping build a cat behavior dataset. "
        "Generate diverse **Dailymotion** search queries (short keyword phrases, "
        "like real users type in Dailymotion search) that would find **uploaded pet/home videos** "
        "of cats exhibiting the behavioral categories below. "
        "Queries should be natural for Dailymotion — not long YouTube-style sentences. "
        "PRIORITIZE queries that surface: owner-filmed clips, close-up, single-pet focus, "
        "clear sounds (meow, purr, hiss), living room / vet exam room / window bird — "
        "these match clips that pass video+audio screening. "
        "AVOID query wording that mainly matches **TV news, broadcasters, disaster alerts, or sports** "
        "(e.g. vague “alert”, “breaking”, standalone “rescue” without “cat”); avoid generic "
        "“funny compilation” as the *only* content (add a concrete cat behavior word). "
        "CRITICAL: Each query must be clearly distinct—avoid near-duplicates within a category. "
        "Do not repeat the same short phrase across categories. "
        "Vary angles: home vs vet, indoor vs window, sounds, body language, breed-agnostic. "
        'Return ONLY a JSON object with key "queries" whose value is an array of objects with '
        '"behavioral_category" (string, one of the five keys below) and "query" (string).'
    )
    user = (
        f"{BEHAVIORAL_PROMPT}\n\nGenerate exactly {n} distinct queries per category "
        f"({5 * n} total). Maximize lexical diversity. Return JSON only."
    )

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.62,
            response_format={"type": "json_object"},
        )

    resp = _retry_call(call, max_retries=3)
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
    out = _dedupe_seeds_per_category(raw_rows, n)
    if len(out) < 5:
        logger.warning("GPT returned few seeds; padding with hardcoded seeds")
        out = []
        for cat, seeds in HARDCODED_SEEDS.items():
            for q in seeds[:n]:
                out.append({"behavioral_category": cat, "query": q})
    return out


def dedupe_expanded_query_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep first occurrence per normalized search string (drops redundant API work)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = str(r.get("query", "")).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    dropped = len(rows) - len(out)
    return out, dropped


def prepare_query_rows_for_search(
    rows: list[dict[str, Any]],
    search_cfg: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Dedupe identical query strings and optionally shuffle order before search."""
    out = list(rows)
    if search_cfg.get("dedupe_identical_queries", True):
        out, n_drop = dedupe_expanded_query_rows(out)
        if n_drop:
            logger.info(
                "Deduped identical search strings: removed %d rows → %d unique queries",
                n_drop,
                len(out),
            )
    if search_cfg.get("shuffle_query_order", True):
        seed = search_cfg.get("shuffle_seed")
        if seed is not None:
            rng = random.Random(int(seed))
            rng.shuffle(out)
            logger.info("Shuffled search query order (deterministic shuffle_seed=%s)", seed)
        else:
            random.shuffle(out)
            logger.info("Shuffled search query order (shuffle_seed unset — order differs each run)")
    return out


def _expand_seed_multilingual(
    seed: dict[str, Any],
    target_langs: list[str],
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
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
    system = (
        "Translate the following Dailymotion search query into the specified languages. "
        "Make translations natural — how a native speaker would search for **home pet cat videos**, "
        "not TV news. "
        "IMPORTANT: In Romance languages avoid literal word pairs that mean **news/weather alerts** "
        "(e.g. Portuguese/Spanish: do not mirror English “alert” + “cat” as “alerta de gato” — "
        "prefer phrases like “gato miando”, “gato bravo em casa”, “gato no veterinário”). "
        "In French prefer “chat qui miaule”, “chat en colère” over ambiguous “alerte” unless it clearly "
        "means pet emergency. "
        'Return a JSON object with key "translations" containing an array of objects with '
        "fields: language, query."
    )
    user = json.dumps({"query": base_q, "languages": target_langs})

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

    try:
        resp = _retry_call(call, max_retries=3)
        text = resp.choices[0].message.content or "{}"
        data = json.loads(text)
        translations = data.get("translations", data) if isinstance(data, dict) else []
        if isinstance(translations, dict):
            translations = [translations]
    except Exception as e:
        logger.warning("Multilingual expand failed for %r: %s — English only", base_q, e)
        translations = []

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
    Load cached queries if search config hash matches; else generate via GPT + multilingual expand.
    Returns rows with behavioral_category, query_language, query, seed_query.
    """
    root = project_root()
    logs_dir = root / "src" / "scrapers" / "dailymotion" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    meta_path = logs_dir / QUERY_CACHE_META
    cache_path = logs_dir / QUERY_CACHE_JSONL

    current_hash = hash_config_section(cfg, "search")

    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("search_config_hash") == current_hash and cache_path.is_file():
                logger.info("Using cached Dailymotion generated queries (search config hash matches).")
                rows = load_jsonl(cache_path)
                if rows:
                    return rows
        except (json.JSONDecodeError, OSError):
            pass

    search_cfg = cfg["search"]
    langs_n = int(search_cfg.get("languages_per_query", 5))
    target_langs = (search_cfg.get("target_languages") or ["English"])[:langs_n]

    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    all_queries: list[dict[str, Any]] = []

    if api_key:
        try:
            seeds = _generate_seed_queries_openai(cfg, logger)
        except Exception as e:
            logger.warning("GPT seed generation failed: %s — using hardcoded English seeds", e)
            seeds = []
            for cat, qs in HARDCODED_SEEDS.items():
                n = int(search_cfg.get("seed_queries_per_category", 6))
                for q in qs[:n]:
                    seeds.append({"behavioral_category": cat, "query": q})
    else:
        logger.warning("OPENAI_API_KEY not set — using hardcoded English seeds only.")
        seeds = []
        n = int(search_cfg.get("seed_queries_per_category", 6))
        for cat, qs in HARDCODED_SEEDS.items():
            for q in qs[:n]:
                seeds.append({"behavioral_category": cat, "query": q})

    if not seeds:
        for cat, qs in HARDCODED_SEEDS.items():
            for q in qs:
                seeds.append({"behavioral_category": cat, "query": q})

    for seed in tqdm(seeds, desc="expand_queries", unit="seed"):
        expanded = _expand_seed_multilingual(seed, target_langs, cfg, logger)
        all_queries.extend(expanded)

    all_queries, _ = dedupe_expanded_query_rows(all_queries)
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
    logger.info("Wrote Dailymotion query cache: %s", cache_path)

    return all_queries
