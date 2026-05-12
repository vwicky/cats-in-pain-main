"""Hashtag-oriented query generation + TikTok listing via Playwright (DOM).

The yt-dlp ``tiktok:tag`` playlist path is unreliable on hashtag URLs; single-video
download in ``downloader.py`` still uses yt-dlp.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.downloader import collect_skip_video_ids
from src.utils import (
    append_jsonl,
    hash_config_section,
    load_jsonl,
    pipeline_package_root,
    project_root,
    resolve_path,
    retry_with_backoff,
    save_jsonl,
)

QUERY_CACHE_META = "tiktok_generated_queries.meta.json"
QUERY_CACHE_JSONL = "tiktok_generated_queries.jsonl"

HARDCODED_TAG_SEEDS: dict[str, list[str]] = {
    "Pain/Vet": [
        "cat vet",
        "sick cat",
        "cat at vet",
        "injured cat",
        "cat crying",
        "rescue cat",
        "cat surgery",
        "cat emergency vet",
        "cat in pain",
        "limping cat",
        "cat anesthesia",
        "cat dental",
        "cat spay recovery",
        "cat wound",
        "cat iv fluid",
        "cat xray",
        "gato veterinario",
        "chat veto",
        "kot weterynarz",
        "katze tierarzt",
    ],
    "Agonistic": ["cats fighting", "cat hiss", "angry cat", "feral cat", "cat attack", "cat vs cat"],
    "Vocalizing": ["cat meowing", "cat yowling", "loud cat", "kitten meow", "siamese cat", "cat sounds"],
    "Positive_Baseline": ["cat purring", "happy cat", "cat playing", "cute cat", "cat grooming", "sleepy cat"],
    "HuntingMind": ["cat hunting", "cat stalking", "cat chattering", "cat window", "cat toy", "playful cat"],
}

BEHAVIOR_CATEGORIES: tuple[str, ...] = (
    "Pain/Vet",
    "Agonistic",
    "Vocalizing",
    "Positive_Baseline",
    "HuntingMind",
)

BEHAVIORAL_PROMPT_LINES: dict[str, str] = {
    "Pain/Vet": "- Pain/Vet: cats at vets, post-surgery, injured, in pain, paining, limping, emergency, medication, cone of shame, dental, grooming vet, distress",
    "Agonistic": "- Agonistic: cats hissing, fighting, defensive, aggressive",
    "Vocalizing": "- Vocalizing: cats yowling, calling, warning, mating sounds",
    "Positive_Baseline": "- Positive_Baseline: cats relaxed, purring, content, resting",
    "HuntingMind": "- HuntingMind: cats stalking, chattering, predatory focus",
}

BEHAVIORAL_PROMPT = """Behavioral categories (use these exact keys in output):
- Pain/Vet: cats at vets, post-surgery, injured, in pain, limping, emergency, medication, cone of shame, dental, grooming vet, distress
- Agonistic: cats hissing, fighting, defensive, aggressive
- Vocalizing: cats yowling, calling, warning, mating sounds
- Positive_Baseline: cats relaxed, purring, content, resting
- HuntingMind: cats stalking, chattering, predatory focus
"""


def _behavioral_prompt_for_active(active: tuple[str, ...]) -> str:
    lines = [BEHAVIORAL_PROMPT_LINES[c] for c in active if c in BEHAVIORAL_PROMPT_LINES]
    return "Behavioral categories (use these exact keys in output):\n" + "\n".join(lines)


def _dedupe_tags_preserve_order(tags: list[str]) -> list[str]:
    """Case-insensitive dedupe; keep first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        s = str(t).lstrip("#").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _tag_to_url(tag: str) -> str:
    """Build TikTok hashtag page URL (strip #, encode)."""
    t = (tag or "").strip().lstrip("#").strip()
    if not t:
        return ""
    return f"https://www.tiktok.com/tag/{quote(t, safe='')}"


def _openai_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _target_n_by_category(
    search_cfg: dict[str, Any],
    active: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Per-category hashtag counts; Pain/Vet gets more slots when boost/extra config is set.
    Categories not listed in *active* get count 0 (no GPT/hardcoded hashtags for them).
    """
    if active is None:
        active = BEHAVIOR_CATEGORIES
    n_base = max(1, int(search_cfg.get("seed_queries_per_category", 6)))
    explicit = search_cfg.get("seed_queries_per_category_pain_vet")
    if explicit is not None and str(explicit).strip() != "":
        n_pain = max(1, int(explicit))
    else:
        boost = float(search_cfg.get("pain_vet_hashtag_boost", 1.5))
        n_pain = max(n_base, int(round(n_base * boost)))
    targets: dict[str, int] = {}
    for c in BEHAVIOR_CATEGORIES:
        if c not in active:
            targets[c] = 0
        elif c == "Pain/Vet":
            targets[c] = n_pain
        else:
            targets[c] = n_base
    return targets


def _normalize_behavioral_category(raw: str) -> str:
    s = (raw or "").strip()
    for k in BEHAVIOR_CATEGORIES:
        if s == k or s.lower() == k.lower():
            return k
    low = s.lower().replace(" ", "")
    if "pain" in low and "vet" in low:
        return "Pain/Vet"
    if "agonistic" in low or "aggressive" in low:
        return "Agonistic"
    if "vocal" in low or "meow" in low:
        return "Vocalizing"
    if "positive" in low or "baseline" in low:
        return "Positive_Baseline"
    if "hunt" in low:
        return "HuntingMind"
    return "Positive_Baseline"


def _active_behavior_categories(search_cfg: dict[str, Any], logger: logging.Logger) -> tuple[str, ...]:
    """Config search.behavior_categories: subset of BEHAVIOR_CATEGORIES; default = all five."""
    raw = search_cfg.get("behavior_categories")
    if raw is None or raw == "" or raw == []:
        return BEHAVIOR_CATEGORIES
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if not s:
            continue
        norm = _normalize_behavioral_category(s)
        if norm not in BEHAVIOR_CATEGORIES:
            logger.warning("Ignoring unknown behavior category %r (expected one of %s)", s, BEHAVIOR_CATEGORIES)
            continue
        if norm not in out:
            out.append(norm)
    if not out:
        logger.warning("behavior_categories was empty after validation; using all categories")
        return BEHAVIOR_CATEGORIES
    return tuple(out)


def _load_prior_hashtag_labels(cfg: dict[str, Any], logger: logging.Logger) -> list[str]:
    """Tag labels from the last saved query cache — GPT should avoid repeating these pools."""
    search_cfg = cfg.get("search") or {}
    if not bool(search_cfg.get("query_generation_avoid_prior_hashtags", True)):
        return []
    max_n = max(0, int(search_cfg.get("prior_hashtag_hints_max", 250)))
    if max_n == 0:
        return []
    cache_path = pipeline_package_root() / "cache" / QUERY_CACHE_JSONL
    if not cache_path.is_file():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for row in load_jsonl(cache_path):
        lab = row.get("tag_label") or row.get("tag")
        if not lab:
            continue
        s = str(lab).lstrip("#").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_n:
            break
    if out:
        logger.info(
            "GPT hashtag generation: passing %d prior tag_label(s) as avoid-list "
            "(seek new angles vs cached hashtag pages).",
            len(out),
        )
    return out


def _generate_seed_hashtags_openai(cfg: dict[str, Any], logger: logging.Logger) -> list[dict[str, Any]]:
    """Returns list of {behavioral_category, hashtags: list[str]}."""
    search_cfg = cfg["search"]
    active = _active_behavior_categories(search_cfg, logger)
    targets = _target_n_by_category(search_cfg, active)
    total_hashtags = sum(targets[c] for c in BEHAVIOR_CATEGORIES)
    model = search_cfg.get("query_generation_model", "gpt-4o-mini")
    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("OpenAI key required for GPT hashtag generation")

    prior_labels = _load_prior_hashtag_labels(cfg, logger)
    avoid_block = ""
    if prior_labels:
        joined = ", ".join(prior_labels)
        max_chars = int(search_cfg.get("prior_hashtag_hints_max_chars", 12000))
        if len(joined) > max_chars:
            joined = joined[: max_chars - 3] + "..."
        avoid_block = (
            "\n\nALREADY USED IN PREVIOUS PIPELINE RUNS (do NOT repeat these strings or trivial edits; "
            "pick novel medical/slang/regional angles so TikTok hashtag pages surface different videos):\n"
            f"{joined}\n"
        )

    client = _openai_client(api_key)
    qtemp = float(search_cfg.get("query_generation_temperature", 0.62))
    require_cat = bool(search_cfg.get("query_generation_require_cat_token", False))
    cat_token_block = ""
    if require_cat:
        cat_token_block = (
            "CAT TOKEN (mandatory): Every hashtag must clearly refer to cats. "
            "Include the substring \"cat\" (e.g. cats, catmom, catvet, scaredycat) OR a common "
            "non-English cat word (gato, gatto, chat, kot, neko, кот, kedi, mèo, mačka, kočka, kissa, katt). "
            "Do not emit vet/dog/human-only tags with no cat marker. "
        )
    solo_pain_block = ""
    if active == ("Pain/Vet",):
        solo_pain_block = (
            "SCOPE: This run uses ONLY the Pain/Vet category—hashtags must target cats in pain, "
            "paining, at the vet, injured, post-surgery, emergency care, medication, limping, dental, "
            "anesthesia, or medical distress. Omit playful, hunting, aggression-only, or relaxed "
            "baseline angles. "
        )
    system = (
        "You are a research assistant building a cat behavior dataset from TikTok. "
        f"{solo_pain_block}"
        "Generate diverse TikTok **hashtag** search terms (not full sentences) that would surface "
        "videos of cats in the behavioral categories below. Prefer short, realistic hashtag-style "
        "tokens people use on TikTok (e.g. catsoftiktok, meow, kitten). "
        f"{cat_token_block}"
        "CRITICAL: Each hashtag must be clearly distinct—avoid near-duplicates within a category "
        "(no trivial plural/suffix variants of the same idea). Vary angles: medical, emotional, "
        "breed/context, sounds, rescue, everyday life, and mixed-language romanizations where useful. "
        "For **Pain/Vet**, heavily favor pain, injury, veterinary exam, surgery recovery, emergency clinic, "
        "medication, limping, dental, anesthesia, distress vocalization at vet, and regional vet terms. "
        "Avoid repeating ultra-generic mega-tags that all surface the same viral clips across many "
        "searches (vary specificity: procedures, settings, breeds, emotions, slang). "
        "Include a substantial share of non-English or romanized regional tags (e.g. gato, kot, "
        "chat, neko, кот) so hashtag *pages* differ and results overlap less with English-only pools. "
        "Across the full JSON output, hashtags must be unique (case-insensitive); overlap across "
        "categories is OK only if the angle truly differs. "
        "If an avoid-list of prior hashtags is given, treat it as hard inspiration to stay away from "
        "those exact pools. "
        'Return ONLY a JSON object with key "items" whose value is an array of objects with '
        '"behavioral_category" (string, one of: '
        + ", ".join(f'"{c}"' for c in active)
        + ') and "hashtags" (array of strings, no # prefix).'
    )
    counts_lines = "\n".join(
        f"- {cat}: exactly {targets[cat]} hashtags"
        for cat in BEHAVIOR_CATEGORIES
        if targets[cat] > 0
    )
    count_note = (
        "(Pain/Vet is intentionally larger than other categories when multiple categories are used) "
        if len(active) > 1 and "Pain/Vet" in active
        else ""
    )
    prompt_block = _behavioral_prompt_for_active(active)
    user = (
        f"{prompt_block}\n\n"
        f"Produce exactly these counts {count_note}:\n{counts_lines}\n"
        f"Total distinct hashtags across all categories: {total_hashtags}.\n"
        "Maximize lexical diversity: include niche and regional tags, not only generic words like "
        "cat or kitten repeated with small edits."
        f"{avoid_block}\n"
        "Return JSON only."
    )

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=qtemp,
            response_format={"type": "json_object"},
        )

    resp = retry_with_backoff(call, max_retries=3)
    text = resp.choices[0].message.content or "{}"
    data = json.loads(text)
    if isinstance(data, list):
        items = data
    else:
        items = data.get("items", data.get("queries", data.get("hashtags", [])))
    if isinstance(items, dict):
        items = [items]
    out: list[dict[str, Any]] = []
    by_cat: dict[str, list[str]] = {c: [] for c in BEHAVIOR_CATEGORIES}
    for item in items:
        if not isinstance(item, dict):
            continue
        tags = item.get("hashtags") or item.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        cat = _normalize_behavioral_category(str(item.get("behavioral_category", "")))
        if tags:
            by_cat.setdefault(cat, [])
            by_cat[cat].extend(str(t) for t in tags)

    global_seen: set[str] = set()
    for cat in BEHAVIOR_CATEGORIES:
        want = targets[cat]
        if want <= 0:
            continue
        cleaned = _dedupe_tags_preserve_order(by_cat.get(cat, []))
        merged: list[str] = []
        for t in cleaned:
            k = t.lower()
            if k in global_seen:
                continue
            global_seen.add(k)
            merged.append(t)
            if len(merged) >= want:
                break
        if merged:
            out.append({"behavioral_category": cat, "hashtags": merged[:want]})

    if len(out) < 1:
        logger.warning("GPT returned no hashtag groups; using hardcoded tag seeds")
        return []
    return out


def _expand_seed_multilingual(
    seed: dict[str, Any],
    target_langs: list[str],
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """One row per hashtag URL; assign query_language by cycling target_langs for diversity."""
    tags = seed.get("hashtags") or []
    cat = seed.get("behavioral_category", "Positive_Baseline")
    rows: list[dict[str, Any]] = []
    if not tags:
        tags = ["cat"]
    langs = target_langs if target_langs else ["English"]
    for i, t in enumerate(tags):
        u = _tag_to_url(str(t))
        if not u:
            continue
        lang = langs[i % len(langs)]
        rows.append(
            {
                "behavioral_category": cat,
                "query_language": lang,
                "source_hashtag_url": u,
                "tag_label": str(t).lstrip("#"),
                "seed_tags": list(tags),
            }
        )
    return rows if rows else [
        {
            "behavioral_category": cat,
            "query_language": "English",
            "source_hashtag_url": _tag_to_url("cat"),
            "tag_label": "cat",
            "seed_tags": [],
        }
    ]


def load_or_generate_queries(
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load cached queries if search config hash matches; else generate."""
    pkg = pipeline_package_root()
    cache_dir = pkg / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / QUERY_CACHE_META
    cache_path = cache_dir / QUERY_CACHE_JSONL

    current_hash = hash_config_section(cfg, "search")

    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("search_config_hash") == current_hash and cache_path.is_file():
                logger.info("Using cached generated TikTok hashtag queries (search config hash matches).")
                rows = load_jsonl(cache_path)
                if rows:
                    return rows
        except (json.JSONDecodeError, OSError):
            pass

    search_cfg = cfg["search"]
    active = _active_behavior_categories(search_cfg, logger)
    logger.info("Search: behavioral_categories=%s", ", ".join(active))
    langs_n = int(search_cfg.get("languages_per_query", 5))
    target_langs = (search_cfg.get("target_languages") or ["English"])[:langs_n]

    api_key = (search_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    seeds: list[dict[str, Any]] = []

    if api_key:
        try:
            g = _generate_seed_hashtags_openai(cfg, logger)
            if g:
                seeds = g
        except Exception as e:
            logger.warning("GPT hashtag generation failed: %s — using hardcoded seeds", e)
    if not seeds:
        logger.warning("Using hardcoded TikTok-style tag seeds (no or failed GPT).")
        targets = _target_n_by_category(search_cfg, active)
        for cat, tag_list in HARDCODED_TAG_SEEDS.items():
            n = targets.get(cat, 0)
            if n <= 0:
                continue
            expanded: list[str] = []
            i = 0
            while len(expanded) < n:
                expanded.append(tag_list[i % len(tag_list)])
                i += 1
            seeds.append({"behavioral_category": cat, "hashtags": expanded[:n]})

    all_queries: list[dict[str, Any]] = []
    for seed in tqdm(seeds, desc="expand_hashtag_urls", unit="seed"):
        expanded = _expand_seed_multilingual(seed, target_langs, cfg, logger)
        all_queries.extend(expanded)

    seen_u: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for q in all_queries:
        key = (q.get("source_hashtag_url", ""), q.get("behavioral_category", ""))
        if key[0] and key not in seen_u:
            seen_u.add(key)
            deduped.append(q)
    all_queries = deduped

    extra = search_cfg.get("extra_seed_urls") or []
    extra_behavior = (
        "Pain/Vet"
        if "Pain/Vet" in active
        else (active[0] if active else "Positive_Baseline")
    )
    for u in extra:
        u = str(u).strip()
        if u:
            all_queries.append(
                {
                    "behavioral_category": extra_behavior,
                    "query_language": "English",
                    "source_hashtag_url": u,
                    "tag_label": u,
                    "seed_tags": [],
                }
            )

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
    logger.info("Wrote TikTok query cache: %s", cache_path)

    if run_dir:
        gq = run_dir / "stage_1_search" / "generated_queries.jsonl"
        save_jsonl(all_queries, gq, mode="w")

    return all_queries


def _normalize_tiktok_video_href(href: str | None) -> str | None:
    """Return canonical https URL for a TikTok /video/ link, or None."""
    if not href or "/video/" not in href:
        return None
    h = href.strip().split("?")[0]
    if h.startswith("//"):
        h = "https:" + h
    elif h.startswith("/"):
        h = "https://www.tiktok.com" + h
    elif h.startswith("http://") or h.startswith("https://"):
        pass
    else:
        return None
    if "tiktok.com" not in h.lower().replace("www.", ""):
        return None
    return h


def scrape_hashtag_page_playwright(
    hashtag_url: str,
    max_results: int,
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Scroll a TikTok hashtag page in Chromium and collect ``/video/`` links.
    Use ``playwright.headless: false`` locally if TikTok shows CAPTCHA/login/cookie UI.
    Deep metadata comes later from yt-dlp in the download stage.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(
            "Playwright is required for TikTok hashtag search. "
            "Install: pip install playwright && playwright install chromium"
        ) from e

    sc = cfg.get("search") or {}
    pw_cfg = sc.get("playwright") or {}
    headless = bool(pw_cfg.get("headless", True))
    goto_timeout = int(pw_cfg.get("goto_timeout_ms", 20000))
    wait_vid = int(pw_cfg.get("wait_for_video_timeout_ms", 15000))
    post_goto_sleep = float(pw_cfg.get("post_goto_sleep_sec", 3))
    max_scroll = int(pw_cfg.get("max_scroll_attempts", 10))
    pause = float(pw_cfg.get("scroll_pause_sec", 2.5))
    wheel_dy = int(pw_cfg.get("wheel_delta_y", 3000))
    vw = int(pw_cfg.get("viewport_width", 1280))
    vh = int(pw_cfg.get("viewport_height", 720))
    ua = str(pw_cfg.get("user_agent", "")).strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    user_data_rel = str(pw_cfg.get("user_data_dir", "src/scrapers/tiktok/tiktok_profile")).strip()
    profile_dir = resolve_path(project_root(), user_data_rel)
    profile_dir.mkdir(parents=True, exist_ok=True)

    video_urls: set[str] = set()

    with sync_playwright() as p:
        # Persistent profile: cookies/session survive across runs (log in / CAPTCHA once).
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=headless,
            viewport={"width": vw, "height": vh},
            user_agent=ua,
            locale="en-US",
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(hashtag_url, timeout=goto_timeout, wait_until="domcontentloaded")
                if post_goto_sleep > 0:
                    time.sleep(post_goto_sleep)
                page.wait_for_selector('a[href*="/video/"]', timeout=wait_vid)
                scroll_attempts = 0
                while len(video_urls) < max_results and scroll_attempts < max_scroll:
                    elements = page.query_selector_all('a[href*="/video/"]')
                    for el in elements:
                        href = el.get_attribute("href")
                        nu = _normalize_tiktok_video_href(href)
                        if nu:
                            video_urls.add(nu)
                    if len(video_urls) >= max_results:
                        break
                    page.mouse.wheel(0, wheel_dy)
                    time.sleep(pause)
                    scroll_attempts += 1
            except Exception as e:
                logger.warning("Playwright extraction error on %s: %s", hashtag_url, e)
        finally:
            context.close()

    entries: list[dict[str, Any]] = []
    for url in list(video_urls)[:max_results]:
        m = re.search(r"/video/(\d+)", url)
        if not m:
            continue
        vid_id = m.group(1)
        entries.append(
            {
                "video_id": vid_id,
                "webpage_url": url,
                "title": "",
                "description": "",
                "hashtags": [],
                "tags": [],
                "duration_seconds": None,
            }
        )
    return entries


def _entry_to_candidate(
    entry: dict[str, Any],
    source_url: str,
    behavioral: str,
    qlang: str,
) -> dict[str, Any] | None:
    vid = entry.get("id") or entry.get("video_id") or entry.get("display_id")
    if not vid:
        return None
    vid = str(vid)
    title = entry.get("title") or entry.get("description") or ""
    if title and len(str(title)) > 500:
        title = str(title)[:500]
    desc = entry.get("description") or ""
    if desc and len(str(desc)) > 500:
        desc = str(desc)[:500]
    hashtags = entry.get("hashtags") or []
    if isinstance(hashtags, str):
        hashtags = [hashtags]
    elif not isinstance(hashtags, list):
        hashtags = []
    tags = list(entry.get("tags") or []) if isinstance(entry.get("tags"), list) else []
    duration = entry.get("duration")
    try:
        dur_f = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        dur_f = 0.0
    webpage_url = (
        entry.get("webpage_url")
        or entry.get("url")
        or entry.get("original_url")
        or f"https://www.tiktok.com/video/{vid}"
    )
    if isinstance(webpage_url, str) and webpage_url.startswith("http"):
        pass
    else:
        webpage_url = f"https://www.tiktok.com/video/{vid}"

    channel = entry.get("uploader") or entry.get("channel") or entry.get("uploader_id") or ""

    return {
        "video_id": vid,
        "title": str(title) if title is not None else "",
        "description": str(desc) if desc is not None else "",
        "tags": tags,
        "hashtags": [str(h).lstrip("#") for h in hashtags if h is not None],
        "category_id": "",
        "category_name": "",
        "duration_iso": "",
        "duration_seconds": dur_f,
        "view_count": entry.get("view_count"),
        "like_count": entry.get("like_count"),
        "published_at": str(entry.get("upload_date") or entry.get("timestamp") or ""),
        "channel_title": str(channel) if channel is not None else "",
        "webpage_url": str(webpage_url).strip(),
        "search_query": source_url,
        "source_hashtag_url": source_url,
        "behavioral_category": behavioral,
        "query_language": qlang,
        "track": entry.get("track"),
        "artist": entry.get("artist"),
        "album": entry.get("album"),
    }


def candidate_from_ytdlp_info(row: dict[str, Any], info: dict[str, Any]) -> dict[str, Any] | None:
    """Merge yt-dlp ``extract_info(..., download=False)`` into a search-stage row.

    Keeps ``video_id`` and ``webpage_url`` from the row when present so hashtag discovery
    stays aligned with the URL we collected.
    """
    vid = str(row.get("video_id") or "").strip()
    if not vid:
        return None
    source_url = str(row.get("source_hashtag_url") or row.get("search_query") or "").strip()
    behavioral = str(row.get("behavioral_category") or "")
    qlang = str(row.get("query_language") or "")
    entry = dict(info)
    entry["id"] = vid
    page = row.get("webpage_url")
    if isinstance(page, str) and page.strip().startswith("http"):
        entry.setdefault("webpage_url", page.strip())
    c = _entry_to_candidate(entry, source_url, behavioral, qlang)
    if not c:
        return None
    c["video_id"] = vid
    if isinstance(page, str) and page.strip().startswith("http"):
        c["webpage_url"] = page.strip()
    return c


def run_search(
    cfg: dict[str, Any],
    logger: logging.Logger,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sns.set_theme(style="whitegrid")
    search_cfg = cfg["search"]
    max_results = int(search_cfg.get("max_results_per_query", 50))

    logger.info(
        "Search stage: hashtag video links via Playwright (Chromium). "
        "Ensure `playwright install chromium` has been run once."
    )
    pw_cfg = (search_cfg.get("playwright") or {})
    udr = str(pw_cfg.get("user_data_dir", "src/scrapers/tiktok/tiktok_profile")).strip()
    logger.info("Playwright persistent profile (cookies/session): %s", resolve_path(project_root(), udr))

    queries = load_or_generate_queries(cfg, logger, run_dir=run_dir)

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        by_cat[q.get("behavioral_category", "?")].append(q)
    print(
        f"\nTikTok hashtag URLs ({len(queries)} total from {len(by_cat)} categories):\n"
    )
    for cat, items in list(by_cat.items())[:5]:
        for item in items[:3]:
            print(f"    [{cat}] {item.get('query_language', '?')}: {item.get('source_hashtag_url', '')!r}")

    seen_ids: set[str] = set()
    prior_skip = collect_skip_video_ids(cfg)
    if prior_skip:
        seen_ids.update(prior_skip)
        logger.info(
            "Search: excluding %d video_id(s) already in metadata / pipeline_log / legacy dataset "
            "(scraped TikTok IDs are skipped so you do not re-queue processed videos).",
            len(prior_skip),
        )

    dup_hits: dict[str, list[str]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    per_lang_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"queries": 0, "results": 0, "unique": 0})
    lang_unique: dict[str, set[str]] = defaultdict(set)

    candidates_path = run_dir / "stage_1_search" / "candidates.jsonl"
    if candidates_path.exists():
        candidates_path.unlink()

    for qrow in tqdm(queries, desc="TikTok hashtag scrape (Playwright)", unit="url"):
        url = qrow.get("source_hashtag_url") or ""
        behavioral = qrow.get("behavioral_category", "")
        qlang = qrow.get("query_language", "English")
        if not url:
            continue
        per_lang_stats[qlang]["queries"] += 1

        try:
            rows = scrape_hashtag_page_playwright(url, max_results, cfg, logger)
        except ImportError:
            raise
        except Exception as e:
            logger.warning("Hashtag scrape failed for %r: %s", url, e)
            rows = []

        n_dup = 0
        for entry in rows:
            cand = _entry_to_candidate(entry, url, behavioral, qlang)
            if not cand:
                continue
            vid = cand["video_id"]
            if vid in seen_ids:
                n_dup += 1
                dup_hits[vid].append(url)
                continue
            seen_ids.add(vid)
            per_lang_stats[qlang]["results"] += 1
            lang_unique[qlang].add(vid)
            candidates.append(cand)
            append_jsonl(cand, candidates_path)

        tqdm.write(
            f'[{behavioral} {qlang}] {url[:70]}... -> {len(rows)} entries ({n_dup} dupes)'
        )

    total_unique = len(seen_ids)
    print(f"\nSearch complete: {total_unique} unique candidates from {len(queries)} hashtag URLs\n")

    summary_lines = [
        "Language    | Queries | Results | Unique  | % of total",
        "------------+---------+---------+---------+----------",
    ]
    for lang in sorted(per_lang_stats.keys()):
        st = per_lang_stats[lang]
        u = len(lang_unique[lang])
        pct = 100.0 * u / total_unique if total_unique else 0.0
        summary_lines.append(
            f"{lang[:11]:11} | {st['queries']:7} | {st['results']:7} | {u:7} | {pct:5.1f}%"
        )
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (run_dir / "stage_1_search" / "search_summary.txt").write_text(summary_text, encoding="utf-8")

    if lang_unique:
        langs = list(lang_unique.keys())
        counts = [len(lang_unique[l]) for l in langs]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(langs, counts, color="teal")
        ax.set_ylabel("Unique candidates")
        ax.set_title("TikTok search yield by language (unique video_ids)")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(run_dir / "stage_1_search" / "search_yield_by_language.png", dpi=150)
        plt.close(fig)
        rep = project_root() / "src" / "scrapers" / "tiktok" / "reports"
        rep.mkdir(parents=True, exist_ok=True)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(langs, counts, color="teal")
        ax2.set_ylabel("Unique candidates")
        ax2.set_title("TikTok search yield by language (unique video_ids)")
        plt.xticks(rotation=45, ha="right")
        fig2.tight_layout()
        fig2.savefig(rep / "search_yield_by_language.png", dpi=150)
        plt.close(fig2)

    stats = {
        "total_queries": len(queries),
        "unique_candidates": total_unique,
        "per_language": {k: len(v) for k, v in lang_unique.items()},
        "summary_text": summary_text,
    }
    return candidates, stats
