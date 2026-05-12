"""Text-only GPT filter for Dailymotion candidates (metadata batching)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from config import (
    GPT_BATCH_SIZE,
    GPT_COST_PER_1K_TOKENS,
    GPT_MAX_RETRIES,
    GPT_MODEL,
    GPT_PARALLEL_WORKERS,
    GPT_TEMPERATURE,
)

from utils import save_jsonl

_RULES_PATH = Path(__file__).resolve().parent / "config" / "gpt_filter_rules.md"


def _safe_str(x: Any, limit: int = 400) -> str:
    if x is None:
        return ""
    try:
        s = str(x).strip()
        return s[:limit] if len(s) > limit else s
    except Exception:
        return ""


def _build_user_message(batch: list[dict]) -> str:
    lines: list[str] = []
    lines.append(
        "Return a JSON object with a single key \"evaluations\" whose value is an array of objects, "
        "one per video below, in order. Each object MUST include:\n"
        '  "video_id" (string),\n'
        '  "decision" ("keep" | "discard"),\n'
        '  "reason" (string),\n'
        '  "confidence" ("high" | "medium" | "low"),\n'
        '  "is_cat_behavior" (boolean),\n'
        '  "behavior_category" (string — one of: Vocalizing, Agonistic, Pain_or_vet, '
        "Play_or_grooming, Resting_or_ambient, Other_cat_behavior, Uncertain),\n"
        '  "reject_reason" (string or null).\n'
    )
    for i, v in enumerate(batch, 1):
        vid = _safe_str(v.get("video_id"), 80)
        title = _safe_str(v.get("title"), 220)
        desc = _safe_str(v.get("description"), 320)
        tags = v.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags_s = [str(t) for t in tags[:15]]
        ch = _safe_str(v.get("channel_title") or v.get("channel"), 120)
        dur = v.get("duration_sec", v.get("duration_seconds", 0))
        views = v.get("views_total", 0)
        wurl = _safe_str(v.get("webpage_url") or v.get("url"), 300)
        lines.append(
            f"{i}. video_id={vid}\n"
            f"   title={title!r}\n"
            f"   description={desc!r}\n"
            f"   tags={tags_s}\n"
            f"   channel_title={ch!r}\n"
            f"   duration_sec={dur}\n"
            f"   views_total={views}\n"
            f"   webpage_url={wurl!r}\n"
        )
    return "\n".join(lines)


def _parse_evaluations(raw_text: str, batch: list[dict]) -> list[dict[str, Any]]:
    """Parse model JSON; on failure return default keep-low for each."""
    default = [
        {
            "video_id": v.get("video_id"),
            "decision": "keep",
            "reason": "parse fallback",
            "confidence": "low",
            "is_cat_behavior": True,
            "behavior_category": "Uncertain",
            "reject_reason": None,
        }
        for v in batch
    ]
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return default
        else:
            return default

    if isinstance(data, dict) and "evaluations" in data:
        arr = data["evaluations"]
    elif isinstance(data, list):
        arr = data
    else:
        return default

    if not isinstance(arr, list) or not arr:
        return default

    by_id = {str(e.get("video_id")): e for e in arr if isinstance(e, dict)}
    out: list[dict[str, Any]] = []
    for v in batch:
        vid = v.get("video_id")
        ev = by_id.get(str(vid), {})
        out.append(
            {
                "video_id": vid,
                "decision": str(ev.get("decision", "keep")).lower(),
                "reason": ev.get("reason", ""),
                "confidence": str(ev.get("confidence", "medium")).lower(),
                "is_cat_behavior": bool(ev.get("is_cat_behavior", True)),
                "behavior_category": ev.get("behavior_category") or "Uncertain",
                "reject_reason": ev.get("reject_reason"),
            }
        )
    return out


def _call_batch(
    client: OpenAI,
    system_prompt: str,
    batch: list[dict],
) -> tuple[list[dict[str, Any]], int, int]:
    user_content = _build_user_message(batch)
    last_err: Exception | None = None
    for attempt in range(GPT_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=GPT_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or "{}"
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) if usage else 0
            ct = getattr(usage, "completion_tokens", 0) if usage else 0
            return _parse_evaluations(text, batch), pt, ct
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    logging.getLogger("dailymotion_pipeline").warning("GPT batch failed after retries: %s", last_err)
    return _parse_evaluations("{}", batch), 0, 0


def load_system_prompt() -> str:
    return _RULES_PATH.read_text(encoding="utf-8")


def _merge_gpt_eval(c: dict, gpt_eval: dict[str, Any], decision: str) -> dict[str, Any]:
    o = dict(c)
    o["gpt_decision"] = decision
    o["gpt_reason"] = gpt_eval.get("reason", "")
    o["gpt_confidence"] = gpt_eval.get("confidence", "medium")
    o["gpt_filter"] = gpt_eval
    return o


def run_gpt_filter(
    candidates: list[dict],
    api_key: str | None,
    logger: logging.Logger,
    run_dir: Path,
    *,
    parallel_workers: int | None = None,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """
    Text-only GPT filter (pipeline v2 style).
    Writes ``stage_3_gpt_filter/kept.jsonl`` and ``discarded.jsonl``.
    Returns (kept_rows, discarded_rows, stats).
    """
    if not candidates:
        return [], [], {"input": 0, "kept": 0, "discarded": 0, "cost": 0.0, "tokens": 0}

    if not (api_key or "").strip():
        logger.warning(
            "OPENAI_API_KEY not set — GPT filter skipped; passing all tag-filtered candidates."
        )
        kept_rows: list[dict] = []
        for c in tqdm(candidates, desc="gpt_filter (pass-through)", unit="vid"):
            ge = {
                "video_id": c.get("video_id"),
                "decision": "keep",
                "reason": "OpenAI key missing — pass-through",
                "confidence": "low",
                "is_cat_behavior": True,
                "behavior_category": "Uncertain",
                "reject_reason": None,
            }
            kept_rows.append(_merge_gpt_eval(c, ge, "keep"))
        save_jsonl(kept_rows, run_dir / "stage_3_gpt_filter" / "kept.jsonl", mode="w")
        save_jsonl([], run_dir / "stage_3_gpt_filter" / "discarded.jsonl", mode="w")
        stats = {
            "input": len(candidates),
            "kept": len(kept_rows),
            "discarded": 0,
            "cost": 0.0,
            "tokens": 0,
        }
        return kept_rows, [], stats

    system_prompt = load_system_prompt()
    client = OpenAI(api_key=api_key.strip())

    workers = parallel_workers if parallel_workers is not None else GPT_PARALLEL_WORKERS
    env_w = os.environ.get("GPT_FILTER_PARALLEL", "").strip()
    if env_w.isdigit():
        workers = int(env_w)
    workers = max(1, int(workers))

    batches = [
        candidates[i : i + GPT_BATCH_SIZE] for i in range(0, len(candidates), GPT_BATCH_SIZE)
    ]
    n_batches = len(batches)

    logger.info(
        "Stage 3 GPT filter: %d videos in %d API batches (batch_size=%d, parallel_workers=%d)",
        len(candidates),
        n_batches,
        GPT_BATCH_SIZE,
        workers,
    )

    def _one_batch(batch: list[dict]) -> tuple[list[dict], int, int]:
        return _call_batch(client, system_prompt, batch)

    kept_rows = []
    discarded: list[dict] = []
    total_tokens = 0
    total_cost = 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch_outputs = list(
            tqdm(
                pool.map(_one_batch, batches),
                total=n_batches,
                desc="gpt_filter",
                unit="batch",
                mininterval=0.5,
            )
        )

    for batch, (evals, pt, ct) in zip(batches, batch_outputs):
        total_tokens += pt + ct
        total_cost += (pt + ct) / 1000.0 * GPT_COST_PER_1K_TOKENS
        by_id = {str(e.get("video_id")): e for e in evals}

        for c in batch:
            vid = c.get("video_id")
            ev = by_id.get(str(vid), {})
            decision = str(ev.get("decision", "keep")).lower()
            gpt_eval = {
                "video_id": vid,
                "decision": decision,
                "reason": ev.get("reason", ""),
                "confidence": str(ev.get("confidence", "medium")).lower(),
                "is_cat_behavior": ev.get("is_cat_behavior", True),
                "behavior_category": ev.get("behavior_category") or "Uncertain",
                "reject_reason": ev.get("reject_reason"),
            }
            if decision == "discard":
                discarded.append(_merge_gpt_eval(c, gpt_eval, "discard"))
            else:
                kept_rows.append(_merge_gpt_eval(c, gpt_eval, "keep"))

    save_jsonl(kept_rows, run_dir / "stage_3_gpt_filter" / "kept.jsonl", mode="w")
    save_jsonl(discarded, run_dir / "stage_3_gpt_filter" / "discarded.jsonl", mode="w")

    ni, nk, nd = len(candidates), len(kept_rows), len(discarded)
    box = f"""
┌────────────────────────────────────────────────────────┐
│  GPT FILTER SUMMARY (Dailymotion metadata)             │
│  Input:     {ni:>6} candidates                           │
│  Kept:      {nk:>6} ({100.0 * nk / ni if ni else 0:.1f}%)                              │
│  Discarded: {nd:>6} ({100.0 * nd / ni if ni else 0:.1f}%)                              │
│  Estimated API cost:       ${total_cost:.4f}                      │
│  Total tokens used:        {total_tokens:>6}                        │
└────────────────────────────────────────────────────────┘"""
    logger.info(box)

    stats = {
        "input": ni,
        "kept": nk,
        "discarded": nd,
        "cost": total_cost,
        "tokens": total_tokens,
    }
    return kept_rows, discarded, stats
