"""GPT-4o-mini metadata-only filtering."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import append_jsonl, load_jsonl, project_root, resolve_path, retry_with_backoff, save_jsonl


def build_batch_prompt(videos: list[dict]) -> str:
    """User message for a batch of videos."""
    lines: list[str] = []
    for i, v in enumerate(videos, 1):
        vid = v.get("video_id", "")
        title = v.get("title", "")
        desc = (v.get("description") or "")[:300]
        tags = (v.get("tags") or [])[:10]
        dur = v.get("duration_seconds", 0)
        ch = v.get("channel_title", "")
        ql = v.get("query_language", "")
        lines.append(
            f"{i}. video_id={vid}\n"
            f"   title={title!r}\n"
            f"   description (truncated)={desc!r}\n"
            f"   tags={tags}\n"
            f"   duration_seconds={dur}\n"
            f"   channel_title={ch!r}\n"
            f"   query_language={ql!r}\n"
        )
    return (
        "Evaluate each video for the dataset. "
        'Return a JSON object with a single key "evaluations" whose value is an array of objects '
        'with video_id, decision (keep|discard), reason, confidence.\n\n' + "\n".join(lines)
    )


def call_gpt_filter(
    batch: list[dict],
    cfg: dict,
    rules_text: str,
    client: Any,
    model: str,
) -> tuple[list[dict], int, int]:
    """
    Call OpenAI. Returns (evaluations, prompt_tokens, completion_tokens).
    On malformed JSON: default all to keep with confidence=low.
    """
    gcfg = cfg.get("gpt_filter", {})
    max_retries = int(gcfg.get("max_retries", 3))
    temperature = float(gcfg.get("temperature", 0.0))
    user_content = build_batch_prompt(batch)

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": rules_text},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    try:
        resp = retry_with_backoff(call, max_retries=max_retries)
    except Exception:
        return (
            [
                {
                    "video_id": v.get("video_id"),
                    "decision": "keep",
                    "reason": "API failure default",
                    "confidence": "low",
                }
                for v in batch
            ],
            0,
            0,
        )

    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    text = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # try extract array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None
    if data is None:
        return (
            [
                {
                    "video_id": v.get("video_id"),
                    "decision": "keep",
                    "reason": "malformed JSON default",
                    "confidence": "low",
                }
                for v in batch
            ],
            pt,
            ct,
        )

    if isinstance(data, dict):
        if "evaluations" in data:
            arr = data["evaluations"]
        elif "videos" in data:
            arr = data["videos"]
        else:
            # single wrapper
            arr = list(data.values())[0] if len(data) == 1 and isinstance(list(data.values())[0], list) else []
    elif isinstance(data, list):
        arr = data
    else:
        arr = []

    if not arr:
        return (
            [
                {
                    "video_id": v.get("video_id"),
                    "decision": "keep",
                    "reason": "empty response default",
                    "confidence": "low",
                }
                for v in batch
            ],
            pt,
            ct,
        )

    return arr, pt, ct


def run_gpt_filter(
    kept_candidates: list[dict],
    cfg: dict,
    logger: logging.Logger,
    run_dir: Any,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    gcfg = cfg.get("gpt_filter", {})
    rules_path = resolve_path(project_root(), gcfg.get("rules_file", "src/scrapers/data_pipeline_v2/config/gpt_filter_rules.md"))
    rules_text = rules_path.read_text(encoding="utf-8")
    batch_size = int(gcfg.get("batch_size", 10))
    workers = max(1, int(gcfg.get("gpt_filter_workers", 1)))
    model = gcfg.get("model", "gpt-4o-mini")
    cost_per_1k = float(gcfg.get("cost_estimate_per_1k_tokens", 0.00015))

    api_key = (gcfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — GPT filter skipped; passing all tag-filtered candidates.")
        kept_out = [dict(x) for x in kept_candidates]
        for x in kept_out:
            x.setdefault("gpt_decision", "keep")
            x.setdefault("gpt_confidence", "low")
            x.setdefault("gpt_reason", "OpenAI key missing — pass-through")
        save_jsonl(kept_out, run_dir / "stage_3_gpt_filter" / "kept.jsonl")
        save_jsonl([], run_dir / "stage_3_gpt_filter" / "discarded.jsonl")
        save_jsonl(kept_out, run_dir / "stage_3_gpt_filter" / "low_confidence.jsonl")
        return kept_out, [], {"input": len(kept_candidates), "kept": len(kept_out), "discarded": 0, "cost": 0.0, "tokens": 0}

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    kept_out: list[dict] = []
    discarded_out: list[dict] = []
    low_conf: list[dict] = []
    total_tokens = 0
    total_cost = 0.0
    high_conf_keeps = 0
    low_conf_keeps = 0

    batch_starts = list(range(0, len(kept_candidates), batch_size))

    def run_one_batch(start: int) -> tuple[list[dict], list[dict], int, int, int, int]:
        """Returns (kept_chunk, discarded_chunk, low_conf_chunk, pt, ct, high_k, low_k)."""
        batch = kept_candidates[start : start + batch_size]
        evals, pt, ct = call_gpt_filter(batch, cfg, rules_text, client, model)
        kc: list[dict] = []
        dc: list[dict] = []
        lc: list[dict] = []
        hk = 0
        lk = 0
        by_id = {e.get("video_id"): e for e in evals if isinstance(e, dict)}
        for v in batch:
            vid = v.get("video_id")
            ev = by_id.get(vid, {})
            decision = str(ev.get("decision", "keep")).lower()
            if decision == "discard":
                d = dict(v)
                d["gpt_reason"] = ev.get("reason", "")
                d["gpt_confidence"] = ev.get("confidence", "medium")
                dc.append(d)
            else:
                o = dict(v)
                o["gpt_decision"] = "keep"
                o["gpt_reason"] = ev.get("reason", "")
                conf = str(ev.get("confidence", "medium")).lower()
                o["gpt_confidence"] = conf
                kc.append(o)
                if conf == "high":
                    hk += 1
                else:
                    lk += 1
                    lc.append(o)
        return kc, dc, lc, pt, ct, hk, lk

    if workers <= 1:
        batch_results = [run_one_batch(s) for s in tqdm(batch_starts, desc="GPT filter", unit="batch")]
    else:
        logger.info("GPT filter: %d parallel worker(s) for API batches", workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(
                tqdm(
                    pool.map(run_one_batch, batch_starts),
                    total=len(batch_starts),
                    desc="GPT filter",
                    unit="batch",
                )
            )

    for kc, dc, lc, pt, ct, hk, lk in batch_results:
        kept_out.extend(kc)
        discarded_out.extend(dc)
        low_conf.extend(lc)
        total_tokens += pt + ct
        total_cost += (pt + ct) / 1000.0 * cost_per_1k
        high_conf_keeps += hk
        low_conf_keeps += lk

    save_jsonl(kept_out, run_dir / "stage_3_gpt_filter" / "kept.jsonl")
    save_jsonl(discarded_out, run_dir / "stage_3_gpt_filter" / "discarded.jsonl")
    save_jsonl(low_conf, run_dir / "stage_3_gpt_filter" / "low_confidence.jsonl")

    ni, nk, nd = len(kept_candidates), len(kept_out), len(discarded_out)
    box = f"""
┌────────────────────────────────────────────────────────┐
│  GPT FILTER SUMMARY (metadata-only filtering)          │
│  Input:     {ni:>6} candidates                           │
│  Kept:      {nk:>6} ({100.0*nk/ni if ni else 0:.1f}%)                              │
│  Discarded: {nd:>6} ({100.0*nd/ni if ni else 0:.1f}%)                              │
│  High confidence keeps:   {high_conf_keeps:>6}                        │
│  Low confidence keeps:      {low_conf_keeps:>6} (review recommended)   │
│  Estimated API cost:       ${total_cost:.3f}                      │
│  Total tokens used:        {total_tokens:>6}                        │
└────────────────────────────────────────────────────────┘"""
    print(box)
    logger.info(box)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        [nk, nd] if nk + nd else [1],
        labels=["keep", "discard"] if nk + nd else ["ok"],
        autopct="%1.1f%%",
    )
    ax.set_title("GPT filter (keep vs discard)")
    gpt_dir = run_dir / "stage_3_gpt_filter"
    gpt_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(gpt_dir / "gpt_filter_breakdown.png", dpi=150)
    plt.close(fig)

    stats = {
        "input": ni,
        "kept": nk,
        "discarded": nd,
        "high_conf_keeps": high_conf_keeps,
        "low_conf_keeps": low_conf_keeps,
        "cost": total_cost,
        "tokens": total_tokens,
    }
    return kept_out, discarded_out, stats
