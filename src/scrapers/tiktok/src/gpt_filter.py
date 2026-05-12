"""GPT metadata-only filtering — TikTok fields."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import append_jsonl, load_jsonl, project_root, resolve_path, retry_with_backoff, save_jsonl


def _safe_str(x: Any, limit: int = 400) -> str:
    if x is None:
        return ""
    try:
        s = str(x).strip()
        return s[:limit] if len(s) > limit else s
    except Exception:
        return ""


def _format_video_batch_lines(videos: list[dict]) -> str:
    """Numbered metadata lines only (user message body)."""
    lines: list[str] = []
    for i, v in enumerate(videos, 1):
        vid = _safe_str(v.get("video_id"), 80)
        title = _safe_str(v.get("title"), 200)
        desc = _safe_str(v.get("description"), 300)
        tags = v.get("tags") or []
        htags = v.get("hashtags") or []
        if not isinstance(tags, list):
            tags = []
        if not isinstance(htags, list):
            htags = []
        tags_s = [str(t) for t in tags[:10] if t is not None]
        ht_s = [str(h) for h in htags[:15] if h is not None]
        dur = v.get("duration_seconds", 0)
        ch = _safe_str(v.get("channel_title"), 120)
        ql = _safe_str(v.get("query_language"), 40)
        track = _safe_str(v.get("track"), 120)
        artist = _safe_str(v.get("artist"), 120)
        wurl = _safe_str(v.get("webpage_url"), 300)
        lines.append(
            f"{i}. video_id={vid}\n"
            f"   title={title!r}\n"
            f"   description={desc!r}\n"
            f"   tags={tags_s}\n"
            f"   hashtags={ht_s}\n"
            f"   duration_seconds={dur}\n"
            f"   channel/uploader={ch!r}\n"
            f"   sound_track={track!r}\n"
            f"   sound_artist={artist!r}\n"
            f"   query_language={ql!r}\n"
            f"   webpage_url={wurl!r}\n"
        )
    return "\n".join(lines)


def build_user_message(videos: list[dict], user_prompt_prefix: str) -> str:
    """Full user message: prefix from config file + numbered video metadata."""
    prefix = user_prompt_prefix.rstrip()
    body = _format_video_batch_lines(videos)
    return f"{prefix}\n{body}"


def call_gpt_filter(
    batch: list[dict],
    cfg: dict,
    system_prompt: str,
    user_prompt_prefix: str,
    client: Any,
    model: str,
) -> tuple[list[dict], int, int]:
    gcfg = cfg.get("gpt_filter", {})
    max_retries = int(gcfg.get("max_retries", 3))
    temperature = float(gcfg.get("temperature", 0.0))
    user_content = build_user_message(batch, user_prompt_prefix)

    def call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
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


def _outcomes_for_batch(batch: list[dict], evals: list[Any]) -> tuple[list[dict], list[dict], list[dict], int, int]:
    """Split one batch into kept / discarded / low_conf lists and confidence counts."""
    by_id = {e.get("video_id"): e for e in evals if isinstance(e, dict)}
    kept_chunk: list[dict] = []
    disc_chunk: list[dict] = []
    low_chunk: list[dict] = []
    high_conf_keeps = 0
    low_conf_keeps = 0
    for v in batch:
        vid = v.get("video_id")
        ev = by_id.get(vid, {})
        decision = str(ev.get("decision", "keep")).lower()
        if decision == "discard":
            d = dict(v)
            d["gpt_reason"] = ev.get("reason", "")
            d["gpt_confidence"] = ev.get("confidence", "medium")
            disc_chunk.append(d)
        else:
            o = dict(v)
            o["gpt_decision"] = "keep"
            o["gpt_reason"] = ev.get("reason", "")
            conf = str(ev.get("confidence", "medium")).lower()
            o["gpt_confidence"] = conf
            kept_chunk.append(o)
            if conf == "high":
                high_conf_keeps += 1
            else:
                low_conf_keeps += 1
                low_chunk.append(o)
    return kept_chunk, disc_chunk, low_chunk, high_conf_keeps, low_conf_keeps


def run_gpt_filter(
    kept_candidates: list[dict],
    cfg: dict,
    logger: logging.Logger,
    run_dir: Any,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    gcfg = cfg.get("gpt_filter", {})
    root = project_root()
    if gcfg.get("system_prompt_file"):
        system_path = resolve_path(root, gcfg["system_prompt_file"])
    elif gcfg.get("rules_file"):
        system_path = resolve_path(root, gcfg["rules_file"])
    else:
        system_path = resolve_path(root, "src/scrapers/tiktok/config/gpt_filter_system_prompt.md")
    user_prefix_path = resolve_path(
        root,
        gcfg.get("user_prompt_prefix_file", "src/scrapers/tiktok/config/gpt_filter_user_prompt_prefix.md"),
    )
    system_prompt = system_path.read_text(encoding="utf-8")
    user_prompt_prefix = user_prefix_path.read_text(encoding="utf-8")
    batch_size = int(gcfg.get("batch_size", 10))
    parallel_workers = max(1, int(gcfg.get("parallel_workers", 5)))
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

    batches: list[list[dict]] = [
        kept_candidates[i : i + batch_size] for i in range(0, len(kept_candidates), batch_size)
    ]
    n_batches = len(batches)
    workers = min(parallel_workers, n_batches) if n_batches else 1

    def _run_one(idx_batch: tuple[int, list[dict]]) -> tuple[int, list[Any], int, int, list[dict]]:
        idx, batch = idx_batch
        evals, pt, ct = call_gpt_filter(batch, cfg, system_prompt, user_prompt_prefix, client, model)
        return idx, evals, pt, ct, batch

    if n_batches == 0:
        pass
    elif workers <= 1:
        pbar = tqdm(range(n_batches), desc="GPT filter", unit="batch")
        for i in pbar:
            batch = batches[i]
            evals, pt, ct = call_gpt_filter(batch, cfg, system_prompt, user_prompt_prefix, client, model)
            total_tokens += pt + ct
            total_cost += (pt + ct) / 1000.0 * cost_per_1k
            kc, dc, lc, hi, lo = _outcomes_for_batch(batch, evals)
            kept_out.extend(kc)
            discarded_out.extend(dc)
            low_conf.extend(lc)
            high_conf_keeps += hi
            low_conf_keeps += lo
            pbar.set_postfix(kept=len(kept_out), discarded=len(discarded_out), cost=f"${total_cost:.3f}")
    else:
        logger.info("GPT filter: %d batches, %d parallel workers", n_batches, workers)
        results: list[tuple[int, list[Any], int, int, list[dict]]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, (i, batches[i])) for i in range(n_batches)]
            pbar = tqdm(as_completed(futs), total=n_batches, desc="GPT filter", unit="batch")
            for fut in pbar:
                results.append(fut.result())
                _, _, pt, ct, _ = results[-1]
                total_tokens += pt + ct
                total_cost += (pt + ct) / 1000.0 * cost_per_1k
                pbar.set_postfix(cost=f"${total_cost:.3f}")
        results.sort(key=lambda x: x[0])
        for _idx, evals, _pt, _ct, batch in results:
            kc, dc, lc, hi, lo = _outcomes_for_batch(batch, evals)
            kept_out.extend(kc)
            discarded_out.extend(dc)
            low_conf.extend(lc)
            high_conf_keeps += hi
            low_conf_keeps += lo

    save_jsonl(kept_out, run_dir / "stage_3_gpt_filter" / "kept.jsonl")
    save_jsonl(discarded_out, run_dir / "stage_3_gpt_filter" / "discarded.jsonl")
    save_jsonl(low_conf, run_dir / "stage_3_gpt_filter" / "low_confidence.jsonl")

    ni, nk, nd = len(kept_candidates), len(kept_out), len(discarded_out)
    box = f"""
┌────────────────────────────────────────────────────────┐
│  GPT FILTER SUMMARY (TikTok metadata)                  │
│  Input:     {ni:>6} candidates                           │
│  Kept:      {nk:>6} ({100.0*nk/ni if ni else 0:.1f}%)                              │
│  Discarded: {nd:>6} ({100.0*nd/ni if ni else 0:.1f}%)                              │
│  High confidence keeps:   {high_conf_keeps:>6}                        │
│  Low confidence keeps:      {low_conf_keeps:>6}                        │
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
    fig.savefig(run_dir / "stage_3_gpt_filter" / "gpt_filter_breakdown.png", dpi=150)
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
