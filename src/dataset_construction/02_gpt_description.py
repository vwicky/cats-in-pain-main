#!/usr/bin/env python3
"""
GPT-4o vision descriptions for snippet frames. Run from REPO_ROOT.

  python dataset_construction/02_gpt_description.py
  python dataset_construction/02_gpt_description.py --dry-run
  python dataset_construction/02_gpt_description.py --limit 100
  python dataset_construction/02_gpt_description.py --rebuild-cache
  python dataset_construction/02_gpt_description.py --workers 5
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from openai import OpenAI
from openai import APIError, APIStatusError, RateLimitError
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm")
SNIPPET_PATH_KEYS = (
    "path",
    "video_path",
    "file",
    "snippet_path",
    "video_file",
    "local_path",
)

GPT_TOP_LEVEL_KEYS = frozenset(
    {"video_quality", "cats", "environment", "behavior", "dataset_flags"}
)
RESULT_META_KEYS = frozenset(
    {
        "snippet_id",
        "model_used",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_sec",
        "parse_error",
        "raw_response",
        "error",
        "api_error",
    }
)


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _candidate_paths_from_snippet(snippet: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in SNIPPET_PATH_KEYS:
        v = snippet.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for k, v in snippet.items():
        if k in SNIPPET_PATH_KEYS or k == "id":
            continue
        if isinstance(v, str) and ("/" in v or "\\" in v):
            low = v.lower()
            if any(low.endswith(ext) for ext in VIDEO_EXTS):
                out.append(v.strip())
    return out


def resolve_snippet_video(
    repo_root: Path,
    snippet: dict[str, Any],
    snippets_dirs: list[Path],
) -> str | None:
    sid = snippet.get("id")
    if not isinstance(sid, str) or not sid:
        return None

    for raw in _candidate_paths_from_snippet(snippet):
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
        q = repo_root / raw
        if q.is_file():
            return str(q.resolve())

    for base in snippets_dirs:
        for ext in VIDEO_EXTS:
            cand = base / f"{sid}{ext}"
            if cand.is_file():
                return str(cand.resolve())
    return None


def infer_platform(record: dict[str, Any], resolved_path: str | None) -> str:
    src = (record.get("source") or "").strip().lower()
    if src == "tiktok_metadata":
        return "TikTok"
    if src == "metadata_v2":
        pass
    for key in ("platform", "source_platform"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            pl = v.strip().lower()
            if "tiktok" in pl:
                return "TikTok"
            if "dailymotion" in pl or "daily_motion" in pl:
                return "DailyMotion"
            if "youtube" in pl:
                return "YouTube"

    path = (resolved_path or "").lower()
    if "tiktok_snippets" in path or "/tiktok/" in path:
        return "TikTok"
    if "dailymotion" in path:
        return "DailyMotion"

    vid = str(record.get("video_id", ""))
    if vid.isdigit() and len(vid) >= 15:
        return "TikTok"

    return "YouTube"


# Snippet clips are short; cap avoids runaway reads on corrupt streams.
# If a file exceeds this, fractional positions apply to the decoded prefix only.
_MAX_FRAMES_SEQUENTIAL = 8000


def _count_frames_sequential(video_path: str, max_frames: int) -> int:
    """Decode sequentially (no seeking) until EOF or max_frames. Returns frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = 0
    try:
        while n < max_frames:
            ret, _ = cap.read()
            if not ret:
                break
            n += 1
    finally:
        cap.release()
    return n


def _read_bgr_at_frame_indices(
    video_path: str,
    frame_indices: list[int],
) -> list[np.ndarray] | None:
    """
    Second pass: sequential read only, stopping once all unique indices are collected.
    ``frame_indices`` may contain duplicates; order in the output matches the input.
    """
    if not frame_indices:
        return []
    need_sorted = sorted(set(frame_indices))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    saved: dict[int, np.ndarray] = {}
    try:
        fi = 0
        need_ptr = 0
        while need_ptr < len(need_sorted):
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            if fi == need_sorted[need_ptr]:
                saved[need_sorted[need_ptr]] = frame.copy()
                need_ptr += 1
            fi += 1
        return [saved[i] for i in frame_indices]
    finally:
        cap.release()


def _resize_rgb_longest_side(rgb: np.ndarray, max_size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest > max_size and longest > 0:
        scale = max_size / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return rgb


def extract_three_frames(
    video_path: str,
    positions: list[float],
    max_size: int,
) -> list[np.ndarray] | None:
    """
    Extract frames at fractional positions [0..1] of the **actually decodable**
    frame count (sequential decode — no index seeking).

    OpenCV often misreports ``CAP_PROP_FRAME_COUNT`` and seeking with
    ``CAP_PROP_POS_FRAMES`` fails on many H.264/MP4 snippets; this matches the
    approach in ``01_static_detection.py`` (decode in order, then subsample).

    Convert BGR→RGB. Resize longest side to max_size. Return list of RGB arrays,
    or None if the video cannot be read.
    Never raises — catches all exceptions.
    """
    try:
        n = _count_frames_sequential(video_path, _MAX_FRAMES_SEQUENTIAL)
        if n <= 0:
            return None
        idx_list: list[int] = []
        for pos in positions:
            if n == 1:
                idx_list.append(0)
            else:
                idx = int(round(float(pos) * (n - 1)))
                idx_list.append(max(0, min(idx, n - 1)))
        bgr_list = _read_bgr_at_frame_indices(video_path, idx_list)
        if bgr_list is None or len(bgr_list) != len(positions):
            return None
        frames_out: list[np.ndarray] = []
        for frame in bgr_list:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_out.append(_resize_rgb_longest_side(rgb, max_size))
        return frames_out
    except Exception:
        return None


def encode_frame_base64(frame: np.ndarray, quality: int = 85) -> str:
    """
    Encode RGB numpy array to base64 JPEG string.
    Compress to quality=85 to reduce token cost.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok or buf is None:
        raise ValueError("imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _parse_gpt_json(content: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = _strip_code_fences(content)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
        return None, "JSON root is not an object"
    except json.JSONDecodeError as e:
        return None, str(e)


class _MinuteRateLimiter:
    """Thread-safe global RPM cap (shared across parallel workers)."""

    def __init__(self, requests_per_minute: int) -> None:
        self.rpm = max(1, int(requests_per_minute))
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                while self._times and self._times[0] < cutoff:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(time.monotonic())
                    return
                sleep_s = self._times[0] + 60.0 - now + 0.05
            time.sleep(max(sleep_s, 0.0))

    def current_rpm(self) -> int:
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            return sum(1 for t in self._times if t >= cutoff)


def describe_snippet(
    snippet_id: str,
    frames_b64: list[str],
    system_prompt: str,
    cfg: dict[str, Any],
    client: OpenAI,
    rate_limiter: _MinuteRateLimiter,
) -> dict[str, Any]:
    """
    Call GPT-4o vision; parse JSON. On parse failure: return error dict with
    raw_response, parse_error=True, snippet_id.

    Return dict with parsed fields plus:
    snippet_id, model_used, input_tokens, output_tokens,
    cost_usd, latency_sec, parse_error, raw_response
    """
    model = str(cfg["model"])
    image_quality = str(cfg.get("image_quality") or "auto")
    max_retries = int(cfg.get("max_retries", 3))
    base_delay = float(cfg.get("retry_base_delay", 2.0))
    cost_in = float(cfg.get("cost_per_1k_input_tokens", 0.0))
    cost_out = float(cfg.get("cost_per_1k_output_tokens", 0.0))

    user_text = (
        "These are three frames from a cat behavior video clip "
        "(beginning, middle, end). "
        f"Snippet ID: {snippet_id}. Analyze and respond with JSON."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64 in frames_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": image_quality,
                },
            }
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    last_err: str | None = None
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            rate_limiter.acquire()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.0,
            )
            latency_sec = time.perf_counter() - t0
            msg = resp.choices[0].message
            raw_content = (msg.content or "").strip()
            usage = resp.usage
            in_tok = int(usage.prompt_tokens or 0) if usage else 0
            out_tok = int(usage.completion_tokens or 0) if usage else 0
            cost_usd = (in_tok / 1000.0) * cost_in + (out_tok / 1000.0) * cost_out

            parsed, perr = _parse_gpt_json(raw_content)
            if parsed is None:
                return {
                    "snippet_id": snippet_id,
                    "model_used": model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost_usd,
                    "latency_sec": latency_sec,
                    "parse_error": True,
                    "raw_response": raw_content,
                    "parse_message": perr,
                }

            out: dict[str, Any] = {
                "snippet_id": snippet_id,
                "model_used": model,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": cost_usd,
                "latency_sec": latency_sec,
                "parse_error": False,
                "raw_response": raw_content,
            }
            for k in GPT_TOP_LEVEL_KEYS:
                if k in parsed:
                    out[k] = parsed[k]
            return out
        except RateLimitError as e:
            last_err = str(e)
            if attempt >= max_retries:
                break
            delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
            continue
        except APIStatusError as e:
            last_err = str(e)
            code = getattr(e, "status_code", None)
            if code in (429, 500, 502, 503) and attempt < max_retries:
                delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            break
        except APIError as e:
            last_err = str(e)
            break
        except Exception as e:
            last_err = str(e)
            break

    latency_sec = time.perf_counter() - t0
    return {
        "snippet_id": snippet_id,
        "model_used": model,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_sec": latency_sec,
        "parse_error": True,
        "raw_response": last_err or "",
        "api_error": True,
        "error": last_err or "api_error",
    }


def _gpt_payload_for_manifest(result: dict[str, Any]) -> dict[str, Any]:
    """Parsed GPT JSON only (no token/cost metadata), plus error stubs if needed."""
    if result.get("parse_error"):
        out = {
            k: v
            for k, v in result.items()
            if k not in RESULT_META_KEYS and k != "parse_message"
        }
        stub: dict[str, Any] = {
            "parse_error": True,
        }
        if result.get("raw_response"):
            stub["raw_response"] = result["raw_response"]
        if result.get("error"):
            stub["error"] = result["error"]
        if result.get("parse_message"):
            stub["parse_message"] = result["parse_message"]
        return {**stub, **out}
    return {k: result[k] for k in GPT_TOP_LEVEL_KEYS if k in result}


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("snippet_id")
            if isinstance(sid, str) and sid:
                out[sid] = obj
    return out


def _append_cache_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _plot_exclusion_reasons(results: list[dict[str, Any]], out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    reasons: list[str] = []
    for r in results:
        if r.get("parse_error"):
            continue
        df = r.get("dataset_flags") or {}
        if not isinstance(df, dict):
            continue
        if df.get("suitable_for_training") is True:
            continue
        er = df.get("exclusion_reason")
        if er is None or er == "null":
            reasons.append("(none)")
        else:
            reasons.append(str(er))
    if not reasons:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no exclusions", ha="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    c = Counter(reasons)
    labels = list(c.keys())
    vals = [c[k] for k in labels]
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.35)))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="#c44e52")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("Exclusion reasons (unsuitable snippets)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_location_by_category(
    tasks: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    rows: list[tuple[str, str]] = []
    for t in tasks:
        sid = t["snippet_id"]
        r = result_by_id.get(sid)
        if not r or r.get("parse_error"):
            continue
        cat = str(t.get("behavioral_category") or "unknown")
        env = r.get("environment") or {}
        loc = "unknown"
        if isinstance(env, dict):
            loc = str(env.get("location_type") or "unknown")
        rows.append((cat, loc))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no data", ha="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    df = pd.DataFrame(rows, columns=["behavioral_category", "location_type"])
    pivot = pd.crosstab(df["behavioral_category"], df["location_type"])
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[0] * 0.45), 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.legend(title="location_type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylabel("count")
    ax.set_xlabel("behavioral_category")
    ax.set_title("location_type by behavioral_category")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_breed_distribution(results: list[dict[str, Any]], out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    breeds: list[str] = []
    for r in results:
        if r.get("parse_error"):
            continue
        cats = r.get("cats") or {}
        if not isinstance(cats, dict):
            continue
        pc = cats.get("primary_cat") or {}
        if isinstance(pc, dict):
            bg = pc.get("breed_guess")
            if bg is not None and str(bg).strip().lower() not in ("", "null", "none"):
                breeds.append(str(bg).strip())
    if not breeds:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no breed data", ha="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    c = Counter(breeds)
    top = c.most_common(15)
    labels = [x[0] for x in top]
    vals = [x[1] for x in top]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), vals, color="#4682b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Top 15 breed_guess values")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_suitability_by_platform(
    tasks: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    platforms = ["YouTube", "TikTok", "DailyMotion"]
    ok_c = {p: 0 for p in platforms}
    bad_c = {p: 0 for p in platforms}
    for t in tasks:
        pl = t.get("platform") or "YouTube"
        if pl not in ok_c:
            continue
        r = result_by_id.get(t["snippet_id"])
        if not r or r.get("parse_error"):
            bad_c[pl] += 1
            continue
        df = r.get("dataset_flags") or {}
        suit = True
        if isinstance(df, dict):
            suit = df.get("suitable_for_training") is True
        if suit:
            ok_c[pl] += 1
        else:
            bad_c[pl] += 1
    x = np.arange(len(platforms))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [ok_c[p] for p in platforms], w, label="suitable", color="#2ca02c")
    ax.bar(x + w / 2, [bad_c[p] for p in platforms], w, label="unsuitable", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(["YouTube", "TikTok", "DM"])
    ax.set_ylabel("snippet count")
    ax.legend()
    ax.set_title("Suitable vs unsuitable by platform")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _safe_get_flags(r: dict[str, Any]) -> dict[str, Any]:
    df = r.get("dataset_flags")
    return df if isinstance(df, dict) else {}


def _safe_get_vq(r: dict[str, Any]) -> dict[str, Any]:
    vq = r.get("video_quality")
    return vq if isinstance(vq, dict) else {}


def _safe_get_cats(r: dict[str, Any]) -> dict[str, Any]:
    c = r.get("cats")
    return c if isinstance(c, dict) else {}


def _safe_get_env(r: dict[str, Any]) -> dict[str, Any]:
    e = r.get("environment")
    return e if isinstance(e, dict) else {}


def _safe_get_behavior(r: dict[str, Any]) -> dict[str, Any]:
    b = r.get("behavior")
    return b if isinstance(b, dict) else {}


def _write_report(
    path: Path,
    model: str,
    date_s: str,
    tasks: list[dict[str, Any]],
    result_by_id: dict[str, dict[str, Any]],
    total_in_tok: int,
    total_out_tok: int,
    total_cost: float,
) -> None:
    n = len(tasks)
    parse_err = 0
    for t in tasks:
        r = result_by_id.get(t["snippet_id"])
        if r and r.get("parse_error"):
            parse_err += 1

    suitable_n = 0
    ai_gen = 0
    no_cat = 0
    sev_occ = 0
    multi_only = 0

    breed_counter: Counter[str] = Counter()
    color_counter: Counter[str] = Counter()
    age_counts = Counter({"kitten": 0, "juvenile": 0, "adult": 0, "senior": 0, "unknown": 0})
    setting_counts = Counter()
    loc_type_counts = Counter()
    behavior_counts = Counter()
    pain_vis = 0
    face_vis = 0

    removed_breakdown = Counter(
        {
            "ai_generated": 0,
            "no_cat_visible": 0,
            "severe_occlusion": 0,
            "severe_blur": 0,
            "other": 0,
        }
    )
    _known_exclusion = frozenset(
        {"ai_generated", "no_cat_visible", "severe_occlusion", "severe_blur"}
    )

    pv_true = pv_false = 0
    pv_total = 0

    for t in tasks:
        r = result_by_id.get(t["snippet_id"])
        if not r or r.get("parse_error"):
            continue
        df = _safe_get_flags(r)
        vq = _safe_get_vq(r)
        cats = _safe_get_cats(r)
        env = _safe_get_env(r)
        beh = _safe_get_behavior(r)

        if df.get("suitable_for_training") is True:
            suitable_n += 1
        if vq.get("is_ai_generated") is True:
            ai_gen += 1
        ncv = cats.get("n_cats_visible")
        ncv_zero = False
        try:
            ncv_zero = int(ncv) == 0
        except (TypeError, ValueError):
            ncv_zero = False
        if ncv_zero or df.get("exclusion_reason") == "no_cat_visible":
            no_cat += 1
        if vq.get("occlusion") == "severe":
            sev_occ += 1
        if df.get("exclusion_reason") == "multiple_cats_only":
            multi_only += 1

        pc = cats.get("primary_cat")
        if isinstance(pc, dict):
            bg = pc.get("breed_guess")
            if bg and str(bg).lower() not in ("null", "none", ""):
                breed_counter[str(bg).strip()] += 1
            coats = pc.get("coat_color")
            if isinstance(coats, list):
                for c in coats:
                    color_counter[str(c)] += 1
            ag = str(pc.get("age_guess") or "unknown").lower()
            if ag in age_counts:
                age_counts[ag] += 1
            else:
                age_counts["unknown"] += 1

        st = str(env.get("setting") or "unclear").lower()
        setting_counts[st] += 1
        lt = str(env.get("location_type") or "other")
        loc_type_counts[lt] += 1

        pb = " ".join(str(beh.get("primary_behavior") or "unknown").split())
        behavior_counts[pb] += 1

        if df.get("pain_indicators_visible") is True:
            pain_vis += 1
        if beh.get("face_clearly_visible") is True:
            face_vis += 1

        if df.get("suitable_for_training") is False:
            er = df.get("exclusion_reason")
            es = str(er).strip() if er is not None else ""
            if es in _known_exclusion:
                removed_breakdown[es] += 1
            else:
                removed_breakdown["other"] += 1

        bc = str(t.get("behavioral_category") or "")
        if bc == "Pain/Vet":
            pv_total += 1
            if df.get("vet_clinic_confirmed") is True:
                pv_true += 1
            elif df.get("vet_clinic_confirmed") is False:
                pv_false += 1

    after = suitable_n
    removed = n - after
    pct = lambda a, b: (100.0 * a / b) if b else 0.0

    lines: list[str] = []
    lines.append("════════════════════════════════════════════════════════")
    lines.append("GPT DESCRIPTION REPORT")
    lines.append(f"Model: {model} | Snippets: {n} | Date: {date_s}")
    lines.append("════════════════════════════════════════════════════════")
    lines.append("")
    lines.append("COST SUMMARY")
    lines.append("────────────")
    lines.append(f"Total input tokens:   {total_in_tok}")
    lines.append(f"Total output tokens:  {total_out_tok}")
    lines.append(f"Total cost:           ${total_cost:.2f}")
    mean_c = total_cost / n if n else 0.0
    lines.append(f"Mean cost/snippet:    ${mean_c:.4f}")
    lines.append(f"Parse errors:         {parse_err} ({pct(parse_err, n):.1f}%)")
    lines.append("")
    lines.append("QUALITY FLAGS")
    lines.append("─────────────")
    lines.append(f"Suitable for training:    {suitable_n} ({pct(suitable_n, n):.1f}%)")
    lines.append(f"AI generated (flagged):   {ai_gen} ({pct(ai_gen, n):.1f}%)")
    lines.append(f"No cat visible:           {no_cat} ({pct(no_cat, n):.1f}%)")
    lines.append(f"Severe occlusion:         {sev_occ} ({pct(sev_occ, n):.1f}%)")
    lines.append(f"Multiple cats only:       {multi_only} ({pct(multi_only, n):.1f}%)")
    lines.append("")
    lines.append("CAT CHARACTERISTICS")
    lines.append("───────────────────")
    lines.append("Top breeds detected:")
    for breed, cnt in breed_counter.most_common(20):
        lines.append(f"  [{breed}]: {cnt} snippets")
    lines.append("")
    col_parts = [f"{k}: {v}" for k, v in color_counter.most_common(30)]
    lines.append(f"Coat colors: {', '.join(col_parts) if col_parts else '(none)'}")
    tot_age = sum(age_counts[k] for k in ("kitten", "juvenile", "adult", "senior", "unknown"))
    lines.append(
        "Age distribution: "
        f"kitten {pct(age_counts['kitten'], tot_age):.1f}% | "
        f"juvenile {pct(age_counts['juvenile'], tot_age):.1f}% | "
        f"adult {pct(age_counts['adult'], tot_age):.1f}% | "
        f"senior {pct(age_counts['senior'], tot_age):.1f}%"
    )
    lines.append("")
    lines.append("ENVIRONMENT")
    lines.append("───────────")
    sc = sum(setting_counts.get(k, 0) for k in ("indoor", "outdoor", "mixed"))
    lines.append(
        f"Indoor: {pct(setting_counts.get('indoor', 0), sc):.1f}% | "
        f"Outdoor: {pct(setting_counts.get('outdoor', 0), sc):.1f}% | "
        f"Mixed: {pct(setting_counts.get('mixed', 0), sc):.1f}%"
    )
    lines.append("")
    lines.append("Location types:")
    lt_total = sum(loc_type_counts.values()) or 1
    core_labels = ("home", "vet_clinic", "shelter", "street")
    for label in core_labels:
        cnt = loc_type_counts.get(label, 0)
        lines.append(f"  {label + ':':<12} {cnt} ({pct(cnt, lt_total):.1f}%)")
    other_lt = sum(
        loc_type_counts.get(k, 0) for k in loc_type_counts if k not in core_labels
    )
    lines.append(f"  {'other:':<12} {other_lt} ({pct(other_lt, lt_total):.1f}%)")
    lines.append("")
    lines.append("Vet clinic confirmed (for Pain/Vet category check):")
    pvt = pv_total or 1
    lines.append(
        "  behavioral_category=Pain/Vet AND vet_clinic_confirmed=true: "
        f"{pv_true} ({pct(pv_true, pvt):.1f}%)"
    )
    lines.append(
        "  behavioral_category=Pain/Vet AND vet_clinic_confirmed=false: "
        f"{pv_false} ({pct(pv_false, pvt):.1f}%)"
    )
    lines.append("")
    lines.append("BEHAVIOR")
    lines.append("────────")
    lines.append("Primary behavior distribution:")
    btot = sum(behavior_counts.values()) or 1
    for beh, cnt in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {beh}: {cnt} ({pct(cnt, btot):.1f}%)")
    lines.append("")
    lines.append(f"Pain indicators visible: {pain_vis} snippets ({pct(pain_vis, n):.1f}%)")
    lines.append(f"Face clearly visible:    {face_vis} snippets ({pct(face_vis, n):.1f}%)")
    lines.append("")
    lines.append("DATASET COMPOSITION AFTER GPT FILTERING")
    lines.append("─────────────────────────────────────────")
    lines.append(f"Before: {n} snippets")
    lines.append(f"After removing unsuitable: {after} snippets ({removed} removed)")
    lines.append("")
    lines.append("Removed breakdown:")
    lines.append(f"  ai_generated:       {removed_breakdown['ai_generated']}")
    lines.append(f"  no_cat_visible:     {removed_breakdown['no_cat_visible']}")
    lines.append(f"  severe_occlusion:   {removed_breakdown['severe_occlusion']}")
    lines.append(f"  severe_blur:        {removed_breakdown['severe_blur']}")
    lines.append(f"  other:              {removed_breakdown['other']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _process_snippet_row(
    t: dict[str, Any],
    positions: list[float],
    max_size: int,
    system_prompt: str,
    gd: dict[str, Any],
    client: OpenAI,
    rate_limiter: _MinuteRateLimiter,
) -> dict[str, Any]:
    """Extract frames, encode, and call GPT for one snippet (runs in a worker thread)."""
    sid = t["snippet_id"]
    vpath = t["video_path"]
    try:
        if vpath is None:
            return {
                "snippet_id": sid,
                "model_used": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "latency_sec": 0.0,
                "parse_error": True,
                "raw_response": "",
                "error": "missing_video_file",
            }
        frames = extract_three_frames(vpath, positions, max_size)
        if frames is None:
            return {
                "snippet_id": sid,
                "model_used": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "latency_sec": 0.0,
                "parse_error": True,
                "raw_response": "",
                "error": "frame_extraction_failed",
            }
        try:
            b64s = [encode_frame_base64(fr) for fr in frames]
        except Exception as e:
            return {
                "snippet_id": sid,
                "model_used": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "latency_sec": 0.0,
                "parse_error": True,
                "raw_response": str(e),
                "error": "encode_failed",
            }
        return describe_snippet(sid, b64s, system_prompt, gd, client, rate_limiter)
    except Exception as e:
        return {
            "snippet_id": sid,
            "model_used": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "latency_sec": 0.0,
            "parse_error": True,
            "raw_response": repr(e),
            "error": "worker_exception",
        }


def main() -> int:
    _load_dotenv(REPO_ROOT)
    parser = argparse.ArgumentParser(description="GPT-4o frame descriptions for snippets.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve tasks only; no API I/O.")
    parser.add_argument("--limit", type=int, default=None, help="Max snippets to process (after skip).")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete cache file and re-describe all snippets.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel thread workers (default: gpt_description.workers in config).",
    )
    args = parser.parse_args()

    cfg_all = load_config()
    sources = cfg_all["sources"]
    gd = cfg_all["gpt_description"]
    n_workers = max(1, args.workers if args.workers is not None else int(gd.get("workers", 5)))

    repo_root = REPO_ROOT
    meta_path = repo_root / gd["input_manifest"]
    out_manifest = repo_root / gd["output_manifest"]
    descriptions_path = repo_root / gd["descriptions_jsonl"]
    report_path = repo_root / gd["report_txt"]
    cache_path = repo_root / gd["cache_file"]
    prompt_path = repo_root / gd["prompt_file"]
    reports_dir = repo_root / cfg_all["output"]["reports_dir"]

    snippets_dirs = [repo_root / p for p in sources["snippets_dirs"]]

    if args.rebuild_cache and cache_path.is_file():
        cache_path.unlink()

    system_prompt = prompt_path.read_text(encoding="utf-8")

    positions = [float(x) for x in gd["frame_positions"]]
    max_size = int(gd["image_size"])
    skip_cached = bool(gd.get("skip_already_described", True)) and not args.rebuild_cache

    if not meta_path.is_file():
        print(f"Missing input manifest: {meta_path}", file=sys.stderr)
        return 1

    records_in = _load_jsonl_records(meta_path)

    cache: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        cache = _load_cache(cache_path)

    tasks: list[dict[str, Any]] = []
    for rec in records_in:
        behavioral_category = rec.get("behavioral_category")
        for sn in rec.get("snippets") or []:
            if not isinstance(sn, dict):
                continue
            sid = sn.get("id")
            if not isinstance(sid, str):
                continue
            resolved = resolve_snippet_video(repo_root, sn, snippets_dirs)
            platform = infer_platform(rec, resolved)
            tasks.append(
                {
                    "snippet_id": sid,
                    "platform": platform,
                    "behavioral_category": behavioral_category,
                    "video_path": resolved,
                }
            )

    n_snippets = len(tasks)
    skip_ids = set(cache.keys()) if skip_cached else set()
    n_skip = len(skip_ids) if skip_cached else 0
    to_process = [t for t in tasks if t["snippet_id"] not in skip_ids]
    if args.limit is not None:
        to_process = to_process[: max(0, int(args.limit))]

    n_todo = len(to_process)
    print(
        f"Loaded {n_snippets} snippets | {n_skip} already described | {n_todo} to process "
        f"| workers={n_workers}"
    )

    if args.dry_run:
        n_miss = sum(1 for t in tasks if t["video_path"] is None)
        print(
            f"Dry run: would process {n_todo} snippets ({n_miss} missing video paths total) "
            f"| workers={n_workers}"
        )
        return 0

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("OPENAI_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    rate_limiter = _MinuteRateLimiter(int(gd["requests_per_minute"]))

    result_by_id: dict[str, dict[str, Any]] = dict(cache)
    pending_cache: list[dict[str, Any]] = []
    total_in_tok = 0
    total_out_tok = 0
    total_cost = 0.0
    n_err = 0
    processed_since_flush = 0

    def flush_cache() -> None:
        nonlocal pending_cache, processed_since_flush
        if pending_cache:
            _append_cache_lines(cache_path, pending_cache)
            pending_cache = []
        processed_since_flush = 0

    ex = ThreadPoolExecutor(max_workers=n_workers)
    interrupted = False
    try:
        future_to_task = {
            ex.submit(
                _process_snippet_row,
                t,
                positions,
                max_size,
                system_prompt,
                gd,
                client,
                rate_limiter,
            ): t
            for t in to_process
        }
        pbar = tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="GPT Description",
            unit="snippet",
        )
        for fut in pbar:
            t = future_to_task[fut]
            sid = t["snippet_id"]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "snippet_id": sid,
                    "model_used": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_sec": 0.0,
                    "parse_error": True,
                    "raw_response": repr(e),
                    "error": "future_exception",
                }

            result_by_id[sid] = res
            total_in_tok += int(res.get("input_tokens") or 0)
            total_out_tok += int(res.get("output_tokens") or 0)
            total_cost += float(res.get("cost_usd") or 0.0)
            if res.get("parse_error"):
                n_err += 1

            pending_cache.append(res)
            processed_since_flush += 1
            if processed_since_flush >= 50:
                flush_cache()
                tqdm.write(
                    f"[cache flush] cost=${total_cost:.2f} | errors={n_err} | rpm={rate_limiter.current_rpm()}"
                )

            rpm = rate_limiter.current_rpm()
            pbar.set_postfix(
                cost=f"${total_cost:.2f}",
                errors=n_err,
                rpm=rpm,
                refresh=True,
            )
        flush_cache()
    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("\nInterrupted — saving cache...", flush=True)
        flush_cache()
        tqdm.write("Cache saved.", flush=True)
    finally:
        if sys.version_info >= (3, 9):
            ex.shutdown(wait=not interrupted, cancel_futures=interrupted)
        else:
            ex.shutdown(wait=not interrupted)

    if interrupted:
        return 130

    total_in_tok = sum(
        int((result_by_id.get(t["snippet_id"]) or {}).get("input_tokens") or 0)
        for t in tasks
    )
    total_out_tok = sum(
        int((result_by_id.get(t["snippet_id"]) or {}).get("output_tokens") or 0)
        for t in tasks
    )
    total_cost = sum(
        float((result_by_id.get(t["snippet_id"]) or {}).get("cost_usd") or 0.0)
        for t in tasks
    )

    # Full descriptions jsonl (ordered)
    descriptions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(descriptions_path, "w", encoding="utf-8") as f:
        for t in tasks:
            sid = t["snippet_id"]
            row = result_by_id.get(sid)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as mf:
        for rec in records_in:
            new_snips = []
            for sn in rec.get("snippets") or []:
                if not isinstance(sn, dict):
                    continue
                sid = sn.get("id")
                if not isinstance(sid, str):
                    continue
                out_sn = dict(sn)
                r = result_by_id.get(sid)
                if r:
                    out_sn["gpt_description"] = _gpt_payload_for_manifest(r)
                else:
                    out_sn["gpt_description"] = {"parse_error": True, "error": "not_processed"}
                new_snips.append(out_sn)
            out_rec = dict(rec)
            out_rec["snippets"] = new_snips
            out_rec["cleaning_stage"] = "02_gpt_description"
            mf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    date_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_results = [result_by_id[t["snippet_id"]] for t in tasks if t["snippet_id"] in result_by_id]
    _write_report(
        report_path,
        str(gd["model"]),
        date_s,
        tasks,
        result_by_id,
        total_in_tok,
        total_out_tok,
        total_cost,
    )

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _plot_exclusion_reasons(all_results, reports_dir / "gpt_exclusion_reasons.png")
    _plot_location_by_category(tasks, result_by_id, reports_dir / "location_by_category.png")
    _plot_breed_distribution(all_results, reports_dir / "breed_distribution.png")
    _plot_suitability_by_platform(tasks, result_by_id, reports_dir / "suitability_by_platform.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
