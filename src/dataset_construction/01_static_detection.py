#!/usr/bin/env python3
"""
Strict static detection: only clips where **no** visually meaningful change vs the first
frame (full BGR, every decoded frame, short-circuit). Loads ``sources.metadata_merged``.

  python dataset_construction/00_merge_metadata.py
  python dataset_construction/01_static_detection.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# Reduce FFmpeg/OpenCV H.264 decode noise on stderr (e.g. mmco: unref short failure)
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
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


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    """Return absolute path to snippet video if found."""
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
        pass  # YouTube / DailyMotion / mixed — use path + heuristics below
    for key in ("platform", "source_platform",):
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


# Cap sequential decode length per snippet (avoids pathological long files).
_MAX_FRAMES_TO_DECODE = 600


def is_static_video(
    video_path: str,
    variance_threshold: float = 0.5,
) -> tuple[bool, float]:
    """
    Strict static detection: compare every decoded frame to the **first** frame in
    full **BGR** (not grayscale). Tracks the maximum mean absolute difference across
    all pixels and channels (0–255 scale). Short-circuits as soon as the max meets
    or exceeds ``variance_threshold`` → not static.

    Static = no frame differs enough from the anchor (codec noise: use a small
    threshold such as 0.1–0.5, not 0.0).

    Returns ``(is_static, max_mean_absdiff_bgr)``. The second value is stored in
    reports as ``mean_variance`` for JSONL compatibility.

    Falls back to (True, 0.0) if the video cannot be read. Never raises.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return True, 0.0

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            return True, 0.0

        max_variance_found = 0.0
        frames_decoded = 1

        while frames_decoded < _MAX_FRAMES_TO_DECODE:
            ret, curr_frame = cap.read()
            if not ret or curr_frame is None:
                break

            diff = cv2.absdiff(curr_frame, first_frame)
            current_mean_diff = float(np.mean(diff))

            if current_mean_diff > max_variance_found:
                max_variance_found = current_mean_diff

            if max_variance_found >= variance_threshold:
                cap.release()
                return False, max_variance_found

            frames_decoded += 1

        cap.release()
        return True, max_variance_found

    except Exception:
        return True, 0.0


def _worker_scan(task: dict[str, Any]) -> dict[str, Any]:
    """Pool worker: run static detection for one resolved snippet."""
    path = task["video_path"]
    if path is None:
        return {
            **task,
            "status": "missing",
            "mean_variance": None,
        }
    is_st, mv = is_static_video(path, variance_threshold=task["variance_threshold"])
    status = "static" if is_st else "ok"
    return {
        **task,
        "status": status,
        "mean_variance": mv,
    }


def _percentile_table(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct = np.percentile(arr, qs)
    return {f"p{q}": float(pct[i]) for i, q in enumerate(qs)}


def _plot_histogram(
    variances: list[float],
    threshold: float,
    n_static: int,
    n_valid: int,
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_vals = np.array([max(v, 1e-9) for v in variances], dtype=np.float64)
    if plot_vals.size == 0:
        plot_vals = np.array([1e-6])
    lo = max(float(plot_vals.min()), 1e-9)
    hi = max(float(plot_vals.max()), lo * 1.001)
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    _counts, edges, patches = ax.hist(
        plot_vals,
        bins=bins,
        edgecolor="white",
        linewidth=0.5,
    )
    edge_centers = np.sqrt(np.maximum(edges[:-1], 1e-12) * np.maximum(edges[1:], 1e-12))
    for patch, c in zip(patches, edge_centers):
        patch.set_facecolor("#c44e52" if c < threshold else "#4682b4")
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("max mean |BGR − frame0| (0–255, log scale)")
    ax.set_ylabel("count")
    pct_s = 100.0 * n_static / n_valid if n_valid else 0.0
    ax.set_title("Strict static detection — max anchor BGR diff")
    ax.text(
        0.5,
        -0.15,
        f"threshold={threshold} | static={n_static} ({pct_s:.1f}%)",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_platform_bars(
    results: list[dict[str, Any]],
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    platforms = ["YouTube", "TikTok", "DailyMotion"]
    ok_c = {p: 0 for p in platforms}
    st_c = {p: 0 for p in platforms}
    for r in results:
        pl = r.get("platform", "YouTube")
        if pl not in ok_c:
            continue
        if r["status"] == "missing":
            continue
        if r["status"] == "ok":
            ok_c[pl] += 1
        elif r["status"] == "static":
            st_c[pl] += 1
    x = np.arange(len(platforms))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [ok_c[p] for p in platforms], w, label="ok", color="#4682b4")
    ax.bar(x + w / 2, [st_c[p] for p in platforms], w, label="static", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.set_ylabel("snippet count")
    ax.legend()
    ax.set_title("ok vs static by platform (non-missing only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_category_boxes(
    results: list[dict[str, Any]],
    threshold: float,
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    rows: list[tuple[str, float]] = []
    for r in results:
        if r["status"] == "missing":
            continue
        mv = r.get("mean_variance")
        if mv is None:
            continue
        cat = str(r.get("behavioral_category") or "unknown")
        rows.append((cat, float(mv)))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no data", ha="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    plot_df = pd.DataFrame(rows, columns=["behavioral_category", "mean_variance"])
    cat_order = sorted(plot_df["behavioral_category"].unique())
    fig, ax = plt.subplots(figsize=(max(10, len(cat_order) * 0.4), 5))
    sns.boxplot(
        data=plot_df,
        x="behavioral_category",
        y="mean_variance",
        order=cat_order,
        ax=ax,
    )
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.2)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("max mean BGR diff vs first frame")
    ax.set_title("Anchor BGR diff by behavioral_category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary_box(
    threshold: float,
    total: int,
    n_ok: int,
    n_static: int,
    n_missing: int,
    by_platform: dict[str, tuple[int, int, int]],
) -> None:
    def pct(n: int) -> str:
        return f"{100.0 * n / total:.1f}%" if total else "0.0%"

    lines = [
        "┌─────────────────────────────────────────────────────┐",
        "│  STATIC DETECTION (strict anchor BGR)                │",
        f"│  Threshold: {threshold:<40}│",
        "│                                                     │",
        f"│  Total snippets scanned:    {total:<23,}│",
        f"│  Clean (ok):                {n_ok:<5}  ({pct(n_ok):<14})│",
        f"│  Static detected:           {n_static:<5}  ({pct(n_static):<14})│",
        f"│  Missing files:             {n_missing:<5}  ({pct(n_missing):<14})│",
        "│                                                     │",
        "│  By platform:                                       │",
    ]
    for pl in ("YouTube", "TikTok", "DailyMotion"):
        ok, st, miss = by_platform.get(pl, (0, 0, 0))
        lines.append(f"│    {pl:<12} {ok} ok | {st} static | {miss} missing{' ' * 11}│")
    lines.append("└─────────────────────────────────────────────────────┘")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect static snippet videos.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override BGR anchor-diff threshold from config (mean abs diff 0–255).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths only; skip OpenCV and file outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count from config.",
    )
    args = parser.parse_args()

    cfg = load_config()
    sources = cfg["sources"]
    out_cfg = cfg["output"]
    sd_cfg = cfg["static_detection"]

    repo_root = REPO_ROOT
    meta_path = repo_root / sources["metadata_merged"]
    snippets_dirs = [repo_root / p for p in sources["snippets_dirs"]]
    reports_dir = repo_root / out_cfg["reports_dir"]
    manifests_dir = repo_root / out_cfg["manifests_dir"]

    variance_threshold = float(args.threshold if args.threshold is not None else sd_cfg["variance_threshold"])
    workers = int(args.workers if args.workers is not None else sd_cfg["workers"])
    workers = max(1, workers)

    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records_in: list[dict[str, Any]] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records_in.append(json.loads(line))

    input_records = len(records_in)
    input_snippets = sum(len(r.get("snippets") or []) for r in records_in)

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
            dur = sn.get("duration")
            duration_sec = float(dur) if isinstance(dur, (int, float)) else None
            video_stem = Path(resolved).stem if resolved else sid
            tasks.append(
                {
                    "snippet_id": sid,
                    "video_stem": video_stem,
                    "platform": platform,
                    "behavioral_category": behavioral_category,
                    "video_path": resolved,
                    "duration_sec": duration_sec,
                    "variance_threshold": variance_threshold,
                }
            )

    if args.dry_run:
        n_miss = sum(1 for t in tasks if t["video_path"] is None)
        n_found = len(tasks) - n_miss
        print(f"Dry run: {len(tasks)} snippets, resolved={n_found}, missing={n_miss}")
        print(f"Before cleaning: Records: {input_records}  |  Snippets: {input_snippets}")
        return 0

    results: list[dict[str, Any]] = []
    jsonl_path = reports_dir / "static_scan_results.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if workers == 1:
        pbar = tqdm(tasks, desc="Static detection", unit="snippet")
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for t in pbar:
                r = _worker_scan(t)
                results.append(r)
                row = {
                    "snippet_id": r["snippet_id"],
                    "video_stem": r["video_stem"],
                    "platform": r["platform"],
                    "behavioral_category": r["behavioral_category"],
                    "video_path": r["video_path"],
                    "status": r["status"],
                    "mean_variance": r["mean_variance"],
                    "duration_sec": r["duration_sec"],
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                n_st = sum(1 for x in results if x["status"] == "static")
                n_m = sum(1 for x in results if x["status"] == "missing")
                pbar.set_postfix_str(f"static={n_st} missing={n_m}", refresh=True)
    else:
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            with Pool(workers) as pool:
                for r in tqdm(
                    pool.imap(_worker_scan, tasks),
                    total=len(tasks),
                    desc="Static detection",
                    unit="snippet",
                ):
                    results.append(r)
                    row = {
                        "snippet_id": r["snippet_id"],
                        "video_stem": r["video_stem"],
                        "platform": r["platform"],
                        "behavioral_category": r["behavioral_category"],
                        "video_path": r["video_path"],
                        "status": r["status"],
                        "mean_variance": r["mean_variance"],
                        "duration_sec": r["duration_sec"],
                    }
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jf.flush()

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_static = sum(1 for r in results if r["status"] == "static")
    n_missing = sum(1 for r in results if r["status"] == "missing")

    valid_vars = [
        float(r["mean_variance"])
        for r in results
        if r["status"] != "missing" and r.get("mean_variance") is not None
    ]
    pct_table = _percentile_table(valid_vars)
    print(
        f"\nMax anchor BGR diff (mean |pixel| 0–255) — {len(valid_vars)} snippets with valid video:"
    )
    for key in ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
        if key in pct_table:
            label = key.upper() if key != "p50" else "p50   ← median"
            v = pct_table[key]
            fmt = f"{v:.4f}" if v < 2.0 else f"{v:.2f}"
            print(f"  {label}: {fmt}")

    sns.set_theme(style="whitegrid")
    _plot_histogram(
        valid_vars,
        variance_threshold,
        n_static,
        len(valid_vars),
        reports_dir / "static_detection_histogram.png",
    )
    _plot_platform_bars(results, reports_dir / "static_by_platform.png")
    _plot_category_boxes(results, variance_threshold, reports_dir / "variance_by_category.png")

    by_platform: dict[str, tuple[int, int, int]] = {
        "YouTube": (0, 0, 0),
        "TikTok": (0, 0, 0),
        "DailyMotion": (0, 0, 0),
    }
    for r in results:
        pl = r.get("platform", "YouTube")
        if pl not in by_platform:
            continue
        ok, st, miss = by_platform[pl]
        if r["status"] == "ok":
            ok += 1
        elif r["status"] == "static":
            st += 1
        else:
            miss += 1
        by_platform[pl] = (ok, st, miss)

    _print_summary_box(
        variance_threshold,
        len(results),
        n_ok,
        n_static,
        n_missing,
        by_platform,
    )

    static_ids = [r["snippet_id"] for r in results if r["status"] == "static"]
    missing_ids = [r["snippet_id"] for r in results if r["status"] == "missing"]
    with open(reports_dir / "static_flagged.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(static_ids) + ("\n" if static_ids else ""))
    with open(reports_dir / "missing_files.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(missing_ids) + ("\n" if missing_ids else ""))

    result_by_id = {r["snippet_id"]: r for r in results}
    records_out: list[dict[str, Any]] = []
    removed_static_total = 0
    removed_missing_total = 0

    for rec in records_in:
        new_snips = []
        rs = 0
        rm = 0
        for sn in rec.get("snippets") or []:
            if not isinstance(sn, dict):
                continue
            sid = sn.get("id")
            if not isinstance(sid, str):
                continue
            info = result_by_id.get(sid)
            if info is None:
                new_snips.append(sn)
                continue
            if info["status"] == "missing":
                rm += 1
                continue
            if info["status"] == "static":
                rs += 1
                continue
            new_snips.append(sn)

        removed_static_total += rs
        removed_missing_total += rm

        if not new_snips:
            continue

        out_rec = dict(rec)
        out_rec["snippets"] = new_snips
        out_rec["removed_snippets"] = {"static": rs, "missing": rm}
        out_rec["cleaning_stage"] = "01_static_detection"
        records_out.append(out_rec)

    output_records = len(records_out)
    output_snippets = sum(len(r.get("snippets") or []) for r in records_out)

    n_rec_all_static = 0
    n_rec_all_missing = 0
    for rec in records_in:
        snips = [
            s
            for s in rec.get("snippets") or []
            if isinstance(s, dict) and isinstance(s.get("id"), str)
        ]
        if not snips:
            continue
        statuses: list[str] = []
        for s in snips:
            info = result_by_id.get(s["id"])
            if info is None:
                statuses = []
                break
            statuses.append(info["status"])
        if not statuses:
            continue
        if all(s == "static" for s in statuses):
            n_rec_all_static += 1
        if all(s == "missing" for s in statuses):
            n_rec_all_missing += 1

    manifest_path = manifests_dir / "metadata_clean_01.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for rec in records_out:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary = {
            "stage": "01_static_detection",
            "detection": "strict_anchor_bgr",
            "max_frames_cap": _MAX_FRAMES_TO_DECODE,
            "threshold": variance_threshold,
            "input_records": input_records,
            "output_records": output_records,
            "input_snippets": input_snippets,
            "output_snippets": output_snippets,
            "removed_static": removed_static_total,
            "removed_missing": removed_missing_total,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        mf.write("# " + json.dumps(summary, ensure_ascii=False) + "\n")

    print("\nFinal comparison")
    print(f"  Before cleaning:")
    print(f"    Records:  {input_records:,}  |  Snippets: {input_snippets:,}")
    print(f"  After cleaning:")
    print(f"    Records:  {output_records:,}  |  Snippets: {output_snippets:,}")
    print(f"  Removed:")
    print(
        f"    Static:   {removed_static_total} snippets  "
        f"({n_rec_all_static} records fully dropped)"
    )
    print(
        f"    Missing:  {removed_missing_total} snippets  "
        f"({n_rec_all_missing} records fully dropped)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
