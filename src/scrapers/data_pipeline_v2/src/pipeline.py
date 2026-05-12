#!/usr/bin/env python3
"""Main orchestrator for data pipeline v2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Allow `python data_pipeline_v2/src/pipeline.py` from repo root
_V2_ROOT = Path(__file__).resolve().parent.parent
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

from src.audio_classifier import AudioClassifier
from src.downloader import download_batch, resolve_single_file
from src.gpt_filter import run_gpt_filter
from src.processor import process_batch
from src.search import resolve_seed_query_limits, run_search, total_seed_queries_from_limits
from src.tag_filter import run_tag_filter
from src.utils import (
    REPORT_METRICS_COMMENT,
    deep_merge_dict,
    list_recent_run_dirs,
    load_config,
    load_jsonl,
    make_run_dir,
    project_root,
    resolve_path,
    save_funnel_plot,
    save_jsonl,
    setup_logger,
)


STAGES = ("search", "tag_filter", "gpt_filter", "download", "process")


def _rows_for_yolo_process(dl_results: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Videos to run YOLO+audio on: ``success`` or ``already_exists`` with media under ``download.output_dir``.

    On resume, most rows are ``already_exists`` without paths in the log; paths are resolved from disk.
    Deduplicate by ``video_id`` (last row wins).
    """
    root = project_root()
    dc = cfg.get("download", {})
    out_dir = resolve_path(root, dc.get("output_dir", "data/dataset/videos_v2"))
    by_vid: dict[str, dict[str, Any]] = {}
    for d in dl_results:
        vid = d.get("video_id")
        if not vid:
            continue
        st = d.get("status")
        if st not in ("success", "already_exists"):
            continue
        vp = d.get("video_path") or resolve_single_file(str(out_dir / f"{vid}_video.*"))
        ap = d.get("audio_path") or resolve_single_file(str(out_dir / f"{vid}_audio.*"))
        if not vp or not ap:
            continue
        by_vid[vid] = {**d, "video_path": vp, "audio_path": ap}
    return list(by_vid.values())


def _stage_idx(name: str) -> int:
    return STAGES.index(name)


def _require_resume_paths(start_from: str, resume_dir: Path) -> None:
    """Validate JSONL inputs exist for resume."""
    need: dict[str, list[str]] = {
        "tag_filter": ["stage_1_search/candidates.jsonl"],
        "gpt_filter": ["stage_2_tag_filter/kept.jsonl"],
        "download": ["stage_3_gpt_filter/kept.jsonl"],
        "process": ["stage_4_download/download_log.jsonl"],
    }
    if start_from not in need:
        return
    for rel in need[start_from]:
        p = resume_dir / rel
        if not p.is_file():
            raise FileNotFoundError(f"Resume run dir missing required file: {p}")


def _error_resume_missing(runs_root: Path) -> None:
    recent = list_recent_run_dirs(runs_root, n=3)
    lines = "\n".join(str(p) for p in recent) if recent else "(no run directories found)"
    msg = (
        "When --start-from is not 'search', you must pass --resume-run-dir PATH "
        "to an existing run directory containing prior stage outputs.\n\n"
        "Recent run directories (newest first):\n"
        f"{lines}\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)


def _build_category_language_tables(
    candidates: list[dict],
    meta_rows: list[dict],
    dl_results: list[dict],
) -> tuple[list[str], list[str]]:
    """Report lines for behavioral category and language (computed)."""
    cat_cand = Counter()
    lang_cand = Counter()
    for c in candidates:
        cat_cand[str(c.get("behavioral_category") or "Unknown")] += 1
        lang_cand[str(c.get("query_language") or "Unknown")] += 1

    cat_snip = Counter()
    for m in meta_rows:
        cat = str(m.get("behavioral_category") or "Unknown")
        cat_snip[cat] += len(m.get("snippets") or [])

    dl_ok_by_cat = Counter()
    for d in dl_results:
        if d.get("status") == "success":
            dl_ok_by_cat[str(d.get("behavioral_category") or "Unknown")] += 1

    cats = sorted(set(cat_cand.keys()) | set(cat_snip.keys()) | set(dl_ok_by_cat.keys()))
    h = f"{'Category':<18} │ {'Candidates':>10} │ {'Downloaded':>10} │ {'Snippets':>10}"
    lines_cat = [h, "─" * 60]
    for cat in cats:
        lines_cat.append(
            f"{cat[:18]:<18} │ {cat_cand[cat]:>10} │ {dl_ok_by_cat[cat]:>10} │ {cat_snip[cat]:>10}"
        )

    langs = sorted(lang_cand.keys())
    total_u = sum(lang_cand.values()) or 1
    h2 = f"{'Language':<12} │ {'Candidates':>10} │ {'% of total':>12}"
    lines_lang = [h2, "─" * 40]
    for lang in langs:
        p = 100.0 * lang_cand[lang] / total_u
        lines_lang.append(f"{lang[:12]:<12} │ {lang_cand[lang]:>10} │ {p:>11.1f}%")

    return lines_cat, lines_lang


def write_final_report(
    run_dir: Path,
    cfg: dict,
    funnel: dict[str, Any],
    stats: dict[str, Any],
    elapsed_sec: float,
    logger: Any,
) -> None:
    # All metrics computed; no template literals for counts.
    _ = REPORT_METRICS_COMMENT
    root = project_root()
    h = int(elapsed_sec // 3600)
    m = int((elapsed_sec % 3600) // 60)
    run_label = run_dir.name
    date_s = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "=" * 60,
        "DATA PIPELINE v2 FINAL REPORT",
        f"Run: {run_label}",
        f"Date: {date_s} | Elapsed: {h}h {m}m",
        "=" * 60,
        "",
        "QUERY GENERATION",
        "-" * 16,
    ]
    qg = stats.get("query_gen", {})
    lines.append(f"Seed queries generated:   {qg.get('seed_count', 0)}")
    lines.append(f"Languages expanded to:     {qg.get('langs', 0)}")
    lines.append(f"Total queries executed:  {qg.get('total_queries', 0)}")
    lines.append("")
    lines.append("PIPELINE FUNNEL")
    lines.append("-" * 15)
    lines.append(f"{'Stage':<28}│ {'In':>7} │ {'Out':>7} │ {'Retention':>10}")
    lines.append("-" * 60)
    for row in funnel.get("rows", []):
        o = row.get("out", "—")
        o_str = o if isinstance(o, str) else f"{o}"
        i = row.get("in", "—")
        i_str = i if isinstance(i, str) else f"{i}"
        lines.append(
            f"{str(row['name'])[:28]:<28}│ {i_str:>7} │ {o_str:>7} │ {str(row.get('retention', '—')):>10}"
        )
    lines.append("-" * 60)
    oy = funnel.get("overall_yield")
    if oy:
        lines.append(
            f"{'Overall yield':<28}│ {oy['in']:>7} │ {oy['out']:>7} │ {oy['retention']:>10}"
        )
    lines.append("")
    lines.append("SNIPPET OUTPUT")
    lines.append("-" * 14)
    so = stats.get("snippets", {})
    lines.append(f"Total snippets saved:    {so.get('total', 0)}")
    lines.append(f"Mean snippets/video:      {so.get('mean_per_video', 0):.1f}")
    lines.append(f"Mean snippet duration:     {so.get('mean_duration', 0):.1f}s")
    lines.append("")
    lines.append("BY BEHAVIORAL CATEGORY (first-match query labels)")
    lines.append("-" * 22)
    for line in stats.get("by_category_lines", ["(no data)"]):
        lines.append(line)
    lines.append("")
    lines.append("BY LANGUAGE")
    lines.append("-" * 11)
    for line in stats.get("by_language_lines", ["(no data)"]):
        lines.append(line)
    lines.append("")
    lines.append("QUALITY NOTES")
    lines.append("-" * 13)
    for n in stats.get("quality_notes", []):
        lines.append(f"- {n}")
    lines.append("")
    lines.append("OUTPUT LOCATIONS")
    lines.append("-" * 16)
    lines.append(f"Snippets:  {resolve_path(root, cfg['output']['snippets_dir'])}")
    lines.append(f"Metadata:  {resolve_path(root, cfg['output']['metadata_file'])}")
    lines.append(f"Full log:  {run_dir / 'pipeline.log'}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    (run_dir / "final_report.txt").write_text(text, encoding="utf-8")
    logger.info(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cat behavior data pipeline v2")
    parser.add_argument("--config", default="src/scrapers/data_pipeline_v2/config/pipeline.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-from", default="search", choices=STAGES)
    parser.add_argument("--stop-after", default="process", choices=STAGES)
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        type=str,
        help="Continue an existing run: required when --start-from is after search. "
        "With --start-from search, optionally reuse that run dir to append YouTube search "
        "(skip queries already present in stage_1_search/candidates.jsonl).",
    )
    parser.add_argument(
        "--search-random-sample",
        default=None,
        type=int,
        metavar="N",
        help="Override search.random_sample_queries: randomly use N expanded queries before YouTube search (0 = all).",
    )
    parser.add_argument(
        "--search-random-sample-seed",
        default=None,
        type=int,
        metavar="SEED",
        help="Override search.random_sample_seed for reproducible subsample (omit for nondeterministic).",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = resolve_path(root, args.config)
    user_cfg = load_config(config_path)
    cfg = user_cfg
    run_name = args.run_name or cfg.get("run_name", "pipeline_v2_run")
    runs_root = resolve_path(root, cfg.get("output", {}).get("runs_dir", "src/scrapers/data_pipeline_v2/runs"))

    if args.start_from != "search":
        if not args.resume_run_dir:
            _error_resume_missing(runs_root)
        resume_dir = Path(args.resume_run_dir).resolve()
        if not resume_dir.is_dir():
            print(f"ERROR: --resume-run-dir is not a directory: {resume_dir}", file=sys.stderr)
            _error_resume_missing(runs_root)
        try:
            _require_resume_paths(args.start_from, resume_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        run_dir = resume_dir
        cu = run_dir / "config_used.yaml"
        if cu.is_file():
            frozen_cfg = load_config(cu)
            cfg = deep_merge_dict(frozen_cfg, user_cfg)
            logger = setup_logger(run_dir)
            logger.info(
                "Resume: config = merge(base=%s, override=%s); override (--config) wins on overlapping keys.",
                cu.name,
                config_path.name,
            )
        else:
            logger = setup_logger(run_dir)
    elif args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            print(f"ERROR: --resume-run-dir is not a directory: {run_dir}", file=sys.stderr)
            sys.exit(1)
        cfg["_resume_partial_search"] = True
        cu = run_dir / "config_used.yaml"
        if cu.is_file():
            frozen_cfg = load_config(cu)
            cfg = deep_merge_dict(frozen_cfg, user_cfg)
            logger = setup_logger(run_dir)
            logger.info(
                "Resume search: run_dir=%s; config merge(base=%s, override=%s).",
                run_dir,
                cu.name,
                config_path.name,
            )
        else:
            logger = setup_logger(run_dir)
            logger.info(
                "Resume search: run_dir=%s (no config_used.yaml; using --config only).",
                run_dir,
            )
        (run_dir / "stage_1_search").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg, run_name)
        logger = setup_logger(run_dir)
        with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
            import yaml

            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    logger.info("Started pipeline v2 run_dir=%s", run_dir)

    if args.search_random_sample is not None:
        cfg.setdefault("search", {})["random_sample_queries"] = args.search_random_sample
    if args.search_random_sample_seed is not None:
        cfg.setdefault("search", {})["random_sample_seed"] = args.search_random_sample_seed

    start_i = _stage_idx(args.start_from)
    stop_i = _stage_idx(args.stop_after)
    if stop_i < start_i:
        print("ERROR: --stop-after must be >= --start-from", file=sys.stderr)
        sys.exit(1)

    banner = f"""
╔══════════════════════════════════════════════════════════╗
║  CAT BEHAVIOR DATA PIPELINE v2                          ║
║  Run: {run_dir.name:<44} ║
║  Stages: search → tag_filter → gpt_filter →            ║
║          download → process                             ║
║  Starting from: {args.start_from:<20}                  ║
╚══════════════════════════════════════════════════════════╝"""
    print(banner)
    logger.info(banner)

    t0 = datetime.now().timestamp()
    stats_accum: dict[str, Any] = {
        "query_gen": {},
        "snippets": {},
        "by_category_lines": [],
        "by_language_lines": [],
        "quality_notes": [],
    }
    funnel_rows: list[dict[str, Any]] = []
    candidates: list[dict] = []
    kept_tag: list[dict] = []
    kept_gpt: list[dict] = []
    dl_results: list[dict] = []

    if start_i <= _stage_idx("search") <= stop_i:
        candidates, search_stats = run_search(cfg, logger, run_dir)
        seed_limits = resolve_seed_query_limits(cfg["search"])
        stats_accum["query_gen"] = {
            "seed_count": total_seed_queries_from_limits(seed_limits),
            "langs": cfg["search"].get("languages_per_query", 5),
            "total_queries": search_stats.get("total_queries", 0),
        }
        funnel_rows.append(
            {
                "name": f"Search ({search_stats.get('total_queries', 0)} queries)",
                "in": "—",
                "out": search_stats.get("unique_candidates", len(candidates)),
                "retention": "—",
            }
        )
        save_jsonl(candidates, run_dir / "stage_1_search" / "candidates.jsonl", mode="w")
    elif start_i > _stage_idx("search"):
        candidates = load_jsonl(run_dir / "stage_1_search" / "candidates.jsonl")
        funnel_rows.append(
            {
                "name": "Search (loaded)",
                "in": "—",
                "out": len(candidates),
                "retention": "—",
            }
        )

    if start_i <= _stage_idx("tag_filter") <= stop_i and candidates:
        kept_tag, _discarded_tag, tf_stats = run_tag_filter(candidates, cfg, logger, run_dir)
        save_jsonl(kept_tag, run_dir / "stage_2_tag_filter" / "kept.jsonl", mode="w")
        save_jsonl(_discarded_tag, run_dir / "stage_2_tag_filter" / "discarded.jsonl", mode="w")
        with open(run_dir / "stage_2_tag_filter" / "tag_filter_summary.txt", "w") as f:
            f.write(json.dumps(tf_stats, indent=2))
        funnel_rows.append(
            {
                "name": "Tag filter (metadata)",
                "in": tf_stats.get("input", len(candidates)),
                "out": tf_stats.get("kept", len(kept_tag)),
                "retention": f"{100.0 * tf_stats.get('kept', 0) / max(1, tf_stats.get('input', 1)):.1f}%",
            }
        )
    elif start_i > _stage_idx("tag_filter"):
        kept_tag = load_jsonl(run_dir / "stage_2_tag_filter" / "kept.jsonl")

    if start_i <= _stage_idx("gpt_filter") <= stop_i and kept_tag:
        kept_gpt, _discarded_gpt, gf_stats = run_gpt_filter(kept_tag, cfg, logger, run_dir)
        funnel_rows.append(
            {
                "name": "GPT filter (metadata)",
                "in": gf_stats.get("input", len(kept_tag)),
                "out": gf_stats.get("kept", len(kept_gpt)),
                "retention": f"{100.0 * gf_stats.get('kept', 0) / max(1, gf_stats.get('input', 1)):.1f}%",
            }
        )
        stats_accum["quality_notes"].append(
            f"GPT filter: {gf_stats.get('discarded', 0)} discarded, est. cost ${gf_stats.get('cost', 0):.4f}"
        )
    elif start_i > _stage_idx("gpt_filter"):
        kept_gpt = load_jsonl(run_dir / "stage_3_gpt_filter" / "kept.jsonl")

    meta_path = resolve_path(root, cfg["output"]["metadata_file"])
    log_path = resolve_path(root, cfg["output"]["pipeline_log"])

    if start_i <= _stage_idx("download") <= stop_i and kept_gpt:
        dc = cfg.get("download", {})
        max_videos = int(dc.get("max_videos") or 0)
        kept_for_download = kept_gpt[:max_videos] if max_videos > 0 else kept_gpt
        if max_videos > 0:
            logger.info(
                "Download: max_videos=%s — using %s of %s GPT-kept row(s)",
                max_videos,
                len(kept_for_download),
                len(kept_gpt),
            )
        dl_results = download_batch(kept_for_download, cfg, logger, run_dir, meta_path, log_path)
        ok = sum(1 for d in dl_results if d.get("status") == "success")
        funnel_rows.append(
            {
                "name": "Download",
                "in": len(kept_for_download),
                "out": ok,
                "retention": f"{100.0 * ok / max(1, len(kept_for_download)):.1f}%",
            }
        )
    elif start_i > _stage_idx("download"):
        dl_results = load_jsonl(run_dir / "stage_4_download" / "download_log.jsonl")

    if start_i <= _stage_idx("process") <= stop_i:
        dl_ok = _rows_for_yolo_process(dl_results, cfg)
        if dl_ok:
            ac = AudioClassifier(cfg, logger)
            proc_results = process_batch(dl_ok, cfg, logger, run_dir, ac)
            with_snip = sum(1 for p in proc_results if p.get("status") == "success")
            funnel_rows.append(
                {
                    "name": "YOLO + Audio",
                    "in": len(dl_ok),
                    "out": with_snip,
                    "retention": f"{100.0 * with_snip / max(1, len(dl_ok)):.1f}%",
                }
            )
            meta_all = load_jsonl(meta_path)
            total_snip = sum(len(r.get("snippets") or []) for r in meta_all)
            durs: list[float] = []
            spv: list[int] = []
            for r in meta_all:
                sn = r.get("snippets") or []
                spv.append(len(sn))
                for s in sn:
                    durs.append(float(s.get("duration", 0)))
            stats_accum["snippets"] = {
                "total": total_snip,
                "mean_per_video": float(np.mean(spv)) if spv else 0.0,
                "mean_duration": float(np.mean(durs)) if durs else 0.0,
            }
            stats_accum["quality_notes"].append(
                f"Audio model: {ac.model_name}; YOLO weights: {cfg.get('yolo', {}).get('weights', '')}"
            )

    if not candidates and (run_dir / "stage_1_search" / "candidates.jsonl").is_file():
        candidates = load_jsonl(run_dir / "stage_1_search" / "candidates.jsonl")
    cat_lines, lang_lines = _build_category_language_tables(candidates, load_jsonl(meta_path), dl_results)
    stats_accum["by_category_lines"] = cat_lines
    stats_accum["by_language_lines"] = lang_lines

    elapsed = datetime.now().timestamp() - t0

    first_out = funnel_rows[0]["out"] if funnel_rows else 0
    last_out = funnel_rows[-1]["out"] if funnel_rows else 0
    overall = None
    if isinstance(first_out, int) and isinstance(last_out, int) and first_out:
        overall = {
            "in": first_out,
            "out": last_out,
            "retention": f"{100.0 * last_out / max(1, first_out):.1f}%",
        }

    write_final_report(
        run_dir,
        cfg,
        {"rows": funnel_rows, "overall_yield": overall},
        stats_accum,
        elapsed,
        logger,
    )
    save_funnel_plot(
        {str(r["name"]): {"out": r["out"]} for r in funnel_rows if isinstance(r.get("out"), int)},
        run_dir,
    )


if __name__ == "__main__":
    main()
