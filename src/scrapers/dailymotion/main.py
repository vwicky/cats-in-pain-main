"""
Dailymotion cat video scraper — same stages as data_pipeline_v2 / TikTok:

search → tag_filter → gpt_filter → download → per-video metadata JSON → process (YOLO + audio)

Prerequisites: ffmpeg on PATH; optional OPENAI_API_KEY; for process: same deps as
``data_pipeline_v2`` (torch, ultralytics, …).

Usage:
  python dailymotion_scraper/main.py
  python dailymotion_scraper/main.py --config dailymotion_scraper/config/pipeline.yaml
  cd dailymotion_scraper && python main.py --config config/pipeline.yaml
  python dailymotion_scraper/main.py --dry-run --max-videos 10
  python dailymotion_scraper/main.py --skip-process
  python dailymotion_scraper/main.py --start-from tag_filter --resume-run-dir dailymotion_scraper/runs/<previous_run>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_args() -> argparse.Namespace:
    from config import DEFAULT_PIPELINE_CONFIG, DEFAULT_RUN_NAME

    p = argparse.ArgumentParser(
        description="Dailymotion cat video scraper (YouTube/TikTok-shaped pipeline)"
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_PIPELINE_CONFIG,
        help="YAML config with yolo, audio, download, output (process stage)",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Stop search after this many unique candidates (default: no cap)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Search only: write stage_1 candidates + report; no tag/GPT/download/process",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=f"Run directory prefix (default: {DEFAULT_RUN_NAME})",
    )
    p.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip YOLO + audio snippet stage (download + per-video JSON only)",
    )
    p.add_argument(
        "--start-from",
        type=str,
        default="search",
        choices=("search", "tag_filter", "gpt_filter", "download"),
        help="Resume from a stage using artifacts under --resume-run-dir (not search).",
    )
    p.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Previous run directory containing stage_* outputs (required if --start-from is not search).",
    )
    return p.parse_args()


def _metadata_record(row: dict, video_path: str, audio_path: str) -> dict:
    """Single JSON record: Dailymotion fields + YouTube-shaped aliases + filter outputs."""
    vid = row["video_id"]
    gf = row.get("gpt_filter") or {}
    return {
        "video_id": vid,
        "source": "dailymotion",
        "source_platform": row.get("source_platform", "dailymotion"),
        "video_path": video_path,
        "audio_path": audio_path,
        "duration_sec": row.get("duration_sec"),
        "duration_seconds": row.get("duration_seconds"),
        "title": row.get("title", ""),
        "webpage_url": row.get("webpage_url") or row.get("url"),
        "dm_id": row.get("dm_id"),
        "views_total": row.get("views_total"),
        "view_count": row.get("view_count"),
        "likes_total": row.get("likes_total"),
        "like_count": row.get("like_count"),
        "tags": row.get("tags", []),
        "channel": row.get("channel", ""),
        "channel_title": row.get("channel_title", ""),
        "language": row.get("language", ""),
        "query_language": row.get("query_language", "unknown"),
        "description": row.get("description", ""),
        "created_time": row.get("created_time"),
        "search_query": row.get("search_query"),
        "behavioral_category": row.get("behavioral_category"),
        "category_name": row.get("category_name", "Dailymotion"),
        "tag_filter": row.get("tag_filter") or {"passed": True},
        "gpt_decision": row.get("gpt_decision"),
        "gpt_reason": row.get("gpt_reason"),
        "gpt_confidence": row.get("gpt_confidence"),
        "gpt_filter": gf,
    }


def main() -> None:
    os.chdir(_ROOT)

    load_dotenv()
    args = parse_args()

    import yaml

    from config import DEFAULT_RUN_NAME, METADATA_DIR, SEARCH_QUERIES, VIDEO_DIR
    from dm_client import fetch_video_candidates, guess_behavioral_category
    from downloader import download_batch
    from gpt_filter import run_gpt_filter
    from tag_filter import run_tag_filter
    from utils import (
        load_jsonl,
        load_pipeline_config,
        make_run_dir,
        project_root,
        resolve_path,
        save_jsonl,
        setup_logger,
        write_final_report,
    )

    pipe_cfg = load_pipeline_config(args.config)

    for d in (VIDEO_DIR, METADATA_DIR):
        os.makedirs(d, exist_ok=True)
    repo = project_root()

    run_name = args.run_name or pipe_cfg.get("run_name") or DEFAULT_RUN_NAME
    run_dir = make_run_dir(run_name)
    logger = setup_logger(run_dir)

    cfg_used_path = run_dir / "config_used.yaml"
    with open(cfg_used_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(pipe_cfg, f, allow_unicode=True, sort_keys=False)

    logger.info("Started Dailymotion pipeline run_dir=%s", run_dir.resolve())

    banner = f"""
╔══════════════════════════════════════════════════════════╗
║  CAT BEHAVIOR DATA PIPELINE — Dailymotion source         ║
║  Run: {run_dir.name:<44} ║
║  Stages: search → tag_filter → gpt_filter → download →   ║
║          process (YOLO + audio)                         ║
╚══════════════════════════════════════════════════════════╝"""
    print(banner)
    for line in banner.strip().split("\n"):
        logger.debug(line)
    logger.info(
        "CAT BEHAVIOR DATA PIPELINE — Dailymotion | run=%s",
        run_dir.name,
    )

    t0 = datetime.now().timestamp()
    funnel_rows: list[dict] = []

    search_cfg = pipe_cfg.get("search") or {}
    start_from = args.start_from
    resume_dir = Path(args.resume_run_dir).expanduser().resolve() if args.resume_run_dir else None

    if start_from != "search":
        if not resume_dir or not resume_dir.is_dir():
            print(
                "ERROR: --resume-run-dir must point to an existing run directory when "
                "--start-from is not search.",
                file=sys.stderr,
            )
            sys.exit(2)
        logger.info("Resuming from prior run: %s (start_from=%s)", resume_dir, start_from)

    candidates: list[dict] = []
    tag_passed: list[dict] = []
    kept_gpt: list[dict] = []

    from query_generation import prepare_query_rows_for_search

    if start_from == "search":
        if search_cfg.get("use_static_queries"):
            query_rows = [
                {
                    "query": q,
                    "behavioral_category": guess_behavioral_category(q),
                    "query_language": "English",
                    "seed_query": q,
                }
                for q in SEARCH_QUERIES
            ]
            logger.info(
                "Search: static queries from config.py (%d strings, no GPT query gen)",
                len(query_rows),
            )
        else:
            from query_generation import load_or_generate_queries

            query_rows = load_or_generate_queries(pipe_cfg, logger, run_dir)
            logger.info("Search: GPT query generation → %d expanded search rows", len(query_rows))

        query_rows = prepare_query_rows_for_search(query_rows, search_cfg, logger)
        save_jsonl(query_rows, run_dir / "stage_1_search" / "generated_queries.jsonl", mode="w")

        candidates = fetch_video_candidates(
            max_candidates=args.max_videos,
            logger=logger,
            query_rows=query_rows,
            search_options=search_cfg,
        )
        save_jsonl(candidates, run_dir / "stage_1_search" / "candidates.jsonl", mode="w")
        funnel_rows.append(
            {
                "name": "Search (Dailymotion API)",
                "in": "—",
                "out": len(candidates),
                "retention": "—",
            }
        )
        logger.info("Search complete: %s unique candidates", len(candidates))

    elif start_from == "tag_filter":
        src_c = resume_dir / "stage_1_search" / "candidates.jsonl"
        if not src_c.is_file():
            print(f"ERROR: missing {src_c}", file=sys.stderr)
            sys.exit(2)
        candidates = load_jsonl(src_c)
        save_jsonl(candidates, run_dir / "stage_1_search" / "candidates.jsonl", mode="w")
        gq = resume_dir / "stage_1_search" / "generated_queries.jsonl"
        if gq.is_file():
            shutil.copy2(gq, run_dir / "stage_1_search" / "generated_queries.jsonl")
        funnel_rows.append(
            {
                "name": "Search (resumed — same candidates as prior run)",
                "in": "—",
                "out": len(candidates),
                "retention": "—",
            }
        )
        logger.info(
            "Resumed %s candidates from %s (duration cap changes do not apply — re-run search for 180s discovery)",
            len(candidates),
            src_c,
        )

    elif start_from == "gpt_filter":
        src_t = resume_dir / "stage_2_tag_filter" / "kept.jsonl"
        if not src_t.is_file():
            print(f"ERROR: missing {src_t}", file=sys.stderr)
            sys.exit(2)
        tag_passed = load_jsonl(src_t)
        n_search = len(load_jsonl(resume_dir / "stage_1_search" / "candidates.jsonl"))
        funnel_rows.append(
            {
                "name": "Search (Dailymotion API)",
                "in": "—",
                "out": n_search,
                "retention": "—",
            }
        )
        funnel_rows.append(
            {
                "name": "Tag filter (metadata)",
                "in": n_search,
                "out": len(tag_passed),
                "retention": f"{100.0 * len(tag_passed) / max(1, n_search):.1f}%",
            }
        )
        logger.info("Resumed %s tag-filtered rows from %s", len(tag_passed), src_t)

    elif start_from == "download":
        src_g = resume_dir / "stage_3_gpt_filter" / "kept.jsonl"
        if not src_g.is_file():
            print(f"ERROR: missing {src_g}", file=sys.stderr)
            sys.exit(2)
        kept_gpt = load_jsonl(src_g)
        n_search = len(load_jsonl(resume_dir / "stage_1_search" / "candidates.jsonl"))
        n_tag = len(load_jsonl(resume_dir / "stage_2_tag_filter" / "kept.jsonl"))
        funnel_rows.append(
            {
                "name": "Search (Dailymotion API)",
                "in": "—",
                "out": n_search,
                "retention": "—",
            }
        )
        funnel_rows.append(
            {
                "name": "Tag filter (metadata)",
                "in": n_search,
                "out": n_tag,
                "retention": f"{100.0 * n_tag / max(1, n_search):.1f}%",
            }
        )
        funnel_rows.append(
            {
                "name": "GPT filter (metadata)",
                "in": n_tag,
                "out": len(kept_gpt),
                "retention": f"{100.0 * len(kept_gpt) / max(1, n_tag):.1f}%",
            }
        )
        logger.info("Resumed %s GPT-kept rows from %s", len(kept_gpt), src_g)

    extra_report: dict = {
        "metadata_dir": str(Path(METADATA_DIR).resolve()),
        "video_dir": str(Path(VIDEO_DIR).resolve()),
        "snippets_dir": str(resolve_path(repo, pipe_cfg["output"]["snippets_dir"])),
        "metadata_jsonl": str(resolve_path(repo, pipe_cfg["output"]["metadata_file"])),
    }

    if args.dry_run:
        elapsed = datetime.now().timestamp() - t0
        write_final_report(run_dir, funnel_rows, elapsed, logger, extra=extra_report)
        print("\n[DONE] dry-run — tag_filter, gpt_filter, download, process skipped")
        return

    if start_from in ("search", "tag_filter"):
        logger.info(
            "Starting stage 2 (tag filter); GPT runs after this completes — watch the tag_filter progress bar"
        )
        tag_passed, _tag_disc, tf_stats = run_tag_filter(candidates, logger, run_dir)
        logger.info(
            "Tag filter done (%d kept). Starting stage 3 (GPT filter)…",
            tf_stats["kept"],
        )
        funnel_rows.append(
            {
                "name": "Tag filter (metadata)",
                "in": tf_stats["input"],
                "out": tf_stats["kept"],
                "retention": f"{100.0 * tf_stats['kept'] / max(1, tf_stats['input']):.1f}%",
            }
        )

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if (pipe_cfg.get("gpt_filter") or {}).get("openai_api_key"):
        api_key = (pipe_cfg["gpt_filter"].get("openai_api_key") or api_key).strip()

    if start_from in ("search", "tag_filter", "gpt_filter"):
        _gw = (pipe_cfg.get("gpt_filter") or {}).get("parallel_workers")
        kept_gpt, disc_gpt, gf_stats = run_gpt_filter(
            tag_passed,
            api_key,
            logger,
            run_dir,
            parallel_workers=int(_gw) if _gw is not None else None,
        )
        for item in disc_gpt:
            logger.info(
                "GPT reject %s — %s",
                item.get("video_id"),
                (item.get("gpt_filter") or {}).get("reason", ""),
            )

        funnel_rows.append(
            {
                "name": "GPT filter (metadata)",
                "in": gf_stats["input"],
                "out": gf_stats["kept"],
                "retention": f"{100.0 * gf_stats['kept'] / max(1, gf_stats['input']):.1f}%",
            }
        )
        logger.info(
            "GPT filter: %s discarded, est. cost $%.4f, tokens=%s",
            gf_stats["discarded"],
            gf_stats["cost"],
            gf_stats["tokens"],
        )

    log_rows, dl_counts = download_batch(
        kept_gpt,
        logger,
        run_dir,
        METADATA_DIR,
        pipe_cfg.get("download") or {},
    )
    n_dl_in = len(kept_gpt)
    n_dl_out = dl_counts["success"] + dl_counts["skipped"]
    funnel_rows.append(
        {
            "name": "Download",
            "in": n_dl_in,
            "out": n_dl_out,
            "retention": f"{100.0 * n_dl_out / max(1, n_dl_in):.1f}%",
        }
    )

    n_meta_written = 0
    for row, rec in zip(kept_gpt, log_rows):
        vid = row.get("video_id")
        meta_path = os.path.join(METADATA_DIR, f"{vid}.json")
        st = rec.get("status")
        if st == "success":
            vp = rec.get("video_path")
            ap = rec.get("audio_path")
            if vp and ap:
                record = _metadata_record(row, vp, ap)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                n_meta_written += 1
                logger.info(
                    "Saved metadata %s behavior=%s conf=%s",
                    vid,
                    (row.get("gpt_filter") or {}).get("behavior_category"),
                    (row.get("gpt_filter") or {}).get("confidence"),
                )
        elif st == "already_exists":
            logger.info("Metadata already present for %s — skip write", vid)

    quality_notes: list[str] = []
    snippet_stats: dict | None = None

    if not args.skip_process and kept_gpt:
        try:
            from pipeline_bridge import import_process_stack, rows_for_yolo_process

            AudioClassifier, process_batch = import_process_stack()
            dl_ok = rows_for_yolo_process(log_rows, pipe_cfg)
            if dl_ok:
                ac = AudioClassifier(pipe_cfg, logger)
                proc_results = process_batch(dl_ok, pipe_cfg, logger, run_dir, ac)
                with_snip = sum(1 for p in proc_results if p.get("status") == "success")
                funnel_rows.append(
                    {
                        "name": "YOLO + Audio",
                        "in": len(dl_ok),
                        "out": with_snip,
                        "retention": f"{100.0 * with_snip / max(1, len(dl_ok)):.1f}%",
                    }
                )
                meta_path_v2 = resolve_path(repo, pipe_cfg["output"]["metadata_file"])
                meta_all = load_jsonl(meta_path_v2)
                total_snip = sum(len(r.get("snippets") or []) for r in meta_all)
                durs: list[float] = []
                spv: list[int] = []
                for r in meta_all:
                    sn = r.get("snippets") or []
                    spv.append(len(sn))
                    for s in sn:
                        durs.append(float(s.get("duration", 0)))
                snippet_stats = {
                    "total": total_snip,
                    "mean_per_video": float(np.mean(spv)) if spv else 0.0,
                    "mean_duration": float(np.mean(durs)) if durs else 0.0,
                }
                quality_notes.append(
                    f"Audio model: {ac.model_name}; YOLO weights: {pipe_cfg.get('yolo', {}).get('weights', '')}"
                )
            else:
                logger.warning("Process stage: no rows with video+audio paths (check download log).")
        except ImportError as e:
            logger.warning(
                "Process stage skipped (import error — install data_pipeline_v2 deps): %s",
                e,
            )
        except Exception as e:
            logger.exception("Process stage failed: %s", e)

    if snippet_stats:
        extra_report["snippet_stats"] = snippet_stats
    if quality_notes:
        extra_report["quality_notes"] = quality_notes

    elapsed = datetime.now().timestamp() - t0
    logger.info(
        "Run finished: metadata_written=%s download_ok=%s skipped=%s failed=%s",
        n_meta_written,
        dl_counts["success"],
        dl_counts["skipped"],
        dl_counts["failed"],
    )

    write_final_report(run_dir, funnel_rows, elapsed, logger, extra=extra_report)
    print(
        f"\n[DONE] metadata_json_written={n_meta_written} "
        f"(download success={dl_counts['success']} skipped={dl_counts['skipped']} "
        f"failed={dl_counts['failed']})"
    )


if __name__ == "__main__":
    main()
