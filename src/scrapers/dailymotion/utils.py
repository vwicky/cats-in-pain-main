"""Run directories, logging, JSONL — aligned with data_pipeline_v2 patterns."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Repo root: ``parents[3]`` of ``src/scrapers/dailymotion/utils.py``."""
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, p: str | Path) -> Path:
    """Resolve *p* relative to *root* unless absolute (same idea as data_pipeline_v2)."""
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def hash_config_section(cfg: dict[str, Any], key: str) -> str:
    """Stable SHA-256 of canonical JSON for cfg[key] (query cache invalidation)."""
    sub = cfg.get(key)
    if sub is None:
        sub = {}
    canonical = json.dumps(sub, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def setup_logger(run_dir: str | Path, name: str = "dailymotion_pipeline") -> logging.Logger:
    """Dual handler: console INFO + file DEBUG with timestamps (matches pipeline v2)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def make_run_dir(run_name: str = "dailymotion_run") -> Path:
    """``dailymotion_scraper/runs/{run_name}_{timestamp}/`` with stage subdirs."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(__file__).resolve().parent / "runs"
    run_dir = base / f"{run_name}_{ts}"
    for sub in (
        "stage_1_search",
        "stage_2_tag_filter",
        "stage_3_gpt_filter",
        "stage_4_download",
        "stage_5_process",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML pipeline config (yolo, audio, download, output).

    Relative paths are resolved in order: (1) repo root, (2) ``dailymotion_scraper/`` package dir
    — so ``--config config/pipeline.yaml`` works when cwd is ``dailymotion_scraper/`` or repo root.
    """
    import os

    import yaml

    root = project_root()
    package_root = Path(__file__).resolve().parent

    if path:
        raw = Path(path)
        if raw.is_absolute():
            p = raw.resolve()
        else:
            cand_repo = (root / raw).resolve()
            cand_pkg = (package_root / raw).resolve()
            if cand_repo.is_file():
                p = cand_repo
            elif cand_pkg.is_file():
                p = cand_pkg
            else:
                raise FileNotFoundError(
                    f"Pipeline config not found. Tried: {cand_repo} | {cand_pkg}"
                )
    else:
        p = (package_root / "config" / "pipeline.yaml").resolve()

    if not p.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {p}")

    with open(p, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    # Merge API keys from environment (same keys as data_pipeline_v2)
    gpt = cfg.setdefault("gpt_filter", {})
    if not (gpt.get("openai_api_key") or "").strip():
        gpt["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    search = cfg.setdefault("search", {})
    if not (search.get("openai_api_key") or "").strip():
        search["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    return cfg


def save_jsonl(records: list[dict[str, Any]], path: str | Path, mode: str = "w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logging.getLogger("dailymotion_pipeline").warning("Skipping malformed JSONL line in %s", path)
    return out


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_final_report(
    run_dir: Path,
    funnel_rows: list[dict[str, Any]],
    elapsed_sec: float,
    logger: logging.Logger,
    extra: dict[str, str] | None = None,
) -> None:
    """Plain-text final report (pipeline v2 style)."""
    h = int(elapsed_sec // 3600)
    m = int((elapsed_sec % 3600) // 60)
    s = int(elapsed_sec % 60)
    date_s = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "=" * 60,
        "DAILYMOTION SCRAPER — FINAL REPORT",
        f"Run: {run_dir.name}",
        f"Date: {date_s} | Elapsed: {h}h {m}m {s}s",
        "=" * 60,
        "",
        "PIPELINE FUNNEL",
        "-" * 15,
        f"{'Stage':<32}│ {'In':>7} │ {'Out':>7} │ {'Retention':>10}",
        "-" * 60,
    ]
    for row in funnel_rows:
        o = row.get("out", "—")
        o_str = o if isinstance(o, str) else f"{o}"
        i = row.get("in", "—")
        i_str = i if isinstance(i, str) else f"{i}"
        lines.append(
            f"{str(row['name'])[:32]:<32}│ {i_str:>7} │ {o_str:>7} │ {str(row.get('retention', '—')):>10}"
        )
    lines.append("-" * 60)
    oy = None
    if funnel_rows:
        first = funnel_rows[0].get("out")
        last = funnel_rows[-1].get("out")
        if isinstance(first, int) and isinstance(last, int) and first > 0:
            oy = {
                "in": first,
                "out": last,
                "retention": f"{100.0 * last / max(1, first):.1f}%",
            }
            lines.append(
                f"{'Overall yield':<32}│ {oy['in']:>7} │ {oy['out']:>7} │ {oy['retention']:>10}"
            )
    lines.append("")
    lines.append("OUTPUT LOCATIONS")
    lines.append("-" * 16)
    ex = extra or {}
    lines.append(f"Run directory:  {run_dir.resolve()}")
    lines.append(f"Full log:       {run_dir / 'pipeline.log'}")
    if ex.get("metadata_dir"):
        lines.append(f"Metadata JSON:  {ex['metadata_dir']}")
    if ex.get("video_dir"):
        lines.append(f"Videos:         {ex['video_dir']}")
    if ex.get("snippets_dir"):
        lines.append(f"Snippets:       {ex['snippets_dir']}")
    if ex.get("metadata_jsonl"):
        lines.append(f"Metadata JSONL: {ex['metadata_jsonl']}")
    ss = ex.get("snippet_stats")
    if isinstance(ss, dict) and ss:
        lines.append("")
        lines.append("SNIPPET OUTPUT (process stage)")
        lines.append("-" * 28)
        lines.append(f"Total snippets:          {ss.get('total', 0)}")
        lines.append(f"Mean snippets/video:      {ss.get('mean_per_video', 0):.2f}")
        lines.append(f"Mean snippet duration:     {ss.get('mean_duration', 0):.2f}s")
    if ex.get("quality_notes"):
        lines.append("")
        lines.append("QUALITY NOTES")
        lines.append("-" * 13)
        for n in ex["quality_notes"]:
            lines.append(f"- {n}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    (run_dir / "final_report.txt").write_text(text, encoding="utf-8")
    logger.info(text)
