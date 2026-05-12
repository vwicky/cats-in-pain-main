"""Logging, config, paths, JSONL, retries, funnel plots."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# All metrics in final_report must be computed at runtime — no template literals for counts.
REPORT_METRICS_COMMENT = "All metrics computed; no template literals for counts."


def project_root() -> Path:
    """Repo root: ``parents[4]`` of ``src/scrapers/data_pipeline_v2/src/utils.py``."""
    return Path(__file__).resolve().parents[4]


def resolve_path(root: Path, p: str | Path) -> Path:
    """Resolve config path relative to repo root unless absolute."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def setup_logger(run_dir: str | Path, name: str = "pipeline_v2") -> logging.Logger:
    """Dual handler: console INFO + file DEBUG with timestamps."""
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


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; values in *override* win for leaves."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML, apply optional overrides, merge env for API keys."""
    root = project_root()
    cfg_path = Path(path) if Path(path).is_absolute() else (root / path).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = deep_merge_dict(cfg, overrides)

    import os

    if not (cfg.get("search") or {}).get("youtube_api_key"):
        cfg.setdefault("search", {})["youtube_api_key"] = os.environ.get("YOUTUBE_API_KEY", "")
    if not (cfg.get("search") or {}).get("openai_api_key"):
        cfg.setdefault("search", {})["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not (cfg.get("gpt_filter") or {}).get("openai_api_key"):
        cfg.setdefault("gpt_filter", {})["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")

    return cfg


def hash_config_section(
    cfg: dict[str, Any],
    key: str,
    exclude_subkeys: frozenset[str] | None = None,
) -> str:
    """Stable SHA-256 of canonical JSON for cfg[key].

    ``exclude_subkeys`` drops keys that affect runtime only (e.g. parallelism) so caches stay valid.
    """
    sub = cfg.get(key)
    if sub is None:
        sub = {}
    if exclude_subkeys:
        sub = {k: v for k, v in sub.items() if k not in exclude_subkeys}
    canonical = json.dumps(sub, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_run_dir(cfg: dict[str, Any], run_name: str | None = None) -> Path:
    """Create runs/{name}_{timestamp}/ with stage subdirs."""
    root = project_root()
    name = run_name or cfg.get("run_name", "run")
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_parent = resolve_path(root, cfg.get("output", {}).get("runs_dir", "src/scrapers/data_pipeline_v2/runs"))
    run_dir = runs_parent / f"{name}_{ts}"
    for sub in (
        "stage_1_search",
        "stage_2_tag_filter",
        "stage_3_gpt_filter",
        "stage_4_download",
        "stage_5_process",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def list_recent_run_dirs(runs_root: Path | str, n: int = 3) -> list[Path]:
    """Newest first by mtime."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[:n]


def save_jsonl(records: list[dict[str, Any]], path: str | Path, mode: str = "w") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    """Append one JSON object as a line. Line order is append order only.

    Parallel pipeline stages may complete tasks out of submission order, so JSONL rows are not
    sorted by time or video_id unless the caller enforces it. Downstream code should sort or
    index by fields (e.g. ``video_id``, ``processed_at``) instead of assuming stable file order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logging.getLogger("pipeline_v2").warning("Skipping malformed JSONL line %s in %s", line_no, path)
    return out


def print_funnel_table(stage_counts: dict[str, dict[str, int | float]], logger: logging.Logger) -> None:
    """Pretty-print funnel (stage -> in/out/retention)."""
    lines = ["Stage funnel:"]
    for stage, data in stage_counts.items():
        lines.append(f"  {stage}: {data}")
    msg = "\n".join(lines)
    logger.info(msg)
    print(msg)


def save_funnel_plot(stage_counts: dict[str, Any], run_dir: str | Path) -> Path:
    """Horizontal bar chart of counts at each stage."""
    sns.set_theme(style="whitegrid")
    run_dir = Path(run_dir)
    stages = list(stage_counts.keys())
    values = [stage_counts[s].get("out", stage_counts[s].get("count", 0)) for s in stages]

    fig, ax = plt.subplots(figsize=(10, max(4, len(stages) * 0.4)))
    ax.barh(stages[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Pipeline funnel (out per stage)")
    out_path = run_dir / "pipeline_funnel.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple[type, ...] = (Exception,),
):
    """Call fn with exponential backoff + jitter on failure."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retry_on as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            delay = min(base_delay * (2**attempt) + random.random(), max_delay)
            time.sleep(delay)
    raise last_exc  # pragma: no cover
