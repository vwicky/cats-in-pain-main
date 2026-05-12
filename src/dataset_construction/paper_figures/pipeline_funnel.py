"""Pipeline survivor funnel counts (snippet-level) for thesis fig/table.

Stages follow the subsection schema: Search → … → Final Labeled Set.
Upstream scraper totals (optional) live under ``paper_figures.pipeline_funnel`` in
``config.yaml``. In-repo manifests supply Download through GPT; the loaded
final manifest (--manifest) supplies suitability and training-ready label counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_extract import extract_suitable_for_training

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def count_snippets_nested_manifest(path: Path) -> int | None:
    """Count snippet dicts with string ``id`` in a video-style JSONL manifest."""
    if not path.is_file():
        return None
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            for sn in rec.get("snippets") or []:
                if isinstance(sn, dict) and isinstance(sn.get("id"), str) and sn["id"].strip():
                    n += 1
    return n


@dataclass(frozen=True)
class FunnelRow:
    stage: str
    count: int | None
    source_note: str
    include_in_plot: bool


def build_pipeline_funnel_rows(
    cfg: dict[str, Any],
    *,
    final_rows: list[dict[str, Any]],
) -> list[FunnelRow]:
    """Assemble per-stage counts and provenance notes."""
    pf = cfg.get("paper_figures")
    pf = pf if isinstance(pf, dict) else {}
    fn = pf.get("pipeline_funnel")
    fn = fn if isinstance(fn, dict) else {}

    def _int_opt(key: str) -> int | None:
        v = fn.get(key)
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    skip_dup = bool(fn.get("skip_duplicate_plateau_in_plot", True))

    merged_rel = str(
        fn.get("download_manifest")
        or (cfg.get("sources") or {}).get("metadata_merged")
        or "src/dataset_construction/manifests/metadata_merged.jsonl"
    )
    clean_01_rel = str(fn.get("static_removal_manifest") or "src/dataset_construction/manifests/metadata_clean_01.jsonl")
    gpt_rel = str(
        fn.get("gpt_manifest")
        or (cfg.get("gpt_description") or {}).get("output_manifest")
        or "src/dataset_construction/manifests/metadata_clean_02.jsonl"
    )

    merged_p = _resolve(merged_rel)
    c01_p = _resolve(clean_01_rel)
    gpt_p = _resolve(gpt_rel)

    n_merged = count_snippets_nested_manifest(merged_p)
    n_static = count_snippets_nested_manifest(c01_p)
    n_gpt = count_snippets_nested_manifest(gpt_p)

    suitable_n = sum(1 for r in final_rows if extract_suitable_for_training(r) is True)
    labeled_any = sum(1 for r in final_rows if r.get("final_label_5"))
    training_labeled = sum(
        1
        for r in final_rows
        if r.get("final_label_5") and extract_suitable_for_training(r) is True
    )

    n_final_rows = len(final_rows)

    search = _int_opt("search")
    kw = _int_opt("keyword_filter")
    llm = _int_opt("llm_filter")
    processing_override = _int_opt("processing")

    download_c = n_merged
    processing_c = processing_override if processing_override is not None else download_c

    out: list[FunnelRow] = []

    def add(label: str, count: int | None, note: str, plot: bool = True) -> None:
        out.append(FunnelRow(stage=label, count=count, source_note=note, include_in_plot=plot))

    _scraper_source = (
        "Unique video candidates across all pipeline v2 runs "
        "(runs/data-pipeline-v2/*/stage_{}; dl_01..06 sub-runs excluded as duplicates of parent run)."
    )
    add(
        "Search",
        search,
        _scraper_source.format("1_search/candidates.jsonl") if search is not None
        else "Set paper_figures.pipeline_funnel.search in config.yaml from scraper export.",
        plot=search is not None,
    )
    add(
        "Keyword Filter",
        kw,
        _scraper_source.format("2_tag_filter/kept.jsonl") if kw is not None
        else "Set paper_figures.pipeline_funnel.keyword_filter in config.yaml from scraper export.",
        plot=kw is not None,
    )
    add(
        "LLM Filter",
        llm,
        _scraper_source.format("3_gpt_filter/kept.jsonl") if llm is not None
        else "Set paper_figures.pipeline_funnel.llm_filter in config.yaml from scraper export.",
        plot=llm is not None,
    )
    add(
        "Download",
        download_c,
        f"Snippet count in nested manifest: {merged_p.relative_to(REPO_ROOT) if merged_p.is_relative_to(REPO_ROOT) else merged_p}",
    )
    add(
        "Processing",
        processing_c,
        "Default = Download count (no separate in-repo manifest). Override with processing.",
        plot=not (skip_dup and processing_c is not None and download_c is not None and processing_c == download_c),
    )
    add(
        "Static Removal",
        n_static,
        f"Snippet count after static scan: {c01_p.relative_to(REPO_ROOT) if c01_p.is_relative_to(REPO_ROOT) else c01_p}",
    )
    add(
        "GPT Description",
        n_gpt,
        f"Snippet count in manifest after GPT enrichment: {gpt_p.relative_to(REPO_ROOT) if gpt_p.is_relative_to(REPO_ROOT) else gpt_p}",
        plot=not (skip_dup and n_gpt is not None and n_static is not None and n_gpt == n_static),
    )
    add(
        "Suitability Filter",
        suitable_n,
        "suitable_for_training == True on final manifest rows (GPT + cat-ID + audio rules).",
        plot=not (skip_dup and n_gpt is not None and suitable_n == n_gpt),
    )

    skip_final_plot = bool(skip_dup and training_labeled == suitable_n)

    final_note = (
        f"Training subset: final_label_5 and suitable_for_training True "
        f"(rows with final_label_5: {labeled_any}; manifest rows: {n_final_rows})."
    )

    add(
        "Final Labeled Set",
        training_labeled,
        final_note,
        plot=not skip_final_plot,
    )

    return out


def funnel_rows_for_plot(rows: list[FunnelRow]) -> list[FunnelRow]:
    return [r for r in rows if r.include_in_plot and r.count is not None]


def write_pipeline_funnel_table(
    path: Path,
    rows: list[FunnelRow],
    *,
    final_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Plain-text table: stage, count, retention vs previous, data source."""
    lines: list[str] = [
        "Pipeline funnel — survivor counts per stage",
        "=" * 100,
        "Stages Search / Keyword Filter / LLM Filter: unique VIDEO candidates (cross-run deduplicated).",
        "Stages Download onwards: SNIPPET counts from nested-JSONL manifests.",
        "Retention = count / previous non-null count (first filled row has no retention).",
        "",
    ]
    w_stage, w_cnt, w_ret, w_src = 28, 8, 10, 170
    hdr = (
        f"{'Stage':<{w_stage}} {'Count':>{w_cnt}} {'Retention':>{w_ret}} "
        f"{'Data source / note':<{w_src}}"
    )
    lines.append(hdr)
    lines.append("-" * (w_stage + w_cnt + w_ret + w_src + 6))
    prev: int | None = None
    for r in rows:
        cnt_s = "—" if r.count is None else str(int(r.count))
        if r.count is None:
            ret_s = "—"
        elif prev is None or prev == 0:
            ret_s = "—"
        else:
            ret_s = f"{100.0 * float(r.count) / float(prev):.1f}%"
        if r.count is not None:
            prev = int(r.count)
        note = (r.source_note.replace("\n", " "))[:w_src]
        lines.append(
            f"{r.stage:<{w_stage}} {cnt_s:>{w_cnt}} {ret_s:>{w_ret}} {note:<{w_src}}"
        )
    lines.append("")
    if final_rows:
        lx = sum(1 for rr in final_rows if rr.get("final_label_5"))
        tr = sum(
            1
            for rr in final_rows
            if rr.get("final_label_5") and extract_suitable_for_training(rr) is True
        )
        if lx > tr:
            lines.append(
                "Note: Rows with merged label final_label_5 excluded by suitability filtering: "
                f"{lx - tr}."
            )
            lines.append("")
    lines.append(
        "Bar chart omits optional upstream stages until counts are set in config, "
        "and may omit adjacent plateaus when skip_duplicate_plateau_in_plot is true."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
