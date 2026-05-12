#!/usr/bin/env python3
"""
Generate thesis PNG figures and TXT tables from final_dataset_v2.jsonl.

Label figures (09–12) and **table 04** use ``audio_label_10``. ``n_missing_audio10`` applies to those
and to the face table.

**Tables 01–03** pair with **figures 01–03**: suitability, exclusion reasons, top-10 breeds.

``n_missing_label5`` counts null ``final_label_5`` on the manifest (reference only).

CLI thresholds (defaults in parentheses):
  --location-other-min (50)    Collapse rare gpt_location_type to "other".
  --behavior-min-count (10)  Merge rare gpt_primary_behavior to "other".
  --location-top-k (6)         Top location types before "other" for fig 11 & 20 audio×location heatmaps.

**Table 08**: ``table_08_lighting_blur_counts.txt`` — lighting × blur snippet counts (replaces grouped bar).

``n_missing_audio10`` is computed once after loading the manifest
and is referenced in figure_report.txt for audio ten-class plots and table 4 (face).

**Fig 11 / 20**: audio ten-class × GPT location (row-normalized vs raw counts); table 4 is face × audio class.

**Fig 21 / 22**: primary behavior × setting (`setting_tri`: Indoor / Outdoor / Mixed/Unclear), same behavior collapse as fig 12 (min-count + top-8 + other).

**Tables 11–13**: binary method comparison, Track A per-class CV (``cv_summary.json`` or report snapshot), pairwise ST-GCN CSV (``.txt`` only).

If PyYAML is not installed, ``cost_report.load_dataset_config`` falls back to
fixed relative paths under the repo for the cost table (same defaults as
config.yaml). Install ``pyyaml`` to read the full config file.

At the end of a run, prints a file checklist and exits with code 1 if any
expected output is missing or a PNG is zero bytes.

Examples::

  python src/dataset_construction/paper_figures/generate_plots.py
  python src/dataset_construction/paper_figures/generate_plots.py \\
    --manifest src/dataset_construction/manifests/final_dataset_v2.jsonl \\
    --location-other-min 40 --behavior-min-count 10 --location-top-k 6
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.style as mpl_style
import numpy as np
import pandas as pd
import seaborn as sns

from cost_report import build_cost_table_txt, load_dataset_config, write_cost_table
from pipeline_funnel import (
    build_pipeline_funnel_rows,
    funnel_rows_for_plot,
    write_pipeline_funnel_table,
)
from thesis_extract import (
    BINARY_ORDER,
    BLUR_ORDER,
    LABEL10_ORDER,
    LIGHTING_ORDER,
    aggregate_platform_suitability,
    build_thesis_dataframe,
    collapse_behaviors_for_emotion_heatmap,
    extract_face_visible,
    extract_suitable_for_training,
    heatmap_annotations_row_norm,
    row_norm_matrix,
    top_k_plus_other,
)
from thesis_io import two_column_table, write_txt_table
from thesis_report import attach_derived_columns, build_figure_report
from metrics_paper_tables import write_metrics_paper_tables

PAPER_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PAPER_DIR.parent / "manifests" / "final_dataset_v2.jsonl"
OUTPUT_DIR = PAPER_DIR / "output_pngs"
REPORT_FILENAME = "figure_report.txt"

EXPECTED_TABLES = (
    "table_01_suitability.txt",
    "table_02_exclusion_reasons.txt",
    "table_03_top10_breeds.txt",
    "table_04_face_by_audio_label_10.txt",
    "table_05_gpt_costs.txt",
    "table_06_pipeline_funnel.txt",
    "table_07_platform_suitability.txt",
    "table_08_lighting_blur_counts.txt",
    "table_09_emotion_labeling_class_distribution.txt",
    "table_10_audiosep_raw_vs_proc_per_class_agreement.txt",
    "table_11_binary_method_comparison.txt",
    "table_12_track_a_per_class_f1.txt",
    "table_13_pairwise_binary_results.txt",
)
EXPECTED_PNGS = (
    "fig_01_suitability.png",
    "fig_02_exclusion_reasons.png",
    "fig_03_breed_distribution.png",
    "fig_04_environment.png",
    "fig_05_behavior.png",
    "fig_06_lighting_blur_heatmap.png",
    "fig_07_face_visibility.png",
    "fig_08_behavior_by_setting.png",
    "fig_09_audio10_labels_twopanel.png",
    "fig_10_audio10_setting_rownorm_heatmap.png",
    "fig_11_audio10_location_rownorm_heatmap.png",
    "fig_20_audio10_location_counts_heatmap.png",
    "fig_12_audio10_behavior_rownorm_heatmap.png",
    "fig_21_behavior_setting_rownorm_heatmap.png",
    "fig_22_behavior_setting_counts_heatmap.png",
    "fig_13_pipeline_funnel.png",
    "fig_14_platform_suitable_stack.png",
    "fig_15_emotion_labeling_class_distribution_table.png",
    "fig_16_audiosep_per_class_prediction_agreement_table.png",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def apply_publication_style() -> None:
    try:
        mpl_style.use("seaborn-v0_8-whitegrid")
    except OSError:
        sns.set_theme(style="whitegrid")
    sns.set_theme(style="whitegrid", palette="crest")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
        }
    )


def _bar_palette(n: int):
    """Discrete bar colors (same as fig_07: seaborn ``crest`` sequential palette)."""
    n = max(int(n), 1)
    return sns.color_palette("crest", n_colors=n)


def _save(out_dir: Path, fig_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / fig_name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


# --- Default path to 04_audio_classification batch report ---
DEFAULT_EMOTION_LABELING_REPORT = PAPER_DIR.parent / "reports" / "emotion_labeling_report.txt"
_EMOTION_COUNT_LINE = re.compile(r"^\s+([A-Za-z]+)\s*:\s*(\d+)\s*$")
# If report file is absent, use counts from 2026-04-19 committed report (7329 files).
_FALLBACK_EMOTION_COUNTS: dict[str, int] = {
    "Angry": 509,
    "Defence": 46,
    "Fighting": 196,
    "Happy": 1155,
    "HuntingMind": 997,
    "Mating": 1409,
    "MotherCall": 764,
    "Paining": 1055,
    "Resting": 1116,
    "Warning": 82,
}
# Committed notebook stdout: mean pred_agrees per class; ascending by agreement %.
# Source: _archive/top_level_notebooks/checking_extraction_real_vs_distribution.ipynb
_AUDIOSEP_PER_CLASS_AGREEMENT_ASC: list[tuple[str, float]] = [
    ("Warning", 82.67),
    ("MotherCall", 89.19),
    ("Paining", 91.75),
    ("Fighting", 95.00),
    ("Defence", 95.19),
    ("Happy", 96.63),
    ("HuntingMind", 97.23),
    ("Resting", 97.30),
    ("Angry", 97.67),
    ("Mating", 98.34),
]


def parse_emotion_labeling_report_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    found: dict[str, int] = {}
    for line in text.splitlines():
        m = _EMOTION_COUNT_LINE.match(line)
        if m:
            found[m.group(1)] = int(m.group(2))
    return found


def build_emotion_labeling_count_table(
    emotion_report: Path | None = None,
) -> tuple[list[tuple[str, int, float]], int, str]:
    """Return (rows as class, count, pct), total, source note for caption."""
    p = Path(emotion_report or DEFAULT_EMOTION_LABELING_REPORT).resolve()
    if p.is_file():
        raw = parse_emotion_labeling_report_counts(p)
        src = str(p)
    else:
        raw = dict(_FALLBACK_EMOTION_COUNTS)
        src = f"{p} (missing on disk; fallback = 2026-04-19 report snapshot)"
    ordered: list[tuple[str, int, float]] = []
    for cls in LABEL10_ORDER:
        if cls not in raw:
            raise KeyError(f"Class {cls!r} not found in emotion report; keys={sorted(raw)!r}")
        ordered.append((cls, raw[cls], 0.0))
    total = sum(t[1] for t in ordered)
    if total <= 0:
        raise ValueError("Total count must be positive")
    out = [(c, n, 100.0 * n / total) for c, n, _ in ordered]
    return out, total, src


def _plot_pub_table_png(
    title: str,
    col_labels: list[str],
    cell_rows: list[list[str]],
    out_dir: Path,
    fig_name: str,
    *,
    figsize: tuple[float, float],
    cell_fontsize: float = 11,
    scale_xy: tuple[float, float] = (1.14, 2.08),
    footnote: str | None = None,
) -> None:
    """Matplotlib table figure aligned with thesis style (crest header, zebra rows)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(cell_fontsize)
    tbl.scale(scale_xy[0], scale_xy[1])
    header_bg = "#1b4332"
    row_even, row_odd = "#f0f7f4", "#ffffff"
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.25")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color="white", weight="bold", fontsize=cell_fontsize)
        else:
            cell.set_facecolor(row_even if r % 2 else row_odd)
    ax.set_title(title, fontsize=14, pad=12)
    if footnote:
        fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=9)
    plt.subplots_adjust(bottom=0.12 if footnote else 0.06, top=0.92)
    _save(out_dir, fig_name)


def write_table_09_emotion_labeling_class_distribution(
    out_dir: Path, emotion_report: Path | None = None
) -> None:
    rows_pct, total, src = build_emotion_labeling_count_table(emotion_report)
    lines = [
        "Table 9 — Emotion model predicted class distribution",
        "",
        f"Caption: Batch inference on separated audios (CatEmotionModel / 04_audio_classification).",
        f"Source: {src}",
        "",
        f"{'Class':<14} {'Count':>8}  {'% of total':>10}",
        "-" * 36,
    ]
    cells: list[list[str]] = []
    for cls, n, pct in rows_pct:
        lines.append(f"{cls:<14} {n:>8}  {pct:>9.2f}%")
        cells.append([cls, f"{n:,}", f"{pct:.2f}%"])
    lines.extend(["-" * 36, f"{'Total':<14} {total:>8}  {'100.00':>9}%"])
    write_txt_table(out_dir / "table_09_emotion_labeling_class_distribution.txt", lines)
    _plot_pub_table_png(
        "Predicted class distribution (separated-audio batch)",
        ["Class", "Count", "% of total"],
        cells,
        out_dir,
        "fig_15_emotion_labeling_class_distribution_table.png",
        figsize=(7.6, 5.6),
    )


def write_table_10_audiosep_raw_vs_proc_per_class_agreement(out_dir: Path) -> None:
    lines = [
        "Table 10 — Raw vs AudioSep-processed: per-class prediction agreement",
        "",
        "Caption: Mean pred_agrees (raw vs paired processed clip, same stems). Lower = more brittle.",
        "Source: _archive/top_level_notebooks/checking_extraction_real_vs_distribution.ipynb (committed stdout)",
        "Sort: ascending by pred_agrees (%).",
        "",
        f"{'Class':<14} {'pred_agrees (%)':>18}",
        "-" * 34,
    ]
    cells: list[list[str]] = []
    for cls, pct in _AUDIOSEP_PER_CLASS_AGREEMENT_ASC:
        lines.append(f"{cls:<14} {pct:>18.2f}")
        cells.append([cls, f"{pct:.2f}%"])
    write_txt_table(out_dir / "table_10_audiosep_raw_vs_proc_per_class_agreement.txt", lines)
    _plot_pub_table_png(
        "Per-class agreement: raw vs AudioSep-processed (NAYA paired run)",
        ["Class", "pred_agrees (%)"],
        cells,
        out_dir,
        "fig_16_audiosep_per_class_prediction_agreement_table.png",
        figsize=(6.9, 5.4),
    )


def plot_fig_01_suitability(rows: list[dict[str, Any]], out_dir: Path) -> None:
    vals = [extract_suitable_for_training(r) for r in rows]
    ctr: Counter[str] = Counter()
    for v in vals:
        if v is True:
            ctr["True"] += 1
        elif v is False:
            ctr["False"] += 1
        else:
            ctr["Missing / unspecified"] += 1
    order = ["True", "False"]
    if ctr["Missing / unspecified"]:
        order.append("Missing / unspecified")
    heights = [ctr[k] for k in order]
    colors = _bar_palette(len(order))
    _, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(order, heights, color=colors, edgecolor="0.25", linewidth=0.6)
    ax.set_ylabel("Number of snippets")
    ax.set_title("Suitable for training")
    ax.bar_label(bars, padding=3, fontsize=11)
    sns.despine(ax=ax)
    _save(out_dir, "fig_01_suitability.png")


def plot_fig_02_exclusion_reasons(df: pd.DataFrame, out_dir: Path) -> None:
    s = df["gpt_exclusion_reason"].dropna()
    s = s[s.astype(str).str.strip().ne("")]
    if s.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No non-null exclusion reasons", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_02_exclusion_reasons.png")
        return
    ctr = s.value_counts().sort_values(ascending=False)
    labels = ctr.index.tolist()
    counts = ctr.values.astype(int)

    def _ytick(lab: Any, maxlen: int = 76) -> str:
        t = str(lab)
        return t if len(t) <= maxlen else (t[: max(0, maxlen - 1)] + "…")

    y = np.arange(len(labels))
    _, ax = plt.subplots(figsize=(10, max(4.2, len(labels) * 0.38)))
    ax.barh(y, counts, color=_bar_palette(len(labels)), edgecolor="0.25", linewidth=0.5)
    ax.set_yticks(y, [_ytick(L) for L in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("GPT exclusion reasons (non-null gpt_exclusion_reason)")
    sns.despine(ax=ax)
    _save(out_dir, "fig_02_exclusion_reasons.png")


def plot_fig_03_breed_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    s = df["breed_extracted"].dropna()
    if s.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No breed guesses (breed_extracted)", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_03_breed_distribution.png")
        return
    ctr = s.value_counts().head(10)
    labels = ctr.index.tolist()
    counts = ctr.values.astype(int)

    def _ytick(lab: Any, maxlen: int = 40) -> str:
        t = str(lab)
        return t if len(t) <= maxlen else (t[: max(0, maxlen - 1)] + "…")

    y = np.arange(len(labels))
    _, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(y, counts, color=_bar_palette(len(labels)), edgecolor="0.25", linewidth=0.5)
    ax.set_yticks(y, [_ytick(L) for L in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Top 10 breed guesses (breed_extracted)")
    sns.despine(ax=ax)
    _save(out_dir, "fig_03_breed_distribution.png")


def plot_fig_04_environment(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["setting_stack"].notna() & df["_loc_col"].notna()]
    if sub.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No complete setting / location pairs", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_04_environment.png")
        return
    ct = pd.crosstab(sub["setting_stack"], sub["_loc_col"])
    preferred = [c for c in ("Indoor", "Outdoor") if c in ct.index]
    rest = sorted(c for c in ct.index if c not in preferred)
    ct = ct.reindex(preferred + rest)
    n_loc = ct.shape[1]
    cmap = _bar_palette(max(n_loc, 3))
    _, ax = plt.subplots(figsize=(8, 4))
    ct.plot(kind="bar", stacked=True, width=0.65, color=cmap[:n_loc], linewidth=0.4, ax=ax)
    ax.set_xlabel("Environment setting")
    ax.set_ylabel("Number of snippets")
    ax.set_title("Location type by setting (rare types → other)")
    ax.legend(title="Location type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    sns.despine(ax=ax)
    _save(out_dir, "fig_04_environment.png")


def plot_fig_05_behavior(df: pd.DataFrame, out_dir: Path) -> None:
    beh = df["gpt_primary_behavior"].dropna()
    if beh.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No primary behaviors recorded", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_05_behavior.png")
        return
    ctr = beh.value_counts()
    labels = ctr.index.tolist()
    counts = ctr.values
    y = np.arange(len(labels))
    _, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.32)))
    ax.barh(y, counts, color=_bar_palette(len(labels)), edgecolor="0.25", linewidth=0.5)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Primary behavior (GPT)")
    sns.despine(ax=ax)
    _save(out_dir, "fig_05_behavior.png")


def plot_fig_06_lighting_blur_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Lighting × blur counts — seaborn crest heatmap styling aligned with figs 10–12."""
    sub = df[df["gpt_lighting"].isin(LIGHTING_ORDER) & df["gpt_blur"].isin(BLUR_ORDER)]
    if sub.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No rows in fixed lighting×blur grid", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_06_lighting_blur_heatmap.png")
        return
    ct = pd.crosstab(sub["gpt_lighting"], sub["gpt_blur"])
    ct = ct.reindex(index=LIGHTING_ORDER, columns=BLUR_ORDER, fill_value=0).astype(int)
    nrows, ncols = int(ct.shape[0]), int(ct.shape[1])
    vmax = float(ct.to_numpy().max())
    if vmax <= 0:
        vmax = 1.0
    _, ax = plt.subplots(figsize=(max(6.0, ncols * 1.1), max(4.5, nrows * 0.42)))
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="crest",
        vmin=0.0,
        vmax=vmax,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Snippet count"},
        annot_kws={"fontsize": 10},
    )
    ax.set_title("Lighting × blur (snippet counts)")
    ax.set_xlabel("Blur (gpt_blur)")
    ax.set_ylabel("Lighting (gpt_lighting)")
    _save(out_dir, "fig_06_lighting_blur_heatmap.png")


def plot_fig_07_face_visibility(rows: list[dict[str, Any]], out_dir: Path) -> None:
    vals = [extract_face_visible(r) for r in rows]
    ctr: Counter[str] = Counter()
    for v in vals:
        if v is True:
            ctr["True"] += 1
        elif v is False:
            ctr["False"] += 1
        else:
            ctr["Missing / unspecified"] += 1
    order = ["True", "False"]
    if ctr["Missing / unspecified"]:
        order.append("Missing / unspecified")
    heights = [ctr[k] for k in order]
    colors = _bar_palette(len(order))
    _, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(order, heights, color=colors, edgecolor="0.25", linewidth=0.6)
    ax.set_ylabel("Number of snippets")
    ax.set_title("Face clearly visible")
    ax.bar_label(bars, padding=3, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    sns.despine(ax=ax)
    _save(out_dir, "fig_07_face_visibility.png")


def plot_fig_08_behavior_setting(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["setting_io"].notna() & df["_beh_col"].notna()].copy()
    if sub.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No Indoor/Outdoor behavior rows", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_08_behavior_by_setting.png")
        return
    wide = pd.crosstab(sub["_beh_col"], sub["setting_io"])
    for c in ("Indoor", "Outdoor"):
        if c not in wide.columns:
            wide[c] = 0
    wide = wide[["Indoor", "Outdoor"]]
    wide = wide.loc[wide.sum(axis=1).sort_values(ascending=False).index]
    beh_order = wide.index.tolist()
    x = np.arange(len(beh_order))
    w = 0.36
    duo = _bar_palette(2)
    _, ax = plt.subplots(figsize=(max(8, len(beh_order) * 0.45), 4))
    ax.bar(x - w / 2, wide["Indoor"], width=w, label="Indoor", color=duo[0], edgecolor="0.25", linewidth=0.45)
    ax.bar(x + w / 2, wide["Outdoor"], width=w, label="Outdoor", color=duo[1], edgecolor="0.25", linewidth=0.45)
    ax.set_xticks(x, beh_order, rotation=35, ha="right")
    ax.set_ylabel("Number of snippets")
    ax.set_title("Primary behavior by Indoor vs Outdoor")
    ax.legend(title="Setting")
    sns.despine(ax=ax)
    _save(out_dir, "fig_08_behavior_by_setting.png")


def plot_fig_09_audio10_labels(df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
    s10 = df["audio_label_10"].dropna()
    c10 = s10.astype(str).value_counts().reindex(LABEL10_ORDER).fillna(0).astype(int)
    c10 = c10.sort_values(ascending=False)
    labels_sorted = c10.index.tolist()
    palette10 = _bar_palette(len(LABEL10_ORDER))
    color_by_label = {lab: palette10[i] for i, lab in enumerate(LABEL10_ORDER)}
    colors10 = [color_by_label[l] for l in labels_sorted]
    bars1 = ax1.bar(labels_sorted, c10.values, color=colors10, edgecolor="0.25", linewidth=0.5)
    ax1.set_title("Audio ten-class label (audio_label_10)")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=55)
    ax1.bar_label(bars1, fontsize=8, rotation=0)
    sns.despine(ax=ax1)

    s2 = df["audio_label_binary"].dropna()
    c2 = s2.astype(str).value_counts().reindex(BINARY_ORDER).fillna(0).astype(int)
    cols2 = _bar_palette(len(BINARY_ORDER))
    bars2 = ax2.bar(BINARY_ORDER, c2.values, color=cols2, edgecolor="0.25", linewidth=0.5)
    ax2.set_title("Audio binary label (audio_label_binary)")
    ax2.set_ylabel("Count")
    ax2.bar_label(bars2, fontsize=10)
    sns.despine(ax=ax2)
    plt.tight_layout()
    _save(out_dir, "fig_09_audio10_labels_twopanel.png")


def _plot_row_norm_heatmap(
    ct: pd.DataFrame,
    out_dir: Path,
    fname: str,
    title: str,
    xlab: str,
    ylab: str,
) -> None:
    prop_plot, row_sums = row_norm_matrix(ct)
    ann = heatmap_annotations_row_norm(ct, row_sums)
    _, ax = plt.subplots(figsize=(max(6, ct.shape[1] * 1.1), max(4.5, ct.shape[0] * 0.42)))
    sns.heatmap(
        prop_plot,
        annot=ann,
        fmt="",
        cmap="crest",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Row-normalized proportion"},
    )
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    _save(out_dir, fname)


def plot_fig_10_audio10_setting(
    df: pd.DataFrame, out_dir: Path, n_missing_audio10: int
) -> None:
    sub = df[df["audio_label_10"].notna()].copy()
    ct = pd.crosstab(sub["audio_label_10"], sub["setting_tri"])
    cols = ["Indoor", "Outdoor", "Mixed/Unclear"]
    for c in cols:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct.reindex(index=LABEL10_ORDER, columns=cols, fill_value=0).astype(int)
    _plot_row_norm_heatmap(
        ct,
        out_dir,
        "fig_10_audio10_setting_rownorm_heatmap.png",
        f"Ten-class × setting (row-normalized); n_missing_audio10={n_missing_audio10}",
        "Setting",
        "Ten-class label (audio_label_10)",
    )


def _crosstab_audio10_gpt_location(df: pd.DataFrame, location_top_k: int) -> pd.DataFrame:
    """Audio ten-class × GPT location (top-k + other); same column order as fig 11 / 20."""
    sub = df[df["audio_label_10"].notna() & df["gpt_location_type"].notna()].copy()
    sub["_loc_top"] = top_k_plus_other(sub["gpt_location_type"], location_top_k)
    ct = pd.crosstab(sub["audio_label_10"], sub["_loc_top"]).reindex(index=LABEL10_ORDER, fill_value=0).astype(int)
    col_order = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    if "other" in col_order:
        col_order.remove("other")
        col_order.append("other")
    return ct.reindex(columns=col_order, fill_value=0)


def plot_fig_11_audio10_location(
    df: pd.DataFrame,
    out_dir: Path,
    n_missing_audio10: int,
    location_top_k: int,
) -> None:
    ct = _crosstab_audio10_gpt_location(df, location_top_k)
    _plot_row_norm_heatmap(
        ct,
        out_dir,
        "fig_11_audio10_location_rownorm_heatmap.png",
        f"Ten-class × location (row-normalized); n_missing_audio10={n_missing_audio10}; top-{location_top_k}+other",
        "Location type",
        "Ten-class label (audio_label_10)",
    )


def plot_fig_20_audio10_location_counts(
    df: pd.DataFrame,
    out_dir: Path,
    n_missing_audio10: int,
    location_top_k: int,
) -> None:
    ct = _crosstab_audio10_gpt_location(df, location_top_k)
    vmax = float(ct.to_numpy().max()) if ct.size and ct.to_numpy().max() > 0 else 1.0
    _, ax = plt.subplots(figsize=(max(6, ct.shape[1] * 1.1), max(4.5, ct.shape[0] * 0.42)))
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="crest",
        vmin=0.0,
        vmax=vmax,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Snippet count"},
        annot_kws={"fontsize": 10},
    )
    ax.set_title(
        f"Ten-class × location (counts); n_missing_audio10={n_missing_audio10}; top-{location_top_k}+other"
    )
    ax.set_xlabel("Location type")
    ax.set_ylabel("Ten-class label (audio_label_10)")
    _save(out_dir, "fig_20_audio10_location_counts_heatmap.png")


def plot_fig_12_audio10_behavior(
    df: pd.DataFrame,
    out_dir: Path,
    n_missing_audio10: int,
    behavior_min_count: int,
) -> None:
    sub = df[df["audio_label_10"].notna() & df["gpt_primary_behavior"].notna()].copy()
    sub["_bt"] = collapse_behaviors_for_emotion_heatmap(
        sub["gpt_primary_behavior"],
        behavior_min_count,
        top_k=8,
    )
    ct = pd.crosstab(sub["audio_label_10"], sub["_bt"]).reindex(index=LABEL10_ORDER, fill_value=0).astype(int)
    col_order = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    if "other" in col_order:
        col_order.remove("other")
        col_order.append("other")
    ct = ct.reindex(columns=col_order, fill_value=0)
    _plot_row_norm_heatmap(
        ct,
        out_dir,
        "fig_12_audio10_behavior_rownorm_heatmap.png",
        f"Ten-class × behavior (row-normalized); n_missing_audio10={n_missing_audio10}; min_count={behavior_min_count}",
        "Primary behavior (collapsed)",
        "Ten-class label (audio_label_10)",
    )


def _crosstab_behavior_setting_tri(df: pd.DataFrame, behavior_min_count: int) -> pd.DataFrame:
    """Collapsed primary behavior × setting_tri; row order matches fig 12-style (descending mass, other last)."""
    sub = df[df["gpt_primary_behavior"].notna() & df["setting_tri"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_bt"] = collapse_behaviors_for_emotion_heatmap(
        sub["gpt_primary_behavior"],
        behavior_min_count,
        top_k=8,
    )
    ct = pd.crosstab(sub["_bt"], sub["setting_tri"]).astype(int)
    cols = ["Indoor", "Outdoor", "Mixed/Unclear"]
    for c in cols:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[cols]
    row_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    if "other" in row_order:
        row_order.remove("other")
        row_order.append("other")
    return ct.reindex(index=row_order, fill_value=0).astype(int)


def plot_fig_21_behavior_setting_rownorm(
    df: pd.DataFrame,
    out_dir: Path,
    behavior_min_count: int,
) -> None:
    ct = _crosstab_behavior_setting_tri(df, behavior_min_count)
    if ct.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No rows with primary behavior and setting", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_21_behavior_setting_rownorm_heatmap.png")
        return
    _plot_row_norm_heatmap(
        ct,
        out_dir,
        "fig_21_behavior_setting_rownorm_heatmap.png",
        f"Behavior × setting (row-normalized); min_count={behavior_min_count}; top-8+other",
        "Setting",
        "Primary behavior (collapsed)",
    )


def plot_fig_22_behavior_setting_counts(
    df: pd.DataFrame,
    out_dir: Path,
    behavior_min_count: int,
) -> None:
    ct = _crosstab_behavior_setting_tri(df, behavior_min_count)
    if ct.empty:
        _, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No rows with primary behavior and setting", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_22_behavior_setting_counts_heatmap.png")
        return
    vmax = float(ct.to_numpy().max()) if ct.size and ct.to_numpy().max() > 0 else 1.0
    _, ax = plt.subplots(figsize=(max(6, ct.shape[1] * 1.1), max(4.5, ct.shape[0] * 0.42)))
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="crest",
        vmin=0.0,
        vmax=vmax,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Snippet count"},
        annot_kws={"fontsize": 10},
    )
    ax.set_title(f"Behavior × setting (counts); min_count={behavior_min_count}; top-8+other")
    ax.set_xlabel("Setting")
    ax.set_ylabel("Primary behavior (collapsed)")
    _save(out_dir, "fig_22_behavior_setting_counts_heatmap.png")


def plot_fig_13_pipeline_funnel(plot_rows: list, out_dir: Path) -> None:
    """Horizontal survivor funnel — top bar = earliest measured stage in this run."""
    if not plot_rows:
        _, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No funnel stages with counts — check manifests / config.yaml", ha="center", va="center")
        ax.axis("off")
        _save(out_dir, "fig_13_pipeline_funnel.png")
        return
    stages = [r.stage for r in plot_rows]
    counts = [int(r.count) for r in plot_rows if r.count is not None]
    y = np.arange(len(stages))
    _, ax = plt.subplots(figsize=(10, max(4.0, len(stages) * 0.5)))
    ax.barh(y, counts, color=_bar_palette(len(stages)), edgecolor="0.25", linewidth=0.5)
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xlabel("Surviving snippet count")
    ax.set_title("Pipeline funnel (per-stage survivor counts)")
    sns.despine(ax=ax)
    _save(out_dir, "fig_13_pipeline_funnel.png")


def plot_fig_14_platform_suitable_stack(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Stacked vertical bars: YouTube vs TikTok, suitable vs not (vs missing if any)."""
    agg, n_skip = aggregate_platform_suitability(rows)
    plat_order = [("YouTube", "youtube"), ("TikTok", "tiktok")]
    labels = [po[0] for po in plat_order]
    x = np.arange(len(labels))

    suites = np.array([agg[k][0] for _, k in plat_order], dtype=float)
    nos = np.array([agg[k][1] for _, k in plat_order], dtype=float)
    miss = np.array([agg[k][2] for _, k in plat_order], dtype=float)
    totals = suites + nos + miss

    use_miss = bool(np.any(miss > 0))
    pal_2 = sns.color_palette("crest", n_colors=2)
    pal_3 = sns.color_palette("crest", n_colors=3)

    _, ax = plt.subplots(figsize=(6.2, 4.6))

    stacks: list[tuple[str, np.ndarray, tuple[float, float, float]]]
    if use_miss:
        stacks = [
            ("Suitable", suites, pal_3[2]),
            ("Not suitable", nos, pal_3[0]),
            ("Suitability missing", miss, pal_3[1]),
        ]
    else:
        stacks = [
            ("Suitable", suites, pal_2[1]),
            ("Not suitable", nos, pal_2[0]),
        ]

    bottoms = np.zeros_like(x, dtype=float)
    for title, vals, color in stacks:
        ax.bar(x, vals, bottom=bottoms, label=title, color=color, edgecolor="0.25", linewidth=0.55)
        for xi, yi, hi in zip(x, bottoms, vals):
            if hi <= 0:
                continue
            tot_i = totals[int(xi)]
            pct = 100.0 * hi / float(tot_i) if tot_i else 0.0
            y_text = yi + hi / 2.0
            ax.text(float(xi), y_text, f"{int(hi)}\n({pct:.1f}%)", ha="center", va="center", fontsize=10)
        bottoms = bottoms + vals

    ax.set_xticks(x, labels)
    ax.set_ylabel("Snippet count")
    ax.set_title("Suitable vs not suitable by platform")
    ax.legend(loc="upper right")
    if n_skip:
        fig = ax.figure
        fig.text(
            0.5,
            0.02,
            f"{n_skip} manifest rows omitted (could not resolve platform to YouTube/TikTok).",
            ha="center",
            fontsize=10,
        )
        plt.subplots_adjust(bottom=0.16)
    sns.despine(ax=ax)
    _save(out_dir, "fig_14_platform_suitable_stack.png")


def write_table_platform_suitability(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    agg, n_skip = aggregate_platform_suitability(rows)
    plat_order = [("YouTube", "youtube"), ("TikTok", "tiktok")]
    lines: list[str] = [
        "Platform × suitability for training",
        "=" * 90,
        f"Manifest rows (total): {len(rows)}",
        (
            "Logical fields: top-level `platform` (fallback `source` for URL shorthand) + same "
            "suitability extraction as fig_01 / table_01."
        ),
        "Platforms shown: YouTube and TikTok only.",
        "",
    ]
    w_pl, w_a, w_b, w_c, w_tot, w_ps = 12, 10, 12, 8, 8, 10
    hdr = (
        f"{'Platform':<{w_pl}} {'Suitable':>{w_a}} {'Not suitable':>{w_b}} {'Missing':>{w_c}} "
        f"{'Total':>{w_tot}} {'% suit':>{w_ps}}"
    )
    lines.append(hdr)
    sep_w = w_pl + w_a + w_b + w_c + w_tot + w_ps + 5
    lines.append("-" * sep_w)
    for label, key in plat_order:
        su, nf, mz = agg[key]
        tot = su + nf + mz
        ps = (100.0 * su / float(tot)) if tot else 0.0
        lines.append(
            f"{label:<{w_pl}} {su:>{w_a}} {nf:>{w_b}} {mz:>{w_c}} {tot:>{w_tot}} {ps:>{w_ps}.1f}%"
        )
    lines.append("")
    lines.append("% suit = Suitable / Total (within each row), Total = Suitable + Not suitable + Missing.")
    if n_skip:
        lines.append(
            f"Rows excluded from this table (could not resolve platform to youtube/tiktok): {n_skip}."
        )
    path = out_dir / "table_07_platform_suitability.txt"
    write_txt_table(path, lines)
    return path


def write_table_suitability(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    vals = [extract_suitable_for_training(r) for r in rows]
    ctr: Counter[str] = Counter()
    for v in vals:
        if v is True:
            ctr["True"] += 1
        elif v is False:
            ctr["False"] += 1
        else:
            ctr["Missing / unspecified"] += 1
    order = ["True", "False"]
    if ctr["Missing / unspecified"]:
        order.append("Missing / unspecified")
    pairs = [(lab, str(int(ctr[lab]))) for lab in order]
    body = [
        "Suitable for training (same logic as fig_01)",
        "=" * 60,
        f"Manifest rows (total): {len(rows)}",
        "Fields: dataset_flags.suitable_for_training, suitable_for_training, gpt_suitable_for_training",
        "",
    ]
    body += two_column_table(pairs, "category", "count")
    path = out_dir / "table_01_suitability.txt"
    write_txt_table(path, body)
    return path


def write_table_exclusion_reasons(df: pd.DataFrame, out_dir: Path) -> Path:
    s = df["gpt_exclusion_reason"].dropna()
    s = s[s.astype(str).str.strip().ne("")]
    ctr = s.value_counts().sort_values(ascending=False)
    rows = [(lab, str(int(cnt))) for lab, cnt in ctr.items()]
    body = [
        "Exclusion reason counts (same data as fig_02; non-null gpt_exclusion_reason)",
        "=" * 60,
        "",
    ]
    body += two_column_table(rows, "reason", "count")
    path = out_dir / "table_02_exclusion_reasons.txt"
    write_txt_table(path, body)
    return path


def write_table_top10_breeds(df: pd.DataFrame, out_dir: Path) -> Path:
    s = df["breed_extracted"].dropna()
    top = s.value_counts().head(10)
    rows = [(lab, str(int(cnt))) for lab, cnt in top.items()]
    body = [
        "Top 10 breed guesses (same as fig_03; breed_extracted / GPT)",
        "=" * 60,
        "",
    ]
    body += two_column_table(rows, "breed_guess", "count")
    path = out_dir / "table_03_top10_breeds.txt"
    write_txt_table(path, body)
    return path


def write_table_face_by_audio_label_10(
    df: pd.DataFrame,
    out_dir: Path,
    n_missing_audio10: int,
) -> Path:
    sub = df[df["audio_label_10"].notna()].copy()
    lines = [
        "Face visibility by audio ten-class label (gpt_face_clearly_visible)",
        "=" * 60,
        "(Table uses rows with non-null audio_label_10 only.)",
        f"n_missing_audio10 (rows without audio ten-way label): {n_missing_audio10}",
        "",
        f"{'audio_label':<26} {'face=True':>10} {'face=False':>10} {'missing':>10} {'total':>10}",
        "-" * 66,
    ]
    for lab in LABEL10_ORDER:
        part = sub[sub["audio_label_10"].astype(str) == lab]
        fv = part["gpt_face_clearly_visible"]
        ft = int((fv == True).sum())  # noqa: E712
        ff = int((fv == False).sum())  # noqa: E712
        ms = int(fv.isna().sum())
        tot = len(part)
        lines.append(f"{str(lab):<26} {ft:>10} {ff:>10} {ms:>10} {tot:>10}")
    path = out_dir / "table_04_face_by_audio_label_10.txt"
    write_txt_table(path, lines)
    return path


def write_table_lighting_blur_counts(df: pd.DataFrame, out_dir: Path) -> Path:
    """Cross-tab snippet counts for gpt_lighting × gpt_blur; lighting sorted by row total descending."""

    def _pct(num: float, denom: float) -> str:
        if denom <= 0:
            return "—"
        return f"{100.0 * num / denom:.1f}%"

    n_manifest = len(df)
    sub = df[df["gpt_lighting"].isin(LIGHTING_ORDER) & df["gpt_blur"].isin(BLUR_ORDER)]
    lines = [
        "Lighting × blur — snippet counts and percentages",
        "=" * 88,
        "Same filter as fig_06_lighting_blur_heatmap: lighting and blur must be in fixed vocabularies.",
        "Lighting rows sorted by row total (blur none + mild + severe), descending.",
        f"gpt_blur column order: {', '.join(BLUR_ORDER)}.",
        "",
    ]
    path = out_dir / "table_08_lighting_blur_counts.txt"
    if sub.empty:
        lines.append("(No rows matched the lighting/blur vocabularies.)")
        lines.extend(["", f"Manifest rows (total): {n_manifest}", "Snippet rows in table: 0"])
        write_txt_table(path, lines)
        return path

    ct = pd.crosstab(sub["gpt_lighting"], sub["gpt_blur"])
    ct = ct.reindex(index=LIGHTING_ORDER, columns=BLUR_ORDER, fill_value=0).astype(int)
    light_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    ct_ord = ct.reindex(light_order)

    w_lab = 16
    w_cell = 9
    w_tot = 10
    b1, b2, b3 = BLUR_ORDER[0], BLUR_ORDER[1], BLUR_ORDER[2]
    hdr = (
        f"{'gpt_lighting':<{w_lab}} "
        f"{b1:>{w_cell}} {b2:>{w_cell}} {b3:>{w_cell}} "
        f"{'row_total':>{w_tot}}"
    )
    lines.append("Counts")
    lines.append(hdr)
    lines.append("-" * (w_lab + 3 * w_cell + w_tot + 4))
    for lit in light_order:
        r = ct_ord.loc[lit]
        c1, c2, c3 = int(r[b1]), int(r[b2]), int(r[b3])
        rt = c1 + c2 + c3
        lines.append(
            f"{str(lit):<{w_lab}} {c1:>{w_cell}} {c2:>{w_cell}} {c3:>{w_cell}} {rt:>{w_tot}}"
        )
    t1 = int(ct_ord[b1].sum())
    t2 = int(ct_ord[b2].sum())
    t3 = int(ct_ord[b3].sum())
    gt = t1 + t2 + t3
    lines.append("-" * (w_lab + 3 * w_cell + w_tot + 4))
    lines.append(
        f"{'column_total':<{w_lab}} {t1:>{w_cell}} {t2:>{w_cell}} {t3:>{w_cell}} {gt:>{w_tot}}"
    )

    w_pc = 10
    lines.extend(
        [
            "",
            "Within-row blur share (% of snippets in that lighting row; columns sum to 100%).",
            f"{'gpt_lighting':<{w_lab}} "
            f"{b1 + ' %':>{w_pc}} {b2 + ' %':>{w_pc}} {b3 + ' %':>{w_pc}}",
            "-" * (w_lab + 3 * w_pc + 2),
        ]
    )
    for lit in light_order:
        r = ct_ord.loc[lit]
        c1, c2, c3 = int(r[b1]), int(r[b2]), int(r[b3])
        rt = c1 + c2 + c3
        lines.append(
            f"{str(lit):<{w_lab}} "
            f"{_pct(c1, rt):>{w_pc}} {_pct(c2, rt):>{w_pc}} {_pct(c3, rt):>{w_pc}}"
        )
    lines.extend(
        [
            "",
            "Marginal blur share (% of all snippets in table, same column totals as above).",
            f"{b1:<{w_lab}} {_pct(t1, float(gt)):>{w_pc}}",
            f"{b2:<{w_lab}} {_pct(t2, float(gt)):>{w_pc}}",
            f"{b3:<{w_lab}} {_pct(t3, float(gt)):>{w_pc}}",
            "",
            (
                "Grand total snippets in counts table: "
                f"{gt} (manifest rows {n_manifest}; rows excluded when lighting/blur missing or "
                "outside vocabularies)."
            ),
        ]
    )
    write_txt_table(path, lines)
    return path


def validate_outputs(out_dir: Path, include_report: bool, report_path: Path | None) -> int:
    bad = False
    for name in EXPECTED_TABLES:
        p = out_dir / name
        ok = p.is_file()
        sz = p.stat().st_size if ok else 0
        status = "OK" if ok and sz > 0 else "MISSING"
        if not ok or sz == 0:
            bad = True
        print(f"[{status}] {p} ({sz} bytes)")
    for name in EXPECTED_PNGS:
        p = out_dir / name
        ok = p.is_file()
        sz = p.stat().st_size if ok else 0
        status = "OK" if ok and sz > 0 else "MISSING_OR_EMPTY"
        if not ok or sz == 0:
            bad = True
        print(f"[{status}] {p} ({sz} bytes)")
    if include_report and report_path is not None:
        ok = report_path.is_file()
        sz = report_path.stat().st_size if ok else 0
        status = "OK" if ok and sz > 0 else "MISSING_OR_EMPTY"
        if not ok or sz == 0:
            bad = True
        print(f"[{status}] {report_path} ({sz} bytes)")
    return 1 if bad else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thesis tables and plots from dataset manifest.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--location-other-min", type=int, default=50)
    p.add_argument("--behavior-min-count", type=int, default=10)
    p.add_argument("--location-top-k", type=int, default=6, help="Top location types before folding to other")
    p.add_argument(
        "--emotion-report",
        type=Path,
        default=None,
        help=(
            "Path to emotion_labeling_report.txt for table 9 / fig 15 "
            f"(default: {DEFAULT_EMOTION_LABELING_REPORT})"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    apply_publication_style()
    rows = load_jsonl(manifest)
    df = build_thesis_dataframe(rows)
    n_missing_label5 = int(df["final_label_5"].isna().sum())
    n_missing_audio10 = int(df["audio_label_10"].isna().sum())
    df = attach_derived_columns(df, args.location_other_min, args.behavior_min_count)

    cfg = load_dataset_config()

    write_table_suitability(rows, out_dir)
    write_table_exclusion_reasons(df, out_dir)
    write_table_top10_breeds(df, out_dir)
    write_table_face_by_audio_label_10(df, out_dir, n_missing_audio10)
    write_table_lighting_blur_counts(df, out_dir)
    write_table_09_emotion_labeling_class_distribution(out_dir, args.emotion_report)
    write_table_10_audiosep_raw_vs_proc_per_class_agreement(out_dir)
    write_metrics_paper_tables(out_dir)
    write_cost_table(out_dir / "table_05_gpt_costs.txt", cfg)
    _, cost_notes = build_cost_table_txt(cfg)

    funnel_rows = build_pipeline_funnel_rows(cfg, final_rows=rows)
    write_pipeline_funnel_table(
        out_dir / "table_06_pipeline_funnel.txt", funnel_rows, final_rows=rows
    )

    write_table_platform_suitability(rows, out_dir)

    plot_fig_01_suitability(rows, out_dir)
    plot_fig_02_exclusion_reasons(df, out_dir)
    plot_fig_03_breed_distribution(df, out_dir)
    plot_fig_04_environment(df, out_dir)
    plot_fig_05_behavior(df, out_dir)
    plot_fig_06_lighting_blur_heatmap(df, out_dir)
    plot_fig_07_face_visibility(rows, out_dir)
    plot_fig_08_behavior_setting(df, out_dir)
    plot_fig_09_audio10_labels(df, out_dir)
    plot_fig_10_audio10_setting(df, out_dir, n_missing_audio10)
    plot_fig_11_audio10_location(df, out_dir, n_missing_audio10, args.location_top_k)
    plot_fig_20_audio10_location_counts(df, out_dir, n_missing_audio10, args.location_top_k)
    plot_fig_12_audio10_behavior(df, out_dir, n_missing_audio10, args.behavior_min_count)
    plot_fig_21_behavior_setting_rownorm(df, out_dir, args.behavior_min_count)
    plot_fig_22_behavior_setting_counts(df, out_dir, args.behavior_min_count)
    plot_fig_13_pipeline_funnel(funnel_rows_for_plot(funnel_rows), out_dir)
    plot_fig_14_platform_suitable_stack(rows, out_dir)

    report_path: Path | None = None
    if not args.no_report:
        if args.report is not None:
            rp = Path(args.report).expanduser().resolve()
            if rp.is_dir():
                rp = rp / REPORT_FILENAME
            report_path = rp
        else:
            report_path = (out_dir / REPORT_FILENAME).resolve()
        text = build_figure_report(
            manifest=manifest,
            df=df,
            rows=rows,
            n_missing_label5=n_missing_label5,
            n_missing_audio10=n_missing_audio10,
            n_rows=len(rows),
            location_other_min=args.location_other_min,
            behavior_min_count=args.behavior_min_count,
            location_top_k=args.location_top_k,
            cost_footer_notes=cost_notes,
            funnel_rows=funnel_rows,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
        print(f"Report:   {report_path}")

    code = validate_outputs(out_dir, include_report=not args.no_report, report_path=report_path)
    print(
        f"Done. Rows={len(rows)} n_missing_label5={n_missing_label5} "
        f"n_missing_audio10={n_missing_audio10} out_dir={out_dir}"
    )
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
