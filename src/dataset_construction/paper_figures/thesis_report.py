"""Build figure_report.txt matching the thesis schema."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from thesis_extract import (
    BLUR_ORDER,
    LABEL10_ORDER,
    LIGHTING_ORDER,
    aggregate_platform_suitability,
    collapse_rare_categories,
    extract_suitable_for_training,
    location_collapsed,
)
from pipeline_funnel import FunnelRow, funnel_rows_for_plot


def _sep(title: str, width: int = 78) -> list[str]:
    line = "-" * width
    return ["", line, title, line, ""]


def build_figure_report(
    *,
    manifest: Path,
    df: pd.DataFrame,
    rows: list[dict[str, Any]],
    n_missing_label5: int,
    n_missing_audio10: int,
    n_rows: int,
    location_other_min: int,
    behavior_min_count: int,
    location_top_k: int,
    cost_footer_notes: list[str],
    funnel_rows: list[FunnelRow],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks: list[str] = []

    preamble = [
        "Bachelor thesis — figure report (auto-generated)",
        "=" * 78,
        "",
        f"Manifest:              {manifest}",
        f"Rows read:             {n_rows}",
        f"n_missing_label5:      {n_missing_label5} "
        "(null final_label_5 on manifest; reference only for merged five-class target)",
        f"n_missing_audio10:     {n_missing_audio10} "
        "(rows with null audio_label_10; excluded from audio ten-class plots and table 4 face)",
        f"Generated:               {ts}",
        "",
        "CLI thresholds",
        "-" * 78,
        f"  location_other_min:    {location_other_min}",
        f"  behavior_min_count:    {behavior_min_count}",
        f"  location_top_k:        {location_top_k}",
        "",
        "Shared methodology",
        "-" * 78,
        "Outputs from generate_plots.py: PNG (300 dpi, bbox tight), seaborn-whitegrid;",
        "flat gpt_* fields from merged manifest unless nested under gpt_description.",
        "Figures 09–12: audio ten-class (audio_label_10); table 4 is face × audio class.",
        "Figures 13–14: pipeline funnel + platform × suitability stacks; tables 6–8 (includes lighting×blur grid).",
        "",
    ]
    blocks.extend(preamble)

    blocks.extend(_sep("Figure 1 — fig_01_suitability.png"))
    blocks.extend([
        "In-figure title: Suitable for training",
        "",
        "Description:",
        "  Bar chart counts suitable vs not suitable for training (GPT / merged flags).",
        "",
        "Logical fields used:",
        "  dataset_flags.suitable_for_training, suitable_for_training, gpt_suitable_for_training",
        "",
        "Exclusions applied:",
        "  None (all manifest rows). Boolean coercion failures -> 'Missing / unspecified' if bar shown.",
        "",
        "Key counts:",
    ])
    vals = [extract_suitable_for_training(r) for r in rows]
    ctr: Counter[str] = Counter()
    for v in vals:
        if v is True:
            ctr["True"] += 1
        elif v is False:
            ctr["False"] += 1
        else:
            ctr["Missing / unspecified"] += 1
    for k in ("True", "False", "Missing / unspecified"):
        if k in ctr:
            blocks.append(f"  {k}: {ctr[k]}")

    blocks.extend(_sep("Table 1 — table_01_suitability.txt"))
    blocks.extend([
        "Caption: Suitable for training counts",
        "",
        "Same categories as fig_01 (two-column category, count).",
        "Logical fields: dataset_flags.suitable_for_training, suitable_for_training, gpt_suitable_for_training",
        "Exclusions: none (all manifest rows).",
    ])
    for k in ("True", "False", "Missing / unspecified"):
        if k in ctr:
            blocks.append(f"  {k}: {ctr[k]}")

    s_excl = df["gpt_exclusion_reason"].dropna()
    s_excl = s_excl[s_excl.astype(str).str.strip().ne("")]
    blocks.extend(_sep("Figure 2 — fig_02_exclusion_reasons.png"))
    blocks.extend([
        "In-figure title: GPT exclusion reasons (non-null gpt_exclusion_reason)",
        "Logical fields: gpt_exclusion_reason",
        "Pairs with table_02_exclusion_reasons.txt (horizontal bars, crest via _bar_palette).",
        "Exclusions applied: null / empty strings omitted.",
        f"Key counts: non-null reason rows {len(s_excl)}; distinct reasons {s_excl.nunique()}.",
    ])

    blocks.extend(_sep("Table 2 — table_02_exclusion_reasons.txt"))
    s = df["gpt_exclusion_reason"].dropna()
    s = s[s.astype(str).str.strip().ne("")]
    blocks.extend([
        "Caption: Exclusion reason counts",
        "",
        "Description: Two-column table (reason, count), non-null gpt_exclusion_reason only.",
        "",
        "Logical fields used: gpt_exclusion_reason",
        "",
        "Exclusions applied:",
        "  Null / empty exclusion strings omitted.",
        "",
        "Key counts:",
        f"  Non-null reasons: {len(s)}",
        f"  Distinct reasons: {s.nunique()}",
    ])

    br = df["breed_extracted"].dropna()
    blocks.extend(_sep("Figure 3 — fig_03_breed_distribution.png"))
    blocks.extend([
        "In-figure title: Top 10 breed guesses (breed_extracted)",
        "",
        "Horizontal bar chart (crest); pairs with table_03_top10_breeds.txt.",
        "",
        "Logical fields used: breed_extracted (nested cats / gpt_breed_guess)",
        "",
        "Exclusions applied:",
        "  Rows without breed guess omitted.",
        "",
        f"Key counts: rows with breed guess {len(br)}",
    ])

    blocks.extend(_sep("Table 3 — table_03_top10_breeds.txt"))
    blocks.extend([
        "Caption: Top 10 breed guesses",
        "",
        "Logical fields used: nested cats / gpt_breed_guess (breed_extracted)",
        "",
        "Exclusions applied:",
        "  Rows without breed guess omitted from ranking.",
        "",
        f"Key counts: rows with breed {len(br)}",
    ])

    pairs = df[df["setting_stack"].notna() & df["_loc_col"].notna()]
    blocks.extend(_sep("Figure 4 — fig_04_environment.png"))
    blocks.extend([
        "In-figure title: Location type by setting (stacked)",
        "",
        "Description: Stacked bars; rare location types collapsed to 'other' globally.",
        "",
        "Logical fields: gpt_setting (setting_stack), gpt_location_type (_loc_col)",
        "",
        "Exclusions applied:",
        "  Rows missing setting or location after collapse omitted from chart.",
        f"  location_other_min = {location_other_min}",
        "",
        f"Key counts: chart rows {len(pairs)}",
    ])

    blocks.extend(_sep("Figure 5 — fig_05_behavior.png"))
    nb = df["gpt_primary_behavior"].notna().sum()
    blocks.extend([
        "In-figure title: Primary behavior (GPT)",
        "Logical fields: gpt_primary_behavior",
        "Exclusions: non-null behavior only",
        f"Key counts: {int(nb)} rows with behavior",
    ])

    vq = df[df["gpt_lighting"].isin(LIGHTING_ORDER) & df["gpt_blur"].isin(BLUR_ORDER)]
    blocks.extend(_sep("Figure 6 — fig_06_lighting_blur_heatmap.png"))
    blocks.extend([
        "In-figure title: Lighting × blur (snippet counts)",
        "Seaborn crest heatmap — same linewidth / sizing pattern as Figures 10–12 (no despine strip).",
        "Logical fields: gpt_lighting, gpt_blur",
        "Exclusions: only rows in fixed lighting/blur vocabularies",
        f"Key counts: {len(vq)} rows",
        "",
        "Table: table_08_lighting_blur_counts.txt — same cross-tab as plain text (lighting rows sorted by row total).",
    ])

    blocks.extend(_sep("Figure 7 — fig_07_face_visibility.png"))
    blocks.extend([
        "In-figure title: Face clearly visible",
        "Logical fields: gpt_face_clearly_visible",
        "Exclusions: none (all rows)",
    ])

    sub_io = df[df["setting_io"].notna() & df["_beh_col"].notna()]
    blocks.extend(_sep("Figure 8 — fig_08_behavior_by_setting.png"))
    blocks.extend([
        "In-figure title: Behavior × setting (Indoor vs Outdoor)",
        "Logical fields: gpt_primary_behavior (_beh_col), gpt_setting (setting_io)",
        "Exclusions:",
        "  Mixed/unclear/null setting excluded (Indoor/Outdoor only).",
        f"  behavior_min_count = {behavior_min_count} (rare behaviors -> other).",
        f"Key counts: rows in chart {len(sub_io)}",
    ])

    sub_a10 = df[df["audio_label_10"].notna()]
    blocks.extend(_sep("Figure 9 — fig_09_audio10_labels_twopanel.png"))
    blocks.extend([
        "In-figure title: Audio ten-class + audio binary label distribution",
        "Logical fields: audio_label_10, audio_label_binary",
        f"Exclusions: n_missing_audio10 = {n_missing_audio10}; panels use non-null values only.",
        f"  Rows with non-null audio_label_10: {len(sub_a10)}",
        "  Left: crest palette, bars sorted by descending count; colors keyed to canonical class order.",
        f"    Canonical reference order: {', '.join(LABEL10_ORDER)}",
        "  Right: first two crest stops (binary; same ramp as fig 07).",
    ])

    sub_a10_set = df[df["audio_label_10"].notna()]
    blocks.extend(_sep("Figure 10 — fig_10_audio10_setting_rownorm_heatmap.png"))
    blocks.extend([
        "Row-normalized audio ten-class × Indoor/Outdoor/Mixed",
        "Logical fields: audio_label_10, setting_tri",
        f"Exclusions: n_missing_audio10 = {n_missing_audio10}; rows used {len(sub_a10_set)}",
        "Annotation: '—' for undefined row; '0.00' for true zeros.",
    ])

    sub_a10_loc = df[df["audio_label_10"].notna() & df["gpt_location_type"].notna()]
    blocks.extend(_sep("Figure 11 — fig_11_audio10_location_rownorm_heatmap.png"))
    blocks.extend([
        "Row-normalized audio ten-class × location (top-k + other)",
        f"location_top_k = {location_top_k}",
        f"Rows used: {len(sub_a10_loc)}",
    ])

    blocks.extend(_sep("Figure 20 — fig_20_audio10_location_counts_heatmap.png"))
    blocks.extend([
        "Snippet counts: audio ten-class × location (same top-k + other as fig 11)",
        f"location_top_k = {location_top_k}",
        f"Rows used: {len(sub_a10_loc)}",
    ])

    sub_a10_b = df[df["audio_label_10"].notna() & df["gpt_primary_behavior"].notna()]
    blocks.extend(_sep("Figure 12 — fig_12_audio10_behavior_rownorm_heatmap.png"))
    blocks.extend([
        "Row-normalized audio ten-class × primary behavior (top 8 + min-count other)",
        f"behavior_min_count = {behavior_min_count}",
        f"Rows used: {len(sub_a10_b)}",
    ])

    sub_bs = df[df["gpt_primary_behavior"].notna() & df["setting_tri"].notna()]
    blocks.extend(_sep("Figure 21 — fig_21_behavior_setting_rownorm_heatmap.png"))
    blocks.extend([
        "Row-normalized primary behavior × setting (Indoor / Outdoor / Mixed/Unclear)",
        "Same behavior collapse as fig 12 (min_count + top-8 + other)",
        f"behavior_min_count = {behavior_min_count}",
        f"Rows used: {len(sub_bs)}",
    ])
    blocks.extend(_sep("Figure 22 — fig_22_behavior_setting_counts_heatmap.png"))
    blocks.extend([
        "Snippet counts: primary behavior × setting (same collapse as fig 21)",
        f"behavior_min_count = {behavior_min_count}",
        f"Rows used: {len(sub_bs)}",
    ])

    fplot = funnel_rows_for_plot(funnel_rows)
    blocks.extend(_sep("Figure 13 — fig_13_pipeline_funnel.png"))
    blocks.extend([
        "In-figure title: Pipeline funnel (per-stage survivor counts)",
        "Horizontal bars (crest); top bar = earliest measured stage in this run.",
        "",
        "Stages on the figure (plateau / optional upstream stages may be omitted — see config):",
    ])
    if not fplot:
        blocks.append("  (empty — set manifest paths or paper_figures.pipeline_funnel.search, etc.)")
    for r in fplot:
        blocks.append(f"  {r.stage}: {r.count}")
    blocks.extend([
        "",
        "Table: table_06_pipeline_funnel.txt (all nine stage names; optional rows show — until set in config).",
        f"Final manifest rows (--manifest): {n_rows}",
    ])

    plat_agg, plat_skip = aggregate_platform_suitability(rows)
    blocks.extend(_sep("Figure 14 — fig_14_platform_suitable_stack.png"))
    youtube_n = sum(plat_agg["youtube"])
    tiktok_n = sum(plat_agg["tiktok"])
    blocks.extend([
        "In-figure title: Suitable vs not suitable by platform",
        "",
        "Description:",
        "  Stacked counts for YouTube vs TikTok: suitable for training vs not (third stack only if",
        "  suitability flag is missing). Compares how much each host contributes as unsuitable volume",
        "  relative to its own total.",
        "",
        "Logical fields:",
        "  manifest ``platform`` (fallback ``source`` parsed as YouTube/TikTok URL or shorthand);",
        "  suitability: dataset_flags.suitable_for_training, suitable_for_training, gpt_suitable_for_training",
        "",
        "Rows used (resolved platform): "
        f"{youtube_n + tiktok_n} (YouTube {youtube_n}, TikTok {tiktok_n}).",
    ])
    if plat_skip:
        blocks.append(f"Excluded (unknown platform/source): {plat_skip}")
    blocks.extend([
        "",
        "Key counts (suitable / not suitable / missing suitability):",
        f"  YouTube: {plat_agg['youtube'][0]} / {plat_agg['youtube'][1]} / {plat_agg['youtube'][2]}",
        f"  TikTok: {plat_agg['tiktok'][0]} / {plat_agg['tiktok'][1]} / {plat_agg['tiktok'][2]}",
        "",
        "Table: table_07_platform_suitability.txt",
    ])

    blocks.extend(_sep("Table 4 — table_04_face_by_audio_label_10.txt"))
    blocks.extend([
        "Face True/False/missing by audio ten-class label (audio_label_10)",
        f"Exclusions: n_missing_audio10 = {n_missing_audio10}; non-null audio_label_10 rows only.",
        f"Canonical row order: {', '.join(LABEL10_ORDER)}",
    ])

    blocks.extend(_sep("Table 5 — table_05_gpt_costs.txt"))
    blocks.extend([
        "GPT usage and USD estimates from gpt_descriptions.jsonl, cat_id_report, config.",
        "",
    ])
    for n in cost_footer_notes:
        blocks.append(n)

    blocks.extend(_sep("Table 6 — table_06_pipeline_funnel.txt"))
    blocks.extend([
        "Pipeline funnel: stage, count, retention vs previous, data-source notes.",
        "Optional scraper stages — set paper_figures.pipeline_funnel (search, keyword_filter, llm_filter) in config.yaml.",
        "",
    ])

    blocks.extend(_sep("Table 7 — table_07_platform_suitability.txt"))
    blocks.extend([
        "Cross-tab counts: YouTube/TikTok × suitable / not suitable / missing (same aggregation as Fig. 14).",
        "",
    ])

    blocks.extend(_sep("Table 8 — table_08_lighting_blur_counts.txt"))
    blocks.extend([
        "Snippet counts: gpt_lighting × gpt_blur (fixed vocabularies; same subset as Fig. 6 heatmap).",
        "Lighting rows sorted by descending row total; includes within-row % and marginal % of corpus.",
        "",
    ])

    blocks.extend(["", "=" * 78, "End of report", ""])
    return "\n".join(blocks)


def attach_derived_columns(
    df: pd.DataFrame,
    location_other_min: int,
    behavior_min_count: int,
) -> pd.DataFrame:
    """In-place style: add _loc_col, _beh_col for report + fig 8."""
    out = df.copy()
    out["_loc_col"] = location_collapsed(out["gpt_location_type"], location_other_min)
    out["_beh_col"] = collapse_rare_categories(out["gpt_primary_behavior"], behavior_min_count)
    return out
