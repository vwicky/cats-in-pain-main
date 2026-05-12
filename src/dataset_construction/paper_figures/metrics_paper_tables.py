"""Thesis tables: binary method comparison, Track A per-class CV, pairwise ST-GCN metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

CV_SUMMARY_REL_PATHS = (
    "video/pose-models/runs/pose-models/p4_audio_10class_macrof1_lr3e5_20260428_183753/P4/fraction_100/cv_summary.json",
    "runs/pose-models/p4_audio_10class_macrof1_lr3e5_20260428_183753/P4/fraction_100/cv_summary.json",
)

# Snapshot from video_pipeline_comprehensive_report.txt (same run path as above) when JSON is absent locally.
_CV_SUMMARY_FALLBACK: dict[str, Any] | None = None


def _fallback_cv_summary() -> dict[str, Any]:
    global _CV_SUMMARY_FALLBACK
    if _CV_SUMMARY_FALLBACK is not None:
        return _CV_SUMMARY_FALLBACK
    report = REPO_ROOT / "src/dataset_construction/reports/video_pipeline_comprehensive_report.txt"
    if not report.is_file():
        raise FileNotFoundError(
            f"No cv_summary.json on disk and no comprehensive report at {report}"
        )
    text = report.read_text(encoding="utf-8")
    m = re.search(
        r"C\.\s+runs/pose-models/p4_audio_10class_macrof1_lr3e5_20260428_183753/P4/fraction_100/cv_summary\.json\s*-+\s*(\{.*?\})\s*-+",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Could not extract cv_summary JSON block from video_pipeline_comprehensive_report.txt")
    _CV_SUMMARY_FALLBACK = json.loads(m.group(1))
    return _CV_SUMMARY_FALLBACK


def load_cv_summary_track_a() -> dict[str, Any]:
    for rel in CV_SUMMARY_REL_PATHS:
        p = REPO_ROOT / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return _fallback_cv_summary()


LABEL10_ORDER = [
    "Angry",
    "Defence",
    "Fighting",
    "Happy",
    "HuntingMind",
    "Mating",
    "MotherCall",
    "Paining",
    "Resting",
    "Warning",
]


def _excerpt_block(rel: str, start_line: int, end_line: int) -> list[str]:
    """Return a banner plus a 1-based inclusive line slice from REPO_ROOT / rel."""
    path = REPO_ROOT / rel
    if not path.is_file():
        return [f"--- Excerpt: {rel} (file not found under repo root) ---", ""]
    raw = path.read_text(encoding="utf-8").splitlines()
    n = len(raw)
    lo = max(0, start_line - 1)
    hi = min(n, end_line)
    last = hi if hi > 0 else 0
    banner = f"--- Excerpt: {rel} (lines {start_line}-{min(end_line, n)} of {n}) ---"
    return [banner, *raw[lo:hi], ""]


def write_table_11_binary_method_comparison(out_dir: Path) -> None:
    """Four-row binary comparison; EfficientNet v2 from last reported CNN run in comprehensive report."""
    rows_spec = [
        ("EfficientNet-B3 (v2)", "0.438", "0.691", "0.460"),
        ("GPT-4o FGS (legacy cohort†)", "0.353", "0.702", "0.332"),
        ("GPT-4o label-derived (legacy†)", "0.532", "0.155", "0.905"),
        ("Neural F1-OOF (pose+CLIP)", "0.679", "0.447", "0.915"),
    ]
    lines = [
        "Table 11 — Binary pain classification: method comparison",
        "",
        "Context: Compares the v2 EfficientNet-B3 binary head (5-class CV, cat_id-grouped folds) against fixed",
        "baseline rows carried in the CNN report and literature snapshots. GPT FGS uses the Feline Grimace–style",
        "vision protocol; GPT label-derived maps model behavior labels to pain vs not pain (legacy cohort only).",
        "Neural F1-OOF is the out-of-fold pose+CLIP benchmark cited next to the CNN table.",
        "",
        "Related:",
        "  - video/cnn-finetuning/config/v2.yaml — data: final_dataset_v2.jsonl, five-class + binary head",
        "  - video/cnn-finetuning/src/train.py, video/cnn-finetuning/src/evaluate.py — CV + comparison block",
        "  - video/gpt-fgs/cat_pain_llm_eval*.ipynb — GPT-4o + FGS scoring pipeline",
        "  - src/dataset_construction/reports/video_pipeline_comprehensive_report.txt — CNN row quote + same constants",
        "",
        "Source excerpts (verbatim from repository files):",
        "",
    ]
    lines.extend(_excerpt_block("video/cnn-finetuning/config/v2.yaml", 1, 50))
    lines.extend(_excerpt_block("video/cnn-finetuning/src/evaluate.py", 342, 355))
    lines.extend(
        _excerpt_block(
            "src/dataset_construction/reports/video_pipeline_comprehensive_report.txt",
            765,
            773,
        )
    )
    lines.extend(
        [
        "Macro F1 / sensitivity / specificity (binary head where applicable).",
        "Sources: EfficientNet row = CNN finetune on final_dataset_v2 (video_pipeline_comprehensive_report.txt); "
        "GPT + Neural rows = evaluate.py comparison constants / prior pose+CLIP OOF.",
        "",
            f"{'Method':<36} {'Macro F1':>10} {'Sensitivity':>12} {'Specificity':>12}",
            "-" * 74,
        ]
    )
    for method, mf1, sens, spec in rows_spec:
        lines.append(f"{method:<36} {mf1:>10} {sens:>12} {spec:>12}")
    lines.extend(
        [
            "",
            "† Evaluated on legacy manifest, not directly comparable to v2 results.",
        ]
    )
    from thesis_io import write_txt_table

    write_txt_table(out_dir / "table_11_binary_method_comparison.txt", lines)


def write_table_12_track_a_per_class_f1(out_dir: Path) -> None:
    agg = load_cv_summary_track_a()
    per_rows: list[tuple[str, float, float, float, float]] = []
    for cls in LABEL10_ORDER:
        mf = float(agg[f"mean_per_class_f1_{cls}"])
        sf = float(agg[f"std_per_class_f1_{cls}"])
        mr = float(agg[f"mean_per_class_recall_{cls}"])
        mp = float(agg[f"mean_per_class_precision_{cls}"])
        per_rows.append((cls, mf, sf, mr, mp))
    per_rows.sort(key=lambda t: t[1], reverse=True)

    lines = [
        "Table 12 — Track A (10-class): per-class F1 / recall / precision (CV mean ± std)",
        "",
        "Context: Track A trains the P4 ST-GCN backbone on audio_label_10 (10 fine-grained vocal classes)",
        "from final_dataset_v2.jsonl with ViT-normalized poses; metrics are mean ± std across 5 CV folds at",
        "100% data (fraction_100). Per-class F1/recall/precision are aggregated in cv_summary.json by",
        "video/pose-models/training_loop.py.",
        "",
        "Related:",
        "  - video/pose-models/config_p4_10class.yaml — 10-way head, manifest + pose paths",
        "  - video/pose-models/run_scaling_experiment.py — example driver (see header in config_p4_10class.yaml)",
        "  - video/pose-models/training_loop.py — writes cv_summary.json beside fold outputs",
        "  - Documented run (metrics source): runs/pose-models/p4_audio_10class_macrof1_lr3e5_20260428_183753/…",
        "  - Fallback if JSON missing on disk: embedded block in video_pipeline_comprehensive_report.txt (section C)",
        "",
        "Source excerpts (verbatim from repository files):",
        "",
    ]
    lines.extend(_excerpt_block("video/pose-models/config_p4_10class.yaml", 1, 42))
    lines.extend(_excerpt_block("video/pose-models/training_loop.py", 1316, 1352))
    lines.extend(
        [
        "Source: cv_summary.json values — Pose-ST-GCN P4, p4_audio_10class_macrof1_lr3e5_20260428_183753, "
        "fraction 100%, 5 folds. If cv_summary.json is absent locally, values are parsed from the embedded "
        "snapshot in video_pipeline_comprehensive_report.txt.",
        "Sort: descending by mean F1.",
        "",
            f"{'Class':<14} {'Mean F1':>10} {'Std F1':>10} {'Mean Recall':>12} {'Mean Precision':>14}",
            "-" * 64,
        ]
    )
    for cls, mf, sf, mr, mp in per_rows:
        lines.append(
            f"{cls:<14} {mf:>10.3f} {sf:>10.3f} {mr:>12.3f} {mp:>14.3f}"
        )
    lines.extend(
        [
            "",
            "Defence and Warning F1 = 0.0 across all folds due to insufficient validation support "
            "(6 and 10 clips respectively in fold 0).",
        ]
    )
    from thesis_io import write_txt_table

    write_txt_table(out_dir / "table_12_track_a_per_class_f1.txt", lines)


def write_table_13_pairwise_results(out_dir: Path) -> None:
    csv_path = REPO_ROOT / "src/dataset_construction/reports/pairwise_run_summary_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing pairwise CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["_pi"] = df["pain_inclusive"].str.strip().str.lower().isin(("true", "1", "yes"))
    df["_f1"] = pd.to_numeric(df["val_macro_f1_binary"], errors="coerce")
    pain = df[df["_pi"]].sort_values("_f1", ascending=False)
    non = df[~df["_pi"]].sort_values("_f1", ascending=False)
    out_df = pd.concat([pain, non], ignore_index=True)

    lines = [
        "Table 13 — Pairwise binary ST-GCN: all 45 class pairs",
        "",
        "Context: Track C fits C(10,2)=45 unordered one-vs-one binary classifiers on audio_label_10 clips,",
        "each pair with training.binary_only: true (see run_p4_pairwise.py). This table is the aggregated CSV:",
        "for each pair it keeps the training with highest val_macro_f1_binary when multiple grids exist.",
        "",
        "Related:",
        "  - video/pose-models/config_p4_pairwise.yaml — pairwise defaults (same manifest/pose wiring as 10-class)",
        "  - video/pose-models/run_p4_pairwise.py — launches pair-specific binary runs",
        "  - src/dataset_construction/reports/export_pairwise_run_summary_metrics.py — builds this CSV from a bundle",
        "  - Typical bundle root (per export script docstring): runs/pose-models/p4_pairwise_ensemble_bundle_20260428/",
        "",
        "Source excerpts (verbatim from repository files):",
        "",
    ]
    lines.extend(_excerpt_block("video/pose-models/config_p4_pairwise.yaml", 1, 45))
    lines.extend(
        _excerpt_block("src/dataset_construction/reports/export_pairwise_run_summary_metrics.py", 1, 95)
    )
    lines.extend(
        [
        "Note: export_pairwise_run_summary_metrics.py sorts by (pain_inclusive, pair name). This .txt table",
        "re-sorts rows by Macro F1 descending within pain-inclusive and non-pain groups.",
        "",
        "Sort: pain-inclusive pairs first (Macro F1 descending), then non-pain pairs (Macro F1 descending).",
        f"Source: {csv_path.relative_to(REPO_ROOT)} (verbatim val_auc_roc / val_macro_f1_binary / counts).",
        "",
            f"{'Pair':<22} {'Pain-inclusive':>14} {'N train':>8} {'N val':>7} {'Macro F1':>10} {'AUC-ROC':>10}",
            "-" * 78,
        ]
    )
    for _, r in out_df.iterrows():
        pair = str(r["pair"])
        pi = "Yes" if r["_pi"] else "No"
        nt = str(r["n_train"]).strip()
        nv = str(r["n_val"]).strip()
        f1 = str(r["val_macro_f1_binary"]).strip()
        auc = str(r["val_auc_roc"]).strip()
        lines.append(f"{pair:<22} {pi:>14} {nt:>8} {nv:>7} {f1:>10} {auc:>10}")
    from thesis_io import write_txt_table

    write_txt_table(out_dir / "table_13_pairwise_binary_results.txt", lines)


def write_metrics_paper_tables(out_dir: Path) -> None:
    out_dir = Path(out_dir).resolve()
    write_table_11_binary_method_comparison(out_dir)
    write_table_12_track_a_per_class_f1(out_dir)
    write_table_13_pairwise_results(out_dir)
