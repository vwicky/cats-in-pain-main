"""
Train every pose model on each Paining-inclusive class subset (15 combinations),
one stratified group train/val split (no cross-validation), full data for that subset.

By default, only rows with **strictly higher** than 0.7 on the manifest *numeric* field
``audio_confidence`` (YAMNet/clip top-class score, 0--1) are used. The boolean
``audio_high_confidence`` is a separate gate, not a graded score—use ``--audio-confidence-field``
if you need a different column. Override the threshold with ``--min-audio-confidence`` or turn
off with ``--no-audio-confidence-filter``.

Usage:
  python model_training_v2/run_class_subset_experiment.py
  python model_training_v2/run_class_subset_experiment.py --models P1 P4
  python model_training_v2/run_class_subset_experiment.py --device cuda
  python model_training_v2/run_class_subset_experiment.py --no-audio-confidence-filter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from class_subset_utils import  (
    filter_dataframe_by_audio_confidence,
    filter_remap_dataframe,
    first_stratified_group_split,
    iter_paining_inclusive_subsets,
    split_viable,
    subset_to_dirname,
)
from models import  DEFAULT_MODEL_IDS, MODEL_REGISTRY
from training_loop import  train_single_split


def make_run_dir(cfg: dict, experiment_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPO_ROOT / cfg["output"]["runs_dir"]
    run_dir = base / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    log_path = run_dir / "experiment.log"
    logger = logging.getLogger("class_subset_experiment")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def build_model_kwargs(cfg: dict, n_classes: int, use_kinematics: bool = True) -> dict:
    nc = int(cfg["data"]["n_channels"]) * (2 if use_kinematics else 1)
    return {
        "n_frames": int(cfg["data"]["n_frames"]),
        "n_keypoints": int(cfg["data"]["n_keypoints"]),
        "n_channels": nc,
        "n_classes": int(n_classes),
    }


def make_master_row(
    out: dict,
    model_id: str,
    class_subset: tuple[str, ...],
    output_dir: Path,
    *,
    master_meta: dict | None = None,
) -> dict:
    meta = master_meta or {}
    return {
        "model_id": model_id,
        "model_name": out.get("model_name"),
        "class_subset": ",".join(class_subset),
        "K": out.get("n_classes"),
        "n_train": out.get("n_train"),
        "n_val": out.get("n_val"),
        "best_epoch": out.get("best_epoch"),
        "val_macro_f1_5": out.get("val_macro_f1_5"),
        "val_sensitivity": out.get("val_sensitivity"),
        "val_specificity": out.get("val_specificity"),
        "val_auc_roc": out.get("val_auc_roc"),
        "val_loss_total": out.get("val_loss_total"),
        "val_accuracy_5": out.get("val_accuracy_5"),
        "val_accuracy_binary": out.get("val_accuracy_binary"),
        "val_macro_f1_binary": out.get("val_macro_f1_binary"),
        "per_class_f1_paining": out.get("val_per_class_f1_Paining"),
        "min_audio_confidence": meta.get("min_audio_confidence"),
        "audio_confidence_field": meta.get("audio_confidence_field"),
        "audio_confidence_filter": meta.get("audio_confidence_filter"),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)) if _under_repo(output_dir) else str(output_dir),
    }


def _under_repo(p: Path) -> bool:
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


def save_master_results(
    all_results: list[dict], run_dir: Path, start_time: float, models_to_run: list[str]
) -> None:
    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv(run_dir / "master_results.csv", index=False)
        with open(run_dir / "master_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    n_sub = 15
    n_done = len(all_results)
    print(
        f"\n════════════════════════════════════════════════════════════\n"
        f"CLASS-SUBSET EXPERIMENT — saved partial or full results\n"
        f"Run: {run_dir.name}\n"
        f"Models: {len(models_to_run)} | Combinations each: {n_sub} | Completed runs: {n_done}\n"
        f"Elapsed: {h}h {m}m {s}s\n"
        f"════════════════════════════════════════════════════════════\n"
    )


def write_aggregated_report(
    run_dir: Path, results: list[dict], top_n: int = 30
) -> None:
    path = run_dir / "aggregated_report.txt"
    if not results:
        path.write_text("No completed runs to aggregate.\n", encoding="utf-8")
        return
    lines: list[str] = [
        "Class-subset training — comparison report",
    ]
    if results and results[0].get("audio_confidence_filter") is not None:
        f0 = results[0]
        lines.append(
            f"Data filter: audio_conf={f0.get('audio_confidence_field')!r}  "
            f"min={f0.get('min_audio_confidence')}  enabled={f0.get('audio_confidence_filter')}"
        )
    lines.append(f"Total completed runs: {len(results)}")
    lines.append("")

    df = pd.DataFrame(results)
    for col, title in [
        ("val_macro_f1_5", "Top runs by val_macro_f1_5 (multiclass macro)"),
        ("val_sensitivity", "Top runs by val_sensitivity (Paining recall)"),
    ]:
        if col not in df.columns:
            continue
        lines.append(f"--- {title} (up to {top_n}) ---")
        sub = df.sort_values(col, ascending=False, na_position="last").head(top_n)
        for _, r in sub.iterrows():
            lines.append(
                f"  {r.get('model_id', '?'):<5}  subset={r.get('class_subset', '?'):<50}  "
                f"macroF1_5={float(r.get('val_macro_f1_5', 0) or 0):.4f}  "
                f"sens={float(r.get('val_sensitivity', 0) or 0):.4f}  "
                f"spec={float(r.get('val_specificity', 0) or 0):.4f}  "
                f"auc={r.get('val_auc_roc', 'nan')}"
            )
        lines.append("")

    lines.append("--- Best val_macro_f1_5 per model ---")
    for mid in sorted(df["model_id"].unique()):
        m = df[df["model_id"] == mid]
        if m.empty:
            continue
        best = m.sort_values("val_macro_f1_5", ascending=False, na_position="last").iloc[0]
        lines.append(
            f"  {mid}: macroF1_5={float(best.get('val_macro_f1_5', 0) or 0):.4f}  "
            f"subset={best.get('class_subset')}  "
            f"sens={float(best.get('val_sensitivity', 0) or 0):.4f}  "
            f"dir={best.get('output_dir', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paining-inclusive class subset training (no CV).")
    parser.add_argument("--config", default="video/pose-models/config.yaml")
    parser.add_argument("--models", nargs="+", help="Model IDs. Default: all registered.")
    parser.add_argument(
        "--experiment-name",
        default="class_subset_hiconf_70",
        help="Run folder base name (appended with timestamp). Default: class_subset_hiconf_70 (distinct from unfiltered / parallel jobs).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-kinematics", action="store_true", help="Use 3 channels (no velocity).")
    parser.add_argument(
        "--audio-confidence-field",
        default="audio_confidence",
        help="Manifest column to threshold. Use ``audio_confidence`` (float 0-1) for a real score cut. "
        "The ``audio_high_confidence`` field is a boolean only; 0.7> keeps all-True and drops nothing. "
        "Config ``labels.confidence_field`` (often audio_high_confidence) is for pose, not this filter—do not use it here.",
    )
    parser.add_argument(
        "--min-audio-confidence",
        type=float,
        default=0.7,
        help="Keep rows with confidence *strictly greater* than this. "
        "0.0–1.0: fraction; if the column is 0–100, 0.7 is mapped to 70. "
        "1–100: use as 0–100 when max(conf)>1, else divide by 100. Ignored with --no-audio-confidence-filter.",
    )
    parser.add_argument(
        "--no-audio-confidence-filter",
        action="store_true",
        help="Use all rows from load_dataset (no confidence threshold).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override config split.random_state (StratifiedGroupKFold).",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    if args.split_seed is not None:
        cfg = dict(cfg)
        cfg["split"] = {**cfg["split"], "random_state": int(args.split_seed)}

    use_kinematics = not args.no_kinematics
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    run_dir = make_run_dir(cfg, args.experiment_name)
    logger = setup_logger(run_dir)
    start_time = time.time()
    conf_field = str(args.audio_confidence_field)
    filter_enabled = not bool(args.no_audio_confidence_filter)
    cfg_to_save: dict = dict(cfg)
    cfg_to_save["class_subset_experiment"] = {
        "min_audio_confidence": None
        if not filter_enabled
        else float(args.min_audio_confidence),
        "confidence_field": conf_field,
        "filter_enabled": filter_enabled,
    }
    (run_dir / "config_used.yaml").write_text(
        yaml.dump(cfg_to_save, default_flow_style=False), encoding="utf-8"
    )
    master_meta: dict = {
        "min_audio_confidence": None
        if not filter_enabled
        else float(args.min_audio_confidence),
        "audio_confidence_field": conf_field,
        "audio_confidence_filter": filter_enabled,
    }

    classes_5: list[str] = list(cfg["labels"]["classes_5"])
    label_field: str = str(cfg["labels"]["label_field"])
    pain_class: str = str(cfg["labels"].get("binary_pain_class", "Paining"))
    batch_size = int(cfg["training"]["batch_size"])
    models_to_run: list[str] = list(args.models) if args.models else list(DEFAULT_MODEL_IDS)
    subsets = iter_paining_inclusive_subsets(classes_5, pain_class=pain_class)
    n_sub = len(subsets)

    _cbox = 58
    _line_audio = f"  Audio: {conf_field}  filter={filter_enabled}  min={master_meta.get('min_audio_confidence')}"[:_cbox]
    print(
        f"""
╔══════════════════════════════════════════════════════════╗
║  CLASS-SUBSET (Paining-inclusive)                      ║
║  Run: {run_dir.name:<46}║
║  Device: {str(device):<44}║
║  Models: {len(models_to_run):<2} | Subsets each: {n_sub:<2}  Total: {len(models_to_run) * n_sub:<5}║
║{_line_audio:<{_cbox}}║
╚══════════════════════════════════════════════════════════╝
""",
        flush=True,
    )
    from data_loading import  load_dataset, print_dataset_statistics

    df = load_dataset(cfg, logger)
    print_dataset_statistics(df, logger)
    if filter_enabled:
        n_before = len(df)
        df = filter_dataframe_by_audio_confidence(
            df,
            conf_field,
            float(args.min_audio_confidence),
            logger=logger,
        )
        if len(df) < 1:
            logger.error(
                "No rows left after audio confidence filter; "
                "loosen --min-audio-confidence or use --no-audio-confidence-filter."
            )
            raise SystemExit(1)
        print_dataset_statistics(df, logger)
        logger.info("Rows after high-confidence filter: %d (was %d).", len(df), n_before)
    else:
        logger.info("Audio confidence filter: disabled (using full load_dataset output).")

    from data_loading import  plot_dataset_overview

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    plot_dataset_overview(df, plot_dir)

    all_results: list[dict] = []
    try:
        pbar = tqdm(
            total=len(models_to_run) * n_sub,
            desc="model×subset",
        )
        for model_id in models_to_run:
            if model_id not in MODEL_REGISTRY:
                logger.warning("Unknown model_id %s — skipping", model_id)
                pbar.update(n_sub)
                continue
            model_class, extra_kw = MODEL_REGISTRY[model_id]
            for class_subset in subsets:
                pbar.update(1)
                k = len(class_subset)
                try:
                    dsub = filter_remap_dataframe(
                        df, class_subset, label_field=label_field, pain_class=pain_class
                    )
                except ValueError as e:
                    logger.warning("subset %s: %s", class_subset, e)
                    continue
                if len(dsub) < 2:
                    logger.warning("subset %s: too few rows after filter (%d)", class_subset, len(dsub))
                    continue

                train_df, val_df, err = first_stratified_group_split(dsub, cfg)
                if err:
                    logger.warning("subset %s: %s", class_subset, err)
                    continue
                ok, reason = split_viable(train_df, val_df, k, batch_size=batch_size)
                if not ok:
                    logger.warning("subset %s: skip: %s", class_subset, reason)
                    continue

                sub_dir = run_dir / model_id / subset_to_dirname(class_subset)
                sub_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs = {**build_model_kwargs(cfg, k, use_kinematics), **extra_kw}
                probe = model_class(**model_kwargs)
                mname = probe.model_name
                tr_recs = train_df.to_dict("records")
                va_recs = val_df.to_dict("records")
                logger.info("Training %s | %s | n_train=%d n_val=%d", model_id, class_subset, len(tr_recs), len(va_recs))
                out = train_single_split(
                    model_class,
                    model_kwargs,
                    tr_recs,
                    va_recs,
                    sub_dir,
                    cfg,
                    device,
                    logger,
                    class_names=list(class_subset),
                    use_kinematics=use_kinematics,
                )
                row = make_master_row(
                    out, model_id, class_subset, sub_dir, master_meta=master_meta
                )
                row["model_name"] = mname
                all_results.append(row)
        pbar.close()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — writing partial results.")
        save_master_results(all_results, run_dir, start_time, models_to_run)
        if all_results:
            write_aggregated_report(run_dir, all_results)
        raise SystemExit(130)

    save_master_results(all_results, run_dir, start_time, models_to_run)
    write_aggregated_report(run_dir, all_results)
    if all_results:
        print(f"Wrote: {run_dir / 'master_results.csv'}, {run_dir / 'aggregated_report.txt'}")


if __name__ == "__main__":
    main()
