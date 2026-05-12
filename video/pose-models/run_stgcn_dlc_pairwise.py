"""
Train ST-GCN–DeepLabCut (P6) on **every unordered pair** of multiclass labels
(e.g. Paining vs Happy): binary task, one stratified group split per pair.

Each pair writes under a **separate subfolder** of one master run directory
(``pair_to_dirname`` → ``binary__ClassA__ClassB/training/``), mirroring
``run_stgcn_deeplabcut_train.py`` layout.

Label convention (same as thesis pain-vs-rest when Paining is in the pair):
- If ``Paining`` is in the pair, it is positive (1); the other class is (0).
- Otherwise lexicographic order: earlier name → 0, later → 1.

Load uses ``training.binary_only: false`` so the manifest keeps full
``label_int`` until this script filters to two classes and remaps.

Train/val split: ``first_stratified_group_split`` on ``cfg`` (same as single-run
DLC). If ``split.val_fraction`` is set, the fold whose |val|/N is closest to
that target is chosen (grouped by ``split.group_field``, stratified on the
remapped binary ``label_int``).

Usage:
  python model_training_v2/run_stgcn_dlc_pairwise.py --dry-run
  python model_training_v2/run_stgcn_dlc_pairwise.py --config model_training_v2/config_stgcn_dlc.yaml --device cuda
  python model_training_v2/run_stgcn_dlc_pairwise.py --pair Happy,Paining --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from class_subset_utils import  (
    filter_dataframe_by_audio_confidence,
    filter_remap_binary_pair,
    first_stratified_group_split,
    iter_sorted_class_pairs,
    pair_binary_neg_pos,
    pair_to_dirname,
    split_val_fraction_from_cfg,
    split_viable,
)
from data_loading import  (
    get_multiclass_class_names,
    load_dataset_for_deeplabcut,
    plot_dataset_overview,
    print_dataset_statistics,
)
from run_stgcn_deeplabcut_train import  (
    apply_clip_quality_filter,
    build_model_kwargs,
    setup_logger,
    write_data_filter_summary,
    _bodypart_audit_text,
)


def make_master_run_dir(cfg: dict, experiment_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPO_ROOT / cfg["output"]["runs_dir"]
    run_dir = base / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    return run_dir


def make_master_row(
    out: dict,
    *,
    class_lo: str,
    class_hi: str,
    class_neg: str,
    class_pos: str,
    output_dir: Path,
    master_meta: dict | None = None,
) -> dict:
    meta = master_meta or {}
    return {
        "class_lo": class_lo,
        "class_hi": class_hi,
        "class_neg_label_int_0": class_neg,
        "class_pos_label_int_1": class_pos,
        "pair": f"{class_lo}|{class_hi}",
        "n_train": out.get("n_train"),
        "n_val": out.get("n_val"),
        "best_epoch": out.get("best_epoch"),
        "n_epochs_ran": out.get("n_epochs_ran"),
        "early_stop_patience": out.get("early_stop_patience"),
        "ce_ratio": out.get("ce_ratio"),
        "ce_ratio_heuristic": out.get("ce_ratio_heuristic"),
        "val_sensitivity": out.get("val_sensitivity"),
        "val_specificity": out.get("val_specificity"),
        "val_auc_roc": out.get("val_auc_roc"),
        "val_loss_total": out.get("val_loss_total"),
        "val_accuracy_binary": out.get("val_accuracy_binary"),
        "val_macro_f1_binary": out.get("val_macro_f1_binary"),
        "min_audio_confidence": meta.get("min_audio_confidence"),
        "audio_confidence_field": meta.get("audio_confidence_field"),
        "audio_confidence_filter": meta.get("audio_confidence_filter"),
        "output_dir": str(output_dir.relative_to(REPO_ROOT))
        if _under_repo(output_dir)
        else str(output_dir),
    }


def _under_repo(p: Path) -> bool:
    try:
        p.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


def save_master_results(all_results: list[dict], run_dir: Path, start_time: float, n_pairs: int) -> None:
    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv(run_dir / "master_results.csv", index=False)
        with open(run_dir / "master_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(
        f"\n════════════════════════════════════════════════════════════\n"
        f"DLC BINARY PAIRWISE — saved partial or full results\n"
        f"Run: {run_dir.name}\n"
        f"Pairs planned: {n_pairs} | Completed runs: {len(all_results)}\n"
        f"Elapsed: {h}h {m}m {s}s\n"
        f"════════════════════════════════════════════════════════════\n"
    )


def write_pairwise_report(
    run_dir: Path, results: list[dict], top_n: int = 30, *, report_title: str | None = None
) -> None:
    path = run_dir / "aggregated_report.txt"
    if not results:
        path.write_text("No completed pairwise runs to aggregate.\n", encoding="utf-8")
        return
    default_title = "ST-GCN–DLC binary pairwise training — comparison report"
    lines: list[str] = [
        report_title or default_title,
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
    col = "val_macro_f1_binary"
    if col in df.columns:
        lines.append(f"--- Top runs by {col} (up to {top_n}) ---")
        sub = df.sort_values(col, ascending=False, na_position="last").head(top_n)
        for _, r in sub.iterrows():
            lines.append(
                f"  pair={r.get('pair', '?'):<40}  macroF1_bin={float(r.get(col, 0) or 0):.4f}  "
                f"sens={float(r.get('val_sensitivity', 0) or 0):.4f}  "
                f"spec={float(r.get('val_specificity', 0) or 0):.4f}  "
                f"auc={r.get('val_auc_roc', 'nan')}"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_single_pair(s: str) -> tuple[str, str]:
    raw = s.replace(";", ",").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--pair expects two comma-separated class names, got: {s!r}")
    a, b = sorted(parts)
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GCN–DLC: train all binary class pairs.")
    parser.add_argument("--config", default="video/pose-models/config_stgcn_dlc.yaml")
    parser.add_argument("--experiment-name", default="stgcn_dlc_pairwise")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-kinematics", action="store_true")
    parser.add_argument(
        "--pair",
        metavar="A,B",
        help="Run a single pair (comma-separated). Default: all C(K,2) pairs from config labels.classes.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Stop after this many pairs (order: lexicographic on sorted class list).",
    )
    parser.add_argument(
        "--min-pair-rows",
        type=int,
        default=None,
        help="Skip a pair if fewer than this many rows after clip QC (default: max(24, 2*batch_size+4)).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a pair if pair_dir/training/run_summary.json already exists.",
    )
    parser.add_argument(
        "--audio-confidence-field",
        default="audio_confidence",
        help="Optional numeric column for strict thresholding (same semantics as class-subset script).",
    )
    parser.add_argument(
        "--min-audio-confidence",
        type=float,
        default=0.7,
        help="Keep rows strictly above this threshold when audio filter is enabled.",
    )
    parser.add_argument(
        "--audio-confidence-filter",
        action="store_true",
        help="Apply strict audio confidence filter before clip QC (off by default; matches single-run DLC script).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override config split.random_state for StratifiedGroupKFold.",
    )
    parser.add_argument(
        "--clip-qc-keep",
        action="store_true",
        help="Force clip_quality.on_low_confidence=keep (no HDF5 exclusion). Useful if strict QC removes all rows.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    if args.clip_qc_keep or args.split_seed is not None:
        cfg = copy.deepcopy(cfg)
    if args.clip_qc_keep:
        cfg.setdefault("clip_quality", {})["on_low_confidence"] = "keep"
    if args.split_seed is not None:
        cfg["split"] = {**cfg["split"], "random_state": int(args.split_seed)}

    load_cfg = copy.deepcopy(cfg)
    load_cfg.setdefault("training", {})["binary_only"] = False

    use_kinematics = not args.no_kinematics
    classes_mc = get_multiclass_class_names(load_cfg)
    label_field = str(load_cfg["labels"]["label_field"])
    pain_class = str(load_cfg["labels"].get("binary_pain_class", "Paining"))
    batch_size = int(load_cfg["training"]["batch_size"])
    min_pair = args.min_pair_rows
    if min_pair is None:
        min_pair = max(24, 2 * batch_size + 4)

    if args.pair:
        pairs: list[tuple[str, str]] = [_parse_single_pair(args.pair)]
        for lo, hi in pairs:
            if lo not in classes_mc or hi not in classes_mc:
                raise SystemExit(f"Unknown class in --pair; expected names from config: {lo!r}, {hi!r}")
    else:
        pairs = iter_sorted_class_pairs(classes_mc)
        if args.limit_pairs is not None:
            pairs = pairs[: max(0, int(args.limit_pairs))]

    run_dir = make_master_run_dir(cfg, args.experiment_name)
    logger = setup_logger(run_dir)
    start_time = time.time()

    conf_field = str(args.audio_confidence_field)
    filter_enabled = bool(args.audio_confidence_filter)
    cfg_to_save = copy.deepcopy(cfg)
    cfg_to_save["pairwise_experiment"] = {
        "min_audio_confidence": None if not filter_enabled else float(args.min_audio_confidence),
        "confidence_field": conf_field,
        "filter_enabled": filter_enabled,
        "clip_qc_keep_override": bool(args.clip_qc_keep),
        "pairs": [f"{a}|{b}" for a, b in pairs],
        "min_pair_rows": min_pair,
    }
    (run_dir / "config_used.yaml").write_text(
        yaml.dump(cfg_to_save, default_flow_style=False), encoding="utf-8"
    )
    master_meta = {
        "min_audio_confidence": None if not filter_enabled else float(args.min_audio_confidence),
        "audio_confidence_field": conf_field,
        "audio_confidence_filter": filter_enabled,
    }

    logger.info(
        "Pairwise DLC binary | %d pairs | label_field=%s | pain_class=%s | min_pair_rows=%d",
        len(pairs),
        label_field,
        pain_class,
        min_pair,
    )

    df = load_dataset_for_deeplabcut(load_cfg, logger)
    print_dataset_statistics(df, logger, load_cfg)
    plot_dataset_overview(df, run_dir / "plots", load_cfg)

    if filter_enabled:
        n0 = len(df)
        df = filter_dataframe_by_audio_confidence(
            df,
            conf_field,
            float(args.min_audio_confidence),
            logger=logger,
        )
        logger.info("Audio confidence filter: %d rows (was %d).", len(df), n0)
        if len(df) < 1:
            logger.error("No rows left after audio confidence filter.")
            raise SystemExit(1)
        print_dataset_statistics(df, logger, load_cfg)

    bodypart_audit = ""
    # Bodypart audit loads the quadruped graph (torch-backed models package). Skip on --dry-run
    # so environments without torch can still validate splits and row counts.
    if (
        not args.dry_run
        and len(df) > 0
        and "dlc_h5_path" in df.columns
        and df["dlc_h5_path"].iloc[0]
    ):
        sample = REPO_ROOT / str(df["dlc_h5_path"].iloc[0])
        if sample.is_file():
            bodypart_audit = _bodypart_audit_text(sample)
            logger.info("Bodypart audit (first row):\n%s", bodypart_audit.strip())

    n_h5 = len(df)
    df_filt, clip_stats = apply_clip_quality_filter(df, cfg, logger)
    write_data_filter_summary(run_dir, clip_stats, n_rows_with_h5=n_h5, bodypart_audit=bodypart_audit)
    logger.info("After clip quality: %d rows (was %d)", len(df_filt), len(df))

    with open(run_dir / "reports" / "pair_list.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "class_lo": lo,
                    "class_hi": hi,
                    "class_neg": pair_binary_neg_pos(lo, hi, pain_class=pain_class)[0],
                    "class_pos": pair_binary_neg_pos(lo, hi, pain_class=pain_class)[1],
                    "subdir": pair_to_dirname(lo, hi),
                }
                for lo, hi in pairs
            ],
            f,
            indent=2,
        )

    if args.dry_run:
        logger.info("Dry run — checking each pair (no PyTorch training).")
        for lo, hi in pairs:
            sub = filter_remap_binary_pair(
                df_filt, lo, hi, label_field=label_field, pain_class=pain_class
            )
            logger.info(
                "pair %s vs %s → n=%d (after filter; clip QC already applied)",
                lo,
                hi,
                len(sub),
            )
            if len(sub) < min_pair:
                logger.warning("  skip: fewer than min_pair_rows=%d", min_pair)
                continue
            tr, va, err = first_stratified_group_split(sub, cfg)
            if err:
                logger.warning("  split: %s", err)
                continue
            ok, reason = split_viable(tr, va, 2, batch_size=batch_size)
            logger.info("  split OK=%s  n_train=%d n_val=%d  %s", ok, len(tr), len(va), reason or "")
        logger.info("Dry run done.")
        return

    import torch

    from data_engineering import  build_dlc_dataloaders
    from models.stgcn_deeplabcut_model import  P6PoseSTGCNDeepLabCut
    from training_loop import  train_single_split

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    train_cfg = copy.deepcopy(cfg)
    train_cfg.setdefault("training", {})["binary_only"] = True

    all_results: list[dict] = []
    try:
        from tqdm import tqdm

        for lo, hi in tqdm(pairs, desc="pairs"):
            pdir_name = pair_to_dirname(lo, hi)
            pair_dir = run_dir / pdir_name
            pair_train_dir = pair_dir / "training"
            if args.skip_existing and (pair_train_dir / "run_summary.json").is_file():
                logger.info("skip existing: %s", pdir_name)
                continue

            neg, pos = pair_binary_neg_pos(lo, hi, pain_class=pain_class)
            sub = filter_remap_binary_pair(
                df_filt, lo, hi, label_field=label_field, pain_class=pain_class
            )
            if len(sub) < min_pair:
                logger.warning("%s: too few rows (%d < %d)", pdir_name, len(sub), min_pair)
                continue

            train_df, val_df, err = first_stratified_group_split(sub, cfg)
            if err:
                logger.warning("%s: split failed: %s", pdir_name, err)
                continue
            ok, reason = split_viable(train_df, val_df, 2, batch_size=batch_size)
            if not ok:
                logger.warning("%s: not viable: %s", pdir_name, reason)
                continue

            pair_dir.mkdir(parents=True, exist_ok=True)
            pair_train_dir.mkdir(parents=True, exist_ok=True)

            def _counts_two_classes(frame: pd.DataFrame) -> dict[str, int]:
                return {
                    neg: int((frame[label_field] == neg).sum()),
                    pos: int((frame[label_field] == pos).sum()),
                }

            n_tr, n_va = len(train_df), len(val_df)
            n_all = n_tr + n_va
            vf_split = split_val_fraction_from_cfg(cfg)
            meta = {
                "class_lo": lo,
                "class_hi": hi,
                "class_neg_label_int_0": neg,
                "class_pos_label_int_1": pos,
                "pain_class": pain_class,
                "label_field": label_field,
                "split_group_field": str(cfg["split"]["group_field"]),
                "split_val_fraction_target": vf_split,
                "n_filtered": int(len(sub)),
                "class_counts_filtered": _counts_two_classes(sub),
                "n_train": int(n_tr),
                "n_val": int(n_va),
                "class_counts_train": _counts_two_classes(train_df),
                "class_counts_val": _counts_two_classes(val_df),
                "train_fraction_of_filtered": round(n_tr / n_all, 6) if n_all else None,
                "val_fraction_of_filtered": round(n_va / n_all, 6) if n_all else None,
                "train_to_val_ratio": round(n_tr / n_va, 6) if n_va else None,
            }
            (pair_dir / "pair_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            model_kwargs = build_model_kwargs(train_cfg, use_kinematics=use_kinematics)
            class_names = [neg, pos]
            tr = train_df.to_dict("records")
            va = val_df.to_dict("records")
            logger.info(
                "Training %s | neg=%s pos=%s | n_train=%d n_val=%d",
                pdir_name,
                neg,
                pos,
                len(tr),
                len(va),
            )
            split_tag = (
                f"stratified_group_val_frac_{vf_split:.4f}_cat_id_dlc_pairwise"
                if vf_split is not None
                else "stratified_group_first_fold_cat_id_dlc_pairwise"
            )
            t0 = time.time()
            out = train_single_split(
                P6PoseSTGCNDeepLabCut,
                model_kwargs,
                tr,
                va,
                pair_train_dir,
                train_cfg,
                device,
                logger,
                class_names=class_names,
                use_kinematics=use_kinematics,
                split_tag=split_tag,
                build_dataloaders_fn=build_dlc_dataloaders,
            )
            logger.info("Finished %s in %.1f s", pdir_name, time.time() - t0)
            row = make_master_row(
                out,
                class_lo=lo,
                class_hi=hi,
                class_neg=neg,
                class_pos=pos,
                output_dir=pair_dir,
                master_meta=master_meta,
            )
            all_results.append(row)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — writing partial results.")
        save_master_results(all_results, run_dir, start_time, len(pairs))
        if all_results:
            write_pairwise_report(run_dir, all_results)
        raise SystemExit(130)

    save_master_results(all_results, run_dir, start_time, len(pairs))
    write_pairwise_report(run_dir, all_results)
    if all_results:
        print(f"Wrote: {run_dir / 'master_results.csv'}, {run_dir / 'aggregated_report.txt'}")


if __name__ == "__main__":
    main()
