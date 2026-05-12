"""
Train P4 (17-keypoint ST-GCN) on **every unordered pair** of 10 original audio
classes (``audio_label_10``): C(10,2)=**45** binary tasks, one
stratified group train/val split per pair (same contract as
``run_stgcn_dlc_pairwise.py``).

Label remapping: :func:`pair_binary_neg_pos` (Paining positive when in pair; else
lexicographic 0/1). Training uses ``training_loop.train_single_split`` with
``binary_only: true`` and a 2-class unused multiclass head (n_classes=2).

**Config** — use ``model_training_v2/config_p4_pairwise.yaml`` (small LR, batch
4, 120 epochs, early stopping patience 30).

Usage:
  python model_training_v2/run_p4_pairwise.py --config model_training_v2/config_p4_pairwise.yaml
  python model_training_v2/run_p4_pairwise.py --pair Happy,Paining --device cuda
  python model_training_v2/run_p4_pairwise.py --dry-run
  python model_training_v2/run_p4_pairwise.py --limit-pairs 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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
    load_dataset,
    plot_dataset_overview,
    print_dataset_statistics,
)
from models.p4_pose_stgcn import  P4PoseSTGCN
from run_stgcn_deeplabcut_train import  setup_logger
from run_stgcn_dlc_pairwise import  (
    make_master_run_dir,
    make_master_row,
    save_master_results,
    write_pairwise_report,
    _parse_single_pair,
)


def train_p4_one_pair(
    lo: str,
    hi: str,
    df: pd.DataFrame,
    base_cfg: dict,
    parent_run_dir: Path,
    device,
    logger,
    *,
    use_kinematics: bool,
    min_pair: int,
    pain_class: str,
    label_field: str,
    master_meta: dict,
    dry_run: bool = False,
    skip_if_exists: bool = False,
) -> dict | None:
    """
    Run one C(K,2) P4 binary task: filter ``df`` to the two classes, split, train.

    ``base_cfg`` is the on-disk YAML (without ``binary_only: true``); a deep copy
    is made and ``training.binary_only`` is set. ``parent_run_dir`` is the
    run root (same as ``make_master_run_dir`` return value); this function writes
    ``{parent}/binary__A__B/``.
    """
    from models.p4_pose_stgcn import  P4PoseSTGCN
    from training_loop import  train_single_split

    train_cfg = copy.deepcopy(base_cfg)
    train_cfg.setdefault("training", {})["binary_only"] = True
    if str(train_cfg.get("training", {}).get("early_stop_metric", "")).lower() in (
        "val_macro_f1",
        "macro_f1",
    ):
        train_cfg["training"]["early_stop_metric"] = "val_macro_f1_binary"
    pdir_name = pair_to_dirname(lo, hi)
    pair_dir = parent_run_dir / pdir_name
    train_dir = pair_dir / "training"
    if dry_run:
        sub = filter_remap_binary_pair(df, lo, hi, label_field=label_field, pain_class=pain_class)
        if len(sub) < min_pair:
            logger.warning("dry: %s too few rows (%d < %d)", pdir_name, len(sub), min_pair)
            return None
        tr, va, err = first_stratified_group_split(sub, base_cfg)
        if err:
            logger.warning("dry: %s split: %s", pdir_name, err)
            return None
        bs = int(base_cfg["training"]["batch_size"])
        ok, reason = split_viable(tr, va, 2, batch_size=bs)
        logger.info("dry: %s viable=%s n_tr=%d n_val=%d %s", pdir_name, ok, len(tr), len(va), reason or "")
        return None
    if skip_if_exists and (train_dir / "run_summary.json").is_file():
        logger.info("skip existing: %s", pair_dir)
        return None
    neg, pos = pair_binary_neg_pos(lo, hi, pain_class=pain_class)
    sub = filter_remap_binary_pair(df, lo, hi, label_field=label_field, pain_class=pain_class)
    if len(sub) < min_pair:
        logger.warning("%s: too few rows (%d < %d)", pdir_name, len(sub), min_pair)
        return None
    train_df, val_df, err = first_stratified_group_split(sub, base_cfg)
    if err:
        logger.warning("%s: split: %s", pdir_name, err)
        return None
    batch_size = int(train_cfg["training"]["batch_size"])
    ok, reason = split_viable(train_df, val_df, 2, batch_size=batch_size)
    if not ok:
        logger.warning("%s: not viable: %s", pdir_name, reason)
        return None
    pair_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    n_tr, n_va = len(train_df), len(val_df)
    n_all = n_tr + n_va
    vf = split_val_fraction_from_cfg(base_cfg)
    meta: dict = {
        "class_lo": lo,
        "class_hi": hi,
        "class_neg_label_int_0": neg,
        "class_pos_label_int_1": pos,
        "pain_class": pain_class,
        "label_field": label_field,
        "n_filtered": int(len(sub)),
        "n_train": int(n_tr),
        "n_val": int(n_va),
        "val_fraction_target": vf,
    }
    if n_all:
        meta["val_fraction_achieved"] = round(n_va / n_all, 6)
    (pair_dir / "pair_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    model_kwargs = build_p4_model_kwargs(train_cfg, use_kinematics=use_kinematics)
    tr_recs = train_df.to_dict("records")
    va_recs = val_df.to_dict("records")
    logger.info(
        "Train %s | neg=%s pos=%s | n_tr=%d n_val=%d",
        pdir_name,
        neg,
        pos,
        len(tr_recs),
        len(va_recs),
    )
    split_tag = (
        f"stratified_group_val_frac_{vf:.4f}_p4_pairwise" if vf is not None else "p4_pairwise_first_fold"
    )
    out = train_single_split(
        P4PoseSTGCN,
        model_kwargs,
        tr_recs,
        va_recs,
        train_dir,
        train_cfg,
        device,
        logger,
        class_names=[neg, pos],
        use_kinematics=use_kinematics,
        split_tag=split_tag,
    )
    return make_master_row(
        out,
        class_lo=lo,
        class_hi=hi,
        class_neg=neg,
        class_pos=pos,
        output_dir=pair_dir,
        master_meta=master_meta,
    )


def build_p4_model_kwargs(cfg: dict, *, use_kinematics: bool) -> dict:
    """
    n_classes=2 when training.binary_only (pairwise), else |multiclass| from config.
    """
    nc = int(cfg["data"]["n_channels"]) * (2 if use_kinematics else 1)
    bo = bool(cfg.get("training", {}).get("binary_only", False))
    n_cls = 2 if bo else len(get_multiclass_class_names(cfg))
    return {
        "n_frames": int(cfg["data"]["n_frames"]),
        "n_keypoints": int(cfg["data"]["n_keypoints"]),
        "n_channels": nc,
        "n_classes": n_cls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 pose: binary one-vs-one for all 10-class pairs.")
    parser.add_argument("--config", default="video/pose-models/config_p4_pairwise.yaml")
    parser.add_argument("--experiment-name", default="p4_pairwise_10class")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-kinematics", action="store_true")
    parser.add_argument(
        "--pair",
        metavar="A,B",
        help="Run a single pair (comma-separated). Default: all C(K,2) from labels.classes.",
    )
    parser.add_argument("--limit-pairs", type=int, default=None, help="Stop after this many pairs (lex order).")
    parser.add_argument(
        "--min-pair-rows",
        type=int,
        default=None,
        help="Skip if fewer than this many rows in pair (default: max(24, 2*batch+4)).",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip if training/run_summary.json exists.")
    parser.add_argument("--audio-confidence-field", default="audio_confidence")
    parser.add_argument("--min-audio-confidence", type=float, default=0.7)
    parser.add_argument(
        "--audio-confidence-filter",
        action="store_true",
        help="Optional strict YAMNet-style score filter before pairing.",
    )
    parser.add_argument("--split-seed", type=int, default=None, help="Override config split.random_state.")
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    if args.split_seed is not None:
        cfg = copy.deepcopy(cfg)
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
                raise SystemExit(f"Unknown class in --pair; use names from config: {lo!r}, {hi!r}")
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
    cfg_to_save["p4_pairwise"] = {
        "min_audio_confidence": None if not filter_enabled else float(args.min_audio_confidence),
        "confidence_field": conf_field,
        "filter_enabled": filter_enabled,
        "pairs": [f"{a}|{b}" for a, b in pairs],
        "min_pair_rows": min_pair,
    }
    (run_dir / "config_used.yaml").write_text(
        yaml.dump(cfg_to_save, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    master_meta: dict = {
        "min_audio_confidence": None if not filter_enabled else float(args.min_audio_confidence),
        "audio_confidence_field": conf_field,
        "audio_confidence_filter": filter_enabled,
    }

    logger.info(
        "P4 pairwise | %d pairs | label_field=%s | pain_class=%s | min_pair_rows=%d",
        len(pairs),
        label_field,
        pain_class,
        min_pair,
    )

    df = load_dataset(load_cfg, logger)
    print_dataset_statistics(df, logger, load_cfg)
    plot_dataset_overview(df, run_dir / "plots", load_cfg)

    if filter_enabled:
        n0 = len(df)
        df = filter_dataframe_by_audio_confidence(
            df, conf_field, float(args.min_audio_confidence), logger=logger
        )
        logger.info("Audio filter: %d rows (was %d).", len(df), n0)
        if len(df) < 1:
            logger.error("No rows after audio filter.")
            raise SystemExit(1)
        print_dataset_statistics(df, logger, load_cfg)

    with open(run_dir / "reports" / "pair_list.json", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
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
                indent=2,
            )
        )

    if args.dry_run:
        for lo, hi in pairs:
            sub = filter_remap_binary_pair(df, lo, hi, label_field=label_field, pain_class=pain_class)
            logger.info("pair %s vs %s → n=%d", lo, hi, len(sub))
            if len(sub) < min_pair:
                logger.warning("  skip: n < min_pair_rows=%d", min_pair)
                continue
            tr, va, err = first_stratified_group_split(sub, cfg)
            if err:
                logger.warning("  split: %s", err)
                continue
            ok, reason = split_viable(tr, va, 2, batch_size=batch_size)
            logger.info("  viable=%s  n_tr=%d n_val=%d  %s", ok, len(tr), len(va), reason or "")
        logger.info("Dry run done.")
        return

    import torch

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    all_results: list[dict] = []
    n_pairs_total = len(pairs)
    try:
        for pi, (lo, hi) in enumerate(pairs):
            logger.info("=== pair %d / %d: %s vs %s ===", pi + 1, n_pairs_total, lo, hi)
            pdir_name = pair_to_dirname(lo, hi)
            t0 = time.time()
            row = train_p4_one_pair(
                lo,
                hi,
                df,
                cfg,
                run_dir,
                device,
                logger,
                use_kinematics=use_kinematics,
                min_pair=min_pair,
                pain_class=pain_class,
                label_field=label_field,
                master_meta=master_meta,
                dry_run=False,
                skip_if_exists=args.skip_existing,
            )
            if row is not None:
                logger.info(
                    "Finished %s in %.1f s | best_epoch=%s val_macro_f1_binary=%s",
                    pdir_name,
                    time.time() - t0,
                    row.get("best_epoch"),
                    row.get("val_macro_f1_binary"),
                )
                all_results.append(row)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — writing partial results.")
        save_master_results(all_results, run_dir, start_time, len(pairs))
        if all_results:
            write_pairwise_report(
                run_dir, all_results, report_title="P4 (17-kp) binary pairwise — comparison report"
            )
        raise SystemExit(130)

    save_master_results(all_results, run_dir, start_time, len(pairs))
    write_pairwise_report(
        run_dir, all_results, report_title="P4 (17-kp) binary pairwise — comparison report"
    )
    if all_results:
        print(f"Wrote: {run_dir / 'master_results.csv'}")


if __name__ == "__main__":
    main()
