"""
Train ST-GCN–DeepLabCut (P6) on DeepLabCut HDF5 poses — one stratified group split on cat_id.

Usage:
  python model_training_v2/run_stgcn_deeplabcut_train.py --dry-run
  python model_training_v2/run_stgcn_deeplabcut_train.py --inspect-bodyparts dataset/deeplabcut_really_labeled/foo.h5
  python model_training_v2/run_stgcn_deeplabcut_train.py --config model_training_v2/config_stgcn_dlc.yaml --device cuda

``--dry-run`` avoids importing PyTorch so you can validate paths, manifest, clip QC,
and bodypart names in any environment with pandas + yaml + numpy only.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from class_subset_utils import  (
    first_stratified_group_split,
    split_val_fraction_from_cfg,
    split_viable,
)
from data_loading import  (
    get_multiclass_class_names,
    load_dataset_for_deeplabcut,
    plot_dataset_overview,
    print_dataset_statistics,
)
from deeplabcut_pose_io import  (
    compare_bodyparts_to_spec,
    dlc_expected_num_keypoints,
    list_hdf_bodypart_names,
    passes_clip_quality_hdf5,
)


def make_run_dir(cfg: dict, experiment_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPO_ROOT / cfg["output"]["runs_dir"]
    run_dir = base / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "training").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    log_path = run_dir / "experiment.log"
    logger = logging.getLogger("stgcn_dlc_train")
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


def build_model_kwargs(cfg: dict, *, use_kinematics: bool) -> dict:
    nc = int(cfg["data"]["n_channels"]) * (2 if use_kinematics else 1)
    bo = bool(cfg.get("training", {}).get("binary_only", False))
    n_cls = 2 if bo else len(get_multiclass_class_names(cfg))
    return {
        "n_frames": int(cfg["data"]["n_frames"]),
        "n_keypoints": int(cfg["data"]["n_keypoints"]),
        "n_channels": nc,
        "n_classes": n_cls,
        "binary_only": bo,
    }


def _clip_q_cfg(cfg: dict) -> dict:
    return cfg.get("clip_quality") or {}


def apply_clip_quality_filter(
    df: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict]:
    """Filter rows by DLC clip confidence; returns (df_kept, stats dict)."""
    cq = _clip_q_cfg(cfg)
    mode = str(cq.get("on_low_confidence", "exclude")).strip().lower()
    mean_thr = float(cq.get("clip_mean_likelihood_threshold", 0.4))
    frac_thr = float(cq.get("min_fraction_high_conf_frames", 0.6))
    pft = cq.get("per_frame_mean_likelihood_threshold")
    pft_f = float(pft) if pft is not None else None

    if mode == "mask_only":
        logger.warning("on_low_confidence=mask_only is not implemented; keeping all rows.")
        mode = "keep"

    stats: dict = {
        "mode": mode,
        "mean_threshold": mean_thr,
        "fraction_threshold": frac_thr,
        "per_frame_threshold": pft_f if pft_f is not None else mean_thr,
    }

    if mode == "keep":
        stats["n_kept"] = len(df)
        stats["n_excluded"] = 0
        return df, stats

    logger.info("Clip QC (exclude): using likelihood-only HDF5 scan (no full 39-keypoint load).")
    h5_col = "dlc_h5_path"
    keep_mask = []
    for _, row in df.iterrows():
        rel = row.get(h5_col)
        path = REPO_ROOT / str(rel) if rel else None
        if path is None or not path.is_file():
            keep_mask.append(False)
            continue
        ok = passes_clip_quality_hdf5(
            path,
            clip_mean_likelihood_threshold=mean_thr,
            min_fraction_high_conf_frames=frac_thr,
            per_frame_threshold=pft_f,
        )
        keep_mask.append(ok)

    km = np.array(keep_mask, dtype=bool)
    out = df.loc[km].copy().reset_index(drop=True)
    stats["n_excluded"] = int((~km).sum())
    stats["n_kept"] = int(km.sum())

    ex = df.loc[~km]
    if len(ex) > 0 and "binary_label_int" in ex.columns:
        pain_ex = int((ex["binary_label_int"] == 1).sum())
        nop_ex = int((ex["binary_label_int"] == 0).sum())
        stats["excluded_pain"] = pain_ex
        stats["excluded_no_pain"] = nop_ex
    else:
        stats["excluded_pain"] = 0
        stats["excluded_no_pain"] = 0

    if len(df) > 0 and "binary_label_int" in df.columns:
        pain_all = int((df["binary_label_int"] == 1).sum())
        nop_all = int((df["binary_label_int"] == 0).sum())
        stats["exclusion_rate_pain"] = stats["excluded_pain"] / pain_all if pain_all else 0.0
        stats["exclusion_rate_no_pain"] = stats["excluded_no_pain"] / nop_all if nop_all else 0.0
        rp = stats["exclusion_rate_pain"]
        rn = stats["exclusion_rate_no_pain"]
        if rn > 1e-6 and rp / rn >= 3.0:
            stats["imbalance_warning"] = (
                f"Pain clips excluded at {rp / rn:.1f}× the No_Pain rate — check selection bias for the thesis."
            )
        elif rp > 1e-6 and rn / rp >= 3.0:
            stats["imbalance_warning"] = (
                f"No_Pain clips excluded at {rn / rp:.1f}× the Pain rate — check selection bias for the thesis."
            )
        else:
            stats["imbalance_warning"] = None

    return out, stats


def _bodypart_audit_text(sample_h5: Path) -> str:
    cmp = compare_bodyparts_to_spec(sample_h5)
    lines = [
        "",
        "Bodypart alignment (sample HDF5)",
        "--------------------------------",
        f"Sample file: {sample_h5}",
        f"Matched {cmp['matched_count']} / {dlc_expected_num_keypoints()} spec keypoints to HDF columns.",
    ]
    if cmp["missing_in_hdf"]:
        lines.append(f"missing_in_hdf ({len(cmp['missing_in_hdf'])}): {cmp['missing_in_hdf'][:12]!r}")
        if len(cmp["missing_in_hdf"]) > 12:
            lines.append("  ...")
    if cmp["extra_in_hdf"]:
        lines.append(f"extra_in_hdf ({len(cmp['extra_in_hdf'])}): {cmp['extra_in_hdf'][:12]!r}")
        if len(cmp["extra_in_hdf"]) > 12:
            lines.append("  ...")
    lines.append(f"hdf_raw_bodyparts ({len(cmp['hdf_raw'])}): {cmp['hdf_raw']!r}")
    return "\n".join(lines) + "\n"


def write_data_filter_summary(
    run_dir: Path,
    clip_stats: dict,
    *,
    n_rows_with_h5: int,
    bodypart_audit: str = "",
) -> None:
    lines = [
        "DeepLabCut ST-GCN data filter summary",
        "====================================",
        "",
        f"n_rows_with_resolved_h5 (before clip QC): {n_rows_with_h5}",
        f"clip_quality.mode: {clip_stats.get('mode')}",
        f"n_kept_after_clip_q: {clip_stats.get('n_kept')}",
        f"n_excluded_low_confidence: {clip_stats.get('n_excluded')}",
        f"excluded_pain / excluded_no_pain: {clip_stats.get('excluded_pain')} / {clip_stats.get('excluded_no_pain')}",
        f"exclusion_rate_pain: {clip_stats.get('exclusion_rate_pain', 0):.4f}",
        f"exclusion_rate_no_pain: {clip_stats.get('exclusion_rate_no_pain', 0):.4f}",
    ]
    w = clip_stats.get("imbalance_warning")
    if w:
        lines += ["", "WARNING:", w]
    lines.append("")
    body = "\n".join(lines) + "\n"
    if bodypart_audit.strip():
        body += bodypart_audit
    (run_dir / "reports" / "data_filter_summary.txt").write_text(body, encoding="utf-8")

    thesis = run_dir / "reports" / "thesis_notes.txt"
    thesis.write_text(
        "Methods notes (single split + clip QC)\n"
        "========================================\n"
        "- Validation metrics use one stratified group hold-out by cat_id; variance is higher than k-fold CV.\n"
        "- If clip_quality exclusions skew by pain label, the filtered train/val set may not match raw label prevalence.\n",
        encoding="utf-8",
    )


def _run_split_audit(df_filt: pd.DataFrame, cfg: dict, logger: logging.Logger) -> None:
    train_df, val_df, err = first_stratified_group_split(df_filt, cfg)
    if err:
        logger.warning("Stratified group split (audit): %s", err)
        return
    bs = int(cfg["training"]["batch_size"])
    k_mc = 2 if bool(cfg.get("training", {}).get("binary_only", False)) else len(get_multiclass_class_names(cfg))
    ok, reason = split_viable(train_df, val_df, k_mc, batch_size=bs)
    if ok:
        vf = split_val_fraction_from_cfg(cfg)
        if vf is not None:
            got = len(val_df) / max(len(train_df) + len(val_df), 1)
            logger.info(
                "Split audit OK: n_train=%d n_val=%d (val_fraction target=%.4f, achieved=%.4f).",
                len(train_df),
                len(val_df),
                vf,
                got,
            )
        else:
            logger.info(
                "Split audit OK: n_train=%d n_val=%d (first stratified group fold).",
                len(train_df),
                len(val_df),
            )
    else:
        logger.warning("Split audit: not viable — %s", reason)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="video/pose-models/config_stgcn_dlc.yaml")
    parser.add_argument("--experiment-name", default="stgcn_deeplabcut")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-kinematics", action="store_true")
    parser.add_argument(
        "--inspect-bodyparts",
        metavar="PATH.h5",
        help="Print bodypart level from one HDF5 and exit (no training). Repo-relative or absolute.",
    )
    args = parser.parse_args()

    if args.inspect_bodyparts:
        p = Path(args.inspect_bodyparts)
        if not p.is_file():
            p = REPO_ROOT / args.inspect_bodyparts
        if not p.is_file():
            raise SystemExit(f"File not found: {args.inspect_bodyparts}")
        raw = list_hdf_bodypart_names(p)
        print("Unique bodyparts in HDF5 (raw strings):", raw)
        cmp = compare_bodyparts_to_spec(p)
        print("matched_count:", cmp["matched_count"])
        print("missing_in_hdf (would zero-fill):", cmp["missing_in_hdf"])
        print("extra_in_hdf (unused columns):", cmp["extra_in_hdf"])
        return

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    use_kinematics = not args.no_kinematics

    run_dir = make_run_dir(cfg, args.experiment_name)
    logger = setup_logger(run_dir)
    (run_dir / "config_used.yaml").write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    df = load_dataset_for_deeplabcut(cfg, logger)
    print_dataset_statistics(df, logger, cfg)
    plot_dataset_overview(df, run_dir / "plots", cfg)

    bodypart_audit = ""
    if len(df) > 0 and df["dlc_h5_path"].iloc[0]:
        sample = REPO_ROOT / str(df["dlc_h5_path"].iloc[0])
        if sample.is_file():
            bodypart_audit = _bodypart_audit_text(sample)
            logger.info("Bodypart audit (first row):\n%s", bodypart_audit.strip())

    n_h5 = len(df)
    df_filt, clip_stats = apply_clip_quality_filter(df, cfg, logger)
    write_data_filter_summary(run_dir, clip_stats, n_rows_with_h5=n_h5, bodypart_audit=bodypart_audit)
    logger.info("After clip quality: %d rows (was %d)", len(df_filt), len(df))

    _run_split_audit(df_filt, cfg, logger)

    if args.dry_run:
        logger.info("Dry run — skipping PyTorch import and training.")
        return

    if len(df_filt) < 2:
        logger.error("Too few rows after clip QC to train.")
        raise SystemExit(1)

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

    train_df, val_df, err = first_stratified_group_split(df_filt, cfg)
    if err:
        logger.error("Split failed: %s", err)
        raise SystemExit(1)

    bs = int(cfg["training"]["batch_size"])
    k_mc = 2 if bool(cfg.get("training", {}).get("binary_only", False)) else len(get_multiclass_class_names(cfg))
    ok, reason = split_viable(train_df, val_df, k_mc, batch_size=bs)
    if not ok:
        logger.error("Split not viable: %s", reason)
        raise SystemExit(1)

    model_kwargs = build_model_kwargs(cfg, use_kinematics=use_kinematics)
    tr = train_df.to_dict("records")
    va = val_df.to_dict("records")
    logger.info("Training P6 | n_train=%d n_val=%d", len(tr), len(va))

    t0 = time.time()
    bo = bool(cfg.get("training", {}).get("binary_only", False))
    train_single_split(
        P6PoseSTGCNDeepLabCut,
        model_kwargs,
        tr,
        va,
        run_dir / "training",
        cfg,
        device,
        logger,
        class_names=["No_Pain", "Pain"] if bo else get_multiclass_class_names(cfg),
        use_kinematics=use_kinematics,
        split_tag="stratified_group_first_fold_cat_id_dlc",
        build_dataloaders_fn=build_dlc_dataloaders,
    )
    logger.info("Finished in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
