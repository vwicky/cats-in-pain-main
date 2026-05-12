"""
Outer loop: trains all registered models across all data fractions.
Produces scaling curves showing how metrics grow with data volume.

Usage:
  python model_training_v2/run_scaling_experiment.py
  python model_training_v2/run_scaling_experiment.py --models dummy
  python model_training_v2/run_scaling_experiment.py --fractions 0.5 1.0
  python model_training_v2/run_scaling_experiment.py --dry-run
  python model_training_v2/run_scaling_experiment.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_loading import  get_multiclass_class_names
from models import  DEFAULT_MODEL_IDS, MODEL_REGISTRY


def make_run_dir(cfg: dict, experiment_name: str) -> Path:
    """
    Create and return:
      model_training_v2/runs/{experiment_name}_{YYYYMMDD_HHMMSS}/
    With subdirectories:
      plots/
      reports/ (optional)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPO_ROOT / cfg["output"]["runs_dir"]
    run_dir = base / f"{experiment_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    """Dual handler: console INFO + file DEBUG."""
    log_path = run_dir / "experiment.log"
    logger = logging.getLogger("scaling_experiment")
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


def build_model_kwargs(cfg: dict, use_kinematics: bool = True) -> dict:
    """
    Build kwargs for model instantiation from config:
      n_frames=cfg.data.n_frames,
      n_keypoints=cfg.data.n_keypoints,
      n_channels=cfg.data.n_channels * (2 if kinematics else 1),
      n_classes=len(labels.classes) or len(labels.classes_5) via get_multiclass_class_names
    """
    nc = int(cfg["data"]["n_channels"]) * (2 if use_kinematics else 1)
    n_cls = len(get_multiclass_class_names(cfg))
    return {
        "n_frames": int(cfg["data"]["n_frames"]),
        "n_keypoints": int(cfg["data"]["n_keypoints"]),
        "n_channels": nc,
        "n_classes": n_cls,
    }


def plot_scaling_curves(
    results_df: pd.DataFrame,
    plots_dir: Path,
    *,
    title_prefix: str = "",
):
    """
    Scaling curve plots (macro F1, sensitivity, 2x2 metrics, per-class F1).

    ``plots_dir`` is the directory that receives the PNG files (e.g. per-model
    ``run_dir/P0/plots`` or aggregated ``run_dir/plots``).
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    from data_loading import  LABEL_COLORS

    if results_df.empty:
        return

    tp = f"{title_prefix} " if title_prefix else ""
    models = results_df["model_id"].unique()
    palette = sns.color_palette("husl", max(len(models), 1))

    def _line(metric_mean: str, metric_std: str, title: str, fname: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, mid in enumerate(models):
            sub = results_df[results_df["model_id"] == mid].sort_values("fraction")
            x = sub["fraction"].values
            y = sub[metric_mean].values
            s = sub[metric_std].values if metric_std in sub.columns else np.zeros(len(y))
            name = sub["model_name"].iloc[0] if "model_name" in sub.columns else mid
            ax.plot(x, y, marker="o", label=name, color=palette[i % len(palette)])
            ax.fill_between(x, y - s, y + s, alpha=0.15, color=palette[i % len(palette)])
        ax.set_xlabel("Training data fraction")
        ax.set_ylabel(metric_mean.replace("mean_", ""))
        ax.set_title(f"{tp}{title}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _line(
        "mean_macro_f1_5",
        "std_macro_f1_5",
        "5-Class Macro F1 vs Training Data Fraction",
        "scaling_macro_f1_5.png",
    )
    _line(
        "mean_sensitivity",
        "std_sensitivity",
        "Pain Sensitivity vs Training Data Fraction",
        "scaling_sensitivity.png",
    )

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("mean_macro_f1_5", "std_macro_f1_5", "Macro F1 (5-class)"),
        ("mean_macro_f1_binary", "std_macro_f1_binary", "Macro F1 (binary)"),
        ("mean_sensitivity", "std_sensitivity", "Sensitivity"),
        ("mean_specificity", "std_specificity", "Specificity"),
    ]
    for ax, (mm, sm, ttl) in zip(axes.ravel(), metrics):
        for i, mid in enumerate(models):
            sub = results_df[results_df["model_id"] == mid].sort_values("fraction")
            x = sub["fraction"].values
            y = sub[mm].values if mm in sub.columns else np.zeros(len(x))
            s = sub[sm].values if sm in sub.columns else np.zeros(len(x))
            name = sub["model_name"].iloc[0] if "model_name" in sub.columns else mid
            ax.plot(x, y, marker="o", label=name, color=palette[i % len(palette)])
            ax.fill_between(x, y - s, y + s, alpha=0.12, color=palette[i % len(palette)])
        ax.set_title(f"{tp}{ttl}")
        ax.set_xlabel("Fraction")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{tp}Scaling — key metrics")
    fig.tight_layout()
    fig.savefig(plots_dir / "scaling_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-class F1
    classes = ["Paining", "Positive_Baseline", "Agonistic", "Vocalizing", "HuntingMind"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for j, cname in enumerate(classes):
        ax = axes[j]
        col_m = f"mean_per_class_f1_{cname}"
        col_s = f"std_per_class_f1_{cname}"
        for i, mid in enumerate(models):
            sub = results_df[results_df["model_id"] == mid].sort_values("fraction")
            x = sub["fraction"].values
            y = sub[col_m].values if col_m in sub.columns else np.zeros(len(x))
            s = sub[col_s].values if col_s in sub.columns else np.zeros(len(x))
            nm = sub["model_name"].iloc[0] if "model_name" in sub.columns else mid
            lw = 2.5 if cname == "Paining" else 1.2
            ax.plot(x, y, marker="o", label=nm, color=palette[i % len(palette)], linewidth=lw)
            ax.fill_between(x, y - s, y + s, alpha=0.1, color=palette[i % len(palette)])
        if cname == "Paining":
            for spine in ax.spines.values():
                spine.set_edgecolor(LABEL_COLORS.get("Paining", "red"))
                spine.set_linewidth(2.5)
        ax.set_title(f"{tp}{cname}")
        ax.set_xlabel("Fraction")
        ax.set_ylabel("F1")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "scaling_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_master_results(results: list[dict], run_dir: Path, start_time: float, models_run: list[str]):
    """Save master_results.csv / .json and print summary table."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(run_dir / "master_results.csv", index=False)
        with open(run_dir / "master_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    n_cv = len(results) * 5 if results else 0  # approximate

    frac_set = {r.get("fraction") for r in results}
    print(
        f"""
════════════════════════════════════════════════════════════
SCALING EXPERIMENT COMPLETE
Run: {run_dir.name}
Models: {len(models_run)} | Fractions: {len(frac_set)} | Total CV rows: {len(results)}
Elapsed: {h}h {m}m {s}s
════════════════════════════════════════════════════════════
"""
    )

    sub = [r for r in results if abs(float(r.get("fraction", 0)) - 1.0) < 1e-6]
    if sub:
        print("BEST RESULTS AT 100% DATA (mean over 5 folds)")
        print("──────────────────────────────────────────────────────────")
        print(f"{'Model':<14} │ {'MacroF1-5':^9} │ {'Sensitivity':^11} │ {'Specificity':^11} │ {'AUC-ROC':^7}")
        print("─────────────┼───────────┼─────────────┼─────────────┼─────────")
        for r in sub:
            mn = r.get("model_name", r.get("model_id", "?"))
            print(
                f"{str(mn):<14} │ {r.get('mean_macro_f1_5', 0):^9.3f} │ "
                f"{r.get('mean_sensitivity', 0):^11.3f} │ {r.get('mean_specificity', 0):^11.3f} │ "
                f"{r.get('mean_auc_roc', float('nan')):^7.3f}"
            )

        print("\nPAINING CLASS F1 AT 100% DATA")
        print("──────────────────────────────")
        print(f"{'Model':<14} │ {'Precision':^9} │ {'Recall':^7} │ {'F1':^5}")
        for r in sub:
            mn = r.get("model_name", r.get("model_id", "?"))
            print(
                f"{str(mn):<14} │ {r.get('mean_per_class_precision_Paining', 0):^9.3f} │ "
                f"{r.get('mean_per_class_recall_Paining', 0):^7.3f} │ {r.get('mean_per_class_f1_Paining', 0):^5.3f}"
            )
        print("════════════════════════════════════════════════════════════\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="video/pose-models/config.yaml")
    parser.add_argument("--models", nargs="+", help="Model IDs to run. Default: all registered.")
    parser.add_argument("--fractions", nargs="+", type=float, help="Data fractions to run. Default: from config.")
    parser.add_argument("--experiment-name", default="scaling_experiment")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print plan, no training.")
    parser.add_argument("--no-kinematics", action="store_true", help="Use 3 channels (no velocity).")
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    use_kinematics = not args.no_kinematics

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

    run_dir = make_run_dir(cfg, args.experiment_name)
    logger = setup_logger(run_dir)
    start_time = time.time()

    (run_dir / "config_used.yaml").write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    models_to_run: list[str] = list(args.models) if args.models else list(DEFAULT_MODEL_IDS)
    fractions = args.fractions or cfg["scaling"]["fractions"]
    n_folds = int(cfg["split"]["n_folds"])
    n_repeats = int(cfg["scaling"].get("n_repeats", 1))
    total_runs = len(models_to_run) * len(fractions) * n_folds * n_repeats

    print(
        f"""
╔══════════════════════════════════════════════════════════╗
║  POSE MODEL SCALING EXPERIMENT                          ║
║  Run: {run_dir.name:<46}║
║  Device: {str(device):<44}║
║  Models: {len(models_to_run):<2} | Fractions: {len(fractions):<2} | Total CV runs: {total_runs:<5}║
╚══════════════════════════════════════════════════════════╝
    """
    )

    from data_loading import  load_dataset, plot_dataset_overview, print_dataset_statistics

    df = load_dataset(cfg, logger)
    print_dataset_statistics(df, logger, cfg)
    plot_dataset_overview(df, run_dir / "plots", cfg)

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    from training_loop import  run_cv_sweep

    all_results: list[dict] = []
    try:
        for model_id in tqdm(models_to_run, desc="models"):
            if model_id not in MODEL_REGISTRY:
                logger.warning("Unknown model_id: %s — skipping", model_id)
                continue

            model_class, extra_kwargs = MODEL_REGISTRY[model_id]
            model_kwargs = {**build_model_kwargs(cfg, use_kinematics), **extra_kwargs}
            probe = model_class(**model_kwargs)
            mname = probe.model_name

            for repeat_idx in range(n_repeats):
                for fraction in tqdm(fractions, desc=f"{model_id} fr", leave=False):
                    pct = int(round(fraction * 100))
                    logger.info("%s", "=" * 60)
                    logger.info("Model: %s | Fraction: %d%% | repeat %d", model_id, pct, repeat_idx)
                    logger.info("%s", "=" * 60)

                    frac_run_dir = run_dir / model_id / f"fraction_{pct:03d}"
                    if n_repeats > 1:
                        frac_run_dir = frac_run_dir / f"repeat_{repeat_idx:02d}"
                    frac_run_dir.mkdir(parents=True, exist_ok=True)

                    result = run_cv_sweep(
                        model_class=model_class,
                        model_kwargs=model_kwargs,
                        df=df,
                        fraction=fraction,
                        run_dir=frac_run_dir,
                        cfg=cfg,
                        device=device,
                        logger=logger,
                        use_kinematics=use_kinematics,
                        repeat_idx=repeat_idx,
                    )
                    result["model_id"] = model_id
                    result["model_name"] = mname
                    result["fraction"] = fraction
                    result["repeat_idx"] = repeat_idx
                    all_results.append(result)

                    logger.info(
                        "Done: %s @ %s%% | MacroF1=%.3f ± %.3f | Sens=%.3f",
                        model_id,
                        pct,
                        result.get("mean_macro_f1_5", 0),
                        result.get("std_macro_f1_5", 0),
                        result.get("mean_sensitivity", 0),
                    )

            model_results = [r for r in all_results if r.get("model_id") == model_id]
            model_plots = run_dir / model_id / "plots"
            model_plots.mkdir(parents=True, exist_ok=True)
            mdf = pd.DataFrame(model_results)
            if not mdf.empty:
                mdf.to_csv(run_dir / model_id / "model_results.csv", index=False)
                plot_scaling_curves(mdf, model_plots, title_prefix=f"{model_id}")
                logger.info("Wrote scaling plots for %s → %s", model_id, model_plots)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving partial results.")
        save_master_results(all_results, run_dir, start_time, models_to_run)
        raise SystemExit(130)

    results_df = pd.DataFrame(all_results)
    save_master_results(all_results, run_dir, start_time, models_to_run)
    if not results_df.empty:
        plot_scaling_curves(results_df, run_dir / "plots", title_prefix="All models")


if __name__ == "__main__":
    main()
