"""Evaluation metrics, plotting, and report generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_COLORS = {
    "Paining": "red",
    "Positive_Baseline": "green",
    "Agonistic": "orange",
    "Vocalizing": "purple",
    "HuntingMind": "blue",
    "Pain": "red",
    "No_Pain": "steelblue",
}

sns.set_theme(style="whitegrid")


# ── Metrics ─────────────────────────────────────────────────────────────


def compute_all_metrics(
    y_true_5: np.ndarray,
    y_pred_5: np.ndarray,
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    class_names: list[str],
    y_prob_binary: np.ndarray | None = None,
) -> dict:
    """Compute 5-class and binary metrics as a flat dict."""
    metrics: dict = {}

    # 5-class metrics
    metrics["accuracy_5"] = accuracy_score(y_true_5, y_pred_5)
    metrics["macro_f1_5"] = f1_score(y_true_5, y_pred_5, average="macro", zero_division=0)
    metrics["weighted_f1_5"] = f1_score(y_true_5, y_pred_5, average="weighted", zero_division=0)

    per_prec = precision_score(y_true_5, y_pred_5, average=None, zero_division=0, labels=range(len(class_names)))
    per_rec = recall_score(y_true_5, y_pred_5, average=None, zero_division=0, labels=range(len(class_names)))
    per_f1 = f1_score(y_true_5, y_pred_5, average=None, zero_division=0, labels=range(len(class_names)))

    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = float(per_prec[i]) if i < len(per_prec) else 0.0
        metrics[f"recall_{name}"] = float(per_rec[i]) if i < len(per_rec) else 0.0
        metrics[f"f1_{name}"] = float(per_f1[i]) if i < len(per_f1) else 0.0
        mask = y_true_5 == i
        metrics[f"support_{name}"] = int(mask.sum())

    # Binary metrics (Pain=1, No_Pain=0)
    metrics["accuracy_binary"] = accuracy_score(y_true_binary, y_pred_binary)
    metrics["macro_f1_binary"] = f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0)
    metrics["sensitivity"] = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)

    tn_mask = (y_true_binary == 0)
    if tn_mask.sum() > 0:
        metrics["specificity"] = float((y_pred_binary[tn_mask] == 0).sum()) / float(tn_mask.sum())
    else:
        metrics["specificity"] = 0.0

    metrics["precision_pain"] = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)

    if y_prob_binary is not None and len(np.unique(y_true_binary)) > 1:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true_binary, y_prob_binary)
        except ValueError:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    return metrics


# ── Plotting ────────────────────────────────────────────────────────────


def plot_training_curves(
    history: dict,
    fold: int,
    save_path: str,
    unfreeze_epoch: int | None = None,
    best_epoch: int | None = None,
) -> None:
    """4-panel training curves: loss, macro F1, sensitivity, specificity."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Fold {fold} Training Curves", fontsize=14, fontweight="bold")

    panels = [
        ("train_loss", "val_loss", "Loss (5-class focal)", axes[0, 0]),
        ("train_macro_f1", "val_macro_f1", "Macro F1 (5-class)", axes[0, 1]),
        ("train_sensitivity", "val_sensitivity", "Sensitivity (binary)", axes[1, 0]),
        ("train_specificity", "val_specificity", "Specificity (binary)", axes[1, 1]),
    ]

    for train_key, val_key, title, ax in panels:
        train_vals = history.get(train_key, [])
        val_vals = history.get(val_key, [])
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, label="Train", linewidth=1.5)
        ax.plot(epochs, val_vals, label="Val", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

        if best_epoch is not None:
            ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6, label="best")
        if unfreeze_epoch is not None:
            ax.axvline(unfreeze_epoch, color="gray", linestyle=":", alpha=0.6, label="unfreeze")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    save_path: str,
) -> None:
    """Normalized heatmap with both normalized values and raw counts."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_curves(
    all_fold_histories: list[dict],
    save_path: str,
) -> None:
    """4-panel mean +/- std shaded band across all folds."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Summary Across Folds (mean \u00b1 std)", fontsize=14, fontweight="bold")

    panels = [
        ("val_loss", "Val Loss", axes[0, 0]),
        ("val_macro_f1", "Val Macro F1", axes[0, 1]),
        ("val_sensitivity", "Val Sensitivity", axes[1, 0]),
        ("val_specificity", "Val Specificity", axes[1, 1]),
    ]

    for key, title, ax in panels:
        all_vals = []
        for h in all_fold_histories:
            vals = h.get(key, [])
            if vals:
                all_vals.append(vals)

        if not all_vals:
            ax.set_title(f"{title} (no data)")
            continue

        max_len = max(len(v) for v in all_vals)
        padded = np.full((len(all_vals), max_len), np.nan)
        for i, v in enumerate(all_vals):
            padded[i, :len(v)] = v

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(1, max_len + 1)

        ax.plot(epochs, mean, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Reports ─────────────────────────────────────────────────────────────


def write_per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str,
    fold: int = 0,
    model_name: str = "efficientnet_b3",
) -> None:
    """Write sklearn classification_report to a .txt file."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )
    lines = [
        "\u2550" * 45,
        "Per-Class Classification Report",
        f"Fold {fold} | Model: {model_name}",
        "\u2550" * 45,
        "",
        report,
    ]
    Path(save_path).write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    all_fold_metrics: list[dict],
    cfg: dict,
    run_dir: str,
    elapsed_sec: float,
    data_summary: dict | None = None,
) -> None:
    """Write final_report.txt with full results and comparison table."""
    from datetime import datetime

    n_folds = len(all_fold_metrics)
    classes = cfg["classes_5"]
    binary_map = cfg["binary_map"]

    def _mean_std(key: str) -> tuple[float, float]:
        vals = [m[key] for m in all_fold_metrics if key in m]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    elapsed_m = int(elapsed_sec // 60)
    elapsed_s = int(elapsed_sec % 60)
    date_str = datetime.now().strftime("%Y-%m-%d")

    lines: list[str] = []
    sep = "\u2550" * 56
    lines.append(sep)
    lines.append(f"TRAINING REPORT: {cfg['run_name']}")
    lines.append(f"Run directory: {run_dir}")
    lines.append(f"Date: {date_str} | Elapsed: {elapsed_m}m {elapsed_s}s")
    lines.append(sep)
    lines.append("")

    lines.append("CONFIGURATION")
    lines.append("\u2500" * 15)
    lines.append(f"Backbone:         EfficientNet-B3 ({cfg.get('backbone_weights', 'IMAGENET1K_V1')})")
    lines.append(f"Pooling:          {cfg.get('pooling', 'attention')} ({cfg.get('frames_per_clip', 3)} frames)")
    bbox_str = "yes" if cfg.get("use_bbox_crop") else "no"
    lines.append(f"Bbox crop:        {bbox_str} (YOLOv8x, conf={cfg.get('yolo_conf', 0.4)}, pad={int(cfg.get('bbox_padding', 0.2)*100)}%)")
    lines.append(f"Image size:       {cfg.get('image_size', 300)}\u00d7{cfg.get('image_size', 300)}")
    lines.append(f"Freeze epochs:    {cfg.get('freeze_backbone_epochs', 5)} | Unfreeze blocks: {cfg.get('unfreeze_blocks', 2)}")
    lines.append(f"Head hidden dim:  {cfg.get('head_hidden_dim', 256)} | Dropout: {cfg.get('head_dropout', 0.5)}")
    lines.append(f"Epochs:           {cfg.get('epochs', 60)} (early stop patience={cfg.get('early_stop_patience', 12)})")
    lines.append(f"Batch size:       {cfg.get('batch_size', 16)}")
    lines.append(f"LR backbone:      {cfg.get('lr_backbone', 1e-5)} | LR head: {cfg.get('lr_head', 1e-3)}")
    lines.append(f"Loss:             Focal (\u03b3={cfg.get('focal_gamma', 2.0)}) + 0.3\u00d7binary")
    lines.append(f"CV:               {n_folds}-fold StratifiedGroupKFold (cat_id groups)")
    lines.append("")

    if data_summary:
        lines.append("DATA")
        lines.append("\u2500" * 4)
        lines.append(f"Total clips:      {data_summary.get('total', 0)}")
        for cls in classes:
            count = data_summary.get(cls, 0)
            total = data_summary.get("total", 1)
            pct = 100.0 * count / total if total else 0
            lines.append(f"{cls:<22s} {count:>4d}  ({pct:>4.1f}%) \u2192 {binary_map.get(cls, '?')}")
        lines.append("")

    # 5-class results
    lines.append("5-CLASS RESULTS (mean \u00b1 std over {} folds)".format(n_folds))
    lines.append("\u2500" * 45)
    m, s = _mean_std("accuracy_5")
    lines.append(f"Accuracy:     {m:.3f} \u00b1 {s:.3f}")
    m, s = _mean_std("macro_f1_5")
    lines.append(f"Macro F1:     {m:.3f} \u00b1 {s:.3f}")
    m, s = _mean_std("weighted_f1_5")
    lines.append(f"Weighted F1:  {m:.3f} \u00b1 {s:.3f}")
    lines.append("")
    lines.append("Per-class F1:")
    for cls in classes:
        m, s = _mean_std(f"f1_{cls}")
        lines.append(f"  {cls:<22s} {m:.3f} \u00b1 {s:.3f}")
    lines.append("")

    # Binary results (from binary head)
    lines.append("BINARY RESULTS (mean \u00b1 std over {} folds)".format(n_folds))
    lines.append("\u2500" * 45)
    lines.append("[from binary head]")
    m, s = _mean_std("sensitivity")
    lines.append(f"Sensitivity:  {m:.3f} \u00b1 {s:.3f}")
    m, s = _mean_std("specificity")
    lines.append(f"Specificity:  {m:.3f} \u00b1 {s:.3f}")
    m, s = _mean_std("macro_f1_binary")
    lines.append(f"Macro F1:     {m:.3f} \u00b1 {s:.3f}")
    lines.append("")

    # Binary derived from 5-class
    lines.append("[derived from 5-class argmax via binary_map]")
    m, s = _mean_std("sensitivity_derived")
    lines.append(f"Sensitivity:  {m:.3f} \u00b1 {s:.3f}")
    m, s = _mean_std("specificity_derived")
    lines.append(f"Specificity:  {m:.3f} \u00b1 {s:.3f}")
    m, s = _mean_std("macro_f1_binary_derived")
    lines.append(f"Macro F1:     {m:.3f} \u00b1 {s:.3f}")
    lines.append("")

    # Best fold
    best_fold_idx = max(range(n_folds), key=lambda i: all_fold_metrics[i].get("macro_f1_5", 0))
    bm = all_fold_metrics[best_fold_idx]
    lines.append("BEST FOLD")
    lines.append("\u2500" * 10)
    lines.append(
        f"Fold {best_fold_idx + 1} | val_macro_f1={bm.get('macro_f1_5', 0):.3f} "
        f"| sensitivity={bm.get('sensitivity', 0):.3f} "
        f"| epoch={bm.get('best_epoch', '?')}"
    )
    lines.append("")

    # Comparison table
    lines.append("COMPARISON WITH PREVIOUS RESULTS")
    lines.append("\u2500" * 35)
    lines.append(f"{'Method':<30s} \u2502 {'Macro F1':>8s} \u2502 {'Sensitivity':>11s} \u2502 {'Specificity':>11s}")
    lines.append("\u2500" * 30 + "\u253c" + "\u2500" * 10 + "\u253c" + "\u2500" * 13 + "\u253c" + "\u2500" * 12)

    eff_mf1, _ = _mean_std("macro_f1_binary")
    eff_sens, _ = _mean_std("sensitivity")
    eff_spec, _ = _mean_std("specificity")
    lines.append(f"{'EfficientNet-B3 (this run)':<30s} \u2502  {eff_mf1:.3f}   \u2502    {eff_sens:.3f}    \u2502   {eff_spec:.3f}")
    lines.append(f"{'Neural F1-OOF (pose+CLIP)':<30s} \u2502  0.679   \u2502    0.447    \u2502   0.915")
    lines.append(f"{'GPT-4o FGS (corrected)':<30s} \u2502  0.353   \u2502    0.702    \u2502   0.332")
    lines.append(f"{'GPT-4o label-derived':<30s} \u2502  0.532   \u2502    0.155    \u2502   0.905")
    lines.append(sep)

    report_path = Path(run_dir) / "final_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
