"""
Shared training and evaluation loop.
Identical behavior across all models — models are injected via model_base.py.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

import torch.optim.lr_scheduler as lr_sched

from data_engineering import  build_dataloaders
from data_loading import  get_multiclass_class_names

DEFAULT_CLASS_NAMES = [
    "Paining",
    "Positive_Baseline",
    "Agonistic",
    "Vocalizing",
    "HuntingMind",
]


def _default_class_names() -> list[str]:
    return list(DEFAULT_CLASS_NAMES)


# ── Loss ──────────────────────────────────────────────────────


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multiclass focal loss.
    logits: (B, C), targets: (B,) long
    Identical implementation to finetuning_after_ssl focal_loss_soft
    but for hard integer targets — mean reduction.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    fl = ((1.0 - pt) ** gamma) * ce
    if class_weights is not None:
        fl = fl * class_weights[targets]
    return fl.mean()


# ── Metrics ───────────────────────────────────────────────────


def compute_metrics(
    y_true_5: np.ndarray,
    y_pred_5: np.ndarray,
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    y_prob_pain: np.ndarray | None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute and return all metrics as a flat dict.

    5-class metrics:
      accuracy_5, macro_f1_5, weighted_f1_5
      per_class_f1_{classname} for each class
      per_class_precision_{classname}
      per_class_recall_{classname}

    Binary metrics:
      accuracy_binary, macro_f1_binary, weighted_f1_binary
      sensitivity (Pain recall), specificity (No_Pain recall)
      precision_pain, auc_roc (if y_prob_pain provided)
    """
    if class_names is None:
        class_names = _default_class_names()
    out: dict = {}
    y_true_5 = np.asarray(y_true_5).ravel()
    y_pred_5 = np.asarray(y_pred_5).ravel()
    y_true_bin = np.asarray(y_true_bin).ravel()
    y_pred_bin = np.asarray(y_pred_bin).ravel()

    out["accuracy_5"] = float(accuracy_score(y_true_5, y_pred_5))
    out["macro_f1_5"] = float(f1_score(y_true_5, y_pred_5, average="macro", zero_division=0))
    out["weighted_f1_5"] = float(f1_score(y_true_5, y_pred_5, average="weighted", zero_division=0))

    p_s, r_s, f_s, _ = precision_recall_fscore_support(
        y_true_5, y_pred_5, labels=list(range(len(class_names))), zero_division=0
    )
    for i, name in enumerate(class_names):
        out[f"per_class_precision_{name}"] = float(p_s[i]) if i < len(p_s) else 0.0
        out[f"per_class_recall_{name}"] = float(r_s[i]) if i < len(r_s) else 0.0
        out[f"per_class_f1_{name}"] = float(f_s[i]) if i < len(f_s) else 0.0

    out["accuracy_binary"] = float(accuracy_score(y_true_bin, y_pred_bin))
    out["macro_f1_binary"] = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
    out["weighted_f1_binary"] = float(f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    out["sensitivity"] = sens
    out["specificity"] = spec

    prec_p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    out["precision_pain"] = prec_p

    if y_prob_pain is not None and len(np.unique(y_true_bin)) > 1:
        try:
            out["auc_roc"] = float(roc_auc_score(y_true_bin, y_prob_pain))
        except ValueError:
            out["auc_roc"] = float("nan")
    else:
        out["auc_roc"] = float("nan")

    return out


# ── Early stopping ────────────────────────────────────────────


class EarlyStopping:
    """
    Monitors val_macro_f5 (maximize) with patience and min_delta.
    Same idea as EarlyStopping in finetuning_after_ssl.
    """

    def __init__(self, patience: int, min_delta: float, mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: float | None = None
        self.best_epoch = 0
        self.counter = 0

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        """
        Returns (should_stop, is_new_best).
        """
        v = float(value)
        if self.best_value is None:
            self.best_value = v
            self.best_epoch = int(epoch)
            return False, True
        if self.mode == "min":
            improved = v < self.best_value - self.min_delta
        else:
            improved = v > self.best_value + self.min_delta
        if improved:
            self.best_value = v
            self.best_epoch = int(epoch)
            self.counter = 0
            return False, True
        self.counter += 1
        return self.counter >= self.patience, False


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict, epochs: int
) -> lr_sched.LRScheduler | lr_sched.ReduceLROnPlateau | None:
    """
    ``training.scheduler``:

    - ``cosine`` (default) — cosine anneal over ``epochs``
    - ``cosine_with_warmup`` — linear warmup then cosine
    - ``plateau`` / ``reduce_on_plateau`` — :class:`ReduceLROnPlateau` on
      ``val_loss_total`` (``mode=min``; step **after** each validation in the
      main loop, not in ``train_epoch``)
    - ``none`` (``off`` / ``constant``) — fixed LR, no step
    """
    tr = cfg.get("training", {})
    name = str(tr.get("scheduler", "cosine")).strip().lower().replace("-", "_")
    eta_min = float(tr.get("cosine_eta_min", 1e-7))
    wu = int(tr.get("warmup_epochs", 0))

    if name in ("none", "off", "no", "disable", "disabled", "constant", "null", ""):
        return None
    if name in ("plateau", "reduce_on_plateau", "val_plateau"):
        p_pat = int(tr.get("lr_plateau_scheduler_patience", 5))
        fac = float(tr.get("lr_plateau_factor", 0.5))
        min_lr = float(tr.get("lr_plateau_min", eta_min))
        thr = float(tr.get("lr_plateau_threshold", 1e-4))
        return lr_sched.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=fac,
            patience=p_pat,
            min_lr=min_lr,
            threshold=thr,
        )
    if name in ("cosine_with_warmup", "cosine_warmup"):
        wu = max(1, wu) if wu < 1 else wu
        cos_epochs = max(1, int(epochs) - wu)
        warm = lr_sched.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=wu)
        cos = lr_sched.CosineAnnealingLR(optimizer, T_max=cos_epochs, eta_min=eta_min)
        return lr_sched.SequentialLR(optimizer, schedulers=[warm, cos], milestones=[wu])
    return lr_sched.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=eta_min)


def _step_lr_scheduler(
    sched: lr_sched.LRScheduler | lr_sched.ReduceLROnPlateau | None,
    val_d: dict,
) -> None:
    """Call once per epoch (after val). ``ReduceLROnPlateau`` uses val loss; cosine uses no args."""
    if sched is None:
        return
    if isinstance(sched, lr_sched.ReduceLROnPlateau):
        sched.step(float(val_d.get("val_loss_total", 0.0)))
    else:
        sched.step()


def _cfg_binary_only(cfg: dict) -> bool:
    return bool(cfg.get("training", {}).get("binary_only", False))


def _metric_class_names(cfg: dict, class_names: list[str] | None) -> list[str]:
    if _cfg_binary_only(cfg):
        # Pairwise runs pass [neg, pos] (label_int 0/1); use them in CSV so keys match CM/report.
        if class_names is not None and len(class_names) == 2:
            return [str(c) for c in class_names]
        return ["No_Pain", "Pain"]
    return class_names if class_names is not None else _default_class_names()


def _map_early_stop_metric(cfg: dict) -> str:
    """
    Map config ``training.early_stop_metric`` to a key present in ``val_d``.

    Multiclass (``binary_only: false``) extras:
      - ``val_top3_accuracy_5`` — fraction of samples whose true class is among the
        top-3 logits of the multiclass head (for K<3, identical to top-1 accuracy).
    """
    m = str(cfg["training"].get("early_stop_metric", "val_macro_f1")).strip()
    if _cfg_binary_only(cfg):
        if m in ("val_top3_accuracy_5", "val_top3_accuracy", "top3_accuracy"):
            return "val_macro_f1_binary"
        if m in ("val_macro_f1", "macro_f1", "val_macro_f1_5", "val_macro_f1_binary"):
            return "val_macro_f1_binary"
        if m == "val_loss_total":
            return "val_loss_total"
        return "val_macro_f1_binary"
    if m in ("val_top3_accuracy_5", "val_top3_accuracy", "top3_accuracy"):
        return "val_top3_accuracy_5"
    if m in ("val_macro_f1", "macro_f1", "val_macro_f1_5"):
        return "val_macro_f1_5"
    if m == "val_macro_f1_binary":
        return "val_macro_f1_binary"
    if m == "val_loss_total":
        return "val_loss_total"
    return "val_macro_f1_5"


def _epoch_metrics_to_flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}


def _fmt_metric_scalar(v) -> str:
    try:
        if isinstance(v, (float, np.floating)) and (v != v or np.isnan(float(v))):
            return "n/a"
        x = float(v)
        if x != x:
            return "n/a"
        if abs(x) >= 1e4 or (abs(x) < 1e-4 and x != 0.0):
            return f"{x:.4e}"
        return f"{x:.4f}"
    except (TypeError, ValueError):
        return str(v)


def _essential_val_log_keys(bo: bool) -> list[str]:
    """Subset of val_* to print (CSV still has full history)."""
    if bo:
        return [
            "val_loss_total",
            "val_macro_f1_binary",
            "val_accuracy_binary",
            "val_sensitivity",
            "val_specificity",
            "val_precision_pain",
            "val_auc_roc",
        ]
    return [
        "val_loss_total",
        "val_macro_f1_5",
        "val_accuracy_5",
        "val_top3_accuracy_5",
        "val_macro_f1_binary",
        "val_sensitivity",
        "val_specificity",
        "val_auc_roc",
    ]


def _format_compact_epoch(
    tr: dict,
    val_d: dict,
    *,
    epoch: int,
    n_epochs: int,
    bo: bool,
    es_metric: str,
    monitor: float,
    auc_s: str,
) -> str:
    """Two short lines: losses + key validation metrics (no per-class table)."""
    t_loss = _fmt_metric_scalar(tr.get("train_loss_total"))
    v_loss = _fmt_metric_scalar(val_d.get("val_loss_total"))
    lr = _fmt_metric_scalar(tr.get("lr"))
    line1 = (
        f"  epoch {epoch + 1}/{n_epochs}  {es_metric}={_fmt_metric_scalar(monitor)}  "
        f"train_loss={t_loss}  val_loss={v_loss}  lr={lr}  auc={auc_s}"
    )
    parts: list[str] = []
    for k in _essential_val_log_keys(bo):
        if k in val_d and k not in ("val_loss_total", "val_auc_roc"):
            parts.append(f"{k.replace('val_', '')}={_fmt_metric_scalar(val_d[k])}")
    line2 = "    " + "  ".join(parts)
    return line1 + "\n" + line2


def train_epoch(
    model,
    loader,
    optimizer,
    class_weights_5,
    class_weights_bin,
    cfg,
    device,
    logger,
    epoch: int,
    class_names: list[str] | None = None,
) -> dict:
    """
    One training epoch.

    Returns dict with:
      train_loss_5, train_loss_binary, train_loss_total,
      train_macro_f1_5, train_macro_f1_binary,
      train_sensitivity, train_specificity,
      lr (current learning rate)
    """
    names = _metric_class_names(cfg, class_names)
    bo = _cfg_binary_only(cfg)
    model.train()
    gamma = float(cfg["training"]["focal_gamma"])
    use_w = bool(cfg["training"]["use_class_weights"])
    w5 = class_weights_5.to(device) if use_w and not bo else None
    wb = class_weights_bin.to(device) if use_w else None
    clip = cfg["training"].get("grad_clip_norm")

    log_int = max(1, int(cfg["output"].get("log_interval_batches", 20)))
    running_5 = 0.0
    running_b = 0.0
    running_t = 0.0
    n_batches = 0

    y5_t, y5_p = [], []
    yb_t, yb_p = [], []
    top3_correct = 0
    n_top3_total = 0

    for bi, batch in enumerate(loader):
        pose = batch["pose"].to(device)
        mask = batch["mask"].to(device)
        y5 = batch["label_5"].to(device)
        yb = batch["label_binary"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(pose, mask)
        lb = focal_loss(out["logits_binary"], yb, gamma, wb)
        if bo:
            loss = lb
            l5 = torch.zeros((), device=device)
        else:
            l5 = focal_loss(out["logits_5"], y5, gamma, w5)
            loss = l5 + 0.3 * lb
        loss.backward()
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        optimizer.step()

        running_5 += float(l5.item()) if not bo else 0.0
        running_b += float(lb.item())
        running_t += float(loss.item())
        n_batches += 1

        with torch.no_grad():
            yb_hat = torch.argmax(out["logits_binary"], dim=1).cpu().numpy().tolist()
            yb_p.extend(yb_hat)
            yb_t.extend(yb.cpu().numpy().tolist())
            if bo:
                y5_p.extend(yb_hat)
                y5_t.extend(yb.cpu().numpy().tolist())
            else:
                y5_p.extend(torch.argmax(out["logits_5"], dim=1).cpu().numpy().tolist())
                y5_t.extend(y5.cpu().numpy().tolist())
                k_mc = min(3, int(out["logits_5"].shape[1]))
                _, pred_topk = torch.topk(out["logits_5"], k=k_mc, dim=1)
                top3_correct += int((pred_topk == y5.view(-1, 1)).any(dim=1).sum().item())
                n_top3_total += int(y5.numel())

        if log_int > 0 and (bi + 1) % log_int == 0:
            logger.debug(
                "Epoch %d | batch %d/%d | loss=%.4f",
                epoch,
                bi + 1,
                len(loader),
                running_t / n_batches,
            )
    # LR scheduler: stepped in main loop after validation (cosine) or on val loss (plateau)
    lr = optimizer.param_groups[0]["lr"]
    m = compute_metrics(
        np.array(y5_t),
        np.array(y5_p),
        np.array(yb_t),
        np.array(yb_p),
        None,
        names,
    )
    out_tr = {
        "train_loss_5": running_5 / max(n_batches, 1),
        "train_loss_binary": running_b / max(n_batches, 1),
        "train_loss_total": running_t / max(n_batches, 1),
        "train_macro_f1_5": m["macro_f1_5"],
        "train_macro_f1_binary": m["macro_f1_binary"],
        "train_sensitivity": m["sensitivity"],
        "train_specificity": m["specificity"],
        "lr": lr,
    }
    if not bo and n_top3_total > 0:
        out_tr["train_top3_accuracy_5"] = float(top3_correct / n_top3_total)
    return out_tr


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    class_weights_5,
    class_weights_bin,
    cfg,
    device,
    class_names: list[str] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One evaluation epoch.

    Returns:
      metrics dict (val_* keys), including ``val_top3_accuracy_5`` when not
      ``binary_only``: fraction of samples whose true multiclass label appears
      among the top-3 ``logits_5`` indices (for K<3 classes, same as top-1 accuracy).

      y_true_5, y_pred_5, y_true_bin, y_pred_bin, y_prob_pain
    """
    model.eval()
    bo = _cfg_binary_only(cfg)
    names = _metric_class_names(cfg, class_names)
    top3_correct = 0
    n_top3_total = 0
    gamma = float(cfg["training"]["focal_gamma"])
    use_w = bool(cfg["training"]["use_class_weights"])
    w5 = class_weights_5.to(device) if use_w and not bo else None
    wb = class_weights_bin.to(device) if use_w else None

    running_5 = 0.0
    running_b = 0.0
    running_t = 0.0
    n_batches = 0

    y5_t, y5_p = [], []
    yb_t, yb_p = [], []
    prob_pain = []

    for batch in loader:
        pose = batch["pose"].to(device)
        mask = batch["mask"].to(device)
        y5 = batch["label_5"].to(device)
        yb = batch["label_binary"].to(device)

        out = model(pose, mask)
        lb = focal_loss(out["logits_binary"], yb, gamma, wb)
        if bo:
            loss = lb
            l5 = torch.zeros((), device=device)
        else:
            l5 = focal_loss(out["logits_5"], y5, gamma, w5)
            loss = l5 + 0.3 * lb

        running_5 += float(l5.item()) if not bo else 0.0
        running_b += float(lb.item())
        running_t += float(loss.item())
        n_batches += 1

        pr = F.softmax(out["logits_binary"], dim=1)[:, 1].cpu().numpy()
        prob_pain.extend(pr.tolist())
        yb_hat = torch.argmax(out["logits_binary"], dim=1).cpu().numpy().tolist()
        yb_p.extend(yb_hat)
        yb_t.extend(yb.cpu().numpy().tolist())
        if bo:
            y5_p.extend(yb_hat)
            y5_t.extend(yb.cpu().numpy().tolist())
        else:
            y5_p.extend(torch.argmax(out["logits_5"], dim=1).cpu().numpy().tolist())
            y5_t.extend(y5.cpu().numpy().tolist())
            k_mc = min(3, int(out["logits_5"].shape[1]))
            _, pred_topk = torch.topk(out["logits_5"], k=k_mc, dim=1)
            top3_correct += int((pred_topk == y5.view(-1, 1)).any(dim=1).sum().item())
            n_top3_total += int(y5.numel())

    y_true_5 = np.array(y5_t)
    y_pred_5 = np.array(y5_p)
    y_true_bin = np.array(yb_t)
    y_pred_bin = np.array(yb_p)
    y_prob_pain = np.array(prob_pain)

    m = compute_metrics(y_true_5, y_pred_5, y_true_bin, y_pred_bin, y_prob_pain, names)
    val_d = {f"val_{k}": v for k, v in m.items()}
    val_d["val_loss_5"] = running_5 / max(n_batches, 1)
    val_d["val_loss_binary"] = running_b / max(n_batches, 1)
    val_d["val_loss_total"] = running_t / max(n_batches, 1)
    if not bo and n_top3_total > 0:
        val_d["val_top3_accuracy_5"] = float(top3_correct / n_top3_total)

    return val_d, y_true_5, y_pred_5, y_true_bin, y_pred_bin, y_prob_pain


def _append_metrics_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _atomic_torch_save(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_confusion_matrix_txt(
    path: Path,
    cm: np.ndarray,
    row_names: list[str],
    col_names: list[str],
    title: str,
) -> None:
    """ASCII confusion matrix: raw counts and row-normalized proportions."""
    lines: list[str] = [title, ""]
    k = int(cm.shape[0])
    width = max(4, max(len(n) for n in row_names + col_names) + 2)
    head = f"{'T/P':<{width}}" + "".join(f"{c:>{width + 2}}" for c in col_names)
    lines.append(head)
    lines.append("-" * len(head))
    for i, rn in enumerate(row_names):
        row = f"{rn:<{width}}" + "".join(f"{int(cm[i, j]):>{width + 2}}" for j in range(k))
        lines.append(row)
    lines.append("")

    rsu = cm.sum(axis=1, keepdims=True).astype(float)
    rsu[rsu == 0] = 1.0
    cmn = cm.astype(float) / rsu
    lines.append("Row-normalized (proportions within true class):")
    lines.append(head)
    for i, rn in enumerate(row_names):
        row = f"{rn:<{width}}" + "".join(f"{cmn[i, j]:>{width + 2}.4f}" for j in range(k))
        lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_validation_metrics_txt(
    path: Path,
    class_names: list[str],
    best_tr: dict | None,
    best_val: dict | None,
    be: int,
) -> None:
    lines: list[str] = [
        "Metrics at best epoch (early-stopping best by configured metric on validation).",
        f"best_epoch: {be}",
        f"classes: {', '.join(class_names)}",
        "",
        "--- train (at best epoch) ---",
    ]
    if best_tr:
        for k in sorted(best_tr.keys()):
            lines.append(f"  {k}: {best_tr[k]}")
    else:
        lines.append("  (no train row captured)")
    lines += ["", "--- validation (at best epoch) ---"]
    if best_val:
        for k in sorted(best_val.keys()):
            lines.append(f"  {k}: {best_val[k]}")
    else:
        lines.append("  (no validation row)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_training_to_completion(
    model_class,
    model_kwargs: dict,
    train_records: list,
    val_records: list,
    out_dir: Path,
    cfg: dict,
    device: torch.device,
    logger: logging.Logger,
    *,
    class_names: list[str],
    use_kinematics: bool = True,
    progress_epoch_desc: str = "epochs",
    ax_title_tag: str = "Val",
    summary_json_name: str = "fold_summary.json",
    summary_extras: dict | None = None,
    write_txt_artifacts: bool = False,
    build_dataloaders_fn=None,
) -> dict:
    """
    One train/val training loop: same artifacts as train_fold, under ``out_dir`` (flat).
    """
    _REPO = Path(__file__).resolve().parent.parent.parent
    summary_extras = dict(summary_extras) if summary_extras else {}
    t_cfg0 = cfg.get("training", {})
    if t_cfg0.get("init_weights_path"):
        summary_extras["init_weights_path"] = t_cfg0.get("init_weights_path")
    _fe = int(t_cfg0.get("freeze_backbone_epochs") or 0)
    if _fe > 0:
        summary_extras["freeze_backbone_epochs"] = _fe
    if t_cfg0.get("unfreeze_lr") is not None:
        summary_extras["unfreeze_lr"] = t_cfg0.get("unfreeze_lr")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bo = _cfg_binary_only(cfg)
    K = len(class_names)
    csv_path = out_dir / "metrics_history.csv"

    _dl_fn = build_dataloaders_fn if build_dataloaders_fn is not None else build_dataloaders
    train_loader, val_loader, cw5, cwb = _dl_fn(
        train_records, val_records, cfg, use_kinematics=use_kinematics, n_multiclass=K
    )

    t_cfg = cfg["training"]
    model = model_class(**model_kwargs).to(device)
    # Optional: load P4 (or other) state dict (e.g. best_weights from a sweep / pair)
    _iw = t_cfg.get("init_weights_path")
    if _iw is not None and str(_iw).strip() and str(_iw).lower() not in ("none", "null", "false", ""):
        wpath = Path(str(_iw).strip()).expanduser()
        if not wpath.is_absolute():
            wpath = (_REPO / wpath).resolve()
        if not wpath.is_file():
            raise FileNotFoundError(f"training.init_weights_path not found: {wpath}")
        blob = torch.load(wpath, map_location=device)
        sd = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
        if not isinstance(sd, dict):
            raise TypeError(f"Invalid checkpoint at {wpath}")
        model.load_state_dict(sd, strict=True)
        logger.info("Loaded init_weights_path: %s", wpath)

    freeze_e = int(t_cfg.get("freeze_backbone_epochs") or 0)
    epochs = int(t_cfg["epochs"])
    u_lr_f: float | None
    if t_cfg.get("unfreeze_lr", None) is not None and str(t_cfg.get("unfreeze_lr")).strip() != "":
        u_lr_f = float(t_cfg["unfreeze_lr"])
    else:
        u_lr_f = None

    if freeze_e > 0 and freeze_e >= epochs:
        logger.warning("freeze_backbone_epochs (%d) >= epochs (%d); no frozen phase", freeze_e, epochs)
        freeze_e = 0
    can_freeze = bool(getattr(model, "encoder", None) is not None)
    if freeze_e > 0 and not can_freeze:
        logger.warning("Model has no .encoder; freeze_backbone_epochs ignored.")
        freeze_e = 0

    for p in model.parameters():
        p.requires_grad = True
    if freeze_e > 0 and can_freeze:
        for p in model.encoder.parameters():
            p.requires_grad = False
        _params = [x for x in model.parameters() if x.requires_grad]
        opt = torch.optim.AdamW(
            _params,
            lr=float(t_cfg["lr"]),
            weight_decay=float(t_cfg["weight_decay"]),
        )
        c_p1 = copy.deepcopy(cfg)
        c_p1.setdefault("training", {})["lr"] = float(t_cfg["lr"])
        sched = build_lr_scheduler(opt, c_p1, freeze_e)
        u_after = u_lr_f if u_lr_f is not None else float(t_cfg["lr"])
        rem_after = max(0, epochs - freeze_e)
        logger.info(
            "Backbone frozen for epochs 1..%d (train heads only) | then lr=%.6e for %d remaining epoch(s); "
            "scheduler resets at unfreeze",
            freeze_e,
            u_after,
            rem_after,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(t_cfg["lr"]),
            weight_decay=float(t_cfg["weight_decay"]),
        )
        sched = build_lr_scheduler(opt, cfg, epochs)

    es_metric = _map_early_stop_metric(cfg)
    mode = "min" if "loss" in es_metric else "max"
    es = EarlyStopping(
        patience=int(cfg["training"]["early_stop_patience"]),
        min_delta=float(cfg["training"]["early_stop_min_delta"]),
        mode=mode,
    )

    hist_train_loss, hist_val_loss = [], []
    hist_tr_f1, hist_val_f1 = [], []
    hist_tr_sens, hist_val_sens = [], []
    hist_tr_spec, hist_val_spec = [], []

    best_snapshot = None
    best_val_for_fold: dict | None = None
    best_tr_for_fold: dict | None = None
    val_d: dict = {}

    if bo:
        f1_ylabel = "Macro F1 (binary pain)"
    elif K == 5:
        f1_ylabel = "Macro F1 (5-class)"
    else:
        f1_ylabel = f"Macro F1 ({K}-class)"

    es_pat = int(cfg["training"]["early_stop_patience"])
    logger.info(
        "Starting training: %d epochs | model=%s | early_stop_metric=%s | patience=%d | %s",
        epochs,
        getattr(model_class, "__name__", str(model_class)),
        es_metric,
        es.patience,
        progress_epoch_desc,
    )
    last_epoch_0 = -1  # 0-based index of last *completed* epoch; -1 if no training steps
    for epoch in range(epochs):
        if freeze_e > 0 and can_freeze and epoch == freeze_e:
            for p in model.parameters():
                p.requires_grad = True
            u_run = u_lr_f if u_lr_f is not None else float(t_cfg["lr"])
            rem = epochs - epoch
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=u_run,
                weight_decay=float(t_cfg["weight_decay"]),
            )
            c_p2 = copy.deepcopy(cfg)
            c_p2.setdefault("training", {})["lr"] = u_run
            sched = build_lr_scheduler(opt, c_p2, rem)
            logger.info(
                "Unfroze backbone at start of epoch %d / %d | AdamW lr=%.6e | new scheduler T_max=%d (match remaining epochs)",
                epoch + 1,
                epochs,
                u_run,
                rem,
            )
        if epoch > 0:
            logger.info("%s", "—" * 72)
        tr = train_epoch(
            model,
            train_loader,
            opt,
            cw5,
            cwb,
            cfg,
            device,
            logger,
            epoch,
            class_names=class_names,
        )
        val_d, _, _, _, _, _ = eval_epoch(
            model, val_loader, cw5, cwb, cfg, device, class_names=class_names
        )
        _step_lr_scheduler(sched, val_d)

        auc_raw = val_d.get("val_auc_roc")
        try:
            auc_f = float(auc_raw)
            auc_s = f"{auc_f:.4f}" if auc_f == auc_f else "n/a"
        except (TypeError, ValueError):
            auc_s = "n/a"

        monitor = float(val_d.get(es_metric, val_d["val_macro_f1_binary" if bo else "val_macro_f1_5"]))
        should_stop, is_best = es.step(monitor, epoch)
        last_epoch_0 = int(epoch)

        block = _format_compact_epoch(
            tr,
            val_d,
            epoch=epoch,
            n_epochs=epochs,
            bo=bo,
            es_metric=es_metric,
            monitor=monitor,
            auc_s=auc_s,
        )
        for line in block.split("\n"):
            logger.info(line)
        if is_best:
            logger.info(
                "  *** NEW BEST WEIGHTS ***  %s = %s  (best epoch will be set to this epoch; "
                "weights saved to best_model.pth at end of training)",
                es_metric,
                _fmt_metric_scalar(monitor),
            )
        else:
            bmv = _fmt_metric_scalar(es.best_value) if es.best_value is not None else "—"
            logger.info(
                "  No improvement on %s  (this epoch: %s | best: %s @ epoch %d | no-improve: %d/%d)",
                es_metric,
                _fmt_metric_scalar(monitor),
                bmv,
                es.best_epoch + 1,
                es.counter,
                es.patience,
            )

        row = {"epoch": epoch, **tr, **val_d}
        _append_metrics_row(csv_path, row)

        hist_train_loss.append(tr["train_loss_total"])
        hist_val_loss.append(val_d["val_loss_total"])
        hist_tr_f1.append(tr["train_macro_f1_5"])
        hist_val_f1.append(val_d["val_macro_f1_5"])
        hist_tr_sens.append(tr["train_sensitivity"])
        hist_val_sens.append(val_d["val_sensitivity"])
        hist_tr_spec.append(tr["train_specificity"])
        hist_val_spec.append(val_d["val_specificity"])
        if is_best:
            best_snapshot = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict() if sched is not None else {},
                "epoch": epoch,
                "config": cfg,
            }
            best_val_for_fold = dict(val_d)
            best_tr_for_fold = dict(tr)
        if should_stop:
            logger.info("Early stopping triggered after epoch %d/%d", epoch + 1, epochs)
            break
    n_epochs_ran = int(last_epoch_0) + 1  # 1-based count of completed epoch loops

    if best_snapshot is None:
        best_snapshot = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict() if sched is not None else {},
            "epoch": epochs - 1,
            "config": cfg,
        }
        best_val_for_fold = dict(val_d)
        best_tr_for_fold = best_tr_for_fold or {}

    logger.info("%s", "─" * 72)
    logger.info(
        "End of training | best epoch: %d  |  %s: %s  |  writing: %s  +  best_weights.pth",
        es.best_epoch + 1,
        es_metric,
        _fmt_metric_scalar(es.best_value) if es.best_value is not None else "n/a",
        (out_dir / "best_model.pth").resolve(),
    )
    if best_val_for_fold:
        keys = _essential_val_log_keys(bo)
        bvk_lines = [
            f"  {k}: {_fmt_metric_scalar(best_val_for_fold[k])}"
            for k in keys
            if k in best_val_for_fold
        ]
        if bvk_lines:
            logger.info("Metrics at best checkpoint (same as CSV):")
            for line in bvk_lines:
                logger.info(line)

    _atomic_torch_save(best_snapshot, out_dir / "best_model.pth")
    torch.save(best_snapshot["model_state_dict"], out_dir / "best_weights.pth")

    be = int(es.best_epoch)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].plot(hist_train_loss, label="train")
    axes[0, 0].plot(hist_val_loss, label="val")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].axvline(be, color="gray", linestyle="--")
    axes[0, 0].legend()

    axes[0, 1].plot(hist_tr_f1, label="train")
    axes[0, 1].plot(hist_val_f1, label="val")
    axes[0, 1].set_title(f1_ylabel)
    axes[0, 1].axvline(be, color="gray", linestyle="--")
    axes[0, 1].legend()

    axes[1, 0].plot(hist_tr_sens, label="train")
    axes[1, 0].plot(hist_val_sens, label="val")
    axes[1, 0].set_title("Sensitivity (binary)")
    axes[1, 0].axvline(be, color="gray", linestyle="--")
    axes[1, 0].legend()

    axes[1, 1].plot(hist_tr_spec, label="train")
    axes[1, 1].plot(hist_val_spec, label="val")
    axes[1, 1].set_title("Specificity (binary)")
    axes[1, 1].axvline(be, color="gray", linestyle="--")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    model.load_state_dict(best_snapshot["model_state_dict"])
    _, yt5, yp5, ytb, ypb, ypp = eval_epoch(
        model, val_loader, cw5, cwb, cfg, device, class_names=class_names
    )
    names = class_names

    cm5_best: np.ndarray | None = None
    if not bo:
        cm5_best = confusion_matrix(yt5, yp5, labels=list(range(K)))
        row_sum = cm5_best.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm5n = cm5_best.astype(float) / row_sum
        fig, ax = plt.subplots(figsize=(10, 8))
        ann = np.empty_like(cm5_best, dtype=object)
        for i in range(cm5_best.shape[0]):
            for j in range(cm5_best.shape[1]):
                ann[i, j] = f"{cm5n[i, j]:.2f}\n(n={cm5_best[i, j]})"
        sns.heatmap(
            cm5n,
            annot=ann,
            fmt="",
            cmap="Blues",
            xticklabels=names,
            yticklabels=names,
            ax=ax,
        )
        k_tag = f"{K}-class" if K != 5 else "5-class"
        ax.set_title(f"{ax_title_tag} — best epoch {be} ({k_tag}, row-normalized)")
        fig.tight_layout()
        cm_mc_png = "confusion_matrix_5.png" if K == 5 else f"confusion_matrix_{K}class.png"
        fig.savefig(out_dir / cm_mc_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    cmb = confusion_matrix(ytb, ypb, labels=[0, 1])
    rs = cmb.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    cmbn = cmb.astype(float) / rs
    fig, ax = plt.subplots(figsize=(6, 5))
    annb = np.empty_like(cmb, dtype=object)
    for i in range(cmb.shape[0]):
        for j in range(cmb.shape[1]):
            annb[i, j] = f"{cmbn[i, j]:.2f}\n(n={cmb[i, j]})"
    bin_labels = names if len(names) == 2 else ["No_Pain", "Pain"]
    sns.heatmap(
        cmbn,
        annot=annb,
        fmt="",
        cmap="Blues",
        xticklabels=bin_labels,
        yticklabels=bin_labels,
        ax=ax,
    )
    ax.set_title(f"{ax_title_tag} — binary")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_binary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if write_txt_artifacts:
        if not bo:
            cm_mc_txt = "confusion_matrix_5.txt" if K == 5 else f"confusion_matrix_{K}class.txt"
            _write_confusion_matrix_txt(
                out_dir / cm_mc_txt,
                confusion_matrix(yt5, yp5, labels=list(range(K))),
                names,
                names,
                f"Multiclass confusion matrix (counts) — {ax_title_tag} best epoch {be}",
            )
        _bin_lbl = names if len(names) == 2 else ["No_Pain", "Pain"]
        _write_confusion_matrix_txt(
            out_dir / "confusion_matrix_binary.txt",
            cmb,
            _bin_lbl,
            _bin_lbl,
            f"Binary confusion matrix (counts) — {ax_title_tag} best epoch {be}",
        )
        _write_validation_metrics_txt(
            out_dir / "validation_metrics.txt",
            names,
            best_tr_for_fold,
            best_val_for_fold,
            be,
        )

    if not bo:
        report = classification_report(
            yt5,
            yp5,
            labels=list(range(K)),
            target_names=names,
            zero_division=0,
        )
    else:
        report = classification_report(
            ytb,
            ypb,
            labels=[0, 1],
            target_names=(names if len(names) == 2 else ["No_Pain", "Pain"]),
            zero_division=0,
        )
    (out_dir / "per_class_report.txt").write_text(report, encoding="utf-8")

    b0 = int(be)  # 0-based (same as metrics CSV "epoch" column; run_summary best_epoch)
    b1 = b0 + 1  # 1-based epoch index of the best val checkpoint
    n_er = int(n_epochs_ran)
    p_int = int(es_pat)
    # Convergence efficiency: 1-based best position relative to *actual* run length
    ce = float(b1) / max(n_er, 1)
    # User heuristic (0-based in numerator & denominator): best_epoch / (best_epoch + patience)
    ce_h = float(b0) / max(float(b0) + float(p_int), 1e-9)

    summary = {
        "best_epoch": be,
        "n_epochs_ran": n_er,
        "early_stop_patience": p_int,
        "ce_ratio": round(ce, 6),
        "ce_ratio_heuristic": round(ce_h, 6),
        "model_id": model.model_id,
        "model_name": model.model_name,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "class_names": names,
        "n_classes": K,
        "metrics": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in (best_val_for_fold or {}).items()
        },
        **summary_extras,
    }
    (out_dir / summary_json_name).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Convergence: ce_ratio=%.4f (1-based best epoch / n_epochs_ran)  "
        "ce_ratio_heuristic=%.4f (0-based best / (best+patience))  |  n_epochs_ran=%d  best_epoch(0)=%d",
        ce,
        ce_h,
        n_er,
        b0,
    )

    out = dict(best_val_for_fold or {})
    out.update(
        {
            "best_epoch": be,
            "n_epochs_ran": n_er,
            "early_stop_patience": p_int,
            "ce_ratio": round(float(ce), 6),
            "ce_ratio_heuristic": round(float(ce_h), 6),
            "model_id": model.model_id,
            "model_name": model.model_name,
            "n_train": len(train_records),
            "n_val": len(val_records),
            "class_names": names,
            "n_classes": K,
            "cm5": cm5_best,
            **summary_extras,
        }
    )
    return out


def train_single_split(
    model_class,
    model_kwargs: dict,
    train_records: list,
    val_records: list,
    run_dir: Path,
    cfg: dict,
    device: torch.device,
    logger: logging.Logger,
    *,
    class_names: list[str] | None = None,
    use_kinematics: bool = True,
    split_tag: str = "stratified_group_first_of_5",
    build_dataloaders_fn=None,
) -> dict:
    """
    Single train/val split, artifacts in ``run_dir`` (no fold_*/ nesting).

    Extra vs CV fold: run_summary.json, confusion_matrix_*.txt, validation_metrics.txt
    (multiclass CMs use filename confusion_matrix_5.png for parity with run_scaling_experiment).
    """
    names = class_names or _default_class_names()
    return _run_training_to_completion(
        model_class,
        model_kwargs,
        train_records,
        val_records,
        run_dir,
        cfg,
        device,
        logger,
        class_names=names,
        use_kinematics=use_kinematics,
        progress_epoch_desc="epochs",
        ax_title_tag="Val",
        summary_json_name="run_summary.json",
        summary_extras={"split": split_tag},
        write_txt_artifacts=True,
        build_dataloaders_fn=build_dataloaders_fn,
    )


def train_fold(
    model_class,
    model_kwargs: dict,
    train_records: list,
    val_records: list,
    fold: int,
    run_dir: Path,
    cfg: dict,
    device: torch.device,
    logger: logging.Logger,
    *,
    use_kinematics: bool = True,
    class_names: list[str] | None = None,
) -> dict:
    """
    Full training loop for one fold.

    Saves to run_dir/fold_{fold}/:
      best_model.pth, metrics_history.csv, loss_curve.png,
      confusion_matrix_5.png, confusion_matrix_binary.png,
      per_class_report.txt, fold_summary.json
    """
    names = class_names or _default_class_names()
    fold_dir = run_dir / f"fold_{fold}"
    return _run_training_to_completion(
        model_class,
        model_kwargs,
        train_records,
        val_records,
        fold_dir,
        cfg,
        device,
        logger,
        class_names=names,
        use_kinematics=use_kinematics,
        progress_epoch_desc=f"fold {fold} epochs",
        ax_title_tag=f"Fold {fold}",
        summary_json_name="fold_summary.json",
        summary_extras={"fold": fold},
        write_txt_artifacts=False,
    )


def stratified_group_subsample(train_df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Sample ~fraction of snippets using whole cat_id groups, stratified by label_int."""
    rng = np.random.default_rng(seed)
    if fraction >= 0.999999:
        return train_df
    gcol, lcol = "cat_id", "label_int"
    collected = []
    for c in np.sort(train_df[lcol].unique()):
        sub = train_df[train_df[lcol] == c]
        if len(sub) == 0:
            continue
        cats = sub[gcol].unique()
        rng.shuffle(cats)
        n_take = max(1, int(round(fraction * len(cats))))
        chosen = set(cats[:n_take])
        collected.append(sub[sub[gcol].isin(chosen)])
    if not collected:
        return train_df.iloc[:0].copy()
    out = pd.concat(collected).drop_duplicates()
    return out.sort_index()


def run_cv_sweep(
    model_class,
    model_kwargs: dict,
    df: pd.DataFrame,
    fraction: float,
    run_dir: Path,
    cfg: dict,
    device: torch.device,
    logger: logging.Logger,
    *,
    use_kinematics: bool = True,
    repeat_idx: int = 0,
) -> dict:
    """
    Run n_folds cross-validation for one (model, fraction) combination.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gcol = cfg["split"]["group_field"]
    n_folds_cfg = int(cfg["split"]["n_folds"])
    rs = int(cfg["split"]["random_state"]) + repeat_idx * 7919

    df = df.copy()
    if "label_int" not in df.columns:
        raise ValueError("df must have label_int")

    n_samples = len(df)
    n_groups = int(df[gcol].nunique())
    # StratifiedGroupKFold requires n_splits <= n_samples; also cannot exceed n_groups.
    n_folds = min(n_folds_cfg, n_samples, n_groups)
    if n_folds < 2:
        raise ValueError(
            f"Cannot run CV: need at least 2 folds, but n_samples={n_samples}, "
            f"n_unique_{gcol}={n_groups}, split.n_folds={n_folds_cfg} → effective n_folds={n_folds}. "
            "Ensure pose files exist for enough snippets/cats, or lower split.n_folds in config."
        )
    if n_folds < n_folds_cfg:
        logger.warning(
            "Reducing n_folds from %d to %d (n_samples=%d, n_unique_%s=%d)",
            n_folds_cfg,
            n_folds,
            n_samples,
            gcol,
            n_groups,
        )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    X = np.zeros(len(df))
    groups = df[gcol].values
    y = df["label_int"].values

    fold_rows = []
    cms5 = []
    class_names = get_multiclass_class_names(cfg)
    k_cls = len(class_names)

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        train_df = df.iloc[tr_idx]
        val_df = df.iloc[va_idx]
        sub_seed = rs + fold * 97 + int(fraction * 10000)
        train_sub = stratified_group_subsample(train_df, fraction, sub_seed)
        tr_recs = train_sub.to_dict("records")
        va_recs = val_df.to_dict("records")
        logger.info(
            "=== CV fold %d / %d  |  train %d (sub %d%%)  val %d  ===",
            fold + 1,
            n_folds,
            len(tr_recs),
            int(fraction * 100),
            len(va_recs),
        )

        fd = train_fold(
            model_class,
            model_kwargs,
            tr_recs,
            va_recs,
            fold,
            run_dir,
            cfg,
            device,
            logger,
            use_kinematics=use_kinematics,
            class_names=class_names,
        )
        cms5.append(fd.pop("cm5"))
        fold_rows.append(fd)

    cm_sum = np.sum(cms5, axis=0)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    rsu = cm_sum.sum(axis=1, keepdims=True)
    rsu[rsu == 0] = 1
    cmn = cm_sum.astype(float) / rsu
    names = class_names
    ann = np.empty_like(cm_sum, dtype=object)
    for i in range(cm_sum.shape[0]):
        for j in range(cm_sum.shape[1]):
            ann[i, j] = f"{cmn[i, j]:.2f}\n(n={cm_sum[i, j]})"
    sns.heatmap(cmn, annot=ann, fmt="", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(f"CV summed confusion ({k_cls}-class, row-normalized)")
    fig.tight_layout()
    out_cm = run_dir / (f"summary_confusion_{k_cls}class.png" if k_cls != 5 else "summary_confusion_5.png")
    fig.savefig(out_cm, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # aggregate mean/std
    keys = [
        "val_macro_f1_5",
        "val_macro_f1_binary",
        "val_sensitivity",
        "val_specificity",
        "val_accuracy_5",
        "val_accuracy_binary",
        "val_auc_roc",
    ]
    mprobe = model_class(**model_kwargs)
    agg: dict = {
        "model_id": mprobe.model_id,
        "model_name": mprobe.model_name,
        "fraction": fraction,
        "n_folds": n_folds,
    }
    for k in keys:
        vals = [float(fr.get(k, np.nan)) for fr in fold_rows]
        agg[f"mean_{k.replace('val_', '')}"] = float(np.nanmean(vals))
        agg[f"std_{k.replace('val_', '')}"] = float(np.nanstd(vals))

    for name in names:
        for short, pref in [
            ("f1", "val_per_class_f1_"),
            ("recall", "val_per_class_recall_"),
            ("precision", "val_per_class_precision_"),
        ]:
            kk = f"{pref}{name}"
            vals = [float(fr.get(kk, np.nan)) for fr in fold_rows]
            agg[f"mean_per_class_{short}_{name}"] = float(np.nanmean(vals))
            agg[f"std_per_class_{short}_{name}"] = float(np.nanstd(vals))

    pd.DataFrame(fold_rows).to_csv(run_dir / "cv_results.csv", index=False)
    (run_dir / "cv_summary.json").write_text(
        json.dumps(agg, indent=2, default=str), encoding="utf-8"
    )

    # summary curves across folds — mean ± std of metrics from fold summaries
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # We don't have per-epoch cross-fold history in one array; plot fold-index bars as proxy
    xs = np.arange(n_folds)
    f1s = [float(fr.get("val_macro_f1_5", 0)) for fr in fold_rows]
    axes[0, 0].bar(xs, f1s)
    axes[0, 0].set_title("Val macro F1 per fold")
    axes[0, 1].bar(xs, [float(fr.get("val_sensitivity", 0)) for fr in fold_rows])
    axes[0, 1].set_title("Sensitivity per fold")
    axes[1, 0].bar(xs, [float(fr.get("val_specificity", 0)) for fr in fold_rows])
    axes[1, 0].set_title("Specificity per fold")
    axes[1, 1].bar(xs, [float(fr.get("val_loss_total", 0)) for fr in fold_rows])
    axes[1, 1].set_title("Val total loss per fold")
    fig.tight_layout()
    fig.savefig(run_dir / "summary_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return agg
