"""Main training script for EfficientNet-B3 cat pain classification.

Usage:
    python cnn_finetune/src/train.py --config cnn_finetune/config/default.yaml
    python cnn_finetune/src/train.py --config cnn_finetune/config/default.yaml --batch_size 32
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from src.dataset import (
    CatBehaviorDataset,
    build_dataset,
    build_transforms,
    load_manifest,
)
from src.evaluate import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_summary_curves,
    plot_training_curves,
    write_final_report,
    write_per_class_report,
)
from src.model import CatPainCNN
from src.utils import (
    EarlyStopping,
    load_config,
    make_run_dir,
    save_config,
    save_metrics_csv,
    setup_logger,
)


# ── Focal Loss ──────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ── Device detection ────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Seed ────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Collate (handles mixed types in batch) ──────────────────────────────


def collate_fn(batch: list[dict]) -> dict:
    frames = torch.stack([b["frames"] for b in batch])
    label_5 = torch.stack([b["label_5"] for b in batch])
    label_binary = torch.stack([b["label_binary"] for b in batch])
    stems = [b["stem"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    frame_indices = [b["frame_indices"] for b in batch]
    frame_timestamps = [b["frame_timestamps"] for b in batch]
    bbox_used = [b["bbox_used"] for b in batch]
    return {
        "frames": frames,
        "label_5": label_5,
        "label_binary": label_binary,
        "stem": stems,
        "video_id": video_ids,
        "frame_indices": frame_indices,
        "frame_timestamps": frame_timestamps,
        "bbox_used": bbox_used,
    }


# ── Load YOLO model ────────────────────────────────────────────────────


def load_yolo(cfg: dict, device: torch.device):
    """Load the YOLO model for bbox cropping."""
    if not cfg.get("use_bbox_crop", True):
        return None
    try:
        from ultralytics import YOLO
        project_root = Path(__file__).resolve().parents[3]
        weights_path = project_root / cfg.get("yolo_weights", "models/yolo/yolov8x.pt")
        yolo_device = "cpu"
        if device.type == "cuda":
            yolo_device = "cuda"
        elif device.type == "mps":
            yolo_device = "mps"
        model = YOLO(str(weights_path))
        return model, yolo_device
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")
        return None, "cpu"


# ── Training one fold ───────────────────────────────────────────────────


def train_fold(
    fold: int,
    train_records: list[dict],
    val_records: list[dict],
    cfg: dict,
    device: torch.device,
    run_dir: str,
    logger,
    yolo_model=None,
    yolo_device: str = "cpu",
) -> tuple[dict, dict]:
    """Train one fold. Returns (best_metrics, fold_history)."""

    classes_5 = cfg["classes_5"]
    binary_map = cfg["binary_map"]
    binary_classes = sorted(set(binary_map.values()))
    fold_dir = Path(run_dir) / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True)

    # Datasets — use PrecomputedCatDataset when precomputed_frames_dir is set
    train_ds = build_dataset(train_records, cfg, is_train=True)
    val_ds = build_dataset(val_records, cfg, is_train=False)

    # Attach YOLO to legacy CatBehaviorDataset if needed
    if isinstance(train_ds, CatBehaviorDataset) and yolo_model is not None:
        train_ds.yolo_model = yolo_model
        train_ds.set_yolo_device(yolo_device)
    if isinstance(val_ds, CatBehaviorDataset) and yolo_model is not None:
        val_ds.yolo_model = yolo_model
        val_ds.set_yolo_device(yolo_device)

    # num_workers: use 4+ for pre-computed (pure I/O), 0 for on-the-fly (YOLO/MPS unsafe)
    from src.dataset import PrecomputedCatDataset
    precomputed = isinstance(train_ds, PrecomputedCatDataset)
    num_workers = cfg.get("num_workers", 4 if precomputed else 0)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # Model
    model = CatPainCNN(cfg).to(device)
    model.freeze_backbone()

    # Class weights
    train_labels = np.array([r["label_5_idx"] for r in train_records])
    if cfg.get("use_class_weights", True):
        cw = compute_class_weight("balanced", classes=np.arange(len(classes_5)), y=train_labels)
        class_weights_5 = torch.tensor(cw, dtype=torch.float32).to(device)
        logger.info(f"Class weights: {[f'{w:.2f}' for w in cw]}")
    else:
        class_weights_5 = None

    train_binary = np.array([r["label_binary_idx"] for r in train_records])
    if cfg.get("use_class_weights", True):
        bw = compute_class_weight("balanced", classes=np.arange(len(binary_classes)), y=train_binary)
        class_weights_bin = torch.tensor(bw, dtype=torch.float32).to(device)
    else:
        class_weights_bin = None

    # Loss
    gamma = cfg.get("focal_gamma", 2.0)
    criterion_5 = FocalLoss(gamma=gamma, weight=class_weights_5)
    criterion_bin = FocalLoss(gamma=gamma, weight=class_weights_bin)

    # Optimizer (head only initially)
    head_params = list(model.head_5class.parameters()) + list(model.head_binary.parameters())
    if hasattr(model.frame_pool, "parameters"):
        head_params += list(model.frame_pool.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": cfg["lr_head"]},
    ], weight_decay=cfg.get("weight_decay", 0.01))

    # Scheduler
    sched_type = cfg.get("scheduler", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-7,
        )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    else:
        scheduler = None

    # Early stopping
    early_stop = EarlyStopping(
        patience=cfg.get("early_stop_patience", 12),
        min_delta=cfg.get("early_stop_min_delta", 0.001),
        mode="max",
    )

    freeze_epochs = cfg.get("freeze_backbone_epochs", 5)
    unfreeze_blocks = cfg.get("unfreeze_blocks", 2)
    epochs = cfg["epochs"]
    log_interval = cfg.get("log_interval_batches", 10)

    history = {
        "train_loss": [], "val_loss": [],
        "train_macro_f1": [], "val_macro_f1": [],
        "train_sensitivity": [], "val_sensitivity": [],
        "train_specificity": [], "val_specificity": [],
    }
    csv_rows: list[dict] = []
    best_state = None
    unfreeze_epoch = freeze_epochs + 1

    for epoch in range(1, epochs + 1):
        phase = "freeze" if epoch <= freeze_epochs else "finetune"

        # Unfreeze at the transition epoch
        if epoch == freeze_epochs + 1:
            model.unfreeze_last_n_blocks(unfreeze_blocks, cfg["lr_backbone"])
            # Rebuild optimizer with backbone params
            backbone_params = [
                p for p in model.backbone.parameters() if p.requires_grad
            ]
            head_params = list(model.head_5class.parameters()) + list(model.head_binary.parameters())
            if hasattr(model.frame_pool, "parameters"):
                head_params += list(model.frame_pool.parameters())

            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": cfg["lr_backbone"]},
                {"params": head_params, "lr": cfg["lr_head"]},
            ], weight_decay=cfg.get("weight_decay", 0.01))

            if sched_type == "cosine":
                remaining = epochs - epoch + 1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=1e-7,
                )
            elif sched_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

            logger.info(f"\U0001f513 Unfreezing last {unfreeze_blocks} blocks at epoch {epoch}")
            params = model.count_trainable_params()
            logger.info(f"Trainable params: {params}")

        # ── Train ───────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        all_pred_5, all_true_5 = [], []
        all_pred_bin, all_true_bin = [], []
        n_batches = len(train_loader)

        pbar = tqdm(
            train_loader, desc=f"Fold {fold} Epoch {epoch}/{epochs}",
            leave=False,
        )
        for batch_i, batch in enumerate(pbar, 1):
            frames = batch["frames"].to(device)
            lbl_5 = batch["label_5"].to(device)
            lbl_bin = batch["label_binary"].to(device)

            out = model(frames)
            loss_5 = criterion_5(out["logits_5"], lbl_5)
            loss_bin = criterion_bin(out["logits_binary"], lbl_bin)
            loss = loss_5 + 0.3 * loss_bin

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred_5 = out["logits_5"].argmax(dim=1).cpu().numpy()
            pred_bin = out["logits_binary"].argmax(dim=1).cpu().numpy()
            all_pred_5.extend(pred_5)
            all_true_5.extend(lbl_5.cpu().numpy())
            all_pred_bin.extend(pred_bin)
            all_true_bin.extend(lbl_bin.cpu().numpy())

            if batch_i % log_interval == 0:
                logger.debug(
                    f"Epoch {epoch:>3d}/{epochs} | phase={phase} | "
                    f"batch {batch_i}/{n_batches} | loss={loss.item():.3f}"
                )

            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / n_batches
        all_true_5_np = np.array(all_true_5)
        all_pred_5_np = np.array(all_pred_5)
        all_true_bin_np = np.array(all_true_bin)
        all_pred_bin_np = np.array(all_pred_bin)

        train_metrics = compute_all_metrics(
            all_true_5_np, all_pred_5_np,
            all_true_bin_np, all_pred_bin_np,
            classes_5,
        )

        # ── Validate ────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_pred_5, val_true_5 = [], []
        val_pred_bin, val_true_bin = [], []
        val_pred_5_for_derived, val_true_5_for_derived = [], []

        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                lbl_5 = batch["label_5"].to(device)
                lbl_bin = batch["label_binary"].to(device)

                out = model(frames)
                loss_5 = criterion_5(out["logits_5"], lbl_5)
                loss_bin = criterion_bin(out["logits_binary"], lbl_bin)
                loss = loss_5 + 0.3 * loss_bin
                val_loss_sum += loss.item()

                p5 = out["logits_5"].argmax(dim=1).cpu().numpy()
                pb = out["logits_binary"].argmax(dim=1).cpu().numpy()
                val_pred_5.extend(p5)
                val_true_5.extend(lbl_5.cpu().numpy())
                val_pred_bin.extend(pb)
                val_true_bin.extend(lbl_bin.cpu().numpy())
                val_pred_5_for_derived.extend(p5)
                val_true_5_for_derived.extend(lbl_5.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader)
        val_true_5_np = np.array(val_true_5)
        val_pred_5_np = np.array(val_pred_5)
        val_true_bin_np = np.array(val_true_bin)
        val_pred_bin_np = np.array(val_pred_bin)

        val_metrics = compute_all_metrics(
            val_true_5_np, val_pred_5_np,
            val_true_bin_np, val_pred_bin_np,
            classes_5,
        )

        # Derived binary from 5-class argmax
        class_to_binary_idx = {}
        for i, cls in enumerate(classes_5):
            bin_label = binary_map[cls]
            class_to_binary_idx[i] = 0 if bin_label == "No_Pain" else 1

        derived_pred_bin = np.array([class_to_binary_idx[p] for p in val_pred_5_np])
        derived_true_bin = np.array([class_to_binary_idx[t] for t in val_true_5_np])

        from sklearn.metrics import f1_score, recall_score
        val_metrics["sensitivity_derived"] = recall_score(
            derived_true_bin, derived_pred_bin, pos_label=1, zero_division=0,
        )
        tn_mask = derived_true_bin == 0
        if tn_mask.sum() > 0:
            val_metrics["specificity_derived"] = float((derived_pred_bin[tn_mask] == 0).sum()) / float(tn_mask.sum())
        else:
            val_metrics["specificity_derived"] = 0.0
        val_metrics["macro_f1_binary_derived"] = f1_score(
            derived_true_bin, derived_pred_bin, average="macro", zero_division=0,
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_metrics["macro_f1_5"])
        history["val_macro_f1"].append(val_metrics["macro_f1_5"])
        history["train_sensitivity"].append(train_metrics["sensitivity"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["train_specificity"].append(train_metrics["specificity"])
        history["val_specificity"].append(val_metrics["specificity"])

        # Get current LRs
        lr_bb = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0
        lr_hd = optimizer.param_groups[-1]["lr"]

        csv_rows.append({
            "fold": fold, "epoch": epoch, "phase": phase,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_macro_f1": train_metrics["macro_f1_5"],
            "val_macro_f1": val_metrics["macro_f1_5"],
            "train_sensitivity": train_metrics["sensitivity"],
            "val_sensitivity": val_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "val_specificity": val_metrics["specificity"],
            "lr_backbone": lr_bb, "lr_head": lr_hd,
            **{f"val_f1_{c}": val_metrics.get(f"f1_{c}", 0) for c in classes_5},
        })

        # Log
        logger.info(
            f"Epoch {epoch:>3d}/{epochs} | train_loss={train_loss:.3f} | val_loss={val_loss:.3f}"
        )
        logger.info(
            f"             | train_macro_f1={train_metrics['macro_f1_5']:.3f} | "
            f"val_macro_f1={val_metrics['macro_f1_5']:.3f}"
        )
        logger.info(
            f"             | train_binary_sens={train_metrics['sensitivity']:.3f} | "
            f"val_binary_sens={val_metrics['sensitivity']:.3f}"
        )
        logger.info(
            f"             | train_binary_spec={train_metrics['specificity']:.3f} | "
            f"val_binary_spec={val_metrics['specificity']:.3f}"
        )
        logger.info(
            f"             | lr_backbone={lr_bb:.3e} | lr_head={lr_hd:.3e}"
        )

        # Early stopping check
        val_f1 = val_metrics["macro_f1_5"]
        if early_stop._is_improvement(val_f1):
            logger.info(
                f"\u2713 New best val_macro_f1={val_f1:.3f} \u2192 checkpoint saved"
            )
            best_state = {
                "epoch": epoch,
                "fold": fold,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_macro_f1": val_f1,
                "val_binary_sensitivity": val_metrics["sensitivity"],
                "config": cfg,
                "class_names": classes_5,
                "binary_map": binary_map,
            }
            torch.save(best_state, fold_dir / "best_model.pth")

        if early_stop.step(val_f1, epoch):
            logger.info(
                f"\u26a0 Early stopping at epoch {epoch} "
                f"(patience={early_stop.patience}, best={early_stop.best_value:.3f} "
                f"@ epoch {early_stop.best_epoch})"
            )
            break

        if scheduler:
            scheduler.step()

    # End of fold
    best_val_f1 = early_stop.best_value or 0.0
    best_epoch = early_stop.best_epoch

    logger.info(
        f"Fold {fold} complete | best val_macro_f1={best_val_f1:.3f} | best_epoch={best_epoch}"
    )
    per_class_f1 = " ".join(
        f"{c[:4]}={val_metrics.get(f'f1_{c}', 0):.2f}" for c in classes_5
    )
    logger.info(f"  Per-class F1: {per_class_f1}")
    logger.info(
        f"  Binary: sens={val_metrics['sensitivity']:.3f} "
        f"spec={val_metrics['specificity']:.3f} "
        f"macro_f1={val_metrics['macro_f1_binary']:.3f}"
    )

    # Save plots and reports for this fold
    plot_training_curves(
        history, fold, str(fold_dir / f"fold_{fold}_curves.png"),
        unfreeze_epoch=unfreeze_epoch, best_epoch=best_epoch,
    )

    if len(val_true_5_np) > 0:
        plot_confusion_matrix(
            val_true_5_np, val_pred_5_np, classes_5,
            f"Fold {fold} \u2014 5-Class Confusion",
            str(fold_dir / f"fold_{fold}_confusion_5class.png"),
        )
        plot_confusion_matrix(
            val_true_bin_np, val_pred_bin_np, binary_classes,
            f"Fold {fold} \u2014 Binary Confusion",
            str(fold_dir / f"fold_{fold}_confusion_binary.png"),
        )
        write_per_class_report(
            val_true_5_np, val_pred_5_np, classes_5,
            str(fold_dir / f"fold_{fold}_per_class_report.txt"),
            fold=fold,
        )

    val_metrics["best_epoch"] = best_epoch
    val_metrics["sensitivity_derived"] = val_metrics.get("sensitivity_derived", 0)
    val_metrics["specificity_derived"] = val_metrics.get("specificity_derived", 0)
    val_metrics["macro_f1_binary_derived"] = val_metrics.get("macro_f1_binary_derived", 0)

    return val_metrics, history, csv_rows


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B3 Cat Pain Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run_name", default=None, help="Override run name")
    args, unknown = parser.parse_known_args()

    # Parse overrides from unknown args
    overrides = {}
    if args.run_name:
        overrides["run_name"] = args.run_name
    i = 0
    while i < len(unknown):
        key = unknown[i].lstrip("-")
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
            overrides[key] = unknown[i + 1]
            i += 2
        else:
            overrides[key] = True
            i += 1

    cfg = load_config(args.config, overrides)
    set_seed(cfg.get("cv_random_state", 42))
    device = get_device()

    run_dir = make_run_dir(cfg)
    save_config(cfg, run_dir)
    logger = setup_logger(run_dir)

    # Banner
    run_name = f"{cfg['run_name']}_{Path(run_dir).name.split('_')[-2]}_{Path(run_dir).name.split('_')[-1]}"
    banner = (
        "\u2554" + "\u2550" * 54 + "\u2557\n"
        f"\u2551  CAT PAIN CNN \u2014 EfficientNet-B3 Fine-tuning{' ' * 9}\u2551\n"
        f"\u2551  Run: {Path(run_dir).name:<47s}\u2551\n"
        f"\u2551  Device: {device!s:<5s} | Folds: {cfg['cv_folds']} | Epochs: {cfg['epochs']:<13d}\u2551\n"
        f"\u2551  Backbone: frozen for {cfg.get('freeze_backbone_epochs', 5)} epochs, "
        f"then {cfg.get('unfreeze_blocks', 2)} blocks free{' ' * 2}\u2551\n"
        "\u255a" + "\u2550" * 54 + "\u255d"
    )
    logger.info(f"\n{banner}")
    logger.info(f"Run: {Path(run_dir).name}")
    logger.info(f"Config loaded from: {args.config}")
    logger.info(f"Device: {device}")

    # Load manifest
    df = load_manifest(cfg)
    records = df.to_dict("records")

    logger.info(f"Dataset: {len(records)} records | {len(cfg['classes_5'])} classes | {cfg['cv_folds']}-fold CV")

    # Load YOLO only when using the legacy on-the-fly dataset
    precomp_dir = cfg.get("precomputed_frames_dir")
    if precomp_dir and (PROJECT_ROOT / precomp_dir).is_dir():
        logger.info("precomputed_frames_dir found — skipping YOLO model load")
        yolo_model, yolo_device = None, "cpu"
    else:
        yolo_result = load_yolo(cfg, device)
        if yolo_result:
            yolo_model, yolo_device = yolo_result
        else:
            yolo_model, yolo_device = None, "cpu"

    # Cross-validation — group by cat_id to prevent same-cat leakage
    # across different video_ids
    labels = np.array([r["label_5_idx"] for r in records])
    groups = np.array([r["cat_id"] for r in records])
    sgkf = StratifiedGroupKFold(
        n_splits=cfg["cv_folds"], shuffle=True,
        random_state=cfg.get("cv_random_state", 42),
    )

    all_fold_metrics: list[dict] = []
    all_fold_histories: list[dict] = []
    all_csv_rows: list[dict] = []
    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(records, labels, groups), 1):
        logger.info(f"\n\u2500\u2500 FOLD {fold_idx}/{cfg['cv_folds']} " + "\u2500" * 40)

        train_recs = [records[i] for i in train_idx]
        val_recs = [records[i] for i in val_idx]
        logger.info(f"Train: {len(train_recs)} clips | Val: {len(val_recs)} clips")

        fold_metrics, fold_history, fold_csv = train_fold(
            fold=fold_idx,
            train_records=train_recs,
            val_records=val_recs,
            cfg=cfg,
            device=device,
            run_dir=run_dir,
            logger=logger,
            yolo_model=yolo_model,
            yolo_device=yolo_device,
        )

        all_fold_metrics.append(fold_metrics)
        all_fold_histories.append(fold_history)
        all_csv_rows.extend(fold_csv)

    elapsed = time.time() - start_time

    # Save CSV
    csv_path = Path(run_dir) / "metrics_history.csv"
    save_metrics_csv(all_csv_rows, str(csv_path))

    # Summary plots
    plot_summary_curves(all_fold_histories, str(Path(run_dir) / "summary_curves.png"))

    # Summary table
    logger.info("\n" + "\u2550" * 60)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("\u2550" * 60)
    metric_keys = [
        "accuracy_5", "macro_f1_5", "weighted_f1_5",
        "sensitivity", "specificity", "macro_f1_binary",
    ]
    for key in metric_keys:
        vals = [m.get(key, 0) for m in all_fold_metrics]
        mean, std = np.mean(vals), np.std(vals)
        logger.info(f"  {key:<25s} = {mean:.3f} \u00b1 {std:.3f}")

    # Data summary for report
    data_summary = {"total": len(records)}
    for cls in cfg["classes_5"]:
        data_summary[cls] = sum(1 for r in records if r["label_5"] == cls)

    write_final_report(all_fold_metrics, cfg, run_dir, elapsed, data_summary)
    logger.info(f"\nFinal report saved to: {Path(run_dir) / 'final_report.txt'}")
    logger.info(f"Run completed in {int(elapsed // 60)}m {int(elapsed % 60)}s")


if __name__ == "__main__":
    main()
