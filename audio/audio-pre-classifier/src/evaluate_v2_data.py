#!/usr/bin/env python3
"""
Evaluate PyTorch YAMNet P(cat) against directory or manifest labels (V2-style data).

Creates ``runs/run_YYYYMMDD_HHMMSS/`` with logs, CSV, metrics, and plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# Allow ``python audio_preclassification_v3/src/evaluate_v2_data.py``
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src import config
from src.yamnet_runner import YamNetRunner

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


def _walk_audio_files(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(AUDIO_EXTENSIONS):
                out.append(Path(dirpath) / f)
    return sorted(out)


def _parse_labeled_root(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Labeled root must be LABEL=path, got: {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip().lower()
    path = Path(path.strip()).expanduser().resolve()
    return label, path


def _normalize_binary_label(raw: str) -> int | None:
    s = raw.strip().lower()
    if s in ("cat", "1", "true", "yes"):
        return 1
    if s in ("non-cat", "noncat", "other", "0", "false", "no"):
        return 0
    return None


def build_manifest(
    data_root: Path,
    manifest_path: Path | None,
    labeled_roots: list[str] | None,
    label_all: str | None,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns ``filepath`` (str) and ``label`` (0 or 1).

    Priority:
    1. ``--manifest`` CSV with columns filepath, label
    2. ``--label-all`` + ``data_root``: every audio file under ``data_root`` gets that label
       (use for trees like ``NAYA_DATA_AUG1X/*/*.mp3`` where subfolders are emotions, not cat/non-cat)
    3. ``data_root/cat`` and ``data_root/non-cat`` (or ``other``) exist
    4. ``--labeled-root`` entries (repeatable ``label=path``)
    5. Walk ``data_root``: subfolder basename maps to category; ``cat`` -> 1 else 0
    """
    if manifest_path is not None:
        df = pd.read_csv(manifest_path)
        if "filepath" not in df.columns or "label" not in df.columns:
            raise ValueError("Manifest CSV must contain columns: filepath, label")
        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            fp = Path(str(r["filepath"])).expanduser().resolve()
            lab = _normalize_binary_label(str(r["label"]))
            if lab is None:
                logger.warning("Skipping unknown label %r for %s", r["label"], fp)
                continue
            rows.append({"filepath": str(fp), "label": lab})
        return pd.DataFrame(rows)

    if label_all is not None:
        lab = _normalize_binary_label(label_all.replace("_", "-"))
        if lab is None:
            raise ValueError("--label-all must be cat or non-cat")
        data_root = data_root.resolve()
        if not data_root.is_dir():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        files = _walk_audio_files(data_root)
        return pd.DataFrame([{"filepath": str(p), "label": lab} for p in files])

    cat_dir = data_root / "cat"
    non_dir = data_root / "non-cat"
    other_dir = data_root / "other"
    if cat_dir.is_dir() and (non_dir.is_dir() or other_dir.is_dir()):
        neg = non_dir if non_dir.is_dir() else other_dir
        pos_files = _walk_audio_files(cat_dir)
        neg_files = _walk_audio_files(neg)
        rows = [{"filepath": str(p), "label": 1} for p in pos_files]
        rows.extend({"filepath": str(p), "label": 0} for p in neg_files)
        return pd.DataFrame(rows)

    if labeled_roots:
        rows = []
        for spec in labeled_roots:
            label_s, path = _parse_labeled_root(spec)
            lab = _normalize_binary_label(label_s)
            if lab is None:
                raise ValueError(f"Unknown label in --labeled-root: {label_s}")
            for p in _walk_audio_files(path):
                rows.append({"filepath": str(p), "label": lab})
        return pd.DataFrame(rows)

    # Walk data_root: folder name -> category
    rows = []
    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        cat_name = sub.name.lower()
        lab = 1 if cat_name == "cat" else 0
        for p in _walk_audio_files(sub):
            rows.append({"filepath": str(p), "label": lab})
    return pd.DataFrame(rows)


def setup_run_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "execution.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def main() -> None:
    p = argparse.ArgumentParser(description="YAMNet P(cat) evaluation on V2-style audio data")
    p.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_ROOT,
        help="Default dataset root (used for mode 2 / 4)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV with columns filepath,label (overrides directory modes)",
    )
    p.add_argument(
        "--labeled-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeatable. E.g. --labeled-root cat=/path/to/cats --labeled-root non-cat=/path/to/neg",
    )
    p.add_argument(
        "--label-all",
        choices=("cat", "non-cat"),
        default=None,
        help="Assign this label to every audio file under --data-root (recursive). "
        "Use for NAYA-style folders where subdirs are emotions, not cat vs non-cat.",
    )
    p.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    p.add_argument("--batch-size", type=int, default=config.DEFAULT_PATCH_BATCH_SIZE)
    p.add_argument(
        "--include-roaring-cats",
        action="store_true",
        help="Include YAMNet class Roaring cats (lions, tigers) in P(cat)",
    )
    p.add_argument(
        "--aggregate-cat-classes",
        choices=("sum", "max"),
        default=config.AGGREGATE_CAT_CLASSES,
        help="Combine cat-related class probs within each frame",
    )
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.RUNS_DIR / f"run_{stamp}"
    setup_run_logging(run_dir)
    logging.getLogger("src").setLevel(logging.INFO)

    logger.info("Project root: %s", config.PROJECT_ROOT)
    logger.info("Run directory: %s", run_dir)
    logger.info("Torch device: %s", config.device)

    df = build_manifest(
        args.data_root,
        args.manifest,
        args.labeled_root or None,
        args.label_all,
    )
    if df.empty:
        logger.error("No audio files found. Check paths and labeling mode.")
        sys.exit(1)

    n_cat = int((df["label"] == 1).sum())
    n_nc = int((df["label"] == 0).sum())
    logger.info("Manifest: %d files (cat=%d, non-cat=%d)", len(df), n_cat, n_nc)
    if n_cat == 0 or n_nc == 0:
        logger.warning(
            "Only one class present — ROC-AUC will be skipped; add non-cat or cat sources "
            "(see README / --labeled-root / --manifest)."
        )

    runner = YamNetRunner(
        patch_batch_size=args.batch_size,
        aggregate_cat_classes=args.aggregate_cat_classes,
        include_roaring_cats=args.include_roaring_cats,
    )

    scores: list[float] = []
    for fp in tqdm(df["filepath"].tolist(), desc="infer"):
        try:
            s = runner.predict_p_cat_path(Path(fp))
            scores.append(s)
        except Exception:
            logger.exception("Failed: %s", fp)
            scores.append(float("nan"))

    df = df.copy()
    df["p_cat_score"] = scores
    df["filename"] = df["filepath"].apply(lambda x: Path(x).name)
    df["true_label"] = df["label"].map({1: "cat", 0: "non-cat"})

    thresh = args.threshold
    df["predicted_label"] = np.where(
        df["p_cat_score"].isna(),
        "",
        np.where(df["p_cat_score"] >= thresh, "cat", "non-cat"),
    )

    out_csv = run_dir / "predictions.csv"
    df.to_csv(
        out_csv,
        columns=["filename", "true_label", "p_cat_score", "predicted_label"],
        index=False,
    )

    valid = df["p_cat_score"].notna()
    df_valid = df.loc[valid].copy()
    y_true = df_valid["label"].values.astype(int)
    y_score = df_valid["p_cat_score"].values.astype(float)
    y_pred = (y_score >= thresh).astype(int)

    metrics: dict[str, Any] = {
        "threshold": thresh,
        "n_files": int(len(df)),
        "n_scored": int(len(df_valid)),
        "n_failed": int(len(df) - len(df_valid)),
        "device": str(config.device),
    }

    if len(np.unique(y_true)) >= 2 and len(y_true) > 0:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError as e:
            logger.warning("ROC-AUC not defined: %s", e)
            metrics["roc_auc"] = None
    else:
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["f1"] = None
        metrics["roc_auc"] = None

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if len(df_valid) > 0 and len(np.unique(y_true)) >= 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["pred non-cat", "pred cat"],
            yticklabels=["true non-cat", "true cat"],
            ax=ax,
        )
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(df_valid) > 0:
        cat_scores = df_valid.loc[df_valid["label"] == 1, "p_cat_score"]
        nc_scores = df_valid.loc[df_valid["label"] == 0, "p_cat_score"]
        ax.hist(cat_scores, bins=40, alpha=0.6, label="true cat", density=True)
        ax.hist(nc_scores, bins=40, alpha=0.6, label="true non-cat", density=True)
    ax.axvline(thresh, color="k", linestyle="--", label=f"threshold={thresh}")
    ax.set_xlabel("P(cat)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("P(cat) distribution")
    fig.tight_layout()
    fig.savefig(run_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    if metrics["n_failed"]:
        logger.warning("%d files failed (see log for stack traces)", metrics["n_failed"])

    logger.info("Wrote artifacts to %s", run_dir)


if __name__ == "__main__":
    main()
