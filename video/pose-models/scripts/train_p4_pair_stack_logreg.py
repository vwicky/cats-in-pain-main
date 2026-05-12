"""
P4 Stacked Logistic Regression — Ensemble meta-learner for pain detection.

Conceptual basis
----------------
Eight binary P4 ST-GCN (spatiotemporal graph convolutional) models were trained
as one-vs-one (OvO) classifiers, each distinguishing Paining from a single
confounding class (Angry, Happy, HuntingMind, MotherCall, Mating, Resting,
Warning, Fighting).  Each model outputs P(Paining) via softmax[:, 1].

A logistic regression is trained on the stacked eight-dimensional probability
vector to produce a final binary pain / no-pain decision.

Hold-out data
-------------
The pairwise P4 models were trained on ``dataset/embeddings/pose/v2`` poses
resolved via the pose_extraction_index (manifest: final_dataset_v2.jsonl).
To avoid leakage, the meta-learner is trained on a DISJOINT hold-out:
``dataset/final_dataset.jsonl`` — an older labeled dataset whose clips and
pose files (``dataset/embeddings/pose/labeled/``) have zero overlap with
final_dataset_v2.jsonl at the video_id level.  This is the correct leakage-free
strategy because the sub-models have never seen these clips.

The hold-out uses a 5-class label scheme (Paining, Positive_Baseline, Vocalizing,
Agonistic, HuntingMind) which maps cleanly to binary: Paining=1, others=0.
Quality filtering is done via ``low_quality_pose`` (replaces ViTPose QC) and
``label_confidence`` (replaces audio_confidence).  Groups are formed by
``video_id`` (one video ≈ one cat, serving the same role as cat_id in v2).

Ensemble quality framing
------------------------
  6 strong voters  : Paining-vs-Resting, Angry, Happy, HuntingMind,
                     MotherCall, Mating
  1 marginal       : Fighting (AUC=0.739, specificity=0.385 — low specificity
                     but above random; Fighting and pain poses are genuinely
                     confusable)
  1 problematic    : Warning (AUC=0.657, specificity=0.300 — near-degenerate
                     specificity; almost always votes Paining regardless of
                     ground truth)

Defence–Paining exclusion (quantitative)
-----------------------------------------
The Defence–Paining baseline run (p4_defence_paining_baseline_20260425_110826)
showed training collapse: best_epoch=0 (no improvement over 101 epochs),
AUC=0.346 (sub-random), specificity=0.000 (model predicts all samples as
Paining), macro-F1=0.490.  A finetune run confirmed the same pattern.
Defence–Paining is excluded from the ensemble.

Usage
-----
  python model_training_v2/scripts/train_p4_pair_stack_logreg.py \\
    --model-dir model_training_v2/runs/p4_sweep_paining_restsing_20260423_235721/hparams_grid/gs0030_lr5e-05_bs8_wd0.05_g0.0_cosine/binary__Paining__Resting \\
    --model-dir model_training_v2/runs/p4_pain_finetune_8_20260424_192600/binary__Angry__Paining \\
    ... (8 total) \\
    --audio-confidence-filter

Dry run (no torch, fast):
  python ... --dry-run
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from class_subset_utils import  first_stratified_group_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_N_MODELS = 8
DEGENERATE_SPECIFICITY_THRESHOLD = 0.35
DEGENERATE_COEF_THRESHOLD = 0.3

# 5-class order for label_int in the old labeled manifest
LABELED_CLASSES = ["Agonistic", "HuntingMind", "Paining", "Positive_Baseline", "Vocalizing"]
PAIN_CLASS = "Paining"

DEFAULT_HOLD_OUT_MANIFEST = "data/dataset/final_dataset.jsonl"
DEFAULT_BASE_CONFIG = "video/pose-models/config_p4_pain_finetune.yaml"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _make_run_dir(base: Path, prefix: str = "p4_stack_logreg") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_logger(run_dir: Path) -> logging.Logger:
    log_path = run_dir / "experiment.log"
    lg = logging.getLogger("p4_stack_logreg")
    lg.setLevel(logging.DEBUG)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    lg.addHandler(ch)
    lg.addHandler(fh)
    return lg


# ---------------------------------------------------------------------------
# Hold-out data loader (dataset/final_dataset.jsonl)
# ---------------------------------------------------------------------------

def load_labeled_holdout(
    manifest_path: Path,
    *,
    audio_filter: bool,
    min_audio_confidence: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load the older labeled dataset (dataset/final_dataset.jsonl).

    Schema differences from final_dataset_v2.jsonl:
      - snippet_id  -> ``stem``
      - audio_label_10 -> ``label`` (5-class: Agonistic/HuntingMind/Paining/Positive_Baseline/Vocalizing)
      - audio_confidence -> ``label_confidence`` (0-1 scale)
      - suitable_for_training -> ``label is not None``
      - vitpose_qc -> ``low_quality_pose`` (bool, True = BAD)
      - cat_id -> ``video_id`` (one video ≈ one cat; used as group for splitting)

    Returns a DataFrame with columns compatible with PoseDataset:
      snippet_id, pose_path, pose_mask_path (None), label_int, binary_label_int,
      cat_id (= video_id), label_confidence, audio_label_10 (= label).
    """
    rows: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n_total = len(rows)

    # Filter: labeled only (label not None)
    rows = [r for r in rows if r.get("label") is not None]
    n_labeled = len(rows)

    # Filter: pose quality (low_quality_pose == False means good pose)
    rows = [r for r in rows if not r.get("low_quality_pose", True)]
    n_good_pose = len(rows)

    logger.info(
        "Hold-out manifest: %d total -> %d labeled -> %d good_pose",
        n_total, n_labeled, n_good_pose,
    )

    # Audio confidence filter
    if audio_filter:
        before = len(rows)
        rows = [r for r in rows if (r.get("label_confidence") or 0) > min_audio_confidence]
        logger.info(
            "Audio confidence filter (label_confidence > %.2f): %d -> %d rows",
            min_audio_confidence, before, len(rows),
        )
    else:
        logger.info("Audio confidence filter: disabled")

    if not rows:
        raise SystemExit("ERROR: No rows survived filtering — hold-out set is empty.")

    name_to_idx = {c: i for i, c in enumerate(LABELED_CLASSES)}

    records: list[dict] = []
    for r in rows:
        label = str(r.get("label", ""))
        if label not in name_to_idx:
            continue
        pose_path = r.get("pose_path")
        if not pose_path:
            continue
        # Verify file exists
        pp = REPO_ROOT / pose_path if not Path(pose_path).is_absolute() else Path(pose_path)
        if not pp.is_file():
            continue
        records.append({
            "snippet_id": str(r.get("stem", "")),
            "pose_path": str(pose_path).replace("\\", "/"),
            "pose_mask_path": None,
            "pose_version": "v1",
            "label_int": name_to_idx[label],
            "binary_label_int": 1 if label == PAIN_CLASS else 0,
            "cat_id": str(r.get("video_id", r.get("stem", ""))),
            "label_confidence": float(r.get("label_confidence") or 0),
            "audio_label_10": label,
        })

    df = pd.DataFrame(records)
    logger.info(
        "Hold-out: %d rows | Pain=%d NoPain=%d | unique video_ids=%d",
        len(df),
        int((df["binary_label_int"] == 1).sum()),
        int((df["binary_label_int"] == 0).sum()),
        int(df["cat_id"].nunique()),
    )
    return df


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def _load_pair_meta(model_dir: Path) -> dict:
    p = model_dir / "pair_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"pair_meta.json not found in {model_dir}")
    with open(p) as f:
        return json.load(f)


def _load_run_summary(model_dir: Path) -> dict:
    p = model_dir / "training" / "run_summary.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def _pair_name(meta: dict) -> str:
    neg = meta.get("class_neg_label_int_0", "?")
    pos = meta.get("class_pos_label_int_1", "?")
    return f"{neg}-{pos}"


def load_model(model_dir: Path, cfg: dict, device) -> object:
    """Load a P4PoseSTGCN binary checkpoint from model_dir/training/best_weights.pth."""
    import torch
    from models.p4_pose_stgcn import  P4PoseSTGCN

    wpath = model_dir / "training" / "best_weights.pth"
    if not wpath.is_file():
        raise FileNotFoundError(f"best_weights.pth not found: {wpath}")

    nc = int(cfg["data"]["n_channels"]) * 2  # use_kinematics=True always
    model = P4PoseSTGCN(
        n_frames=int(cfg["data"]["n_frames"]),
        n_keypoints=int(cfg["data"]["n_keypoints"]),
        n_channels=nc,
        n_classes=2,
    ).to(device)

    blob = torch.load(wpath, map_location=device)
    sd = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict):
        raise TypeError(f"Invalid checkpoint format at {wpath}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _run_inference(
    model,
    records: list[dict],
    cfg: dict,
    device,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray]:
    """
    Batch inference over records.  Returns (snippet_ids, probs_pain) in record order.
    snippet_ids must match the order of `records` exactly for the alignment assertion.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from data_engineering import  PoseDataset

    infer_cfg = copy.deepcopy(cfg)
    infer_cfg.setdefault("training", {})["binary_only"] = True
    infer_cfg.setdefault("data", {})["pose_cache"] = {"backend": "ram"}

    ds = PoseDataset(records, infer_cfg, is_train=False, use_kinematics=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    snippet_ids: list[str] = []
    probs: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            pose = batch["pose"].to(device)
            mask = batch["mask"].to(device)
            out = model(pose, mask)
            p = F.softmax(out["logits_binary"], dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            snippet_ids.extend(batch["snippet_id"])

    return snippet_ids, np.array(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Calibration (Platt scaling — fit on train only, apply to val)
# ---------------------------------------------------------------------------

def platt_calibrate(
    probs_train: np.ndarray,
    y_train: np.ndarray,
    probs_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a 1D Platt scaler on train probabilities, apply to val probabilities.

    IMPORTANT: fitting uses ONLY probs_train / y_train.
    probs_val is held out completely from the fit.
    This is the caller's responsibility — CalibratedClassifierCV(cv='prefit')
    does NOT do this split internally; the caller must enforce it.
    """
    from sklearn.linear_model import LogisticRegression as _LR
    scaler = _LR(C=1.0, solver="lbfgs", max_iter=500)
    scaler.fit(probs_train.reshape(-1, 1), y_train)
    cal_train = scaler.predict_proba(probs_train.reshape(-1, 1))[:, 1].astype(np.float32)
    cal_val = scaler.predict_proba(probs_val.reshape(-1, 1))[:, 1].astype(np.float32)
    return cal_train, cal_val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "sensitivity": sens,
        "specificity": spec,
        "precision_pain": prec,
        "auc_roc": auc,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n": int(len(y_true)),
        "n_pain": int(np.sum(y_true == 1)),
        "n_nopain": int(np.sum(y_true == 0)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_stack_report(
    run_dir: Path,
    run_summaries: list[dict],
    pair_names: list[str],
    split_info: dict,
    degenerate_warnings: list[str],
    coef_pairs: list[tuple[str, float]],
    val_metrics: dict,
    ablation_df: pd.DataFrame,
    calibration_mode: str,
    sub_model_metrics: list[dict],
) -> None:
    lines: list[str] = []

    def h(text: str) -> None:
        lines.append("")
        lines.append(text)
        lines.append("=" * len(text))

    def p(text: str = "") -> None:
        lines.append(text)

    h("P4 Stacked Logistic Regression — Scientific Report")
    p(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    p(f"Run directory: {run_dir}")

    h("1. Hold-out Data (Leakage-free)")
    p("Source: dataset/final_dataset.jsonl  (older labeled dataset)")
    p("Pose files: dataset/embeddings/pose/labeled/  (v1 ViTPose, pixel coords)")
    p("")
    p("The pairwise P4 sub-models were trained on final_dataset_v2.jsonl clips")
    p("(v2 poses via pose_extraction_index).  The two datasets have ZERO overlap")
    p("at the video_id level — confirmed by cross-reference.  This is the correct")
    p("leakage-free design: the sub-models have never seen any of these clips.")
    p("")
    p("Label scheme: 5-class (Paining, Positive_Baseline, Vocalizing, Agonistic,")
    p("HuntingMind).  Binary: Paining=1, all others=0.")
    p("Group variable: video_id (one video ≈ one cat, analogous to cat_id in v2).")

    h("2. Ensemble Composition")
    p("8 binary P4 ST-GCN models (OvO: each Paining vs one confounding class).")
    p("")
    p(f"  {'Pair':<28} {'Trn-AUC':>8}  {'Trn-Sens':>8}  {'Trn-Spec':>8}  Role")
    p(f"  {'-'*28} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*25}")
    for pname, rs, sm in zip(pair_names, run_summaries, sub_model_metrics):
        m = rs.get("metrics", {})
        auc = m.get("val_auc_roc", float("nan"))
        sens = m.get("val_sensitivity", float("nan"))
        spec = m.get("val_specificity", float("nan"))
        hol_auc = sm.get("raw_auc_val", float("nan"))
        role = "strong"
        if spec < DEGENERATE_SPECIFICITY_THRESHOLD:
            role = "PROBLEMATIC (spec < 0.35)"
        elif spec < 0.5 or auc < 0.70:
            role = "marginal"
        p(f"  {pname:<28} {auc:>8.3f}  {sens:>8.3f}  {spec:>8.3f}  {role}")
    p("")
    p("Note: AUC/Sens/Spec are from each model's OWN pairwise train/val split")
    p("(v2 data), NOT from this hold-out.  Hold-out per-model AUC is in coefficients.csv.")

    h("3. Defence-Paining Exclusion (Quantitative Justification)")
    p("Model: p4_defence_paining_baseline_20260425_110826 / binary__Defence__Paining")
    p("")
    p("  best_epoch : 0      (never improved beyond random initialization)")
    p("  AUC        : 0.346  (sub-chance; worse than a coin flip)")
    p("  specificity: 0.000  (always predicts Paining; trivial majority predictor)")
    p("  macro-F1   : 0.490  (near-degenerate)")
    p("  n_train    : 604    n_val: 151")
    p("")
    p("A finetune run confirmed the same collapse pattern.  The Defence class has")
    p("too few samples relative to Paining for this pair to learn a discriminative")
    p("boundary.  Including this classifier would inject pure Paining-biased noise.")
    p("Exclusion is statistically and scientifically justified.")

    h("4. Warning Sub-Classifier (Problematic Voter)")
    p("Pair: Paining-Warning  |  OvO training split metrics:")
    p("  AUC : 0.657   specificity : 0.300   sensitivity : 0.979")
    p("")
    p("Near-degenerate specificity (0.300 < 0.35 threshold).  The classifier")
    p("almost always votes Paining regardless of actual label.  It is retained")
    p("because AUC > 0.5 indicates some real signal, but the logistic regression")
    p("meta-learner is expected to assign it a low or negative weight.")
    if degenerate_warnings:
        p("")
        p("Degenerate-voter coefficient check TRIGGERED:")
        for w in degenerate_warnings:
            p(f"  !! {w}")
    else:
        p("")
        p("Coefficient check: PASS — no degenerate voter received suspiciously high |coef|.")

    h("5. Logistic Regression Coefficients")
    p(f"  {'Pair':<28} {'Coefficient':>12}  {'|Coef|':>8}  {'Rank':>5}")
    p(f"  {'-'*28} {'-'*12}  {'-'*8}  {'-'*5}")
    for rank, (pname, coef) in enumerate(
        sorted(coef_pairs, key=lambda x: abs(x[1]), reverse=True), 1
    ):
        p(f"  {pname:<28} {coef:>12.4f}  {abs(coef):>8.4f}  {rank:>5}")

    h("6. Hold-out Split Statistics")
    p(f"  Total rows after all filters  : {split_info['n_total']}")
    p(f"  Train rows                    : {split_info['n_train_rows']}")
    p(f"  Val rows                      : {split_info['n_val_rows']}")
    p(f"  Train unique video_ids        : {split_info['n_train_cats']}")
    p(f"  Val unique video_ids          : {split_info['n_val_cats']}")
    p(f"  Train Pain rows               : {split_info['n_train_pain']}")
    p(f"  Val Pain rows                 : {split_info['n_val_pain']}")
    p(f"  Achieved val fraction         : {split_info['val_fraction_achieved']:.4f}")
    p(f"  Split seed                    : {split_info['split_seed']}")
    p(f"  Calibration mode              : {calibration_mode}")

    h("7. Stacked Ensemble Validation Metrics")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            p(f"  {k:<25}: {v:.4f}")
        else:
            p(f"  {k:<25}: {v}")

    h("8. Per-Model Ablation (column i set to 0.5 = maximally uncertain)")
    p(f"  {'Pair':<28} {'Baseline F1':>11}  {'Ablated F1':>10}  {'Delta F1':>9}")
    p(f"  {'-'*28} {'-'*11}  {'-'*10}  {'-'*9}")
    for _, row in ablation_df.sort_values("delta_f1").iterrows():
        p(
            f"  {row['pair_name']:<28} {row['baseline_macro_f1']:>11.4f}  "
            f"{row['ablated_macro_f1']:>10.4f}  {row['delta_f1']:>+9.4f}"
        )
    p("")
    p("Most negative delta_f1 = most important sub-classifier.")
    p("Near-zero or positive delta = meta-learner learned to ignore this model.")

    h("9. Interpretation Notes for Thesis Defence")
    p("- The stacked ensemble treats each OvO vote as a soft probability signal.")
    p("  A positive logistic coefficient means higher P(Paining) from that model")
    p("  increases the ensemble's global pain score.")
    p("- Coefficients near zero indicate the meta-learner ignored that sub-classifier")
    p("  (expect Warning to receive low or negative weight given its specificity=0.30).")
    p("- AUC-ROC is the primary calibration-independent metric; macro-F1 is the")
    p("  primary threshold-dependent metric given class imbalance.")
    p("- The hold-out label scheme (5-class) does not match the training label scheme")
    p("  (10-class).  This means sub-models trained against classes absent in the hold-out")
    p("  (e.g., Fighting, Mating) are evaluated on a distribution shift.  The ensemble's")
    p("  meta-learner absorbs this; individual sub-model AUCs on the hold-out should be")
    p("  interpreted cautiously.  The binary pain/no-pain signal is unaffected.")
    p("- See ablation_results.csv and calibration_curve.{png,csv} for visual evidence.")

    (run_dir / "stack_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a stacked logistic regression on 8 P4 pairwise pain-vs-class models. "
            "Uses dataset/final_dataset.jsonl as leakage-free hold-out (v1 poses, disjoint "
            "from the v2 training data used for the pairwise sub-models)."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_BASE_CONFIG,
        help="Base YAML config (used for normalization / model architecture settings).",
    )
    parser.add_argument(
        "--hold-out-manifest",
        default=DEFAULT_HOLD_OUT_MANIFEST,
        help="Path to the labeled hold-out manifest (default: dataset/final_dataset.jsonl).",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dirs",
        action="append",
        metavar="DIR",
        required=True,
        help="Path to a binary__* directory with pair_meta.json + training/best_weights.pth. "
             "Repeat for each of the 8 models.",
    )
    parser.add_argument(
        "--audio-confidence-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter by label_confidence > --min-audio-confidence (default: on).",
    )
    parser.add_argument("--min-audio-confidence", type=float, default=0.7,
                        help="Minimum label_confidence to keep a row (default: 0.7).")
    parser.add_argument("--split-val-fraction", type=float, default=1.0 / 3.0,
                        help="Target val fraction for video_id-stratified split (default: ~0.333).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply per-sub-model Platt scaling (fit on train, apply to val).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--experiment-name", default="p4_stack_logreg")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip torch inference; print data/split stats and exit.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit inference to first N rows (CPU smoke test).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0. Startup assertion — fail loudly before touching any data
    # ------------------------------------------------------------------
    model_dirs = [Path(d) for d in args.model_dirs]

    if len(model_dirs) != EXPECTED_N_MODELS:
        raise SystemExit(
            f"Expected exactly {EXPECTED_N_MODELS} model directories, got {len(model_dirs)}. "
            "If you added or removed a pair, update the stacker intentionally -- "
            "a silent shape change in X is hard to debug downstream."
        )

    for md in model_dirs:
        if not (md / "pair_meta.json").is_file():
            raise SystemExit(f"pair_meta.json missing in: {md}")
        if not (md / "training" / "best_weights.pth").is_file():
            raise SystemExit(f"training/best_weights.pth missing in: {md}")

    # ------------------------------------------------------------------
    # 1. Config and run directory
    # ------------------------------------------------------------------
    cfg_path = REPO_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    runs_dir = REPO_ROOT / cfg.get("output", {}).get("runs_dir", "video/pose-models/runs")
    run_dir = _make_run_dir(runs_dir, args.experiment_name)

    # Need logger before data loading (it logs to run_dir)
    lg = _setup_logger(run_dir)
    lg.info("P4 stacked logreg | run_dir=%s", run_dir)
    lg.info("Hold-out manifest: %s", args.hold_out_manifest)
    lg.info("Models:")
    for i, md in enumerate(model_dirs):
        lg.info("  [%d] %s", i, md)

    # ------------------------------------------------------------------
    # 2. Load hold-out dataset
    # ------------------------------------------------------------------
    manifest_path = REPO_ROOT / args.hold_out_manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Hold-out manifest not found: {manifest_path}")

    df = load_labeled_holdout(
        manifest_path,
        audio_filter=args.audio_confidence_filter,
        min_audio_confidence=args.min_audio_confidence,
        logger=lg,
    )

    if args.limit is not None:
        df = df.iloc[: args.limit].copy()
        lg.info("--limit: using first %d rows", len(df))

    # ------------------------------------------------------------------
    # 3. Dry-run: print stats and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        lg.info("=== DRY RUN MODE ===")
        lg.info("Rows after all filters: %d", len(df))
        lg.info("Pain rows (binary_label_int==1): %d", int((df["binary_label_int"] == 1).sum()))
        lg.info("Unique video_ids: %d", int(df["cat_id"].nunique()))

        split_cfg = copy.deepcopy(cfg)
        split_cfg.setdefault("split", {})["val_fraction"] = args.split_val_fraction
        split_cfg["split"]["group_field"] = "cat_id"
        split_cfg["split"]["random_state"] = split_cfg.get("split", {}).get("random_state", 42)

        tr, va, err = first_stratified_group_split(df, split_cfg)
        if err:
            lg.warning("Split error: %s", err)
        else:
            lg.info(
                "Split (val_fraction=%.4f): train=%d rows / %d video_ids  |  val=%d rows / %d video_ids",
                args.split_val_fraction,
                len(tr), tr["cat_id"].nunique(),
                len(va), va["cat_id"].nunique(),
            )
            lg.info(
                "Pain in train=%d  Pain in val=%d",
                int((tr["binary_label_int"] == 1).sum()),
                int((va["binary_label_int"] == 1).sum()),
            )
        for md in model_dirs:
            meta = _load_pair_meta(md)
            lg.info("  Model checkpoint OK: %s", _pair_name(meta))
        lg.info("Dry run complete.")
        return

    # ------------------------------------------------------------------
    # 4. Load pair metadata and run summaries
    # ------------------------------------------------------------------
    model_metas = [_load_pair_meta(md) for md in model_dirs]
    run_summaries = [_load_run_summary(md) for md in model_dirs]
    pair_names = [_pair_name(m) for m in model_metas]
    lg.info("Pair names: %s", pair_names)

    # ------------------------------------------------------------------
    # 5. Device
    # ------------------------------------------------------------------
    import torch

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    lg.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 6. Inference — all 8 models over full hold-out df
    # ------------------------------------------------------------------
    records = df.to_dict("records")
    model_outputs: list[tuple[list[str], np.ndarray]] = []

    infer_cfg = copy.deepcopy(cfg)
    infer_cfg.setdefault("training", {})["binary_only"] = True

    for i, (md, pname) in enumerate(zip(model_dirs, pair_names)):
        lg.info("Loading model [%d/%d]: %s", i + 1, EXPECTED_N_MODELS, pname)
        model = load_model(md, infer_cfg, device)
        lg.info("  Running inference on %d records ...", len(records))
        t0 = time.time()
        sids, probs = _run_inference(model, records, infer_cfg, device, batch_size=args.batch_size)
        lg.info("  Done in %.1f s | P(pain) mean=%.4f std=%.4f",
                time.time() - t0, probs.mean(), probs.std())
        model_outputs.append((sids, probs))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 7. snippet_id alignment assertion (hard failure on mismatch)
    # ------------------------------------------------------------------
    ref_sids = model_outputs[0][0]
    for i, (sids, _) in enumerate(model_outputs[1:], 1):
        if sids != ref_sids:
            first_mismatch = next(
                (j for j, (a, b) in enumerate(zip(sids, ref_sids)) if a != b),
                len(ref_sids),
            )
            raise AssertionError(
                f"snippet_id mismatch between model 0 ({pair_names[0]}) and "
                f"model {i} ({pair_names[i]}) at row index {first_mismatch}. "
                "Cannot build stacked X safely — all 8 models must produce "
                "output in identical snippet_id order."
            )
    lg.info("snippet_id alignment: OK (all %d models, %d rows)", EXPECTED_N_MODELS, len(ref_sids))

    snippet_ids_full = list(df["snippet_id"].astype(str))
    assert list(ref_sids) == snippet_ids_full, (
        "Internal error: inference snippet_id list does not match df['snippet_id'] order."
    )

    X_full = np.column_stack([probs for _, probs in model_outputs]).astype(np.float32)
    y_full = df["binary_label_int"].values.astype(np.int64)

    # ------------------------------------------------------------------
    # 8. Train/val split by video_id (group)
    # ------------------------------------------------------------------
    split_cfg = copy.deepcopy(cfg)
    split_cfg.setdefault("split", {})["val_fraction"] = args.split_val_fraction
    split_cfg["split"]["group_field"] = "cat_id"  # cat_id = video_id in this dataset
    split_cfg["split"].setdefault("random_state", 42)

    train_df, val_df, split_err = first_stratified_group_split(df, split_cfg)
    if split_err:
        raise SystemExit(f"Split failed: {split_err}")

    # Hard Pain-balance check
    n_train_pain = int((train_df["binary_label_int"] == 1).sum())
    n_val_pain = int((val_df["binary_label_int"] == 1).sum())
    if n_train_pain == 0:
        raise SystemExit("FATAL: No Pain rows in meta-train split — stacker cannot be trained.")
    if n_val_pain == 0:
        raise SystemExit("FATAL: No Pain rows in meta-val split — stacker cannot be evaluated.")

    sid_to_idx = {sid: i for i, sid in enumerate(snippet_ids_full)}
    train_idx = [sid_to_idx[str(sid)] for sid in train_df["snippet_id"]]
    val_idx = [sid_to_idx[str(sid)] for sid in val_df["snippet_id"]]

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    n_train_cats = int(train_df["cat_id"].nunique())
    n_val_cats = int(val_df["cat_id"].nunique())
    val_frac_achieved = len(val_idx) / len(snippet_ids_full)

    split_info = {
        "n_total": len(snippet_ids_full),
        "n_train_rows": len(train_idx),
        "n_val_rows": len(val_idx),
        "n_train_cats": n_train_cats,
        "n_val_cats": n_val_cats,
        "n_train_pain": n_train_pain,
        "n_val_pain": n_val_pain,
        "val_fraction_target": args.split_val_fraction,
        "val_fraction_achieved": round(val_frac_achieved, 6),
        "split_seed": int(split_cfg["split"]["random_state"]),
        "group_field": "video_id (mapped to cat_id for compatibility)",
    }
    lg.info(
        "Split: train=%d rows/%d vids/%d pain  |  val=%d rows/%d vids/%d pain  |  frac=%.4f",
        split_info["n_train_rows"], n_train_cats, n_train_pain,
        split_info["n_val_rows"], n_val_cats, n_val_pain,
        val_frac_achieved,
    )

    # ------------------------------------------------------------------
    # 9. Optional Platt calibration (train-only fit, val-only eval)
    # ------------------------------------------------------------------
    calibration_mode = "none"
    sub_model_metrics: list[dict] = []

    try:
        from sklearn.metrics import roc_auc_score as _auc
        _sklearn_ok = True
    except ImportError:
        _sklearn_ok = False

    X_train_m = X_train.copy()
    X_val_m = X_val.copy()

    for i, pname in enumerate(pair_names):
        raw_auc = float(_auc(y_val, X_val[:, i])) if _sklearn_ok and len(np.unique(y_val)) > 1 else float("nan")
        entry: dict = {"pair_name": pname, "raw_auc_val": raw_auc}

        if args.calibrate:
            cal_tr, cal_va = platt_calibrate(X_train[:, i], y_train, X_val[:, i])
            cal_auc = float(_auc(y_val, cal_va)) if _sklearn_ok and len(np.unique(y_val)) > 1 else float("nan")
            X_train_m[:, i] = cal_tr
            X_val_m[:, i] = cal_va
            entry["cal_auc_val"] = cal_auc
            lg.info("  Platt [%s]: raw_auc=%.4f -> cal_auc=%.4f", pname, raw_auc, cal_auc)

        sub_model_metrics.append(entry)

    if args.calibrate:
        calibration_mode = "platt"
        X_train, X_val = X_train_m, X_val_m
        lg.info("Platt calibration applied (fit on train, scored on val).")

    # ------------------------------------------------------------------
    # 10. Logistic regression meta-learner
    # ------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression

    lg.info("Fitting LogisticRegression (max_iter=2000, balanced) ...")
    lr = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=int(split_cfg["split"]["random_state"]),
    )
    lr.fit(X_train, y_train)

    converged = bool(lr.n_iter_[0] < 2000)
    if not converged:
        lg.warning(
            "LogisticRegression did not converge in 2000 iterations — "
            "results may be unreliable."
        )
    else:
        lg.info("LogisticRegression converged in %d iterations.", int(lr.n_iter_[0]))

    coef = lr.coef_[0]
    coef_pairs = list(zip(pair_names, [float(c) for c in coef]))
    for pname, c in sorted(coef_pairs, key=lambda x: abs(x[1]), reverse=True):
        lg.info("  coef[%s] = %.4f", pname, c)

    # ------------------------------------------------------------------
    # 11. Degenerate-voter coefficient check
    # ------------------------------------------------------------------
    degenerate_warnings: list[str] = []
    for i, (pname, c) in enumerate(coef_pairs):
        rs = run_summaries[i]
        spec = rs.get("metrics", {}).get("val_specificity", None)
        if spec is None:
            continue
        if spec < DEGENERATE_SPECIFICITY_THRESHOLD and abs(c) > DEGENERATE_COEF_THRESHOLD:
            msg = (
                f"'{pname}' specificity={spec:.3f} < {DEGENERATE_SPECIFICITY_THRESHOLD} "
                f"but |coef|={abs(c):.3f} > {DEGENERATE_COEF_THRESHOLD} — "
                "this low-specificity classifier received a large weight"
            )
            degenerate_warnings.append(msg)
            lg.warning("DEGENERATE VOTER: %s", msg)

    if not degenerate_warnings:
        lg.info("Degenerate-voter coefficient check: PASS")

    # ------------------------------------------------------------------
    # 12. Validation metrics
    # ------------------------------------------------------------------
    y_prob_val = lr.predict_proba(X_val)[:, 1]
    y_pred_val = lr.predict(X_val)
    val_metrics = compute_binary_metrics(y_val, y_pred_val, y_prob_val)

    y_prob_train = lr.predict_proba(X_train)[:, 1]
    y_pred_train = lr.predict(X_train)
    train_metrics = compute_binary_metrics(y_train, y_pred_train, y_prob_train)

    lg.info(
        "Val: acc=%.4f macro_f1=%.4f auc=%.4f sens=%.4f spec=%.4f",
        val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["auc_roc"],
        val_metrics["sensitivity"], val_metrics["specificity"],
    )

    # ------------------------------------------------------------------
    # 13. Calibration curve (reliability diagram)
    # ------------------------------------------------------------------
    from sklearn.calibration import calibration_curve as _cal_curve

    if len(np.unique(y_val)) > 1 and len(y_val) >= 10:
        frac_pos, mean_pred = _cal_curve(y_val, y_prob_val, n_bins=10, strategy="uniform")
    else:
        frac_pos = mean_pred = np.array([])

    cal_curve_df = pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": frac_pos,
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    if len(mean_pred):
        ax.plot(mean_pred, frac_pos, "s-", label="Stacked ensemble")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted P(Paining)")
    ax.set_ylabel("Fraction of Paining samples")
    ax.set_title("Reliability Diagram — Stacked Logistic Regression")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(run_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 14. Per-model ablation
    # ------------------------------------------------------------------
    from sklearn.metrics import f1_score as _f1

    baseline_f1 = float(_f1(y_val, y_pred_val, average="macro", zero_division=0))
    ablation_rows: list[dict] = []

    for i, pname in enumerate(pair_names):
        X_abl = X_val.copy()
        X_abl[:, i] = 0.5  # maximally uncertain / neutral
        y_pred_abl = lr.predict(X_abl)
        abl_f1 = float(_f1(y_val, y_pred_abl, average="macro", zero_division=0))
        delta = abl_f1 - baseline_f1
        ablation_rows.append({
            "pair_name": pname,
            "baseline_macro_f1": round(baseline_f1, 6),
            "ablated_macro_f1": round(abl_f1, 6),
            "delta_f1": round(delta, 6),
        })
        lg.info(
            "  Ablation [%s]: baseline=%.4f ablated=%.4f delta=%+.4f",
            pname, baseline_f1, abl_f1, delta,
        )

    ablation_df = pd.DataFrame(ablation_rows).sort_values("delta_f1")

    # ------------------------------------------------------------------
    # 15. Probability output tables (always written)
    # ------------------------------------------------------------------
    def _prob_df(ids: list[str], X: np.ndarray, y_prob: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
        d: dict = {"snippet_id": ids}
        for j, pname in enumerate(pair_names):
            col = pname.replace("-", "_").replace(" ", "_")
            d[f"prob_{col}"] = X[:, j].tolist()
        d["prob_stack"] = y_prob.tolist()
        d["y_true"] = y_true.tolist()
        return pd.DataFrame(d)

    train_sids = [str(sid) for sid in train_df["snippet_id"]]
    val_sids = [str(sid) for sid in val_df["snippet_id"]]

    _prob_df(train_sids, X_train, y_prob_train, y_train).to_csv(run_dir / "prob_train.csv", index=False)
    _prob_df(val_sids, X_val, y_prob_val, y_val).to_csv(run_dir / "prob_val.csv", index=False)

    # ------------------------------------------------------------------
    # 16. stack_meta.json
    # ------------------------------------------------------------------
    stack_meta: dict = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(args.config),
        "hold_out_manifest": str(args.hold_out_manifest),
        "hold_out_note": (
            "data/dataset/final_dataset.jsonl — older labeled dataset with zero video_id "
            "overlap with final_dataset_v2.jsonl (training set). Leakage-free by design."
        ),
        "model_dirs": [str(md) for md in model_dirs],
        "pair_names": pair_names,
        "audio_confidence_filter": {
            "enabled": bool(args.audio_confidence_filter),
            "field": "label_confidence",
            "min_value": args.min_audio_confidence,
        },
        "split": split_info,
        "calibration_mode": calibration_mode,
        "sub_model_metrics": sub_model_metrics,
        "lr_converged": converged,
        "lr_n_iter": int(lr.n_iter_[0]),
        "degenerate_voter_warnings": degenerate_warnings,
        "coefficients": {pname: float(c) for pname, c in coef_pairs},
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
        "label_scheme_note": (
            "Hold-out uses 5-class labels (Paining/Positive_Baseline/Vocalizing/Agonistic/"
            "HuntingMind). Sub-models were trained on 10-class (audio_label_10). "
            "Binary pain/no-pain signal is consistent; per-model AUC on hold-out reflects "
            "a distribution shift (non-Paining classes are different from training)."
        ),
    }

    # ------------------------------------------------------------------
    # 17. Write all artifacts
    # ------------------------------------------------------------------
    cfg_save = copy.deepcopy(cfg)
    cfg_save["stack_logreg"] = {
        "hold_out_manifest": str(args.hold_out_manifest),
        "model_dirs": [str(md) for md in model_dirs],
        "pair_names": pair_names,
        "audio_confidence_filter": args.audio_confidence_filter,
        "calibrate": args.calibrate,
        "split_val_fraction": args.split_val_fraction,
    }
    (run_dir / "config_used.yaml").write_text(
        yaml.dump(cfg_save, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    (run_dir / "stack_meta.json").write_text(
        json.dumps(stack_meta, indent=2, default=str), encoding="utf-8"
    )

    coef_df = pd.DataFrame(
        [{"pair_name": pn, "coef": c, "abs_coef": abs(c)} for pn, c in coef_pairs]
    ).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coef_df["rank"] = range(1, len(coef_df) + 1)
    coef_df.to_csv(run_dir / "coefficients.csv", index=False)

    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    ablation_df.to_csv(run_dir / "ablation_results.csv", index=False)
    cal_curve_df.to_csv(run_dir / "calibration_curve.csv", index=False)

    write_stack_report(
        run_dir,
        run_summaries,
        pair_names,
        split_info,
        degenerate_warnings,
        coef_pairs,
        val_metrics,
        ablation_df,
        calibration_mode,
        sub_model_metrics,
    )

    lg.info("=== All artifacts written to: %s ===", run_dir)
    for fname in [
        "stack_report.txt", "stack_meta.json", "val_metrics.json",
        "coefficients.csv", "ablation_results.csv",
        "calibration_curve.png", "calibration_curve.csv",
        "prob_train.csv", "prob_val.csv",
    ]:
        lg.info("  %s", run_dir / fname)


if __name__ == "__main__":
    main()
