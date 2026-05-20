"""
Load the 9 pairwise P4PoseSTGCN base models and the LogReg L2 meta-learner
from ``runs/pose-models/stgcn_dlc_stack_20260504_235843/run_05/``.

Inference flow
--------------
1. For each of the 9 pairwise models, call ``run_pairwise_inference`` to get
   P(Paining | pair) ∈ [0, 1].
2. Assemble a feature vector in the order matching the stacking training columns
   (alphabetically sorted pair names, confirmed from prob_val.csv).
3. Feed into the meta-learner (joblib-serialised sklearn LogisticRegression).

Column order (from stacking run prob_val.csv):
  Angry|Paining, Defence|Paining, Fighting|Paining, Happy|Paining,
  HuntingMind|Paining, Mating|Paining, MotherCall|Paining,
  Paining|Resting, Paining|Warning
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from inference.import_hygiene import (
    drop_cached_models_package_unless_origin_contains,
    prioritize_sys_path,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"

# Default stack run to use
DEFAULT_STACK_RUN = (
    REPO_ROOT / "runs" / "pose-models" / "stgcn_dlc_stack_20260504_235843" / "run_05"
)
DEFAULT_BUNDLE = (
    REPO_ROOT / "runs" / "pose-models" / "p4_pairwise_ensemble_bundle_20260428"
)

# Exact column order that the meta-learner was trained on (alphabetical by pair name)
FEATURE_COLUMNS = [
    "Angry|Paining",
    "Defence|Paining",
    "Fighting|Paining",
    "Happy|Paining",
    "HuntingMind|Paining",
    "Mating|Paining",
    "MotherCall|Paining",
    "Paining|Resting",
    "Paining|Warning",
]

# Minimal config matching the pairwise training config (config_p4_pairwise.yaml)
_PAIRWISE_CFG: dict[str, Any] = {
    "data": {
        "n_frames": 35,
        "n_keypoints": 17,
        "n_channels": 3,
        "pose_cache": {"backend": "off"},  # avoid RAM cache side-effects
    },
    "normalization": {
        "root_joint": 0,
        "scale_joints": [5, 6],
        "clamp_xy": [-0.5, 1.5],
        "drop_low_confidence_frames": True,
        "confidence_threshold": 0.10,
    },
    "training": {
        "binary_only": True,
    },
}


def _bootstrap_pose_models() -> None:
    """
    Prefer ``video/pose-models`` on ``sys.path`` and drop a foreign ``models``
    package (e.g. from AudioSep) left in ``sys.modules`` by an earlier window.
    """
    prioritize_sys_path(POSE_MODELS_ROOT)
    drop_cached_models_package_unless_origin_contains("pose-models")


def _remap_old_path(p: str) -> Path:
    """
    Remap stale ``model_training_v2/runs/`` paths to the new ``runs/pose-models/`` layout.
    If the path already exists as-is, return it unchanged.
    """
    candidate = Path(p)
    if candidate.is_dir():
        return candidate
    # Remap: replace old repo prefix + model_training_v2/runs/ with new layout
    new = p.replace(
        str(REPO_ROOT / "model_training_v2" / "runs"),
        str(REPO_ROOT / "runs" / "pose-models"),
    )
    remapped = Path(new)
    if remapped.is_dir():
        return remapped
    raise FileNotFoundError(
        f"Pairwise model dir not found (tried original and remapped):\n"
        f"  original: {p}\n"
        f"  remapped: {new}"
    )


def _load_pair_meta(model_dir: Path) -> dict[str, Any]:
    p = model_dir / "pair_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"pair_meta.json not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _pair_name(meta: dict[str, Any]) -> str:
    """Return canonical ``min|max`` pair name from pair_meta."""
    neg = str(meta.get("class_neg_label_int_0", ""))
    pos = str(meta.get("class_pos_label_int_1", ""))
    a, b = min(neg, pos), max(neg, pos)
    return f"{a}|{b}"


def load_pairwise_models(
    stack_run_dir: str | Path = DEFAULT_STACK_RUN,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load the 9 pairwise P4PoseSTGCN models from the pain-subset bundle.

    Returns a dict mapping pair_name → {model, meta, dir, pain_class_idx}.
    ``pain_class_idx`` is the index in the binary [neg, pos] output whose
    softmax value equals P(Paining).
    """
    _bootstrap_pose_models()
    from models.p4_pose_stgcn import P4PoseSTGCN  # noqa: E402 (under POSE_MODELS_ROOT)

    if device is None:
        device = torch.device("cpu")

    stack_run_dir = Path(stack_run_dir)
    pain_subset_json = stack_run_dir.parent / "pairwise_pain_subset.json"
    if not pain_subset_json.is_file():
        raise FileNotFoundError(f"pairwise_pain_subset.json not found: {pain_subset_json}")

    with open(pain_subset_json, encoding="utf-8") as fh:
        subset = json.load(fh)

    kept_paths: list[str] = subset.get("kept_paths", [])
    if not kept_paths:
        raise ValueError("pairwise_pain_subset.json has no kept_paths")

    loaded: dict[str, Any] = {}
    cfg = _PAIRWISE_CFG

    for raw_path in kept_paths:
        model_dir = _remap_old_path(raw_path)
        meta = _load_pair_meta(model_dir)
        pair = _pair_name(meta)

        ckpt = model_dir / "training" / "best_weights.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        n_channels = int(cfg["data"]["n_channels"]) * 2  # kinematics doubles channels
        model = P4PoseSTGCN(
            n_frames=int(cfg["data"]["n_frames"]),
            n_keypoints=int(cfg["data"]["n_keypoints"]),
            n_channels=n_channels,
            n_classes=2,
        ).to(device)

        blob = torch.load(str(ckpt), map_location=device)
        state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
        model.load_state_dict(state, strict=True)
        model.eval()

        # The stacking always uses probs[:, 1] as P(positive).
        # pair_meta tells us which class is at index 1 (class_pos_label_int_1).
        pain_class_idx = 1  # always index 1 = positive class = Paining

        loaded[pair] = {
            "model": model,
            "meta": meta,
            "dir": str(model_dir),
            "pain_class_idx": pain_class_idx,
        }
        logger.info("Loaded pairwise model %s from %s", pair, model_dir)

    missing = [col for col in FEATURE_COLUMNS if col not in loaded]
    if missing:
        raise ValueError(f"Missing pairwise models for columns: {missing}")

    return loaded


def run_pairwise_inference(
    pair_models: dict[str, Any],
    record: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """
    Run each pairwise model on ``record`` and return {pair_name: P(Paining)}.
    """
    _bootstrap_pose_models()
    from data_engineering import PoseDataset  # noqa: E402

    cfg = _PAIRWISE_CFG
    pair_probs: dict[str, float] = {}

    for pair, info in pair_models.items():
        model: torch.nn.Module = info["model"]
        pain_idx: int = info["pain_class_idx"]

        ds = PoseDataset([record], cfg, is_train=False, use_kinematics=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                pose = batch["pose"].to(device)
                mask = batch["mask"].to(device)
                out = model(pose, mask)
                logits = out["logits_binary"]
                probs = F.softmax(logits, dim=1)
                p_pain = float(probs[0, pain_idx].item())
                pair_probs[pair] = p_pain

    return pair_probs


def load_meta_model(stack_run_dir: str | Path = DEFAULT_STACK_RUN) -> Any:
    """Load the LogReg L2 meta-learner from run_05/model.pkl."""
    pkl = Path(stack_run_dir) / "model.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"Meta-model not found: {pkl}")
    meta_model = joblib.load(str(pkl))
    logger.info("Loaded meta-model from %s", pkl)
    return meta_model


def run_meta_inference(
    meta_model: Any,
    pair_probs: dict[str, float],
) -> dict[str, Any]:
    """
    Assemble the feature vector in FEATURE_COLUMNS order and run the meta-model.

    Returns a dict with per-pair probs, meta-model probabilities, and final decision.
    """
    feature_vec = np.array(
        [pair_probs[col] for col in FEATURE_COLUMNS], dtype=np.float64
    ).reshape(1, -1)

    meta_proba = meta_model.predict_proba(feature_vec)[0]  # shape (n_classes,)
    meta_classes_raw: list[Any] = list(meta_model.classes_)
    meta_classes: list[str] = [str(cls) for cls in meta_classes_raw]

    # Map class labels to probabilities
    class_probs = {str(cls): float(p) for cls, p in zip(meta_classes, meta_proba)}

    # Infer pain class robustly:
    # - prefer explicit labels containing "pain"/"Paining"
    # - for numeric binary labels (0/1), treat label 1 as pain
    pain_idx: int | None = None
    for i, cls in enumerate(meta_classes):
        cl = cls.lower()
        if "pain" in cl or cls == "Paining":
            pain_idx = i
            break
    if pain_idx is None and len(meta_classes_raw) == 2:
        numeric_pairs: list[tuple[int, float]] = []
        for i, cls in enumerate(meta_classes_raw):
            try:
                numeric_pairs.append((i, float(cls)))
            except (TypeError, ValueError):
                continue
        if len(numeric_pairs) == 2:
            pain_idx = max(numeric_pairs, key=lambda x: x[1])[0]

    if pain_idx is not None and 0 <= pain_idx < len(meta_proba):
        p_pain = float(meta_proba[pain_idx])
        decision = "pain" if p_pain >= 0.5 else "non_pain"
    else:
        p_pain = 0.0
        decision = "non_pain"

    # Export per-feature meta-model coefficients for the inferred pain class.
    feature_weights: dict[str, float] = {}
    try:
        coef = np.asarray(meta_model.coef_, dtype=np.float64)
        if coef.ndim == 2 and coef.shape[1] == len(FEATURE_COLUMNS):
            weight_row: np.ndarray | None = None
            if coef.shape[0] == 1:
                # sklearn binary logistic stores one row for the positive class
                # (class index 1 when there are two classes).
                weight_row = coef[0]
            elif pain_idx is not None and pain_idx < coef.shape[0]:
                weight_row = coef[pain_idx]
            if weight_row is not None:
                feature_weights = {
                    col: float(weight_row[j]) for j, col in enumerate(FEATURE_COLUMNS)
                }
    except Exception:
        feature_weights = {}

    return {
        "feature_vector": {col: float(pair_probs[col]) for col in FEATURE_COLUMNS},
        "meta_class_probs": class_probs,
        "decision": decision,
        "p_pain": p_pain,
        "meta_model_classes": meta_classes,
        "meta_feature_weights": feature_weights,
        "meta_feature_weight_class": meta_classes[pain_idx] if pain_idx is not None else None,
    }
