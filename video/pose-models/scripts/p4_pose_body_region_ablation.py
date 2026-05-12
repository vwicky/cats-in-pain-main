"""
P4 pairwise models — body-region ablation on a fixed evaluation set.

For each checkpoint, runs batched inference under:
  - full body (baseline)
  - head removed (APT-36K indices 0,1,2 = L_Eye, R_Eye, Nose)
  - left front leg removed (5,6,7 = L_Shoulder, L_Elbow, L_F_Paw)
  - right front leg removed (8,9,10 = R_Shoulder, R_Elbow, R_F_Paw)
  - both front legs removed (5,6,7,8,9,10)
  - left hind leg removed (11,12,13 = L_Hip, L_Knee, L_B_Paw)
  - right hind leg removed (14,15,16 = R_Hip, R_Knee, R_B_Paw)
  - both hind legs removed (11,12,13,14,15,16)
  - all four legs / paws removed (5–16; same as both front + both hind)
  - all four legs and head removed (0–2 and 5–16)

Joint removal is applied **after** the usual ``load_and_normalize_pose`` pipeline
(same root joint and shoulder scale as training), by zeroing the selected joint
channels (including kinematics), via ``data.inference_zero_joint_indices`` in
:class:`model_training_v2.data_engineering.PoseDataset`.

Evaluation protocols (documented in ``ablation_report.txt``)
-------------------------------------------------------------
**A) Audio-10 holdout** (recommended; pass ``--audio10-manifest``):

  Built by ``src/dataset_construction/build_holdout_audio10_human_validation.py`` from
  ``human_validation_app/cat_audios_copy_processed`` (CNN14 labels, v1 poses joined).

  - Keep rows with ``audio_confidence > --min-audio10-confidence`` (default **>0.65**),
    non-null ``pose_path``, and pose file on disk.
  - Optional **training-aligned pose QC** (see ``--exclude-low-quality-pose``,
    ``--vitpose-qc-filter``): ``low_quality_pose`` from v1 join + same ViTPose gates as
    ``model_training_v2/scripts/build_vitpose_qc_enrichment.py`` (``vitpose_qc.py``).
    Holdout clips missing from the enrichment JSONL are scored after ``normalize_pose_array``
    (``--config``) so v1 pixel ``.npy`` tensors use the same scale as v2 QC in the build script.
  - For each pairwise model, restrict to rows whose ``audio_label_10`` is one of the
    two classes in ``pair_meta.json``, then **OvO** metrics: ``y_true = 1`` iff
    ``audio_label_10 == class_pos_label_int_1``, score ``= softmax[:, 1]`` (positive
    class logit). **Poses are v1** (same as manifest); models were trained on v2
    for some pairs — expect domain shift; this is still a clean **10-class-disjoint**
    clip pool vs ``final_dataset_v2.jsonl`` snippet_ids.
  - ``--audio10-preset default`` (three models) or ``paining_pairs`` (five: Paining–Resting,
    Fighting–Paining, Warning–Paining, HuntingMind–Paining, Mating–Paining on the same holdout).

**B) Legacy** (omit ``--audio10-manifest``):

  1) **Paining–Resting** and **Fighting–Paining** use ``dataset/final_dataset.jsonl``
     with **pain vs not** proxy labels (see script body). **Fighting–Resting** uses a
     v2 Fighting/Resting slice (in-distribution).

Examples
--------
  .venv/bin/python3 model_training_v2/scripts/p4_pose_body_region_ablation.py \\
    --audio10-manifest dataset_construction/manifests/holdout_audio10_20260428T205847Z/holdout_manifest_audio10.jsonl \\
    --exclude-low-quality-pose --vitpose-qc-filter \\
    --vitpose-qc-enrichment dataset_construction/reports/vitpose_qc_enrichment.jsonl \\
    --device auto --batch-size 32

  .venv/bin/python3 model_training_v2/scripts/p4_pose_body_region_ablation.py \\
    --audio10-manifest dataset_construction/manifests/holdout_audio10_20260428T205847Z/holdout_manifest_audio10.jsonl \\
    --audio10-preset paining_pairs \\
    --device auto --batch-size 32

  .venv/bin/python3 model_training_v2/scripts/p4_pose_body_region_ablation.py \\
    --device auto --batch-size 32
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# APT-36K 17-keypoint cat layout (matches ``data_engineering._FLIP_PAIRS_17``).
# Index map (APT-36K):
#   0: L_Eye, 1: R_Eye, 2: Nose, 3: Neck, 4: Root_of_tail,
#   5: L_Shoulder, 6: L_Elbow, 7: L_F_Paw,
#   8: R_Shoulder, 9: R_Elbow, 10: R_F_Paw,
#   11: L_Hip, 12: L_Knee, 13: L_B_Paw,
#   14: R_Hip, 15: R_Knee, 16: R_B_Paw.
KP_HEAD = (0, 1, 2)
KP_FRONT_LEG_L = (5, 6, 7)
KP_FRONT_LEG_R = (8, 9, 10)
KP_HIND_LEG_L = (11, 12, 13)
KP_HIND_LEG_R = (14, 15, 16)
KP_ALL_PAWS = tuple(sorted(KP_FRONT_LEG_L + KP_FRONT_LEG_R + KP_HIND_LEG_L + KP_HIND_LEG_R))
KP_ALL_PAWS_AND_HEAD = tuple(sorted(KP_HEAD + KP_ALL_PAWS))

ABLATIONS: dict[str, tuple[int, ...]] = {
    "full": (),
    "drop_head": KP_HEAD,
    "drop_front_leg_L": KP_FRONT_LEG_L,
    "drop_front_leg_R": KP_FRONT_LEG_R,
    "drop_both_front_legs": tuple(sorted(KP_FRONT_LEG_L + KP_FRONT_LEG_R)),
    "drop_hind_leg_L": KP_HIND_LEG_L,
    "drop_hind_leg_R": KP_HIND_LEG_R,
    "drop_both_hind_legs": tuple(sorted(KP_HIND_LEG_L + KP_HIND_LEG_R)),
    "drop_all_paws": KP_ALL_PAWS,
    "drop_all_paws_and_head": KP_ALL_PAWS_AND_HEAD,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_dir: Path
    eval_mode: str  # "audio10_ovo" | "pain_holdout_v1" | "fighting_restsing_ovo_v2"


def build_p4_pose_ablation_specs(*, use_audio10: bool, audio10_preset: str) -> list[ModelSpec]:
    """Resolve checkpoint directories for ablation (audio-10 vs legacy)."""
    sweep_pr = (
        REPO_ROOT
        / "video/pose-models/runs/p4_sweep_paining_restsing_20260423_235721/hparams_grid"
        / "gs0030_lr5e-05_bs8_wd0.05_g0.0_cosine/binary__Paining__Resting"
    )
    fp = REPO_ROOT / "video/pose-models/runs/p4_fighting_paining_finetune_20260425_105604/binary__Fighting__Paining"
    fr = (
        REPO_ROOT
        / "video/pose-models/runs/p4_nonpain36_grid_16_20260424_211353/hparams_grid"
        / "gs0318__binary__Fighting__Resting__lr1e-04_bs8_wd0.05_g0.0_cosine/binary__Fighting__Resting"
    )
    ft8 = REPO_ROOT / "video/pose-models/runs/p4_pain_finetune_8_20260424_192600"

    if use_audio10:
        if audio10_preset == "default":
            return [
                ModelSpec("Paining_Resting", sweep_pr, "audio10_ovo"),
                ModelSpec("Fighting_Paining", fp, "audio10_ovo"),
                ModelSpec("Fighting_Resting", fr, "audio10_ovo"),
            ]
        if audio10_preset == "paining_pairs":
            return [
                ModelSpec("Paining_Resting", sweep_pr, "audio10_ovo"),
                ModelSpec("Fighting_Paining", fp, "audio10_ovo"),
                ModelSpec("Warning_Paining", ft8 / "binary__Paining__Warning", "audio10_ovo"),
                ModelSpec("HuntingMind_Paining", ft8 / "binary__HuntingMind__Paining", "audio10_ovo"),
                ModelSpec("Mating_Paining", ft8 / "binary__Mating__Paining", "audio10_ovo"),
            ]
        raise ValueError(f"unknown audio10_preset: {audio10_preset!r}")

    return [
        ModelSpec("Paining_Resting", sweep_pr, "pain_holdout_v1"),
        ModelSpec("Fighting_Paining", fp, "pain_holdout_v1"),
        ModelSpec("Fighting_Resting", fr, "fighting_restsing_ovo_v2"),
    ]


def _setup_run_dir(prefix: str = "p4_pose_region_ablation") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = REPO_ROOT / "video" / "pose-models" / "runs" / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_pair_meta(model_dir: Path) -> dict[str, Any]:
    p = model_dir / "pair_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"pair_meta.json not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_model_p4(model_dir: Path, cfg: dict, device) -> Any:
    import torch
    from models.p4_pose_stgcn import  P4PoseSTGCN

    wpath = model_dir / "training" / "best_weights.pth"
    if not wpath.is_file():
        raise FileNotFoundError(f"best_weights.pth not found: {wpath}")

    nc = int(cfg["data"]["n_channels"]) * 2
    model = P4PoseSTGCN(
        n_frames=int(cfg["data"]["n_frames"]),
        n_keypoints=int(cfg["data"]["n_keypoints"]),
        n_channels=nc,
        n_classes=2,
    ).to(device)
    blob = torch.load(wpath, map_location=device)
    sd = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict):
        raise TypeError(f"Invalid checkpoint at {wpath}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _pain_logit_index(meta: dict[str, Any]) -> int:
    """Index into logits_binary such that softmax[:, i] is P(that class)."""
    if str(meta.get("class_pos_label_int_1", "")) == "Paining":
        return 1
    if str(meta.get("class_neg_label_int_0", "")) == "Paining":
        return 0
    raise ValueError(f"Paining not found in pair_meta classes: {meta}")


def _positive_class_logit_index(meta: dict[str, Any]) -> int:
    """pairwise training uses label 1 = class_pos_label_int_1 (see run_p4_pairwise)."""
    return 1


def _load_audio10_holdout_df(
    manifest: Path,
    *,
    min_confidence: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Rows from holdout_manifest_audio10.jsonl with confident pseudo-labels and usable v1 pose."""
    rows: list[dict[str, Any]] = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df.loc[df["audio_confidence"].astype(float) > float(min_confidence)].copy()
    n1 = len(df)
    df = df.loc[df["pose_path"].notna() & (df["pose_path"].astype(str).str.len() > 0)].copy()
    n2 = len(df)
    kept: list[bool] = []
    for _, r in df.iterrows():
        pp = REPO_ROOT / str(r["pose_path"]).replace("\\", "/")
        kept.append(pp.is_file())
    df = df.loc[kept].copy()
    n3 = len(df)
    logger.info(
        "Audio-10 holdout %s: rows %d -> conf>%.3f -> %d -> with pose_path -> %d -> file exists -> %d",
        manifest,
        n0,
        min_confidence,
        n1,
        n2,
        n3,
    )
    if df.empty:
        raise SystemExit("No rows left after audio-10 holdout filters — check manifest and confidence threshold.")
    return df


def _vitpose_qc_ok_from_pose_npy(
    pose_path: Path,
    *,
    cfg: dict,
    min_mean_conf: float,
    max_y_range: float,
    visible_thr: float,
) -> tuple[bool, list[str]]:
    """Same gates as ``build_vitpose_qc_enrichment.py`` / training ``data.vitpose_qc``.

    Raw v1 ``.npy`` clips are in pixel space (``y_range`` hundreds+); enrichment was built
    on v2 tensors already in model scale. We therefore run ``normalize_pose_array`` first
    so holdout v1 poses are judged in the same coordinate regime as training QC.
    """
    from data_engineering import  normalize_pose_array
    from vitpose_qc import  apply_vitpose_qc_gates, compute_vitpose_clip_metrics

    if not pose_path.is_file():
        return False, ["pose_file_missing"]
    pose_raw = np.load(pose_path)
    if pose_raw.ndim != 3 or pose_raw.shape[-1] < 3:
        return False, ["pose_shape_invalid"]
    pose = normalize_pose_array(pose_raw, cfg, drop_low_confidence_frames=False)
    mask = np.ones(pose.shape[0], dtype=bool)
    mets = compute_vitpose_clip_metrics(pose, mask, visible_thr=float(visible_thr))
    if mets is None:
        return False, ["pose_metrics_none"]
    ok, reasons = apply_vitpose_qc_gates(
        mets, min_mean_conf=float(min_mean_conf), max_y_range=float(max_y_range)
    )
    return ok, list(reasons)


def _audio10_apply_pose_pipeline_filters(
    df: pd.DataFrame,
    *,
    logger: logging.Logger,
    pose_qc_cfg: dict,
    exclude_low_quality_pose: bool,
    vitpose_qc_filter: bool,
    vitpose_qc_enrichment: Path | None,
    min_mean_conf: float,
    max_y_range: float,
    visible_thr: float,
) -> pd.DataFrame:
    """
    Optional alignment with P4 training data cleaning:

    - ``exclude_low_quality_pose``: same keep rule as ``train_p4_pair_stack_logreg`` —
      keep row iff ``not row.get("low_quality_pose", True)`` (missing key → dropped).
    - ``vitpose_qc_filter``: drop clips failing ``mean_conf`` / ``y_range`` gates. If
      ``vitpose_qc_enrichment`` is set, reuse ``vitpose_qc_ok`` for ``snippet_id`` when
      present; otherwise load the pose ``.npy``, run ``normalize_pose_array`` (same
      ``--config`` as inference), then score — so v1 pixel tensors match the scale used
      when ``build_vitpose_qc_enrichment.py`` scored v2 files.
    """
    from vitpose_qc import  load_vitpose_enrichment

    out = df
    n0 = len(out)

    if exclude_low_quality_pose:
        if "low_quality_pose" not in out.columns:
            logger.warning(
                "exclude_low_quality_pose: column low_quality_pose missing — no rows dropped."
            )
        else:
            def _keep_lq(r: pd.Series) -> bool:
                return not bool(r.get("low_quality_pose", True))

            m = out.apply(_keep_lq, axis=1)
            n_drop = int((~m).sum())
            out = out.loc[m].copy()
            logger.info(
                "low_quality_pose filter: kept %d / %d (dropped %d; missing key counts as low quality, "
                "same as train_p4_pair_stack_logreg)",
                len(out),
                n0,
                n_drop,
            )
            n0 = len(out)

    if vitpose_qc_filter:
        enr: dict[str, dict[str, Any]] = {}
        if vitpose_qc_enrichment is not None:
            ep = vitpose_qc_enrichment if vitpose_qc_enrichment.is_absolute() else REPO_ROOT / vitpose_qc_enrichment
            enr = load_vitpose_enrichment(ep)
            logger.info("ViTPose QC: loaded enrichment %s (%d snippet_ids)", ep, len(enr))

        kept_idx: list[int] = []
        n_cached_ok = n_cached_bad = n_scored_ok = n_scored_bad = 0
        for i, r in out.iterrows():
            sid = str(r.get("snippet_id", "")).strip()
            pose_rel = str(r.get("pose_path", "")).replace("\\", "/")
            ppose = REPO_ROOT / pose_rel if pose_rel else Path()

            if enr and sid in enr:
                ok = bool(enr[sid].get("vitpose_qc_ok", True))
                if ok:
                    n_cached_ok += 1
                else:
                    n_cached_bad += 1
                if ok:
                    kept_idx.append(i)
                continue

            ok, _reasons = _vitpose_qc_ok_from_pose_npy(
                ppose,
                cfg=pose_qc_cfg,
                min_mean_conf=min_mean_conf,
                max_y_range=max_y_range,
                visible_thr=visible_thr,
            )
            if ok:
                n_scored_ok += 1
                kept_idx.append(i)
            else:
                n_scored_bad += 1

        out = out.loc[kept_idx].copy()
        logger.info(
            "ViTPose QC gates (mean_conf>=%.3f, y_range<%g): kept %d / %d — "
            "enrichment_ok=%d enrichment_fail=%d disk_ok=%d disk_fail=%d",
            min_mean_conf,
            max_y_range,
            len(out),
            n0,
            n_cached_ok,
            n_cached_bad,
            n_scored_ok,
            n_scored_bad,
        )
        n0 = len(out)

    if out.empty:
        raise SystemExit("No rows left after pose pipeline filters — relax flags or check manifest.")
    return out


def _subset_audio10_for_pair(
    df: pd.DataFrame,
    meta: dict[str, Any],
    *,
    model_key: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Restrict to OvO rows; y_true=1 iff audio_label_10 == positive class (index 1 in model)."""
    neg = str(meta["class_neg_label_int_0"])
    pos = str(meta["class_pos_label_int_1"])
    lab_col = df["audio_label_10"].astype(str)
    sub = df.loc[lab_col.isin([neg, pos])].copy()
    if sub.empty:
        raise SystemExit(f"{model_key}: no rows with audio_label_10 in {{{neg}, {pos}}}.")
    y = (sub["audio_label_10"].astype(str) == pos).astype(np.int64).values
    li = _positive_class_logit_index(meta)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "%s: single-class OvO subset (n_pos=%d n_neg=%d) — metrics will be degenerate.",
            model_key,
            n_pos,
            n_neg,
        )
    logger.info(
        "%s OvO on audio-10 holdout: n=%d (%s=%d, %s=%d)",
        model_key,
        len(sub),
        pos,
        n_pos,
        neg,
        n_neg,
    )
    sub["_y_true"] = y
    return sub, y, li


def _pose_records_from_audio10_subset(sub: pd.DataFrame) -> list[dict]:
    """PoseDataset rows: v1 pose tensors; labels unused at inference but set for debugging."""
    recs: list[dict] = []
    for _, r in sub.iterrows():
        yb = int(r["_y_true"])
        recs.append({
            "snippet_id": str(r["snippet_id"]),
            "pose_path": str(r["pose_path"]).replace("\\", "/"),
            "pose_mask_path": None,
            "pose_version": "v1",
            "label_int": yb,
            "binary_label_int": yb,
            "cat_id": str(r.get("video_id", r.get("snippet_id", ""))),
        })
    return recs


def _run_inference_ablation(
    model: Any,
    records: list[dict],
    cfg: dict,
    device,
    *,
    joint_drop: tuple[int, ...],
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from data_engineering import  PoseDataset

    infer_cfg = copy.deepcopy(cfg)
    infer_cfg.setdefault("training", {})["binary_only"] = True
    infer_cfg.setdefault("data", {})["pose_cache"] = {"backend": "off"}
    ddata = infer_cfg.setdefault("data", {})
    if joint_drop:
        ddata["inference_zero_joint_indices"] = [int(j) for j in joint_drop]
    else:
        ddata.pop("inference_zero_joint_indices", None)

    ds = PoseDataset(records, infer_cfg, is_train=False, use_kinematics=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    snippet_ids: list[str] = []
    prob_rows: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            pose = batch["pose"].to(device)
            mask = batch["mask"].to(device)
            out = model(pose, mask)
            pr = F.softmax(out["logits_binary"], dim=1).cpu().numpy().astype(np.float32)
            snippet_ids.extend(batch["snippet_id"])
            prob_rows.append(pr)

    probs2 = np.concatenate(prob_rows, axis=0)
    return snippet_ids, probs2


def _metrics_block(
    y_true: np.ndarray,
    y_prob_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob_score >= threshold).astype(np.int64)
    out: dict[str, float] = {
        "n": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_pain": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "precision_pain": float(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "recall_pain": float(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
    }
    try:
        if len(np.unique(y_true)) > 1:
            out["auc_roc"] = float(roc_auc_score(y_true, y_prob_score))
        else:
            out["auc_roc"] = float("nan")
    except ValueError:
        out["auc_roc"] = float("nan")
    return out


def _load_holdout_records(
    manifest: Path,
    *,
    audio_filter: bool,
    min_audio_confidence: float,
    logger: logging.Logger,
) -> tuple[list[dict], pd.DataFrame]:
    from scripts.train_p4_pair_stack_logreg import  load_labeled_holdout

    df = load_labeled_holdout(
        manifest,
        audio_filter=audio_filter,
        min_audio_confidence=min_audio_confidence,
        logger=logger,
    )
    records = df.to_dict("records")
    return records, df


def _load_fighting_resting_v2(
    cfg_path: Path,
    model_dir: Path,
    logger: logging.Logger,
) -> tuple[list[dict], pd.DataFrame, dict[str, str]]:
    from data_loading import  load_dataset

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    lg = logging.getLogger("data_loading_ablation")
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())
    df = load_dataset(cfg, lg)
    lf = str(cfg["labels"]["label_field"])
    sub = df[df[lf].isin(["Fighting", "Resting"])].copy()
    if sub.empty:
        raise SystemExit("No Fighting/Resting rows after load_dataset — cannot evaluate FR model.")
    logger.info(
        "Fighting–Resting v2 subset: %d rows (Fighting=%d Resting=%d)",
        len(sub),
        int((sub[lf] == "Fighting").sum()),
        int((sub[lf] == "Resting").sum()),
    )

    meta_fr = _load_pair_meta(model_dir)
    pos = str(meta_fr["class_pos_label_int_1"])
    neg = str(meta_fr["class_neg_label_int_0"])

    records: list[dict] = []
    for _, r in sub.iterrows():
        pm = r.get("pose_mask_path")
        records.append({
            "snippet_id": str(r["snippet_id"]),
            "pose_path": str(r["pose_path"]).replace("\\", "/"),
            "pose_mask_path": str(pm).replace("\\", "/") if pm and str(pm).strip() else None,
            "pose_version": str(r.get("pose_version", "v2")),
            "label_int": int(r["label_int"]),
            "binary_label_int": int(r["binary_label_int"]),
            "cat_id": str(r.get("cat_id", "")),
        })
    return records, sub, {"positive_class": pos, "negative_class": neg, "label_field": lf}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P4 pose body-region ablation (pairwise ST-GCN × joint-drop conditions)."
    )
    parser.add_argument("--config", default="video/pose-models/config_p4_pain_finetune.yaml",
                        help="Base YAML (architecture + normalization).")
    parser.add_argument(
        "--audio10-manifest",
        default=None,
        help="If set, evaluate on this holdout (CNN14 audio_label_10 + v1 pose); "
             "filter with --min-audio10-confidence. Each model uses OvO on pseudo-labels.",
    )
    parser.add_argument(
        "--audio10-preset",
        default="default",
        choices=("default", "paining_pairs"),
        help="Audio-10 only: default = Paining–Resting + Fighting–Paining + Fighting–Resting; "
             "paining_pairs = those first two plus Warning–Paining, HuntingMind–Paining, "
             "Mating–Paining (p4_pain_finetune_8 checkpoints).",
    )
    parser.add_argument(
        "--min-audio10-confidence",
        type=float,
        default=0.65,
        help="Keep rows with audio_confidence **strictly greater than** this (default: >0.65).",
    )
    parser.add_argument(
        "--exclude-low-quality-pose",
        action="store_true",
        help="Audio-10 mode only: drop rows with low_quality_pose=True or missing key "
             "(same rule as train_p4_pair_stack_logreg: not row.get(\"low_quality_pose\", True)).",
    )
    parser.add_argument(
        "--vitpose-qc-filter",
        action="store_true",
        help="Audio-10 mode only: drop clips failing ViTPose QC gates (mean_conf, y_range; "
             "see model_training_v2/vitpose_qc.py). Cached rows: --vitpose-qc-enrichment. "
             "Otherwise each .npy is scored after normalize_pose_array (--config) so v1 pixel "
             "poses match training-scale QC.",
    )
    parser.add_argument(
        "--vitpose-qc-enrichment",
        type=str,
        default=None,
        help="Optional JSONL from build_vitpose_qc_enrichment.py (repo-relative or absolute). "
             "Only used with --vitpose-qc-filter to avoid re-reading poses already in the file.",
    )
    parser.add_argument("--vitpose-qc-min-mean-conf", type=float, default=0.30)
    parser.add_argument("--vitpose-qc-max-y-range", type=float, default=10.0)
    parser.add_argument("--vitpose-qc-visible-thr", type=float, default=0.30)
    parser.add_argument("--hold-out-manifest", default="data/dataset/final_dataset.jsonl",
                        help="Legacy v1 5-way holdout (ignored when --audio10-manifest is set).")
    parser.add_argument("--v2-config", default="video/pose-models/config_p4_10class.yaml",
                        help="Config for Fighting–Resting v2 slice (legacy mode only).")
    parser.add_argument("--audio-confidence-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-audio-confidence", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary predictions (score = P(positive class)).")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows per eval set (debug).")
    parser.add_argument("--dry-run", action="store_true", help="Load data + models, skip torch inference.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("p4_pose_ablation")

    if (
        args.exclude_low_quality_pose
        or args.vitpose_qc_filter
        or args.vitpose_qc_enrichment
    ) and not args.audio10_manifest:
        raise SystemExit(
            "Pose pipeline flags (--exclude-low-quality-pose, --vitpose-qc-filter, "
            "--vitpose-qc-enrichment) require --audio10-manifest."
        )
    if args.vitpose_qc_enrichment and not args.vitpose_qc_filter:
        logger.warning("--vitpose-qc-enrichment has no effect without --vitpose-qc-filter")

    run_dir = _setup_run_dir()
    logger.info("run_dir=%s", run_dir)

    cfg_path = REPO_ROOT / args.config
    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    use_audio10 = bool(args.audio10_manifest)
    if not use_audio10 and args.audio10_preset != "default":
        raise SystemExit("--audio10-preset other than 'default' requires --audio10-manifest.")

    specs = build_p4_pose_ablation_specs(use_audio10=use_audio10, audio10_preset=args.audio10_preset)
    logger.info("Ablation models (%d): %s", len(specs), ", ".join(s.key for s in specs))

    import torch

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    eval_pack: dict[str, dict[str, Any]] = {}
    audio10_df: pd.DataFrame | None = None

    if use_audio10:
        a10_path = REPO_ROOT / str(args.audio10_manifest)
        if not a10_path.is_file():
            raise SystemExit(f"--audio10-manifest not found: {a10_path}")
        audio10_df = _load_audio10_holdout_df(
            a10_path,
            min_confidence=float(args.min_audio10_confidence),
            logger=logger,
        )
        audio10_df = _audio10_apply_pose_pipeline_filters(
            audio10_df,
            logger=logger,
            pose_qc_cfg=base_cfg,
            exclude_low_quality_pose=bool(args.exclude_low_quality_pose),
            vitpose_qc_filter=bool(args.vitpose_qc_filter),
            vitpose_qc_enrichment=(
                Path(args.vitpose_qc_enrichment)
                if args.vitpose_qc_enrichment
                else None
            ),
            min_mean_conf=float(args.vitpose_qc_min_mean_conf),
            max_y_range=float(args.vitpose_qc_max_y_range),
            visible_thr=float(args.vitpose_qc_visible_thr),
        )
        if args.limit is not None:
            audio10_df = audio10_df.iloc[: args.limit].copy()
            logger.info("--limit: using first %d rows of filtered audio-10 holdout", len(audio10_df))

        for sp in specs:
            meta = _load_pair_meta(sp.model_dir)
            sub, y_true, li = _subset_audio10_for_pair(
                audio10_df, meta, model_key=sp.key, logger=logger
            )
            records = _pose_records_from_audio10_subset(sub)
            eval_pack[sp.key] = {
                "records": records,
                "y_true": y_true,
                "logit_index": int(li),
                "meta": meta,
                "n_rows": len(records),
                "eval_subset_csv": sub[
                    ["snippet_id", "audio_label_10", "audio_confidence", "pose_path", "_y_true"]
                ].rename(columns={"_y_true": "y_true_positive_is_class_pos"}),
            }
    else:
        hold_path = REPO_ROOT / args.hold_out_manifest
        hold_records, hold_df = _load_holdout_records(
            hold_path,
            audio_filter=args.audio_confidence_filter,
            min_audio_confidence=args.min_audio_confidence,
            logger=logger,
        )
        if args.limit is not None:
            hold_df = hold_df.iloc[: args.limit].copy()
            hold_records = hold_df.to_dict("records")

        fr_records, fr_df, fr_lab = _load_fighting_resting_v2(
            REPO_ROOT / args.v2_config,
            specs[2].model_dir,
            logger,
        )
        if args.limit is not None:
            fr_df = fr_df.iloc[: args.limit].copy()
            fr_records = []
            for _, r in fr_df.iterrows():
                pm = r.get("pose_mask_path")
                fr_records.append({
                    "snippet_id": str(r["snippet_id"]),
                    "pose_path": str(r["pose_path"]).replace("\\", "/"),
                    "pose_mask_path": str(pm).replace("\\", "/") if pm and str(pm).strip() else None,
                    "pose_version": str(r.get("pose_version", "v2")),
                    "label_int": int(r["label_int"]),
                    "binary_label_int": int(r["binary_label_int"]),
                    "cat_id": str(r.get("cat_id", "")),
                })

        for sp in specs:
            meta = _load_pair_meta(sp.model_dir)
            if sp.eval_mode == "pain_holdout_v1":
                li = _pain_logit_index(meta)
                y_true = hold_df["binary_label_int"].values.astype(np.int64)
                eval_pack[sp.key] = {
                    "records": hold_records,
                    "y_true": y_true,
                    "logit_index": li,
                    "meta": meta,
                    "n_rows": len(hold_records),
                }
            else:
                lf = fr_lab["label_field"]
                pos = str(meta["class_pos_label_int_1"])
                y_true = (fr_df[lf] == pos).astype(np.int64).values
                eval_pack[sp.key] = {
                    "records": fr_records,
                    "y_true": y_true,
                    "logit_index": 1,
                    "meta": meta,
                    "n_rows": len(fr_records),
                    "fr_pos_class": fr_lab,
                }

    if use_audio10:
        protocol = (
            "Audio-10 human-validation holdout: y_true from CNN14 pseudo-labels (audio_label_10) "
            "restricted to each model's OvO pair; score = P(class_pos_label_int_1). "
            f"Rows require audio_confidence > {args.min_audio10_confidence} and existing v1 pose_path. "
            "Clip IDs are disjoint from final_dataset_v2 snippet_id set (see holdout build script)."
        )
        if args.exclude_low_quality_pose:
            protocol += " Applied low_quality_pose exclusion (train_p4_pair_stack_logreg rule)."
        if args.vitpose_qc_filter:
            protocol += (
                f" Applied ViTPose QC gates mean_conf>={args.vitpose_qc_min_mean_conf}, "
                f"y_range<{args.vitpose_qc_max_y_range} (enrichment cache: "
                f"{args.vitpose_qc_enrichment or 'none — scored every pose from disk'})."
            )
        meta_out = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": str(cfg_path),
            "eval_mode": "audio10_holdout_ovo",
            "audio10_manifest": str(REPO_ROOT / args.audio10_manifest),
            "audio10_preset": str(args.audio10_preset),
            "min_audio10_confidence": float(args.min_audio10_confidence),
            "audio10_pose_filters": {
                "exclude_low_quality_pose": bool(args.exclude_low_quality_pose),
                "vitpose_qc_filter": bool(args.vitpose_qc_filter),
                "vitpose_qc_enrichment": args.vitpose_qc_enrichment,
                "vitpose_qc_min_mean_conf": float(args.vitpose_qc_min_mean_conf),
                "vitpose_qc_max_y_range": float(args.vitpose_qc_max_y_range),
                "vitpose_qc_visible_thr": float(args.vitpose_qc_visible_thr),
            },
            "joint_groups": {k: list(v) for k, v in ABLATIONS.items()},
            "keypoint_legend": (
                "APT-36K 17-kp cat: 0 L_Eye, 1 R_Eye, 2 Nose, 3 Neck, 4 Root_of_tail, "
                "5 L_Shoulder, 6 L_Elbow, 7 L_F_Paw, 8 R_Shoulder, 9 R_Elbow, 10 R_F_Paw, "
                "11 L_Hip, 12 L_Knee, 13 L_B_Paw, 14 R_Hip, 15 R_Knee, 16 R_B_Paw"
            ),
            "models": [
                {
                    "key": sp.key,
                    "model_dir": str(sp.model_dir),
                    "eval_mode": sp.eval_mode,
                    "pair_meta": _load_pair_meta(sp.model_dir),
                    "n_eval_rows": eval_pack[sp.key]["n_rows"],
                }
                for sp in specs
            ],
            "protocol_notes": protocol,
        }
    else:
        hold_path = REPO_ROOT / args.hold_out_manifest
        protocol = (
            "Paining_Resting and Fighting_Paining: y_true = (label==Paining) on final_dataset.jsonl; "
            "score = P(Paining). Fighting_Resting: v2 OvO slice (in-distribution)."
        )
        meta_out = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": str(cfg_path),
            "eval_mode": "legacy",
            "hold_out_manifest": str(hold_path),
            "v2_config": str(REPO_ROOT / args.v2_config),
            "joint_groups": {k: list(v) for k, v in ABLATIONS.items()},
            "keypoint_legend": (
                "APT-36K 17-kp cat: 0 L_Eye, 1 R_Eye, 2 Nose, 3 Neck, 4 Root_of_tail, "
                "5 L_Shoulder, 6 L_Elbow, 7 L_F_Paw, 8 R_Shoulder, 9 R_Elbow, 10 R_F_Paw, "
                "11 L_Hip, 12 L_Knee, 13 L_B_Paw, 14 R_Hip, 15 R_Knee, 16 R_B_Paw"
            ),
            "models": [
                {
                    "key": sp.key,
                    "model_dir": str(sp.model_dir),
                    "eval_mode": sp.eval_mode,
                    "pair_meta": _load_pair_meta(sp.model_dir),
                }
                for sp in specs
            ],
            "protocol_notes": protocol,
        }

    (run_dir / "ablation_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    if use_audio10:
        for sp in specs:
            ep = eval_pack[sp.key]
            if "eval_subset_csv" in ep:
                ep["eval_subset_csv"].to_csv(run_dir / f"eval_subset_{sp.key}.csv", index=False)

    rows_out: list[dict[str, Any]] = []
    all_probs_npz: dict[str, Any] = {}

    if args.dry_run:
        logger.info("Dry run: would evaluate %d models × %d ablations", len(specs), len(ABLATIONS))
        (run_dir / "ablation_report.txt").write_text("Dry run — no inference.\n", encoding="utf-8")
        return

    for sp in specs:
        pack = eval_pack[sp.key]
        records = pack["records"]
        y_true = pack["y_true"]
        li = int(pack["logit_index"])
        sid_order = [str(r["snippet_id"]) for r in records]

        model = load_model_p4(sp.model_dir, base_cfg, device)
        for abl_name, joints in ABLATIONS.items():
            t0 = time.time()
            sids, probs2 = _run_inference_ablation(
                model, records, base_cfg, device, joint_drop=tuple(joints), batch_size=args.batch_size,
            )
            if list(sids) != sid_order:
                raise RuntimeError(f"snippet_id order mismatch [{sp.key} / {abl_name}]")
            score = probs2[:, li].astype(np.float32)
            m = _metrics_block(y_true, score, threshold=args.threshold)
            row = {
                "model_key": sp.key,
                "ablation": abl_name,
                "joints_zeroed": ";".join(map(str, joints)) if joints else "",
                "threshold": args.threshold,
                **{k: round(v, 6) if isinstance(v, float) and np.isfinite(v) else v for k, v in m.items()},
                "seconds": round(time.time() - t0, 2),
            }
            rows_out.append(row)
            key = f"{sp.key}__{abl_name}"
            all_probs_npz[key + "_snippet_id"] = np.array(sids, dtype=object)
            all_probs_npz[key + "_y_true"] = y_true.astype(np.int64)
            all_probs_npz[key + "_score"] = score
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dfm = pd.DataFrame(rows_out)
    dfm.to_csv(run_dir / "metrics_by_model_ablation.csv", index=False)

    # Delta vs full for key metrics
    deltas: list[dict[str, Any]] = []
    for spk in dfm["model_key"].unique():
        sub = dfm[dfm["model_key"] == spk].set_index("ablation")
        if "full" not in sub.index:
            continue
        base = sub.loc["full"]
        for abl in sub.index:
            if abl == "full":
                continue
            r = sub.loc[abl]
            deltas.append({
                "model_key": spk,
                "ablation": abl,
                "delta_macro_f1": round(float(r["macro_f1"] - base["macro_f1"]), 6),
                "delta_macro_precision": round(float(r["macro_precision"] - base["macro_precision"]), 6),
                "delta_macro_recall": round(float(r["macro_recall"] - base["macro_recall"]), 6),
                "delta_f1_pain": round(float(r["f1_pain"] - base["f1_pain"]), 6),
                "delta_precision_pain": round(float(r["precision_pain"] - base["precision_pain"]), 6),
                "delta_recall_pain": round(float(r["recall_pain"] - base["recall_pain"]), 6),
            })
    pd.DataFrame(deltas).to_csv(run_dir / "metric_deltas_vs_full.csv", index=False)

    np.savez_compressed(run_dir / "raw_scores.npz", **all_probs_npz)

    # Wide pivot table for the paper-style view
    pivot_f1 = dfm.pivot(index="model_key", columns="ablation", values="macro_f1")
    pivot_f1.to_csv(run_dir / "pivot_macro_f1.csv")
    pivot_pr = dfm.pivot(index="model_key", columns="ablation", values="macro_precision")
    pivot_pr.to_csv(run_dir / "pivot_macro_precision.csv")
    pivot_rc = dfm.pivot(index="model_key", columns="ablation", values="macro_recall")
    pivot_rc.to_csv(run_dir / "pivot_macro_recall.csv")

    report_lines = [
        "P4 pose body-region ablation",
        "===========================",
        "",
        str(meta_out["protocol_notes"]),
        "",
        "Per-model × ablation (macro_* = sklearn macro average over binary classes; ",
        " f1_pain / precision_pain / recall_pain = **positive OvO class** (pair_meta class_pos_label_int_1)):",
        "",
        dfm.to_string(index=False),
        "",
        "Deltas vs full:",
        "",
        pd.DataFrame(deltas).to_string(index=False) if deltas else "(none)",
        "",
        f"Artifacts: {run_dir}",
    ]
    (run_dir / "ablation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Wrote %s", run_dir)


if __name__ == "__main__":
    main()
