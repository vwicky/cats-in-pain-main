"""
Video branch of the inference pipeline.

Steps
-----
1. Extract poses frame-by-frame via VitInference (ViTPose + YOLO8x), simultaneously
   writing a pose-overlay video for visualisation.
2. Save the raw (T, 17, 3) pose array as a .npy file.
3. Run the 9 pairwise P4PoseSTGCN models to get P(Paining) per pair.
4. Feed the 9-dimensional feature vector into the LogReg L2 meta-model.
5. Return final pain/non-pain decision with all intermediate probabilities.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

# Ensure src/ is on sys.path so absolute inference.* imports work whether this
# module is imported as a package or loaded transitively from the script.
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from inference.pose_assembler import (  # noqa: E402
    DEFAULT_VITPOSE_ARCH,
    DEFAULT_VITPOSE_DATASET,
    DEFAULT_VITPOSE_MODEL,
    DEFAULT_YOLO,
    build_pose_record,
    extract_poses_and_render,
)
from inference.stgcn_loader import (  # noqa: E402
    DEFAULT_STACK_RUN,
    load_meta_model,
    load_pairwise_models,
    run_meta_inference,
    run_pairwise_inference,
)

logger = logging.getLogger(__name__)


def run_video_branch(
    video_path: str | Path,
    run_dir: Path,
    device: torch.device,
    timer: Any,
    *,
    vitpose_model: str = DEFAULT_VITPOSE_MODEL,
    vitpose_dataset: str = DEFAULT_VITPOSE_DATASET,
    vitpose_arch: str = DEFAULT_VITPOSE_ARCH,
    yolo_model: str = DEFAULT_YOLO,
    stack_run_dir: str | Path = DEFAULT_STACK_RUN,
    pair_models: dict[str, Any] | None = None,
    meta_model: Any | None = None,
    yolo_size: int = 320,
    single_pose: bool = True,
    conf_threshold: float = 0.35,
) -> dict[str, Any]:
    """
    Full video branch: pose extraction + pose video + pairwise STGCN + meta-learner.

    Parameters
    ----------
    pair_models:
        Pre-loaded dict from ``load_pairwise_models`` (avoids reloading across calls).
        Loaded here if None.
    meta_model:
        Pre-loaded sklearn LogisticRegression from ``load_meta_model``.
        Loaded here if None.

    Returns
    -------
    dict with keys: branch, pose_video, pose_npy, pose_meta,
                    pairwise_probs, meta_result.
    """
    video_path = Path(video_path)

    # ── Load models if not pre-loaded ────────────────────────────────────────
    if pair_models is None:
        with timer.step("pairwise_models_load"):
            pair_models = load_pairwise_models(stack_run_dir, device=device)

    if meta_model is None:
        with timer.step("meta_model_load"):
            meta_model = load_meta_model(stack_run_dir)

    # ── Pose extraction + overlay video ──────────────────────────────────────
    pose_npy_path = run_dir / "raw_poses.npy"
    pose_video_path = run_dir / "pose_video.mp4"

    with timer.step("pose_extraction_and_render"):
        _, _, pose_meta = extract_poses_and_render(
            video_path,
            pose_npy_path,
            pose_video_path,
            vitpose_model=vitpose_model,
            yolo_model=yolo_model,
            dataset=vitpose_dataset,
            model_name=vitpose_arch,
            yolo_size=yolo_size,
            single_pose=single_pose,
            conf_threshold=conf_threshold,
        )

    mask_p = pose_meta.get("pose_mask_path")
    gpt_res = f"{pose_meta['frame_width']}x{pose_meta['frame_height']}"
    record = build_pose_record(
        pose_npy_path,
        snippet_id=video_path.stem,
        pose_mask_path=mask_p,
        gpt_resolution=gpt_res,
    )

    with timer.step("pairwise_stgcn_inference"):
        pair_probs = run_pairwise_inference(pair_models, record, device)

    logger.info("Pairwise P(pain): %s", {k: f"{v:.3f}" for k, v in pair_probs.items()})

    # ── Meta-learner ──────────────────────────────────────────────────────────
    with timer.step("meta_learner_inference"):
        meta_result = run_meta_inference(meta_model, pair_probs)

    logger.info(
        "Meta-model decision: %s  P(pain)=%.3f",
        meta_result["decision"],
        meta_result["p_pain"],
    )

    return {
        "branch": "video",
        "pose_video": str(pose_video_path),
        "pose_npy": str(pose_npy_path),
        "pose_mask_path": pose_meta.get("pose_mask_path", ""),
        "pose_meta": pose_meta,
        "pairwise_probs": pair_probs,
        "meta_result": meta_result,
    }
