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
import os
import sys
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
    REPO_ROOT,
    build_pose_record,
    detect_cats_and_bboxes,
    extract_cat_poses_cropped,
    extract_poses_and_render,
    _bootstrap_vitpose,
)
from inference.stgcn_loader import (  # noqa: E402
    DEFAULT_STACK_RUN,
    load_meta_model,
    load_pairwise_models,
    run_meta_inference,
    run_pairwise_inference,
)

logger = logging.getLogger(__name__)

_VITPOSE_ROOT = REPO_ROOT / "video" / "easyViTPose"


def _video_probe_basic(video_path: Path) -> tuple[int, int, int, float]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0, 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, fr = cap.read()
    cap.release()
    if not ret or fr is None:
        return 0, 0, n, fps
    h, w = fr.shape[:2]
    return w, h, n, fps


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
        timer=timer,
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


def run_video_branch_multicat(
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
    conf_threshold: float = 0.35,
    min_track_coverage: float = 0.15,
    max_cats: int = 8,
    window_index: int | None = None,
) -> dict[str, Any]:
    """Two-pass multicat: YOLO+SORT tracks, then shared ViTPose on crops per track."""
    video_path = Path(video_path)
    run_dir = Path(run_dir)

    if pair_models is None:
        with timer.step("pairwise_models_load"):
            pair_models = load_pairwise_models(stack_run_dir, device=device)

    if meta_model is None:
        with timer.step("meta_model_load"):
            meta_model = load_meta_model(stack_run_dir)

    vit_device = (
        "cuda"
        if device.type == "cuda"
        else "mps"
        if device.type == "mps"
        else "cpu"
    )

    with timer.step("multicat_pass1_detect"):
        kept_tracks, multicat_diag = detect_cats_and_bboxes(
            video_path,
            yolo_model=yolo_model,
            vitpose_dataset=vitpose_dataset,
            det_class=None,
            yolo_size=yolo_size,
            conf_threshold=conf_threshold,
            min_track_coverage=min_track_coverage,
            max_cats=max_cats,
            device=device,
        )

    n_cats = len(kept_tracks)
    logger.info("[multicat] %d cat(s) will be scored", n_cats)

    w0, h0, nframes, fps0 = _video_probe_basic(video_path)
    pose_meta_branch: dict[str, Any] = {
        "frame_width": w0,
        "frame_height": h0,
        "n_frames_source_video": nframes,
        "source_video_fps": fps0,
    }

    cats_out: list[dict[str, Any]] = []

    if n_cats == 0:
        multicat_diag.setdefault("status", "no_cats")
    else:
        multicat_diag["status"] = "ok"
        _bootstrap_vitpose()
        from easy_ViTPose import VitInference  # noqa: E402
        from inference.artifact_io import save_json  # noqa: E402

        _orig_cwd = os.getcwd()
        try:
            os.chdir(_VITPOSE_ROOT)
            vit_model = VitInference(
                str(vitpose_model),
                str(yolo_model),
                model_name=vitpose_arch,
                dataset=vitpose_dataset,
                det_class=None,
                yolo_size=yolo_size,
                is_video=False,
                single_pose=True,
                yolo_step=1,
                device=vit_device,
            )
            vit_model.reset()
        finally:
            os.chdir(_orig_cwd)

        gpt_res = f"{w0}x{h0}"
        for tr in kept_tracks:
            cat_dir = run_dir / "cats" / str(tr.track_id)
            tid = tr.track_id
            with timer.step(f"multicat_pass2_track_{tid}"):
                pose_npy_path, pose_video_path, pose_mask_path, n_det, pose_meta = (
                    extract_cat_poses_cropped(
                        video_path,
                        tr,
                        cat_dir,
                        vit_model,
                        conf_threshold=conf_threshold,
                    )
                )
            pose_meta_branch.update(
                {
                    k: v
                    for k, v in pose_meta.items()
                    if k
                    in (
                        "training_pose_sampling",
                        "pose_npy_shape",
                    )
                }
            )

            mask_p = str(pose_mask_path.resolve())
            record = build_pose_record(
                pose_npy_path,
                snippet_id=f"{video_path.stem}_track_{tid}",
                pose_mask_path=mask_p,
                gpt_resolution=gpt_res,
            )

            with timer.step(f"pairwise_stgcn_track_{tid}"):
                pair_probs = run_pairwise_inference(pair_models, record, device)

            with timer.step(f"meta_learner_track_{tid}"):
                meta_result = run_meta_inference(meta_model, pair_probs)

            cat_row: dict[str, Any] = {
                "branch": "video",
                "local_track_id": int(tid),
                "window_index": window_index,
                "detection_rate_sampled": float(tr.detection_rate),
                "n_detected_frames": int(n_det),
                "pose_npy": str(pose_npy_path.resolve()),
                "pose_video": str(pose_video_path.resolve()),
                "pose_mask_path": mask_p,
                "pairwise_probs": pair_probs,
                "meta_result": meta_result,
            }
            cats_out.append(cat_row)

            save_json(
                {
                    "pairwise_probs": pair_probs,
                    "meta_result": meta_result,
                    "local_track_id": int(tid),
                    "detection_rate_sampled": float(tr.detection_rate),
                    "n_detected_frames": int(n_det),
                },
                cat_dir / "result.json",
            )

    out: dict[str, Any] = {
        "branch": "video",
        "multicat_video_only": True,
        "multicat_cat_count": n_cats,
        "multicat_diag": multicat_diag,
        "pose_meta": pose_meta_branch,
        "cats": cats_out,
    }
    return out
