"""
Pose assembler: run VitInference frame-by-frame on a video, collecting per-frame
keypoints while simultaneously writing a pose-overlay video.

Returns:
  - ``pose_npy_path`` – saved (T, 17, 3) float32 array (x, y, confidence)
  - ``pose_video_path`` – rendered overlay video
  - ``pose_meta`` – dict with frame counts, detection stats

The assembler also produces a PoseDataset-compatible record dict that can be
fed directly into ``PoseDataset`` / ``run_inference`` from the stacking script.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
_VITPOSE_ROOT = REPO_ROOT / "video" / "easyViTPose"

# Defaults match ``src/dataset_construction/config.yaml`` (animal pose extraction).
DEFAULT_VITPOSE_MODEL = str(REPO_ROOT / "models" / "pose_est" / "vitpose-h-apt36k.pth")
DEFAULT_VITPOSE_DATASET = "apt36k"
DEFAULT_VITPOSE_ARCH = "h"
DEFAULT_YOLO = str(REPO_ROOT / "models" / "yolo" / "yolov8x.pt")

# Number of keypoints expected by the pairwise ST-GCN base models
N_KEYPOINTS = 17

# Match ``src/dataset_construction/config.yaml`` pose_extraction (same as pairwise training).
_STGCN_N_FRAMES = 35
_STGCN_SAMPLE_TARGET_FPS = 5.0
_STGCN_SAMPLE_MAX_DURATION_SEC = 7.0


def _stgcn_dense_frame_indices(
    video_fps: float,
    dense_frame_count: int,
    *,
    target_fps: float = _STGCN_SAMPLE_TARGET_FPS,
    n_target_frames: int = _STGCN_N_FRAMES,
    max_duration_sec: float = _STGCN_SAMPLE_MAX_DURATION_SEC,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    """
    Which dense frame indices feed ST-GCN, mirroring
    ``sample_frames_fixed_rate`` in ``video/easyViTPose/extraction/06_pose_extraction.py``.
    Returns (indices length n_sampled≤n_target, mask of shape (n_target_frames,), stats).
    """
    if dense_frame_count <= 0:
        mask = np.zeros((n_target_frames,), dtype=bool)
        stats: dict[str, Any] = {
            "dense_frame_count": 0,
            "training_pose_n_frames": n_target_frames,
            "training_target_fps": target_fps,
            "training_sampled_frames": 0,
            "training_padded_frames": n_target_frames,
        }
        return [], mask, stats

    fps = float(video_fps)
    if fps <= 1e-6:
        fps = 30.0

    clip_duration = dense_frame_count / fps
    effective_d = min(clip_duration, max_duration_sec)
    dt = 1.0 / target_fps if target_fps > 0 else 1.0
    times: list[float] = []
    t = 0.0
    while t < effective_d - 1e-9 and len(times) < n_target_frames:
        times.append(t)
        t += dt

    indices: list[int] = []
    for tv in times:
        fi = int(round(tv * fps))
        fi = max(0, min(dense_frame_count - 1, fi))
        indices.append(fi)

    n_sampled = len(indices)
    mask = np.zeros((n_target_frames,), dtype=bool)
    mask[:n_sampled] = True

    stats = {
        "dense_frame_count": dense_frame_count,
        "dense_est_duration_sec": clip_duration,
        "training_pose_n_frames": n_target_frames,
        "training_target_fps": target_fps,
        "training_max_clip_sec": max_duration_sec,
        "training_sampled_frames": n_sampled,
        "training_padded_frames": n_target_frames - n_sampled,
    }
    return indices, mask, stats


def _bootstrap_vitpose() -> None:
    """Add video/easyViTPose/ to sys.path so ``easy_ViTPose`` is importable."""
    vp = str(_VITPOSE_ROOT)
    if vp not in sys.path:
        sys.path.insert(0, vp)


def _open_writer(
    path: Path, fps: float, frame_size_wh: tuple[int, int]
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for tag in ("mp4v", "avc1", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(str(path), fourcc, float(fps), frame_size_wh)
        if w.isOpened():
            return w
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def _reencode_pose_video_h264_inplace(path: Path) -> None:
    """
    OpenCV usually writes MPEG-4 Part 2 (mp4v). Browsers, Quick Look, and editor
    media previews expect H.264; re-encode in place when ffmpeg is available.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning(
            "ffmpeg not on PATH; pose video is mp4v — use VLC/QuickTime or install ffmpeg for H.264."
        )
        return

    tmp = path.with_name(f"{path.stem}._h264_tmp{path.suffix}")
    if sys.platform == "darwin":
        vcodec_args = ["-c:v", "h264_videotoolbox", "-b:v", "8M", "-pix_fmt", "yuv420p"]
    else:
        vcodec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(path),
        "-an",
        *vcodec_args,
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.warning(
            "ffmpeg H.264 re-encode failed; keeping OpenCV mp4v file. stderr:\n%s",
            (e.stderr or "").strip() or str(e),
        )
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return
    try:
        tmp.replace(path)
    except OSError as e:
        logger.warning("Could not replace pose video with H.264 version: %s", e)


def extract_poses_and_render(
    video_path: str | Path,
    pose_npy_out: str | Path,
    pose_video_out: str | Path,
    *,
    vitpose_model: str = DEFAULT_VITPOSE_MODEL,
    yolo_model: str = DEFAULT_YOLO,
    model_name: str = DEFAULT_VITPOSE_ARCH,
    dataset: str = DEFAULT_VITPOSE_DATASET,
    det_class: str = "cat",
    yolo_size: int = 320,
    yolo_step: int = 1,
    single_pose: bool = True,
    conf_threshold: float = 0.35,
    show_yolo: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Run VitInference on training-aligned sampled frames (5 Hz, up to 35 bins),
    collect keypoints, and write a pose-overlay video at the same sampled rate.

    Keypoints shape per frame: (17, 3) → (x, y, confidence).
    Missing-detection frames are filled with zeros.

    Returns:
        pose_npy_path, pose_video_path, meta_dict
    """
    _bootstrap_vitpose()
    from easy_ViTPose import VitInference  # noqa: E402

    video_path = Path(video_path).resolve()
    pose_npy_out = Path(pose_npy_out)
    pose_video_out = Path(pose_video_out)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(vitpose_model).is_file():
        raise FileNotFoundError(
            f"ViTPose model not found: {vitpose_model}\n"
            f"Expected e.g. {DEFAULT_VITPOSE_MODEL} (dataset={DEFAULT_VITPOSE_DATASET}, arch={DEFAULT_VITPOSE_ARCH}). "
            "See README / dataset_construction config for download links."
        )
    if not Path(yolo_model).is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {yolo_model}\n"
            "Download yolov8x.pt and place it at models/yolo/"
        )

    # Video metadata
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first = cap.read()
    cap.release()
    if not ret or first is None:
        raise RuntimeError(f"Cannot read first frame from {video_path}")
    h, w = first.shape[:2]

    # VitInference needs to be constructed from easyViTPose root because
    # VitInference.__init__ does os.path.isfile(model) – absolute paths work fine.
    _orig_cwd = os.getcwd()
    try:
        os.chdir(_VITPOSE_ROOT)

        model_obj = VitInference(
            str(vitpose_model),
            str(yolo_model),
            model_name=model_name,
            dataset=dataset,
            det_class=det_class,
            yolo_size=yolo_size,
            is_video=True,
            single_pose=single_pose,
            yolo_step=yolo_step,
        )
        model_obj.reset()
    finally:
        os.chdir(_orig_cwd)

    # Training-aligned sampling: uniform @ 5 Hz over the first <=7s, capped to 35 bins.
    idx_tr, pose_mask_vec, samp_stats = _stgcn_dense_frame_indices(fps, total_frames)

    writer = _open_writer(pose_video_out, _STGCN_SAMPLE_TARGET_FPS, (w, h))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for sampled inference: {video_path}")

    all_keypoints: list[np.ndarray] = []
    n_detected = 0
    n_missed = 0

    blank_kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)

    for fi in idx_tr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img_bgr = cap.read()
        if not ok or img_bgr is None:
            img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frame_kp_dict: dict[Any, np.ndarray] = model_obj.inference(img)

        # Pick the highest-confidence detection (by mean keypoint confidence)
        if frame_kp_dict:
            best_kp = max(
                frame_kp_dict.values(),
                key=lambda kp: float(np.mean(kp[:, 2])),
            )
            kp = best_kp[:N_KEYPOINTS].astype(np.float32)
            if kp.shape[0] < N_KEYPOINTS:
                pad = np.zeros((N_KEYPOINTS - kp.shape[0], 3), dtype=np.float32)
                kp = np.concatenate([kp, pad], axis=0)
            all_keypoints.append(kp)
            n_detected += 1
        else:
            all_keypoints.append(blank_kp.copy())
            n_missed += 1

        # Render overlay frame → BGR for VideoWriter
        try:
            rgb_overlay = model_obj.draw(
                show_yolo=show_yolo,
                show_raw_yolo=False,
                confidence_threshold=conf_threshold,
            )
            bgr = rgb_overlay[..., ::-1]
        except Exception:
            bgr = img_bgr

        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h))
        writer.write(bgr)

    cap.release()
    writer.release()
    _reencode_pose_video_h264_inplace(pose_video_out)

    n_sampled = len(all_keypoints)

    pose_stgcn = np.zeros((_STGCN_N_FRAMES, N_KEYPOINTS, 3), dtype=np.float32)
    for ti, kp in enumerate(all_keypoints):
        pose_stgcn[ti] = kp

    pose_mask_path = pose_npy_out.parent / f"{pose_npy_out.stem}_mask.npy"
    pose_npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(pose_npy_out), pose_stgcn)
    np.save(str(pose_mask_path), pose_mask_vec.astype(np.bool_))

    meta: dict[str, Any] = {
        "n_frames_source_video": total_frames,
        "source_video_fps": fps,
        "n_frames_sampled_inference": n_sampled,
        "n_frames_detected": n_detected,
        "n_frames_missed": n_missed,
        "detection_rate_sampled": n_detected / max(n_sampled, 1),
        # Back-compat aliases
        "n_frames_total": n_sampled,
        "detection_rate": n_detected / max(n_sampled, 1),
        "fps": _STGCN_SAMPLE_TARGET_FPS,
        "frame_width": w,
        "frame_height": h,
        "pose_npy_shape": list(pose_stgcn.shape),
        "pose_mask_path": str(pose_mask_path.resolve()),
        "training_pose_sampling": samp_stats,
        "video_path": str(video_path),
    }

    logger.info(
        "Pose extraction: sampled %d frames (%.1f%% det), "
        "ST-GCN pose %s (target_fps=%s, bins=%s, mask %s)",
        n_sampled,
        100.0 * meta["detection_rate_sampled"],
        pose_stgcn.shape,
        _STGCN_SAMPLE_TARGET_FPS,
        _STGCN_N_FRAMES,
        pose_mask_path.name,
    )

    return pose_npy_out, pose_video_out, meta


def build_pose_record(
    pose_npy_path: str | Path,
    snippet_id: str,
    *,
    pose_mask_path: str | Path | None = None,
    gpt_resolution: str | None = None,
) -> dict[str, Any]:
    """
    Build a minimal PoseDataset-compatible record from a saved pose .npy file.

    The dummy label fields are required by ``PoseDataset.__getitem__`` but are
    not used by the stacking ``run_inference`` function.
    """
    rec: dict[str, Any] = {
        "pose_path": str(Path(pose_npy_path).resolve()),
        "snippet_id": snippet_id,
        "label_int": 0,
        "binary_label_int": 0,
        "pose_mask_path": (
            str(Path(pose_mask_path).resolve()) if pose_mask_path else None
        ),
        "cat_id": snippet_id,
    }
    if gpt_resolution:
        rec["gpt_resolution"] = str(gpt_resolution)
    return rec
