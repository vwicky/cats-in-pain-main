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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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

# Match easy_ViTPose VitInference det_class → YOLO COCO indices.
_DETC_TO_YOLO_YOLOC: dict[str, list[int]] = {
    "human": [0],
    "cat": [15],
    "dog": [16],
    "horse": [17],
    "sheep": [18],
    "cow": [19],
    "elephant": [20],
    "bear": [21],
    "zebra": [22],
    "giraffe": [23],
    "animals": [15, 16, 17, 18, 19, 20, 21, 22, 23],
}


@dataclass
class TrackedCat:
    """One SORT track after coverage filter (Pass 1 multicat)."""

    track_id: int
    detection_rate: float
    bboxes: dict[int, np.ndarray] = field(default_factory=dict)
    # bboxes[sample_index] = (x1, y1, x2, y2, conf) float32 full-frame xyxy


def _yolo_device_arg(device: torch.device | None) -> str | int | None:
    if device is None:
        return None
    if device.type == "cuda":
        return 0
    return str(device)


def _expand_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding_frac: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    pad_w = bw * padding_frac
    pad_h = bh * padding_frac
    xe1 = int(max(0, np.floor(x1 - pad_w)))
    ye1 = int(max(0, np.floor(y1 - pad_h)))
    xe2 = int(min(img_w, np.ceil(max(x1, x2) + pad_w)))
    ye2 = int(min(img_h, np.ceil(max(y1, y2) + pad_h)))
    return xe1, ye1, xe2, ye2


def _letterbox_rgb(img_rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0 = max(0, (out_h - h) // 2)
    x0 = max(0, (out_w - w) // 2)
    y1 = min(out_h, y0 + h)
    x1 = min(out_w, x0 + w)
    sh = y1 - y0
    sw = x1 - x0
    canvas[y0:y1, x0:x1] = img_rgb[:sh, :sw]
    return canvas


def detect_cats_and_bboxes(
    clip_path: str | Path,
    *,
    yolo_model: str,
    vitpose_dataset: str = DEFAULT_VITPOSE_DATASET,
    det_class: str | None = None,
    yolo_size: int = 320,
    yolo_step: int = 1,
    conf_threshold: float = 0.35,
    min_track_coverage: float = 0.15,
    max_cats: int = 8,
    device: torch.device | None = None,
) -> tuple[list[TrackedCat], dict[str, Any]]:
    """
    Pass 1 multicat: YOLO + SORT on ST-GCN-sampled frames only (no ViTPose).
    """
    _bootstrap_vitpose()
    from easy_ViTPose.sort import Sort  # noqa: PLC0415

    clip_path = Path(clip_path).resolve()
    if not clip_path.is_file():
        raise FileNotFoundError(f"Video not found: {clip_path}")
    if not Path(yolo_model).is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model}")

    if det_class is None:
        det_class = "animals" if vitpose_dataset in ("ap10k", "apt36k") else "human"
    yolo_classes = _DETC_TO_YOLO_YOLOC.get(det_class, [15])

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {clip_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, _first = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {clip_path}")

    idx_tr, _, _ = _stgcn_dense_frame_indices(fps, total_frames)
    n_sampled = len(idx_tr)
    per_frame: list[dict[int, np.ndarray]] = [{} for _ in range(n_sampled)]

    yolo = YOLO(yolo_model, task="detect")
    dev = _yolo_device_arg(device)

    min_hits = 3 if yolo_step == 1 else 1
    tracker = Sort(max_age=yolo_step, min_hits=min_hits, iou_threshold=0.3)
    frame_counter = 0

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for sampling: {clip_path}")

    for sample_i, fi in enumerate(idx_tr):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img_bgr = cap.read()
        if not ok or img_bgr is None:
            img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        res_pd = np.empty((0, 5))
        if frame_counter % yolo_step == 0 or frame_counter < 3:
            ykw: dict[str, Any] = {
                "verbose": False,
                "imgsz": yolo_size,
                "classes": yolo_classes,
            }
            if dev is not None:
                ykw["device"] = dev
            results = yolo(img_bgr, **ykw)[0]
            if results.boxes is not None and len(results.boxes):
                rows = [
                    r[:5].tolist()
                    for r in results.boxes.data.cpu().numpy()
                    if r[4] > conf_threshold
                ]
                if rows:
                    res_pd = np.array(rows, dtype=np.float32).reshape((-1, 5))
        frame_counter += 1
        res_trk = tracker.update(res_pd)
        if res_trk.size == 0:
            continue
        res_trk = np.atleast_2d(res_trk)
        for row in res_trk:
            x1, y1, x2, y2, score, tid = [float(row[j]) for j in range(6)]
            kid = int(tid)
            per_frame[sample_i][kid] = np.array(
                [x1, y1, x2, y2, score], dtype=np.float32
            )

    cap.release()

    track_ids: set[int] = set()
    for fd in per_frame:
        track_ids.update(fd.keys())
    raw_track_count = len(track_ids)
    track_rates: dict[int, float] = {}
    for tid in track_ids:
        pres = sum(1 for fd in per_frame if tid in fd)
        track_rates[tid] = pres / max(n_sampled, 1)

    seen_before = sorted(track_ids, key=lambda t: (-track_rates[t], t))
    eligible = [t for t in seen_before if track_rates[t] >= float(min_track_coverage)]
    truncated = False
    if len(eligible) > int(max_cats):
        eligible = eligible[: int(max_cats)]
        truncated = True

    kept_tracks: list[TrackedCat] = []
    for tid in eligible:
        bmap: dict[int, np.ndarray] = {}
        for si, fd in enumerate(per_frame):
            if tid in fd:
                bmap[si] = fd[tid].copy()
        kept_tracks.append(
            TrackedCat(
                track_id=int(tid),
                detection_rate=float(track_rates[tid]),
                bboxes=bmap,
            )
        )

    kept_ids = [t.track_id for t in kept_tracks]
    logger.info(
        "[multicat] clip=%s sampled_frames=%d raw_tracks=%d kept_tracks=%d ids=%s",
        clip_path.name,
        n_sampled,
        raw_track_count,
        len(kept_tracks),
        kept_ids,
    )

    diag: dict[str, Any] = {
        "n_sampled_frames": n_sampled,
        "raw_track_count": raw_track_count,
        "kept_track_count": len(kept_tracks),
        "kept_track_ids": kept_ids,
        "truncated": truncated,
    }
    return kept_tracks, diag


def extract_cat_poses_cropped(
    video_path: str | Path,
    tracked_cat: TrackedCat,
    out_dir: Path,
    vit_model: Any,
    *,
    padding_frac: float = 0.2,
    conf_threshold: float = 0.35,
    show_yolo: bool = True,
) -> tuple[Path, Path, Path, int, dict[str, Any]]:
    """
    Pass 2 multicat: ViTPose on expanded bbox crops; full-frame keypoints for ST-GCN.
    Reuse one ``VitInference`` (single_pose=True, is_video=False) from video_branch.
    """
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_npy_out = out_dir / "raw_poses.npy"
    pose_mask_path = out_dir / "pose_mask.npy"
    pose_video_out = out_dir / "pose_video.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first = cap.read()
    cap.release()
    if not ret or first is None:
        raise RuntimeError(f"Cannot read first frame from {video_path}")
    full_h, full_w = first.shape[:2]

    idx_tr, pose_mask_vec, samp_stats = _stgcn_dense_frame_indices(fps, total_frames)
    n_sampled = len(idx_tr)
    blank_kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)

    max_out_w, max_out_h = 1, 1
    for si in range(n_sampled):
        bb = tracked_cat.bboxes.get(si)
        if bb is None:
            continue
        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        xe1, ye1, xe2, ye2 = _expand_bbox_xyxy(
            x1, y1, x2, y2, padding_frac, full_w, full_h
        )
        cw, ch = max(1, xe2 - xe1), max(1, ye2 - ye1)
        max_out_w = max(max_out_w, cw)
        max_out_h = max(max_out_h, ch)

    max_out_w = max(max_out_w, 2)
    max_out_h = max(max_out_h, 2)

    pose_stgcn = np.zeros((_STGCN_N_FRAMES, N_KEYPOINTS, 3), dtype=np.float32)
    overlays_bgr: list[np.ndarray] = []
    n_detected = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for pose extraction: {video_path}")

    for ti, fi in enumerate(idx_tr):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img_bgr = cap.read()
        if not ok or img_bgr is None:
            img_bgr = np.zeros((full_h, full_w, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        bb = tracked_cat.bboxes.get(ti)
        if bb is None:
            pose_stgcn[ti] = blank_kp
            overlays_bgr.append(np.zeros((max_out_h, max_out_w, 3), dtype=np.uint8))
            continue

        x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        xe1, ye1, xe2, ye2 = _expand_bbox_xyxy(
            x1, y1, x2, y2, padding_frac, full_w, full_h
        )
        crop_rgb = img_rgb[ye1:ye2, xe1:xe2]
        if crop_rgb.size == 0:
            pose_stgcn[ti] = blank_kp
            overlays_bgr.append(np.zeros((max_out_h, max_out_w, 3), dtype=np.uint8))
            continue

        kp_dict = vit_model.inference(crop_rgb)
        kp_crop = next(iter(kp_dict.values())) if kp_dict else None
        if kp_crop is None:
            kp_full = blank_kp.copy()
        else:
            k = kp_crop[:N_KEYPOINTS].astype(np.float32)
            if k.shape[0] < N_KEYPOINTS:
                pad = np.zeros((N_KEYPOINTS - k.shape[0], 3), dtype=np.float32)
                k = np.concatenate([k, pad], axis=0)
            # Required full-frame remap for ST-GCN — training expects full-frame keypoints (not crop-local).
            kp_full = k.copy()
            kp_full[:, 0] = k[:, 0] + float(ye1)
            kp_full[:, 1] = k[:, 1] + float(xe1)

        pose_stgcn[ti] = kp_full
        if kp_crop is not None and float(np.mean(kp_full[:, 2])) > 1e-6:
            n_detected += 1

        try:
            rgb_overlay = vit_model.draw(
                show_yolo=show_yolo,
                show_raw_yolo=False,
                confidence_threshold=conf_threshold,
            )
        except Exception:
            rgb_overlay = crop_rgb
        if rgb_overlay.shape[0] != crop_rgb.shape[0] or rgb_overlay.shape[1] != crop_rgb.shape[1]:
            rgb_overlay = cv2.resize(
                rgb_overlay,
                (crop_rgb.shape[1], crop_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        rgb_lb = _letterbox_rgb(rgb_overlay, max_out_w, max_out_h)
        overlays_bgr.append(cv2.cvtColor(rgb_lb, cv2.COLOR_RGB2BGR))

    cap.release()

    for ti2 in range(n_sampled, _STGCN_N_FRAMES):
        pose_stgcn[ti2] = blank_kp
        overlays_bgr.append(np.zeros((max_out_h, max_out_w, 3), dtype=np.uint8))

    writer = _open_writer(
        pose_video_out, _STGCN_SAMPLE_TARGET_FPS, (max_out_w, max_out_h)
    )
    for bgr in overlays_bgr[:_STGCN_N_FRAMES]:
        writer.write(bgr)
    writer.release()
    _reencode_pose_video_h264_inplace(pose_video_out)

    np.save(str(pose_npy_out), pose_stgcn)
    np.save(str(pose_mask_path), pose_mask_vec.astype(np.bool_))

    pose_meta: dict[str, Any] = {
        "n_frames_source_video": total_frames,
        "source_video_fps": fps,
        "n_frames_sampled_inference": n_sampled,
        "n_frames_detected": n_detected,
        "frame_width": full_w,
        "frame_height": full_h,
        "pose_npy_shape": list(pose_stgcn.shape),
        "pose_mask_path": str(pose_mask_path.resolve()),
        "training_pose_sampling": samp_stats,
        "video_path": str(video_path),
    }
    return pose_npy_out, pose_video_out, pose_mask_path, n_detected, pose_meta


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
    timer: Any | None = None,
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
    from contextlib import nullcontext

    yolo_ctx = timer.step("yolo_inference") if timer is not None else nullcontext()
    with yolo_ctx:
        os.chdir(_VITPOSE_ROOT)
        try:
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

    from contextlib import nullcontext

    vit_ctx = timer.step("vitpose_inference") if timer is not None else nullcontext()
    with vit_ctx:
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


def extract_multicat_poses_and_render(
    video_path: str | Path,
    run_dir: Path,
    pose_video_out: str | Path,
    *,
    vitpose_model: str = DEFAULT_VITPOSE_MODEL,
    yolo_model: str = DEFAULT_YOLO,
    model_name: str = DEFAULT_VITPOSE_ARCH,
    dataset: str = DEFAULT_VITPOSE_DATASET,
    det_class: str = "cat",
    yolo_size: int = 320,
    yolo_step: int = 1,
    conf_threshold: float = 0.35,
    show_yolo: bool = True,
    min_track_coverage: float = 0.15,
    max_cats: int = 8,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    """
    Multi-cat pose extraction with SORT (``single_pose=False``).

    Returns composite pose video, shared-style composite meta, per-track file info
    (absolute npy/mask paths), and optional ``multicat_diag`` for empty/truncation.
    """
    _bootstrap_vitpose()
    from easy_ViTPose import VitInference  # noqa: E402

    video_path = Path(video_path).resolve()
    run_dir = Path(run_dir).resolve()
    pose_video_out = Path(pose_video_out)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(vitpose_model).is_file():
        raise FileNotFoundError(f"ViTPose model not found: {vitpose_model}")
    if not Path(yolo_model).is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model}")

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
            single_pose=False,
            yolo_step=yolo_step,
        )
        model_obj.reset()
    finally:
        os.chdir(_orig_cwd)

    idx_tr, pose_mask_vec, samp_stats = _stgcn_dense_frame_indices(fps, total_frames)
    writer = _open_writer(pose_video_out, _STGCN_SAMPLE_TARGET_FPS, (w, h))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for sampled inference: {video_path}")

    per_frame_keypoints: list[dict[int, np.ndarray]] = []
    n_detected_frames = 0

    blank_kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)

    for fi in idx_tr:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img_bgr = cap.read()
        if not ok or img_bgr is None:
            img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frame_kp_dict: dict[Any, np.ndarray] = model_obj.inference(img)
        fd: dict[int, np.ndarray] = {}
        for k, kp_raw in frame_kp_dict.items():
            try:
                kid = int(k)
            except (TypeError, ValueError):
                continue
            kp = kp_raw[:N_KEYPOINTS].astype(np.float32)
            if kp.shape[0] < N_KEYPOINTS:
                pad = np.zeros((N_KEYPOINTS - kp.shape[0], 3), dtype=np.float32)
                kp = np.concatenate([kp, pad], axis=0)
            fd[kid] = kp

        per_frame_keypoints.append(fd)
        if fd:
            n_detected_frames += 1

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

    n_sampled = len(per_frame_keypoints)
    track_ids = set()
    for fd in per_frame_keypoints:
        track_ids.update(fd.keys())

    track_rates: dict[int, float] = {}
    for tid in track_ids:
        pres = sum(1 for fd in per_frame_keypoints if tid in fd)
        track_rates[tid] = pres / max(n_sampled, 1)

    seen_before: list[int] = sorted(track_ids, key=lambda t: (-track_rates[t], t))
    eligible = [t for t in seen_before if track_rates[t] >= float(min_track_coverage)]
    multicat_diag: dict[str, Any] | None = None

    dropped_trunc: list[int] = []
    n_passing_cov = len([t for t in seen_before if track_rates[t] >= float(min_track_coverage)])
    if len(eligible) > int(max_cats):
        dropped_trunc = eligible[int(max_cats) :]
        eligible = eligible[: int(max_cats)]
        multicat_diag = {
            "status": "truncated_to_max_cats",
            "max_cats": int(max_cats),
            "n_tracks_passing_filter": n_passing_cov,
            "dropped_local_track_ids": [int(x) for x in dropped_trunc],
        }
        logger.warning(
            "Multicat: truncated %d tracks to max_cats=%d; dropped ids=%s",
            len(dropped_trunc),
            max_cats,
            dropped_trunc,
        )

    if not eligible:
        diag_empty: dict[str, Any] = {
            "status": "no_tracks_above_coverage",
            "min_track_coverage": float(min_track_coverage),
            "tracks_seen_before_filter": [int(x) for x in seen_before],
            "n_sampled_frames": n_sampled,
        }
        logger.info("Multicat: no tracks passed coverage floor (min=%s)", min_track_coverage)
        meta = {
            "n_frames_source_video": total_frames,
            "source_video_fps": fps,
            "n_frames_sampled_inference": n_sampled,
            "n_frames_detected": n_detected_frames,
            "n_frames_missed": n_sampled - n_detected_frames,
            "detection_rate_sampled": n_detected_frames / max(n_sampled, 1),
            "fps": _STGCN_SAMPLE_TARGET_FPS,
            "frame_width": w,
            "frame_height": h,
            "training_pose_sampling": samp_stats,
            "video_path": str(video_path),
            "multicat_composite": True,
        }
        return pose_video_out, meta, [], diag_empty

    pose_stgcn_blank = np.zeros((_STGCN_N_FRAMES, N_KEYPOINTS, 3), dtype=np.float32)
    track_rows: list[dict[str, Any]] = []

    for tid in eligible:
        seq: list[np.ndarray] = []
        for fd in per_frame_keypoints:
            if tid in fd:
                seq.append(fd[tid])
            else:
                seq.append(blank_kp.copy())
        pose_stgcn = pose_stgcn_blank.copy()
        for ti, kp in enumerate(seq):
            if ti < _STGCN_N_FRAMES:
                pose_stgcn[ti] = kp
        sub = run_dir / "cats" / str(tid)
        sub.mkdir(parents=True, exist_ok=True)
        pose_npy_out = sub / "raw_poses.npy"
        pose_mask_path = sub / "raw_poses_mask.npy"
        np.save(str(pose_npy_out), pose_stgcn)
        np.save(str(pose_mask_path), pose_mask_vec.astype(np.bool_))

        track_rows.append(
            {
                "local_track_id": int(tid),
                "detection_rate_sampled": float(track_rates[tid]),
                "pose_npy": pose_npy_out.resolve(),
                "pose_mask_path": pose_mask_path.resolve(),
            }
        )

    meta = {
        "n_frames_source_video": total_frames,
        "source_video_fps": fps,
        "n_frames_sampled_inference": n_sampled,
        "n_frames_detected": n_detected_frames,
        "n_frames_missed": n_sampled - n_detected_frames,
        "detection_rate_sampled": n_detected_frames / max(n_sampled, 1),
        "fps": _STGCN_SAMPLE_TARGET_FPS,
        "frame_width": w,
        "frame_height": h,
        "pose_npy_shape": [_STGCN_N_FRAMES, N_KEYPOINTS, 3],
        "training_pose_sampling": samp_stats,
        "video_path": str(video_path),
        "multicat_composite": True,
        "multicat_local_track_ids_scored": [int(t) for t in eligible],
    }

    return pose_video_out, meta, track_rows, multicat_diag


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
