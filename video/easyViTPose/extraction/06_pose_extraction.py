#!/usr/bin/env python3
"""
Extract ViTPose keypoints for training-ready snippets (final_dataset_v2).

YOLO runs per frame. ViTPose can micro-batch frames when ``vitpose_forward_batch`` > 1
(uses top-1 YOLO box per frame for throughput). Optional ``decode_prefetch`` overlaps
video decode of the next clip with GPU work. Multi-process clip parallelism is not used:
multiple Python processes on one GPU usually serialize and waste VRAM.

Run from repository root:

  python dataset_construction/06_pose_extraction.py
  python dataset_construction/06_pose_extraction.py --dry-run
  python dataset_construction/06_pose_extraction.py --limit 50
  python dataset_construction/06_pose_extraction.py --device cuda
  python dataset_construction/06_pose_extraction.py --update-manifest
  python dataset_construction/06_pose_extraction.py --rebuild
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_EASY_VTP_ROOT = REPO_ROOT / "video" / "easyViTPose"
if _EASY_VTP_ROOT.is_dir():
    _p = str(_EASY_VTP_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from easy_ViTPose.configs.ViTPose_common import data_cfg  # noqa: E402
from easy_ViTPose.inference import MEAN, STD, VitInference  # noqa: E402
from easy_ViTPose.vit_models.model import ViTPose  # noqa: E402
from easy_ViTPose.vit_utils.inference import pad_image  # noqa: E402
from easy_ViTPose.vit_utils.util import dyn_model_import  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm")


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(preference: str) -> str:
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _yolo_device(device: str) -> int | str:
    return 0 if device == "cuda" else device


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_index_last_wins(path: Path) -> dict[str, dict[str, Any]]:
    d: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("snippet_id")
            if isinstance(sid, str):
                d[sid] = row
    return d


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_write_jsonl_from_dict(path: Path, by_id: dict[str, dict[str, Any]]) -> None:
    keys = sorted(by_id.keys())
    atomic_write_jsonl(path, [by_id[k] for k in keys])


def probe_v1_pose_format(repo_root: Path, legacy_manifest: Path) -> tuple[bool, str]:
    """
    Decide whether new extractions should normalize x,y to [0,1] via divide-by W,H.

    Returns (normalize_xy_to_unit, explanation_message).
    """
    if not legacy_manifest.is_file():
        msg = (
            "Legacy manifest missing; assuming VitInference-style pixel coords → "
            "normalize x,y by frame W,H to [0,1]."
        )
        print(f"WARNING: {msg}")
        return True, msg

    first_pose: Path | None = None
    for row in load_jsonl(legacy_manifest):
        pp = row.get("pose_path")
        if not pp or not isinstance(pp, str):
            continue
        cand = repo_root / pp
        if cand.is_file():
            first_pose = cand
            break

    if first_pose is None:
        msg = (
            "No readable legacy pose_path in legacy manifest; assuming pixel coords → normalize to [0,1]."
        )
        print(f"WARNING: {msg}")
        return True, msg

    v1 = np.load(first_pose)
    xmin, xmax = float(np.min(v1[:, :, 0])), float(np.max(v1[:, :, 0]))
    ymin, ymax = float(np.min(v1[:, :, 1])), float(np.max(v1[:, :, 1]))
    cmin, cmax = float(np.min(v1[:, :, 2])), float(np.max(v1[:, :, 2]))
    print(f"V1 pose sample: {first_pose.relative_to(repo_root)}")
    print(f"  shape: {v1.shape}  dtype: {v1.dtype}")
    print(f"  x range: [{xmin:.4f}, {xmax:.4f}]")
    print(f"  y range: [{ymin:.4f}, {ymax:.4f}]")
    print(f"  conf range: [{cmin:.4f}, {cmax:.4f}]")

    looks_unit = xmax <= 1.5 and ymax <= 1.5 and xmin >= -0.01 and ymin >= -0.01
    if looks_unit:
        return (
            False,
            "Legacy sample x,y in ~[0,1]; new extractions will skip divide-by W,H on x,y.",
        )
    return (
        True,
        "Legacy sample x,y look like pixels; new extractions will divide x,y by frame W,H.",
    )


class CatPoseBackend:
    """Single Ultralytics YOLO + ViTPose (torch), mirroring VitInference internals without duplicate YOLO."""

    def __init__(
        self,
        vitpose_ckpt: Path,
        yolo_weights: Path,
        device: str,
        yolo_size: int,
        yolo_conf: float,
        cat_class: int,
        vitpose_dataset: str = "apt36k",
        vitpose_model_name: str = "h",
    ):
        self.device = torch.device(device)
        self.yolo_size = yolo_size
        self.yolo_conf = float(yolo_conf)
        self.cat_class = int(cat_class)
        if not yolo_weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")
        if not vitpose_ckpt.is_file():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {vitpose_ckpt}")

        self.yolo = YOLO(str(yolo_weights), task="detect")

        model_cfg = dyn_model_import(vitpose_dataset, vitpose_model_name)
        self._vit_pose = ViTPose(model_cfg)
        self._vit_pose.eval()
        try:
            ckpt = torch.load(vitpose_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(vitpose_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self._vit_pose.load_state_dict(ckpt["state_dict"])
        else:
            self._vit_pose.load_state_dict(ckpt)
        self._vit_pose.to(self.device)

        ts = data_cfg["image_size"]
        self.target_size = (int(ts[0]), int(ts[1]))

    def pre_img(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255.0
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    @torch.no_grad()
    def infer_pose_crop(self, img_rgb: np.ndarray) -> np.ndarray | None:
        """Run ViTPose on an RGB crop; return (17, 3) float32 or None if invalid."""
        if img_rgb.size == 0 or img_rgb.shape[0] < 2 or img_rgb.shape[1] < 2:
            return None
        img_input, org_h, org_w = self.pre_img(img_rgb)
        x = torch.from_numpy(img_input).to(self.device)
        heatmaps = self._vit_pose(x).detach().cpu().numpy()
        kpts_all = VitInference.postprocess(heatmaps, org_w, org_h)
        if kpts_all is None or len(kpts_all) == 0:
            return None
        return np.asarray(kpts_all[0], dtype=np.float32)

    @torch.no_grad()
    def infer_pose_crops_batch(self, crops_rgb: list[np.ndarray]) -> list[np.ndarray | None]:
        """Stack crops through ViTPose (same fixed resize per crop). Returns list len(crops_rgb)."""
        if not crops_rgb:
            return []
        ims: list[np.ndarray] = []
        dims: list[tuple[int, int]] = []
        for c in crops_rgb:
            inp, org_h, org_w = self.pre_img(c)
            ims.append(inp)
            dims.append((org_h, org_w))
        x = np.concatenate(ims, axis=0).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        heatmaps = self._vit_pose(xt).detach().cpu().numpy()
        out: list[np.ndarray | None] = []
        B = len(crops_rgb)
        for bi in range(B):
            hm = heatmaps[bi : bi + 1]
            org_h, org_w = dims[bi]
            kpts_all = VitInference.postprocess(hm, org_w, org_h)
            if kpts_all is None or len(kpts_all) == 0:
                out.append(None)
            else:
                out.append(np.asarray(kpts_all[0], dtype=np.float32))
        return out

    def yolo_cat_boxes(self, img_rgb: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Return list of (x1,y1,x2,y2,conf) for cat class after Ultralytics filtering."""
        res = self.yolo(
            img_rgb[..., ::-1],
            verbose=False,
            imgsz=self.yolo_size,
            device=_yolo_device(str(self.device)),
            classes=[self.cat_class],
            conf=self.yolo_conf,
        )[0]
        out: list[tuple[float, float, float, float, float]] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy()
        for i in range(len(xyxy)):
            if int(cls[i]) != self.cat_class:
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            c = float(conf[i])
            if c >= self.yolo_conf:
                out.append((x1, y1, x2, y2, c))
        return out


def expand_xyxy(xyxy: tuple[float, float, float, float], w: int, h: int, frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    px = bw * frac
    py = bh * frac
    nx1 = max(0, int(np.floor(x1 - px)))
    ny1 = max(0, int(np.floor(y1 - py)))
    nx2 = min(w - 1, int(np.ceil(x2 + px)))
    ny2 = min(h - 1, int(np.ceil(y2 + py)))
    if nx2 <= nx1 or ny2 <= ny1:
        return 0, 0, max(1, w - 1), max(1, h - 1)
    return nx1, ny1, nx2, ny2


def lift_kpts_from_crop(kpts_crop: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Map keypoints from cropped/padded coords to full-frame pixels."""
    if meta["mode"] == "full":
        return kpts_crop.copy()
    out = kpts_crop.copy()
    bbox = meta["bbox"]
    left_pad = int(meta["left_pad"])
    top_pad = int(meta["top_pad"])
    out[:, :2] += bbox[:2][::-1] - np.array([top_pad, left_pad], dtype=np.float32)
    return out


def prepare_frame_pose_crop_top1(
    backend: CatPoseBackend,
    img_rgb: np.ndarray,
    bbox_frac: float,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    """Single highest-confidence cat box → crop (+ pad_image). Fallback: full frame."""
    h, w = img_rgb.shape[:2]
    boxes = backend.yolo_cat_boxes(img_rgb)
    if boxes:
        x1, y1, x2, y2, _conf = max(boxes, key=lambda b: b[4])
        ex1, ey1, ex2, ey2 = expand_xyxy((x1, y1, x2, y2), w, h, bbox_frac)
        bbox = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)
        pad_bbox = 10.0
        bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img_rgb.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img_rgb.shape[0])
        crop = img_rgb[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        if crop.size == 0:
            meta = {"mode": "full"}
            return img_rgb.copy(), meta, False
        crop_pad, (left_pad, top_pad) = pad_image(crop, 3 / 4)
        meta = {
            "mode": "crop",
            "bbox": bbox,
            "left_pad": left_pad,
            "top_pad": top_pad,
        }
        return crop_pad, meta, True
    meta = {"mode": "full"}
    return img_rgb.copy(), meta, False


def infer_single_frame_pose(
    backend: CatPoseBackend,
    img_rgb: np.ndarray,
    bbox_frac: float,
) -> tuple[np.ndarray, bool]:
    """
    Returns ((17,3) float32 keypoints in pixel coords, yolo_hit).
    Matches easy_ViTPose bbox padding + pad_image + keypoint offset convention.
    """
    h, w = img_rgb.shape[:2]
    boxes = backend.yolo_cat_boxes(img_rgb)
    if boxes:
        best_kpts = None
        best_score = -1.0
        for x1, y1, x2, y2, _ in sorted(boxes, key=lambda b: -b[4]):
            ex1, ey1, ex2, ey2 = expand_xyxy((x1, y1, x2, y2), w, h, bbox_frac)
            bbox = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)
            pad_bbox = 10.0
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img_rgb.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img_rgb.shape[0])
            crop = img_rgb[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            if crop.size == 0:
                continue
            crop_pad, (left_pad, top_pad) = pad_image(crop, 3 / 4)
            kpts = backend.infer_pose_crop(crop_pad)
            if kpts is None:
                continue
            kpts = kpts.copy()
            kpts[:, :2] += bbox[:2][::-1] - np.array([top_pad, left_pad], dtype=np.float32)
            mean_conf = float(np.mean(kpts[:, 2]))
            if mean_conf > best_score:
                best_score = mean_conf
                best_kpts = kpts
        if best_kpts is not None:
            return best_kpts, True

    full = backend.infer_pose_crop(img_rgb)
    if full is None:
        return np.zeros((17, 3), dtype=np.float32), False
    return full.copy(), False


def normalize_pose_xy(pose_px: np.ndarray, w: int, h: int, normalize_xy: bool) -> np.ndarray:
    out = pose_px.astype(np.float32, copy=True)
    if normalize_xy and w > 0 and h > 0:
        out[:, 0] /= float(w)
        out[:, 1] /= float(h)
    return out


def run_clip_pose_inference(
    backend: CatPoseBackend,
    frames_bgr: list[np.ndarray],
    mask: np.ndarray,
    bbox_frac: float,
    normalize_xy: bool,
    forward_bs: int,
    n_frames: int,
    n_kp: int,
) -> tuple[np.ndarray, int]:
    """Fill pose tensor for real frames; return (pose_acc, n_yolo_frame_hits)."""
    pose_acc = np.zeros((n_frames, n_kp, 3), dtype=np.float32)
    yolo_hits_frame = 0

    real_indices = [i for i in range(n_frames) if mask[i]]

    if forward_bs <= 1:
        for i in real_indices:
            rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            kpts_px, yhit = infer_single_frame_pose(backend, rgb, bbox_frac)
            pose_acc[i] = normalize_pose_xy(kpts_px, w, h, normalize_xy)
            if yhit:
                yolo_hits_frame += 1
        return pose_acc, yolo_hits_frame

    jobs: list[tuple[int, np.ndarray, dict[str, Any], bool, int, int]] = []
    for i in real_indices:
        rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        crop_pad, meta, yhit = prepare_frame_pose_crop_top1(backend, rgb, bbox_frac)
        jobs.append((i, crop_pad, meta, yhit, h, w))
        if yhit:
            yolo_hits_frame += 1

    chunk_sz = max(1, forward_bs)
    for start in range(0, len(jobs), chunk_sz):
        chunk = jobs[start : start + chunk_sz]
        crops = [c for (_, c, _, _, _, _) in chunk]
        kp_list = backend.infer_pose_crops_batch(crops)
        for (fi, _crop, meta, _yh, h, w), kp_crop in zip(chunk, kp_list, strict=True):
            if kp_crop is None:
                kpts_px = np.zeros((n_kp, 3), dtype=np.float32)
            else:
                kpts_px = lift_kpts_from_crop(kp_crop, meta)
            pose_acc[fi] = normalize_pose_xy(kpts_px, w, h, normalize_xy)

    return pose_acc, yolo_hits_frame


def decode_clip_task(args: tuple[Any, ...]) -> tuple[str, Any, Any, Any, str | None]:
    """Runs in worker thread: decode one clip."""
    row, repo_root, snippets_dirs, target_fps, n_frames, max_dur = args
    sid = row.get("snippet_id")
    if not isinstance(sid, str):
        return str(sid), None, None, None, "bad_snippet_id"
    vid = resolve_video_path(repo_root, row, snippets_dirs)
    if vid is None:
        return sid, None, None, None, "video_not_found"
    fs, m, meta = sample_frames_fixed_rate(vid, target_fps, n_frames, max_dur)
    if fs is None or (meta and meta.get("error")):
        err = meta.get("error", "decode_failed") if meta else "decode_failed"
        return sid, None, None, None, str(err)
    return sid, fs, m, meta, None


def iter_decode_pipeline(
    clips: list[dict[str, Any]],
    decode_pf: int,
    repo_root: Path,
    snippets_dirs: list[Path],
    target_fps: float,
    n_frames: int,
    max_dur: float,
):
    """Yield (row, frames_bgr|None, mask|None, meta|None, dec_err|None) per clip."""
    if decode_pf <= 1:
        for row in clips:
            sid = row.get("snippet_id")
            if not isinstance(sid, str):
                yield row, None, None, None, "bad_snippet_id"
                continue
            vid = resolve_video_path(repo_root, row, snippets_dirs)
            if vid is None:
                yield row, None, None, None, "video_not_found"
                continue
            fs, m, meta = sample_frames_fixed_rate(vid, target_fps, n_frames, max_dur)
            if fs is None:
                yield row, None, None, meta, "cannot_open"
                continue
            if meta.get("error"):
                yield row, None, None, meta, str(meta["error"])
                continue
            yield row, fs, m, meta, None
        return

    pack = (repo_root, snippets_dirs, target_fps, n_frames, max_dur)
    with ThreadPoolExecutor(max_workers=min(decode_pf, 8)) as ex:
        next_fut: Future | None = None
        if clips:
            next_fut = ex.submit(decode_clip_task, (clips[0], *pack))
        for i, row in enumerate(clips):
            assert next_fut is not None
            sid_dec, fs, m, meta, err = next_fut.result()
            exp = row.get("snippet_id")
            if sid_dec != exp:
                raise RuntimeError(f"decode pipeline desync: expected {exp!r}, got {sid_dec!r}")
            if i + 1 < len(clips):
                next_fut = ex.submit(decode_clip_task, (clips[i + 1], *pack))
            else:
                next_fut = None
            if err:
                yield row, None, None, meta, err
            elif fs is None or meta is None:
                yield row, None, None, meta, "decode_failed"
            elif meta.get("error"):
                yield row, None, None, meta, str(meta["error"])
            else:
                yield row, fs, m, meta, None


def apply_repeat_last_pose(pose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Repeat last real frame x,y on padded slots; zero confidence on padded slots."""
    out = pose.copy()
    last_i: int | None = None
    for i in range(mask.shape[0]):
        if mask[i]:
            last_i = i
    if last_i is None:
        return out
    fill_xy = out[last_i, :, :2].copy()
    for i in range(mask.shape[0]):
        if not mask[i]:
            out[i, :, :2] = fill_xy
            out[i, :, 2] = 0.0
    return out


def apply_repeat_first_pose(pose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Repeat first real frame x,y on padded slots; zero confidence on padded slots."""
    out = pose.copy()
    first_i: int | None = None
    for i in range(mask.shape[0]):
        if mask[i]:
            first_i = i
            break
    if first_i is None:
        return out
    fill_xy = out[first_i, :, :2].copy()
    for i in range(mask.shape[0]):
        if not mask[i]:
            out[i, :, :2] = fill_xy
            out[i, :, 2] = 0.0
    return out


def apply_zero_pad_pose(pose: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Full zeros for padded timesteps (x, y, and confidence)."""
    out = pose.copy()
    out[~mask] = 0.0
    return out


def apply_pose_pad_strategy(pose: np.ndarray, mask: np.ndarray, strategy: str) -> np.ndarray:
    """Dispatch pad_strategy from config (case-insensitive)."""
    s = strategy.strip().lower()
    if s in ("repeat_last", "last"):
        return apply_repeat_last_pose(pose, mask)
    if s in ("repeat_first", "first"):
        return apply_repeat_first_pose(pose, mask)
    if s in ("zeros", "zero"):
        return apply_zero_pad_pose(pose, mask)
    print(
        f"WARNING: unknown pose_extraction.pad_strategy={strategy!r}; "
        f"using repeat_last.",
        file=sys.stderr,
    )
    return apply_repeat_last_pose(pose, mask)


def sample_frames_fixed_rate(
    video_path: Path,
    target_fps: float,
    n_frames: int,
    max_duration: float,
) -> tuple[list[np.ndarray] | None, np.ndarray | None, dict[str, Any]]:
    """
    Sample frames at fixed rate. Returns (frames_bgr_list, mask_bool, meta).
    Never raises; (None, None, meta) if open failed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, {"error": "cannot_open"}

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if fps <= 1e-6:
        cap.release()
        return None, None, {"error": "invalid_fps", "actual_fps": fps}

    clip_duration = n_total / fps
    effective_d = min(clip_duration, max_duration)
    dt = 1.0 / target_fps if target_fps > 0 else 1.0
    times: list[float] = []
    t = 0.0
    while t < effective_d - 1e-9 and len(times) < n_frames:
        times.append(t)
        t += dt
    n_real = len(times)
    n_padded = n_frames - n_real

    frames: list[np.ndarray] = []
    for tv in times:
        idx = int(round(tv * fps))
        if n_total > 0:
            idx = max(0, min(n_total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, fr = cap.read()
        if not ret or fr is None:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 256
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 256
            fr = np.zeros((h, w, 3), dtype=np.uint8)
        frames.append(fr)

    cap.release()

    shape = frames[0].shape if frames else (256, 256, 3)
    for _ in range(n_padded):
        frames.append(np.zeros(shape, dtype=np.uint8))

    mask = np.zeros((n_frames,), dtype=bool)
    mask[:n_real] = True

    meta = {
        "actual_fps": fps,
        "clip_duration_sec": clip_duration,
        "n_real_frames": n_real,
        "n_padded_frames": n_padded,
    }
    return frames, mask, meta


def resolve_video_path(repo_root: Path, row: dict[str, Any], snippets_dirs: list[Path]) -> Path | None:
    vp = row.get("video_path")
    if isinstance(vp, str) and vp.strip():
        q = repo_root / vp.strip()
        if q.is_file():
            return q
    sid = row.get("snippet_id")
    if isinstance(sid, str):
        for base in snippets_dirs:
            for ext in VIDEO_EXTS:
                c = base / f"{sid}{ext}"
                if c.is_file():
                    return c
    return None


def update_final_manifest(
    repo_root: Path,
    manifest_path: Path,
    index_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Merge pose_* fields into every row; return (n_non_null, n_null)."""
    rows = load_jsonl(manifest_path)
    n_non = 0
    n_null = 0
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        sid = row.get("snippet_id")
        ent = index_by_id.get(sid) if isinstance(sid, str) else None
        if ent and ent.get("status") == "done":
            row["pose_path"] = ent.get("pose_path")
            row["pose_mask_path"] = ent.get("pose_mask_path")
            row["pose_n_real_frames"] = ent.get("n_real_frames")
            row["pose_actual_fps"] = ent.get("actual_fps")
            n_non += 1
        else:
            row["pose_path"] = None
            row["pose_mask_path"] = None
            row["pose_n_real_frames"] = None
            row["pose_actual_fps"] = None
            n_null += 1
        out_rows.append(row)

    tmp = manifest_path.parent / (manifest_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(manifest_path)
    return n_non, n_null


def validate_random_outputs(
    repo_root: Path,
    index_by_id: dict[str, dict[str, Any]],
    n_frames: int,
    n_kp: int,
    normalize_xy: bool,
    k: int = 5,
) -> None:
    done_ids = [s for s, r in index_by_id.items() if r.get("status") == "done"]
    if len(done_ids) < 1:
        print("Validation: no done entries in index to sample.")
        return
    sample = random.sample(done_ids, k=min(k, len(done_ids)))
    print("\n--- Validation (random samples) ---")
    for sid in sample:
        r = index_by_id[sid]
        pp = repo_root / str(r.get("pose_path", ""))
        pm = repo_root / str(r.get("pose_mask_path", ""))
        if not pp.is_file() or not pm.is_file():
            print(f"WARNING: missing files for {sid}")
            continue
        pose = np.load(pp)
        m = np.load(pm)
        ok = True
        if pose.shape != (n_frames, n_kp, 3):
            print(f"WARNING: {sid} bad pose shape {pose.shape}")
            ok = False
        if pose.dtype != np.float32:
            print(f"WARNING: {sid} dtype {pose.dtype}")
            ok = False
        if m.shape != (n_frames,) or not np.issubdtype(m.dtype, np.bool_):
            print(f"WARNING: {sid} bad mask {m.shape} {m.dtype}")
            ok = False
        xmax, ymax = float(np.max(pose[:, :, 0])), float(np.max(pose[:, :, 1]))
        cmax = float(np.max(pose[:, :, 2]))
        if normalize_xy:
            if xmax > 1.01 or ymax > 1.01:
                print(f"WARNING: {sid} expected unit xy but max x,y = {xmax:.4f},{ymax:.4f}")
                ok = False
        zero_frames = int(np.sum(np.all(pose == 0, axis=(1, 2))))
        padded = int(np.sum(~m))
        mean_conf = float(np.mean(pose[:, :, 2]))
        print(
            f"{sid} | shape={pose.shape} dtype={pose.dtype} mean_conf={mean_conf:.4f} "
            f"zero_frames={zero_frames} padded_frames={padded}"
            + (" OK" if ok else " ISSUES")
        )


def compare_v1_v2_sample(repo_root: Path, legacy_manifest: Path, index_by_id: dict[str, dict[str, Any]]) -> None:
    """Load one legacy pose and one new pose when possible; print comparison."""
    legacy_path: Path | None = None
    if legacy_manifest.is_file():
        for row in load_jsonl(legacy_manifest):
            pp = row.get("pose_path")
            if pp and isinstance(pp, str):
                c = repo_root / pp
                if c.is_file():
                    legacy_path = c
                    break
    new_id = next((s for s, r in index_by_id.items() if r.get("status") == "done"), None)
    if legacy_path is None:
        print("\nV1 vs V2 comparison: no legacy pose file found; skipped.")
        return
    if new_id is None:
        print("\nV1 vs V2 comparison: no new pose in index; skipped.")
        return
    v1 = np.load(legacy_path)
    np2 = repo_root / str(index_by_id[new_id]["pose_path"])
    v2 = np.load(np2)
    print("\n--- V1 vs new format (shape / dtype / value ranges) ---")
    try:
        leg_disp = legacy_path.relative_to(repo_root)
    except ValueError:
        leg_disp = legacy_path
    print(f"V1 {leg_disp}: shape {v1.shape} dtype {v1.dtype}")
    print(
        f"  x[{v1[:,:,0].min():.4f},{v1[:,:,0].max():.4f}] "
        f"y[{v1[:,:,1].min():.4f},{v1[:,:,1].max():.4f}] "
        f"conf[{v1[:,:,2].min():.4f},{v1[:,:,2].max():.4f}]"
    )
    print(f"V2 {np2.relative_to(repo_root)}: shape {v2.shape} dtype {v2.dtype}")
    print(
        f"  x[{v2[:,:,0].min():.4f},{v2[:,:,0].max():.4f}] "
        f"y[{v2[:,:,1].min():.4f},{v2[:,:,1].max():.4f}] "
        f"conf[{v2[:,:,2].min():.4f},{v2[:,:,2].max():.4f}]"
    )


def run(args: argparse.Namespace) -> int:
    cfg_all = load_config()
    pe = cfg_all.get("pose_extraction") or {}
    fd = cfg_all.get("final_dataset") or {}
    sources = cfg_all.get("sources") or {}

    manifest_path = REPO_ROOT / fd["output_manifest"]
    cache_index_path = REPO_ROOT / pe["cache_index"]
    output_dir = REPO_ROOT / pe["output_dir"]
    legacy_manifest = REPO_ROOT / pe.get("legacy_pose_manifest", "data/dataset/final_dataset.jsonl")

    n_frames = int(pe["n_frames"])
    target_fps = float(pe["target_fps"])
    max_dur = float(pe["max_clip_duration"])
    pad_strategy = str(pe.get("pad_strategy", "repeat_last"))
    yolo_conf = float(pe["yolo_conf"])
    cat_class = int(pe["yolo_cat_class"])
    yolo_size = int(pe.get("yolo_size", 320))
    bbox_expand = float(pe.get("bbox_expand", 0.10))
    n_kp = int(pe.get("n_keypoints", 17))
    skip_existing = bool(pe.get("skip_existing", True)) and not args.rebuild

    snippets_dirs = [REPO_ROOT / x for x in sources.get("snippets_dirs", [])]

    device = resolve_device(args.device or pe.get("device", "auto"))

    # --- Manifest-only mode ---
    if args.update_manifest:
        idx = load_index_last_wins(cache_index_path)
        n_ok, n_bad = update_final_manifest(REPO_ROOT, manifest_path, idx)
        print(f"pose fields added: {n_ok} rows")
        print(f"pose fields null: {n_bad} rows (not extracted or error)")
        return 0

    rows_v2 = load_jsonl(manifest_path)
    train_ready = [r for r in rows_v2 if r.get("suitable_for_training") is True]
    scheduled = train_ready[: args.limit] if args.limit else train_ready
    scheduled_ids = {r["snippet_id"] for r in scheduled if isinstance(r.get("snippet_id"), str)}

    index_by_id = load_index_last_wins(cache_index_path)

    if args.rebuild:
        for sid in scheduled_ids:
            index_by_id.pop(sid, None)
        atomic_write_jsonl_from_dict(cache_index_path, index_by_id)

    to_process_list: list[dict[str, Any]] = []
    for r in scheduled:
        sid = r.get("snippet_id")
        if not isinstance(sid, str):
            continue
        if skip_existing and not args.rebuild:
            prev = index_by_id.get(sid)
            if prev and prev.get("status") == "done":
                continue
        to_process_list.append(r)

    n_skip = len(scheduled) - len(to_process_list)
    print(f"Pose extraction: {len(to_process_list)} to process, {n_skip} already done (skipping)")

    if args.dry_run:
        print("Dry run: exiting before V1 probe and model load.")
        return 0

    normalize_xy, probe_msg = probe_v1_pose_format(REPO_ROOT, legacy_manifest)
    print(f"Coordinate policy: normalize_xy_to_unit={normalize_xy} ({probe_msg})")

    output_dir.mkdir(parents=True, exist_ok=True)

    def finish_from_index(skip_models: bool = False) -> int:
        idx = load_index_last_wins(cache_index_path)
        n_ok, n_bad = update_final_manifest(REPO_ROOT, manifest_path, idx)
        print(f"\npose fields added: {n_ok} rows")
        print(f"pose fields null: {n_bad} rows (not extracted or error)")
        validate_random_outputs(REPO_ROOT, idx, n_frames, n_kp, normalize_xy)
        compare_v1_v2_sample(REPO_ROOT, legacy_manifest, idx)
        if skip_models:
            print("\n(No clips processed this run — manifest refreshed from index.)")
        return 0

    if len(to_process_list) == 0:
        print("Nothing to process; skipping model load.")
        return finish_from_index(skip_models=True)

    vit_ckpt = REPO_ROOT / pe["vitpose_checkpoint"]
    yolo_w = REPO_ROOT / pe["yolo_weights"]

    backend = CatPoseBackend(
        vit_ckpt,
        yolo_w,
        device,
        yolo_size,
        yolo_conf,
        cat_class,
        vitpose_dataset=str(pe.get("vitpose_dataset", "apt36k")),
        vitpose_model_name=str(pe.get("vitpose_model_name", "h")),
    )

    forward_bs = max(
        1,
        args.forward_batch if args.forward_batch is not None else int(pe.get("vitpose_forward_batch", 1)),
    )
    decode_pf = max(
        1,
        args.decode_prefetch if args.decode_prefetch is not None else int(pe.get("decode_prefetch", 1)),
    )
    if forward_bs > 1:
        print(
            f"ViTPose micro-batch size {forward_bs} (top-1 YOLO box per frame). "
            "Use --forward-batch 1 or vitpose_forward_batch: 1 for multi-candidate ViTPose voting."
        )
    if decode_pf > 1:
        print(f"Decode prefetch threads: {decode_pf} (next clip decodes while GPU runs).")

    done_c = err_c = 0
    hit_all = hit_partial = hit_zero = 0
    pad_none = pad_full = 0
    index_fp = open(cache_index_path, "a", encoding="utf-8")
    t0_run = time.perf_counter()
    processed_in_loop = 0

    pbar = tqdm(total=len(to_process_list), desc="Pose extraction", mininterval=0.5)

    def flush_index() -> None:
        index_fp.flush()

    try:
        for row, frames_bgr, mask, meta, dec_err in iter_decode_pipeline(
            to_process_list,
            decode_pf,
            REPO_ROOT,
            snippets_dirs,
            target_fps,
            n_frames,
            max_dur,
        ):
            sid = row.get("snippet_id")
            if not isinstance(sid, str):
                sid = str(sid)

            if dec_err:
                err_c += 1
                rec = {
                    "snippet_id": sid,
                    "pose_path": None,
                    "pose_mask_path": None,
                    "status": "error",
                    "error_msg": dec_err,
                    "n_frames_extracted": n_frames,
                    "n_real_frames": 0,
                    "n_padded_frames": n_frames,
                    "actual_fps": None,
                    "clip_duration_sec": None,
                    "n_yolo_hits": 0,
                    "latency_sec": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec
                pbar.set_postfix(done=done_c, err=err_c)
                processed_in_loop += 1
                pbar.update(1)
                continue

            t_clip = time.perf_counter()
            try:
                assert frames_bgr is not None and mask is not None and meta is not None

                pose_acc, yolo_hits_frame = run_clip_pose_inference(
                    backend,
                    frames_bgr,
                    mask,
                    bbox_expand,
                    normalize_xy,
                    forward_bs,
                    n_frames,
                    n_kp,
                )

                pose_acc = apply_pose_pad_strategy(pose_acc, mask, pad_strategy)

                rel_pose = output_dir / f"{sid}_pose.npy"
                rel_mask = output_dir / f"{sid}_pose_mask.npy"
                abs_pose = REPO_ROOT / rel_pose
                abs_mask = REPO_ROOT / rel_mask
                np.save(abs_pose, pose_acc.astype(np.float32))
                np.save(abs_mask, mask.astype(np.bool_))

                elapsed = time.perf_counter() - t_clip

                n_real = int(meta["n_real_frames"])
                if n_real > 0 and yolo_hits_frame == n_real:
                    hit_all += 1
                elif yolo_hits_frame == 0:
                    hit_zero += 1
                else:
                    hit_partial += 1

                if int(meta["n_padded_frames"]) == 0:
                    pad_none += 1
                if n_real >= n_frames:
                    pad_full += 1

                rec = {
                    "snippet_id": sid,
                    "pose_path": str(rel_pose).replace("\\", "/"),
                    "pose_mask_path": str(rel_mask).replace("\\", "/"),
                    "status": "done",
                    "error_msg": None,
                    "n_frames_extracted": n_frames,
                    "n_real_frames": meta["n_real_frames"],
                    "n_padded_frames": meta["n_padded_frames"],
                    "actual_fps": meta["actual_fps"],
                    "clip_duration_sec": meta["clip_duration_sec"],
                    "n_yolo_hits": yolo_hits_frame,
                    "latency_sec": round(elapsed, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec
                done_c += 1

            except Exception as ex:  # noqa: BLE001
                err_c += 1
                rec = {
                    "snippet_id": sid,
                    "pose_path": None,
                    "pose_mask_path": None,
                    "status": "error",
                    "error_msg": str(ex)[:500],
                    "n_frames_extracted": n_frames,
                    "n_real_frames": 0,
                    "n_padded_frames": n_frames,
                    "actual_fps": None,
                    "clip_duration_sec": None,
                    "n_yolo_hits": 0,
                    "latency_sec": round(time.perf_counter() - t_clip, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec

            processed_in_loop += 1
            pbar.update(1)
            pbar.set_postfix(done=done_c, err=err_c)

            if processed_in_loop % 100 == 0 and processed_in_loop > 0:
                elapsed_run = time.perf_counter() - t0_run
                rate = processed_in_loop / elapsed_run if elapsed_run > 0 else 0
                remaining = len(to_process_list) - processed_in_loop
                eta_sec = remaining / rate if rate > 0 else 0
                print(
                    f"\n[{processed_in_loop}/{len(to_process_list)}] done={done_c} err={err_c} | "
                    f"elapsed={elapsed_run/60:.1f}m | eta={eta_sec/3600:.1f}h"
                )

    except KeyboardInterrupt:
        print("\nInterrupted; flushing index...")
        flush_index()
        index_fp.close()
        print(f"Progress: done={done_c} err={err_c}")
        return 130
    finally:
        if not index_fp.closed:
            index_fp.close()

    # Reload index last-wins for manifest + validation
    index_by_id = load_index_last_wins(cache_index_path)

    n_ok, n_bad = update_final_manifest(REPO_ROOT, manifest_path, index_by_id)
    print(f"\npose fields added: {n_ok} rows")
    print(f"pose fields null: {n_bad} rows (not extracted or error)")

    validate_random_outputs(REPO_ROOT, index_by_id, n_frames, n_kp, normalize_xy)
    compare_v1_v2_sample(REPO_ROOT, legacy_manifest, index_by_id)

    # Summary (this run)
    total_train = len(train_ready)

    bytes_per = n_frames * n_kp * 3 * 4 + n_frames * 1
    mean_kb = bytes_per / 1024.0

    denom = max(1, len(to_process_list))
    pct_ok = 100.0 * done_c / denom
    pct_err = 100.0 * err_c / denom

    print("\n" + "═" * 56)
    print("POSE EXTRACTION SUMMARY")
    print("═" * 56)
    print(f"Training-ready snippets:     {total_train}")
    print(f"Successfully extracted:      {done_c}  ({pct_ok:.1f}% of queued this run)")
    print(f"Errors (this run):           {err_c}  ({pct_err:.1f}% of queued this run)")
    print(f"Skipped before run:          {n_skip}")
    print("")
    print("YOLO detection rate (this run):")
    if done_c > 0:
        dr = float(done_c)
        print(f"All real frames hit:       {hit_all} clips ({100.0 * hit_all / dr:.1f}% of successes)")
        print(f"Partial hits:              {hit_partial} clips ({100.0 * hit_partial / dr:.1f}%)")
        print(f"Zero detections:           {hit_zero} clips ({100.0 * hit_zero / dr:.1f}%)")
    else:
        print("  (no successful extractions this run)")
    print("")
    print("Frame padding (this run):")
    if done_c > 0:
        dr = float(done_c)
        print(f"No padded frames:          {pad_none} clips ({100.0 * pad_none / dr:.1f}% of successes)")
        print(f"Fully utilized ({n_frames} real): {pad_full} clips ({100.0 * pad_full / dr:.1f}%)")
    else:
        print("  (no successful extractions this run)")
    print("")
    print(f"Output:\nDirectory:  {output_dir.relative_to(REPO_ROOT)}/")
    print(f"File size:  mean ~{mean_kb:.1f} KB per clip (pose + mask)")
    print("")
    print(f"pose fields appended to final_dataset_v2.jsonl: {n_ok} rows")
    print("═" * 56)

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "YOLO is per-frame. ViTPose can micro-batch frames when vitpose_forward_batch > 1 "
            "(top-1 YOLO box per frame). decode_prefetch overlaps OpenCV decode of the next clip "
            "with GPU work. Do not run multiple clip workers on one GPU."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--update-manifest", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument(
        "--forward-batch",
        type=int,
        default=None,
        metavar="N",
        help="Override vitpose_forward_batch (1 = legacy multi-candidate path per frame)",
    )
    p.add_argument(
        "--decode-prefetch",
        type=int,
        default=None,
        metavar="N",
        help="Override decode_prefetch (1 = decode each clip on main thread)",
    )
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
