"""Dataset classes, frame utilities, and transforms.

Two dataset backends are available, selected automatically by config:

PrecomputedCatDataset (fast, recommended)
    Reads pre-cropped JPEGs from ``dataset/cropped_frames_p{N}/``.
    No OpenCV, no YOLO, no video decoding at runtime.
    Enabled when config has ``precomputed_frames_dir``.
    Random-sample ``frames_per_clip`` frames per clip in __getitem__.

CatBehaviorDataset (legacy, on-the-fly)
    Decodes video with OpenCV and optionally runs YOLO every call.
    Kept for compatibility; used when ``precomputed_frames_dir`` is absent.
"""

from __future__ import annotations

import logging
import pathlib
import random
import warnings
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger("cat_cnn")

_MISSING_VIDEO_LOGGED: set[str] = set()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CAT_CLASS_ID = 15


# ── Video path resolution (adapted from cat_pain_llm_eval.ipynb) ────────


def _usable_media_path(p: pathlib.Path) -> bool:
    if not p.exists():
        return False
    if p.suffix.lower() == ".npz":
        return False
    if "embeddings" in p.as_posix():
        return False
    return True


def resolve_video_path(stem: str, cfg: dict) -> str | None:
    """Search for the video file corresponding to a stem.

    Looks in manifest column paths, dataset/frames dir, and common video
    directories.  Returns full path string if found, None if not found.
    Logs once per missing video.
    """
    if not stem:
        return None

    extensions = cfg.get("video_extensions", [".mp4", ".avi", ".mov", ".mkv", ".webm"])

    frames_dir = PROJECT_ROOT / "data" / "dataset" / "frames" / "labeled" / stem
    if frames_dir.exists() and frames_dir.is_dir():
        return str(frames_dir)

    roots = [
        PROJECT_ROOT / "src" / "human_validation" / "video_audio_human_validation" / "cat",
        PROJECT_ROOT / "_archive" / "misc" / "dataset_snippets",
        PROJECT_ROOT / "_archive" / "misc" / "downloaded_snippets",
    ]
    for root in roots:
        for ext in extensions:
            candidate = root / f"{stem}{ext}"
            if _usable_media_path(candidate):
                return str(candidate)

    if stem not in _MISSING_VIDEO_LOGGED:
        _MISSING_VIDEO_LOGGED.add(stem)
        logger.debug(f"Video not found for stem: {stem}")
    return None


# ── Frame timestamp sampling (adapted from cat_pain_llm_eval.ipynb) ─────


def get_frame_timestamps(
    duration_sec: float, n_samples: int = 3,
    strategy: str = "uniform_inner",
) -> list[float]:
    """Sample n_samples timestamps from a clip.

    Strategies:
        uniform_inner: deterministic linspace in [10%, 90%] of duration.
        random_inner:  random sorted samples in [10%, 90%] — temporal jitter.
    """
    duration = float(duration_sec) if duration_sec and duration_sec > 0 else 1.0
    start = 0.1 * duration
    end = 0.9 * duration

    if strategy == "random_inner":
        if n_samples <= 1:
            t = np.random.uniform(start, end)
            return [round(float(t), 3)]
        ts = np.sort(np.random.uniform(start, end, size=n_samples))
        return [round(float(x), 3) for x in ts]

    # uniform_inner (default / validation)
    if n_samples <= 1:
        return [round((start + end) / 2.0, 3)]
    ts = np.linspace(start, end, n_samples)
    return [round(float(x), 3) for x in ts]


# ── Frame extraction (adapted from cat_pain_llm_eval.ipynb) ─────────────


def extract_frames_from_video(
    video_path: str, timestamps_sec: list[float]
) -> tuple[list[np.ndarray], list[int]]:
    """Extract frames at given timestamps.  Returns (frames_rgb, frame_indices)."""
    p = pathlib.Path(video_path)
    if not p.exists():
        return [], []

    if p.suffix.lower() == ".npz" or "embeddings" in p.as_posix():
        return [], []

    if p.is_dir():
        frame_files = sorted(p.glob("frame_*.jpg"))
        if not frame_files:
            return [], []
        meta_path = p / "meta.json"
        duration = None
        if meta_path.exists():
            import json
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                duration = float(meta.get("duration_sec", 0))
            except Exception:
                duration = None
        if not duration or duration <= 0:
            duration = max(len(frame_files) / 30.0, 1.0)

        sampled, indices = [], []
        for t in timestamps_sec:
            frac = min(max(float(t) / duration, 0.0), 1.0)
            idx = int(round(frac * (len(frame_files) - 1)))
            img = Image.open(frame_files[idx]).convert("RGB")
            sampled.append(np.array(img))
            indices.append(idx)
        return sampled, indices

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, indices = [], []
    for t in timestamps_sec:
        frame_idx = max(0, int(round(float(t) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        indices.append(frame_idx)
    cap.release()
    return frames, indices


# ── YOLO bbox crop (adapted from cat_pain_llm_eval.ipynb) ───────────────


def _expand_xyxy_pad_clamp(
    xmin: float, ymin: float, xmax: float, ymax: float,
    H: int, W: int, pad_frac: float = 0.20,
) -> tuple[int, int, int, int] | None:
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0.0 or bh <= 0.0:
        return None
    xmin -= pad_frac * bw
    xmax += pad_frac * bw
    ymin -= pad_frac * bh
    ymax += pad_frac * bh
    x0 = int(np.floor(max(0.0, xmin)))
    y0 = int(np.floor(max(0.0, ymin)))
    x1 = int(np.ceil(min(float(W), xmax)))
    y1 = int(np.ceil(min(float(H), ymax)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_frame_to_cat_bbox(
    frame_rgb: np.ndarray,
    yolo_model: Any,
    cfg: dict,
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    """Detect cat with YOLO, crop to padded bbox.

    Returns (cropped_frame_rgb, bbox_info).
    """
    yolo_conf = cfg.get("yolo_conf", 0.40)
    yolo_imgsz = cfg.get("yolo_imgsz", 640)
    bbox_pad = cfg.get("bbox_padding", 0.20)

    bbox_info: dict[str, Any] = {
        "detected": False, "confidence": 0.0,
        "bbox_xyxy": None, "padded_xyxy": None,
    }

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = yolo_model.predict(
            frame_bgr, imgsz=yolo_imgsz, conf=yolo_conf,
            classes=[CAT_CLASS_ID], verbose=False, device=device,
        )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return frame_rgb, bbox_info

    best_i = int(results.boxes.conf.argmax().item())
    xyxy = results.boxes.xyxy[best_i].cpu().numpy()
    conf_val = float(results.boxes.conf[best_i].item())
    xmin, ymin, xmax, ymax = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    H, W = frame_rgb.shape[0], frame_rgb.shape[1]
    box = _expand_xyxy_pad_clamp(xmin, ymin, xmax, ymax, H, W, bbox_pad)

    if box is None:
        bbox_info["detected"] = True
        bbox_info["confidence"] = conf_val
        bbox_info["bbox_xyxy"] = [xmin, ymin, xmax, ymax]
        return frame_rgb, bbox_info

    x0, y0, x1, y1 = box
    bbox_info.update({
        "detected": True, "confidence": conf_val,
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "padded_xyxy": [x0, y0, x1, y1],
    })
    return np.ascontiguousarray(frame_rgb[y0:y1, x0:x1]), bbox_info


# ── Transforms ──────────────────────────────────────────────────────────


def build_transforms(cfg: dict, is_train: bool) -> transforms.Compose:
    """Build train or val transforms.

    Train: Resize -> RandomHorizontalFlip -> ColorJitter -> RandomRotation
           -> ToTensor -> Normalize(ImageNet) -> RandomErasing
    Val:   Resize -> CenterCrop -> ToTensor -> Normalize(ImageNet)
    """
    size = cfg.get("image_size", 300)
    if is_train:
        t_list = [transforms.Resize((size, size))]
        if cfg.get("aug_horizontal_flip", True):
            t_list.append(transforms.RandomHorizontalFlip())
        if cfg.get("aug_color_jitter", True):
            t_list.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05,
            ))
        rot = cfg.get("aug_random_rotation", 0)
        if rot:
            t_list.append(transforms.RandomRotation(rot))
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        erase_p = cfg.get("aug_random_erasing", 0.0)
        if erase_p > 0:
            t_list.append(transforms.RandomErasing(p=erase_p))
        return transforms.Compose(t_list)

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Manifest loading ────────────────────────────────────────────────────


def _is_v2_manifest(raw_df: pd.DataFrame) -> bool:
    """Return True if the manifest looks like final_dataset_v2 (snippet_id / final_label_5 schema)."""
    return "snippet_id" in raw_df.columns and "final_label_5" in raw_df.columns


def _load_manifest_v2(raw_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adapt final_dataset_v2 rows to the internal schema expected by CatBehaviorDataset.

    v2 field → internal field:
        snippet_id      → stem
        final_label_5   → label  (then label_5)
        suitable_for_training == True → row filter
        video_path      → resolved directly (absolute or repo-relative)
        cat_id          → cat_id
        video_id        → video_id
        duration_sec    → duration_sec
    """
    df = raw_df.copy()

    # Filter to training-suitable rows
    if "suitable_for_training" in df.columns:
        df = df.loc[df["suitable_for_training"] == True].copy()
    if "final_label_5" in df.columns:
        df = df.loc[df["final_label_5"].notna() & (df["final_label_5"].astype(str).str.strip() != "")].copy()

    # Normalise field names
    df["stem"] = df["snippet_id"].astype(str)
    df["label"] = df["final_label_5"].astype(str)

    # Resolve video path: prefer manifest's own video_path column when file exists
    def _resolve_v2_video(row) -> str | None:
        vp = row.get("video_path")
        if vp:
            p = pathlib.Path(str(vp))
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return str(p)
        return resolve_video_path(str(row.get("stem", "")), cfg)

    df["video_path"] = df.apply(_resolve_v2_video, axis=1)

    for col in ["duration_sec", "frame_count", "cat_id", "video_id"]:
        if col not in df.columns:
            df[col] = None

    df["video_id"] = df["video_id"].fillna(df["stem"])
    df["cat_id"] = df["cat_id"].fillna(df["video_id"])

    return df


def load_manifest(cfg: dict) -> pd.DataFrame:
    """Load a manifest JSONL and return a DataFrame ready for CatBehaviorDataset.

    Supports two schemas:
    - **v1** (``final_dataset_with_cat_id.jsonl``): ``stem``, ``label``,
      ``logical_split == "labeled"`` filter.
    - **v2** (``final_dataset_v2.jsonl``): ``snippet_id``, ``final_label_5``,
      ``suitable_for_training == True`` filter. ``video_path`` column is used
      directly (all 5786 clips live in ``dataset/snippets_v2/``).

    Training labels come exclusively from the manifest's label field.
    The predictions_jsonl is NOT used here.
    """
    manifest_path = PROJECT_ROOT / cfg["manifest_path"]
    raw_df = pd.read_json(manifest_path, lines=True)

    classes_5 = cfg["classes_5"]
    binary_map = cfg["binary_map"]
    class_to_idx = {c: i for i, c in enumerate(classes_5)}
    binary_classes = sorted(set(binary_map.values()))
    binary_to_idx = {c: i for i, c in enumerate(binary_classes)}

    if _is_v2_manifest(raw_df):
        logger.info("Detected v2 manifest schema (snippet_id / final_label_5)")
        df = _load_manifest_v2(raw_df, cfg)
    else:
        if "logical_split" in raw_df.columns:
            df = raw_df.loc[raw_df["logical_split"] == "labeled"].copy()
        else:
            df = raw_df.copy()
            logger.warning("No 'logical_split' column — using all rows")
        df["video_path"] = df["stem"].apply(lambda s: resolve_video_path(s, cfg))
        for col in ["duration_sec", "frame_count", "pose_path", "clip_path",
                    "label_confidence", "video_id", "cat_id"]:
            if col not in df.columns:
                df[col] = None
        df["video_id"] = df["video_id"].fillna(df["stem"])
        df["cat_id"] = df["cat_id"].fillna(df["video_id"])

    if "label" not in df.columns:
        raise ValueError("Manifest has no usable label column")

    df["label_5"] = df["label"]
    invalid = df[~df["label_5"].isin(classes_5)]
    if len(invalid) > 0:
        logger.warning(
            f"Dropping {len(invalid)} rows with labels not in classes_5: "
            f"{invalid['label_5'].unique().tolist()}"
        )
        df = df[df["label_5"].isin(classes_5)].copy()

    df["label_5_idx"] = df["label_5"].map(class_to_idx)
    df["label_binary"] = df["label_5"].map(binary_map)
    df["label_binary_idx"] = df["label_binary"].map(binary_to_idx)

    summary_rows = []
    for cls in classes_5:
        count = int((df["label_5"] == cls).sum())
        binary = binary_map[cls]
        summary_rows.append(f"  {cls:<22s} | {count:>5d} | {binary}")
    summary = (
        "\n  Class                  | Count | Binary\n"
        "  ─────────────────────────────────────────\n"
        + "\n".join(summary_rows)
    )
    logger.info(f"Dataset: {len(df)} records | {len(classes_5)} classes")
    logger.info(summary)

    return df.reset_index(drop=True)


# ── Dataset class ───────────────────────────────────────────────────────


class CatBehaviorDataset(Dataset):
    """Cat video clip dataset with YOLO bbox crop and multi-frame stacking.

    Returns dict with frames tensor, 5-class and binary labels, and metadata.
    """

    def __init__(
        self,
        records: list[dict],
        cfg: dict,
        transform: transforms.Compose,
        is_train: bool,
        yolo_model: Any | None = None,
    ):
        self.records = records
        self.cfg = cfg
        self.transform = transform
        self.is_train = is_train
        self.yolo_model = yolo_model
        self.frames_per_clip = cfg.get("frames_per_clip", 3)
        self.use_bbox = cfg.get("use_bbox_crop", True)
        self.image_size = cfg.get("image_size", 300)
        self._yolo_device = "cpu"

    def set_yolo_device(self, device: str) -> None:
        self._yolo_device = device

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        stem = rec.get("stem", "")
        video_id = rec.get("video_id", stem)
        label_5 = int(rec["label_5_idx"])
        label_binary = int(rec["label_binary_idx"])
        video_path = rec.get("video_path")
        duration = rec.get("duration_sec") or 1.0

        strategy = "random_inner" if self.is_train else "uniform_inner"
        timestamps = get_frame_timestamps(duration, self.frames_per_clip, strategy=strategy)
        frame_indices: list[int] = []
        bbox_used: list[bool] = []
        frame_tensors: list[torch.Tensor] = []

        if video_path:
            raw_frames, raw_indices = extract_frames_from_video(
                video_path, timestamps,
            )
        else:
            raw_frames, raw_indices = [], []

        if not raw_frames:
            if stem not in _MISSING_VIDEO_LOGGED:
                _MISSING_VIDEO_LOGGED.add(stem)
                logger.warning(f"No frames extracted for stem={stem}")
            zero = torch.zeros(
                self.frames_per_clip, 3, self.image_size, self.image_size,
            )
            return {
                "frames": zero,
                "label_5": torch.tensor(label_5, dtype=torch.long),
                "label_binary": torch.tensor(label_binary, dtype=torch.long),
                "stem": stem,
                "video_id": video_id,
                "frame_indices": [0] * self.frames_per_clip,
                "frame_timestamps": timestamps[:self.frames_per_clip],
                "bbox_used": [False] * self.frames_per_clip,
            }

        while len(raw_frames) < self.frames_per_clip:
            raw_frames.append(raw_frames[-1])
            raw_indices.append(raw_indices[-1])

        for i in range(self.frames_per_clip):
            frame = raw_frames[i]
            used_bbox = False

            if self.use_bbox and self.yolo_model is not None:
                cropped, info = crop_frame_to_cat_bbox(
                    frame, self.yolo_model, self.cfg, self._yolo_device,
                )
                frame = cropped
                used_bbox = info["detected"]

            pil_img = Image.fromarray(frame)
            tensor = self.transform(pil_img)
            frame_tensors.append(tensor)
            frame_indices.append(raw_indices[i])
            bbox_used.append(used_bbox)

        frames = torch.stack(frame_tensors, dim=0)

        return {
            "frames": frames,
            "label_5": torch.tensor(label_5, dtype=torch.long),
            "label_binary": torch.tensor(label_binary, dtype=torch.long),
            "stem": stem,
            "video_id": video_id,
            "frame_indices": frame_indices,
            "frame_timestamps": timestamps[:self.frames_per_clip],
            "bbox_used": bbox_used,
        }


# ── PrecomputedCatDataset ────────────────────────────────────────────────────


class PrecomputedCatDataset(Dataset):
    """Fast dataset that reads pre-cropped JPEGs from disk.

    Expects the directory layout produced by scripts/02_crop_frames.py::

        {frames_root}/{clip_id}/frame_01.jpg … frame_15.jpg

    In ``__getitem__`` it randomly samples ``frames_per_clip`` (default 3)
    frames from the available files (training) or takes the first N evenly
    spaced ones (validation).  No OpenCV, no YOLO, no video decoding.
    """

    def __init__(
        self,
        records: list[dict],
        cfg: dict,
        transform: transforms.Compose,
        is_train: bool,
        frames_root: pathlib.Path,
    ):
        self.records = records
        self.cfg = cfg
        self.transform = transform
        self.is_train = is_train
        self.frames_root = pathlib.Path(frames_root)
        self.frames_per_clip = int(cfg.get("frames_per_clip", 3))
        self.image_size = int(cfg.get("image_size", 300))

        # Pre-index available frames per clip at init time (avoids repeated
        # glob calls in __getitem__ which would be slow with num_workers > 0)
        self._clip_frames: dict[str, list[pathlib.Path]] = {}
        for rec in records:
            stem = rec.get("stem", "")
            clip_dir = self.frames_root / stem
            if clip_dir.is_dir():
                jpgs = sorted(clip_dir.glob("frame_*.jpg"))
                self._clip_frames[stem] = jpgs
            else:
                self._clip_frames[stem] = []

        missing = sum(1 for v in self._clip_frames.values() if not v)
        if missing:
            logger.warning(
                "%d/%d clips have no pre-computed frames in %s. "
                "Run scripts/02_crop_frames.py first.",
                missing, len(records), self.frames_root,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        stem = rec.get("stem", "")
        video_id = rec.get("video_id", stem)
        label_5 = int(rec["label_5_idx"])
        label_binary = int(rec["label_binary_idx"])

        all_frames = self._clip_frames.get(stem, [])

        frame_tensors: list[torch.Tensor] = []
        selected_names: list[str] = []

        if all_frames:
            n_avail = len(all_frames)
            n_need = self.frames_per_clip

            if self.is_train:
                # Random sample without replacement (with replacement if fewer available)
                replace = n_avail < n_need
                chosen = random.choices(all_frames, k=n_need) if replace else random.sample(all_frames, n_need)
            else:
                # Evenly-spaced deterministic selection
                if n_avail >= n_need:
                    indices = [int(round(i * (n_avail - 1) / max(n_need - 1, 1)))
                               for i in range(n_need)]
                    chosen = [all_frames[i] for i in indices]
                else:
                    chosen = all_frames + [all_frames[-1]] * (n_need - n_avail)

            for fp in chosen:
                img = Image.open(str(fp)).convert("RGB")
                frame_tensors.append(self.transform(img))
                selected_names.append(fp.name)
        else:
            # No pre-computed frames — return zeros and warn once
            if stem not in _MISSING_VIDEO_LOGGED:
                _MISSING_VIDEO_LOGGED.add(stem)
                logger.warning("No pre-computed frames for clip: %s", stem)
            zero = torch.zeros(self.frames_per_clip, 3, self.image_size, self.image_size)
            return {
                "frames": zero,
                "label_5": torch.tensor(label_5, dtype=torch.long),
                "label_binary": torch.tensor(label_binary, dtype=torch.long),
                "stem": stem,
                "video_id": video_id,
                "frame_indices": list(range(self.frames_per_clip)),
                "frame_timestamps": [0.0] * self.frames_per_clip,
                "bbox_used": [False] * self.frames_per_clip,
            }

        return {
            "frames": torch.stack(frame_tensors, dim=0),
            "label_5": torch.tensor(label_5, dtype=torch.long),
            "label_binary": torch.tensor(label_binary, dtype=torch.long),
            "stem": stem,
            "video_id": video_id,
            "frame_indices": [int(n.split("_")[1].split(".")[0]) for n in selected_names],
            "frame_timestamps": [0.0] * self.frames_per_clip,
            "bbox_used": [True] * self.frames_per_clip,
        }


# ── Factory ──────────────────────────────────────────────────────────────────


def build_dataset(
    records: list[dict],
    cfg: dict,
    is_train: bool,
) -> Dataset:
    """Return the right Dataset subclass based on config.

    If ``precomputed_frames_dir`` is set in config and the directory exists,
    uses ``PrecomputedCatDataset`` (fast, no YOLO).  Otherwise falls back to
    ``CatBehaviorDataset`` (on-the-fly video decode + optional YOLO).

    The caller is responsible for passing ``yolo_model`` to
    ``CatBehaviorDataset`` when needed; this factory only handles the
    pre-computed path.
    """
    tf = build_transforms(cfg, is_train=is_train)
    precomp = cfg.get("precomputed_frames_dir")
    if precomp:
        frames_root = PROJECT_ROOT / str(precomp)
        if frames_root.is_dir():
            logger.info(
                "Using PrecomputedCatDataset from %s (%s)",
                frames_root, "train" if is_train else "val",
            )
            return PrecomputedCatDataset(records, cfg, tf, is_train, frames_root)
        else:
            logger.warning(
                "precomputed_frames_dir=%s does not exist — "
                "falling back to on-the-fly CatBehaviorDataset",
                frames_root,
            )
    return CatBehaviorDataset(records, cfg, tf, is_train, yolo_model=None)
