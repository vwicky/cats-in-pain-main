"""
Pose normalization, augmentation, and dataset classes.

Normalization pipeline (applied in this exact order):
  1. Load raw (35, 17, 3) float32 array
  2. Confidence gating: zero out x,y for keypoints where conf < threshold
  3. Clamp x,y to clamp_xy range (catches extraction bugs where y > 1.0)
  4. Root-relative centering: subtract keypoint[root_joint] from all x,y
  5. Scale normalization: divide all x,y by mean_bone_length
     mean_bone_length = mean over frames of euclidean distance between
     scale_joints[0] and scale_joints[1] (left/right shoulder)
     If mean_bone_length < 1e-6 (all zeros): set to 1.0 (no scaling)
  6. Append velocity channels if use_kinematics=True:
     velocity[t] = pose[t] - pose[t-1] for t>0, velocity[0] = 0
     Result shape: (35, 17, 6) — [x, y, conf, vx, vy, v_conf_delta]
  7. Apply pose_mask: zero out padded frames (mask==False) after all transforms

Historical note: legacy ``finetuning_after_ssl.SweepDataset.normalize_pose`` uses a
simpler root + mean-distance scale; this module implements the v2 spec (shoulder bone
length, clamp, conf gating).

``append_pose_kinematics`` (below) appends temporal deltas for x, y, and confidence
Δ — matching [x, y, conf, vx, vy, v_conf_delta].
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_loading import  get_multiclass_class_names
from deeplabcut_pose_io import  load_dlc_h5_tensor
from models.superanimal_quadruped_stgcn_graph import  FLIP_PAIRS as QUADRUPED39_FLIP_PAIRS
from models.superanimal_quadruped_stgcn_graph import  NUM_KEYPOINTS as NUM_KEYPOINTS_QUADRUPED

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def append_pose_kinematics(pose: np.ndarray) -> np.ndarray:
    """
    Append per-joint temporal velocity (root-normalized space) to pose (T, V, 3) -> (T, V, 6).
    """
    if pose.shape[-1] != 3:
        return pose
    t, v, c = pose.shape
    vel = np.zeros((t, v, c), dtype=np.float32)
    if t > 1:
        vel[1:] = pose[1:].astype(np.float32) - pose[:-1].astype(np.float32)
    return np.concatenate([pose.astype(np.float32), vel], axis=-1)

_WARNED_KP_TRUNCATE = False

_WXH_RE = re.compile(r"(\d+)\s*[x×]\s*(\d+)", re.IGNORECASE)


def _parse_resolution_wh(record: dict) -> tuple[float, float] | None:
    gr = record.get("gpt_resolution")
    if gr is None:
        return None
    s = str(gr).strip()
    m = _WXH_RE.search(s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def detect_and_renormalize_v1(pose: np.ndarray, record: dict) -> np.ndarray:
    """
    If max(x) > 2.0, treat as pixel-ish / unnormalized coordinates and divide
    x,y by (W,H) from gpt_resolution when parseable as WxH, else by per-clip
    max(|x|), max(|y|) over frames and joints.
    """
    p = pose.astype(np.float32, copy=True)
    if p.size == 0:
        return p
    xy = p[..., :2]
    if float(np.nanmax(xy[..., 0])) <= 2.0:
        return p

    wh = _parse_resolution_wh(record)
    if wh is not None:
        w, h = wh
        if w > 0 and h > 0:
            p[..., 0] /= float(w)
            p[..., 1] /= float(h)
            return p

    mx = float(np.nanmax(np.abs(xy[..., 0]))) or 1.0
    my = float(np.nanmax(np.abs(xy[..., 1]))) or 1.0
    p[..., 0] /= mx
    p[..., 1] /= my
    return p


def normalize_pose_array(
    pose: np.ndarray,
    cfg: dict,
    *,
    drop_low_confidence_frames: bool | None = None,
) -> np.ndarray:
    """Apply steps 2–5 (no load, no kinematics, no mask) on (T, J, 3)."""
    norm_cfg = cfg["normalization"]
    root_j = int(norm_cfg["root_joint"])
    sj0, sj1 = int(norm_cfg["scale_joints"][0]), int(norm_cfg["scale_joints"][1])
    lo, hi = float(norm_cfg["clamp_xy"][0]), float(norm_cfg["clamp_xy"][1])
    thr = float(norm_cfg["confidence_threshold"])
    drop = norm_cfg["drop_low_confidence_frames"] if drop_low_confidence_frames is None else drop_low_confidence_frames

    p = pose.astype(np.float32, copy=True)
    t, j, c = p.shape
    conf = p[..., 2]

    mask_kp = conf >= thr
    p[~mask_kp, 0] = 0.0
    p[~mask_kp, 1] = 0.0

    p[..., 0] = np.clip(p[..., 0], lo, hi)
    p[..., 1] = np.clip(p[..., 1], lo, hi)

    root = p[:, root_j : root_j + 1, :2]
    p[..., :2] = p[..., :2] - root

    d = np.linalg.norm(p[:, sj0, :2] - p[:, sj1, :2], axis=-1)
    mean_bone = float(np.mean(d)) if d.size else 0.0
    if mean_bone < 1e-6:
        mean_bone = 1.0
    p[..., 0] /= mean_bone
    p[..., 1] /= mean_bone

    if drop:
        low_f = np.any(conf < thr, axis=1)
        p[low_f] = 0.0

    return p


def load_and_normalize_pose(
    pose_path: str,
    mask_path: str | None,
    cfg: dict,
    record: dict | None = None,
    *,
    use_kinematics: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pose .npy file and apply full normalization pipeline.

    Returns:
      pose: float32 array of shape (n_frames, n_keypoints, C)
            C=3 if use_kinematics=False, C=6 if True
      mask: bool array of shape (n_frames,)
            True = real frame, False = padded
            If mask_path is None: all True (assume all real)

    On load/shape/file errors returns zeros array + all-False mask and logs a warning.
    Configuration errors (e.g. missing ``cfg["normalization"]``) are logged and
    re-raised so callers never get a silent all-zero tensor from a mis-built config.
    """
    n_frames = int(cfg["data"]["n_frames"])
    n_kp = int(cfg["data"]["n_keypoints"])
    n_ch = int(cfg["data"]["n_channels"])
    record = record or {}

    try:
        full = REPO_ROOT / pose_path if not Path(pose_path).is_absolute() else Path(pose_path)
        raw = np.load(full, allow_pickle=False).astype(np.float32, copy=False)
        if raw.ndim != 3 or raw.shape[2] < n_ch:
            raise ValueError(f"bad pose shape {raw.shape}")
        global _WARNED_KP_TRUNCATE
        if raw.shape[1] > n_kp:
            if not _WARNED_KP_TRUNCATE:
                logger.warning(
                    "Pose arrays have more keypoints than config (e.g. SuperAnimal 39 vs %d); "
                    "using first %d joints only. Prefer paths from pose_extraction_index / pose/v2.",
                    n_kp,
                    n_kp,
                )
                _WARNED_KP_TRUNCATE = True
            raw = np.ascontiguousarray(raw[:, :n_kp, :])
        elif raw.shape[1] < n_kp:
            raise ValueError(f"bad pose shape {raw.shape} (need at least {n_kp} keypoints)")

        raw = raw[:, :, :n_ch].copy()
        raw = detect_and_renormalize_v1(raw, record)

        if mask_path:
            mf = REPO_ROOT / mask_path if not Path(mask_path).is_absolute() else Path(mask_path)
            m = np.load(mf, allow_pickle=False)
            mask = m.astype(bool).reshape(-1)
        else:
            mask = np.ones(raw.shape[0], dtype=bool)

        T0 = raw.shape[0]
        if len(mask) < T0:
            mask = np.pad(mask, (0, T0 - len(mask)), constant_values=True)
        else:
            mask = mask[:T0]

        T = raw.shape[0]
        if T < n_frames:
            pad = np.zeros((n_frames - T, n_kp, n_ch), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
            mask_pad = np.zeros(n_frames - T, dtype=bool)
            mask = np.concatenate([mask[:T], mask_pad])
        elif T > n_frames:
            raw = raw[:n_frames]
            mask = mask[:n_frames]

        p = normalize_pose_array(raw, cfg)
        if use_kinematics:
            p = append_pose_kinematics(p)

        mfull = np.asarray(mask[:n_frames], dtype=bool)
        if mfull.size < n_frames:
            mfull = np.pad(mfull, (0, n_frames - mfull.size), constant_values=False)

        p[~mfull] = 0.0
        return p.astype(np.float32), mfull
    except (KeyError, TypeError) as e:
        logger.error(
            "load_and_normalize_pose: invalid cfg or unexpected types for %s: %s",
            pose_path,
            e,
        )
        raise
    except Exception as e:
        logger.warning("load_and_normalize_pose failed for %s: %s", pose_path, e)
        c_out = 6 if use_kinematics else 3
        return (
            np.zeros((n_frames, n_kp, c_out), dtype=np.float32),
            np.zeros(n_frames, dtype=bool),
        )


_FLIP_PAIRS_17 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def compute_bone_lengths(
    pose_txj3: np.ndarray,
    edges: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    Euclidean bone length per frame for each edge. Shape ``(T, E)`` using ``pose[..., :2]``.

    ``edges`` defaults to ``SKELETON_EDGES`` from ``quadruped_skeleton_spec`` (ST-GCN graph).
    """
    from quadruped_skeleton_spec import  SKELETON_EDGES

    ed = list(SKELETON_EDGES if edges is None else edges)
    xy = pose_txj3[..., :2].astype(np.float64, copy=False)
    t = xy.shape[0]
    if t == 0:
        return np.zeros((0, len(ed)), dtype=np.float64)
    out = np.zeros((t, len(ed)), dtype=np.float64)
    for e, (a, b) in enumerate(ed):
        d = xy[:, a, :] - xy[:, b, :]
        out[:, e] = np.linalg.norm(d, axis=-1)
    return out


def _temporal_speed_resample_pose(
    pose: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    aug: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the real (prefix) segment in time by a random speed factor, then pad to ``n_frames``."""
    if not bool(aug.get("temporal_speed_jitter", False)):
        return pose, mask
    lo = float(aug.get("temporal_speed_min", 0.8))
    hi = float(aug.get("temporal_speed_max", 1.2))
    nf, n_j, _c = pose.shape
    m = np.asarray(mask, dtype=bool)
    tr = int(np.sum(m))
    if tr < 2:
        return pose, mask
    seg = pose[:tr].astype(np.float32, copy=False)
    factor = float(rng.uniform(lo, hi))
    new_tr = max(2, int(round(tr / max(factor, 1e-6))))
    new_tr = min(new_tr, nf)
    t_old = np.arange(tr, dtype=np.float64)
    t_new = np.linspace(0.0, float(tr - 1), new_tr)
    out = np.zeros_like(pose)
    for ji in range(n_j):
        for cc in (0, 1, 2):
            out[:new_tr, ji, cc] = np.interp(t_new, t_old, seg[:, ji, cc]).astype(np.float32)
    m2 = np.zeros(nf, dtype=bool)
    m2[:new_tr] = True
    out[~m2] = 0.0
    return out, m2


def _rotate_pose_xy_plane(pose: np.ndarray, rng: np.random.Generator, aug: dict) -> np.ndarray:
    """Small in-plane rotation on x,y (after normalization)."""
    md = float(aug.get("rotation_augment_max_deg", 0.0) or 0.0)
    if md <= 0.0:
        return pose
    p = pose.copy()
    deg = float(rng.uniform(-md, md))
    rad = float(np.deg2rad(deg))
    co = float(np.cos(rad))
    si = float(np.sin(rad))
    r = np.array([[co, -si], [si, co]], dtype=np.float32)
    xy = p[:, :, :2].astype(np.float32, copy=False)
    p[:, :, :2] = np.matmul(xy, r.T)
    return p


def _apply_train_augmentation(
    pose: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    n_keypoints: int = 17,
    flip_pairs: list[tuple[int, int]] | None = None,
    enable_horizontal_flip: bool = True,
    aug_cfg: dict | None = None,
) -> np.ndarray:
    """
    Augment a copy of ``pose``. ``aug_cfg`` (subset of ``cfg['augmentation']``) controls extras:

    - ``gaussian_noise_sigma`` (default 0.008): std on x,y for real frames.
    - ``confidence_weighted_gaussian_noise``: scale noise by ``(1 - likelihood)`` on x,y.
    - ``temporal_speed_jitter`` + ``temporal_speed_min`` / ``temporal_speed_max``: resample real segment.
    - ``rotation_augment_max_deg``: small in-plane rotation on x,y.
    - ``temporal_jitter``: roll time axis by ±``temporal_jitter_max_shift`` (default 4).
    - ``joint_dropout_prob`` + ``joint_dropout_conf_threshold``: zero sparse low-confidence joints.
    - ``legacy_random_joint_dropout`` (default True if ``joint_dropout_prob`` is 0): old 2–5 joint + 4-frame dropout.
    """
    if flip_pairs is None:
        flip_pairs = _FLIP_PAIRS_17 if n_keypoints == 17 else []
    a = aug_cfg if aug_cfg is not None else {}
    p = pose.copy()
    m = np.asarray(mask, dtype=bool)
    real_idx = np.flatnonzero(m)
    if real_idx.size == 0:
        return p

    p, m = _temporal_speed_resample_pose(p, m, rng, a)

    sig_n = float(a.get("gaussian_noise_sigma", 0.008))
    m3 = m[:, np.newaxis, np.newaxis].astype(np.float32)
    if bool(a.get("confidence_weighted_gaussian_noise", False)):
        lh = np.clip(p[..., 2:3], 0.0, 1.0).astype(np.float32, copy=False)
        scale = (1.0 - lh) * m3
        p[:, :, :2] += (rng.normal(0.0, sig_n, size=p[:, :, :2].shape) * scale).astype(np.float32)
    else:
        p[:, :, :2] += (rng.normal(0.0, sig_n, size=p[:, :, :2].shape) * m3).astype(np.float32)

    if bool(a.get("temporal_jitter", False)) and p.shape[0] > 0:
        ms = max(1, int(a.get("temporal_jitter_max_shift", 4)))
        sh = int(rng.integers(-ms, ms + 1))
        if sh != 0:
            p = np.roll(p, shift=sh, axis=0)
            m = np.roll(m, shift=sh, axis=0)

    p = _rotate_pose_xy_plane(p, rng, a)

    jp = float(a.get("joint_dropout_prob", 0.0) or 0.0)
    jc_thr = float(a.get("joint_dropout_conf_threshold", 0.30))
    if jp > 0 and n_keypoints > 1:
        for j in range(1, n_keypoints):
            vis = p[m, j, 2]
            mean_c = float(np.nanmean(vis)) if vis.size else 1.0
            if mean_c < jc_thr and rng.random() < jp:
                p[:, j, :] = 0.0
    elif bool(a.get("legacy_random_joint_dropout", True)):
        n_j = int(rng.integers(2, 5))
        upper = max(1, n_keypoints - 1)
        n_pick = min(n_j, upper)
        joints = rng.choice(np.arange(1, upper + 1), size=n_pick, replace=False)
        p[:, joints, :] = 0.0

        ridx = np.flatnonzero(m)
        if ridx.size >= 4:
            pick = rng.choice(ridx, size=4, replace=False)
            p[pick] = 0.0

    if enable_horizontal_flip and flip_pairs and rng.random() < 0.5:
        # Spec: x -> 1-x after normalization (unusual for root-relative coords; negation is common).
        p[..., 0] = 1.0 - p[..., 0]
        for x, y in flip_pairs:
            if x < p.shape[1] and y < p.shape[1]:
                p[:, [x, y]] = p[:, [y, x]]
        p[~m] = 0.0
    else:
        p[~m] = 0.0
    return p


def temporal_pad_truncate(
    pose: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    pose: (T, J, C) → (n_frames, J, C) with zero padding or truncation.
    mask: (n_frames,) True for real (non-padded) rows.
    """
    if pose.ndim != 3:
        raise ValueError(f"expected (T,J,C), got {pose.shape}")
    t, j, c = pose.shape
    mask = np.zeros(n_frames, dtype=bool)
    if t == 0:
        return np.zeros((n_frames, j, c), dtype=np.float32), mask
    if t >= n_frames:
        mask[:] = True
        return pose[:n_frames].astype(np.float32, copy=False), mask
    out = np.zeros((n_frames, j, c), dtype=np.float32)
    out[:t] = pose.astype(np.float32, copy=False)
    mask[:t] = True
    return out, mask


def load_and_normalize_dlc_h5(
    h5_path: str,
    cfg: dict,
    record: dict | None = None,
    *,
    use_kinematics: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load DeepLabCut HDF5, pad/truncate to n_frames, apply same normalization as ViT poses.
    Returns (pose, mask) like load_and_normalize_pose.
    """
    n_frames = int(cfg["data"]["n_frames"])
    n_kp = int(cfg["data"]["n_keypoints"])
    n_ch = int(cfg["data"]["n_channels"])
    record = record or {}

    try:
        full = REPO_ROOT / h5_path if not Path(h5_path).is_absolute() else Path(h5_path)
        raw_full, _meta = load_dlc_h5_tensor(full, expected_joints=n_kp)
        if raw_full.shape[1] != n_kp:
            raise ValueError(f"DLC tensor J={raw_full.shape[1]} != config n_keypoints={n_kp}")

        raw = raw_full[:, :, : min(n_ch, raw_full.shape[2])].copy()
        if raw.shape[2] < n_ch:
            padc = np.zeros((raw.shape[0], raw.shape[1], n_ch - raw.shape[2]), dtype=np.float32)
            raw = np.concatenate([raw, padc], axis=2)

        raw = detect_and_renormalize_v1(raw, record)
        raw, mask = temporal_pad_truncate(raw, n_frames)

        p = normalize_pose_array(raw, cfg)
        if use_kinematics:
            p = append_pose_kinematics(p)

        mfull = np.asarray(mask[:n_frames], dtype=bool)
        if mfull.size < n_frames:
            mfull = np.pad(mfull, (0, n_frames - mfull.size), constant_values=False)

        p[~mfull] = 0.0
        return p.astype(np.float32), mfull
    except Exception as e:
        logger.warning("load_and_normalize_dlc_h5 failed for %s: %s", h5_path, e)
        c_out = 6 if use_kinematics else 3
        return (
            np.zeros((n_frames, n_kp, c_out), dtype=np.float32),
            np.zeros(n_frames, dtype=bool),
        )


# Bump when on-disk tensor layout or semantics change (forces re-read of .pt sidecars).
_DLC_POSE_CACHE_FILE_VERSION = 1
_VIT_POSE_CACHE_FILE_VERSION = 1


def _dlc_pose_record_cache_sig(rec: dict) -> str:
    """Manifest fields that affect ``detect_and_renormalize_v1`` / normalized tensors."""
    gr = rec.get("gpt_resolution")
    return str(gr).strip() if gr is not None else ""


def _dlc_pose_cache_digest(cfg: dict, *, use_kinematics: bool) -> str:
    """Short hash so RAM/disk entries invalidate when preprocessing changes."""
    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    norm = cfg.get("normalization", {}) if isinstance(cfg.get("normalization"), dict) else {}
    blob = {
        "file_v": _DLC_POSE_CACHE_FILE_VERSION,
        "n_frames": data.get("n_frames"),
        "n_keypoints": data.get("n_keypoints"),
        "n_channels": data.get("n_channels"),
        "normalization": norm,
        "use_kinematics": bool(use_kinematics),
    }
    raw = json.dumps(blob, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _vit_pose_cache_digest(cfg: dict, *, use_kinematics: bool) -> str:
    """Hash for :class:`PoseDataset` (ViT ``.npy`` + ``load_and_normalize_pose`` pipeline)."""
    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    norm = cfg.get("normalization", {}) if isinstance(cfg.get("normalization"), dict) else {}
    zj = data.get("inference_zero_joint_indices")
    blob = {
        "kind": "vit_npy",
        "file_v": _VIT_POSE_CACHE_FILE_VERSION,
        "n_frames": data.get("n_frames"),
        "n_keypoints": data.get("n_keypoints"),
        "n_channels": data.get("n_channels"),
        "normalization": norm,
        "use_kinematics": bool(use_kinematics),
    }
    if zj is not None:
        blob["inference_zero_joint_indices"] = [int(x) for x in zj]
    raw = json.dumps(blob, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _vit_disk_cache_path(disk_root: Path, digest: str, snippet_id: str) -> Path:
    return disk_root / digest / f"{_sanitize_dlc_cache_filename(snippet_id)}.pt"


def _sanitize_dlc_cache_filename(snippet_id: str, max_len: int = 200) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(snippet_id), flags=re.UNICODE).strip("_")
    return (s[:max_len] if s else "unknown_snippet")


def _dlc_disk_cache_path(disk_root: Path, digest: str, snippet_id: str) -> Path:
    return disk_root / digest / f"{_sanitize_dlc_cache_filename(snippet_id)}.pt"


def _torch_load_pose_cache(path: Path) -> dict:
    """Load local cache files (trusted paths only)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # Older PyTorch / dict payloads: full unpickler for repo-local ``.pt`` sidecars.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _torch_save_pose_cache_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights using sklearn convention:
      weight[c] = n_samples / (n_classes * count[c])
    Returns FloatTensor of shape (n_classes,).
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    w = np.ones(n_classes, dtype=np.float64)
    for c in range(n_classes):
        cnt = int((labels == c).sum())
        if cnt > 0:
            w[c] = len(labels) / (n_classes * cnt)
        else:
            w[c] = 0.0
    mx = w.max()
    if mx > 0:
        w = np.where(w > 0, w, mx)
    return torch.tensor(w, dtype=torch.float32)


class PoseDataset(Dataset):
    """
    PyTorch Dataset for pose sequences (ViT ``.npy`` + optional mask ``.npy``).

    Optional ``data.pose_cache`` (see ``config_p4_*.yaml``) — same keys as
    :class:`DLCPoseDataset`:

    - ``backend: ram`` (default) — in-process cache of normalized (pose, mask) after
      the first read (large speedup from epoch 2+; same pattern as DLC RAM cache).
    - ``backend: disk`` — ``.pt`` under ``disk_dir/{digest}/``; survives restarts.
    - ``backend: off`` — always ``np.load`` + normalize (slow for many epochs).
    """

    _RAM_NPY: ClassVar[dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]]] = {}
    _RAM_NPY_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, records, cfg, is_train=False, use_kinematics=True):
        self.records = list(records)
        self.cfg = cfg
        self.is_train = is_train
        self.use_kinematics = use_kinematics
        self._rng = np.random.default_rng()

        pc = cfg.get("data", {}).get("pose_cache") if isinstance(cfg.get("data"), dict) else None
        if isinstance(pc, dict):
            backend = str(pc.get("backend", "ram")).strip().lower()
        else:
            backend = "ram"
        if backend in ("false", "none", "0", "off", ""):
            backend = "off"
        if backend not in ("off", "ram", "disk"):
            logger.warning("Unknown data.pose_cache.backend %r — using off for PoseDataset", backend)
            backend = "off"
        self._cache_backend = backend
        self._digest = _vit_pose_cache_digest(cfg, use_kinematics=use_kinematics)
        self._disk_root: Path | None = None
        if backend == "disk":
            raw = (pc or {}).get("disk_dir", "video/pose-models/cache/vit_poses")
            dr = Path(str(raw).strip())
            self._disk_root = dr if dr.is_absolute() else REPO_ROOT / dr

    def __len__(self):
        return len(self.records)

    def _resolve_pose_path(self, pose_path: str) -> Path:
        p = Path(pose_path)
        return p if p.is_absolute() else REPO_ROOT / p

    def _load_pose_mask_npy(self, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        pose_path = str(rec.get("pose_path", "") or "")
        mask_m = rec.get("pose_mask_path")
        mask_path = str(mask_m) if mask_m and str(mask_m).strip() else ""
        rec_sig = _dlc_pose_record_cache_sig(rec)
        pose_f = self._resolve_pose_path(pose_path)
        if not pose_path or not pose_f.is_file():
            p, m = load_and_normalize_pose(
                pose_path, mask_m if mask_m and str(m).strip() else None, self.cfg, record=dict(rec), use_kinematics=self.use_kinematics
            )
            return p, m
        if mask_path:
            mf = Path(mask_m) if Path(str(mask_m)).is_absolute() else REPO_ROOT / str(mask_m)
        else:
            mf = None
        mask_key = str(mf.resolve()) if mf is not None and mf.is_file() else ""
        key = (self._digest, str(pose_f.resolve()), mask_key, rec_sig)
        if self._cache_backend == "off":
            return load_and_normalize_pose(
                pose_path,
                mask_m if mask_m and str(mask_m).strip() else None,
                self.cfg,
                record=dict(rec),
                use_kinematics=self.use_kinematics,
            )
        if self._cache_backend == "ram":
            with self._RAM_NPY_LOCK:
                hit = self._RAM_NPY.get(key)
            if hit is not None:
                pose, mask = hit
                return pose.copy(), mask.copy()
            p, m = load_and_normalize_pose(
                pose_path,
                mask_m if mask_m and str(mask_m).strip() else None,
                self.cfg,
                record=dict(rec),
                use_kinematics=self.use_kinematics,
            )
            with self._RAM_NPY_LOCK:
                if key not in self._RAM_NPY:
                    self._RAM_NPY[key] = (p.copy(), m.copy())
            return p, m
        assert self._disk_root is not None
        sid = str(rec.get("snippet_id", "row"))
        pt = _vit_disk_cache_path(self._disk_root, self._digest, sid)
        if pt.is_file():
            try:
                blob = _torch_load_pose_cache(pt)
                if (
                    isinstance(blob, dict)
                    and blob.get("kind") == "vit_npy"
                    and blob.get("digest") == self._digest
                    and int(blob.get("file_v", -1)) == _VIT_POSE_CACHE_FILE_VERSION
                    and blob.get("resolved_pose") == str(pose_f.resolve())
                    and blob.get("mask_key", "") == mask_key
                    and blob.get("record_sig", "") == rec_sig
                    and "pose" in blob
                    and "mask" in blob
                ):
                    p = np.asarray(blob["pose"], dtype=np.float32)
                    m = np.asarray(blob["mask"], dtype=np.bool_)
                    return p, m
            except Exception as e:
                logger.warning("Corrupt vit pose cache %s (%s) — rebuilding", pt, e)
        p, m = load_and_normalize_pose(
            pose_path,
            mask_m if mask_m and str(mask_m).strip() else None,
            self.cfg,
            record=dict(rec),
            use_kinematics=self.use_kinematics,
        )
        payload = {
            "kind": "vit_npy",
            "digest": self._digest,
            "file_v": _VIT_POSE_CACHE_FILE_VERSION,
            "resolved_pose": str(pose_f.resolve()),
            "mask_key": mask_key,
            "record_sig": rec_sig,
            "snippet_id": sid,
            "pose": torch.from_numpy(np.asarray(p, dtype=np.float32)),
            "mask": torch.from_numpy(np.asarray(m, dtype=np.bool_)),
        }
        try:
            _torch_save_pose_cache_atomic(pt, payload)
        except Exception as e:
            logger.warning("Could not write vit pose cache %s: %s", pt, e)
        return p, m

    def __getitem__(self, idx):
        rec = self.records[idx]

        pose, mask = self._load_pose_mask_npy(rec)

        if self.is_train:
            n_kp = int(self.cfg["data"]["n_keypoints"])
            fp = _FLIP_PAIRS_17 if n_kp == 17 else []
            aug = self.cfg.get("augmentation") if isinstance(self.cfg.get("augmentation"), dict) else {}
            pose = _apply_train_augmentation(
                pose,
                mask,
                self._rng,
                n_keypoints=n_kp,
                flip_pairs=fp,
                enable_horizontal_flip=bool(aug.get("enable_horizontal_flip", True)),
                aug_cfg=aug,
            )

        dcfg = self.cfg.get("data") if isinstance(self.cfg.get("data"), dict) else {}
        zj = dcfg.get("inference_zero_joint_indices")
        if zj is not None and not self.is_train:
            pose = np.asarray(pose, dtype=np.float32, copy=True)
            n_kp = int(self.cfg["data"]["n_keypoints"])
            for j in zj:
                ji = int(j)
                if 0 <= ji < n_kp:
                    pose[:, ji, :] = 0.0

        pose_t = torch.from_numpy(pose).float()
        mask_t = torch.from_numpy(mask.astype(np.bool_))

        return {
            "pose": pose_t,
            "mask": mask_t,
            "label_5": torch.tensor(int(rec["label_int"]), dtype=torch.long),
            "label_binary": torch.tensor(int(rec["binary_label_int"]), dtype=torch.long),
            "snippet_id": str(rec.get("snippet_id", "")),
            "cat_id": str(rec.get("cat_id", "")),
        }


class DLCPoseDataset(Dataset):
    """
    Pose dataset reading DeepLabCut ``.h5`` via ``dlc_h5_path`` on each record
    (falls back to ``pose_path`` if the former is missing).

    Optional ``data.pose_cache`` (see ``config_stgcn_dlc.yaml``):
    - ``backend: ram`` — keep normalized (pose, mask) in an in-process dict after first HDF5 read
      (large speedup for epoch 2+; safe with ``num_workers=0``).
    - ``backend: disk`` — write ``.pt`` sidecars under ``disk_dir/{digest}/`` for reuse across runs;
      still correct when digest / HDF5 path / ``gpt_resolution`` change.
    - ``backend: off`` — always read HDF5 (slowest).
    """

    _RAM_CACHE: ClassVar[dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]] = {}
    _RAM_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, records, cfg, is_train=False, use_kinematics=True):
        self.records = list(records)
        self.cfg = cfg
        self.is_train = is_train
        self.use_kinematics = use_kinematics
        self._rng = np.random.default_rng()

        pc = cfg.get("data", {}).get("pose_cache") if isinstance(cfg.get("data"), dict) else None
        if isinstance(pc, dict):
            backend = str(pc.get("backend", "ram")).strip().lower()
        else:
            backend = "ram"
        if backend in ("false", "none", "0", "off", ""):
            backend = "off"
        if backend not in ("off", "ram", "disk"):
            logger.warning("Unknown data.pose_cache.backend %r — using off", backend)
            backend = "off"
        self._cache_backend = backend
        self._digest = _dlc_pose_cache_digest(cfg, use_kinematics=use_kinematics)
        self._disk_root: Path | None = None
        if backend == "disk":
            raw = (pc or {}).get("disk_dir", "video/pose-models/cache/dlc_poses")
            dr = Path(str(raw).strip())
            self._disk_root = dr if dr.is_absolute() else REPO_ROOT / dr

    def __len__(self):
        return len(self.records)

    def _resolve_h5_path(self, h5_rel: str) -> Path:
        p = Path(h5_rel)
        return p if p.is_absolute() else REPO_ROOT / p

    def _load_pose_mask(self, h5_rel: str, rec: dict) -> tuple[np.ndarray, np.ndarray]:
        full = self._resolve_h5_path(h5_rel)
        sig = _dlc_pose_record_cache_sig(rec)
        key = (self._digest, str(full.resolve()), sig)

        if self._cache_backend == "off":
            return load_and_normalize_dlc_h5(
                h5_rel, self.cfg, record=dict(rec), use_kinematics=self.use_kinematics
            )

        if self._cache_backend == "ram":
            with self._RAM_LOCK:
                hit = self._RAM_CACHE.get(key)
            if hit is not None:
                pose, mask = hit
                return pose.copy(), mask.copy()
            pose, mask = load_and_normalize_dlc_h5(
                h5_rel, self.cfg, record=dict(rec), use_kinematics=self.use_kinematics
            )
            with self._RAM_LOCK:
                if key not in self._RAM_CACHE:
                    self._RAM_CACHE[key] = (pose.copy(), mask.copy())
            return pose, mask

        assert self._disk_root is not None
        sid = str(rec.get("snippet_id", "row"))
        pt = _dlc_disk_cache_path(self._disk_root, self._digest, sid)
        if pt.is_file():
            try:
                blob = _torch_load_pose_cache(pt)
                if (
                    isinstance(blob, dict)
                    and blob.get("digest") == self._digest
                    and int(blob.get("file_v", -1)) == _DLC_POSE_CACHE_FILE_VERSION
                    and blob.get("resolved_h5") == str(full.resolve())
                    and blob.get("record_sig", "") == sig
                    and "pose" in blob
                    and "mask" in blob
                ):
                    pose = np.asarray(blob["pose"], dtype=np.float32)
                    mask = np.asarray(blob["mask"], dtype=np.bool_)
                    return pose, mask
            except Exception as e:
                logger.warning("Corrupt pose cache %s (%s) — rebuilding", pt, e)

        pose, mask = load_and_normalize_dlc_h5(
            h5_rel, self.cfg, record=dict(rec), use_kinematics=self.use_kinematics
        )
        payload = {
            "digest": self._digest,
            "file_v": _DLC_POSE_CACHE_FILE_VERSION,
            "resolved_h5": str(full.resolve()),
            "record_sig": sig,
            "snippet_id": sid,
            "pose": torch.from_numpy(np.asarray(pose, dtype=np.float32)),
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.bool_)),
        }
        try:
            _torch_save_pose_cache_atomic(pt, payload)
        except Exception as e:
            logger.warning("Could not write pose cache %s: %s", pt, e)
        return pose, mask

    def __getitem__(self, idx):
        rec = self.records[idx]
        h5 = rec.get("dlc_h5_path") or rec.get("pose_path", "")
        pose, mask = self._load_pose_mask(str(h5), rec)
        aug_cfg = self.cfg.get("augmentation") if isinstance(self.cfg.get("augmentation"), dict) else {}
        enable_flip = bool(aug_cfg.get("enable_horizontal_flip", True))
        fp = aug_cfg.get("flip_pairs")
        n_kp = int(self.cfg["data"]["n_keypoints"])
        if fp is None and n_kp == NUM_KEYPOINTS_QUADRUPED:
            fp = QUADRUPED39_FLIP_PAIRS
        elif fp is None and n_kp == 17:
            fp = _FLIP_PAIRS_17
        elif fp is None:
            fp = []
        if self.is_train:
            pose = _apply_train_augmentation(
                pose,
                mask,
                self._rng,
                n_keypoints=n_kp,
                flip_pairs=fp,
                enable_horizontal_flip=enable_flip,
                aug_cfg=aug_cfg,
            )

        pose_t = torch.from_numpy(pose).float()
        mask_t = torch.from_numpy(mask.astype(np.bool_))
        return {
            "pose": pose_t,
            "mask": mask_t,
            "label_5": torch.tensor(int(rec["label_int"]), dtype=torch.long),
            "label_binary": torch.tensor(int(rec["binary_label_int"]), dtype=torch.long),
            "snippet_id": str(rec.get("snippet_id", "")),
            "cat_id": str(rec.get("cat_id", "")),
        }


def build_dlc_dataloaders(
    train_records, val_records, cfg, use_kinematics=True, n_multiclass: int | None = None
):
    """Same as ``build_dataloaders`` but uses ``DLCPoseDataset`` (HDF5 DLC paths)."""
    if n_multiclass is None:
        n_multiclass = len(get_multiclass_class_names(cfg))
    bs = int(cfg["training"]["batch_size"])
    digest = _dlc_pose_cache_digest(cfg, use_kinematics=use_kinematics)
    train_ds = DLCPoseDataset(train_records, cfg, is_train=True, use_kinematics=use_kinematics)
    val_ds = DLCPoseDataset(val_records, cfg, is_train=False, use_kinematics=use_kinematics)
    logger.info(
        "DLC pose cache: backend=%s digest=%s disk_root=%s",
        train_ds._cache_backend,
        digest,
        str(train_ds._disk_root) if train_ds._disk_root is not None else None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    y5 = np.array([int(r["label_int"]) for r in train_records], dtype=np.int64)
    yb = np.array([int(r["binary_label_int"]) for r in train_records], dtype=np.int64)
    w5 = compute_class_weights(y5, int(n_multiclass))
    wb = compute_class_weights(yb, 2)
    return train_loader, val_loader, w5, wb


def build_dataloaders(
    train_records, val_records, cfg, use_kinematics=True, n_multiclass: int = 5
):
    """
    Build train and val DataLoaders.
    Train: shuffle=True, drop_last=True (prevents BatchNorm crash on size-1 batch)
    Val:   shuffle=False, drop_last=False
    Both use cfg.training.batch_size and num_workers=0 (MPS safe).
    n_multiclass: number of classes for multiclass head (class weights); default 5.

    **ViT ``.npy`` cache** — :class:`PoseDataset` uses ``data.pose_cache`` (default **ram**;
    set ``backend: off`` to disable, or ``disk`` to persist under ``disk_dir``).
    Returns (train_loader, val_loader, class_weights_tensor_5, class_weights_tensor_bin)
    """
    bs = int(cfg["training"]["batch_size"])
    vdigest = _vit_pose_cache_digest(cfg, use_kinematics=use_kinematics)
    train_ds = PoseDataset(train_records, cfg, is_train=True, use_kinematics=use_kinematics)
    val_ds = PoseDataset(val_records, cfg, is_train=False, use_kinematics=use_kinematics)
    logger.info(
        "ViT pose (npy) cache: backend=%s digest=%s disk_root=%s",
        train_ds._cache_backend,
        vdigest,
        str(train_ds._disk_root) if train_ds._disk_root is not None else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    y5 = np.array([int(r["label_int"]) for r in train_records], dtype=np.int64)
    yb = np.array([int(r["binary_label_int"]) for r in train_records], dtype=np.int64)
    w5 = compute_class_weights(y5, int(n_multiclass))
    wb = compute_class_weights(yb, 2)

    return train_loader, val_loader, w5, wb
