#!/usr/bin/env python3
"""
Cat identity clustering from CLIP embeddings + union–find merge rules.

Dependencies (install if missing): open_clip_torch, faiss-cpu, Pillow, openai,
ultralytics, opencv-python, torch, scikit-learn, matplotlib, seaborn, pyyaml,
tqdm, numpy, python-dotenv (optional, for REPO_ROOT/.env).

  python dataset_construction/03_cat_id.py
  python dataset_construction/03_cat_id.py --dry-run
  # Input manifest is channel-enriched by default (config); see check_channel_ids.py.
  python dataset_construction/03_cat_id.py --rebuild-cache
  python dataset_construction/03_cat_id.py --rebuild-crop-cache
  python dataset_construction/03_cat_id.py --skip-gpt
  python dataset_construction/03_cat_id.py --limit 200

Re-run without redoing YOLO crops or CLIP (use on-disk caches; do **not** pass
``--rebuild-crop-cache`` or ``--rebuild-cache``):

  cd <REPO_ROOT> && source .venv/bin/activate \\
    && KMP_DUPLICATE_LIB_OK=TRUE python3 dataset_construction/03_cat_id.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Reduce OpenMP/thread oversubscription (helps macOS + torch + faiss + ultralytics).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm")
SNIPPET_PATH_KEYS = (
    "path",
    "video_path",
    "file",
    "snippet_path",
    "video_file",
    "local_path",
)

_MAX_FRAMES_SEQUENTIAL = 8000
_EXTRACT_FRAME_ORDER = [0.50, 0.15, 0.85]


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _candidate_paths_from_snippet(snippet: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k in SNIPPET_PATH_KEYS:
        v = snippet.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for k, v in snippet.items():
        if k in SNIPPET_PATH_KEYS or k == "id":
            continue
        if isinstance(v, str) and ("/" in v or "\\" in v):
            low = v.lower()
            if any(low.endswith(ext) for ext in VIDEO_EXTS):
                out.append(v.strip())
    return out


def resolve_snippet_video(
    repo_root: Path,
    snippet: dict[str, Any],
    snippets_dirs: list[Path],
) -> str | None:
    sid = snippet.get("id")
    if not isinstance(sid, str) or not sid:
        return None

    for raw in _candidate_paths_from_snippet(snippet):
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
        q = repo_root / raw
        if q.is_file():
            return str(q.resolve())

    for base in snippets_dirs:
        for ext in VIDEO_EXTS:
            cand = base / f"{sid}{ext}"
            if cand.is_file():
                return str(cand.resolve())
    return None


def infer_platform(record: dict[str, Any], resolved_path: str | None) -> str:
    src = (record.get("source") or "").strip().lower()
    if src == "tiktok_metadata":
        return "TikTok"
    if src == "metadata_v2":
        pass
    for key in ("platform", "source_platform"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            pl = v.strip().lower()
            if "tiktok" in pl:
                return "TikTok"
            if "dailymotion" in pl or "daily_motion" in pl:
                return "DailyMotion"
            if "youtube" in pl:
                return "YouTube"

    path = (resolved_path or "").lower()
    if "tiktok_snippets" in path or "/tiktok/" in path:
        return "TikTok"
    if "dailymotion" in path:
        return "DailyMotion"

    vid = str(record.get("video_id", ""))
    if vid.isdigit() and len(vid) >= 15:
        return "TikTok"

    return "YouTube"


def infer_channel_meta(record: dict[str, Any], video_id: str) -> tuple[str, str]:
    """
    (channel_string, source_key) for merge rules. source_key is which manifest
    field won: channel_id | channel | uploader_id | author_id | video_id_fallback.
    """
    for k in ("channel_id", "channel", "uploader_id", "author_id"):
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip(), k
    return str(video_id), "video_id_fallback"


def infer_channel(record: dict[str, Any], video_id: str) -> str:
    return infer_channel_meta(record, video_id)[0]


def _count_frames_sequential(video_path: str, max_frames: int) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = 0
    try:
        while n < max_frames:
            ret, _ = cap.read()
            if not ret:
                break
            n += 1
    finally:
        cap.release()
    return n


def _read_bgr_at_frame_indices(video_path: str, frame_indices: list[int]) -> list[np.ndarray] | None:
    if not frame_indices:
        return []
    need_sorted = sorted(set(frame_indices))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    saved: dict[int, np.ndarray] = {}
    try:
        fi = 0
        need_ptr = 0
        while need_ptr < len(need_sorted):
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            if fi == need_sorted[need_ptr]:
                saved[need_sorted[need_ptr]] = frame.copy()
                need_ptr += 1
            fi += 1
        return [saved[i] for i in frame_indices]
    finally:
        cap.release()


def extract_cat_crop(
    video_path: str,
    cfg: dict[str, Any],
    yolo: Any,
) -> tuple[np.ndarray | None, bool, float | None]:
    """
    Try frame positions in order [0.50, 0.15, 0.85].
    For each position:
      1. Seek to frame at that fraction of total frames
      2. Read frame with cv2
      3. Run YOLO (class 15, conf threshold from cfg)
      4. If detection found: crop highest-conf box + padding,
         clamp to image, resize to embed_size×embed_size RGB
         Return (crop_array, bbox_found=True, yolo_conf)
    If no YOLO detection at any position:
      Use middle frame, resize full frame to embed_size×embed_size
      Return (full_frame_array, bbox_found=False, None)
    If video cannot be opened at all:
      Return (None, False, None)
    Never raises — catches all exceptions.
    """
    cat_id_cfg = cfg["cat_id"]
    conf_thr = float(cat_id_cfg["yolo_conf"])
    pad = float(cat_id_cfg["bbox_padding"])
    embed_size = int(cat_id_cfg["embed_size"])
    coco_cat = 15

    try:
        n = _count_frames_sequential(video_path, _MAX_FRAMES_SEQUENTIAL)
        if n <= 0:
            return None, False, None

        def frac_to_idx(frac: float) -> int:
            if n == 1:
                return 0
            idx = int(round(float(frac) * (n - 1)))
            return max(0, min(idx, n - 1))

        for frac in _EXTRACT_FRAME_ORDER:
            idx = frac_to_idx(frac)
            bgr_list = _read_bgr_at_frame_indices(video_path, [idx])
            if bgr_list is None or not bgr_list:
                continue
            frame = bgr_list[0]
            h, w = frame.shape[:2]
            results = yolo.predict(
                source=frame,
                conf=conf_thr,
                verbose=False,
                classes=[coco_cat],
            )
            best_conf = 0.0
            best_box: tuple[float, float, float, float] | None = None
            if results:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    cfs = r0.boxes.conf.cpu().numpy()
                    bi = int(np.argmax(cfs))
                    best_conf = float(cfs[bi])
                    x1, y1, x2, y2 = xyxy[bi].tolist()
                    best_box = (x1, y1, x2, y2)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                bw, bh = x2 - x1, y2 - y1
                px, py = pad * bw, pad * bh
                x1i = max(0, int(np.floor(x1 - px)))
                y1i = max(0, int(np.floor(y1 - py)))
                x2i = min(w, int(np.ceil(x2 + px)))
                y2i = min(h, int(np.ceil(y2 + py)))
                crop = frame[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (embed_size, embed_size), interpolation=cv2.INTER_AREA)
                return rgb, True, best_conf

        mid = frac_to_idx(0.50)
        bgr_list = _read_bgr_at_frame_indices(video_path, [mid])
        if bgr_list is None or not bgr_list:
            return None, False, None
        frame = bgr_list[0]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (embed_size, embed_size), interpolation=cv2.INTER_AREA)
        return rgb, False, None
    except Exception:
        return None, False, None


def gpt_field(snippet: dict[str, Any], *keys: str) -> Any:
    cur: Any = snippet.get("gpt_description")
    if not isinstance(cur, dict):
        return None
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def classify_snippet(snippet: dict[str, Any]) -> tuple[str, str | None]:
    """
    Returns (bucket, reason) where bucket is EMBEDDABLE | SKIPPED_NO_CAT |
    SKIPPED_UNSUITABLE and reason is None or a short tag.
    """
    gd = snippet.get("gpt_description")
    if gd is None:
        return "EMBEDDABLE", "gpt_missing"
    if not isinstance(gd, dict):
        return "EMBEDDABLE", "gpt_invalid"
    if gd.get("parse_error") is True:
        return "EMBEDDABLE", "gpt_parse_error"

    cats = gd.get("cats")
    n_cats = None
    if isinstance(cats, dict):
        n_cats = cats.get("n_cats_visible")

    if n_cats == 0:
        return "SKIPPED_NO_CAT", None

    flags = gd.get("dataset_flags")
    suitable = None
    if isinstance(flags, dict):
        suitable = flags.get("suitable_for_training")

    vq = gd.get("video_quality")
    ai_gen = None
    if isinstance(vq, dict):
        ai_gen = vq.get("is_ai_generated")

    if suitable is False or ai_gen is True:
        return "SKIPPED_UNSUITABLE", None

    return "EMBEDDABLE", None


class UnionFind:
    """Standard union-find with path compression and union by rank."""

    def __init__(self, ids: list[str]) -> None:
        self.parent: dict[str, str] = {x: x for x in ids}
        self.rank: dict[str, int] = {x: 0 for x in ids}

    def find(self, x: str) -> str:
        if x not in self.parent:
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        if x not in self.parent or y not in self.parent:
            return False
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def components(self) -> dict[str, list[str]]:
        comp: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            comp[self.find(x)].append(x)
        return dict(comp)


def _atomic_savez_npz(path: Path, snippet_ids: np.ndarray, embeddings: np.ndarray) -> None:
    """numpy.savez_compressed always appends ``.npz`` if the filename lacks it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}_tmp{path.suffix}"
    np.savez_compressed(str(tmp), snippet_ids=snippet_ids, embeddings=embeddings)
    os.replace(str(tmp), str(path))


def _load_embedding_cache(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    if not path.is_file():
        return {}, None, None
    try:
        z = np.load(path, allow_pickle=True)
        ids = z["snippet_ids"]
        emb = z["embeddings"]
        out: dict[str, np.ndarray] = {}
        for i, sid in enumerate(ids.tolist()):
            s = str(sid)
            out[s] = np.asarray(emb[i], dtype=np.float32)
        return out, ids, emb
    except Exception:
        return {}, None, None


def _bbox_crop_cache_signature(ci: dict[str, Any]) -> str:
    """Invalidate crop cache when extraction / YOLO settings change."""
    payload = {
        "embed_size": int(ci["embed_size"]),
        "yolo_conf": float(ci["yolo_conf"]),
        "bbox_padding": float(ci["bbox_padding"]),
        "yolo_weights": str(ci["yolo_weights"]),
        "frame_order": list(_EXTRACT_FRAME_ORDER),
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return h[:20]


def _load_bbox_crop_cache(
    path: Path,
    expect_sig: str,
    embed_size: int,
) -> dict[str, tuple[np.ndarray, bool, float | None]]:
    """
    snippet_id -> (RGB uint8 H×W×3, bbox_found, yolo_conf or None).
    Returns {} if file missing, corrupt, or signature / embed_size mismatch.
    """
    if not path.is_file():
        return {}
    try:
        z = np.load(path, allow_pickle=True)
        got_sig = str(z["cache_sig"].item()) if "cache_sig" in z.files else ""
        got_es = int(z["embed_size"].item()) if "embed_size" in z.files else 0
        if got_sig != expect_sig or got_es != embed_size:
            return {}
        ids = z["snippet_ids"]
        crops = z["crops"]
        bf = z["bbox_found"]
        yc = z["yolo_conf"]
        out: dict[str, tuple[np.ndarray, bool, float | None]] = {}
        for i, sid in enumerate(ids.tolist()):
            s = str(sid)
            crop = np.asarray(crops[i], dtype=np.uint8)
            if crop.shape[:2] != (embed_size, embed_size):
                return {}
            b = bool(bf[i])
            conf_f = float(yc[i])
            conf: float | None = conf_f if np.isfinite(conf_f) else None
            out[s] = (crop, b, conf)
        return out
    except Exception:
        return {}


def _flush_bbox_crop_cache(
    path: Path,
    merged: dict[str, tuple[np.ndarray | None, bool, float | None]],
    cache_sig: str,
    embed_size: int,
) -> None:
    sids = sorted(s for s, t in merged.items() if t[0] is not None)
    if not sids:
        return
    stacks: list[np.ndarray] = []
    bf_l: list[bool] = []
    yc_l: list[float] = []
    for s in sids:
        rgb, b, conf = merged[s]
        assert rgb is not None
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.shape != (embed_size, embed_size, 3):
            raise ValueError(f"bad crop shape for {s}: {arr.shape}")
        stacks.append(arr)
        bf_l.append(bool(b))
        yc_l.append(float(conf) if conf is not None else float("nan"))
    mat = np.stack(stacks, axis=0)
    sid_arr = np.array(sids, dtype=object)
    bf_arr = np.array(bf_l, dtype=np.bool_)
    yc_arr = np.array(yc_l, dtype=np.float32)
    es_arr = np.array(embed_size, dtype=np.int32)
    sig_arr = np.array(cache_sig, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}_tmp{path.suffix}"
    np.savez_compressed(
        str(tmp),
        snippet_ids=sid_arr,
        crops=mat,
        bbox_found=bf_arr,
        yolo_conf=yc_arr,
        embed_size=es_arr,
        cache_sig=sig_arr,
    )
    os.replace(str(tmp), str(path))


def _parse_gpt_verify_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, None
        return None, "root not object"
    except json.JSONDecodeError as e:
        return None, str(e)


@dataclass
class RunStats:
    embeddable: int = 0
    skipped_no_cat: int = 0
    skipped_unsuitable: int = 0
    gpt_missing_warn: int = 0
    yolo_hit: int = 0
    yolo_fallback: int = 0
    bbox_crop_cache_hits: int = 0
    bbox_crop_cache_misses: int = 0
    extract_failed: int = 0
    from_cache: int = 0
    newly_computed: int = 0
    same_video_merges: int = 0
    breed_coat_merges: int = 0
    same_channel_merges: int = 0
    cross_channel_bbox_merges: int = 0
    cross_channel_fallback_merges: int = 0
    gpt_verified_merges: int = 0
    sims_rule2: list[float] = field(default_factory=list)
    sims_rule3: list[float] = field(default_factory=list)
    sims_rule4: list[float] = field(default_factory=list)
    sims_rule5: list[float] = field(default_factory=list)
    gpt_pairs_sent: int = 0
    gpt_verified_same: int = 0
    gpt_verified_different: int = 0
    gpt_uncertain: int = 0
    gpt_parse_errors: int = 0
    gpt_total_cost: float = 0.0
    limit_off_cluster: int = 0


def _strip_frontmatter(prompt_text: str) -> str:
    t = prompt_text.strip()
    if t.startswith("---"):
        parts = t.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return t


def main() -> int:
    _load_dotenv(REPO_ROOT)
    parser = argparse.ArgumentParser(description="Cat ID assignment via CLIP + union–find.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Delete CLIP embedding cache only (see also --rebuild-crop-cache).",
    )
    parser.add_argument(
        "--rebuild-crop-cache",
        action="store_true",
        help="Before frame extraction, delete bbox RGB crop npz (forces YOLO + decode for all snippets).",
    )
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg_all = load_config()
    sources = cfg_all["sources"]
    ci = cfg_all["cat_id"]
    reports_dir = REPO_ROOT / cfg_all["output"]["reports_dir"]
    snippets_dirs = [REPO_ROOT / p for p in sources["snippets_dirs"]]

    in_path = REPO_ROOT / ci["input_manifest"]
    out_manifest = REPO_ROOT / ci["output_manifest"]
    assign_csv = REPO_ROOT / ci["assignments_csv"]
    report_txt = REPO_ROOT / ci["report_txt"]
    cache_path = REPO_ROOT / ci["embedding_cache"]
    crop_cache_path = REPO_ROOT / ci.get(
        "bbox_crop_cache", "src/dataset_construction/reports/cat_id_bbox_crops.npz"
    )
    crop_cache_sig = _bbox_crop_cache_signature(ci)
    embed_size_cfg = int(ci["embed_size"])
    gpt_prompt_path = REPO_ROOT / ci["gpt_verify_prompt_file"]

    if not in_path.is_file():
        print(f"Missing input manifest: {in_path}", file=sys.stderr)
        if "channels" in in_path.name:
            print(
                "Hint: create it with `python dataset_construction/check_channel_ids.py` "
                "(needs metadata_clean_03.jsonl). For a first cat_id pass from GPT rows only, "
                "set cat_id.input_manifest to dataset_construction/manifests/metadata_clean_02.jsonl "
                "in config.yaml.",
                file=sys.stderr,
            )
        return 1

    records = _iter_jsonl(in_path)
    snippets_flat: list[dict[str, Any]] = []
    for rec in records:
        vid = str(rec.get("video_id") or "")
        ch, ch_src = infer_channel_meta(rec, vid)
        bcat = rec.get("behavioral_category")
        for sn in rec.get("snippets") or []:
            if not isinstance(sn, dict):
                continue
            sid = sn.get("id")
            if not isinstance(sid, str) or not sid:
                continue
            resolved = resolve_snippet_video(REPO_ROOT, sn, snippets_dirs)
            platform = infer_platform(rec, resolved)
            row = dict(sn)
            row["_video_id"] = vid
            row["_channel"] = ch
            row["_channel_source"] = ch_src
            row["_platform"] = platform
            row["_behavioral_category"] = bcat
            row["_video_path"] = resolved
            snippets_flat.append(row)

    n_total = len(snippets_flat)
    stats = RunStats()
    bucket_by_id: dict[str, str] = {}
    warn_ids: set[str] = set()

    for sn in snippets_flat:
        sid = str(sn["id"])
        bucket, wr = classify_snippet(sn)
        bucket_by_id[sid] = bucket
        if bucket == "EMBEDDABLE":
            stats.embeddable += 1
            if wr:
                stats.gpt_missing_warn += 1
                warn_ids.add(sid)
        elif bucket == "SKIPPED_NO_CAT":
            stats.skipped_no_cat += 1
        else:
            stats.skipped_unsuitable += 1

    def pct(x: int) -> str:
        return f"{100.0 * x / n_total:.1f}%" if n_total else "0.0%"

    print(
        f"Pre-filter summary:\n"
        f"  Embeddable:        {stats.embeddable} ({pct(stats.embeddable)})\n"
        f"  Skipped no cat:    {stats.skipped_no_cat} ({pct(stats.skipped_no_cat)})\n"
        f"  Skipped unsuitable:{stats.skipped_unsuitable} ({pct(stats.skipped_unsuitable)})\n"
        f"  GPT data missing:  {stats.gpt_missing_warn} (will embed with warning)"
    )

    if args.dry_run:
        print("Dry run: no model I/O or output files.")
        return 0

    # Do not ``import faiss`` here: loading FAISS before PyTorch/Ultralytics can segfault on macOS.
    if importlib.util.find_spec("faiss") is None:
        print(
            "Missing package: faiss. Install with:\n"
            "  python3 -m pip install faiss-cpu\n"
            "On Homebrew-managed Python (PEP 668) you may need:\n"
            "  python3 -m pip install faiss-cpu --break-system-packages",
            file=sys.stderr,
        )
        return 1

    try:
        import torch  # noqa: F401 — initialize torch runtime before ultralytics/faiss native stacks
    except ImportError:
        print("Install torch: pip install torch", file=sys.stderr)
        return 1

    embed_ids = [str(sn["id"]) for sn in snippets_flat if bucket_by_id[str(sn["id"])] == "EMBEDDABLE"]
    id_to_row = {str(sn["id"]): sn for sn in snippets_flat}

    if args.limit is not None:
        lim = max(0, int(args.limit))
        work_ids = embed_ids[:lim]
        off_ids = embed_ids[lim:]
        stats.limit_off_cluster = len(off_ids)
    else:
        work_ids = list(embed_ids)
        off_ids = []

    crops: dict[str, tuple[np.ndarray | None, bool, float | None]] = {}
    bbox_by_id: dict[str, bool] = {}
    yconf_by_id: dict[str, float | None] = {}

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        raise e

    if args.rebuild_crop_cache and crop_cache_path.is_file():
        crop_cache_path.unlink()
        tqdm.write(
            f"[bbox cache] cleared (--rebuild-crop-cache): "
            f"{crop_cache_path.relative_to(REPO_ROOT)}"
        )

    crop_merged: dict[str, tuple[np.ndarray | None, bool, float | None]] = dict(
        _load_bbox_crop_cache(crop_cache_path, crop_cache_sig, embed_size_cfg)
    )
    if crop_merged and not args.rebuild_crop_cache:
        tqdm.write(
            f"[bbox cache] loaded {len(crop_merged)} crops from "
            f"{crop_cache_path.relative_to(REPO_ROOT)} (sig={crop_cache_sig})"
        )

    yolo = YOLO(ci["yolo_weights"])
    yolo_hit = yolo_fb = yolo_fail = 0
    crop_cache_dirty = False
    pbar = tqdm(work_ids, desc="Frame extraction", unit="snippet")
    try:
        for sid in pbar:
            sn = id_to_row[sid]
            vp = sn.get("_video_path")
            if not isinstance(vp, str) or not vp:
                crops[sid] = (None, False, None)
                yolo_fail += 1
                pbar.set_postfix(
                    yolo_hit=yolo_hit,
                    fallback=yolo_fb,
                    failed=yolo_fail,
                    cached=stats.bbox_crop_cache_hits,
                    refresh=True,
                )
                continue

            cached = crop_merged.get(sid)
            if cached is not None and cached[0] is not None:
                crop, bf, yc = cached
                stats.bbox_crop_cache_hits += 1
            else:
                stats.bbox_crop_cache_misses += 1
                crop, bf, yc = extract_cat_crop(vp, cfg_all, yolo)
                if crop is not None:
                    crop_merged[sid] = (crop, bf, yc)
                    crop_cache_dirty = True

            crops[sid] = (crop, bf, yc)
            bbox_by_id[sid] = bf
            yconf_by_id[sid] = yc
            if crop is None:
                yolo_fail += 1
            elif bf:
                yolo_hit += 1
            else:
                yolo_fb += 1
            pbar.set_postfix(
                yolo_hit=yolo_hit,
                fallback=yolo_fb,
                failed=yolo_fail,
                cached=stats.bbox_crop_cache_hits,
                refresh=True,
            )
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted during frame extraction — saving bbox crop cache…")
        if crop_cache_dirty:
            try:
                _flush_bbox_crop_cache(crop_cache_path, crop_merged, crop_cache_sig, embed_size_cfg)
                tqdm.write(
                    f"[bbox cache] wrote {len([s for s, t in crop_merged.items() if t[0] is not None])} "
                    f"crops to {crop_cache_path.relative_to(REPO_ROOT)}"
                )
            except Exception as e:
                tqdm.write(f"[bbox cache] save failed: {e}")
        raise

    if crop_cache_dirty:
        try:
            _flush_bbox_crop_cache(crop_cache_path, crop_merged, crop_cache_sig, embed_size_cfg)
            tqdm.write(
                f"[bbox cache] wrote {len([s for s, t in crop_merged.items() if t[0] is not None])} "
                f"crops to {crop_cache_path.relative_to(REPO_ROOT)}"
            )
        except Exception as e:
            tqdm.write(f"[bbox cache] save failed: {e}", file=sys.stderr)

    print(
        f"Bbox crops: {stats.bbox_crop_cache_hits} from cache, "
        f"{stats.bbox_crop_cache_misses} freshly extracted (YOLO + decode) "
        f"(this run only — first run after a cache clear always shows 0 from disk)."
    )
    n_bbox_cached_on_disk = len([s for s, t in crop_merged.items() if t[0] is not None])
    if n_bbox_cached_on_disk and crop_cache_path.is_file():
        print(
            f"Bbox crop cache on disk: {crop_cache_path.relative_to(REPO_ROOT)} "
            f"({n_bbox_cached_on_disk} entries, sig={crop_cache_sig}) "
            f"— next run with the same settings skips decode+YOLO for those ids."
        )

    embedded_ids = [s for s in work_ids if crops.get(s) and crops[s][0] is not None]
    extract_failed_ids = [s for s in work_ids if s not in embedded_ids]
    stats.extract_failed = len(extract_failed_ids)

    if args.rebuild_cache and cache_path.is_file():
        cache_path.unlink()

    cache_dict, _, _ = _load_embedding_cache(cache_path)
    need_compute = [s for s in embedded_ids if s not in cache_dict]
    stats.from_cache = len(embedded_ids) - len(need_compute)
    stats.newly_computed = len(need_compute)

    def flush_embedding_cache(*, allow_partial: bool = False) -> None:
        if not cache_dict:
            return
        if not allow_partial:
            missing_final = [s for s in embedded_ids if s not in cache_dict]
            if missing_final:
                tqdm.write(f"[cache] incomplete: {len(missing_final)} embeddings missing; skip write")
                return
        all_ids = sorted(cache_dict.keys())
        mat = np.stack([cache_dict[s] for s in all_ids], axis=0).astype(np.float32)
        sid_arr = np.array(all_ids, dtype=object)
        _atomic_savez_npz(cache_path, sid_arr, mat)

    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as e:
        print("Install open_clip_torch and torch: pip install open_clip_torch torch", file=sys.stderr)
        raise e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = preprocess = None
    emb_dim = 512

    def _resolve_emb_dim(m: Any) -> int:
        v = getattr(m, "visual", None)
        if v is not None:
            d = getattr(v, "output_dim", None)
            if isinstance(d, int) and d > 0:
                return d
        return 512

    def run_clip_encode(
        ids_batch: list[str],
        model_: Any,
        preprocess_: Any,
        dev: str,
    ) -> np.ndarray:
        tensors = []
        for sid in ids_batch:
            arr = crops[sid][0]
            assert arr is not None
            pil = Image.fromarray(arr)
            tensors.append(preprocess_(pil).unsqueeze(0))
        xb = torch.cat(tensors, dim=0).to(dev)
        with torch.no_grad():
            feats = model_.encode_image(xb).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.cpu().numpy().astype(np.float32)

    if need_compute:
        model, _, preprocess = open_clip.create_model_and_transforms(
            ci["clip_model"],
            pretrained=ci["clip_pretrained"],
        )
        model.eval()
        model = model.to(device)
        emb_dim = _resolve_emb_dim(model)
        bs = max(1, int(ci["embedding_batch_size"]))
        new_embs: dict[str, np.ndarray] = {}
        pbar2 = tqdm(range(0, len(need_compute), bs), desc="CLIP embedding", unit="batch")
        try:
            for i in pbar2:
                batch = need_compute[i : i + bs]
                try:
                    mat = run_clip_encode(batch, model, preprocess, device)
                    for j, sid in enumerate(batch):
                        new_embs[sid] = mat[j]
                except Exception as e:
                    tqdm.write(f"[CLIP] batch failed {batch[0]}…: {e}")
                    for sid in batch:
                        try:
                            m = run_clip_encode([sid], model, preprocess, device)
                            new_embs[sid] = m[0]
                        except Exception as e2:
                            tqdm.write(f"[CLIP] skip {sid}: {e2}")
                pbar2.set_postfix(new=len(new_embs), refresh=True)
        except KeyboardInterrupt:
            tqdm.write("\nInterrupted during CLIP — saving embedding cache…")
            for sid, vec in new_embs.items():
                cache_dict[sid] = vec
            flush_embedding_cache(allow_partial=True)
            tqdm.write(
                f"Progress: {len(new_embs)}/{len(need_compute)} new embeddings saved to cache "
                f"({len(cache_dict)} total keys)."
            )
            return 130

        for sid, vec in new_embs.items():
            cache_dict[sid] = vec

    # Persist cache (merge with prior keys; always include full cache_dict)
    if embedded_ids:
        missing_final = [s for s in embedded_ids if s not in cache_dict]
        if missing_final:
            print(f"ERROR: missing embeddings for {len(missing_final)} snippets", file=sys.stderr)
            return 1
        try:
            flush_embedding_cache(allow_partial=False)
        except KeyboardInterrupt:
            tqdm.write("KeyboardInterrupt during cache write — saving…")
            flush_embedding_cache(allow_partial=True)
            raise

    print(f"Embeddings: {stats.from_cache} from cache, {stats.newly_computed} newly computed")

    nprobe = int(ci.get("faiss_nprobe", 10))
    th = ci["thresholds"]
    floor = min(
        float(th["same_channel_breed_coat"]),
        float(th["same_channel"]),
        float(th["cross_channel_bbox_both"]),
        float(th["cross_channel_any_fallback"]),
    )
    if nprobe and floor:
        pass  # Flat index: nprobe unused; kept for config parity

    id_list = list(embedded_ids)
    n_emb = len(id_list)

    if n_emb == 0:
        print("No embedded snippets in work set — skipping FAISS/clustering.", file=sys.stderr)
        neighbors = {}
        nearest_sim = {}
        edges_sorted: list[tuple[str, str, float]] = []
        uf = UnionFind([])
    else:
        import faiss

        emb_mat = np.stack([cache_dict[s] for s in id_list], axis=0).astype(np.float32)
        emb_dim = int(emb_mat.shape[1])
        index = faiss.IndexFlatIP(emb_dim)
        index.add(emb_mat)

        k_search = min(50, max(1, n_emb))
        sims, idxs = index.search(emb_mat, k_search)
        neighbors = {s: [] for s in id_list}
        nearest_sim: dict[str, float | None] = {s: None for s in id_list}
        edge_best: dict[tuple[str, str], float] = {}

        for i, sid in enumerate(id_list):
            best = None
            for jj in range(k_search):
                j = int(idxs[i, jj])
                if j < 0:
                    continue
                sim = float(sims[i, jj])
                oid = id_list[j]
                if oid == sid:
                    continue
                if best is None or sim > best:
                    best = sim
                if sim >= floor:
                    neighbors[sid].append((oid, sim))
                    a, b = (sid, oid) if sid < oid else (oid, sid)
                    key = (a, b)
                    prev = edge_best.get(key)
                    if prev is None or sim > prev:
                        edge_best[key] = sim
            if best is not None:
                nearest_sim[sid] = best

        edges_sorted = sorted(
            [(a, b, s) for (a, b), s in edge_best.items()],
            key=lambda t: (-t[2], t[0], t[1]),
        )

        mean_nb = float(np.mean([len(neighbors[s]) for s in id_list])) if id_list else 0.0
        print(f"FAISS search complete: mean neighbors per snippet = {mean_nb:.2f}")

        uf = UnionFind(id_list)

        # Rule 1 same video
        by_video: dict[str, list[str]] = defaultdict(list)
        for sid in id_list:
            by_video[str(id_to_row[sid].get("_video_id") or "")].append(sid)
        for vid, group in by_video.items():
            if len(group) < 2:
                continue
            base = group[0]
            for other in group[1:]:
                if uf.union(base, other):
                    stats.same_video_merges += 1

    th_breed = float(th["same_channel_breed_coat"])
    th_same_ch = float(th["same_channel"])
    th_xbb = float(th["cross_channel_bbox_both"])
    th_xfb = float(th["cross_channel_any_fallback"])

    for a, b, sim in edges_sorted:
        if sim < th_breed:
            continue
        sa, sb = id_to_row[a], id_to_row[b]
        if str(sa.get("_channel")) != str(sb.get("_channel")):
            continue
        breed_a = gpt_field(sa, "cats", "primary_cat", "breed_guess")
        breed_b = gpt_field(sb, "cats", "primary_cat", "breed_guess")
        coat_a = gpt_field(sa, "cats", "primary_cat", "coat_pattern")
        coat_b = gpt_field(sb, "cats", "primary_cat", "coat_pattern")
        breeds_match = bool(breed_a and breed_b and breed_a == breed_b)
        coats_match = bool(coat_a and coat_b and coat_a == coat_b)
        if breeds_match and coats_match and uf.union(a, b):
            stats.breed_coat_merges += 1
            stats.sims_rule2.append(sim)

    for a, b, sim in edges_sorted:
        if sim < th_same_ch:
            continue
        sa, sb = id_to_row[a], id_to_row[b]
        if str(sa.get("_channel")) != str(sb.get("_channel")):
            continue
        if uf.union(a, b):
            stats.same_channel_merges += 1
            stats.sims_rule3.append(sim)

    for a, b, sim in edges_sorted:
        if sim < th_xbb:
            continue
        sa, sb = id_to_row[a], id_to_row[b]
        if str(sa.get("_channel")) == str(sb.get("_channel")):
            continue
        if bbox_by_id.get(a) and bbox_by_id.get(b) and uf.union(a, b):
            stats.cross_channel_bbox_merges += 1
            stats.sims_rule4.append(sim)

    for a, b, sim in edges_sorted:
        if sim < th_xfb:
            continue
        sa, sb = id_to_row[a], id_to_row[b]
        if str(sa.get("_channel")) == str(sb.get("_channel")):
            continue
        if (not bbox_by_id.get(a) or not bbox_by_id.get(b)) and uf.union(a, b):
            stats.cross_channel_fallback_merges += 1
            stats.sims_rule5.append(sim)

    def component_cat_ids() -> dict[str, str]:
        comp = uf.components()
        out: dict[str, str] = {}
        for _root, members in comp.items():
            canon = min(members)
            cid = f"cat_{canon}"
            for m in members:
                out[m] = cid
        return out

    sid_to_cluster = component_cat_ids()

    gpt_skip = args.skip_gpt or not bool(ci.get("gpt_verify", True))
    gpt_verified_snips: set[str] = set()
    sim_lo = float(ci["gpt_verify_sim_low"])
    sim_hi = float(ci["gpt_verify_sim_high"])
    if not gpt_skip and sim_lo >= th_same_ch:
        print(
            f"WARNING: gpt_verify_sim_low ({sim_lo}) >= same_channel threshold ({th_same_ch}); "
            "GPT verification candidate set is usually empty. Lower gpt_verify_sim_low (e.g. 0.82) to enable."
        )

    if not gpt_skip and sim_lo < sim_hi:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            print("OPENAI_API_KEY not set — skipping GPT verification.", file=sys.stderr)
            gpt_skip = True
        else:
            from openai import OpenAI
            from openai import APIStatusError, RateLimitError

            raw_prompt = gpt_prompt_path.read_text(encoding="utf-8")
            system_prompt = _strip_frontmatter(raw_prompt).strip()

            candidates: list[tuple[str, str, float]] = []
            seen_pair: set[tuple[str, str]] = set()
            for a, b, sim in edges_sorted:
                if sim < sim_lo or sim >= sim_hi:
                    continue
                sa, sb = id_to_row[a], id_to_row[b]
                if str(sa.get("_channel")) != str(sb.get("_channel")):
                    continue
                if str(sa.get("_video_id")) == str(sb.get("_video_id")):
                    continue
                if uf.find(a) == uf.find(b):
                    continue
                ra, rb = sid_to_cluster[a], sid_to_cluster[b]
                key = (ra, rb) if ra < rb else (rb, ra)
                if key in seen_pair:
                    continue
                seen_pair.add(key)
                candidates.append((a, b, sim))

            max_pairs = int(ci["gpt_verify_max_pairs"])
            random.seed(42)
            random.shuffle(candidates)
            candidates = candidates[:max_pairs]

            client = OpenAI(api_key=api_key)
            max_retries = int(ci.get("gpt_verify_max_retries", 8))
            base_delay = float(ci.get("gpt_verify_retry_base_delay", 2.0))
            cost_in = float(ci["gpt_verify_cost_per_1k_input"])
            cost_out = float(ci["gpt_verify_cost_per_1k_output"])
            model_g = str(ci["gpt_verify_model"])

            def encode_jpeg_b64(rgb: np.ndarray) -> str:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok or buf is None:
                    return ""
                return base64.standard_b64encode(buf.tobytes()).decode("ascii")

            pbar_g = tqdm(candidates, desc="GPT Verification", unit="pair")
            for a, b, sim in pbar_g:
                stats.gpt_pairs_sent += 1
                ca, _, _ = crops[a]
                cb, _, _ = crops[b]
                if ca is None or cb is None:
                    stats.gpt_parse_errors += 1
                    continue
                b64a, b64b = encode_jpeg_b64(ca), encode_jpeg_b64(cb)
                if not b64a or not b64b:
                    stats.gpt_parse_errors += 1
                    continue
                user_text = (
                    f"Clip A: {a} | Clip B: {b}\n"
                    "Are these the same physical cat?"
                )
                content: list[dict[str, Any]] = [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64a}", "detail": "auto"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64b}", "detail": "auto"},
                    },
                ]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ]
                raw_out = ""
                in_tok = out_tok = 0
                ok_call = False
                for attempt in range(max_retries + 1):
                    try:
                        resp = client.chat.completions.create(
                            model=model_g,
                            messages=messages,
                            max_tokens=200,
                            temperature=0.0,
                        )
                        raw_out = (resp.choices[0].message.content or "").strip()
                        usage = resp.usage
                        in_tok = int(usage.prompt_tokens or 0) if usage else 0
                        out_tok = int(usage.completion_tokens or 0) if usage else 0
                        ok_call = True
                        break
                    except (RateLimitError, APIStatusError) as e:
                        code = getattr(e, "status_code", None) or getattr(e, "code", None)
                        if code in (429, 500, 502, 503) and attempt < max_retries:
                            time.sleep(base_delay * (2**attempt))
                        else:
                            tqdm.write(f"GPT verify API error {a}/{b}: {e}")
                            break
                    except Exception as e:
                        tqdm.write(f"GPT verify error {a}/{b}: {e}")
                        break

                if not ok_call:
                    stats.gpt_parse_errors += 1
                    continue

                cost_pair = (in_tok / 1000.0) * cost_in + (out_tok / 1000.0) * cost_out
                stats.gpt_total_cost += cost_pair

                parsed, _perr = _parse_gpt_verify_json(raw_out)
                if parsed is None:
                    stats.gpt_parse_errors += 1
                    continue
                same = parsed.get("same_cat")
                conf = str(parsed.get("confidence") or "").lower()
                if same is True and conf != "low":
                    if uf.union(a, b):
                        stats.gpt_verified_merges += 1
                        stats.gpt_verified_same += 1
                        gpt_verified_snips.add(a)
                        gpt_verified_snips.add(b)
                elif same is False:
                    stats.gpt_verified_different += 1
                elif same is None:
                    stats.gpt_uncertain += 1
                else:
                    stats.gpt_uncertain += 1

                pbar_g.set_postfix(
                    merged=stats.gpt_verified_merges,
                    cost=f"${stats.gpt_total_cost:.2f}",
                    refresh=True,
                )

    sid_to_cluster = component_cat_ids()

    # cluster sizes (embedded only + off-limit singletons)
    cluster_sizes: dict[str, int] = defaultdict(int)
    for sid in embedded_ids:
        cluster_sizes[sid_to_cluster[sid]] += 1
    for sid in off_ids:
        cluster_sizes[f"cat_{sid}"] += 1

    # per-snippet assignment fields
    out_fields: dict[str, dict[str, Any]] = {}
    for sn in snippets_flat:
        sid = str(sn["id"])
        b = bucket_by_id[sid]
        if b == "SKIPPED_NO_CAT":
            out_fields[sid] = {
                "cat_id": f"no_cat_{sid}",
                "cat_id_method": "no_cat",
                "bbox_found": False,
                "yolo_conf": None,
                "clip_similarity_to_nearest": None,
                "gpt_verified": False,
            }
        elif b == "SKIPPED_UNSUITABLE":
            out_fields[sid] = {
                "cat_id": f"unsuitable_{sid}",
                "cat_id_method": "unsuitable",
                "bbox_found": False,
                "yolo_conf": None,
                "clip_similarity_to_nearest": None,
                "gpt_verified": False,
            }
        elif sid in off_ids:
            out_fields[sid] = {
                "cat_id": f"cat_{sid}",
                "cat_id_method": "singleton",
                "bbox_found": False,
                "yolo_conf": None,
                "clip_similarity_to_nearest": None,
                "gpt_verified": False,
            }
        elif sid in extract_failed_ids:
            out_fields[sid] = {
                "cat_id": f"extract_failed_{sid}",
                "cat_id_method": "extract_failed",
                "bbox_found": bbox_by_id.get(sid, False),
                "yolo_conf": yconf_by_id.get(sid),
                "clip_similarity_to_nearest": None,
                "gpt_verified": False,
            }
        elif sid in embedded_ids:
            cid = sid_to_cluster[sid]
            sz = cluster_sizes[cid]
            method = "singleton" if sz == 1 else "union_find"
            out_fields[sid] = {
                "cat_id": cid,
                "cat_id_method": method,
                "bbox_found": bool(bbox_by_id.get(sid)),
                "yolo_conf": yconf_by_id.get(sid),
                "clip_similarity_to_nearest": nearest_sim.get(sid),
                "gpt_verified": sid in gpt_verified_snips,
            }
        else:
            # embeddable but not in work_ids (should not happen if sid in embed_ids)
            out_fields[sid] = {
                "cat_id": f"cat_{sid}",
                "cat_id_method": "singleton",
                "bbox_found": False,
                "yolo_conf": None,
                "clip_similarity_to_nearest": None,
                "gpt_verified": False,
            }

    # CV validation
    cv_folds = int(ci["cv_folds"])
    cv_min = int(ci["cv_min_cats_per_fold"])
    embed_for_cv = [str(sn["id"]) for sn in snippets_flat if bucket_by_id[str(sn["id"])] == "EMBEDDABLE"]
    X_idx = np.arange(len(embed_for_cv))
    groups = np.array([out_fields[s]["cat_id"] for s in embed_for_cv])
    y_str = np.array(
        [str(id_to_row[s].get("_behavioral_category") or "unknown") for s in embed_for_cv],
        dtype=object,
    )

    cv_lines: list[str] = []
    leakage_notes: list[str] = []
    fold_rows: list[tuple[int, int, int, int, str]] = []
    cv_mode = "skipped"

    try:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

        if len(embed_for_cv) < cv_folds:
            cv_mode = "skipped_too_few_rows"
            leakage_notes.append(f"Embeddable count {len(embed_for_cv)} < cv_folds {cv_folds}")
        else:
            sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            try:
                splits = list(sgkf.split(X_idx, y_str, groups))
                cv_mode = "StratifiedGroupKFold"
            except ValueError as e:
                tqdm.write(f"StratifiedGroupKFold failed ({e}); falling back to GroupKFold.")
                gkf = GroupKFold(n_splits=cv_folds)
                splits = list(gkf.split(X_idx, groups=groups))
                cv_mode = "GroupKFold"
                leakage_notes.append(f"Stratify fallback: {e}")

            for fold_i, (tr, va) in enumerate(splits, start=1):
                tr_set = {embed_for_cv[i] for i in tr}
                va_set = {embed_for_cv[i] for i in va}
                vid_tr = {id_to_row[s]["_video_id"] for s in tr_set}
                vid_va = {id_to_row[s]["_video_id"] for s in va_set}
                leak = sorted(str(v) for v in (vid_tr & vid_va))
                leak_s = "none OK" if not leak else f"LEAK: {len(leak)} videos"
                if leak:
                    leakage_notes.append(f"Fold {fold_i}: {leak[:20]}")
                u_cats = len({out_fields[s]["cat_id"] for s in va_set})
                u_vid = len({id_to_row[s]["_video_id"] for s in va_set})
                fold_rows.append((fold_i, len(va_set), u_cats, u_vid, leak_s))
                if u_cats < cv_min:
                    leakage_notes.append(
                        f"Fold {fold_i}: unique cat_ids in val ({u_cats}) < cv_min_cats_per_fold ({cv_min})"
                    )
    except Exception as e:
        cv_mode = "failed"
        cv_lines.append(str(e))

    # Plots
    sns.set_theme(style="whitegrid")
    reports_dir.mkdir(parents=True, exist_ok=True)

    sizes = list(cluster_sizes.values())
    if sizes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sizes, bins=min(50, max(5, len(set(sizes)))), color="steelblue", edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel("Cluster size (snippets)")
        ax.set_ylabel("Count (log)")
        n_singleton_cats = sum(1 for c, sz in cluster_sizes.items() if sz == 1)
        n_gt5 = sum(1 for sz in sizes if sz > 5)
        ax.set_title("Cat cluster size distribution")
        ax.text(
            0.98,
            0.95,
            f"{n_singleton_cats} singleton cats\n{n_gt5} cats with >5 snippets",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(reports_dir / "cat_cluster_size_distribution.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["Rule2 breed+coat", "Rule3 same_ch", "Rule4 x-ch bbox", "Rule5 x-ch fb"]
    arrs = [stats.sims_rule2, stats.sims_rule3, stats.sims_rule4, stats.sims_rule5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    any_hist = False
    for arr, lab, c in zip(arrs, labels, colors):
        if arr:
            ax.hist(arr, bins=30, alpha=0.45, label=lab, color=c, density=False)
            any_hist = True
    for xv, lab in [
        (th_breed, "0.82"),
        (th_same_ch, "0.88"),
        (th_xbb, "0.95"),
        (th_xfb, "0.97"),
    ]:
        ax.axvline(xv, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    if any_hist:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_title("Similarity distribution for merged pairs by rule")
    fig.tight_layout()
    fig.savefig(reports_dir / "similarity_distribution_by_rule.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    rules = [
        "Rule1 same_video",
        "Rule2 breed+coat",
        "Rule3 same_channel",
        "Rule4 cross_bbox",
        "Rule5 cross_fallback",
        "GPT verified",
    ]
    counts = [
        stats.same_video_merges,
        stats.breed_coat_merges,
        stats.same_channel_merges,
        stats.cross_channel_bbox_merges,
        stats.cross_channel_fallback_merges,
        stats.gpt_verified_merges,
    ]
    y_pos = np.arange(len(rules))
    ax.barh(y_pos, counts, color="teal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rules)
    ax.set_xlabel("Merge count")
    ax.set_title("Merges by rule")
    fig.tight_layout()
    fig.savefig(reports_dir / "merges_by_rule.png", dpi=150)
    plt.close(fig)

    if fold_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [r[0] for r in fold_rows]
        ax.bar(xs, [r[2] for r in fold_rows], color="slategray")
        ax.axhline(cv_min, color="crimson", linestyle="--", label=f"min ({cv_min})")
        ax.set_xticks(xs)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Unique cat_ids in val")
        ax.set_title("Cat ID balance across CV folds (val)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / "cat_id_fold_balance.png", dpi=150)
        plt.close(fig)

    # CSV
    csv_rows: list[dict[str, Any]] = []
    for sn in snippets_flat:
        sid = str(sn["id"])
        of = out_fields[sid]
        cid = of["cat_id"]
        csv_rows.append(
            {
                "snippet_id": sid,
                "cat_id": cid,
                "cat_id_method": of["cat_id_method"],
                "cluster_size": cluster_sizes.get(cid, 1),
                "video_id": sn.get("_video_id"),
                "channel": sn.get("_channel"),
                "platform": sn.get("_platform"),
                "behavioral_category": sn.get("_behavioral_category"),
                "bbox_found": of["bbox_found"],
                "yolo_conf": of["yolo_conf"],
                "clip_similarity_to_nearest": of["clip_similarity_to_nearest"],
                "gpt_verified": of["gpt_verified"],
                "breed_guess": gpt_field(sn, "cats", "primary_cat", "breed_guess"),
                "coat_pattern": gpt_field(sn, "cats", "primary_cat", "coat_pattern"),
            }
        )
    csv_rows.sort(key=lambda r: (-int(r["cluster_size"]), r["snippet_id"]))

    import csv

    assign_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(assign_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "snippet_id",
                "cat_id",
                "cat_id_method",
                "cluster_size",
                "video_id",
                "channel",
                "platform",
                "behavioral_category",
                "bbox_found",
                "yolo_conf",
                "clip_similarity_to_nearest",
                "gpt_verified",
                "breed_guess",
                "coat_pattern",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)

    # Report
    unique_cat_ids = len(set(out_fields[s]["cat_id"] for s in out_fields))
    singleton_cats = sum(1 for c, sz in cluster_sizes.items() if sz == 1)
    multi_clusters = sum(1 for sz in cluster_sizes.values() if sz > 1)
    multi_snips = sum(sz for sz in cluster_sizes.values() if sz > 1)
    largest = max(cluster_sizes.items(), key=lambda kv: kv[1]) if cluster_sizes else ("", 0)

    ch_src_order = ("channel_id", "channel", "uploader_id", "author_id", "video_id_fallback")
    channel_source_counts: defaultdict[str, int] = defaultdict(int)
    for sid in work_ids:
        channel_source_counts[str(id_to_row[sid].get("_channel_source", "?"))] += 1
    n_work_ch = max(1, len(work_ids))
    ch_src_lines: list[str] = [
        "CHANNEL (string used as _channel on every snippet)",
        "-----------------------------------------------------",
        "Source is the first non-empty field on the *parent video row*, in order:",
        "  channel_id > channel > uploader_id > author_id; otherwise video_id (fallback).",
        "Values from \"channel\" (e.g. display names) are used when they appear before",
        "a usable channel_id; stable channel_id is preferred when present.",
        "",
        f"_channel_source among work-set embeddable snippets (N={len(work_ids)}):",
    ]
    for k in ch_src_order:
        n = int(channel_source_counts.get(k, 0))
        ch_src_lines.append(f"  {k}: {n} ({100.0 * n / n_work_ch:.1f}%)")
    for k in sorted(channel_source_counts.keys()):
        if k in ch_src_order:
            continue
        n = int(channel_source_counts[k])
        ch_src_lines.append(f"  {k} (unexpected): {n} ({100.0 * n / n_work_ch:.1f}%)")
    ch_src_lines.extend(
        [
            "",
            "WHY CHANNEL CAN STILL PRODUCE LOW Rule 2–3 OR GPT COUNTS",
            "---------------------------------------------------------",
            "Channels are always compared; they are not silently dropped. Counts depend on",
            "how _channel lines up across *pairs* of snippets, not on whether the field exists:",
            "",
            "  Rule 1 — same video_id: merges all snippets from one video (no _channel test).",
            "",
            "  Rules 2–3 — need the *same* _channel, high CLIP similarity, and different",
            "  components; Rule 2 also needs matching GPT breed + coat. If many parents",
            "use video_id fallback as _channel, different videos almost always have different",
            "_channel, so cross-video \"same channel\" merges are rare (Rule 1 already merged",
            "same-video snippets). That often yields 0 for Rules 2–3 even when \"channel\" is set.",
            "",
            "  Rules 4–5 — need *different* _channel (cross-channel). With video_id fallback,",
            "  different videos usually differ, so these rules can still merge (see Rule 4).",
            "",
            "  GPT verification — only pairs with same _channel and *different* video_id;",
            "  if _channel is always per-video (fallback), the candidate set is empty (0 pairs).",
            "  Use a stable id shared across videos (e.g. channel_id) on the manifest to enable it.",
            "",
        ]
    )

    lines_r = [
        "========================================================",
        "CAT ID ASSIGNMENT REPORT",
        "========================================================",
        "",
        "INPUT",
        "-----",
        f"Total snippets:         {n_total}",
        f"Embeddable:             {stats.embeddable} ({pct(stats.embeddable)})",
        f"Skipped (no cat):       {stats.skipped_no_cat} ({pct(stats.skipped_no_cat)})",
        f"Skipped (unsuitable):   {stats.skipped_unsuitable} ({pct(stats.skipped_unsuitable)})",
        f"Extraction failed:      {len(extract_failed_ids)} "
        f"({100.0 * len(extract_failed_ids) / max(1, stats.embeddable):.1f}% of embeddable in work set)",
        "",
    ]
    lines_r.extend(ch_src_lines)
    lines_r.extend(
        [
        "THIS RUN — FRAME / CLIP I/O (not clustering quality)",
        "------------------------------------------------------",
        f"YOLO detections:        {yolo_hit} "
        f"({100.0 * yolo_hit / max(1, len(work_ids)):.1f}% of work-set embeddable)",
        f"Fallback (full frame):  {yolo_fb}",
        f"Bbox RGB from disk:    {stats.bbox_crop_cache_hits}",
        f"Bbox RGB extracted:    {stats.bbox_crop_cache_misses}",
        f"CLIP loaded from cache: {stats.from_cache}",
        f"CLIP newly computed:   {stats.newly_computed}",
        "",
        "Cache note: 'from disk' counts bbox reuse in this process only. The first",
        "completed run after clearing the bbox npz shows 0 here while writing the file;",
        "the next run with the same cat_id settings should show a large 'from disk' count.",
        "Clustering below matches a prior run whenever CLIP inputs and merge rules match.",
        "",
        "CLUSTERING",
        "------------",
        f"Unique cat_ids:         {unique_cat_ids}",
        f"Singleton cats:         {singleton_cats}",
        f"Multi-snippet cats:     {multi_clusters} cats covering {multi_snips} snippets",
        f"Largest cluster:        {largest[1]} snippets (cat_id={largest[0]})",
        "",
        "Merges by rule (union-find; order is priority, not chronological):",
        f"  Rule 1 same_video:              {stats.same_video_merges} merges",
        f"  Rule 2 channel+breed+coat:      {stats.breed_coat_merges} merges",
        f"  Rule 3 same_channel:            {stats.same_channel_merges} merges",
        f"  Rule 4 cross_channel_bbox:      {stats.cross_channel_bbox_merges} merges",
        f"  Rule 5 cross_channel_fallback:  {stats.cross_channel_fallback_merges} merges",
        f"  GPT verified:                   {stats.gpt_verified_merges} merges (${stats.gpt_total_cost:.2f})",
        "",
        "GPT VERIFICATION",
        "----------------",
        f"Pairs evaluated:    {stats.gpt_pairs_sent}",
        f"Verified same:      {stats.gpt_verified_same} -> merged",
        f"Verified different: {stats.gpt_verified_different} -> kept separate",
        f"Uncertain:          {stats.gpt_uncertain} -> kept separate (conservative)",
        f"Parse errors:       {stats.gpt_parse_errors}",
        f"Total cost:         ${stats.gpt_total_cost:.2f}",
        "",
        "CV VALIDATION",
        "-------------",
        f"Mode: {cv_mode}",
        f"CV VALIDATION ({cv_folds}-fold, groups=cat_id)",
        "Fold | Val snippets | Unique cats | Unique videos | Leakage",
        "-----+--------------+-------------+---------------+----------",
        ]
    )
    for fr in fold_rows:
        lines_r.append(f"  {fr[0]:3d} | {fr[1]:12d} | {fr[2]:11d} | {fr[3]:13d} | {fr[4]}")
    lines_r.append("")
    if leakage_notes:
        lines_r.append("Issues / warnings:")
        for ln in leakage_notes:
            lines_r.append(f"  - {ln}")
    else:
        lines_r.append("Leakage detected: none OK")
    lines_r.extend(
        [
            "",
            "RE-RUNNING WITHOUT REDOING YOLO OR CLIP",
            "----------------------------------------",
            "Use the on-disk bbox crop npz and CLIP embedding npz: run the same script",
            "without --rebuild-crop-cache and without --rebuild-cache (see module docstring).",
            "Keep cat_id.embedding_cache and cat_id.bbox_crop_cache paths in config.yaml",
            "pointing at the files from a completed run; same cat_id signature as when they were built.",
            "",
            "========================================================",
        ]
    )

    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text("\n".join(lines_r) + "\n", encoding="utf-8")

    # Write manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_bits = [
        f"snippets_total={n_total}",
        f"embedded={len(embedded_ids)}",
        f"unique_cat_ids={unique_cat_ids}",
    ]
    with open(out_manifest, "w", encoding="utf-8") as mf:
        for rec in records:
            new_snips = []
            for sn in rec.get("snippets") or []:
                if not isinstance(sn, dict):
                    continue
                sid = sn.get("id")
                if not isinstance(sid, str):
                    continue
                o = dict(sn)
                ex = out_fields.get(sid)
                if ex:
                    o.update(ex)
                new_snips.append(o)
            out_rec = dict(rec)
            out_rec["snippets"] = new_snips
            out_rec["cleaning_stage"] = "03_cat_id"
            mf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        mf.write(
            "# "
            + json.dumps(
                {
                    "stage": "03_cat_id",
                    "summary": summary_bits,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
