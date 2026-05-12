#!/usr/bin/env python3
"""
Load one dataset snippet: video frames, separated-cat audio waveform, and labels
from ``src/dataset_construction/manifests/final_dataset_v2.jsonl``.

Examples:
  python scripts/load_snippet_video_audio.py --snippet-id fKHsL63Br8E_snip_0
  python scripts/load_snippet_video_audio.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "src/dataset_construction/manifests/final_dataset_v2.jsonl"


def _resolve_under_repo(p: str | None) -> Path | None:
    if not p or not str(p).strip():
        return None
    path = Path(p)
    if path.is_file():
        return path.resolve()
    cand = REPO_ROOT / p
    if cand.is_file():
        return cand.resolve()
    return None


def find_manifest_row(
    manifest: Path,
    snippet_id: str | None,
) -> dict[str, Any]:
    wanted = (snippet_id or "").strip()
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("snippet_id", ""))
            if not wanted:
                return row
            if sid == wanted:
                return row
    raise FileNotFoundError(
        f"No row for snippet_id={wanted!r} in {manifest}" if wanted else "Empty manifest"
    )


def load_video_bgr(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise OSError(f"No frames decoded: {video_path}")
    return frames, fps


def load_snippet(
    snippet_id: str | None = None,
    *,
    manifest: Path = DEFAULT_MANIFEST,
    audio_sr: int = 16_000,
) -> dict[str, Any]:
    row = find_manifest_row(manifest, snippet_id)
    sid = str(row["snippet_id"])
    vpath = _resolve_under_repo(row.get("video_path"))
    apath = _resolve_under_repo(row.get("audio_path"))
    if vpath is None:
        raise FileNotFoundError(f"Video missing for {sid}: {row.get('video_path')}")
    if apath is None:
        raise FileNotFoundError(f"Audio missing for {sid}: {row.get('audio_path')}")

    frames, fps = load_video_bgr(vpath)
    waveform, sr = librosa.load(str(apath), sr=audio_sr, mono=True)

    labels = {
        "snippet_id": sid,
        "final_label_5": row.get("final_label_5"),
        "final_label_binary": row.get("final_label_binary"),
        "audio_label_5": row.get("audio_label_5"),
        "audio_label_binary": row.get("audio_label_binary"),
        "audio_label_10": row.get("audio_label_10"),
        "audio_confidence": row.get("audio_confidence"),
        "behavioral_category": row.get("behavioral_category"),
        "suitable_for_training": row.get("suitable_for_training"),
    }

    return {
        "snippet_id": sid,
        "video_path": vpath,
        "audio_path": apath,
        "frames_bgr": frames,
        "video_fps": fps,
        "audio_waveform": waveform.astype(np.float32, copy=False),
        "audio_sample_rate": int(sr),
        "labels": labels,
        "manifest_row": row,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--snippet-id",
        default="",
        help="Snippet id (e.g. fKHsL63Br8E_snip_0). Default: first row in the manifest.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSONL manifest (default: final_dataset_v2.jsonl).",
    )
    p.add_argument("--audio-sr", type=int, default=16_000, help="Resample rate for librosa.load.")
    args = p.parse_args()

    sid = args.snippet_id.strip() or None
    try:
        bundle = load_snippet(sid, manifest=args.manifest, audio_sr=args.audio_sr)
    except (OSError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1

    n = len(bundle["frames_bgr"])
    h, w = bundle["frames_bgr"][0].shape[:2]
    a = bundle["audio_waveform"]
    print(
        f"snippet_id={bundle['snippet_id']}\n"
        f"  video: {bundle['video_path']}  ({n} frames @ {w}x{h}, fps={bundle['video_fps']:.3f})\n"
        f"  audio: {bundle['audio_path']}  ({a.shape[0]} samples @ {bundle['audio_sample_rate']} Hz)\n"
        f"  labels: {bundle['labels']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
