#!/usr/bin/env python3
"""Print keys and shapes for a CLIP *_clip.npz file (validates on-disk format)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz",
        nargs="?",
        type=Path,
        help="Path to .npz (default: first existing clip_path in dataset/final_dataset.jsonl)",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    path = args.npz
    if path is None:
        manifest = repo / "data/dataset/final_dataset.jsonl"
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rel = r.get("clip_path")
                if rel:
                    candidate = repo / rel
                    if candidate.is_file():
                        path = candidate
                        break
        if path is None:
            raise SystemExit("No clip_path found in manifest; pass a .npz path explicitly.")
    path = path.resolve()
    z = np.load(path, allow_pickle=True)
    print(path)
    print("keys:", sorted(z.files))
    for k in z.files:
        a = np.asarray(z[k])
        print(f"  {k}: shape={tuple(a.shape)} dtype={a.dtype}")


if __name__ == "__main__":
    main()
