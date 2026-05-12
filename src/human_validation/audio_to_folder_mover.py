#!/usr/bin/env python3
"""
Copy audio files from video_audio_human_validation/cat into cat_audios_copy.
The source folder is not modified.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "This script requires tqdm. Install with: pip install tqdm"
    ) from e

# Extensions treated as audio (lowercase). Add more if needed.
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".webm"}
)

APP_DIR = Path(__file__).resolve().parent
SOURCE_DIR = APP_DIR / "video_audio_human_validation" / "cat"
DEST_DIR = APP_DIR / "cat_audios_copy"


def main() -> int:
    if not SOURCE_DIR.is_dir():
        print(f"Error: source directory does not exist: {SOURCE_DIR}", file=sys.stderr)
        return 1

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    candidates: list[Path] = []
    for p in sorted(SOURCE_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            candidates.append(p)

    total = len(candidates)
    ok = 0
    failed: list[tuple[Path, str]] = []

    for src in tqdm(candidates, desc="Copying audio", unit="file"):
        dest = DEST_DIR / src.name
        try:
            shutil.copy2(src, dest)
            ok += 1
        except OSError as e:
            failed.append((src, str(e)))

    skipped_non_audio = sum(
        1
        for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() not in AUDIO_EXTENSIONS
    )

    print()
    print("--- Copy statistics ---")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Audio files considered: {total}")
    print(f"Successfully copied: {ok}")
    print(f"Failed: {len(failed)}")
    if skipped_non_audio:
        print(
            f"Non-audio files in folder (not copied, e.g. .mp4): {skipped_non_audio}"
        )

    if failed:
        print("\nFailures:")
        for path, err in failed:
            print(f"  {path.name}: {err}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
