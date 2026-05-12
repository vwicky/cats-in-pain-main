"""
Paths, audio constants, YAMNet-related settings, and compute device (MPS/CPU).

All paths are resolved relative to the ``audio_preclassification_v3`` project root
unless overridden by environment variables or CLI flags.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Project root: parent of ``src/``
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Default V2-style raw data (configurable via CLI / env)
_DEFAULT_DATA = PROJECT_ROOT / ".." / "data" / "audio_cat-classification" / "raw"
DATA_ROOT: Path = Path(
    os.environ.get("AUDIO_PRECLASS_V3_DATA_ROOT", str(_DEFAULT_DATA.resolve()))
).resolve()

RUNS_DIR: Path = PROJECT_ROOT / "runs"

# YAMNet / Google AudioSet frontend
SAMPLE_RATE: int = 16_000

# Binary decision threshold on aggregated P(cat)
DEFAULT_THRESHOLD: float = 0.02

# Patch-level batching inside YAMNet (not file-level; waveforms differ in length)
DEFAULT_PATCH_BATCH_SIZE: int = 32

# Substrings for matching YAMNet class *display names* (case-insensitive).
# "cat" is handled via word-boundary in ``yamnet_runner`` to reduce false positives.
CAT_KEYWORDS: tuple[str, ...] = ("meow", "purr", "caterwaul", "hiss")

# If False, exclude AudioSet class "Roaring cats (lions, tigers)" from the cat bucket
INCLUDE_ROARING_CATS: bool = False

# Within each 0.96s frame: how to combine cat-class probabilities
# "sum" = sum of sigmoid probs; "max" = max across cat-related classes
AGGREGATE_CAT_CLASSES: str = "sum"  # "sum" | "max"

# Hardware: prefer Apple MPS on macOS when available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
