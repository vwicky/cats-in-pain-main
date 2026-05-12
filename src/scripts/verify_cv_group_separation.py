#!/usr/bin/env python3
"""
Verify that StratifiedGroupKFold train/val splits have disjoint group IDs.

Groups match finetuning_after_ssl.stratified_labels_and_groups:
  canonical_cv_group_id: video_id (or stem prefix before '_snip'), with
  ``UNKNOWN_<stem>`` ids normalized to the same parent as the bare YouTube id.

Binary classifier and related CV use the same splitter as binary_classifier.py:
  StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).

Usage:
  python scripts/verify_cv_group_separation.py [manifest.jsonl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from finetuning_after_ssl import load_labeled_records, stratified_labels_and_groups

# Must match binary_classifier.BINARY_RUN_SEED (and fusion finetuning group split seed).
CV_SEED = 42
N_SPLITS = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/dataset/final_dataset.jsonl"),
        help="Labeled manifest (default: dataset/final_dataset.jsonl)",
    )
    args = parser.parse_args()

    records = load_labeled_records(args.manifest, include_low_confidence=False, binary_pain=True)
    if not records:
        print(f"No labeled records in {args.manifest}", file=sys.stderr)
        return 1

    # y only affects stratification targets; group IDs are identical for binary_pain True/False.
    binary_pain = True
    y, groups = stratified_labels_and_groups(records, binary_pain=binary_pain)
    y_arr = np.asarray(y, dtype=np.int64)
    groups_arr = np.asarray(groups, dtype=object)

    # Stems must map to a single group (duplicate stems would inflate leakage checks).
    stem_to_group: dict[str, str] = {}
    bad_stems: list[str] = []
    for r, g in zip(records, groups, strict=True):
        stem = r["stem"]
        if stem in stem_to_group and stem_to_group[stem] != g:
            bad_stems.append(stem)
        stem_to_group[stem] = str(g)

    print(f"Records: {len(records)} | CV: StratifiedGroupKFold({N_SPLITS}, shuffle=True, seed={CV_SEED})")
    print("stratified_labels_and_groups(binary_pain=True)  # same groups as binary_pain=False")
    if bad_stems:
        print(f"WARNING: {len(bad_stems)} stems map to multiple group IDs (data issue): {bad_stems[:10]}...")
    else:
        print("Stem → group: each stem maps to exactly one group ID.")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    indices = np.arange(len(records))

    all_ok = True
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(indices, y_arr, groups_arr)):
        fold_num = fold + 1
        tg = set(groups_arr[train_idx])
        vg = set(groups_arr[val_idx])
        inter = tg & vg
        if inter:
            all_ok = False
            sample = sorted(inter)[:15]
            print(f"\nFold {fold_num}: FAIL — {len(inter)} group ID(s) appear in BOTH train and val.")
            print(f"  Example overlap: {sample}")
        else:
            print(
                f"Fold {fold_num}: OK — disjoint groups "
                f"(train {len(tg)} unique groups, val {len(vg)} unique groups, "
                f"{len(train_idx)} train rows, {len(val_idx)} val rows)"
            )

    # Per-fold: no stem in both sides (follows from disjoint groups if stem→group is unique).
    stem_ok = True
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(indices, y_arr, groups_arr)):
        st = {records[i]["stem"] for i in train_idx}
        sv = {records[i]["stem"] for i in val_idx}
        both = st & sv
        if both:
            stem_ok = False
            print(f"\nFold {fold + 1}: stem overlap (unexpected): {len(both)} stems — {list(both)[:10]}...")

    if stem_ok:
        print("\nStem check: no snippet stem appears in both train and val in any fold.")

    if all_ok and stem_ok:
        print(
            "\nConclusion: train and validation sets do not share the same canonical source group "
            "(parent upload id after UNKNOWN_ normalization), so snippets from the same video are not split across folds."
        )
        return 0

    print("\nConclusion: leakage or splitter issue detected — see messages above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
