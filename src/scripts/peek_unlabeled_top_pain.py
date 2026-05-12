#!/usr/bin/env python3
"""
Rank unlabeled clips by ensemble Pain probability for one binary sweep model (e.g. F4).

Example:
  python scripts/peek_unlabeled_top_pain.py \\
    --binary-results-dir runs/pseudo_loop_20260402_213707/iteration_0/binary_results \\
    --model-id F4 \\
    --top-k 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from finetuning_after_ssl import drop_snip_n1_symlink_clone_records
from models_training import (
    DEFAULT_POSE_COORD_DIMS,
    MODEL_REGISTRY,
    POSE_COORD_DIMS_WITH_KINEMATICS,
)
from pseudo_label_loop import (
    _load_jsonl_records,
    _pick_device,
    _split_human_unlabeled,
    ensemble_softmax_unlabeled,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--binary-results-dir",
        type=Path,
        required=True,
        help="Folder containing F4_Fusion-TCN-3Frame/ etc. (iteration_*/binary_results)",
    )
    p.add_argument("--manifest", type=Path, default=Path("data/dataset/final_dataset.jsonl"))
    p.add_argument("--model-id", type=str, default="F4", help="e.g. F4, P2, F2")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-kinematics", action="store_true")
    args = p.parse_args()

    mid = args.model_id.strip()
    if mid not in MODEL_REGISTRY:
        print(f"Unknown model id {mid!r}; choose one of: {sorted(MODEL_REGISTRY.keys())}", file=sys.stderr)
        return 1
    mname = MODEL_REGISTRY[mid].MODEL_NAME

    all_recs = _load_jsonl_records(Path(args.manifest))
    _, unlabeled = _split_human_unlabeled(all_recs)
    unlabeled = drop_snip_n1_symlink_clone_records(unlabeled)
    # Manifest can list the same stem twice (e.g. labeled/ vs unlabeled/ embedding paths); keep one row.
    _seen: set[str] = set()
    _deduped: list = []
    for r in unlabeled:
        s = r.get("stem", "")
        if not s or s in _seen:
            continue
        _seen.add(s)
        _deduped.append(r)
    if len(_deduped) < len(unlabeled):
        print(
            f"Note: {len(unlabeled) - len(_deduped)} duplicate unlabeled stem row(s) skipped "
            f"(manifest had {len(unlabeled)} unlabeled lines).",
            file=sys.stderr,
        )
    unlabeled = _deduped
    if not unlabeled:
        print("No unlabeled rows after split/dedupe.", file=sys.stderr)
        return 1

    use_kinematics = not args.no_kinematics
    pose_dims = POSE_COORD_DIMS_WITH_KINEMATICS if use_kinematics else DEFAULT_POSE_COORD_DIMS
    device = _pick_device()
    pin_memory = device.type == "cuda"

    br = args.binary_results_dir.resolve()
    if not br.is_dir():
        print(f"Not a directory: {br}", file=sys.stderr)
        return 1

    print(
        f"Model {mid} ({mname}) | unlabeled n={len(unlabeled)} | manifest={args.manifest}\n"
        f"binary_results={br}\n"
    )

    probs, stems = ensemble_softmax_unlabeled(
        binary_results_dir=br,
        model_id=mid,
        model_name=mname,
        unlabeled_records=unlabeled,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        use_kinematics=use_kinematics,
        pose_coord_dims=pose_dims,
    )
    pain = probs[:, 1]
    order = np.argsort(-pain)
    k = min(args.top_k, len(order))

    print(f"Top {k} by Pain softmax (column index 1):\n")
    print(f"{'rank':>4}  {'pain_prob':>10}  {'no_pain':>10}  stem")
    print("-" * 88)
    for rank in range(k):
        j = int(order[rank])
        pp, np_ = float(pain[j]), float(probs[j, 0])
        print(f"{rank + 1:4d}  {pp:10.6f}  {np_:10.6f}  {stems[j]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
