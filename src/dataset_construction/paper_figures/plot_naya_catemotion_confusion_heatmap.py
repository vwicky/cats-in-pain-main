#!/usr/bin/env python3
"""Row-normalized CatEmotion val confusion heatmap (fig-21 style).

The checkpoint and per-snippet predictions are not in-repo. This script builds the
integer confusion matrix from the baseline classification report in
``src/human_validation/naya_catemotion_validation_metrics.txt`` (precision, recall,
support per class): diagonal TP = recall × support, predicted column totals =
TP / precision, off-diagonal filled via a balanced transportation LP (min sum of
mass — scipy). The solution matches sklearn's printed report exactly; other LP
objectives / max-entropy flows can yield different off-diagonal patterns.

Outputs (default ``output_pngs/``):
  - fig_23_naya_catemotion_val_confusion_rownorm_heatmap.png
  - table_14_naya_catemotion_val_confusion_counts.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_DIR = Path(__file__).resolve().parent
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from generate_plots import _plot_row_norm_heatmap, apply_publication_style  # noqa: E402
from thesis_extract import LABEL10_ORDER  # noqa: E402

# Values = four-decimal lines from naya_catemotion_validation_metrics.txt (baseline n=593).
_SUPPORTS = np.array([60, 58, 60, 60, 58, 61, 59, 58, 59, 60], dtype=float)
_RECALLS = np.array([0.8667, 0.9655, 0.9333, 0.8833, 0.9655, 0.9672, 0.8814, 0.7759, 0.9831, 0.8500])
_PRECISIONS = np.array([0.8966, 0.9825, 0.9333, 0.8983, 0.9492, 0.8194, 0.9811, 0.7627, 1.0000, 0.8793])

# Fixed feasible reconstruction (same as LP below); used when SciPy is unavailable.
_CM_FALLBACK: np.ndarray | None = np.array(
    [
        [52, 0, 0, 0, 0, 0, 1, 0, 0, 7],
        [2, 56, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 56, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 53, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 56, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 59, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 7, 52, 0, 0, 0],
        [3, 1, 2, 6, 0, 1, 0, 45, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 58, 0],
        [0, 0, 0, 0, 0, 2, 0, 7, 0, 51],
    ],
    dtype=int,
)


def reconstruct_confusion_from_margins() -> np.ndarray:
    """10×10 non-negative integer confusion with sklearn-consistent margins."""
    n = len(LABEL10_ORDER)
    d = np.rint(_RECALLS * _SUPPORTS).astype(int)
    col_tot = np.rint(d.astype(float) / _PRECISIONS).astype(int)

    assert int(_SUPPORTS.sum()) == 593
    assert d.sum() == 538  # 0.9073 × 593
    assert col_tot.sum() == 593

    row_off = (_SUPPORTS - d).astype(int)
    col_off = (col_tot - d).astype(int)
    assert row_off.sum() == col_off.sum()

    try:
        from scipy.optimize import linprog
    except ImportError:
        return _CM_FALLBACK.copy()

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    m = len(pairs)
    p2k = {p: k for k, p in enumerate(pairs)}

    rows_A: list[np.ndarray] = []
    rows_b: list[float] = []
    for i in range(n - 1):
        r = np.zeros(m)
        for j in range(n):
            if j == i:
                continue
            r[p2k[(i, j)]] = 1.0
        rows_A.append(r)
        rows_b.append(float(row_off[i]))

    for j in range(n):
        r = np.zeros(m)
        for i in range(n):
            if i == j:
                continue
            r[p2k[(i, j)]] = 1.0
        rows_A.append(r)
        rows_b.append(float(col_off[j]))

    A_eq = np.vstack(rows_A)
    b_eq = np.array(rows_b)
    c = np.ones(m)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * m, method="highs")
    if not res.success:
        return _CM_FALLBACK.copy()

    x = res.x
    if np.max(np.abs(x - np.rint(x))) > 1e-8:
        return _CM_FALLBACK.copy()

    cm = np.diag(d).astype(int)
    for (i, j), k in p2k.items():
        cm[i, j] = int(round(x[k]))

    if (not np.array_equal(cm.sum(axis=1), _SUPPORTS.astype(int))) or (
        not np.array_equal(cm.sum(axis=0), col_tot)
    ):
        return _CM_FALLBACK.copy()

    return cm


def _write_counts_table(path: Path, cm: np.ndarray) -> None:
    df = pd.DataFrame(cm, index=LABEL10_ORDER, columns=LABEL10_ORDER)
    lines = [
        "CatEmotionModel — NAYA validation confusion (counts)",
        "Rows: true class; columns: predicted class.",
        "Source: reconstructed from baseline classification report "
        "(naya_catemotion_validation_metrics.txt); off-diagonal from LP.",
        "",
        df.to_string(),
        "",
        f"Row sums: {df.sum(axis=1).tolist()}",
        f"Col sums: {df.sum(axis=0).tolist()}",
        f"Total: {int(df.values.sum())}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PAPER_DIR / "output_pngs",
        help="Directory for PNG and counts table",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = reconstruct_confusion_from_margins()
    ct = pd.DataFrame(cm, index=LABEL10_ORDER, columns=LABEL10_ORDER)

    apply_publication_style()
    _plot_row_norm_heatmap(
        ct,
        out_dir,
        "fig_23_naya_catemotion_val_confusion_rownorm_heatmap.png",
        "Cat emotion (NAYA val, n=593) — confusion matrix row-normalized",
        "Predicted class (CatEmotionModel)",
        "True class",
    )
    _write_counts_table(out_dir / "table_14_naya_catemotion_val_confusion_counts.txt", cm)
    print(f"Wrote {out_dir / 'fig_23_naya_catemotion_val_confusion_rownorm_heatmap.png'}")
    print(f"Wrote {out_dir / 'table_14_naya_catemotion_val_confusion_counts.txt'}")


if __name__ == "__main__":
    main()
