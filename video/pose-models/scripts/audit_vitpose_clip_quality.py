#!/usr/bin/env python3
"""
Per-clip ViTPose quality from ``(T, 17, 3)`` float arrays (x, y, conf) and optional
``pose_mask`` (T,) bool — same layout as P4 / ``data_engineering.PoseDataset``.

**Signals computed (real frames only, after mask):**

- ``mean_conf`` — mean keypoint confidence over all real frames and joints
- ``frac_visible`` — fraction of (frame, joint) with conf > threshold (default 0.3)
- ``zero_frame_rate`` — fraction of *real* frames whose mean joint conf < 0.01
- ``xy_std`` — mean over joints of std of (x, y) across time (motion / stability)
- ``head_conf`` — mean conf on joints 0–4 (nose / eyes / ears; AP-10K–style 17-cat layout)
- ``body_conf`` — mean conf on joints 5–6 (left/right shoulder)
- ``y_range`` — max y minus min y over all real (x, y) — useful for “subject size in box”

Writes a CSV and optional overlay histograms, plus a mean_conf **retention** table
for threshold planning (no manifest mutation — see repo rule on non-destructive pipelines).

Example:
  python model_training_v2/scripts/audit_vitpose_clip_quality.py
  python model_training_v2/scripts/audit_vitpose_clip_quality.py --limit 500 --plot
  python model_training_v2/scripts/audit_vitpose_clip_quality.py --plot custom/out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO / "video" / "pose-models"
for _p in (REPO, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_loading import  LABEL_COLORS, load_pose_extraction_index


def _resolve(repo: Path, p: str | None) -> Path | None:
    if not p or not str(p).strip():
        return None
    path = Path(p)
    return path if path.is_absolute() else (repo / path).resolve()


def _pose_and_mask_for_row(
    repo: Path,
    row: dict,
    index_by_id: dict[str, dict] | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """
    Return (pose, mask_bool, path_used_for_pose).
    Prefers ``pose_extraction_index`` when provided and status is done.
    """
    sid = str(row.get("snippet_id", "")).strip()
    pose_path = None
    mask_path = None
    if index_by_id and sid in index_by_id:
        ent = index_by_id[sid]
        if str(ent.get("status", "")).lower() == "done":
            pose_path = _resolve(repo, ent.get("pose_path"))
            mask_path = _resolve(repo, ent.get("pose_mask_path"))
    if pose_path is None or not pose_path.is_file():
        pose_path = _resolve(repo, row.get("pose_path"))
        mask_path = _resolve(repo, row.get("pose_mask_path"))
    if pose_path is None or not pose_path.is_file():
        return np.array([]), np.array([]), None
    pose = np.load(pose_path)
    if mask_path is not None and mask_path.is_file():
        mask = np.load(mask_path)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.shape[0] != pose.shape[0]:
            mask = np.ones(pose.shape[0], dtype=bool)
    else:
        mask = np.ones(pose.shape[0], dtype=bool)
    return pose, mask, str(pose_path.relative_to(repo)) if repo in pose_path.parents else str(pose_path)


def compute_clip_metrics(
    pose: np.ndarray,
    mask: np.ndarray,
    *,
    visible_thr: float = 0.30,
) -> dict[str, float] | None:
    """Return metric dict or None if pose is unusable."""
    if pose.size == 0 or pose.ndim != 3 or pose.shape[-1] < 3:
        return None
    if pose.shape[0] == 0:
        return None
    if mask.shape[0] != pose.shape[0]:
        return None
    real = pose[mask]
    if real.shape[0] == 0:
        return None
    conf = real[:, :, 2].astype(np.float64, copy=False)
    xy = real[:, :, :2].astype(np.float64, copy=False)
    mean_conf = float(np.mean(conf))
    frac_visible = float(np.mean(conf > float(visible_thr)))
    per_frame_mconf = np.mean(conf, axis=1)
    zero_frames = float(np.mean(per_frame_mconf < 0.01))
    xy_std = float(np.mean(np.std(xy, axis=0)))
    n_j = min(17, conf.shape[1])
    head_hi = min(5, n_j)
    head_conf = float(np.mean(conf[:, :head_hi])) if head_hi else float("nan")
    b0, b1 = min(5, n_j), min(7, n_j)
    body_conf = float(np.mean(conf[:, b0:b1])) if b1 > b0 else float("nan")
    y = xy[:, :, 1]
    y_range = float(np.max(y) - np.min(y)) if y.size else 0.0
    return {
        "mean_conf": mean_conf,
        "frac_visible": frac_visible,
        "zero_frame_rate": zero_frames,
        "xy_std": xy_std,
        "head_conf": head_conf,
        "body_conf": body_conf,
        "y_range": y_range,
    }


def _retention_table(df: pd.DataFrame, mean_col: str, label_col: str, pain_val: str) -> str:
    lines: list[str] = [
        "",
        f"Retention by ``{mean_col}`` (rows with mean_conf >= threshold):",
        f"{'Threshold':>10} {'Kept':>8} {'%':>8} {f'{pain_val} kept':>14}",
    ]
    n0 = max(len(df), 1)
    for thresh in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        sub = df[df[mean_col] >= thresh]
        pain_n = int((sub[label_col] == pain_val).sum()) if label_col in sub.columns else 0
        lines.append(
            f"{thresh:>10.2f} {len(sub):>8d} {100.0 * len(sub) / n0:>7.1f}% {pain_n:>14d}"
        )
    return "\n".join(lines)


def _plot_distributions(
    df: pd.DataFrame,
    label_col: str,
    out_path: Path,
    metrics: list[str],
    titles: list[str],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    labels = sorted(df[label_col].dropna().unique().tolist()) if label_col in df.columns else []
    for ax, metric, title in zip(axes.flat, metrics, titles):
        if metric not in df.columns:
            ax.set_title(f"{title} (missing column)")
            continue
        for lab in labels:
            subset = df[df[label_col] == lab][metric].dropna()
            if len(subset) < 1:
                continue
            col = LABEL_COLORS.get(lab, "#888888")
            ax.hist(subset, bins=40, alpha=0.4, label=str(lab)[:18], color=col, density=True)
        ax.set_title(title)
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("ViTPose quality metrics by class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "manifests" / "final_dataset_v2.jsonl",
    )
    ap.add_argument(
        "--pose-index",
        type=Path,
        default=None,
        help="Optional JSONL (snippet_id -> pose_path) like pose_extraction_index; overrides manifest paths when done.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO / "video" / "pose-models" / "config.yaml",
        help="Used only to read data.pose_extraction_index if --pose-index is omitted.",
    )
    ap.add_argument("--label-field", default="final_label_5", help="Column for coloring (e.g. final_label_5, audio_label_10).")
    ap.add_argument("--pain-label", default="Paining", help="Class name for pain column in retention table.")
    ap.add_argument("--visible-thr", type=float, default=0.30, help="Threshold for frac_visible.")
    ap.add_argument("--limit", type=int, default=0, help="Max clips (0 = all suitable rows with pose).")
    ap.add_argument(
        "--include-unsuitable",
        action="store_true",
        help="Include rows where suitable_for_training is not true.",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=REPO / "src" / "dataset_construction" / "reports" / "vitpose_clip_quality.csv",
    )
    _default_plot = REPO / "src" / "dataset_construction" / "reports" / "vitpose_quality_hist.png"
    ap.add_argument(
        "--plot",
        type=Path,
        nargs="?",
        const=_default_plot,
        default=None,
        metavar="PATH",
        help=(
            "Save a 2x3 histogram grid (matplotlib, seaborn). "
            f"Use ``--plot`` alone for default: {_default_plot.relative_to(REPO)}. "
            "Or pass a path: ``--plot out.png``."
        ),
    )
    args = ap.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else REPO / args.manifest
    index_by_id: dict[str, dict] | None = None
    if args.pose_index is not None:
        p = args.pose_index if args.pose_index.is_absolute() else REPO / args.pose_index
        if not p.is_file():
            print(f"pose index not found: {p}", file=sys.stderr)
            return 1
        try:
            rel = p.resolve().relative_to(REPO.resolve())
        except ValueError:
            rel = p.resolve()
        cfg = {"data": {"pose_extraction_index": rel.as_posix()}}
        index_by_id = load_pose_extraction_index(cfg) or None
    else:
        try:
            cfg = yaml.safe_load((args.config).read_text(encoding="utf-8"))
            index_by_id = load_pose_extraction_index(cfg) or None
        except Exception:
            index_by_id = None

    rows_out: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if args.limit and len(rows_out) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not args.include_unsuitable and not row.get("suitable_for_training"):
                continue
            pose, mask, used = _pose_and_mask_for_row(REPO, row, index_by_id)
            if used is None:
                continue
            m = compute_clip_metrics(pose, mask, visible_thr=args.visible_thr)
            if m is None:
                continue
            lab = row.get(args.label_field)
            rows_out.append(
                {
                    "snippet_id": row.get("snippet_id"),
                    args.label_field: lab,
                    "platform": row.get("platform"),
                    "pose_path_used": used,
                    "n_real_frames": int(mask.sum()),
                    **m,
                }
            )

    if not rows_out:
        print("No clips with loadable pose arrays; check manifest paths and --pose-index.", file=sys.stderr)
        return 1

    quality_df = pd.DataFrame(rows_out)
    out_csv = args.output_csv if args.output_csv.is_absolute() else REPO / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(quality_df)} rows -> {out_csv}")
    print(quality_df[[c for c in quality_df.columns if c in ("mean_conf", "frac_visible", "zero_frame_rate", "xy_std", "head_conf", "body_conf", "y_range")]].describe())

    if args.label_field in quality_df.columns:
        print(_retention_table(quality_df, "mean_conf", args.label_field, args.pain_label))

    if args.plot is not None:
        plot_path = args.plot if args.plot.is_absolute() else REPO / args.plot
        metrics = [
            "mean_conf",
            "frac_visible",
            "zero_frame_rate",
            "head_conf",
            "body_conf",
            "y_range",
        ]
        titles = [
            "Mean confidence",
            f"Fraction visible (>{args.visible_thr})",
            "Zero-frame rate",
            "Head keypoint conf (j 0–4)",
            "Shoulder conf (j 5–6)",
            "Y coordinate range",
        ]
        _plot_distributions(quality_df, args.label_field, plot_path, metrics, titles)
        print(f"Saved plot -> {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
