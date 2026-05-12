"""
Stack ablation analysis — four follow-up experiments on a completed stacker run.

Loads prob_train.csv + prob_val.csv from a prior run of train_p4_pair_stack_logreg.py
and performs, without re-running neural network inference:

  1. 6-voter refit   — drop Warning-Paining and Angry-Paining (both hurt F1 in ablation);
                       refit LogisticRegression on remaining 6 columns; report delta.
  2. Bootstrap CI    — 1000-resample bootstrap of val AUC for 8-model stack, 6-model stack,
                       and single-model baseline (Resting-Paining); report 95% CI.
  3. Single-model baseline — Resting-Paining alone (highest ablation importance) evaluated
                       on the same 148 val rows; direct comparison the thesis committee
                       will ask for first.
  4. Calibration check — Expected Calibration Error (ECE) for all models; comparison
                       reliability diagram overlay; flag overconfidence in high-P regions
                       caused by Warning-Paining's near-degenerate specificity.

All outputs are written to {run_dir}/analysis/.

Usage:
  python model_training_v2/scripts/analyze_stack_ablation.py \\
    --run-dir model_training_v2/runs/p4_stack_logreg_20260425_115412

  # Drop a different set of voters:
  python ... --drop Warning-Paining Angry-Paining HuntingMind-Paining
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Default voters to drop based on ablation results (Warning and Angry both increase F1 when removed)
DEFAULT_DROP = ["Warning-Paining", "Angry-Paining"]

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _prob_col(pair_name: str) -> str:
    return "prob_" + pair_name.replace("-", "_").replace(" ", "_")


def _fit_lr(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> object:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    lr.fit(X_train, y_train)
    return lr


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision_pain": round(prec, 4),
        "auc_roc": round(auc, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _bootstrap_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low_2.5, ci_high_97.5)."""
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng(42)
    point = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    n = len(y_true)
    boot_aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot_aucs.append(float(roc_auc_score(yt, yp)))
        except ValueError:
            pass
    if not boot_aucs:
        return point, float("nan"), float("nan")
    arr = np.array(boot_aucs)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (uniform binning)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        frac = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece_sum += mask.sum() * abs(frac - conf)
    return float(ece_sum / n) if n > 0 else float("nan")


def _cal_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
    return np.array(mean_pred), np.array(frac_pos)


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------

def write_comparison_report(
    out_path: Path,
    run_dir: Path,
    all_metrics: dict[str, dict],
    bootstrap_ci: dict[str, tuple],
    pair_names_8: list[str],
    pair_names_6: list[str],
    dropped: list[str],
    coef_6: list[tuple[str, float]],
    calibration: dict[str, float],
    n_boot: int,
) -> None:
    lines: list[str] = []

    def h(t: str) -> None:
        lines.append("")
        lines.append(t)
        lines.append("=" * len(t))

    def p(t: str = "") -> None:
        lines.append(t)

    h("Stack Ablation Analysis Report")
    p(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    p(f"Source run: {run_dir}")
    p(f"Dropped for 6-voter: {dropped}")

    h("1. Metric Comparison (val set)")
    headers = ["Model", "AUC", "Macro-F1", "Sens", "Spec", "Prec"]
    p(f"  {'Model':<28}  {'AUC':>6}  {'F1':>6}  {'Sens':>6}  {'Spec':>6}  {'Prec':>6}")
    p(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name, m in all_metrics.items():
        p(
            f"  {name:<28}  {m.get('auc_roc', float('nan')):>6.4f}  "
            f"{m.get('macro_f1', 0):>6.4f}  {m.get('sensitivity', 0):>6.4f}  "
            f"{m.get('specificity', 0):>6.4f}  {m.get('precision_pain', 0):>6.4f}"
        )

    h("2. Bootstrap 95% CI — val AUC (n=1000 resamples)")
    p(f"  {'Model':<28}  {'Point':>7}  {'CI-Low':>8}  {'CI-High':>9}")
    p(f"  {'-'*28}  {'-'*7}  {'-'*8}  {'-'*9}")
    for name, (pt, lo, hi) in bootstrap_ci.items():
        p(f"  {name:<28}  {pt:>7.4f}  {lo:>8.4f}  {hi:>9.4f}")
    p("")
    p("Interpretation: overlapping CIs indicate the difference is not reliably")
    p("detectable at this sample size (n_val=148).")

    h("3. 6-voter Logistic Regression Coefficients")
    p(f"Dropped: {', '.join(dropped)}")
    p(f"  {'Pair':<28}  {'Coef':>8}  {'|Coef|':>8}  {'Rank':>5}")
    p(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*5}")
    for rank, (pn, c) in enumerate(
        sorted(coef_6, key=lambda x: abs(x[1]), reverse=True), 1
    ):
        p(f"  {pn:<28}  {c:>8.4f}  {abs(c):>8.4f}  {rank:>5}")

    h("4. Calibration Analysis (Expected Calibration Error)")
    p(f"  {'Model':<28}  {'ECE':>8}")
    p(f"  {'-'*28}  {'-'*8}")
    for name, ece_val in calibration.items():
        p(f"  {name:<28}  {ece_val:>8.4f}")
    p("")
    p("Lower ECE = better calibrated.  ECE > 0.1 is poor for clinical use.")
    p("If the 8-voter ECE is substantially higher than the 6-voter, Warning's")
    p("high P(Paining) is biasing the ensemble toward overconfidence.")

    h("5. Interpretation")
    m8 = all_metrics.get("8-voter stack", {})
    m6 = all_metrics.get("6-voter stack", {})
    m1 = all_metrics.get("Resting-Paining alone", {})
    delta_f1 = m6.get("macro_f1", 0) - m8.get("macro_f1", 0)
    delta_auc = m6.get("auc_roc", 0) - m8.get("auc_roc", 0)
    p(f"  8-voter vs 6-voter: delta_macro_F1={delta_f1:+.4f}  delta_AUC={delta_auc:+.4f}")
    if delta_f1 > 0:
        p("  => Dropping Warning and Angry IMPROVES the ensemble. Report 6-voter as")
        p("     the primary model in the thesis.")
    elif delta_f1 < -0.01:
        p("  => 8-voter is stronger. The ablation improvement on individual removal")
        p("     does not extend to joint removal (feature interaction).")
    else:
        p("  => Difference within noise. Either configuration is defensible.")
    p("")
    p(f"  Single-model (Resting-Paining) vs 8-voter: delta_AUC={m8.get('auc_roc', 0) - m1.get('auc_roc', 0):+.4f}")
    if m8.get("auc_roc", 0) > m1.get("auc_roc", 0) + 0.01:
        p("  => Ensemble adds measurable value over the single best sub-classifier.")
    else:
        p("  => Ensemble AUC is within noise of the single best sub-classifier at this n.")
        p("     Discuss sample size limitations — the ensemble may generalise better.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc ablation analysis on a completed stacker run (no re-inference)."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the p4_stack_logreg_* run directory.",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        default=DEFAULT_DROP,
        metavar="PAIR",
        help="Pair names to drop for the 6-voter refit (default: Warning-Paining Angry-Paining).",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default: 1000).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = REPO_ROOT / args.run_dir if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")

    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load saved probabilities
    # ------------------------------------------------------------------ #
    train_df = pd.read_csv(run_dir / "prob_train.csv")
    val_df = pd.read_csv(run_dir / "prob_val.csv")

    # Discover all pair columns (anything starting with prob_ except prob_stack)
    prob_cols = [c for c in val_df.columns if c.startswith("prob_") and c != "prob_stack"]
    pair_names_8 = [c[len("prob_"):].replace("_", "-") for c in prob_cols]
    # Collapse double-dashes that arise from pairs already containing a dash
    # e.g. prob_Resting_Paining -> Resting-Paining (already correct via single replace)
    # We need the canonical form: replace underscores that are NOT between the two class names
    # Use actual column names directly as-is; pair_name reconstruction is best-effort.
    pair_names_8 = []
    for col in prob_cols:
        stem = col[len("prob_"):]
        # Find the pair: split on '_Paining' suffix if present
        if stem.endswith("_Paining"):
            neg = stem[: -len("_Paining")].replace("_", "")
            pair_names_8.append(f"{neg}-Paining")
        elif stem.startswith("Paining_"):
            pos = stem[len("Paining_"):].replace("_", "")
            pair_names_8.append(f"Paining-{pos}")
        else:
            pair_names_8.append(stem.replace("_", "-"))

    # Reload pair names from stack_meta.json if available (authoritative source)
    meta_path = run_dir / "stack_meta.json"
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        pair_names_8 = meta.get("pair_names", pair_names_8)

    print(f"Pair names: {pair_names_8}")
    print(f"Drop: {args.drop}")

    y_train = train_df["y_true"].values.astype(np.int64)
    y_val = val_df["y_true"].values.astype(np.int64)

    X_train_8 = np.column_stack([train_df[_prob_col(p)] for p in pair_names_8]).astype(np.float32)
    X_val_8 = np.column_stack([val_df[_prob_col(p)] for p in pair_names_8]).astype(np.float32)
    y_prob_stack_8 = val_df["prob_stack"].values
    y_pred_stack_8 = (y_prob_stack_8 >= 0.5).astype(int)

    # ------------------------------------------------------------------ #
    # 1. 6-voter refit
    # ------------------------------------------------------------------ #
    dropped = [d for d in args.drop if d in pair_names_8]
    if len(dropped) != len(args.drop):
        unrecognised = [d for d in args.drop if d not in pair_names_8]
        print(f"WARNING: these drop names not found in pair list: {unrecognised}")
        print(f"Available: {pair_names_8}")

    pair_names_6 = [p for p in pair_names_8 if p not in dropped]
    print(f"6-voter pairs: {pair_names_6}")

    X_train_6 = np.column_stack([train_df[_prob_col(p)] for p in pair_names_6]).astype(np.float32)
    X_val_6 = np.column_stack([val_df[_prob_col(p)] for p in pair_names_6]).astype(np.float32)

    lr6 = _fit_lr(X_train_6, y_train, seed=args.seed)
    y_prob_6 = lr6.predict_proba(X_val_6)[:, 1]
    y_pred_6 = lr6.predict(X_val_6)
    coef_6 = list(zip(pair_names_6, [float(c) for c in lr6.coef_[0]]))

    n_iter_6 = int(lr6.n_iter_[0])
    if n_iter_6 >= 2000:
        print(f"WARNING: 6-voter LR did not converge ({n_iter_6} iterations)")
    else:
        print(f"6-voter LR converged in {n_iter_6} iterations")

    # ------------------------------------------------------------------ #
    # 2. Single-model baseline — Resting-Paining
    # ------------------------------------------------------------------ #
    single_col = _prob_col("Resting-Paining")
    if single_col not in val_df.columns:
        # Fallback: use first pair column
        single_col = prob_cols[0]
        print(f"Resting-Paining not found; using {single_col} as single-model baseline")
    y_prob_single = val_df[single_col].values
    y_pred_single = (y_prob_single >= 0.5).astype(int)

    # ------------------------------------------------------------------ #
    # Metrics for all three
    # ------------------------------------------------------------------ #
    all_metrics: dict[str, dict] = {
        "8-voter stack": _binary_metrics(y_val, y_pred_stack_8, y_prob_stack_8),
        "6-voter stack": _binary_metrics(y_val, y_pred_6, y_prob_6),
        "Resting-Paining alone": _binary_metrics(y_val, y_pred_single, y_prob_single),
    }

    print("\n=== Metric Comparison ===")
    for name, m in all_metrics.items():
        print(f"  {name:<28}  AUC={m['auc_roc']:.4f}  F1={m['macro_f1']:.4f}  "
              f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")

    # ------------------------------------------------------------------ #
    # 3. Bootstrap CI
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(args.seed)
    print(f"\n=== Bootstrap AUC ({args.n_boot} resamples) ===")
    bootstrap_ci: dict[str, tuple[float, float, float]] = {}

    for label, y_prob in [
        ("8-voter stack", y_prob_stack_8),
        ("6-voter stack", y_prob_6),
        ("Resting-Paining alone", y_prob_single),
    ]:
        pt, lo, hi = _bootstrap_auc(y_val, y_prob, n_boot=args.n_boot, rng=rng)
        bootstrap_ci[label] = (pt, lo, hi)
        print(f"  {label:<28}  AUC={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------ #
    # 4. Calibration (ECE + comparison plot)
    # ------------------------------------------------------------------ #
    calibration: dict[str, float] = {}
    cal_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for label, y_prob in [
        ("8-voter stack", y_prob_stack_8),
        ("6-voter stack", y_prob_6),
        ("Resting-Paining alone", y_prob_single),
    ]:
        ece_val = _ece(y_val, y_prob)
        calibration[label] = round(ece_val, 5)
        cal_curves[label] = _cal_curve_data(y_val, y_prob)
        print(f"  ECE [{label}] = {ece_val:.4f}")

    # ------------------------------------------------------------------ #
    # Calibration plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"8-voter stack": "tab:blue", "6-voter stack": "tab:orange",
               "Resting-Paining alone": "tab:green"}
    markers = {"8-voter stack": "s", "6-voter stack": "^", "Resting-Paining alone": "o"}

    for label, (mean_pred, frac_pos) in cal_curves.items():
        if len(mean_pred):
            ece_val = calibration[label]
            ax.plot(mean_pred, frac_pos, f"{markers[label]}-",
                    color=colors[label],
                    label=f"{label} (ECE={ece_val:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted P(Paining)")
    ax.set_ylabel("Fraction of Paining samples")
    ax.set_title("Reliability Diagram — Model Comparison")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Bootstrap AUC distribution plot
    # ------------------------------------------------------------------ #
    rng2 = np.random.default_rng(args.seed)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_b = ["tab:blue", "tab:orange", "tab:green"]
    labels_b = ["8-voter stack", "6-voter stack", "Resting-Paining alone"]
    probs_b = [y_prob_stack_8, y_prob_6, y_prob_single]
    from sklearn.metrics import roc_auc_score as _auc

    for label, y_prob, color in zip(labels_b, probs_b, colors_b):
        boot_aucs = []
        n = len(y_val)
        for _ in range(args.n_boot):
            idx = rng2.integers(0, n, size=n)
            yt = y_val[idx]; yp = y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                boot_aucs.append(float(_auc(yt, yp)))
            except ValueError:
                pass
        if boot_aucs:
            ax.hist(np.array(boot_aucs), bins=40, alpha=0.5, color=color, label=label)

    ax.set_xlabel("Bootstrap AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Bootstrap AUC Distribution (n={args.n_boot} resamples, val n=148)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_auc_distribution.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    coef6_df = pd.DataFrame(
        [{"pair_name": pn, "coef": c, "abs_coef": abs(c)} for pn, c in coef_6]
    ).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coef6_df["rank"] = range(1, len(coef6_df) + 1)
    coef6_df.to_csv(out_dir / "logreg_6voter_coefficients.csv", index=False)

    boot_out = {
        label: {"auc": pt, "ci_low": lo, "ci_high": hi, "n_boot": args.n_boot}
        for label, (pt, lo, hi) in bootstrap_ci.items()
    }
    (out_dir / "bootstrap_ci.json").write_text(json.dumps(boot_out, indent=2), encoding="utf-8")

    comparison_metrics_out = {
        "8_voter": all_metrics["8-voter stack"],
        "6_voter": all_metrics["6-voter stack"],
        "single_resting_paining": all_metrics["Resting-Paining alone"],
        "dropped": dropped,
        "pair_names_8": pair_names_8,
        "pair_names_6": pair_names_6,
        "bootstrap_ci": {k: {"auc": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in bootstrap_ci.items()},
        "ece": calibration,
        "lr6_converged": bool(n_iter_6 < 2000),
        "lr6_n_iter": n_iter_6,
        "lr6_coef": {pn: float(c) for pn, c in coef_6},
    }
    (out_dir / "comparison_metrics.json").write_text(
        json.dumps(comparison_metrics_out, indent=2), encoding="utf-8"
    )

    write_comparison_report(
        out_dir / "comparison_report.txt",
        run_dir,
        all_metrics,
        bootstrap_ci,
        pair_names_8,
        pair_names_6,
        dropped,
        coef_6,
        calibration,
        args.n_boot,
    )

    print(f"\n=== Outputs written to: {out_dir} ===")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
