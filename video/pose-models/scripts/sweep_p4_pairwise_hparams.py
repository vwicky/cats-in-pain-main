#!/usr/bin/env python3
"""
Hyperparameter **grid** for a **single** P4 pairwise binary class pair (e.g. Paining,Resting).

Default compact grid: 4×4×3×3×2 = **288** runs (tune via comma-separated list flags).

**Parallelization**

* **Default** ``--parallel 1`` — one run at a time; one CUDA/MPS/CPU in use. Best for
  a single local GPU: multiple processes on one GPU usually **slow** training down.
* **``--cuda-devices 0,1,2,3`` + ``--parallel 4``** — one worker per GPU (round-robin),
  the practical way to go faster. Each worker sets ``CUDA_VISIBLE_DEVICES`` before
  ``import torch``.
* **MPS (Apple)**: use ``--parallel 1`` (same physical accelerator).
* **``--force-cpu-parallel N``** — N CPU-only jobs; useful only for smoke tests; slow
  for real ST-GCN training.
* **Resume** — if you stopped mid-sweep, reuse the same output folder with
  ``--resume-from /path/to/existing_run`` and usually ``--skip-existing``. The
  DataFrame is reloaded from ``sweep_dataframe_cache.pkl`` in that folder; the
  tqdm bar covers **only cells still to run** (pending ones without
  ``training/run_summary.json`` for this pair). Do **not** use a new timestamped
  run: that would re-run all 288 cells in a new directory.
* **Paining multi-pair** — use ``--paining-pair-sweep`` to run the same
  hparam grid on every (Paining, other) class pair: **9** by default, or **8** if
  you ``--paining-skip Paining,Resting``.
* **Non-pain multi-pair** — use ``--non-pain-pair-sweep`` for **every** unordered
  class pair **among the 9 non-Pain classes** (excludes ``binary_pain_class``,
  default ``Paining``): C(9,2) = **36** pairs, no Paining in any task.
* Add ``--paining-grid-preset`` for a **2×1×2×2×2 = 16** hparam grid: lr 5e-5, 1e-4;
  cosine; focal 0, 2; batch 4, 8; weight_decay 0.01, 0.05. Add ``--from-scratch`` to
  clear any finetune (``init_weights_path`` / freeze) in the config.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[3]
POSE_MODELS_ROOT = REPO / "video" / "pose-models"
for _p in (REPO, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_p4_hparams_task import  apply_sweep_baseline, run_one_sweep_cell


def _write_sweep_best_artifacts(
    results: list[dict],
    run_dir: Path,
    base_for_yaml: dict,
    *,
    top_k: int,
) -> None:
    """
    Persist the top-K hparam runs by val_macro_f1_binary, plus a merge-ready training
    block for the single best (for reuse in a normal :mod:`run_p4_pairwise` config).
    """
    if not results or int(top_k) < 1:
        return
    rows: list[tuple[dict, float | None]] = []
    for r in results:
        f1 = r.get("val_macro_f1_binary")
        if r.get("error") and str(r.get("error")).strip():
            continue
        if f1 is None:
            continue
        try:
            score = float(f1)
        except (TypeError, ValueError):
            continue
        rows.append((r, score))
    rows.sort(key=lambda x: (-(x[1] if x[1] is not None else -1.0), x[0].get("run_index", 0)))
    if not rows:
        return
    k = int(top_k)
    top = [t[0] for t in rows[:k]]
    (run_dir / "hparams_sweep_top_f1.json").write_text(
        json.dumps(
            {
                "metric": "val_macro_f1_binary (desc; tie: lower run_index)",
                "n": len(top),
                "runs": top,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (run_dir / f"hparams_sweep_top{len(top)}.csv").write_text(
        pd.DataFrame(top).to_csv(index=False),
        encoding="utf-8",
    )
    b0, s0 = rows[0]
    tcfg = copy.deepcopy(base_for_yaml)
    tr = tcfg.setdefault("training", {})
    for key, pkey in (
        ("lr", "lr"),
        ("batch_size", "batch_size"),
        ("weight_decay", "weight_decay"),
        ("focal_gamma", "focal_gamma"),
        ("scheduler", "scheduler"),
    ):
        if pkey in b0 and b0[pkey] is not None:
            tr[key] = b0[pkey]
    (run_dir / "hparams_sweep_best_f1_config.yaml").write_text(
        yaml.dump(tcfg, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "hparams_sweep_best_f1_meta.json").write_text(
        json.dumps(
            {
                "val_macro_f1_binary": s0,
                "note": "Best row by val_macro_f1_binary; ce_ratio < 0.2 can indicate very early 'best' vs total length — inspect, not a bug alone.",
                "hparams": {k: b0.get(k) for k in ("run_index", "pair", "lr", "batch_size", "weight_decay", "focal_gamma", "scheduler", "subdir", "ce_ratio", "ce_ratio_heuristic", "best_epoch", "n_epochs_ran", "output_dir", "val_macro_f1_binary")},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

# Default compact grid
DEFAULT_LRS: tuple = (5e-5, 1e-4, 3e-4, 5e-4)
DEFAULT_BATCH: tuple = (4, 8, 16, 32)
DEFAULT_WD: tuple = (0.001, 0.01, 0.05)
DEFAULT_GAMMA: tuple = (0.0, 2.0, 3.0)
DEFAULT_SCHED: tuple = ("cosine", "plateau")
DEFAULT_SWEEP_EPOCHS: int = 1000
DEFAULT_SWEEP_PATIENCE: int = 100
DEFAULT_GRAD_CLIP: float = 1.0


def _hparam_subdir(idx: int, lr: float, bs: int, wd: float, g: float, sched: str) -> str:
    a = f"gs{idx:04d}_lr{lr:.0e}_bs{bs}_wd{wd}_g{g}_{sched}"
    return a.replace("+", "p")


def _hparam_subdir_pained(
    idx: int, pair_dirname: str, lr: float, bs: int, wd: float, g: float, sched: str
) -> str:
    """
    One grid cell; ``pair_dirname`` is e.g. ``binary__Angry__Paining`` (unique
    for this pair; used for Paining- and non-Pain–multi-pair sweeps).
    """
    t = f"lr{lr:.0e}_bs{bs}_wd{wd}_g{g}_{sched}".replace("+", "p")
    a = f"gs{idx:04d}__{pair_dirname}__{t}"
    return a


def _multiclass_names_for_sweep(base_cfg: dict) -> list[str]:
    c = copy.deepcopy(base_cfg)
    c.setdefault("training", {})["binary_only"] = False
    from data_loading import  get_multiclass_class_names

    return get_multiclass_class_names(c)


def _paining_sweep_tuples(
    class_names: list[str],
    skip_pair: tuple[str, str] | None = None,
) -> list[tuple[str, str]]:
    """(Paining, c) in lex-sorted form for every ``c`` in 10-way except *Paining*."""
    p = "Paining"
    if p not in class_names:
        raise SystemExit("labels must include Paining for --paining-pair-sweep")
    oth = [c for c in sorted(class_names) if c != p]
    out = [tuple(sorted((p, c))) for c in oth]  # len 9
    if skip_pair is not None:
        sk = tuple(sorted(skip_pair))
        out = [t for t in out if t != sk]
    return out


def _non_pain_sweep_tuples(base_cfg: dict) -> list[tuple[str, str]]:
    """
    All C(9,2) = 36 pairs from the 10-way label list, excluding
    ``labels.binary_pain_class`` (and thus only among the 9 *other* classes).
    """
    from class_subset_utils import  iter_sorted_class_pairs

    all_n = _multiclass_names_for_sweep(base_cfg)
    pain = str((base_cfg.get("labels") or {}).get("binary_pain_class", "Paining")).strip()
    if pain not in all_n:
        raise SystemExit(f"binary_pain_class {pain!r} is not in labels.classes")
    rest = sorted(c for c in all_n if c != pain)
    if len(rest) < 2:
        raise SystemExit("need at least 2 non-pain classes for --non-pain-pair-sweep")
    return list(iter_sorted_class_pairs(rest))


def _build_grid(lrs, bss, wds, gams, scheds, *, shuffle: bool, seed: int, max_runs: int) -> list:
    combs = list(product(lrs, bss, wds, gams, scheds))
    if shuffle:
        r = random.Random(seed)
        r.shuffle(combs)
    if max_runs and int(max_runs) < len(combs):
        combs = combs[: int(max_runs)]
    return combs


def _parse_float_list(s: str) -> tuple:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_int_list(s: str) -> tuple:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_list(s: str) -> tuple:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _p4_cell_run_summary_path(parent_run_dir: str | Path, pair: str) -> Path:
    from class_subset_utils import  pair_to_dirname
    from run_stgcn_dlc_pairwise import  _parse_single_pair

    lo, hi = _parse_single_pair(str(pair))
    return Path(parent_run_dir) / pair_to_dirname(lo, hi) / "training" / "run_summary.json"


def _read_sweep_jsonl_deduped(path: Path) -> list[dict[str, Any]]:
    """Rows keyed by ``run_index`` (last line in file wins if duplicate)."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    by_idx: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ri = int(row.get("run_index", -1))
            by_idx[ri] = row
    return [by_idx[k] for k in sorted(by_idx.keys())]


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--config", type=Path, default=REPO / "video" / "pose-models" / "config_p4_pairwise.yaml")
    ap.add_argument(
        "--pair",
        default=None,
        metavar="A,B",
        help="One class pair, e.g. Paining,Resting. Not used with --paining-pair-sweep or --non-pain-pair-sweep.",
    )
    ap.add_argument(
        "--paining-pair-sweep",
        action="store_true",
        help="Run the hparam grid on every (Paining, other) pair: 9 (or 8 with --paining-skip).",
    )
    ap.add_argument(
        "--non-pain-pair-sweep",
        action="store_true",
        help="Run on all C(9,2)=36 pairs among the 9 non-pain classes (excludes binary_pain_class, default Paining).",
    )
    ap.add_argument(
        "--paining-skip",
        default="",
        metavar="A,B",
        help="With --paining-pair-sweep, remove this pair (e.g. Paining,Resting → 8 Paining cells).",
    )
    ap.add_argument(
        "--paining-grid-preset",
        action="store_true",
        help="Set grid to 16: lr 5e-5,1e-4 | cosine | focal 0,2 | batch 4,8 | wd 0.01,0.05 (overrides list flags).",
    )
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="Remove init_weights_path, freeze_backbone_epochs, and unfreeze_lr from the base config.",
    )
    ap.add_argument("--experiment-name", default="p4_hparam_sweep_1pair")
    ap.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Existing sweep run directory (the interrupted run’s timestamped folder). With "
        "--skip-existing, only missing cells are trained. Reloads sweep_dataframe_cache.pkl when "
        "present. A new run without this always uses a new directory and does not see prior cells.",
    )
    ap.add_argument(
        "--sweep-subdir",
        default="hparams_grid",
        help="Under the experiment run directory: all grid subfolders go here",
    )
    ap.add_argument("--lrs", default=",".join(f"{x:g}" for x in DEFAULT_LRS))
    ap.add_argument("--batch-sizes", default=",".join(str(x) for x in DEFAULT_BATCH))
    ap.add_argument("--weight-decays", default=",".join(str(x) for x in DEFAULT_WD))
    ap.add_argument("--focal-gammas", default=",".join(str(x) for x in DEFAULT_GAMMA))
    ap.add_argument(
        "--schedulers",
        default=",".join(DEFAULT_SCHED),
        help="e.g. cosine,plateau,none",
    )
    ap.add_argument("--epochs", type=int, default=DEFAULT_SWEEP_EPOCHS)
    ap.add_argument("--early-stop-patience", type=int, default=DEFAULT_SWEEP_PATIENCE)
    ap.add_argument("--max-runs", type=int, default=0, help="0 = all combinations")
    ap.add_argument("--shuffle-grid", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument(
        "--parallel", type=int, default=1, help="Number of workers (see docstring: use with multiple GPUs)"
    )
    ap.add_argument(
        "--cuda-devices",
        default="",
        help="Comma list e.g. 0,1,2,3 — assigned round-robin when --parallel>1. Empty = all visible in each worker.",
    )
    ap.add_argument(
        "--force-cpu-parallel", action="store_true", help="With --parallel>1, run on CPU (slow)"
    )
    ap.add_argument("--no-kinematics", action="store_true")
    ap.add_argument("--device", default="auto", help="Base device string when not using CUDA per-worker")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--audio-confidence-filter", action="store_true")
    ap.add_argument("--min-audio-confidence", type=float, default=0.7)
    ap.add_argument("--audio-confidence-field", default="audio_confidence")
    ap.add_argument("--dry-list", action="store_true", help="Print grid size and exit")
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="After the sweep, write the top-K runs by val_macro_f1_binary and a merge-ready hparams_sweep_best_f1_config.yaml; 0 to skip",
    )
    ap.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable the single grid progress bar (for very clean logs / CI).",
    )
    args = ap.parse_args()

    m_pair = 1 if args.pair else 0
    m_pain = 1 if bool(getattr(args, "paining_pair_sweep", False)) else 0
    m_non = 1 if bool(getattr(args, "non_pain_pair_sweep", False)) else 0
    if m_pair + m_pain + m_non != 1:
        print(
            "error: use exactly one of: --pair A,B | --paining-pair-sweep | --non-pain-pair-sweep",
            file=sys.stderr,
        )
        return 2
    if bool(getattr(args, "paining_grid_preset", False)):
        args.lrs = "0.00005,0.0001"
        args.batch_sizes = "4,8"
        args.weight_decays = "0.01,0.05"
        args.focal_gammas = "0.0,2.0"
        args.schedulers = "cosine"

    cfgp = args.config if args.config.is_absolute() else REPO / args.config
    base = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    base = copy.deepcopy(base)
    if bool(getattr(args, "from_scratch", False)):
        tr0 = base.setdefault("training", {})
        tr0.pop("init_weights_path", None)
        tr0["freeze_backbone_epochs"] = 0
        tr0.pop("unfreeze_lr", None)
    if args.split_seed is not None:
        base["split"] = {**base.get("split", {}), "random_state": int(args.split_seed)}

    paining_skip_t: tuple[str, str] | None = None
    if str(getattr(args, "paining_skip", "")).strip():
        ps = [x.strip() for x in str(args.paining_skip).split(",") if x.strip()]
        if len(ps) == 2:
            paining_skip_t = tuple(sorted((ps[0], ps[1])))

    lrs = _parse_float_list(args.lrs) or DEFAULT_LRS
    bss = _parse_int_list(args.batch_sizes) or DEFAULT_BATCH
    wds = _parse_float_list(args.weight_decays) or DEFAULT_WD
    gms = _parse_float_list(args.focal_gammas) or DEFAULT_GAMMA
    sch = _parse_str_list(args.schedulers) or DEFAULT_SCHED
    combs = _build_grid(
        lrs, bss, wds, gms, sch, shuffle=bool(args.shuffle_grid), seed=int(args.seed), max_runs=int(args.max_runs) or 0
    )
    ncomb = len(combs) if combs else 0
    n_sweep = 1
    sweep_pair_tuples: list[tuple[str, str]] = []
    if bool(getattr(args, "paining_pair_sweep", False)):
        cls = _multiclass_names_for_sweep(base)
        sweep_pair_tuples = _paining_sweep_tuples(cls, skip_pair=paining_skip_t)
        n_sweep = len(sweep_pair_tuples)
        if n_sweep < 1:
            print("error: no Paining pairs to sweep (check --paining-skip)", file=sys.stderr)
            return 1
    elif bool(getattr(args, "non_pain_pair_sweep", False)):
        if str(getattr(args, "paining_skip", "")).strip():
            print("warning: --paining-skip is ignored for --non-pain-pair-sweep", file=sys.stderr)
        sweep_pair_tuples = _non_pain_sweep_tuples(base)
        n_sweep = len(sweep_pair_tuples)
    total_runs = n_sweep * ncomb
    print(
        f"Grid: {len(lrs)}×{len(bss)}×{len(wds)}×{len(gms)}×{len(sch)} = {len(list(product(lrs, bss, wds, gms, sch)))} "
        f"hparam combos; running {ncomb} hparam (after --max-runs / shuffle) × {n_sweep} pair(s) = {total_runs} total cells.",
    )
    if args.dry_list or ncomb == 0:
        return 0

    from data_loading import  load_dataset, plot_dataset_overview, print_dataset_statistics
    from run_p4_pairwise import  make_master_run_dir
    from run_stgcn_deeplabcut_train import  setup_logger

    load_cfg = copy.deepcopy(base)
    load_cfg.setdefault("training", {})["binary_only"] = False
    if args.resume_from is not None:
        run_dir = args.resume_from.expanduser().resolve()
        if not run_dir.is_dir():
            print(f"error: --resume-from is not a directory: {run_dir}", file=sys.stderr)
            return 1
    else:
        run_dir = make_master_run_dir(base, str(args.experiment_name))
    grid_root = (run_dir / str(args.sweep_subdir)).resolve()
    grid_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    apply_sweep_baseline(
        base, epochs=int(args.epochs), patience=int(args.early_stop_patience), grad_clip=DEFAULT_GRAD_CLIP
    )
    (run_dir / "sweep_base_overrides.txt").write_text(
        f"epochs={args.epochs}  early_stop_patience={args.early_stop_patience}  grad_clip={DEFAULT_GRAD_CLIP}\n",
        encoding="utf-8",
    )
    (run_dir / "config_sweep_baseline_applied.yaml").write_text(
        yaml.dump(base, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    main_logger = setup_logger(run_dir)
    t0 = time.time()

    pkl_path = run_dir / "sweep_dataframe_cache.pkl"
    if args.resume_from is not None and pkl_path.is_file():
        df0 = pd.read_pickle(pkl_path)
        main_logger.info("Resume: loaded preloaded DataFrame: %d rows from %s", len(df0), pkl_path)
        print_dataset_statistics(df0, main_logger, load_cfg)
    else:
        df0 = load_dataset(load_cfg, main_logger)
        print_dataset_statistics(df0, main_logger, load_cfg)
        if bool(args.audio_confidence_filter):
            from class_subset_utils import  filter_dataframe_by_audio_confidence

            n0 = len(df0)
            df0 = filter_dataframe_by_audio_confidence(
                df0, str(args.audio_confidence_field), float(args.min_audio_confidence), logger=main_logger
            )
            main_logger.info("Audio filter: %d -> %d rows", n0, len(df0))
        if len(df0) < 2:
            main_logger.error("Empty or tiny dataframe after load/filter.")
            return 1
        plot_dataset_overview(df0, run_dir / "plots", load_cfg)
        df0.to_pickle(pkl_path)
        main_logger.info("Wrote preloaded DataFrame: %s", pkl_path)

    cuda_list = [x.strip() for x in str(args.cuda_devices).split(",") if str(x).strip() != ""]

    def _build_job(
        run_index: int,
        lr: float,
        bs: int,
        wd: float,
        g: float,
        sched: str,
        subdir: str,
        cuda_dev: str | None,
        force_cpu: bool,
        pair: str,
    ) -> dict[str, Any]:
        rep_audio = bool(args.audio_confidence_filter)
        return {
            "run_index": run_index,
            "lr": lr,
            "batch_size": bs,
            "weight_decay": wd,
            "focal_gamma": g,
            "scheduler": sched,
            "subdir": subdir,
            "base_cfg": copy.deepcopy(base),
            "parent_run_dir": str((grid_root / subdir).resolve()),
            "df_pickle_path": str(pkl_path),
            "pair": pair,
            "epochs": int(args.epochs),
            "patience": int(args.early_stop_patience),
            "grad_clip": float(DEFAULT_GRAD_CLIP),
            "use_kinematics": not bool(args.no_kinematics),
            "skip_existing": bool(args.skip_existing),
            "report_audio_filter": rep_audio,
            "min_audio_confidence": float(args.min_audio_confidence) if rep_audio else None,
            "conf_field": str(args.audio_confidence_field),
            "device": str(args.device),
            "cuda_device": cuda_dev,
            "force_cpu": bool(force_cpu),
        }

    from class_subset_utils import  pair_to_dirname

    jobs: list[dict] = []
    multi_sweep = bool(getattr(args, "paining_pair_sweep", False)) or bool(
        getattr(args, "non_pain_pair_sweep", False)
    )
    if multi_sweep and sweep_pair_tuples:
        run_i = 0
        for lo, hi in sweep_pair_tuples:
            pdir = pair_to_dirname(lo, hi)
            pstr = f"{lo},{hi}"
            for lr, bs, wd, g, sched in combs:
                sub = _hparam_subdir_pained(run_i, pdir, lr, bs, wd, g, sched)
                cdev: str | None = None
                fcpu = bool(args.force_cpu_parallel)
                if int(args.parallel) > 1 and cuda_list and not fcpu:
                    cdev = str(cuda_list[run_i % len(cuda_list)])
                jobs.append(
                    _build_job(run_i, lr, bs, wd, g, sched, sub, cdev, fcpu, pstr)
                )
                run_i += 1
    else:
        p_single = str(args.pair)
        for i, (lr, bs, wd, g, sched) in enumerate(combs):
            sub = _hparam_subdir(i, lr, bs, wd, g, sched)
            cdev: str | None = None
            fcpu = bool(args.force_cpu_parallel)
            if int(args.parallel) > 1 and cuda_list and not fcpu:
                cdev = str(cuda_list[i % len(cuda_list)])
            jobs.append(_build_job(i, lr, bs, wd, g, sched, sub, cdev, fcpu, p_single))

    idx_path = run_dir / "hparams_sweep_index.json"
    idx_path.write_text(json.dumps([j for j in jobs], indent=2, default=str), encoding="utf-8")
    if bool(getattr(args, "paining_pair_sweep", False)) and sweep_pair_tuples:
        (run_dir / "paining_sweep_pairs.json").write_text(
            json.dumps([f"{a},{b}" for a, b in sweep_pair_tuples], indent=2) + "\n",
            encoding="utf-8",
        )
    if bool(getattr(args, "non_pain_pair_sweep", False)) and sweep_pair_tuples:
        _allc = _multiclass_names_for_sweep(base)
        _pain = str((base.get("labels") or {}).get("binary_pain_class", "Paining")).strip()
        _nn = len([c for c in _allc if c != _pain])
        (run_dir / "non_pain_sweep_pairs.json").write_text(
            json.dumps(
                {
                    "excluded_pain_class": _pain,
                    "n_non_pain_classes": _nn,
                    "n_pairs": len(sweep_pair_tuples),
                    "expected_pairs_c2": _nn * (_nn - 1) // 2 if _nn >= 2 else 0,
                    "pairs": [f"{a},{b}" for a, b in sweep_pair_tuples],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    n_par = max(1, int(args.parallel))
    use_pool = n_par > 1 and (bool(cuda_list) or bool(args.force_cpu_parallel))
    if int(args.parallel) > 1 and not use_pool:
        main_logger.warning(
            "parallel=%d but use single-process queue: set --cuda-devices 0,1,... (multi-GPU) "
            "or --force-cpu-parallel for CPU worker pool.",
            n_par,
        )

    if bool(args.skip_existing):
        jobs_to_run: list[dict] = [
            j for j in jobs if not _p4_cell_run_summary_path(j["parent_run_dir"], j["pair"]).is_file()
        ]
        main_logger.info(
            "skip-existing: %d already complete, %d pending (tqdm shows pending only)",
            len(jobs) - len(jobs_to_run),
            len(jobs_to_run),
        )
    else:
        jobs_to_run = list(jobs)

    results: list[dict] = []
    jsonl = run_dir / "hparams_sweep_runs.jsonl"
    pbar_kw = {
        "desc": "hparam sweep",
        "total": len(jobs_to_run),
        "unit": "run",
        "dynamic_ncols": True,
    }

    if not jobs_to_run:
        main_logger.info("No pending hparam cells to run (grid empty or all have run_summary.json).")
    elif not use_pool:
        it = jobs_to_run if args.no_tqdm else tqdm(jobs_to_run, **pbar_kw)
        for j in it:
            r = run_one_sweep_cell(j)
            results.append(r)
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=str) + "\n")
    else:
        with ProcessPoolExecutor(max_workers=n_par) as ex:
            futs = [ex.submit(run_one_sweep_cell, j) for j in jobs_to_run]
            acomp = as_completed(futs)
            if not args.no_tqdm:
                acomp = tqdm(acomp, **pbar_kw)
            for fut in acomp:
                r = fut.result()
                results.append(r)
                with open(jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, default=str) + "\n")

    all_results = _read_sweep_jsonl_deduped(jsonl) if jsonl.is_file() and jsonl.stat().st_size else results
    if not all_results and results:
        all_results = list(results)
    if all_results:
        df = pd.DataFrame(all_results)
        if "ce_ratio" in df.columns:
            s = pd.to_numeric(df["ce_ratio"], errors="coerce")
            df["ce_suspect_early_best"] = s.notna() & (s < 0.2)
        else:
            df["ce_suspect_early_best"] = False
        out_csv = run_dir / "hparams_sweep_results.csv"
        df.to_csv(out_csv, index=False)
        if int(args.top_k) > 0:
            _write_sweep_best_artifacts(all_results, run_dir, base, top_k=int(args.top_k))
    main_logger.info(
        "Sweep finished in %.1f s | session cells: %d | total in jsonl: %d | %s | %s | best config: %s (if --top-k>0)",
        time.time() - t0,
        len(results),
        len(all_results),
        run_dir / "hparams_sweep_results.csv",
        jsonl,
        run_dir / "hparams_sweep_best_f1_config.yaml",
    )
    print(
        f"Done. session {len(results)} new runs, {len(all_results)} rows in table. {run_dir / 'hparams_sweep_results.csv'}  "
        f"(ce_ratio=1-based best / n_epochs_ran; ce_suspect_early_best if ce_ratio<0.2). "
        f"Best hparams: {run_dir / 'hparams_sweep_best_f1_config.yaml'}  (use --top-k 0 to skip saving it)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())