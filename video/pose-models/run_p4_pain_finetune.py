#!/usr/bin/env python3
"""
Train the **remaining Paining** binary class pairs, starting from a P4 checkpoint
(typically the best Paining–Resting sweep run).

- **Pairs**: all pairs that include *Paining* except the excluded pretrain pair
  (default (Paining, Resting)) — **8** runs.

- **Config**: :file:`config_p4_pain_finetune.yaml` (freeze encoder 20 epochs, unfreeze
  at 1e-5, 1000 max epochs, early stopping 100, hparams from gs0030: bs=8, lr=5e-5, wd=0.05, γ=0, cosine).

Usage:

  .venv/bin/python3 model_training_v2/run_p4_pain_finetune.py \\
    --config model_training_v2/config_p4_pain_finetune.yaml \\
    --experiment-name p4_pain_finetune_8

From scratch (same 8 pairs, same hparams in config, no checkpoint / no freeze):

  .venv/bin/python3 model_training_v2/run_p4_pain_finetune.py \\
    --config model_training_v2/config_p4_pain_finetune.yaml \\
    --from-scratch --experiment-name p4_pain_8_baseline
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_MODELS_ROOT = REPO_ROOT / "video" / "pose-models"
for _p in (REPO_ROOT, POSE_MODELS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from class_subset_utils import  (
    filter_dataframe_by_audio_confidence,
    filter_remap_binary_pair,
    first_stratified_group_split,
    iter_sorted_class_pairs,
    pair_binary_neg_pos,
    pair_to_dirname,
    split_viable,
)
from data_loading import  get_multiclass_class_names, load_dataset, plot_dataset_overview, print_dataset_statistics
from run_p4_pairwise import  train_p4_one_pair
from run_stgcn_dlc_pairwise import  (
    make_master_run_dir,
    save_master_results,
    write_pairwise_report,
)


def _parse_exclude_pair(s: str) -> tuple[str, str]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit(f"expected two classes in --exclude-pair, got: {s!r}")
    return tuple(sorted((parts[0], parts[1])))  # type: return sorted pair


def _pain_finetune_pairs(
    classes: list[str], *, ex: tuple[str, str]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for lo, hi in iter_sorted_class_pairs(classes):
        if "Paining" in (lo, hi) and (lo, hi) != ex:
            out.append((lo, hi))
    if len(out) != 8:
        raise SystemExit(f"expected 8 Paining finetune pairs (excluding {ex!r}), got {len(out)}: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "video" / "pose-models" / "config_p4_pain_finetune.yaml")
    ap.add_argument("--experiment-name", default="p4_pain_finetune_8")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-kinematics", action="store_true")
    ap.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Override training.init_weights_path in config (best_weights or best_model).",
    )
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore finetune settings: no init_weights, no backbone freeze (random init, same pair list & other hparams).",
    )
    ap.add_argument(
        "--exclude-pair",
        default="Paining,Resting",
        metavar="A,B",
        help="Skip this already-trained pair (pretrain / checkpoint).",
    )
    ap.add_argument("--min-pair-rows", type=int, default=None, help="Default: max(24, 2*batch+4).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if training/run_summary.json exists.")
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--audio-confidence-field", default="audio_confidence")
    ap.add_argument("--min-audio-confidence", type=float, default=0.7)
    ap.add_argument("--audio-confidence-filter", action="store_true")
    args = ap.parse_args()

    cfgp = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    if args.split_seed is not None:
        cfg = copy.deepcopy(cfg)
        cfg["split"] = {**cfg["split"], "random_state": int(args.split_seed)}

    if args.init_weights is not None:
        cfg = copy.deepcopy(cfg)
        p = args.init_weights
        cfg.setdefault("training", {})["init_weights_path"] = str(
            p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
        )

    if bool(getattr(args, "from_scratch", False)):
        cfg = copy.deepcopy(cfg)
        t = cfg.setdefault("training", {})
        t.pop("init_weights_path", None)
        t["freeze_backbone_epochs"] = 0
        t.pop("unfreeze_lr", None)

    ex = _parse_exclude_pair(args.exclude_pair)
    load_cfg = copy.deepcopy(cfg)
    load_cfg.setdefault("training", {})["binary_only"] = False
    use_kin = not args.no_kinematics
    classes_mc = get_multiclass_class_names(load_cfg)
    label_field = str(load_cfg["labels"]["label_field"])
    pain_class = str(load_cfg["labels"].get("binary_pain_class", "Paining"))
    batch_size = int(load_cfg["training"]["batch_size"])
    min_pair = args.min_pair_rows
    if min_pair is None:
        min_pair = max(24, 2 * batch_size + 4)

    pairs = _pain_finetune_pairs(classes_mc, ex=ex)

    run_dir = make_master_run_dir(cfg, args.experiment_name)
    from run_stgcn_deeplabcut_train import  setup_logger

    logger = setup_logger(run_dir)
    t0s = time.time()
    conf_field = str(args.audio_confidence_field)
    filter_enabled = bool(args.audio_confidence_filter)
    cfg_to_save = copy.deepcopy(cfg)
    cfg_to_save["p4_pain_finetune"] = {
        "min_audio_confidence": None if not filter_enabled else float(args.min_audio_confidence),
        "confidence_field": conf_field,
        "filter_enabled": filter_enabled,
        "pairs": [f"{a}|{b}" for a, b in pairs],
        "excluded_pretrain_pair": f"{ex[0]}|{ex[1]}",
        "min_pair_rows": min_pair,
    }
    (run_dir / "config_used.yaml").write_text(
        yaml.dump(cfg_to_save, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    master_meta: dict = {
        "min_audio_confidence": None if not filter_enabled else float(args.min_audio_confidence),
        "audio_confidence_field": conf_field,
        "audio_confidence_filter": filter_enabled,
    }

    tr = cfg.get("training", {})
    logger.info(
        "P4 Paining pairs (8) | excl. %s | min_pair=%d | from_scratch=%s | init=%s | freeze=%s unfreeze_lr=%s",
        f"{ex[0]},{ex[1]}",
        min_pair,
        bool(args.from_scratch),
        tr.get("init_weights_path"),
        tr.get("freeze_backbone_epochs"),
        tr.get("unfreeze_lr"),
    )

    df = load_dataset(load_cfg, logger)
    print_dataset_statistics(df, logger, load_cfg)
    plot_dataset_overview(df, run_dir / "plots", load_cfg)
    if filter_enabled:
        n0 = len(df)
        df = filter_dataframe_by_audio_confidence(
            df, conf_field, float(args.min_audio_confidence), logger=logger
        )
        logger.info("Audio filter: %d -> %d", len(df), n0)
        if len(df) < 1:
            logger.error("No rows after audio filter.")
            raise SystemExit(1)
        print_dataset_statistics(df, logger, load_cfg)

    with open(run_dir / "reports" / "pair_list.json", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                [
                    {
                        "class_lo": lo,
                        "class_hi": hi,
                        "class_neg": pair_binary_neg_pos(lo, hi, pain_class=pain_class)[0],
                        "class_pos": pair_binary_neg_pos(lo, hi, pain_class=pain_class)[1],
                        "subdir": pair_to_dirname(lo, hi),
                    }
                    for lo, hi in pairs
                ],
                indent=2,
            )
        )

    if args.dry_run:
        for lo, hi in pairs:
            sub = filter_remap_binary_pair(df, lo, hi, label_field=label_field, pain_class=pain_class)
            if len(sub) < min_pair:
                logger.info("pair %s vs %s: skip n=%d < %d", lo, hi, len(sub), min_pair)
                continue
            tr, va, err = first_stratified_group_split(sub, cfg)
            if err:
                logger.info("pair %s vs %s: %s", lo, hi, err)
                continue
            ok, rea = split_viable(tr, va, 2, batch_size=batch_size)
            logger.info("pair %s vs %s: n_tr=%d n_val=%d ok=%s %s", lo, hi, len(tr), len(va), ok, rea or "")
        return

    import torch

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    all_results: list[dict] = []
    n = len(pairs)
    for pi, (lo, hi) in enumerate(pairs):
        pdir = pair_to_dirname(lo, hi)
        t0p = time.time()
        row = train_p4_one_pair(
            lo,
            hi,
            df,
            cfg,
            run_dir,
            device,
            logger,
            use_kinematics=use_kin,
            min_pair=min_pair,
            pain_class=pain_class,
            label_field=label_field,
            master_meta=master_meta,
            dry_run=False,
            skip_if_exists=bool(args.skip_existing),
        )
        if row is not None:
            logger.info("Done %s in %.1f s", pdir, time.time() - t0p)
            all_results.append(row)

    save_master_results(all_results, run_dir, t0s, n)
    write_pairwise_report(
        run_dir, all_results, report_title="P4 Paining finetune (from init checkpoint)"
    )
    if all_results:
        print(f"Wrote: {run_dir / 'master_results.csv'}")


if __name__ == "__main__":
    main()
