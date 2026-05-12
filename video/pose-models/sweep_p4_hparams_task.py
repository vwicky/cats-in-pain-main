"""
Picklable single-cell P4 hparam run for :mod:`scripts.sweep_p4_pairwise_hparams` (``ProcessPoolExecutor``).
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_POSE_MODELS = _REPO / "video" / "pose-models"
for _p in (_REPO, _POSE_MODELS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@dataclass
class HParamCell:
    run_index: int
    lr: float
    batch_size: int
    weight_decay: float
    focal_gamma: float
    scheduler: str
    subdir: str
    val_macro_f1_binary: str | None = None
    best_epoch: str | None = None
    output_dir: str | None = None
    error: str | None = None


def apply_sweep_baseline(cfg: dict, *, epochs: int, patience: int, grad_clip: float) -> None:
    tr = cfg.setdefault("training", {})
    tr["epochs"] = int(epochs)
    tr["early_stop_patience"] = int(patience)
    tr["grad_clip_norm"] = float(grad_clip)
    if str(tr.get("early_stop_metric", "")).lower() in ("val_macro_f1", "macro_f1"):
        tr["early_stop_metric"] = "val_macro_f1_binary"


def run_one_sweep_cell(job: dict[str, Any]) -> dict:
    """
    One grid cell. **Must** be defined in a real module (not a ``__main__`` script) so
    :class:`concurrent.futures.ProcessPoolExecutor` can unpickle it on the worker.
    """
    cuda = job.get("cuda_device")
    if cuda is not None and str(cuda).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    if job.get("force_cpu") is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from data_loading import  get_multiclass_class_names
    from run_p4_pairwise import  train_p4_one_pair
    from run_stgcn_dlc_pairwise import  _parse_single_pair
    from run_stgcn_deeplabcut_train import  setup_logger
    import torch

    h = HParamCell(
        run_index=job["run_index"],
        lr=job["lr"],
        batch_size=job["batch_size"],
        weight_decay=job["weight_decay"],
        focal_gamma=job["focal_gamma"],
        scheduler=job["scheduler"],
        subdir=job["subdir"],
    )
    cfg = copy.deepcopy(job["base_cfg"])
    t = cfg.setdefault("training", {})
    t["lr"] = h.lr
    t["batch_size"] = h.batch_size
    t["weight_decay"] = h.weight_decay
    t["focal_gamma"] = h.focal_gamma
    t["scheduler"] = h.scheduler
    apply_sweep_baseline(
        cfg, epochs=job["epochs"], patience=job["patience"], grad_clip=job["grad_clip"]
    )

    out_dir = Path(job["parent_run_dir"])
    if job.get("force_cpu") is True:
        device = torch.device("cpu")
    elif str(job.get("device", "auto")).lower() == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(str(job["device"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    try:
        df = pd.read_pickle(job["df_pickle_path"])
    except Exception as e:  # noqa: BLE001
        h.error = f"load_pickle: {e!r}"
        return asdict(h)

    label_field = str(cfg["labels"]["label_field"])
    lo, hi = _parse_single_pair(str(job["pair"]))

    lcfg = {**cfg, "training": {**cfg.get("training", {}), "binary_only": False}}
    classes_mc = get_multiclass_class_names(lcfg)
    for x in (lo, hi):
        if x not in classes_mc:
            h.error = f"unknown class {x!r}"
            d = asdict(h)
            d["val_macro_f1_binary"] = None
            d["best_epoch"] = None
            d["output_dir"] = None
            return d

    pain_class = str(cfg["labels"].get("binary_pain_class", "Paining"))
    min_pair = max(24, 2 * int(t["batch_size"]) + 4)
    master_meta = {
        "min_audio_confidence": job.get("min_audio_confidence")
        if job.get("report_audio_filter")
        else None,
        "audio_confidence_field": str(job.get("conf_field", "audio_confidence")),
        "audio_confidence_filter": bool(job.get("report_audio_filter", False)),
    }
    (out_dir / "hparam_cell.json").write_text(
        json.dumps(
            {
                "hparams": {
                    k: getattr(h, k)
                    for k in ("lr", "batch_size", "weight_decay", "focal_gamma", "scheduler")
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    row = train_p4_one_pair(
        lo,
        hi,
        df,
        cfg,
        out_dir,
        device,
        logger,
        use_kinematics=bool(job.get("use_kinematics", True)),
        min_pair=min_pair,
        pain_class=pain_class,
        label_field=label_field,
        master_meta=master_meta,
        dry_run=False,
        skip_if_exists=bool(job.get("skip_existing", False)),
    )
    d = asdict(h)
    d["pair"] = str(job.get("pair", ""))
    d["val_macro_f1_binary"] = None
    d["best_epoch"] = None
    d["output_dir"] = None
    d["error"] = None
    if row is not None:
        d["val_macro_f1_binary"] = row.get("val_macro_f1_binary")
        d["best_epoch"] = row.get("best_epoch")
        d["output_dir"] = row.get("output_dir")
        for k in (
            "n_epochs_ran",
            "early_stop_patience",
            "ce_ratio",
            "ce_ratio_heuristic",
        ):
            d[k] = row.get(k)
    else:
        d["error"] = "no_result (skipped, failed, or not viable; see cell experiment.log)"
    return d
