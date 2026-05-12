"""Logging, config loading, checkpointing, and plotting utilities."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml


def setup_logger(run_dir: str, name: str = "cat_cnn") -> logging.Logger:
    """Configure logger writing to console (INFO) and file (DEBUG)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter("%(message)s")

    fh = logging.FileHandler(Path(run_dir) / "train.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    return logger


def load_config(path: str, overrides: dict | None = None) -> dict:
    """Load YAML config, apply CLI overrides, validate required keys."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            if key in cfg:
                target_type = type(cfg[key])
                if target_type is bool:
                    value = str(value).lower() in ("true", "1", "yes")
                elif target_type is int:
                    value = int(value)
                elif target_type is float:
                    value = float(value)
            cfg[key] = value

    required = [
        "run_name", "manifest_path", "classes_5", "binary_map",
        "backbone", "cv_folds", "epochs", "batch_size",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg


def save_config(cfg: dict, run_dir: str) -> None:
    """Save the resolved config to the run directory."""
    out = Path(run_dir) / "config_used.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def make_run_dir(cfg: dict) -> str:
    """Create and return the timestamped run directory with fold subdirs."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{cfg['run_name']}_{ts}"
    run_dir = Path(cfg["runs_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, cfg["cv_folds"] + 1):
        (run_dir / f"fold_{fold}").mkdir(exist_ok=True)

    return str(run_dir)


class EarlyStopping:
    """Track a monitored metric and signal when to stop.

    Supports mode="max" (e.g. macro_f1) or mode="min" (e.g. loss).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = "max"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.best_epoch: int = 0
        self.should_stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        return current < self.best_value - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """Return True when training should stop."""
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_metrics_csv(history: list[dict], path: str) -> None:
    """Save or append epoch metrics to a CSV file."""
    p = Path(path)
    file_exists = p.exists()

    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(history)
