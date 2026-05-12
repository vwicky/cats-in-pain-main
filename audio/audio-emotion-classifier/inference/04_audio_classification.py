#!/usr/bin/env python3
"""
Cat emotion inference on separated audio under dataset_construction/separeted_audios/.

Uses CatEmotionModel + checkpoints/best_model_final.pth (same as audio_classifier.ipynb).

Run from the **repository root** (Cats-in-Pain-Bachelors):

  python dataset_construction/04_audio_classification.py
  python dataset_construction/04_audio_classification.py --dry-run

Or, if your shell is already in ``src/dataset_construction/``, use the script name only (do not
prefix ``src/dataset_construction/`` again — that would double the path):

  python 04_audio_classification.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
EMOTION_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTION_CLASSIFIER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EMOTION_CLASSIFIER_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_classifier_utils.audio_config import AudioConfig
from audio_classifier_utils.models.cnn14 import CatEmotionModel

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")

IDX_TO_CLASS: dict[int, str] = {
    0: "Angry",
    1: "Defence",
    2: "Fighting",
    3: "Happy",
    4: "HuntingMind",
    5: "Mating",
    6: "MotherCall",
    7: "Paining",
    8: "Resting",
    9: "Warning",
}

DEFAULT_AUDIO_ROOT = "src/dataset_construction/separeted_audios"
DEFAULT_CHECKPOINT = "checkpoints/best_model_final.pth"
# Same weights as audio_classifier.ipynb (under REPO_ROOT); alternate layout from labeling_cat_audios.ipynb
FINETUNED_CHECKPOINT_FALLBACKS = (
    "checkpoints/best_model_final.pth",
    "models/audio_emotions/best_model_final.pth",
)
DEFAULT_REPORTS_DIR = "src/dataset_construction/reports"
DEFAULT_LOGS_DIR = "src/dataset_construction/logs"
JSONL_NAME = "separated_audios_emotion_predictions.jsonl"
REPORT_NAME = "emotion_labeling_report.txt"
PLOT_DIST = "emotion_labeling_distribution.png"
PLOT_CONF = "emotion_labeling_confidence.png"


def load_waveform(path: str, config: AudioConfig) -> torch.Tensor:
    """Match CatEmotionDataset (__getitem__) with train=False: load, pad/truncate, no aug."""
    waveform, _ = librosa.load(path, sr=config.sample_rate)
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    target_len = config.target_length
    current_len = waveform.shape[1]
    if current_len > target_len:
        waveform = waveform[:, :target_len]
    elif current_len < target_len:
        waveform = F.pad(waveform, (0, target_len - current_len))
    return waveform.squeeze(0)


def relpath_from_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_finetuned_checkpoint(
    repo_root: Path, checkpoint_arg: str, logger: logging.Logger
) -> tuple[Path | None, str]:
    """
    Resolve --checkpoint to an existing file. If the path is missing but matches a known
    training-export layout, try the other default location (root ``checkpoints/`` vs
    ``models/checkpoints_audio_emotions/``).
    """
    requested = (repo_root / checkpoint_arg).resolve()
    if requested.is_file():
        return requested, checkpoint_arg

    if checkpoint_arg in FINETUNED_CHECKPOINT_FALLBACKS:
        for rel in FINETUNED_CHECKPOINT_FALLBACKS:
            if rel == checkpoint_arg:
                continue
            cand = (repo_root / rel).resolve()
            if cand.is_file():
                logger.info(
                    "Using %s (fine-tuned weights not found at %s)",
                    cand,
                    requested,
                )
                return cand, rel
        return None, checkpoint_arg

    return None, checkpoint_arg


def collect_audio_paths(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but not available; using CPU.")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested}")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("04_audio_classification")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_inference(
    paths: list[Path],
    model: CatEmotionModel,
    device: torch.device,
    config: AudioConfig,
    batch_size: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Returns JSONL rows for successful files and confidences; error rows embedded in list."""
    rows: list[dict[str, Any]] = []
    confidences: list[float] = []
    model.eval()

    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_paths, rows, confidences
        if not batch_tensors:
            return
        inputs = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        for i, p in enumerate(batch_paths):
            idx = int(preds_np[i])
            pr = probs_np[i]
            conf = float(pr[idx])
            confidences.append(conf)
            row: dict[str, Any] = {
                "filepath": str(p.resolve()),
                "relpath": relpath_from_repo(p),
                "predicted_class_idx": idx,
                "predicted_class": IDX_TO_CLASS[idx],
                "confidence": conf,
            }
            for j in range(len(IDX_TO_CLASS)):
                row[f"prob_{IDX_TO_CLASS[j]}"] = float(pr[j])
            rows.append(row)
        batch_tensors = []
        batch_paths = []

    for path in tqdm(paths, desc="Emotion inference", unit="file"):
        try:
            w = load_waveform(str(path), config)
            batch_tensors.append(w)
            batch_paths.append(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            rows.append(
                {
                    "filepath": str(path.resolve()),
                    "relpath": relpath_from_repo(path),
                    "error": str(e),
                }
            )
            continue

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()
    return rows, confidences


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    *,
    audio_root: Path,
    checkpoint: Path,
    pretrained: Path,
    n_files: int,
    n_ok: int,
    n_err: int,
    confidences: list[float],
    class_counts: dict[str, int],
) -> None:
    lines = [
        f"04_audio_classification report",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Audio root: {audio_root}",
        f"Fine-tuned checkpoint: {checkpoint}",
        f"PANNs backbone (AudioConfig): {pretrained}",
        f"Files discovered: {n_files}",
        f"Successfully labeled: {n_ok}",
        f"Failed (load/decode): {n_err}",
        "",
        "Predicted class counts (successful rows only):",
    ]
    for name in sorted(class_counts.keys(), key=lambda k: (-class_counts[k], k)):
        lines.append(f"  {name}: {class_counts[name]}")
    lines.append("")
    if confidences:
        lines.extend(
            [
                f"Confidence (max softmax) — mean: {statistics.mean(confidences):.4f}",
                f"Confidence — median: {statistics.median(confidences):.4f}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_distribution(class_counts: dict[str, int], out_path: Path) -> None:
    if not class_counts:
        return
    sns.set_theme(style="whitegrid")
    names = list(class_counts.keys())
    counts = [class_counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    df_order = sorted(names, key=lambda k: -class_counts[k])
    x_idx = range(len(df_order))
    ax.bar(x_idx, [class_counts[k] for k in df_order], color="steelblue")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(df_order, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_xlabel("Predicted emotion")
    ax.set_title("Emotion predictions (separated audios)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_histogram(confidences: list[float], out_path: Path) -> None:
    if not confidences:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confidences, bins=30, color="coral", edgecolor="white")
    ax.set_xlabel("Max class probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction confidence distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cat emotion labels for separated audios.")
    p.add_argument(
        "--audio-root",
        type=str,
        default=DEFAULT_AUDIO_ROOT,
        help=f"Root directory to scan recursively (default: {DEFAULT_AUDIO_ROOT})",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=(
            f"Fine-tuned CatEmotionModel weights (default: {DEFAULT_CHECKPOINT}). "
            "If that path is missing, the script also tries "
            "models/audio_emotions/best_model_final.pth"
        ),
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Override AudioConfig.checkpoint_path (PANNs Cnn14 pretrained .pth)",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    p.add_argument("--dry-run", action="store_true", help="List file count and exit.")
    p.add_argument("--reports-dir", type=str, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--logs-dir", type=str, default=DEFAULT_LOGS_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    audio_root = (repo_root / args.audio_root).resolve()
    reports_dir = (repo_root / args.reports_dir).resolve()
    logs_dir = (repo_root / args.logs_dir).resolve()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = logs_dir / f"04_audio_classification_{ts}.log"
    logger = setup_logging(log_file)
    logger.info("Log file: %s", log_file)

    ckpt_path, ckpt_label = resolve_finetuned_checkpoint(repo_root, args.checkpoint, logger)

    if not audio_root.is_dir():
        logger.error("Audio root does not exist: %s", audio_root)
        return 1

    all_paths = collect_audio_paths(audio_root)
    if args.limit is not None:
        paths = all_paths[: max(0, args.limit)]
    else:
        paths = all_paths

    logger.info(
        "Discovered %d audio files under %s",
        len(all_paths),
        audio_root,
    )
    if args.limit is not None:
        logger.info("After --limit %s: processing %d files", args.limit, len(paths))

    if args.dry_run:
        logger.info("Dry run: would process %d files.", len(paths))
        return 0

    if not paths:
        logger.info("No files to process; writing empty outputs.")
        cfg_empty = AudioConfig()
        if args.pretrained:
            cfg_empty.checkpoint_path = args.pretrained
        jsonl_path = reports_dir / JSONL_NAME
        write_jsonl(jsonl_path, [])
        write_report(
            reports_dir / REPORT_NAME,
            audio_root=audio_root,
            checkpoint=(repo_root / args.checkpoint).resolve(),
            pretrained=(repo_root / cfg_empty.checkpoint_path).resolve(),
            n_files=0,
            n_ok=0,
            n_err=0,
            confidences=[],
            class_counts={},
        )
        logger.info("Wrote empty %s and %s", jsonl_path, reports_dir / REPORT_NAME)
        return 0

    if ckpt_path is None or not ckpt_path.is_file():
        logger.error(
            "Missing fine-tuned CatEmotionModel weights. Tried:\n"
            "  %s\n"
            "  %s\n"
            "Place best_model_final.pth in one of those locations or pass --checkpoint /path/to.pth",
            repo_root / FINETUNED_CHECKPOINT_FALLBACKS[0],
            repo_root / FINETUNED_CHECKPOINT_FALLBACKS[1],
        )
        return 1
    logger.info("Fine-tuned checkpoint: %s (%s)", ckpt_path, ckpt_label)

    audio_config = AudioConfig()
    if args.pretrained:
        audio_config.checkpoint_path = args.pretrained

    pretrained_path = (repo_root / audio_config.checkpoint_path).resolve()
    if not pretrained_path.is_file():
        logger.error("Missing PANNs pretrained weights (AudioConfig.checkpoint_path): %s", pretrained_path)
        return 1

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    try:
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as e:
        logger.exception("Failed to load checkpoint: %s", e)
        return 1

    model = CatEmotionModel(audio_config)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        logger.exception("load_state_dict failed: %s", e)
        return 1
    model.to(device)

    rows, confidences = run_inference(
        paths,
        model,
        device,
        audio_config,
        max(1, args.batch_size),
        logger,
    )

    jsonl_path = reports_dir / JSONL_NAME
    write_jsonl(jsonl_path, rows)
    logger.info("Wrote %d rows to %s", len(rows), jsonl_path)

    ok_rows = [r for r in rows if "error" not in r]
    err_rows = [r for r in rows if "error" in r]
    class_counts: dict[str, int] = {}
    for r in ok_rows:
        name = r.get("predicted_class", "?")
        class_counts[name] = class_counts.get(name, 0) + 1

    write_report(
        reports_dir / REPORT_NAME,
        audio_root=audio_root,
        checkpoint=ckpt_path,
        pretrained=pretrained_path,
        n_files=len(paths),
        n_ok=len(ok_rows),
        n_err=len(err_rows),
        confidences=confidences,
        class_counts=class_counts,
    )
    logger.info("Wrote report %s", reports_dir / REPORT_NAME)

    plot_distribution(class_counts, reports_dir / PLOT_DIST)
    plot_confidence_histogram(confidences, reports_dir / PLOT_CONF)
    logger.info("Saved plots to %s and %s", reports_dir / PLOT_DIST, reports_dir / PLOT_CONF)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
