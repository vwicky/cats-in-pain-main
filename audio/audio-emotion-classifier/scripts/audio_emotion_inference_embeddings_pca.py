#!/usr/bin/env python3
"""
Standalone inference + embedding export + PCA visualization for the cat audio
emotion model (``CatEmotionModel`` in ``audio_classifier_utils`` — the same stack
as ``audio_classifier.ipynb``; there is no separate ``AudioEmotionClassifier`` type
in the repo, so we alias it below for a single entry point).

Loads fine-tuned ``best_model_final.pth``, scans WAV/MP3/FLAC/M4A under the
separated-audio tree, runs softmax classification and extracts 2048-D backbone
embeddings, saves CSV + NumPy artifacts, plots PCA colored by predicted label, and
writes ``confidence_threshold_plots/`` (PCA at several confidence cutoffs + per-class
counts and a summary CSV).

Run from repository root::

    python scripts/audio_emotion_inference_embeddings_pca.py
    python scripts/audio_emotion_inference_embeddings_pca.py --audio-root dataset_construction/separeted_audios --out-dir dataset_construction/reports/audio_embeddings_pca

The default audio root matches this project (``separeted_audios``). If you pass
``seperated_audios`` or ``separated_audios`` by mistake, the script tries those
spellings automatically when the first path is missing.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
EMOTION_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTION_CLASSIFIER_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(EMOTION_CLASSIFIER_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_classifier_utils.audio_config import AudioConfig
from audio_classifier_utils.models.cnn14 import CatEmotionModel

# Name used in the user-facing spec; the notebook implements this as CatEmotionModel.
AudioEmotionClassifier = CatEmotionModel

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

# Stable display / color order (matches class indices).
EMOTION_CLASSES_ORDERED: tuple[str, ...] = tuple(IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS)))

# Max softmax confidence thresholds (percent); filter samples with confidence >= value.
CONFIDENCE_THRESHOLDS_PCT: tuple[int, ...] = (20, 40, 60, 70, 75, 80, 85, 90)
CONFIDENCE_PLOTS_SUBDIR = "confidence_threshold_plots"

FINETUNED_CHECKPOINT_FALLBACKS = (
    "checkpoints/best_model_final.pth",
    "models/audio_emotions/best_model_final.pth",
)

# This repo uses ``separeted_audios`` (typo in folder name). Users often write ``seperated`` or ``separated``.
_AUDIO_ROOT_TYPOS: tuple[str, ...] = (
    "src/dataset_construction/separeted_audios",
    "src/dataset_construction/seperated_audios",
    "src/dataset_construction/separated_audios",
)


def resolve_audio_root(repo_root: Path, audio_root_arg: str) -> tuple[Path | None, str | None]:
    """
    Return ``(existing_dir, None)`` or ``(None, error_message)``.
    If the requested relative path is missing, tries common alternate spellings under
    ``src/dataset_construction/`` (same as ``04_audio_classification.py``).
    """
    paths_to_try: list[tuple[str, Path]] = []
    seen_resolved: set[str] = set()

    def add(rel: str) -> None:
        p = (repo_root / rel).resolve()
        key = str(p)
        if key not in seen_resolved:
            seen_resolved.add(key)
            paths_to_try.append((rel, p))

    add(audio_root_arg)
    for rel in _AUDIO_ROOT_TYPOS:
        add(rel)

    for rel, p in paths_to_try:
        if p.is_dir():
            if rel != audio_root_arg:
                return p, (
                    f"Note: --audio-root {audio_root_arg!r} was not found; "
                    f"using existing directory {rel!r} instead."
                )
            return p, None

    tried = "\n".join(f"  {p}" for _rel, p in paths_to_try)
    primary = (repo_root / audio_root_arg).resolve()
    msg = (
        f"Audio root does not exist: {primary}\n"
        f"Tried these paths (none is an existing directory):\n{tried}\n"
        "This project’s separated-audio folder is usually "
        "`src/dataset_construction/separeted_audios` (spelling: se-pa-re-ted)."
    )
    return None, msg


def resolve_finetuned_checkpoint(repo_root: Path, checkpoint_arg: str) -> tuple[Path | None, str]:
    requested = (repo_root / checkpoint_arg).resolve()
    if requested.is_file():
        return requested, checkpoint_arg
    if checkpoint_arg in FINETUNED_CHECKPOINT_FALLBACKS:
        for rel in FINETUNED_CHECKPOINT_FALLBACKS:
            if rel == checkpoint_arg:
                continue
            cand = (repo_root / rel).resolve()
            if cand.is_file():
                return cand, rel
        return None, checkpoint_arg
    return None, checkpoint_arg


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


def load_waveform(path: str, config: AudioConfig) -> torch.Tensor:
    """Match ``CatEmotionDataset`` / ``04_audio_classification`` inference (train=False)."""
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


def collect_audio_paths_recursive(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def collect_audio_paths_class_folders(root: Path) -> list[Path]:
    """
    Same discovery rules as ``audio_classifier_utils.data_utils.build_file_list``
    (class subfolders, ``.wav`` / ``.mp3``, skip ``_aug``), but returns **all**
    matching files without a train/validation split.
    """
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        return []
    paths: list[Path] = []
    for cls_path in class_dirs:
        for p in cls_path.iterdir():
            if not p.is_file():
                continue
            name = p.name
            low = name.lower()
            if "_aug" in name:
                continue
            if not (low.endswith(".wav") or low.endswith(".mp3")):
                continue
            paths.append(p)
    return sorted(paths, key=str)


def forward_logits_and_embeddings(
    model: CatEmotionModel, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``inputs``: (batch, time) on device."""
    out = model.backbone(inputs)
    emb = out["embedding"]
    logits = model.classifier(emb)
    return logits, emb


def run_inference_embeddings(
    paths: list[Path],
    model: CatEmotionModel,
    device: torch.device,
    config: AudioConfig,
    batch_size: int,
) -> tuple[
    np.ndarray,
    list[dict[str, Any]],
]:
    """
    Returns ``embeddings`` of shape (N_ok, 2048) and a list of per-file records
    (including error rows without embeddings — those have ``error`` set).
    """
    model.eval()
    records: list[dict[str, Any]] = []
    emb_chunks: list[np.ndarray] = []

    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_paths, records, emb_chunks
        if not batch_tensors:
            return
        inputs = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits, emb = forward_logits_and_embeddings(model, inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        emb_np = emb.cpu().numpy().astype(np.float32)

        for i, p in enumerate(batch_paths):
            idx = int(preds_np[i])
            pr = probs_np[i]
            conf = float(pr[idx])
            rec: dict[str, Any] = {
                "filepath": str(p.resolve()),
                "relpath": relpath_from_repo(p),
                "filename": p.name,
                "predicted_class_idx": idx,
                "predicted_class": IDX_TO_CLASS[idx],
                "confidence": conf,
            }
            for j in range(len(IDX_TO_CLASS)):
                rec[f"prob_{IDX_TO_CLASS[j]}"] = float(pr[j])
            records.append(rec)
            emb_chunks.append(emb_np[i : i + 1])

        batch_tensors = []
        batch_paths = []

    for path in tqdm(paths, desc="Audio emotion + embeddings", unit="file"):
        try:
            w = load_waveform(str(path), config)
            batch_tensors.append(w)
            batch_paths.append(path)
        except Exception as e:
            records.append(
                {
                    "filepath": str(path.resolve()),
                    "relpath": relpath_from_repo(path),
                    "filename": path.name,
                    "error": str(e),
                }
            )
            continue

        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    if emb_chunks:
        embeddings = np.concatenate(emb_chunks, axis=0)
    else:
        embeddings = np.zeros((0, 2048), dtype=np.float32)

    # ``embeddings[i]`` matches the i-th successful record in ``records`` order.
    ok_i = 0
    for r in records:
        if "error" not in r:
            r["embedding_row"] = ok_i
            ok_i += 1

    return embeddings, records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    if not records:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def plot_pca_scatter(
    embeddings: np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    random_state: int = 0,
) -> None:
    if embeddings.shape[0] < 2:
        warnings.warn("Not enough samples for PCA plot; skipping.")
        return
    n_comp = min(2, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    z = pca.fit_transform(embeddings)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    # tab10 / tab20 are designed for categoricals; husl blends together for many classes.
    cmap_name = "tab10" if n_classes <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    n_cmap = cmap.N
    color_map = {
        lab: cmap(i % n_cmap) for i, lab in enumerate(unique_labels)
    }
    colors = [color_map[lab] for lab in labels]
    edge = (0.15, 0.15, 0.15, 0.55)
    if n_comp == 1:
        ax.scatter(
            z[:, 0],
            np.zeros_like(z[:, 0]),
            c=colors,
            s=44,
            alpha=0.88,
            edgecolors=edge,
            linewidths=0.45,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
        ax.set_ylabel("(single component)")
    else:
        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=colors,
            s=44,
            alpha=0.88,
            edgecolors=edge,
            linewidths=0.45,
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
        # Legend from color_map
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[lab], markersize=8, label=lab)
            for lab in sorted(color_map.keys())
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    ax.set_title("PCA of audio emotion embeddings (colored by predicted class)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _emotion_tab10_color_map() -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {c: cmap(i) for i, c in enumerate(EMOTION_CLASSES_ORDERED)}


def class_counts_for_labels(labels: list[str]) -> dict[str, int]:
    counts = {c: 0 for c in EMOTION_CLASSES_ORDERED}
    for lab in labels:
        if lab in counts:
            counts[lab] += 1
    return counts


def write_per_class_counts_by_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_pca_confidence_threshold_panel(
    embeddings: np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    threshold_pct: int,
    random_state: int = 0,
) -> None:
    """
    PCA scatter (left) + horizontal bar chart of sample counts per emotion class (right).
    ``embeddings`` / ``labels`` are already filtered to confidence >= threshold_pct.
    """
    sns.set_theme(style="whitegrid")
    color_by_class = _emotion_tab10_color_map()
    counts = class_counts_for_labels(labels)
    n = embeddings.shape[0]
    if sum(counts.values()) != n or len(labels) != n:
        warnings.warn("Label / embedding length mismatch in confidence plot; check inputs.")

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(3.05, 1.12), wspace=0.26)
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ys = list(range(len(EMOTION_CLASSES_ORDERED)))
    xs = [counts[c] for c in EMOTION_CLASSES_ORDERED]
    bar_colors = [color_by_class[c] for c in EMOTION_CLASSES_ORDERED]
    ax_bar.barh(ys, xs, color=bar_colors, edgecolor=(0.12, 0.12, 0.12, 0.45), linewidth=0.4)
    ax_bar.set_yticks(ys)
    ax_bar.set_yticklabels(EMOTION_CLASSES_ORDERED, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Count")
    ax_bar.set_title("Samples per class", fontsize=10)
    xmax = max(xs) if xs else 0
    ax_bar.set_xlim(left=0, right=max(1.0, xmax * 1.15 + 0.5))
    for y, v in zip(ys, xs):
        if v > 0:
            off = xmax * 0.02 + 0.06 if xmax > 0 else 0.12
            ax_bar.text(v + off, y, str(v), va="center", fontsize=8)

    if n < 2:
        ax_pca.text(
            0.5,
            0.5,
            f"PCA needs ≥2 samples after filtering.\n"
            f"confidence ≥ {threshold_pct}%: n = {n}\n"
            f"(see panel → for per-class counts)",
            ha="center",
            va="center",
            transform=ax_pca.transAxes,
            fontsize=11,
        )
        ax_pca.set_xticks([])
        ax_pca.set_yticks([])
        for s in ax_pca.spines.values():
            s.set_visible(False)
    else:
        n_comp = min(2, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        z = pca.fit_transform(embeddings)
        unique_labels = sorted(set(labels))
        edge = (0.15, 0.15, 0.15, 0.55)
        colors = [color_by_class[lab] for lab in labels]
        if n_comp == 1:
            ax_pca.scatter(
                z[:, 0],
                np.zeros_like(z[:, 0]),
                c=colors,
                s=44,
                alpha=0.88,
                edgecolors=edge,
                linewidths=0.45,
            )
            ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
            ax_pca.set_ylabel("(single component)")
        else:
            ax_pca.scatter(
                z[:, 0],
                z[:, 1],
                c=colors,
                s=44,
                alpha=0.88,
                edgecolors=edge,
                linewidths=0.45,
            )
            ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
            ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_by_class[lab],
                markersize=8,
                label=lab,
            )
            for lab in unique_labels
        ]
        ax_pca.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)

    ax_pca.set_title(
        f"PCA — max softmax confidence ≥ {threshold_pct}% (n = {n})",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_confidence_threshold_plots(
    embeddings: np.ndarray,
    labels: list[str],
    confidences: np.ndarray,
    out_subdir: Path,
    *,
    random_state: int,
) -> list[dict[str, Any]]:
    """
    Writes one figure per threshold and returns rows for ``per_class_counts_by_threshold.csv``
    (long format: threshold, class, count, n_passing_threshold).
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []

    for pct in CONFIDENCE_THRESHOLDS_PCT:
        thr = pct / 100.0
        mask = confidences >= thr
        emb_f = embeddings[mask]
        labels_f = [labels[i] for i in range(len(labels)) if mask[i]]
        counts = class_counts_for_labels(labels_f)
        n_pass = int(mask.sum())

        for c in EMOTION_CLASSES_ORDERED:
            csv_rows.append(
                {
                    "threshold_pct": pct,
                    "predicted_class": c,
                    "count": counts[c],
                    "n_passing_threshold": n_pass,
                }
            )

        out_png = out_subdir / f"pca_conf_ge_{pct}pct.png"
        plot_pca_confidence_threshold_panel(
            emb_f,
            labels_f,
            out_png,
            threshold_pct=pct,
            random_state=random_state,
        )

    return csv_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audio emotion inference, embeddings export, PCA plot.")
    p.add_argument(
        "--audio-root",
        type=str,
        default="src/dataset_construction/separeted_audios",
        help=(
            "Directory of separated snippets (recursive). If missing, also tries "
            "separeted_audios / seperated_audios / separated_audios under dataset_construction/."
        ),
    )
    p.add_argument(
        "--prefer-class-folder-layout",
        action="store_true",
        help="If the root has class subfolders, use the same rules as data_utils.build_file_list (no train/val split).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=FINETUNED_CHECKPOINT_FALLBACKS[0],
        help="Fine-tuned weights (best final). Falls back to models/checkpoints_audio_emotions/ when missing.",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Override AudioConfig.checkpoint_path (PANNs Cnn14 pretrained .pth).",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: dataset_construction/reports/audio_embeddings_pca_<UTC timestamp>).",
    )
    p.add_argument("--pca-seed", type=int, default=0, help="Random seed for PCA (sklearn).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    audio_root, audio_root_note = resolve_audio_root(repo_root, args.audio_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        (repo_root / args.out_dir).resolve()
        if args.out_dir
        else (repo_root / "src" / "dataset_construction" / "reports" / f"audio_embeddings_pca_{ts}").resolve()
    )

    ckpt_path, ckpt_label = resolve_finetuned_checkpoint(repo_root, args.checkpoint)
    if ckpt_path is None or not ckpt_path.is_file():
        print(
            "Missing fine-tuned weights. Tried:",
            repo_root / FINETUNED_CHECKPOINT_FALLBACKS[0],
            "and",
            repo_root / FINETUNED_CHECKPOINT_FALLBACKS[1],
            file=sys.stderr,
        )
        return 1

    if audio_root is None:
        print(audio_root_note, file=sys.stderr)
        return 1
    if audio_root_note:
        print(audio_root_note)

    if args.prefer_class_folder_layout:
        paths = collect_audio_paths_class_folders(audio_root)
        if not paths:
            paths = collect_audio_paths_recursive(audio_root)
    else:
        paths = collect_audio_paths_recursive(audio_root)

    if args.limit is not None:
        paths = paths[: max(0, args.limit)]

    audio_config = AudioConfig()
    if args.pretrained:
        audio_config.checkpoint_path = args.pretrained
    pretrained_path = (repo_root / audio_config.checkpoint_path).resolve()
    if not pretrained_path.is_file():
        print(f"Missing PANNs pretrained weights: {pretrained_path}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path} ({ckpt_label})")
    print(f"Audio files: {len(paths)}")

    try:
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}", file=sys.stderr)
        return 1

    model = AudioEmotionClassifier(audio_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    embeddings, records = run_inference_embeddings(
        paths,
        model,
        device,
        audio_config,
        max(1, args.batch_size),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    npy_path = out_dir / "embeddings.npy"
    npz_path = out_dir / "audio_emotion_bundle.npz"
    pca_path = out_dir / "pca_emotion_scatter.png"
    summary_path = out_dir / "run_summary.txt"

    write_csv(csv_path, records)
    np.save(npy_path, embeddings)

    ok_records = [r for r in records if "error" not in r]
    labels = [r["predicted_class"] for r in ok_records]

    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        filenames=np.array([r["filename"] for r in ok_records], dtype=object),
        relpaths=np.array([r["relpath"] for r in ok_records], dtype=object),
        predicted_class_idx=np.array([r["predicted_class_idx"] for r in ok_records], dtype=np.int32),
        predicted_class=np.array(labels, dtype=object),
        confidence=np.array([r["confidence"] for r in ok_records], dtype=np.float32),
    )

    plot_pca_scatter(embeddings, labels, pca_path, random_state=args.pca_seed)

    conf_arr = np.array([float(r["confidence"]) for r in ok_records], dtype=np.float64)
    conf_plot_dir = out_dir / CONFIDENCE_PLOTS_SUBDIR
    csv_rows_thresholds = run_confidence_threshold_plots(
        embeddings,
        labels,
        conf_arr,
        conf_plot_dir,
        random_state=args.pca_seed,
    )
    write_per_class_counts_by_threshold_csv(
        conf_plot_dir / "per_class_counts_by_threshold.csv",
        csv_rows_thresholds,
    )

    n_err = sum(1 for r in records if "error" in r)
    summary_lines = [
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Audio root: {audio_root}",
        f"Fine-tuned checkpoint: {ckpt_path}",
        f"PANNs pretrained: {pretrained_path}",
        f"Files processed: {len(paths)}",
        f"Successful: {len(ok_records)}",
        f"Failed (load): {n_err}",
        f"Embedding shape: {embeddings.shape}",
        f"Outputs: {csv_path.name}, {npy_path.name}, {npz_path.name}, {pca_path.name}",
        f"Confidence threshold plots: {CONFIDENCE_PLOTS_SUBDIR}/ "
        f"(max softmax ≥ threshold %: {CONFIDENCE_THRESHOLDS_PCT}; PCA + per-class bar chart + CSV)",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {npy_path} shape={embeddings.shape}")
    print(f"Wrote {npz_path}")
    print(f"Wrote {pca_path}")
    print(f"Wrote confidence plots + counts under {conf_plot_dir}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
