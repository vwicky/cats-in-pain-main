#!/usr/bin/env python3
"""
Build a **10-class audio holdout** from ``human_validation_app/cat_audios_copy_processed``.

1. Runs **CatEmotionModel** (CNN14 + fine-tuned head) — same stack as
   ``src/dataset_construction/04_audio_classification.py`` — on all ``*.mp3`` under
   the input directory (flat layout).

2. Writes:
   - ``audio10_predictions_<UTC>.jsonl`` — same row schema as 04 (filepath, probs, …).
   - ``holdout_manifest_audio10.jsonl`` — one row per clip, fields aligned with
     analysis tooling: ``snippet_id``, ``audio_label_10`` (model), optional join
     to v1 ``dataset/final_dataset.jsonl`` for ``pose_path``, human ``label`` (5-way),
     ``label_confidence``, ``low_quality_pose``.

3. Writes ``holdout_audio10_meta.json`` including a **leakage check**: intersection
   of stems with ``src/dataset_construction/manifests/final_dataset_v2.jsonl``
   ``snippet_id`` (expect **0** for this folder; if non-zero, do not treat as
   disjoint holdout without manual review).

**Reuse existing predictions (no torch / no weights):**
  python dataset_construction/build_holdout_audio10_human_validation.py \\
    --from-jsonl human_validation_app/cat_audios_copy_processed/cat_emotion_predictions.jsonl \\
    --skip-v2-leak-check

Run full re-labeling (needs ``checkpoints/best_model_final.pth`` or ``--checkpoint``):
  python dataset_construction/build_holdout_audio10_human_validation.py \\
    --device auto --batch-size 16

Outputs default to ``src/dataset_construction/manifests/holdout_audio10_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

DEFAULT_CHECKPOINT = "checkpoints/best_model_final.pth"
FINETUNED_CHECKPOINT_FALLBACKS = (
    "checkpoints/best_model_final.pth",
    "models/audio_emotions/best_model_final.pth",
)
V2_MANIFEST = "src/dataset_construction/manifests/final_dataset_v2.jsonl"
V1_MANIFEST = "data/dataset/final_dataset.jsonl"


def _collect_audio_paths(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: str(x))


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_checkpoint(repo: Path, arg: str, log: logging.Logger) -> Path | None:
    cand = (repo / arg).resolve()
    if cand.is_file():
        return cand
    if arg in FINETUNED_CHECKPOINT_FALLBACKS:
        for rel in FINETUNED_CHECKPOINT_FALLBACKS:
            if rel == arg:
                continue
            c2 = (repo / rel).resolve()
            if c2.is_file():
                log.info("Using checkpoint %s (requested %s missing)", c2, cand)
                return c2
    return None


def _load_v1_pose_index(repo: Path, log: logging.Logger) -> dict[str, dict[str, Any]]:
    p = repo / V1_MANIFEST
    if not p.is_file():
        log.warning("v1 manifest missing — no pose join: %s", p)
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stem = str(r.get("stem", "") or "")
            if not stem:
                continue
            out[stem] = {
                "pose_path": r.get("pose_path"),
                "low_quality_pose": r.get("low_quality_pose"),
                "label": r.get("label"),
                "label_confidence": r.get("label_confidence"),
                "video_id": r.get("video_id"),
            }
    log.info("Loaded v1 pose index: %d stems from %s", len(out), V1_MANIFEST)
    return out


def _v2_snippet_ids(repo: Path) -> set[str]:
    p = repo / V2_MANIFEST
    if not p.is_file():
        return set()
    s: set[str] = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = str(r.get("snippet_id", "") or "")
            if sid:
                s.add(sid)
    return s


def _stem_from_prediction_row(row: dict[str, Any]) -> str:
    fn = row.get("filename") or Path(str(row.get("filepath", ""))).name
    return Path(str(fn)).stem


def _run_torch_inference(
    paths: list[Path],
    repo: Path,
    ckpt: Path,
    device_s: str,
    batch_size: int,
    log: logging.Logger,
) -> list[dict[str, Any]]:
    from audio_classifier_utils.audio_config import AudioConfig
    from audio_classifier_utils.models.cnn14 import CatEmotionModel

    import librosa
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    def resolve_device(req: str) -> torch.device:
        r = req.lower().strip()
        if r == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if r == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if r == "cpu":
            return torch.device("cpu")
        if r == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device("cpu")

    cfg = AudioConfig()
    pretrained = (repo / cfg.checkpoint_path).resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(f"PANNs pretrained weights missing: {pretrained}")

    try:
        try:
            blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(str(ckpt), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load fine-tuned checkpoint {ckpt}: {e}") from e

    model = CatEmotionModel(cfg)
    model.load_state_dict(blob["model_state_dict"])
    device = resolve_device(device_s)
    model.to(device)
    model.eval()

    def load_wav(path: Path) -> torch.Tensor:
        waveform, _ = librosa.load(str(path), sr=cfg.sample_rate)
        w = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        tl = cfg.target_length
        if w.shape[1] > tl:
            w = w[:, :tl]
        elif w.shape[1] < tl:
            w = F.pad(w, (0, tl - w.shape[1]))
        return w.squeeze(0)

    rows: list[dict[str, Any]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush() -> None:
        if not batch_tensors:
            return
        import torch as T

        x = T.stack(batch_tensors, dim=0).to(device)
        with T.no_grad():
            logits = model(x)
            pr = T.softmax(logits, dim=1).cpu().numpy()
            pred = pr.argmax(axis=1)
        for i, pth in enumerate(batch_paths):
            idx = int(pred[i])
            conf = float(pr[i, idx])
            row: dict[str, Any] = {
                "filepath": str(pth.resolve()),
                "relpath": _relpath(pth),
                "filename": pth.name,
                "predicted_class_idx": idx,
                "predicted_class": IDX_TO_CLASS[idx],
                "confidence": conf,
            }
            for j in range(len(IDX_TO_CLASS)):
                row[f"prob_{IDX_TO_CLASS[j]}"] = float(pr[i, j])
            rows.append(row)
        batch_tensors.clear()
        batch_paths.clear()

    for pth in tqdm(paths, desc="CatEmotion CNN14"):
        try:
            batch_tensors.append(load_wav(pth))
            batch_paths.append(pth)
        except Exception as e:
            log.warning("Failed %s: %s", pth, e)
            rows.append({"filepath": str(pth.resolve()), "relpath": _relpath(pth), "error": str(e)})
            continue
        if len(batch_tensors) >= max(1, batch_size):
            flush()
    flush()
    return rows


def _rows_to_holdout_manifest(
    pred_rows: list[dict[str, Any]],
    v1_index: dict[str, dict[str, Any]],
    *,
    min_conf_for_high: float,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for r in pred_rows:
        if "error" in r:
            continue
        stem = _stem_from_prediction_row(r)
        cls10 = str(r.get("predicted_class", ""))
        conf = float(r.get("confidence", 0.0))
        v1 = v1_index.get(stem, {})
        pose_path = v1.get("pose_path")
        if pose_path:
            pose_path = str(pose_path).replace("\\", "/")
        row: dict[str, Any] = {
            "snippet_id": stem,
            "video_id": v1.get("video_id") or (stem.rsplit("_snip_", 1)[0] if "_snip_" in stem else stem),
            "audio_path": r.get("relpath") or _relpath(Path(str(r["filepath"]))),
            "audio_label_10": cls10,
            "audio_confidence": conf,
            "audio_high_confidence": bool(conf >= min_conf_for_high),
            "suitable_for_training": False,
            "holdout_pool": "src/human_validation/cat_audios_copy_processed",
        }
        for k in (
            "prob_Angry",
            "prob_Defence",
            "prob_Fighting",
            "prob_Happy",
            "prob_HuntingMind",
            "prob_Mating",
            "prob_MotherCall",
            "prob_Paining",
            "prob_Resting",
            "prob_Warning",
        ):
            if k in r:
                row[k] = r[k]
        if pose_path:
            row["pose_path"] = pose_path
            row["low_quality_pose"] = v1.get("low_quality_pose")
            row["human_label_5"] = v1.get("label")
            row["human_label_confidence"] = v1.get("label_confidence")
        manifest.append(row)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build 10-class audio holdout from human-validation MP3s.")
    ap.add_argument(
        "--audio-dir",
        default="src/human_validation/cat_audios_copy_processed",
        help="Directory containing .mp3 (flat).",
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pretrained", default=None, help="Override AudioConfig.checkpoint_path.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--from-jsonl",
        default=None,
        help="Skip torch: read existing prediction rows (same schema as cat_emotion_predictions.jsonl).",
    )
    ap.add_argument(
        "--skip-v2-leak-check",
        action="store_true",
        help="Do not load full v2 id set (faster); not recommended for release artifacts.",
    )
    ap.add_argument(
        "--min-confidence-high",
        type=float,
        default=0.7,
        help="Threshold for audio_high_confidence flag on manifest rows.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: dataset_construction/manifests/holdout_audio10_<UTC>/).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("holdout_audio10")

    repo = REPO_ROOT
    audio_dir = (repo / args.audio_dir).resolve()
    if not audio_dir.is_dir():
        log.error("Audio dir missing: %s", audio_dir)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo / "src/dataset_construction/manifests" / f"holdout_audio10_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _collect_audio_paths(audio_dir)
    if args.limit:
        paths = paths[: args.limit]
    log.info("Audio files: %d under %s", len(paths), audio_dir)

    if args.dry_run:
        log.info("Dry run — would write under %s", out_dir)
        return 0

    if args.from_jsonl:
        jp = Path(args.from_jsonl)
        if not jp.is_file():
            log.error("--from-jsonl not found: %s", jp)
            return 1
        pred_rows = []
        with open(jp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pred_rows.append(json.loads(line))
        log.info("Loaded %d prediction rows from %s", len(pred_rows), jp)
    else:
        ck = _resolve_checkpoint(repo, args.checkpoint, log)
        if ck is None:
            log.error(
                "No fine-tuned weights found. Pass --checkpoint or use --from-jsonl with "
                "src/human_validation/cat_audios_copy_processed/cat_emotion_predictions.jsonl"
            )
            return 1
        pred_rows = _run_torch_inference(paths, repo, ck, args.device, args.batch_size, log)
        pred_path = out_dir / f"audio10_predictions_{ts}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Wrote %s", pred_path)

    v1_index = _load_v1_pose_index(repo, log)
    manifest = _rows_to_holdout_manifest(pred_rows, v1_index, min_conf_for_high=args.min_confidence_high)
    man_path = out_dir / "holdout_manifest_audio10.jsonl"
    with open(man_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %s (%d rows)", man_path, len(manifest))

    stems = {r["snippet_id"] for r in manifest}
    v2_overlap: list[str] = []
    if not args.skip_v2_leak_check:
        v2_ids = _v2_snippet_ids(repo)
        v2_overlap = sorted(stems & v2_ids)
        log.info("v2 snippet_id overlap count: %d", len(v2_overlap))

    counts = Counter(str(r.get("audio_label_10", "")) for r in manifest)
    n_pose = sum(1 for r in manifest if r.get("pose_path"))
    meta = {
        "created_utc": ts,
        "audio_dir": _relpath(audio_dir),
        "n_audio_files_scanned": len(paths),
        "n_manifest_rows": len(manifest),
        "n_with_v1_pose_join": n_pose,
        "class_counts_audio10": dict(counts),
        "v2_manifest_checked": V2_MANIFEST,
        "n_stems_overlapping_v2": len(v2_overlap),
        "v2_overlap_stems_sample": v2_overlap[:50],
        "min_confidence_high": args.min_confidence_high,
        "from_jsonl": args.from_jsonl,
        "class_order": [IDX_TO_CLASS[i] for i in range(10)],
    }
    (out_dir / "holdout_audio10_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "10-class audio holdout (human_validation MP3 pool)",
        f"Manifest: {man_path}",
        f"Rows: {len(manifest)} | with v1 pose_path join: {n_pose}",
        f"Model predictions: {'from --from-jsonl' if args.from_jsonl else 'fresh CNN14 run'}",
        f"v2 training overlap (snippet_id): {len(v2_overlap)} (expect 0 for this folder)",
        "",
        "audio_label_10 counts:",
    ]
    for c, n in counts.most_common():
        lines.append(f"  {c}: {n}")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Done. Output dir: %s", out_dir)
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
