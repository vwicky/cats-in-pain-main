#!/usr/bin/env python3
"""
Extract DeepLabCut SuperAnimal-Quadruped pose for training-ready snippets (final_dataset_v2).

Uses ``deeplabcut.modelzoo.api.superanimal_inference.video_inference`` (full-video inference),
then samples keypoints
at ``target_fps`` into fixed ``n_frames`` with a boolean mask. Does **not** modify
``06_pose_extraction.py`` (ViTPose/YOLO).

Configuration: ``pose_extraction_superanimal`` in ``src/dataset_construction/config.yaml``.

Run from repository root:

  python dataset_construction/06_pose_extraction_superanimal.py
  python dataset_construction/06_pose_extraction_superanimal.py --dry-run
  python dataset_construction/06_pose_extraction_superanimal.py --limit 50
  python dataset_construction/06_pose_extraction_superanimal.py --update-manifest
  python dataset_construction/06_pose_extraction_superanimal.py --rebuild

Conda env (Python 3.11 recommended). DeepLabCut pins ``tables==3.8.0``, which usually **fails**
when installed only with pip; use the helper script (installs ``pytables`` from conda-forge,
then ``deeplabcut --no-deps`` + TensorFlow). Official installation overview:
https://deeplabcut.github.io/DeepLabCut/docs/installation.html (``deeplabcut[tf]``,
``deeplabcut[gui,modelzoo]`` for SuperAnimal, and TensorFlow version notes).

  conda create -n dlc-superanimal python=3.11 -y
  conda activate dlc-superanimal
  cd <YOUR_CLONE_ROOT>
  bash dataset_construction/install_superanimal_deps.sh

Do **not** run ``pip install -r requirements.txt`` afterward in the same env if you need
DeepLabCut: it upgrades NumPy to 2.x and can break this stack.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm")

# SuperAnimal-Quadruped unified vocabulary (id 0..38). ``.npy`` index *i* = this list[i].
# See: https://github.com/AdaptiveMotorControlLab/modelzoo-figures/blob/main/data/superquadruped_dataset.json
SUPERANIMAL_QUADRUPED_BODYPARTS: tuple[str, ...] = (
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "tail_base",
    "tail_end",
    "front_left_thai",
    "front_left_knee",
    "front_left_paw",
    "front_right_thai",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_thai",
    "back_right_thai",
    "back_left_knee",
    "back_right_knee",
    "back_right_paw",
    "belly_bottom",
    "body_middle_right",
    "body_middle_left",
)


def dlc_dataframe_to_pose_array(
    df: pd.DataFrame, n_keypoints: int, warn_flat: bool = True
) -> tuple[np.ndarray | None, str | None]:
    """
    Build (T, K, 3) with K in **vocabulary order** (0 = nose, …). Do not use ``df.values`` order:
    pandas column order is not guaranteed to match keypoint index.
    """
    if n_keypoints != len(SUPERANIMAL_QUADRUPED_BODYPARTS):
        return None, (
            f"config n_keypoints={n_keypoints} but built-in vocabulary has "
            f"{len(SUPERANIMAL_QUADRUPED_BODYPARTS)} bodyparts"
        )
    bodyparts = SUPERANIMAL_QUADRUPED_BODYPARTS

    if isinstance(df.columns, pd.MultiIndex):
        scorer = df.columns.get_level_values(0)[0]
        bp0_cols = [c for c in df.columns if len(c) >= 3 and c[1] == bodyparts[0]]
        zname = next((c[2] for c in bp0_cols if c[2] not in ("x", "y")), "likelihood")
        nrows = len(df)
        out = np.empty((nrows, n_keypoints, 3), dtype=np.float64)
        try:
            for ki, bp in enumerate(bodyparts):
                out[:, ki, 0] = df.loc[:, (scorer, bp, "x")].to_numpy()
                out[:, ki, 1] = df.loc[:, (scorer, bp, "y")].to_numpy()
                try:
                    out[:, ki, 2] = df.loc[:, (scorer, bp, zname)].to_numpy()
                except KeyError:
                    for alt in ("likelihood", "likelihoods", "confidence", "conf"):
                        try:
                            out[:, ki, 2] = df.loc[:, (scorer, bp, alt)].to_numpy()
                            break
                        except KeyError:
                            continue
                    else:
                        raise
            return out.astype(np.float32), None
        except Exception as e:  # noqa: BLE001
            err = str(e)
    else:
        err = "non-MultiIndex columns"

    flat = np.asarray(df.values, dtype=np.float64)
    ncol = flat.shape[1]
    if ncol % 3 != 0 or ncol // 3 != n_keypoints:
        return None, f"{err}; flat fallback shape mismatch"
    if warn_flat:
        print(
            "WARNING: HDF5 MultiIndex read failed; using flat column order "
            f"({err}). Keypoint index order may NOT match visualization vocabulary — "
            "prefer fixing DLC column layout or use ``deeplabcut`` HDF5 MultiIndex.",
            file=sys.stderr,
        )
    return flat.reshape(-1, n_keypoints, 3).astype(np.float32), None


def require_deeplabcut() -> None:
    """Import DeepLabCut once; exit with install hints if missing."""
    try:
        importlib.import_module("deeplabcut")
    except ModuleNotFoundError as e:
        ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(
            "ERROR: The ``deeplabcut`` package is not available in this environment.\n\n"
            "From the repository root, with conda active (Python 3.11 recommended):\n"
            "  bash dataset_construction/install_superanimal_deps.sh\n\n"
            "That installs ``pytables`` from conda-forge (avoids broken pip builds of "
            "``tables==3.8.0``), TensorFlow, then DeepLabCut with ``--no-deps``.\n"
            "Do not run ``pip install -r requirements.txt`` in the same env after that; "
            "it can upgrade NumPy to 2.x and break DeepLabCut.\n"
            "If inference fails with ``keras.legacy_tf_layers`` / ``tf_keras.legacy_tf_layers``,\n"
            "use the install script (Apple Silicon: ``tensorflow-macos`` + ``keras==2.12.0``),\n"
            "or see https://deeplabcut.github.io/DeepLabCut/docs/installation.html for ``deeplabcut[tf]``.\n\n"
            f"Interpreter: {sys.executable}\n"
            f"Python: {ver}\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from e


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_index_last_wins(path: Path) -> dict[str, dict[str, Any]]:
    d: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("snippet_id")
            if isinstance(sid, str):
                d[sid] = row
    return d


def atomic_write_jsonl_from_dict(path: Path, by_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(by_id.keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps(by_id[k], ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_video_path(repo_root: Path, row: dict[str, Any], snippets_dirs: list[Path]) -> Path | None:
    vp = row.get("video_path")
    if isinstance(vp, str) and vp.strip():
        q = repo_root / vp.strip()
        if q.is_file():
            return q
    sid = row.get("snippet_id")
    if isinstance(sid, str):
        for base in snippets_dirs:
            for ext in VIDEO_EXTS:
                c = base / f"{sid}{ext}"
                if c.is_file():
                    return c
    return None


def update_final_manifest(
    repo_root: Path,
    manifest_path: Path,
    index_by_id: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    """Merge pose_* fields into every row; return (n_non_null, n_null)."""
    rows = load_jsonl(manifest_path)
    n_non = 0
    n_null = 0
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        sid = row.get("snippet_id")
        ent = index_by_id.get(sid) if isinstance(sid, str) else None
        if ent and ent.get("status") == "done":
            row["pose_path"] = ent.get("pose_path")
            row["pose_mask_path"] = ent.get("pose_mask_path")
            row["pose_n_real_frames"] = ent.get("n_real_frames")
            row["pose_actual_fps"] = ent.get("actual_fps")
            n_non += 1
        else:
            row["pose_path"] = None
            row["pose_mask_path"] = None
            row["pose_n_real_frames"] = None
            row["pose_actual_fps"] = None
        out_rows.append(row)

    tmp = manifest_path.parent / (manifest_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(manifest_path)
    return n_non, n_null


def extract_and_sample_pose(
    video_path: str,
    target_fps: float,
    n_frames: int,
    max_duration: float,
    n_keypoints: int,
    pad_value: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """
    1. Runs DeepLabCut SuperAnimal-Quadruped on the video (TensorFlow model zoo path).
    2. Loads the resulting .h5 file.
    3. Samples the pose array at fixed target_fps.
    4. Returns padded pose array (n_frames, n_keypoints, 3) and mask (n_frames,).

    Uses ``deeplabcut.modelzoo.api.superanimal_inference.video_inference`` with
    ``destfolder`` so HDF5 output stays in a temp directory. DeepLabCut 2.3.x
    ``video_inference_superanimal`` does not accept ``model_name`` / ``detector_name`` /
    ``destfolder``; model choice is fixed by the ``superanimal_quadruped`` zoo entry.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, {"error": "cannot_open"}
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if actual_fps <= 1e-6:
        return None, None, {"error": "invalid_fps", "actual_fps": actual_fps}

    from deeplabcut.modelzoo.api import superanimal_inference  # noqa: PLC0415

    suf = Path(video_path).suffix.lower()
    videotype = suf if suf else ".mp4"
    if not videotype.startswith("."):
        videotype = "." + videotype

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _init_w, datafiles = superanimal_inference.video_inference(
                [video_path],
                "superanimal_quadruped",
                scale_list=[],
                videotype=videotype,
                destfolder=str(tmpdir),
                batchsize=1,
            )
        except Exception as e:  # noqa: BLE001
            return None, None, {"error": f"DLC inference failed: {e!s}"}

        if not datafiles:
            return None, None, {"error": "DLC did not return h5 paths"}
        h5_path = Path(datafiles[0])
        if not h5_path.is_file():
            return None, None, {"error": f"DLC h5 missing: {h5_path}"}

        df = pd.read_hdf(h5_path)
        ncol = int(np.asarray(df.values).shape[1])
        if ncol % 3 != 0:
            return None, None, {"error": f"h5 column count {ncol} is not a multiple of 3"}
        n_kp_h5 = ncol // 3
        if n_kp_h5 != n_keypoints:
            return None, None, {
                "error": (
                    f"h5 has {n_kp_h5} keypoints (×3 = {ncol} cols) but config n_keypoints={n_keypoints}; "
                    f"set pose_extraction_superanimal.n_keypoints to {n_kp_h5}"
                ),
            }
        raw_pose, conv_err = dlc_dataframe_to_pose_array(df, n_keypoints, warn_flat=True)
        if raw_pose is None:
            return None, None, {"error": conv_err or "hdf_to_pose_failed"}

        clip_duration = min(total_frames / actual_fps, max_duration)
        n_real = min(int(np.floor(clip_duration * target_fps)), n_frames)

        pose_out = np.full((n_frames, n_keypoints, 3), pad_value, dtype=np.float32)
        mask_out = np.zeros(n_frames, dtype=bool)

        for i in range(n_real):
            t_sec = i / target_fps if target_fps > 0 else 0.0
            frame_idx = int(round(t_sec * actual_fps))
            frame_idx = int(np.clip(frame_idx, 0, raw_pose.shape[0] - 1))
            pose_out[i] = raw_pose[frame_idx]
            mask_out[i] = True

        stats: dict[str, Any] = {
            "n_real_frames": n_real,
            "n_padded_frames": n_frames - n_real,
            "actual_fps": actual_fps,
            "clip_duration_sec": clip_duration,
        }
        return pose_out, mask_out, stats


def validate_random_outputs(
    repo_root: Path,
    index_by_id: dict[str, dict[str, Any]],
    n_frames: int,
    n_kp: int,
    k: int = 5,
) -> None:
    """Sanity-check a few completed extractions (pixel x,y; no unit-range assert)."""
    done_ids = [s for s, r in index_by_id.items() if r.get("status") == "done"]
    if len(done_ids) < 1:
        print("Validation: no done entries in index to sample.")
        return
    sample = random.sample(done_ids, k=min(k, len(done_ids)))
    print("\n--- Validation (random samples, SuperAnimal) ---")
    for sid in sample:
        r = index_by_id[sid]
        pp = repo_root / str(r.get("pose_path", ""))
        pm = repo_root / str(r.get("pose_mask_path", ""))
        if not pp.is_file() or not pm.is_file():
            print(f"WARNING: missing files for {sid}")
            continue
        pose = np.load(pp)
        m = np.load(pm)
        ok = True
        if pose.shape != (n_frames, n_kp, 3):
            print(f"WARNING: {sid} bad pose shape {pose.shape}")
            ok = False
        if pose.dtype != np.float32:
            print(f"WARNING: {sid} dtype {pose.dtype}")
            ok = False
        if m.shape != (n_frames,) or not np.issubdtype(m.dtype, np.bool_):
            print(f"WARNING: {sid} bad mask {m.shape} {m.dtype}")
            ok = False
        padded = int(np.sum(~m))
        mean_conf = float(np.mean(pose[:, :, 2]))
        print(
            f"{sid} | shape={pose.shape} dtype={pose.dtype} mean_conf={mean_conf:.4f} "
            f"padded_frames={padded}"
            + (" OK" if ok else " ISSUES")
        )


def compare_v1_superanimal_sample(
    repo_root: Path,
    legacy_manifest: Path,
    index_by_id: dict[str, dict[str, Any]],
) -> None:
    """Print one legacy pose file vs one SuperAnimal pose when both exist."""
    legacy_path: Path | None = None
    if legacy_manifest.is_file():
        for row in load_jsonl(legacy_manifest):
            pp = row.get("pose_path")
            if pp and isinstance(pp, str):
                c = repo_root / pp
                if c.is_file():
                    legacy_path = c
                    break
    new_id = next((s for s, r in index_by_id.items() if r.get("status") == "done"), None)
    if legacy_path is None:
        print("\nLegacy vs SuperAnimal: no legacy pose file found; skipped.")
        return
    if new_id is None:
        print("\nLegacy vs SuperAnimal: no SuperAnimal pose in index; skipped.")
        return
    v1 = np.load(legacy_path)
    np2 = repo_root / str(index_by_id[new_id]["pose_path"])
    v2 = np.load(np2)
    print("\n--- Legacy vs SuperAnimal (shape / dtype / value ranges) ---")
    try:
        leg_disp = legacy_path.relative_to(repo_root)
    except ValueError:
        leg_disp = legacy_path
    print(f"Legacy {leg_disp}: shape {v1.shape} dtype {v1.dtype}")
    print(
        f"  x[{v1[:, :, 0].min():.4f},{v1[:, :, 0].max():.4f}] "
        f"y[{v1[:, :, 1].min():.4f},{v1[:, :, 1].max():.4f}] "
        f"likelihood[{v1[:, :, 2].min():.4f},{v1[:, :, 2].max():.4f}]"
    )
    print(f"SuperAnimal {np2.relative_to(repo_root)}: shape {v2.shape} dtype {v2.dtype}")
    print(
        f"  x[{v2[:, :, 0].min():.4f},{v2[:, :, 0].max():.4f}] "
        f"y[{v2[:, :, 1].min():.4f},{v2[:, :, 1].max():.4f}] "
        f"likelihood[{v2[:, :, 2].min():.4f},{v2[:, :, 2].max():.4f}]"
    )


def run(args: argparse.Namespace) -> int:
    cfg_all = load_config()
    pe = cfg_all.get("pose_extraction_superanimal") or {}
    fd = cfg_all.get("final_dataset") or {}
    sources = cfg_all.get("sources") or {}

    if not pe:
        print(
            "ERROR: config missing ``pose_extraction_superanimal`` block "
            f"in {CONFIG_PATH.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        return 2

    manifest_path = REPO_ROOT / fd["output_manifest"]
    cache_index_path = REPO_ROOT / pe["cache_index"]
    output_dir = REPO_ROOT / pe["output_dir"]
    legacy_manifest = REPO_ROOT / pe.get("legacy_pose_manifest", "data/dataset/final_dataset.jsonl")

    n_frames = int(pe["n_frames"])
    target_fps = float(pe["target_fps"])
    max_dur = float(pe["max_clip_duration"])
    pad_value = float(pe.get("pad_value", 0.0))
    n_kp = int(pe.get("n_keypoints", 39))
    skip_existing = bool(pe.get("skip_existing", True)) and not args.rebuild

    snippets_dirs = [REPO_ROOT / x for x in sources.get("snippets_dirs", [])]

    if args.update_manifest:
        idx = load_index_last_wins(cache_index_path)
        n_ok, n_bad = update_final_manifest(REPO_ROOT, manifest_path, idx)
        print(f"pose fields added: {n_ok} rows")
        print(f"pose fields null: {n_bad} rows (not extracted or error)")
        return 0

    rows_v2 = load_jsonl(manifest_path)
    train_ready = [r for r in rows_v2 if r.get("suitable_for_training") is True]
    scheduled = train_ready[: args.limit] if args.limit else train_ready
    scheduled_ids = {r["snippet_id"] for r in scheduled if isinstance(r.get("snippet_id"), str)}

    index_by_id = load_index_last_wins(cache_index_path)

    if args.rebuild:
        for sid in scheduled_ids:
            index_by_id.pop(sid, None)
        atomic_write_jsonl_from_dict(cache_index_path, index_by_id)

    to_process_list: list[dict[str, Any]] = []
    for r in scheduled:
        sid = r.get("snippet_id")
        if not isinstance(sid, str):
            continue
        if skip_existing and not args.rebuild:
            prev = index_by_id.get(sid)
            if prev and prev.get("status") == "done":
                continue
        to_process_list.append(r)

    n_skip = len(scheduled) - len(to_process_list)
    print(
        f"SuperAnimal pose extraction: {len(to_process_list)} to process, "
        f"{n_skip} already done (skipping)"
    )

    if args.dry_run:
        print("Dry run: exiting before DeepLabCut inference.")
        return 0

    if len(to_process_list) > 0:
        require_deeplabcut()

    output_dir.mkdir(parents=True, exist_ok=True)

    def finish_from_index(skip_inference: bool = False) -> int:
        idx = load_index_last_wins(cache_index_path)
        n_ok, n_bad = update_final_manifest(REPO_ROOT, manifest_path, idx)
        print(f"\npose fields added: {n_ok} rows")
        print(f"pose fields null: {n_bad} rows (not extracted or error)")
        validate_random_outputs(REPO_ROOT, idx, n_frames, n_kp)
        compare_v1_superanimal_sample(REPO_ROOT, legacy_manifest, idx)
        if skip_inference:
            print("\n(No clips processed this run — manifest refreshed from index.)")
        return 0

    if len(to_process_list) == 0:
        print("Nothing to process.")
        return finish_from_index(skip_inference=True)

    done_c = err_c = 0
    pad_none = pad_full = 0
    index_fp = open(cache_index_path, "a", encoding="utf-8")
    t0_run = time.perf_counter()
    processed_in_loop = 0

    pbar = tqdm(total=len(to_process_list), desc="SuperAnimal pose", mininterval=0.5)

    def flush_index() -> None:
        index_fp.flush()

    def base_index_rec(sid: str) -> dict[str, Any]:
        return {
            "snippet_id": sid,
            "pose_path": None,
            "pose_mask_path": None,
            "status": "error",
            "error_msg": None,
            "n_frames_extracted": n_frames,
            "n_real_frames": 0,
            "n_padded_frames": n_frames,
            "actual_fps": None,
            "clip_duration_sec": None,
            "latency_sec": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": "superanimal_quadruped",
        }

    try:
        for row in to_process_list:
            sid = row.get("snippet_id")
            if not isinstance(sid, str):
                sid = str(sid)

            video_path = resolve_video_path(REPO_ROOT, row, snippets_dirs)
            if video_path is None:
                err_c += 1
                rec = base_index_rec(sid)
                rec["error_msg"] = "video_not_found"
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec
                pbar.update(1)
                pbar.set_postfix(done=done_c, err=err_c)
                processed_in_loop += 1
                continue

            t_clip = time.perf_counter()
            pose_array, pose_mask, stats = extract_and_sample_pose(
                video_path=str(video_path),
                target_fps=target_fps,
                n_frames=n_frames,
                max_duration=max_dur,
                n_keypoints=n_kp,
                pad_value=pad_value,
            )

            if pose_array is None or pose_mask is None:
                err_c += 1
                rec = base_index_rec(sid)
                rec["error_msg"] = str(stats.get("error", "unknown"))[:500]
                rec["latency_sec"] = round(time.perf_counter() - t_clip, 4)
                for k, v in stats.items():
                    if k != "error" and k not in rec:
                        rec[k] = v
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec
                pbar.update(1)
                pbar.set_postfix(done=done_c, err=err_c)
                processed_in_loop += 1
                continue

            try:
                rel_pose = output_dir / f"{sid}_pose.npy"
                rel_mask = output_dir / f"{sid}_pose_mask.npy"
                abs_pose = REPO_ROOT / rel_pose
                abs_mask = REPO_ROOT / rel_mask
                np.save(abs_pose, pose_array.astype(np.float32))
                np.save(abs_mask, pose_mask.astype(np.bool_))

                elapsed = time.perf_counter() - t_clip
                n_real = int(stats["n_real_frames"])
                if int(stats["n_padded_frames"]) == 0:
                    pad_none += 1
                if n_real >= n_frames:
                    pad_full += 1

                rec: dict[str, Any] = {
                    "snippet_id": sid,
                    "pose_path": str(rel_pose).replace("\\", "/"),
                    "pose_mask_path": str(rel_mask).replace("\\", "/"),
                    "status": "done",
                    "error_msg": None,
                    "n_frames_extracted": n_frames,
                    "n_real_frames": stats["n_real_frames"],
                    "n_padded_frames": stats["n_padded_frames"],
                    "actual_fps": stats["actual_fps"],
                    "clip_duration_sec": stats["clip_duration_sec"],
                    "latency_sec": round(elapsed, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pipeline": "superanimal_quadruped",
                }
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec
                done_c += 1
            except Exception as ex:  # noqa: BLE001
                err_c += 1
                rec = base_index_rec(sid)
                rec["error_msg"] = str(ex)[:500]
                rec["latency_sec"] = round(time.perf_counter() - t_clip, 4)
                index_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flush_index()
                index_by_id[sid] = rec

            processed_in_loop += 1
            pbar.update(1)
            pbar.set_postfix(done=done_c, err=err_c)

            if processed_in_loop % 50 == 0 and processed_in_loop > 0:
                elapsed_run = time.perf_counter() - t0_run
                rate = processed_in_loop / elapsed_run if elapsed_run > 0 else 0
                remaining = len(to_process_list) - processed_in_loop
                eta_sec = remaining / rate if rate > 0 else 0
                print(
                    f"\n[{processed_in_loop}/{len(to_process_list)}] done={done_c} err={err_c} | "
                    f"elapsed={elapsed_run / 60:.1f}m | eta={eta_sec / 3600:.1f}h"
                )

    except KeyboardInterrupt:
        print("\nInterrupted; flushing index...")
        flush_index()
        index_fp.close()
        print(f"Progress: done={done_c} err={err_c}")
        return 130
    finally:
        if not index_fp.closed:
            index_fp.close()

    index_by_id = load_index_last_wins(cache_index_path)
    n_ok, n_bad = update_final_manifest(REPO_ROOT, manifest_path, index_by_id)
    print(f"\npose fields added: {n_ok} rows")
    print(f"pose fields null: {n_bad} rows (not extracted or error)")
    validate_random_outputs(REPO_ROOT, index_by_id, n_frames, n_kp)
    compare_v1_superanimal_sample(REPO_ROOT, legacy_manifest, index_by_id)

    total_train = len(train_ready)
    bytes_per = n_frames * n_kp * 3 * 4 + n_frames * 1
    mean_kb = bytes_per / 1024.0
    denom = max(1, len(to_process_list))
    pct_ok = 100.0 * done_c / denom
    pct_err = 100.0 * err_c / denom

    print("\n" + "═" * 56)
    print("SUPERANIMAL POSE EXTRACTION SUMMARY")
    print("═" * 56)
    print(f"Training-ready snippets:     {total_train}")
    print(f"Successfully extracted:      {done_c}  ({pct_ok:.1f}% of queued this run)")
    print(f"Errors (this run):           {err_c}  ({pct_err:.1f}% of queued this run)")
    print(f"Skipped before run:          {n_skip}")
    print("")
    print("Frame padding (this run):")
    if done_c > 0:
        dr = float(done_c)
        print(f"No padded frames:          {pad_none} clips ({100.0 * pad_none / dr:.1f}% of successes)")
        print(f"Fully utilized ({n_frames} real): {pad_full} clips ({100.0 * pad_full / dr:.1f}%)")
    else:
        print("  (no successful extractions this run)")
    print("")
    print(f"Output:\nDirectory:  {output_dir.relative_to(REPO_ROOT)}/")
    print(f"File size:  mean ~{mean_kb:.1f} KB per clip (pose + mask)")
    print("")
    print(f"pose fields written to final manifest: {n_ok} rows")
    print("═" * 56)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--update-manifest", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
