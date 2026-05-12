#!/usr/bin/env python3
"""
Batch cat-sound separation with AudioSep. Paths are anchored to this folder
(audios_to_seperate/): metadata, MP3 root, separated WAVs, and batch log.
Config and checkpoint load from the repository root (parent of this folder).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

# This folder (…/AudioSep/audios_to_seperate); repo root is one level up.
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── AudioSep model ───────────────────────────────────────────
AUDIOSEP_CONFIG = REPO_ROOT / "config/audiosep_base.yaml"
AUDIOSEP_CKPT = REPO_ROOT / "checkpoint/audiosep_base_4M_steps.ckpt"
SEPARATION_QUERY = "cat sounds"
USE_CHUNK = False  # set True for very long files (>60s)

# ── Data (all under audios_to_seperate/) ─────────────────────
METADATA_FILE = _HERE / "audios_to_seperate" / "metadata_merged.jsonl"
AUDIO_ROOT = _HERE / "audios_to_seperate"  # root to resolve relative paths / glob MP3s
OUTPUT_DIR = _HERE / "separated_cat_sounds"

# ── Runtime ──────────────────────────────────────────────────
DEVICE = "mps"  # "cuda" | "cpu" | "mps"
WORKERS = 1  # AudioSep is not thread-safe; keep at 1
LOG_FILE = _HERE / "separation_log_batch.jsonl"


class _StdoutViaTqdm:
    """Send stdout through tqdm.write so the progress bar is not torn by prints."""

    __slots__ = ("_buf", "_real")

    def __init__(self, real_stdout) -> None:
        self._buf = ""
        self._real = real_stdout

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        n = len(s)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            # Must pass real stdout: tqdm.write defaults to sys.stdout, which is this object → recursion.
            tqdm.write(line.rstrip("\r"), file=self._real)
        return n

    def flush(self) -> None:
        if self._buf:
            tqdm.write(self._buf.rstrip("\r"), file=self._real)
            self._buf = ""

    def isatty(self) -> bool:
        return False


def resolve_requested_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("DEVICE is 'cuda' but CUDA is not available; using CPU.")
        return torch.device("cpu")
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("DEVICE is 'mps' but MPS is not available; using CPU.")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    warnings.warn(f"Unknown device '{requested}'; using CPU.")
    return torch.device("cpu")


def resolve_audio_path(raw: str, audio_root: Path) -> Path | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        if p.is_file():
            return p.resolve()
        return None
    cand = audio_root / raw
    if cand.is_file():
        return cand.resolve()
    cand2 = audio_root / p.name
    if cand2.is_file():
        return cand2.resolve()
    return None


def snippet_raw_path(snippet: dict) -> str | None:
    for key in ("audio_path", "audio_file", "mp3_path"):
        if key in snippet and snippet[key]:
            return str(snippet[key])
    sid = snippet.get("id")
    if sid:
        return f"{sid}.mp3"
    return None


def collect_from_metadata(
    metadata_path: Path, audio_root: Path
) -> tuple[list[Path], int]:
    """Returns (sorted unique paths, total snippet reference count)."""
    total_refs = 0
    seen: dict[str, Path] = {}

    with metadata_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skip line {line_num} (invalid JSON): {e}", file=sys.stderr)
                continue
            snippets = record.get("snippets") or []
            if not isinstance(snippets, list):
                continue
            for snip in snippets:
                if not isinstance(snip, dict):
                    continue
                total_refs += 1
                raw = snippet_raw_path(snip)
                if not raw:
                    print(
                        f"Warning: snippet without path/id at line {line_num}; skipping.",
                        file=sys.stderr,
                    )
                    continue
                if Path(raw).suffix.lower() != ".mp3":
                    print(
                        f"Warning: non-mp3 path skipped: {raw!r} (line {line_num})",
                        file=sys.stderr,
                    )
                    continue
                resolved = resolve_audio_path(raw, audio_root)
                if resolved is None:
                    print(
                        f"Warning: file not found for snippet path {raw!r} (line {line_num})",
                        file=sys.stderr,
                    )
                    continue
                if resolved.suffix.lower() != ".mp3":
                    print(f"Warning: resolved path is not .mp3: {resolved}", file=sys.stderr)
                    continue
                seen[str(resolved)] = resolved

    unique = sorted(seen.values(), key=lambda p: str(p))
    return unique, total_refs


def collect_from_glob(audio_root: Path) -> tuple[list[Path], int]:
    if not audio_root.is_dir():
        return [], 0
    paths: list[Path] = []
    for p in audio_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp3":
            paths.append(p.resolve())
    unique = sorted(set(paths), key=lambda x: str(x))
    return unique, len(unique)


def build_skip_stems(output_dir: Path) -> set[str]:
    if not output_dir.is_dir():
        return set()
    return {p.stem for p in output_dir.glob("*.wav") if p.is_file()}


def append_log(
    path: Path,
    stem: str,
    source: str,
    output: str,
    status: str,
    error_msg: str | None,
    latency_sec: float | None,
) -> None:
    entry = {
        "stem": stem,
        "source": source,
        "output": output,
        "status": status,
        "error_msg": error_msg,
        "latency_sec": latency_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch separate cat sounds with AudioSep.")
    parser.add_argument("--dry-run", action="store_true", help="List files only; no separation.")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Process first N files only.")
    parser.add_argument("--device", type=str, default=None, choices=("cuda", "cpu", "mps"), help="Override DEVICE.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore skip set and redo all.")
    args = parser.parse_args()

    device_req = args.device if args.device else DEVICE
    device = resolve_requested_device(device_req)
    print(f"Using device: {device}")

    audio_root = AUDIO_ROOT
    output_dir = OUTPUT_DIR
    log_path = LOG_FILE
    meta_path = METADATA_FILE

    if meta_path.is_file():
        print(f"Metadata mode: reading {meta_path}")
        files, total_refs = collect_from_metadata(meta_path, audio_root)
        print(
            f"Snippet references: {total_refs} | Unique MP3 files (existing on disk): {len(files)}"
        )
    else:
        print(f"METADATA_FILE not found ({meta_path}); glob mode: {audio_root}/**/*.mp3")
        files, total_refs = collect_from_glob(audio_root)
        print(f"Unique MP3 files found: {len(files)}")

    if not files:
        print("No input MP3 files to process.", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        files = files[: max(0, args.limit)]

    output_dir.mkdir(parents=True, exist_ok=True)

    skip_stems: set[str] = set()
    if not args.rebuild:
        skip_stems = build_skip_stems(output_dir)
        print(f"Found {len(skip_stems)} existing separated files — will skip these")
    else:
        print("Rebuild mode: not skipping existing outputs.")

    if args.dry_run:
        print(f"Dry run: would process {len(files)} file(s).")
        for p in files[:50]:
            print(f"  {p}")
        if len(files) > 50:
            print(f"  ... and {len(files) - 50} more.")
        return

    from pipeline import build_audiosep, separate_audio

    ckpt = AUDIOSEP_CKPT
    if not ckpt.is_file():
        print(
            f"Error: checkpoint not found at expected path:\n  {ckpt.resolve()}",
            file=sys.stderr,
        )
        sys.exit(1)

    model = build_audiosep(
        str(AUDIOSEP_CONFIG),
        str(AUDIOSEP_CKPT),
        device,
    )

    done = skipped = errors = 0
    error_samples: list[tuple[str, str]] = []
    t0 = time.perf_counter()
    total = len(files)

    pbar = tqdm(
        files,
        desc="Cat sound separation",
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} | {postfix}",
        postfix=dict(done=0, skip=0, err=0),
        dynamic_ncols=True,
    )

    old_stdout = sys.stdout
    sys.stdout = _StdoutViaTqdm(old_stdout)
    try:
        for i, src in enumerate(pbar):
            stem = src.stem
            out_path = output_dir / f"{stem}.wav"
            src_s = str(src)
            out_s = str(out_path.resolve())

            if not args.rebuild and stem in skip_stems:
                skipped += 1
                append_log(
                    log_path,
                    stem,
                    src_s,
                    out_s,
                    "skipped",
                    None,
                    None,
                )
            else:
                t_sep = time.perf_counter()
                try:
                    separate_audio(
                        model,
                        src_s,
                        SEPARATION_QUERY,
                        out_s,
                        device=device,
                        use_chunk=USE_CHUNK,
                    )
                    latency = time.perf_counter() - t_sep
                    if not out_path.is_file() or out_path.stat().st_size <= 0:
                        raise RuntimeError("Output missing or empty after separation.")
                    done += 1
                    append_log(
                        log_path,
                        stem,
                        src_s,
                        out_s,
                        "done",
                        None,
                        round(latency, 4),
                    )
                except Exception as e:
                    errors += 1
                    msg = f"{type(e).__name__}: {e}"
                    if len(error_samples) < 10:
                        error_samples.append((src.name, msg))
                    append_log(
                        log_path,
                        stem,
                        src_s,
                        out_s,
                        "error",
                        msg,
                        round(time.perf_counter() - t_sep, 4),
                    )

            pbar.set_postfix(done=done, skip=skipped, err=errors)

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                n_done = i + 1
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = total - n_done
                eta_sec = remaining / rate if rate > 0 else 0
                print(
                    f"\n[{n_done}/{total}] done={done} skip={skipped} err={errors} "
                    f"| elapsed={format_duration(elapsed)} | eta={format_duration(eta_sec)}"
                )

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t0
        print(
            f"\nInterrupted. Progress: done={done} skipped={skipped} errors={errors} "
            f"| elapsed={format_duration(elapsed)} | log: {log_path}",
            file=sys.stderr,
        )
        sys.exit(130)
    finally:
        sys.stdout = old_stdout

    elapsed = time.perf_counter() - t0
    pct = lambda n: (100.0 * n / total) if total else 0.0

    print()
    print("  ════════════════════════════════════════════════════════")
    print('  CAT SOUND SEPARATION COMPLETE')
    print('  Query: "cat sounds" | Model: AudioSep')
    print("  ════════════════════════════════════════════════════════")
    print(f"  Total files:     {total:,}")
    print(f"  Separated:       {done:,}  ({pct(done):.1f}%)")
    print(f"  Skipped:         {skipped:,}  (already existed)")
    print(f"  Errors:          {errors:,}  ({pct(errors):.1f}%)")
    print(f"  Output dir:      {output_dir}/")
    print(f"  Log:             {log_path}")
    print(f"  Elapsed:         {format_duration(elapsed)}")
    print("  ════════════════════════════════════════════════════════")

    if errors > 0 and error_samples:
        print("\nFirst errors (up to 10):")
        for name, msg in error_samples:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
