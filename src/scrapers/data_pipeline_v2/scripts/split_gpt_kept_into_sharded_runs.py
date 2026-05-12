#!/usr/bin/env python3
"""
Split ``stage_3_gpt_filter/kept.jsonl`` from a finished GPT-filter stage into N run
directories so download+process can run in parallel batches (or sequentially).

Earlier stages are **symlinked** from the source run (no duplicate huge JSONL files).

Example (repo root):

  python data_pipeline_v2/scripts/split_gpt_kept_into_sharded_runs.py \\
    data_pipeline_v2/runs/pipeline_v2_run_20260413_120918 \\
    --shards 6 \\
    --prefix pipeline_v2_run_20260413_120918_dl

Then start each shard with ``--start-from download --resume-run-dir <shard_dir>``.

**Note:** Appending to the same ``output.metadata_file`` / ``pipeline_log`` from multiple
processes at once can interleave lines badly. Prefer running the six commands **one after
another**, or only parallelize download if you use separate metadata paths per shard (advanced).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _chunk_evenly(rows: list[dict], n_chunks: int) -> list[list[dict]]:
    n = len(rows)
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")
    base, rem = divmod(n, n_chunks)
    out: list[list[dict]] = []
    i = 0
    for k in range(n_chunks):
        size = base + (1 if k < rem else 0)
        out.append(rows[i : i + size])
        i += size
    return out


def _symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split GPT-filter kept.jsonl into N sharded run dirs (symlink stages 1–2).",
    )
    p.add_argument(
        "source_run_dir",
        type=Path,
        help="Run directory that has stage_3_gpt_filter/kept.jsonl (and stages 1–2).",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=6,
        help="Number of shards (default 6).",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Run folder name prefix (default: <source_run_dir.name>_shard).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Parent for new dirs (default: same parent as source run).",
    )
    args = p.parse_args()

    src = args.source_run_dir.resolve()
    kept_path = src / "stage_3_gpt_filter" / "kept.jsonl"
    if not kept_path.is_file():
        raise SystemExit(f"Missing {kept_path}")

    text = kept_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {kept_path}")

    n_shards = max(1, int(args.shards))
    chunks = _chunk_evenly(rows, n_shards)
    prefix = args.prefix or f"{src.name}_shard"
    runs_root = (args.runs_root or src.parent).resolve()

    s1 = src / "stage_1_search" / "candidates.jsonl"
    s2 = src / "stage_2_tag_filter" / "kept.jsonl"
    for pth, label in ((s1, "stage_1_search/candidates.jsonl"), (s2, "stage_2_tag_filter/kept.jsonl")):
        if not pth.is_file():
            raise SystemExit(f"Source run missing {label}; cannot symlink.")

    cu = src / "config_used.yaml"
    created: list[Path] = []

    for idx, chunk in enumerate(chunks, start=1):
        name = f"{prefix}_{idx:02d}_of_{n_shards:02d}"
        shard = runs_root / name
        if shard.exists():
            raise SystemExit(f"Refusing to overwrite existing path: {shard}")
        for sub in (
            "stage_1_search",
            "stage_2_tag_filter",
            "stage_3_gpt_filter",
            "stage_4_download",
            "stage_5_process",
        ):
            (shard / sub).mkdir(parents=True, exist_ok=True)

        _symlink_file(s1, shard / "stage_1_search" / "candidates.jsonl")
        _symlink_file(s2, shard / "stage_2_tag_filter" / "kept.jsonl")

        out_kept = shard / "stage_3_gpt_filter" / "kept.jsonl"
        out_kept.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chunk),
            encoding="utf-8",
        )

        if cu.is_file():
            (shard / "config_used.yaml").write_text(cu.read_text(encoding="utf-8"), encoding="utf-8")

        man = shard / "shard_manifest.json"
        man.write_text(
            json.dumps(
                {
                    "source_run_dir": str(src),
                    "source_kept_rows": len(rows),
                    "shard_index": idx,
                    "shard_count": n_shards,
                    "rows_in_shard": len(chunk),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        created.append(shard)
        print(f"{idx}/{n_shards}: {len(chunk):5d} rows -> {shard}")

    print("\n--- Launch download+process for each shard (run from repo root) ---\n")
    for shard in created:
        cmd = (
            f'python data_pipeline_v2/src/pipeline.py \\\n'
            f'  --config data_pipeline_v2/config/pipeline.yaml \\\n'
            f'  --start-from download \\\n'
            f'  --stop-after process \\\n'
            f'  --resume-run-dir {shard}'
        )
        print(cmd)
        print()


if __name__ == "__main__":
    main()
