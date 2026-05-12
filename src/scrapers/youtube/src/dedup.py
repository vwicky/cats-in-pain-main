"""Global video registry: deduplication and atomic persistence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_STATUS_OPTIONS = [
    "candidate",
    "tag_filtered_out",
    "gpt_filtered_out",
    "download_failed",
    "no_valid_segments",
    "snippets_saved",
    "skipped_duplicate",
]

DEFAULT_SKIP_STATUSES = ("snippets_saved", "skipped_duplicate", "no_valid_segments")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s: str | None) -> float:
    if not s or not isinstance(s, str):
        return 0.0
    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(t).timestamp()
    except ValueError:
        return 0.0


class VideoRegistry:
    """Manages global_video_registry.jsonl."""

    def __init__(self, registry_path: Path, machine_id: str):
        self._path = Path(registry_path)
        self._machine_id = machine_id
        self._by_id: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        """Load registry. If file missing, start empty."""
        self._by_id.clear()
        if not self._path.is_file():
            return
        for row in self._read_lines(self._path):
            vid = row.get("video_id")
            if isinstance(vid, str) and vid:
                self._by_id[vid] = row

    @staticmethod
    def _read_lines(path: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def is_known(self, video_id: str) -> bool:
        return str(video_id) in self._by_id

    def get_status(self, video_id: str) -> str | None:
        r = self._by_id.get(str(video_id))
        if not r:
            return None
        st = r.get("pipeline_status")
        return str(st) if st is not None else None

    def get_record(self, video_id: str) -> dict[str, Any] | None:
        r = self._by_id.get(str(video_id))
        return dict(r) if r else None

    def upsert(self, video_id: str, fields: dict[str, Any]) -> None:
        """Insert new record or update existing. Preserves video_id, platform, first_seen_at."""
        vid = str(video_id)
        now = _utc_now_iso()
        incoming = {k: v for k, v in fields.items() if k != "video_id"}
        if vid not in self._by_id:
            row: dict[str, Any] = {
                "video_id": vid,
                "platform": incoming.get("platform", "youtube"),
                "first_seen_at": incoming.get("first_seen_at", now),
            }
            row.update(incoming)
            row["video_id"] = vid
            row["last_updated_at"] = now
            row["machine_id"] = self._machine_id
            self._by_id[vid] = row
            return

        existing = self._by_id[vid]
        merged = dict(existing)
        for k, v in incoming.items():
            if k in ("video_id", "platform", "first_seen_at"):
                continue
            merged[k] = v
        merged["last_updated_at"] = now
        merged["machine_id"] = self._machine_id
        self._by_id[vid] = merged

    def save(self) -> None:
        """Atomic write: tmp → fsync → replace."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=self._path.name + ".",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for vid in sorted(self._by_id.keys()):
                    f.write(json.dumps(self._by_id[vid], ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def build_skip_set(self, skip_statuses: list[str] | None = None) -> set[str]:
        if skip_statuses is None:
            skip_statuses = list(DEFAULT_SKIP_STATUSES)
        skip_s = set(skip_statuses)
        out: set[str] = set()
        for vid, row in self._by_id.items():
            st = row.get("pipeline_status")
            if st in skip_s:
                out.add(vid)
        return out

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s: 0 for s in PIPELINE_STATUS_OPTIONS}
        counts["other"] = 0
        for row in self._by_id.values():
            st = row.get("pipeline_status")
            key = str(st) if st else "other"
            if key in counts:
                counts[key] += 1
            else:
                counts["other"] += 1
        counts["total"] = len(self._by_id)
        return counts


def merge_registries(path_a: Path, path_b: Path, output: Path) -> dict[str, Any]:
    """
    Dedup by video_id — last_updated_at wins on conflict.
    Returns: {total, from_a_only, from_b_only, conflicts_resolved}
    """
    a_rows = VideoRegistry._read_lines(path_a) if path_a.is_file() else []
    b_rows = VideoRegistry._read_lines(path_b) if path_b.is_file() else []

    a_by: dict[str, dict[str, Any]] = {}
    for r in a_rows:
        vid = r.get("video_id")
        if vid:
            a_by[str(vid)] = r
    b_by: dict[str, dict[str, Any]] = {}
    for r in b_rows:
        vid = r.get("video_id")
        if vid:
            b_by[str(vid)] = r

    all_ids = set(a_by.keys()) | set(b_by.keys())
    winners: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for vid in all_ids:
        ra = a_by.get(vid)
        rb = b_by.get(vid)
        if ra is not None and rb is not None:
            conflicts += 1
            ta = _parse_ts(ra.get("last_updated_at"))
            tb = _parse_ts(rb.get("last_updated_at"))
            winners[vid] = rb if tb >= ta else ra
        elif ra is not None:
            winners[vid] = ra
        elif rb is not None:
            winners[vid] = rb

    total = len(winners)
    from_a_only = sum(1 for vid in winners if vid in a_by and vid not in b_by)
    from_b_only = sum(1 for vid in winners if vid in b_by and vid not in a_by)

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=output.name + ".",
        dir=str(output.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for vid in sorted(winners.keys()):
                f.write(json.dumps(winners[vid], ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return {
        "total": total,
        "from_a_only": from_a_only,
        "from_b_only": from_b_only,
        "conflicts_resolved": conflicts,
    }


def _merge_cli(args: argparse.Namespace) -> int:
    stats = merge_registries(args.registry_a, args.registry_b, args.output)
    print(
        json.dumps(
            {
                "message": "Merge complete",
                **stats,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Global video registry utilities")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="Merge two registry JSONL files")
    m.add_argument("--registry-a", type=Path, required=True)
    m.add_argument("--registry-b", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "merge":
        raise SystemExit(_merge_cli(args))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
