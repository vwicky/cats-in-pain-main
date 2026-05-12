"""Append pipeline results to local JSONL files (metadata + pipeline log)."""

from __future__ import annotations

import json
from typing import Any


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def persist_result(
    video_id: str,
    result: dict[str, Any],
    metadata_file: str,
    pipeline_log_file: str,
) -> None:
    """Append success metadata and full result line to local JSONL files."""
    if result.get("status") == "success" and "record" in result:
        _append_jsonl(metadata_file, result["record"])
    _append_jsonl(pipeline_log_file, result)
