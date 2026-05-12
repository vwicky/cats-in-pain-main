"""Plain-text table writers for thesis deliverables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


def write_txt_table(
    path: Path,
    lines: list[str],
    *,
    encoding: str = "utf-8",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding=encoding)


def two_column_table(
    rows: Sequence[tuple[Any, Any]],
    col_a: str = "reason",
    col_b: str = "count",
    width_a: int = 48,
    width_b: int = 12,
) -> list[str]:
    header = f"{col_a:<{width_a}} {col_b:>{width_b}}"
    sep = "-" * (width_a + width_b + 3)
    out = [header, sep]
    for a, b in rows:
        sa = str(a)[:width_a]
        out.append(f"{sa:<{width_a}} {str(b):>{width_b}}")
    return out
