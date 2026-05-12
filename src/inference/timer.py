"""
Step-level wall-clock timer for the inference pipeline.

Usage:
    timer = StepTimer()
    with timer.step("my_step"):
        do_something()
    print(timer.summary())
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


class StepTimer:
    """Accumulates wall-clock time per named pipeline step."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, float]] = []

    @contextmanager
    def step(self, name: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._steps.append((name, elapsed))

    def to_dict(self) -> dict[str, float]:
        """Return {step_name: seconds} preserving insertion order."""
        result: dict[str, float] = {}
        for name, elapsed in self._steps:
            result[name] = result.get(name, 0.0) + elapsed
        return result

    def total(self) -> float:
        return sum(e for _, e in self._steps)

    def summary(self) -> str:
        lines = ["Step timing (seconds):"]
        for name, elapsed in self.to_dict().items():
            lines.append(f"  {name}: {elapsed:.3f}s")
        lines.append(f"  TOTAL: {self.total():.3f}s")
        return "\n".join(lines)
