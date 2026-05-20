from __future__ import annotations

VALID_MULTICAT_SUMMARY_STRATEGIES = frozenset(
    {"max", "mean", "majority_above_threshold", "coverage_weighted_mean"}
)


def parse_form_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")
