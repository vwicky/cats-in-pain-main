from __future__ import annotations

from typing import Any


def extract_p_pain_from_class_probs(
    class_probs: dict[Any, Any] | None,
    positive_label: int = 1,
) -> float | None:
    """
    Robustly read P(pain) from meta_class_probs when keys may be
    "0"/"1", 0/1, "1.0", etc.
    """
    if not class_probs:
        return None
    for k, v in class_probs.items():
        try:
            if float(k) == float(positive_label):
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


def enrich_meta_result_inplace(meta: dict[str, Any] | None, positive_label: int = 1) -> None:
    if not meta:
        return
    probs = meta.get("meta_class_probs")
    if not isinstance(probs, dict):
        return
    orig = meta.get("p_pain")
    computed = extract_p_pain_from_class_probs(probs, positive_label=positive_label)
    if computed is not None:
        meta["p_pain_original"] = orig
        meta["p_pain"] = computed
        meta["positive_label"] = positive_label


def enrich_pipeline_result_inplace(data: dict[str, Any], positive_label: int = 1) -> None:
    """Add normalized p_pain in video meta_result for single run or nested window results."""
    if not data:
        return
    mode = data.get("mode")
    if mode == "split_sliding_windows":
        for w in data.get("windows") or []:
            res = w.get("result")
            if isinstance(res, dict):
                _enrich_single_run_dict(res, positive_label)
        return
    _enrich_single_run_dict(data, positive_label)


def _enrich_single_run_dict(res: dict[str, Any], positive_label: int) -> None:
    meta = res.get("meta_result")
    if isinstance(meta, dict):
        enrich_meta_result_inplace(meta, positive_label=positive_label)
