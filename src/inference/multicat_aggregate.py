"""Multicat headline and split-level aggregation (pipeline + website normalize)."""

from __future__ import annotations

from typing import Any

MULTICAT_STRATEGIES = frozenset(
    {"max", "mean", "majority_above_threshold", "coverage_weighted_mean"}
)
DEFAULT_MULTICAT_SUMMARY_STRATEGY = "coverage_weighted_mean"
DEFAULT_MULTICAT_DECISION_THRESHOLD = 0.5


def _cat_p_pains(cats: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return list of (p_pain, detection_rate_sampled) for each cat row."""
    out: list[tuple[float, float]] = []
    for c in cats:
        meta = c.get("meta_result") or {}
        v = meta.get("p_pain")
        if not isinstance(v, (int, float)):
            continue
        r = c.get("detection_rate_sampled")
        rate = float(r) if isinstance(r, (int, float)) else 0.0
        out.append((float(v), max(0.0, min(1.0, rate))))
    return out


def window_headline_from_cats(
    cats: list[dict[str, Any]],
    *,
    strategy: str,
    pain_threshold: float,
) -> dict[str, Any]:
    """
    Level-1 aggregate for one window / clip from enriched cats[].
    """
    strat = strategy if strategy in MULTICAT_STRATEGIES else DEFAULT_MULTICAT_SUMMARY_STRATEGY
    pairs = _cat_p_pains(cats)
    n = len(pairs)
    if n == 0:
        return {
            "headline_p_pain": None,
            "decision": None,
            "multicat_prevalence_fraction": None,
            "multicat_cats_above_threshold": None,
            "multicat_cats_total": 0,
            "multicat_p_pain_max": None,
            "multicat_p_pain_mean": None,
            "multicat_coverage_weighted_mean": None,
        }

    p_list = [p for p, _ in pairs]
    weights = [w for _, w in pairs]
    th = float(pain_threshold)
    above = sum(1 for p in p_list if p >= th)
    frac = above / n
    p_max = max(p_list)
    p_mean = sum(p_list) / n
    w_sum = sum(weights) or 1e-9
    cw_mean = sum(p_list[i] * weights[i] for i in range(n)) / w_sum

    headline: float | None
    decision: str | None

    if strat == "max":
        headline = p_max
        decision = "pain" if headline >= th else "non_pain"
    elif strat == "mean":
        headline = p_mean
        decision = "pain" if headline >= th else "non_pain"
    elif strat == "coverage_weighted_mean":
        headline = cw_mean
        decision = "pain" if headline >= th else "non_pain"
    else:  # majority_above_threshold
        headline = frac
        decision = "pain" if frac >= 0.5 else "non_pain"

    return {
        "headline_p_pain": headline,
        "decision": decision,
        "multicat_prevalence_fraction": frac,
        "multicat_cats_above_threshold": above,
        "multicat_cats_total": n,
        "multicat_p_pain_max": p_max,
        "multicat_p_pain_mean": p_mean,
        "multicat_coverage_weighted_mean": cw_mean,
    }


def clip_level_from_window_results(
    window_results: list[dict[str, Any]],
    *,
    strategy: str,
    pain_threshold: float,
) -> dict[str, Any]:
    """
    Level-2 aggregation across sliding windows (each item is the inner ``result`` dict).

    Contributing windows: those with non-empty ``cats[]`` only.
    For ``majority_above_threshold``, pools all cat rows across contributing windows.
    """
    strat = strategy if strategy in MULTICAT_STRATEGIES else DEFAULT_MULTICAT_SUMMARY_STRATEGY
    th = float(pain_threshold)

    pooled_cats: list[dict[str, Any]] = []
    l1_headlines: list[float] = []
    l1_weights: list[float] = []

    for res in window_results:
        if not isinstance(res, dict):
            continue
        cats = list(res.get("cats") or [])
        if not cats:
            continue
        pooled_cats.extend(cats)
        wh = window_headline_from_cats(cats, strategy=strat, pain_threshold=th)
        hp = wh.get("headline_p_pain")
        if not isinstance(hp, (int, float)):
            continue
        l1_headlines.append(float(hp))
        w_win = sum(
            float(c.get("detection_rate_sampled") or 0.0)
            for c in cats
            if isinstance(c.get("detection_rate_sampled"), (int, float))
        )
        if w_win <= 0:
            w_win = float(len(cats))
        l1_weights.append(w_win)

    if strat == "majority_above_threshold":
        pairs = _cat_p_pains(pooled_cats)
        n = len(pairs)
        if n == 0:
            return {
                "headline_p_pain": None,
                "decision": None,
                "multicat_prevalence_fraction": None,
                "multicat_cats_above_threshold": None,
                "multicat_cats_total": 0,
                "multicat_n_windows_contributing": 0,
            }
        above = sum(1 for p, _ in pairs if p >= th)
        frac = above / n
        return {
            "headline_p_pain": frac,
            "decision": "pain" if frac >= 0.5 else "non_pain",
            "multicat_prevalence_fraction": frac,
            "multicat_cats_above_threshold": above,
            "multicat_cats_total": n,
            "multicat_n_windows_contributing": len(l1_headlines),
        }

    if not l1_headlines:
        return {
            "headline_p_pain": None,
            "decision": None,
            "multicat_prevalence_fraction": None,
            "multicat_cats_above_threshold": None,
            "multicat_cats_total": 0,
            "multicat_n_windows_contributing": 0,
        }

    if strat == "max":
        h = max(l1_headlines)
    elif strat == "mean":
        h = sum(l1_headlines) / len(l1_headlines)
    else:  # coverage_weighted_mean @ L2: sum(w*s)/sum(w)
        w_tot = sum(l1_weights) or 1e-9
        h = sum(l1_headlines[i] * l1_weights[i] for i in range(len(l1_headlines))) / w_tot

    return {
        "headline_p_pain": h,
        "decision": "pain" if h >= th else "non_pain",
        "multicat_prevalence_fraction": None,
        "multicat_cats_above_threshold": None,
        "multicat_cats_total": len(pooled_cats),
        "multicat_n_windows_contributing": len(l1_headlines),
    }
