"""
GPT usage / cost table for thesis (description jsonl, cat_id report, channel config).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[misc, assignment]

PAPER_FIGURES_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PAPER_FIGURES_DIR.parent / "config.yaml"
REPO_ROOT = PAPER_FIGURES_DIR.parents[3]


def _minimal_config_fallback() -> dict[str, Any]:
    return {
        "gpt_description": {
            "descriptions_jsonl": "src/dataset_construction/reports/gpt_descriptions.jsonl",
            "cost_per_1k_input_tokens": 0.0025,
            "cost_per_1k_output_tokens": 0.01,
        },
        "cat_id": {
            "report_txt": "src/dataset_construction/reports/cat_id_report.txt",
        },
        "paper_figures": {
            "channel_resolution_gpt": {
                "estimated": True,
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_dataset_config() -> dict[str, Any]:
    if yaml is None:
        return _minimal_config_fallback()
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else _minimal_config_fallback()


def aggregate_gpt_descriptions_jsonl(path: Path) -> dict[str, float | int]:
    n = 0
    total_in = 0.0
    total_out = 0.0
    if not path.is_file():
        return {"api_calls": 0, "input_tokens": 0.0, "output_tokens": 0.0}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            total_in += float(r.get("input_tokens") or 0)
            total_out += float(r.get("output_tokens") or 0)
    return {
        "api_calls": n,
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


def parse_cat_id_gpt_verify(path: Path) -> dict[str, Any]:
    """Pairs evaluated and total cost from cat_id_report.txt."""
    out: dict[str, Any] = {
        "api_calls": 0,
        "cost_usd": 0.0,
    }
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8")
    m_pairs = re.search(r"Pairs evaluated:\s*(\d+)", text)
    if m_pairs:
        out["api_calls"] = int(m_pairs.group(1))
    m_cost = re.search(r"Total cost:\s*\$([0-9]+(?:\.[0-9]+)?)", text)
    if m_cost:
        out["cost_usd"] = float(m_cost.group(1))
    return out


def channel_resolution_row(cfg: dict[str, Any]) -> dict[str, Any]:
    block = (cfg.get("paper_figures") or {}).get("channel_resolution_gpt") or {}
    return {
        "estimated": bool(block.get("estimated", True)),
        "api_calls": int(block.get("api_calls") or 0),
        "input_tokens": int(block.get("input_tokens") or 0),
        "output_tokens": int(block.get("output_tokens") or 0),
    }


def usd_from_tokens(
    input_tok: float,
    output_tok: float,
    per_in: float,
    per_out: float,
) -> float:
    return (input_tok / 1000.0) * per_in + (output_tok / 1000.0) * per_out


def build_cost_table_txt(cfg: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """
    Returns (body lines incl. footer footnotes list started empty - footnotes appended in lines).
    """
    if cfg is None:
        cfg = load_dataset_config()
    gd = cfg["gpt_description"]
    cid = cfg["cat_id"]

    desc_path = _resolve(gd["descriptions_jsonl"])
    cat_rep = _resolve(cid["report_txt"])

    agg = aggregate_gpt_descriptions_jsonl(desc_path)
    per_in = float(gd.get("cost_per_1k_input_tokens", 0.0025))
    per_out = float(gd.get("cost_per_1k_output_tokens", 0.01))
    desc_usd = usd_from_tokens(
        float(agg["input_tokens"]),
        float(agg["output_tokens"]),
        per_in,
        per_out,
    )

    cat_info = parse_cat_id_gpt_verify(cat_rep)
    cat_usd = float(cat_info["cost_usd"])

    ch = channel_resolution_row(cfg)
    est = ch["estimated"]
    ch_calls = ch["api_calls"]
    ch_in = ch["input_tokens"]
    ch_out = ch["output_tokens"]

    footer_notes: list[str] = []

    def fmt_num(v: Any) -> str:
        if v == "N/A (not yet logged)":
            return v
        if isinstance(v, float):
            if v == int(v):
                return str(int(v))
            return f"{v:.6g}"
        return str(v)

    def fmt_usd(v: Any) -> str:
        if v == "N/A (not yet logged)":
            return v
        if isinstance(v, str):
            return v
        return f"${v:.2f}"

    if est and ch_calls == 0 and ch_in == 0 and ch_out == 0:
        ch_calls_s = "N/A (not yet logged)"
        ch_in_s = "N/A (not yet logged)"
        ch_out_s = "N/A (not yet logged)"
        ch_usd_s = "N/A (not yet logged)"
        footer_notes.append(
            "* Channel-resolution GPT usage not populated in config (paper_figures.channel_resolution_gpt); "
            "fill api_calls/tokens or set estimated: false after reconciling."
        )
    else:
        ch_calls_s = str(ch_calls)
        ch_in_s = str(ch_in)
        ch_out_s = str(ch_out)
        ch_usd_usd = usd_from_tokens(float(ch_in), float(ch_out), per_in, per_out)
        if est:
            ch_usd_s = f"${ch_usd_usd:.2f} *"
            footer_notes.append(
                "* Estimated channel-resolution GPT cost from config tokens (paper_figures.channel_resolution_gpt)."
            )
        else:
            ch_usd_s = f"${ch_usd_usd:.2f}"

    lines = [
        "GPT pipeline usage and estimated cost",
        "=" * 78,
        f"Pricing (gpt-4o): ${per_in}/1k input, ${per_out}/1k output tokens",
        "",
        f"{'Stage':<32} {'API calls':>12} {'Input tok':>14} {'Output tok':>14} {'USD':>12}",
        "-" * 78,
        f"{'GPT description':<32} {agg['api_calls']:>12} {fmt_num(agg['input_tokens']):>14} {fmt_num(agg['output_tokens']):>14} {fmt_usd(desc_usd):>12}",
        f"{'GPT cat-ID verification':<32} {cat_info['api_calls']:>12} {'N/A':>14} {'N/A':>14} {fmt_usd(cat_usd):>12}",
        f"{'Channel resolution (GPT)':<32} {ch_calls_s:>12} {ch_in_s:>14} {ch_out_s:>14} {ch_usd_s:>12}",
        "",
        f"GPT description JSONL: {desc_path}",
        f"Cat-ID report:         {cat_rep}",
    ]

    lines.append("")
    for note in footer_notes:
        lines.append(note)

    return lines, footer_notes


def write_cost_table(path: Path, cfg: dict[str, Any] | None = None) -> Path:
    lines, _ = build_cost_table_txt(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
