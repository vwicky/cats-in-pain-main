"""
Gradio UI for manual verification of YAMNet chunk scores (adapted from v2
``notebooks/inference_verify.ipynb`` Step 5).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_ordered_chunks(chunks: list[dict]) -> list[dict]:
    """Cat predictions first (highest P(cat)), then non-cat (highest P(cat) first)."""
    cats = [c for c in chunks if c.get("predicted_label") == "cat"]
    noncats = [c for c in chunks if c.get("predicted_label") != "cat"]
    cats.sort(key=lambda x: -float(x["p_cat"]))
    noncats.sort(key=lambda x: -float(x["p_cat"]))
    return cats + noncats


def load_verified_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ids.add(str(obj["chunk_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def append_verification_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stats_from_file(path: Path) -> tuple[int, int, int, int, float]:
    """verified_total, cat_labels, noncat_labels, skip_labels, accuracy (excl. skip)."""
    if not path.exists():
        return 0, 0, 0, 0, 0.0
    cat = noncat = skip = 0
    correct = 0
    compared = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            hl = obj.get("human_label")
            if hl == "cat":
                cat += 1
            elif hl == "non-cat":
                noncat += 1
            else:
                skip += 1
            if hl in ("cat", "non-cat"):
                compared += 1
                if obj.get("predicted_label") == hl:
                    correct += 1
    verified_total = cat + noncat + skip
    acc = (correct / compared) if compared else 0.0
    return verified_total, cat, noncat, skip, acc


def p_cat_html(p: float, thr: float) -> str:
    pct = int(round(min(100, max(0, p * 100))))
    if p >= thr:
        color = "#2e7d32"
    elif p >= thr * 0.85:
        color = "#ef6c00"
    else:
        color = "#c62828"
    pass_txt = "PASS ✅" if p >= thr else "below threshold"
    return (
        f'<div style="width:100%; background:#eee; border-radius:4px; height:24px; position:relative;">'
        f'<div style="width:{pct}%; background:{color}; height:24px; border-radius:4px;"></div>'
        f'<div style="position:absolute; left:8px; top:2px; font-family:monospace; font-size:13px;">'
        f"P(cat): {p:.3f}  [{pass_txt}]"
        f"</div></div>"
    )


def launch_verification_ui(
    ordered_chunks: list[dict],
    *,
    threshold: float,
    results_dir: Path,
    share: bool = False,
    server_port: int | None = None,
) -> None:
    """
    Open a Gradio Blocks UI to verify chunks (audio + metadata + verdict buttons).

    Appends to ``results_dir / "verification_results.jsonl"`` on each verdict.
    """
    try:
        import gradio as gr
    except ImportError:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "gradio", "-q"], check=True)
        import gradio as gr

    results_dir = Path(results_dir)
    verification_path = results_dir / "verification_results.jsonl"
    wav_temp_dir = Path(tempfile.mkdtemp(prefix="gradio_wav_yamnet_"))
    atexit.register(lambda: shutil.rmtree(wav_temp_dir, ignore_errors=True))

    verified_ids: set[str] = load_verified_ids(verification_path)
    _last_wav: Path | None = None
    current_k: int = 0

    def cleanup_wav() -> None:
        nonlocal _last_wav
        if _last_wav is not None and _last_wav.exists():
            try:
                _last_wav.unlink()
            except OSError:
                pass
        _last_wav = None

    def pending_indices() -> list[int]:
        return [i for i in range(len(ordered_chunks)) if ordered_chunks[i]["chunk_id"] not in verified_ids]

    def render_ui():
        nonlocal current_k, _last_wav
        cleanup_wav()
        total = len(ordered_chunks)
        n_cat_pred = sum(1 for c in ordered_chunks if c.get("predicted_label") == "cat")
        verified_total, vc, vnc, vs, acc = stats_from_file(verification_path)
        pend = pending_indices()

        header = (
            "## YAMNet v3 — chunk verification\n\n"
            f"**Progress:** {verified_total} / {total} verified "
            f"({n_cat_pred} cat-predicted first)"
        )

        if total == 0:
            return header + "\n\n*No chunks to verify.*", None, "", "<div></div>", ""

        if not pend:
            completion = (
                f"### All chunks verified\n\n"
                f"Verified: **{verified_total}** | Model agreement (non-skip): **{acc * 100:.1f}%**"
            )
            return (
                header + "\n\n" + completion,
                None,
                "",
                "<div></div>",
                f"Session — Verified: {verified_total} | Cat: {vc} | Not cat: {vnc} | Skip: {vs} | "
                f"Accuracy: {acc * 100:.1f}%",
            )

        current_k = max(0, min(current_k, len(pend) - 1))
        idx_ord = pend[current_k]
        ch = ordered_chunks[idx_ord]
        seg = ch["audio_segment"]
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=wav_temp_dir)
        seg.export(tf.name, format="wav")
        tf.close()
        _last_wav = Path(tf.name)

        pos = idx_ord + 1
        vid_name = ch.get("source_video", f'{ch.get("video_stem", "")}.mp4')
        meta = f"""**Video:** `{vid_name}`

**Chunk:** {pos:03d} of {total}

**Time:** {ch["start_sec"]:.1f}s – {ch["end_sec"]:.1f}s

**Predicted:** {ch["predicted_label"]}
"""
        bar = p_cat_html(float(ch["p_cat"]), threshold)
        stats_md = (
            f"**Session stats:** Verified: {verified_total} | Cat: {vc} | Not cat: {vnc} | Skip: {vs} | "
            f"Model accuracy: **{acc * 100:.1f}%** (excl. skip)"
        )
        return header, str(_last_wav), meta, bar, stats_md

    def apply_verdict(human_label: str):
        nonlocal verified_ids, current_k
        pend = pending_indices()
        if not pend:
            return render_ui()
        if current_k >= len(pend):
            current_k = len(pend) - 1
        idx_ord = pend[current_k]
        ch = ordered_chunks[idx_ord]
        record = {
            "chunk_id": ch["chunk_id"],
            "video_stem": ch["video_stem"],
            "chunk_index": int(ch["chunk_index"]),
            "start_sec": float(ch["start_sec"]),
            "end_sec": float(ch["end_sec"]),
            "p_cat": float(ch["p_cat"]),
            "p_noncat": float(ch.get("p_noncat", 1.0 - float(ch["p_cat"]))),
            "predicted_label": ch["predicted_label"],
            "passed_threshold": bool(ch["passed_threshold"]),
            "human_label": human_label,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        append_verification_line(verification_path, record)
        verified_ids.add(str(ch["chunk_id"]))
        current_k = 0
        return render_ui()

    def on_prev():
        nonlocal current_k
        pend = pending_indices()
        if not pend:
            return render_ui()
        current_k = max(0, current_k - 1)
        return render_ui()

    def on_next():
        nonlocal current_k
        pend = pending_indices()
        if not pend:
            return render_ui()
        current_k = min(len(pend) - 1, current_k + 1)
        return render_ui()

    with gr.Blocks(title="YAMNet v3 verification") as demo:
        title = gr.Markdown()
        audio = gr.Audio(label="Chunk audio", type="filepath", autoplay=True)
        details = gr.Markdown()
        bar = gr.HTML()
        stats = gr.Markdown()
        with gr.Row():
            b_cat = gr.Button("Cat")
            b_nc = gr.Button("Not a cat")
            b_sk = gr.Button("Skip")
        with gr.Row():
            b_prev = gr.Button("Previous")
            b_next = gr.Button("Next")

        outs: list[Any] = [title, audio, details, bar, stats]
        demo.load(render_ui, inputs=None, outputs=outs)
        b_cat.click(lambda: apply_verdict("cat"), outputs=outs)
        b_nc.click(lambda: apply_verdict("non-cat"), outputs=outs)
        b_sk.click(lambda: apply_verdict("skip"), outputs=outs)
        b_prev.click(on_prev, outputs=outs)
        b_next.click(on_next, outputs=outs)

    n_resume = len(load_verified_ids(verification_path))
    n_cat_pred = sum(1 for c in ordered_chunks if c.get("predicted_label") == "cat")
    print("Launching Gradio (YAMNet v3)...")
    print(f"Chunks to verify: {len(ordered_chunks)} ({n_cat_pred} cat-predicted first)")
    print(f"Resume: {n_resume} already verified chunks loaded")
    print(f"Results: {results_dir.resolve()}/")
    demo.launch(share=share, server_port=server_port)
