#!/usr/bin/env python3
"""
Extract audio from ``different/*.mp4``, chunk into 6 s (configurable), run YAMNet P(cat),
save results, and optionally launch the Gradio verification UI (v2 notebook parity).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from src.gradio_verify_yamnet import build_ordered_chunks, launch_verification_ui
from src.video_chunk_pipeline import (
    chunk_audio,
    extract_audio_wav_16k_mono,
    pydub_segment_to_waveform_16k_mono,
)
from src.yamnet_runner import YamNetRunner

logger = logging.getLogger(__name__)


def _json_safe_chunk(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in ("audio_segment",):
            continue
        if isinstance(v, float):
            out[k] = float(v)
        elif isinstance(v, int) and not isinstance(v, bool):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="YAMNet on different/ MP4 chunks + optional Gradio")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=_PROJECT / "different",
        help="Directory containing .mp4 files (default: audio_preclassification_v3/different)",
    )
    parser.add_argument("--chunk-sec", type=float, default=6.0)
    parser.add_argument("--overlap-sec", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write inference_results.jsonl (default: video_inference under project)",
    )
    parser.add_argument("--no-gradio", action="store_true", help="Only run inference; do not open UI")
    parser.add_argument("--share", action="store_true", help="Gradio share=True")
    parser.add_argument("--port", type=int, default=None, help="Gradio server port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    video_dir = args.video_dir.resolve()
    if not video_dir.is_dir():
        logger.error("Video directory not found: %s", video_dir)
        sys.exit(1)

    mp4_files = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    if not mp4_files:
        logger.error("No .mp4 files in %s", video_dir)
        sys.exit(1)

    results_dir = (args.results_dir or (_PROJECT / "video_inference")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    inference_jsonl = results_dir / "inference_results.jsonl"

    temp_root = Path(tempfile.mkdtemp(prefix="yamnet_video_"))
    try:
        all_chunks: list[dict] = []
        for vid_path in tqdm(mp4_files, desc="extract+chunk"):
            sub = temp_root / vid_path.stem
            wav = extract_audio_wav_16k_mono(vid_path, sub)
            if wav is None:
                continue
            chunks = chunk_audio(
                wav,
                args.chunk_sec,
                args.overlap_sec,
                source_video_name=vid_path.name,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.error("No chunks produced.")
            sys.exit(1)

        logger.info("Total chunks: %d — running YAMNet...", len(all_chunks))
        runner = YamNetRunner()

        scored: list[dict] = []
        for ch in tqdm(all_chunks, desc="yamnet"):
            try:
                wf = pydub_segment_to_waveform_16k_mono(ch["audio_segment"])
                p_cat = runner.predict_p_cat_from_waveform(wf, 16000)
                p_noncat = 1.0 - float(p_cat)
                passed = p_cat >= args.threshold
                scored.append(
                    {
                        **ch,
                        "p_cat": float(p_cat),
                        "p_noncat": float(p_noncat),
                        "passed_threshold": passed,
                        "predicted_label": "cat" if passed else "non-cat",
                        "features_ok": True,
                    }
                )
            except Exception:
                logger.exception("Chunk failed: %s", ch.get("chunk_id"))
                scored.append(
                    {
                        **ch,
                        "p_cat": 0.0,
                        "p_noncat": 1.0,
                        "passed_threshold": False,
                        "predicted_label": "non-cat",
                        "features_ok": False,
                    }
                )

        with open(inference_jsonl, "w", encoding="utf-8") as f:
            for row in scored:
                f.write(json.dumps(_json_safe_chunk(row), ensure_ascii=False) + "\n")
        logger.info("Wrote %s", inference_jsonl)

        summary_lines = [
            f"Total chunks: {len(scored)}",
            f"Threshold: {args.threshold}",
            f"Passed (cat): {sum(1 for c in scored if c['predicted_label'] == 'cat')}",
            f"Feature errors: {sum(1 for c in scored if not c['features_ok'])}",
        ]
        (results_dir / "inference_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        ordered = build_ordered_chunks(scored)
        if not args.no_gradio:
            launch_verification_ui(
                ordered,
                threshold=args.threshold,
                results_dir=results_dir,
                share=args.share,
                server_port=args.port,
            )
        else:
            logger.info("Skipping Gradio (--no-gradio). Results in %s", results_dir)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
