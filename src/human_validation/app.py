#!/usr/bin/env python3
"""
Human validation UI for paired .mp3 / .mp4 snippet files.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

LABEL_FOLDERS = ("trash", "non-cat", "cat")
MODE_AUDIO = "Audio only"
MODE_VIDEO = "Video + audio"

MUX_SUBDIR = "human_validation_mux"


def _unlink_mux(state: dict[str, Any]) -> None:
    p = state.get("mux_temp")
    if not p:
        return
    try:
        Path(p).unlink(missing_ok=True)
    except OSError:
        pass
    state["mux_temp"] = None


def mux_mp4_with_mp3(mp4: Path, mp3: Path, out: Path) -> bool:
    """One MP4 with video from mp4 + audio from mp3 (requires ffmpeg on PATH)."""
    if shutil.which("ffmpeg") is None:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-i",
                str(mp3),
                "-c:v",
                "copy",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "aac",
                "-shortest",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return out.is_file() and out.stat().st_size > 0


def discover_pairs(data_dir: Path) -> tuple[list[str], list[str]]:
    """Return (eligible stems, stems skipped because .mp4 was missing)."""
    eligible: list[str] = []
    skipped: list[str] = []
    for mp3 in sorted(data_dir.glob("*.mp3")):
        stem = mp3.stem
        mp4 = data_dir / f"{stem}.mp4"
        if mp4.is_file():
            eligible.append(stem)
        else:
            skipped.append(stem)
    return eligible, skipped


def unique_dest_pair(dest_dir: Path, stem: str) -> tuple[Path, Path]:
    """Return destination paths for .mp3 and .mp4 that do not collide."""
    n = 0
    while True:
        suffix = "" if n == 0 else f"_{n}"
        base = f"{stem}{suffix}"
        mp3 = dest_dir / f"{base}.mp3"
        mp4 = dest_dir / f"{base}.mp4"
        if not mp3.exists() and not mp4.exists():
            return mp3, mp4
        n += 1


def append_jsonl(log_file: Path, record: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def build_state(
    data_dir: Path,
    audio_root: Path,
    video_root: Path,
    seed: int | None,
) -> dict[str, Any]:
    eligible, skipped = discover_pairs(data_dir)
    for s in skipped:
        print(f"Skipping (no paired .mp4): {s}.mp3", file=sys.stderr)
    stems = eligible.copy()
    if seed is None:
        random.shuffle(stems)
    else:
        random.Random(seed).shuffle(stems)
    return {
        "stems": stems,
        "idx": 0,
        "data_dir": str(data_dir.resolve()),
        "audio_root": str(audio_root.resolve()),
        "video_root": str(video_root.resolve()),
        "mux_temp": None,
    }


def output_root_for_mode(mode_ui: str, state: dict[str, Any]) -> Path:
    if mode_ui == MODE_AUDIO:
        return Path(state["audio_root"])
    return Path(state["video_root"])


def mode_key(mode_ui: str) -> str:
    return "audio" if mode_ui == MODE_AUDIO else "video_audio"


def log_file_for_state(state: dict[str, Any], mode_ui: str) -> Path:
    root = output_root_for_mode(mode_ui, state)
    return root / "human_validation_log.jsonl"


def render_sample(
    state: dict[str, Any],
    mode_ui: str,
) -> tuple[dict, dict, str, dict, dict, dict]:
    """Return gr.update() dicts for audio, video, status, and three label buttons."""
    stems: list[str] = state["stems"]
    idx: int = state["idx"]
    data_dir = Path(state["data_dir"])
    total = len(stems)

    if total == 0:
        _unlink_mux(state)
        return (
            gr.update(value=None, label="Audio"),
            gr.update(value=None, label="Video", visible=False),
            "No paired samples found (need .mp3 with same-stem .mp4).",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    if idx >= total:
        _unlink_mux(state)
        return (
            gr.update(value=None, label="Audio"),
            gr.update(value=None, label="Video", visible=False),
            f"Done. Labeled all {total} sample(s).",
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    stem = stems[idx]
    mp3_path = data_dir / f"{stem}.mp3"
    mp4_path = data_dir / f"{stem}.mp4"
    remaining = total - idx
    status = (
        f"Sample {idx + 1} of {total} ({remaining} remaining)\n"
        f"Stem: {stem}"
    )

    if mode_ui == MODE_AUDIO:
        _unlink_mux(state)
        audio_upd = gr.update(
            value=str(mp3_path),
            label=f"Audio — {stem}.mp3",
            visible=True,
            autoplay=True,
        )
        video_upd = gr.update(value=None, label="Video (hidden in audio-only mode)", visible=False)
        return (
            audio_upd,
            video_upd,
            status,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    # Video + audio: single muxed MP4 (video + MP3 soundtrack) for simultaneous playback.
    _unlink_mux(state)
    mux_dir = Path(tempfile.gettempdir()) / MUX_SUBDIR
    mux_path = mux_dir / f"{stem}_{uuid.uuid4().hex}.mp4"
    if mux_mp4_with_mp3(mp4_path, mp3_path, mux_path):
        state["mux_temp"] = str(mux_path)
        extra = (
            "\nPlayback: video + MP3 combined (ffmpeg). Files still move from the original folder."
        )
        return (
            gr.update(value=None, label="Audio (muxed into video below)", visible=False),
            gr.update(
                value=str(mux_path),
                label=f"Video + audio (synced) — {stem}.mp4 + .mp3",
                visible=True,
                autoplay=True,
            ),
            status + extra,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    warn = (
        "\nffmpeg missing or mux failed — showing separate players; "
        "install ffmpeg for a single synced clip."
    )
    state["mux_temp"] = None
    return (
        gr.update(
            value=str(mp3_path),
            label=f"Audio — {stem}.mp3",
            visible=True,
            autoplay=True,
        ),
        gr.update(
            value=str(mp4_path),
            label=f"Video — {stem}.mp4",
            visible=True,
            autoplay=True,
        ),
        status + warn,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def apply_label(
    state: dict[str, Any],
    mode_ui: str,
    label: str,
) -> tuple[dict[str, Any], dict, dict, str, dict, dict, dict]:
    stems: list[str] = state["stems"]
    idx: int = state["idx"]
    data_dir = Path(state["data_dir"])

    new_state = {**state, "stems": list(stems), "idx": idx}

    if not stems or idx >= len(stems):
        a, v, s, b1, b2, b3 = render_sample(new_state, mode_ui)
        return new_state, a, v, s, b1, b2, b3

    stem = stems[idx]
    src_mp3 = data_dir / f"{stem}.mp3"
    src_mp4 = data_dir / f"{stem}.mp4"
    if not src_mp3.is_file() or not src_mp4.is_file():
        new_state["idx"] = idx + 1
        a, v, s, b1, b2, b3 = render_sample(new_state, mode_ui)
        return new_state, a, v, s, b1, b2, b3

    source_mp3 = str(src_mp3.resolve())
    source_mp4 = str(src_mp4.resolve())

    out_root = output_root_for_mode(mode_ui, new_state)
    dest_sub = out_root / label
    dest_sub.mkdir(parents=True, exist_ok=True)
    mp3_dest, mp4_dest = unique_dest_pair(dest_sub, stem)

    shutil.move(str(src_mp3), str(mp3_dest))
    shutil.move(str(src_mp4), str(mp4_dest))

    dest_dir_resolved = mp3_dest.parent.resolve()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode_key(mode_ui),
        "stem": stem,
        "source_mp3": source_mp3,
        "source_mp4": source_mp4,
        "label": label,
        "dest_dir": str(dest_dir_resolved),
        "moved_mp3": str(mp3_dest.resolve()),
        "moved_mp4": str(mp4_dest.resolve()),
    }
    append_jsonl(log_file_for_state(new_state, mode_ui), record)

    new_state["idx"] = idx + 1
    a, v, s, b1, b2, b3 = render_sample(new_state, mode_ui)
    return new_state, a, v, s, b1, b2, b3


def on_mode_change(state: dict[str, Any], mode_ui: str):
    """Refresh display when toggling audio vs video (same queue position)."""
    a, v, s, b1, b2, b3 = render_sample(state, mode_ui)
    return a, v, s, b1, b2, b3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Human validation for paired .mp3/.mp4 snippet files.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing filename.mp3 and filename.mp4 pairs.",
    )
    p.add_argument(
        "--audio-output-root",
        type=Path,
        default=Path("audio_human_validation"),
        help="Root for audio-only validation moves (default: ./audio_human_validation).",
    )
    p.add_argument(
        "--video-output-root",
        type=Path,
        default=Path("video_audio_human_validation"),
        help="Root for video+audio validation moves (default: ./video_audio_human_validation).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling order (omit for nondeterministic shuffle).",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (optional).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        print(f"Error: --data-dir is not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    audio_root = args.audio_output_root.expanduser().resolve()
    video_root = args.video_output_root.expanduser().resolve()
    for root in (audio_root, video_root):
        for name in LABEL_FOLDERS:
            (root / name).mkdir(parents=True, exist_ok=True)

    initial_state = build_state(data_dir, audio_root, video_root, args.seed)

    with gr.Blocks(title="Human validation") as demo:
        gr.Markdown(
            "# Human validation — cat audio / video snippets\n\n"
            "- **Audio only** moves files under the audio output root (default `./audio_human_validation/`), "
            "with separate `trash/`, `non-cat/`, and `cat/` folders.\n"
            "- **Video + audio** uses a **different** output root (default `./video_audio_human_validation/`) "
            "with its own `trash/`, `non-cat/`, and `cat/` folders.\n"
            "- In **Video + audio** mode, the app muxes **picture from `.mp4` + sound from `.mp3`** into one clip "
            "with **ffmpeg** so video and audio play together. Install ffmpeg if you see a fallback warning."
        )
        state = gr.State(initial_state)
        mode = gr.Radio(
            choices=[MODE_AUDIO, MODE_VIDEO],
            value=MODE_AUDIO,
            label="Validation mode",
        )

        status = gr.Textbox(label="Status", lines=3, interactive=False)
        with gr.Row():
            audio = gr.Audio(
                label="Audio",
                type="filepath",
                sources=["upload"],
                autoplay=True,
                interactive=False,
            )
            video = gr.Video(
                label="Video",
                format="mp4",
                sources=["upload"],
                autoplay=True,
                interactive=False,
            )

        with gr.Row():
            btn_trash = gr.Button("Trash", variant="stop")
            btn_noncat = gr.Button("Non-cat")
            btn_cat = gr.Button("Cat", variant="primary")

        outputs = [audio, video, status, btn_trash, btn_noncat, btn_cat]

        def trash_fn(s, m):
            return apply_label(s, m, "trash")

        def noncat_fn(s, m):
            return apply_label(s, m, "non-cat")

        def cat_fn(s, m):
            return apply_label(s, m, "cat")

        btn_trash.click(
            trash_fn,
            inputs=[state, mode],
            outputs=[state, *outputs],
        )
        btn_noncat.click(
            noncat_fn,
            inputs=[state, mode],
            outputs=[state, *outputs],
        )
        btn_cat.click(
            cat_fn,
            inputs=[state, mode],
            outputs=[state, *outputs],
        )

        mode.change(
            on_mode_change,
            inputs=[state, mode],
            outputs=outputs,
        )

        demo.load(
            lambda s, m: render_sample(s, m),
            inputs=[state, mode],
            outputs=outputs,
        )

    # Gradio only serves file paths under cwd, temp, or allowed_paths (absolute).
    mux_root = Path(tempfile.gettempdir()) / MUX_SUBDIR
    mux_root.mkdir(parents=True, exist_ok=True)
    allowed_paths = sorted(
        {str(p) for p in (data_dir, audio_root, video_root, mux_root)}
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
