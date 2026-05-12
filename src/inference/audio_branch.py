"""
Audio branch of the inference pipeline.

Step 1 – AudioSep: separate cat sounds from the input audio.
Step 2 – CatEmotionModel: classify the separated audio into 10 emotion classes.

Both components are loaded lazily on first use so the heavy model weights are
only pulled into memory if the audio branch is actually taken.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from inference.import_hygiene import (
    clear_models_package_cache,
    prioritize_sys_path,
    strip_pose_models_tree_from_sys_path,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── AudioSep paths ────────────────────────────────────────────────────────────
AUDIOSEP_ROOT = REPO_ROOT / "audio" / "AudioSep"
AUDIOSEP_CONFIG = AUDIOSEP_ROOT / "config" / "audiosep_base.yaml"
AUDIOSEP_CKPT = AUDIOSEP_ROOT / "checkpoint" / "audiosep_base_4M_steps.ckpt"
# CLAP / HTSAT weights (path in clap_encoder.py is relative to AudioSep cwd)
AUDIOSEP_CLAP_HTSAT_CKPT = AUDIOSEP_ROOT / "checkpoint" / "music_speech_audioset_epoch_15_esc_89.98.pt"

# ── Audio emotion classifier paths ───────────────────────────────────────────
EMOTION_CLASSIFIER_ROOT = REPO_ROOT / "audio" / "audio-emotion-classifier"
EMOTION_CLASSIFIER_SRC = EMOTION_CLASSIFIER_ROOT / "src"

# Fine-tuned checkpoint search order (same as 04_audio_classification.py)
EMOTION_CKPT_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "best_model_final.pth",
    REPO_ROOT / "models" / "audio_emotions" / "best_model_final.pth",
]

# 10-class index → name (matches training label order)
IDX_TO_CLASS: dict[int, str] = {
    0: "Angry",
    1: "Defence",
    2: "Fighting",
    3: "Happy",
    4: "HuntingMind",
    5: "Mating",
    6: "MotherCall",
    7: "Paining",
    8: "Resting",
    9: "Warning",
}

# ── AudioSep bootstrap ────────────────────────────────────────────────────────


def _bootstrap_audiosep() -> None:
    """
    Prepend ``audio/AudioSep/`` to ``sys.path`` so ``utils``, ``models.*``, etc. import correctly.

    Remove ``video/pose-models`` from ``sys.path`` entirely: if it stays on the
    path, Python can bind ``import models`` to the ST-GCN package and ignore
    AudioSep's ``models/`` tree (regular package wins over namespace portions).

    Always clear the cached ``models`` module before AudioSep imports so a prior
    video window cannot poison this process.

    HTSAT weights in ``checkpoint/*.pt`` are still resolved relative to the
    process **working directory**; use ``_audiosep_working_directory()`` around
    AudioSep model load / separation calls.
    """
    if not AUDIOSEP_ROOT.is_dir():
        raise RuntimeError(f"AudioSep directory not found: {AUDIOSEP_ROOT}")
    strip_pose_models_tree_from_sys_path(REPO_ROOT)
    prioritize_sys_path(AUDIOSEP_ROOT)
    clear_models_package_cache()


@contextmanager
def _audiosep_working_directory():
    """
    ``CLAP_Encoder`` and open_clip resolve ``checkpoint/*.pt`` with
    ``os.path.exists`` relative to the process working directory. Run imports
    and separation with cwd = ``audio/AudioSep/`` so bundled weights are found.
    """
    prev = os.getcwd()
    os.chdir(AUDIOSEP_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


# ── Audio emotion classifier bootstrap ───────────────────────────────────────


def _bootstrap_emotion_classifier() -> None:
    """Register audio_classifier_utils as importable, mirroring 04_audio_classification.py."""
    utils_src = EMOTION_CLASSIFIER_SRC
    if not utils_src.is_dir():
        raise RuntimeError(f"Emotion classifier src not found: {utils_src}")
    if str(utils_src) not in sys.path:
        sys.path.insert(0, str(utils_src))


def _resolve_panns_backbone_path(audio_config: Any) -> None:
    """
    ``AudioConfig.checkpoint_path`` is relative to ``audio-emotion-classifier/src/``,
    but ``CatEmotionModel`` passes it straight to ``torch.load`` (cwd-independent).
    Resolve to an absolute path.
    """
    raw = str(audio_config.checkpoint_path).strip()
    p = Path(raw)
    if p.is_file():
        audio_config.checkpoint_path = str(p.resolve())
        return

    candidates = [
        EMOTION_CLASSIFIER_SRC / raw,
        REPO_ROOT / raw,
        Path.cwd() / raw,
    ]
    for c in candidates:
        if c.is_file():
            audio_config.checkpoint_path = str(c.resolve())
            return

    dest = EMOTION_CLASSIFIER_SRC / raw
    raise FileNotFoundError(
        "PANNs Cnn14 backbone weights not found (required when building CatEmotionModel).\n"
        f"Configured path: {raw!r}\n"
        "Tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\n\nDownload `Cnn14_16k_mAP=0.438.pth` (~359 MB) from the Zenodo link in "
        "`audio/audio-emotion-classifier/src/audio_classifier_utils/models/cnn14.py` "
        f"and save to:\n  {dest}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_audiosep(
    device: torch.device,
    *,
    config_yaml: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> Any:
    """Load and return the AudioSep model."""
    _bootstrap_audiosep()
    reqs_file = Path(__file__).resolve().parent / "requirements-audiosep.txt"
    _AUDIOSEP_MISSING_HINT = (
        f"Missing import for AudioSep / CLAP stack: {{err}}\n\n"
        f"Install vendored AudioSep extras:\n"
        f"  pip install -r src/inference/requirements-audiosep.txt\n"
        f"(see {reqs_file})"
    )

    cfg = str(config_yaml or AUDIOSEP_CONFIG)
    ckpt = str(checkpoint or AUDIOSEP_CKPT)

    if not Path(cfg).is_file():
        raise FileNotFoundError(f"AudioSep config not found: {cfg}")
    if not Path(ckpt).is_file():
        raise FileNotFoundError(
            f"AudioSep checkpoint not found: {ckpt}\n"
            "Download audiosep_base_4M_steps.ckpt from https://huggingface.co/spaces/Audio-AGI/AudioSep "
            "and place it at audio/AudioSep/checkpoint/"
        )
    if not AUDIOSEP_CLAP_HTSAT_CKPT.is_file():
        raise FileNotFoundError(
            f"CLAP HTSAT pretrained weights not found: {AUDIOSEP_CLAP_HTSAT_CKPT}\n"
            "This file is expected next to audiosep_base_4M_steps.ckpt under audio/AudioSep/checkpoint/ "
            "(see audio/AudioSep/models/clap_encoder.py default_pretrained_path)."
        )

    model: Any
    with _audiosep_working_directory():
        try:
            from pipeline import build_audiosep  # noqa: E402 (lives inside AudioSep/)
        except ModuleNotFoundError as e:
            raise ImportError(_AUDIOSEP_MISSING_HINT.format(err=e)) from e

        model = build_audiosep(cfg, ckpt, device)
    model.eval()
    return model


def run_audiosep(
    model: Any,
    audio_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    *,
    text_query: str = "cat sounds",
    use_chunk: bool = False,
) -> Path:
    """
    Run AudioSep on ``audio_path`` and write separated audio to ``output_path``.

    Returns the resolved output path.
    """
    _bootstrap_audiosep()
    with _audiosep_working_directory():
        from pipeline import separate_audio  # noqa: E402 (lives inside AudioSep/)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        separate_audio(model, str(audio_path), text_query, str(out), device=str(device), use_chunk=use_chunk)
        logger.info("AudioSep output: %s", out)
        return out


def load_emotion_model(
    device: torch.device,
    *,
    checkpoint: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Load CatEmotionModel + AudioConfig.

    Returns (model, audio_config).
    """
    _bootstrap_emotion_classifier()
    from audio_classifier_utils.audio_config import AudioConfig  # noqa: E402
    from audio_classifier_utils.models.cnn14 import CatEmotionModel  # noqa: E402

    audio_config = AudioConfig()

    if checkpoint is not None:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.is_file():
            alt = REPO_ROOT / checkpoint
            if alt.is_file():
                ckpt_path = alt
    else:
        ckpt_path = None
        for cand in EMOTION_CKPT_CANDIDATES:
            if cand.is_file():
                ckpt_path = cand
                break

    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(
            "CatEmotionModel checkpoint not found. Tried:\n"
            + "\n".join(f"  {c}" for c in EMOTION_CKPT_CANDIDATES)
            + "\nPlace best_model_final.pth at one of those locations."
        )

    _resolve_panns_backbone_path(audio_config)

    logger.info(
        "Loading emotion model from %s (PANNs backbone: %s)",
        ckpt_path,
        audio_config.checkpoint_path,
    )
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    model = CatEmotionModel(audio_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, audio_config


def run_emotion_inference(
    audio_path: str | Path,
    model: Any,
    audio_config: Any,
    device: torch.device,
) -> dict[str, Any]:
    """
    Classify a single audio file and return a dict with softmax probs + predicted class.
    """
    import librosa

    waveform, _ = librosa.load(str(audio_path), sr=audio_config.sample_rate)
    waveform_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    target_len = audio_config.target_length
    cur_len = waveform_t.shape[1]
    if cur_len > target_len:
        waveform_t = waveform_t[:, :target_len]
    elif cur_len < target_len:
        waveform_t = F.pad(waveform_t, (0, target_len - cur_len))

    x = waveform_t.squeeze(0).unsqueeze(0).to(device)  # [1, T]

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    result: dict[str, Any] = {
        "predicted_class_idx": pred_idx,
        "predicted_class": IDX_TO_CLASS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "softmax": {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(IDX_TO_CLASS))},
    }
    return result


def run_audio_branch(
    audio_path: str | Path,
    run_dir: Path,
    device: torch.device,
    timer: Any,
    *,
    audiosep_model: Any | None = None,
    emotion_model: Any | None = None,
    emotion_config: Any | None = None,
    emotion_ckpt: str | Path | None = None,
    text_query: str = "cat sounds",
) -> dict[str, Any]:
    """
    Full audio branch: AudioSep separation → emotion classification.

    Models can be pre-loaded and passed in (avoids reloading across calls).
    If None, they are loaded here.

    Returns a result dict with all metrics.
    """
    audio_path = Path(audio_path)

    # ── Load models if not provided ───────────────────────────────────────────
    if audiosep_model is None:
        with timer.step("audiosep_model_load"):
            audiosep_model = load_audiosep(device)

    if emotion_model is None:
        with timer.step("emotion_model_load"):
            emotion_model, emotion_config = load_emotion_model(device, checkpoint=emotion_ckpt)

    # ── AudioSep separation ───────────────────────────────────────────────────
    sep_audio_path = run_dir / "separated_audio.wav"
    with timer.step("audiosep_separation"):
        run_audiosep(
            audiosep_model,
            audio_path,
            sep_audio_path,
            device,
            text_query=text_query,
        )

    # ── Emotion classification ────────────────────────────────────────────────
    with timer.step("emotion_classification"):
        emotion_result = run_emotion_inference(
            sep_audio_path, emotion_model, emotion_config, device
        )

    return {
        "branch": "audio",
        "separated_audio": str(sep_audio_path),
        "emotion": emotion_result,
    }
