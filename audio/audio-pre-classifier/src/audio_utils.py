"""
Load audio as mono 16 kHz float32 in [-1, 1].

File decoding uses **librosa** first so we avoid PyTorch 2.9+ routing ``torchaudio.load``
through **TorchCodec** (which often requires FFmpeg dylibs that are missing or mismatched
on macOS Homebrew installs). Optional fallback: **soundfile** for PCM formats.
Resampling uses ``torchaudio.transforms.Resample`` when needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

from . import config

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Minimum waveform length for one YAMNet patch (0.96 s at 16 kHz)
_MIN_SAMPLES_ONE_PATCH: int = int(round(0.96 * config.SAMPLE_RATE))
MIN_SAMPLES_YAMNET_PATCH: int = _MIN_SAMPLES_ONE_PATCH


def pad_waveform_min_yamnet(waveform: torch.Tensor) -> torch.Tensor:
    """Zero-pad ``[1, T]`` on the right so YAMNet can form at least one patch."""
    if waveform.dim() != 2 or waveform.shape[0] != 1:
        return waveform
    if waveform.shape[-1] < _MIN_SAMPLES_ONE_PATCH:
        pad = _MIN_SAMPLES_ONE_PATCH - waveform.shape[-1]
        return torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def load_waveform_16k_mono(
    path: PathLike,
    device: torch.device | None = None,
    *,
    pad_to_min_patch: bool = True,
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and return ``(waveform, sample_rate)``.

    The waveform is shape ``[1, T]``, dtype float32, mono, sampled at
    ``config.SAMPLE_RATE``. Values are clamped to [-1, 1].

    Parameters
    ----------
    path :
        File path (wav, mp3, flac, etc.). MP3/M4A typically need FFmpeg available to librosa.
    device :
        Device for the returned tensor (default: ``config.device``).
    pad_to_min_patch :
        If True, zero-pad short clips so at least one YAMNet patch can be formed.

    Returns
    -------
    waveform : torch.Tensor
        Shape ``[1, num_samples]``.
    sample_rate : int
        Always ``config.SAMPLE_RATE`` after resampling.
    """
    target = device or config.device
    path = Path(path)

    w_cpu = _load_via_librosa_or_soundfile(path)

    w_cpu = torch.clamp(w_cpu, -1.0, 1.0)

    if pad_to_min_patch and w_cpu.shape[-1] < _MIN_SAMPLES_ONE_PATCH:
        pad = _MIN_SAMPLES_ONE_PATCH - w_cpu.shape[-1]
        w_cpu = torch.nn.functional.pad(w_cpu, (0, pad))

    out = w_cpu.to(target)
    return out, config.SAMPLE_RATE


def _load_via_librosa_or_soundfile(path: Path) -> torch.Tensor:
    """
    Return waveform float32 tensor shape ``[1, T]`` at ``config.SAMPLE_RATE`` (CPU).
    """
    try:
        y, sr = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True, dtype=np.float32)
        w = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
        return w
    except Exception as librosa_err:
        logger.debug("librosa.load failed for %s: %s", path, librosa_err)

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as sf_err:
        raise RuntimeError(
            f"Could not decode audio {path}: librosa ({librosa_err!r}); soundfile ({sf_err!r}). "
            "Install FFmpeg (e.g. brew install ffmpeg) for compressed formats."
        ) from sf_err

    # data: [num_samples, num_channels]
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(f"Unexpected soundfile array shape {data.shape} for {path}")
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    w = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)

    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        w = resampler(w)
    return w
