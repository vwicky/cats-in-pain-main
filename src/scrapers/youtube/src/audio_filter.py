"""YAMNet-based cat-sound gate (tensorflow_hub)."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pydub import AudioSegment

from src.utils import project_root

YAMNET_HUB = "https://tfhub.dev/google/yamnet/1"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)


def _pydub_to_float_mono_16k(segment: AudioSegment, target_sr: int = 16000) -> np.ndarray:
    """Export pydub segment to mono float32 waveform at target_sr."""
    seg = segment.set_channels(1)
    raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.frame_width == 2:
        max_val = 32768.0
    else:
        max_val = float(2 ** (8 * seg.frame_width - 1))
    raw = raw / max_val
    sr0 = seg.frame_rate
    if sr0 != target_sr:
        raw = librosa.resample(raw, orig_sr=sr0, target_sr=target_sr)
    return raw.astype(np.float32)


def _ensure_class_map_csv(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve(CLASS_MAP_URL, path)


def _load_display_name_to_index(class_map_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(class_map_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            idx = int(row["index"])
            name = (row.get("display_name") or "").strip()
            if name:
                out[name] = idx
    return out


class YamNetCatGate:
    """Cat-sound gate using YAMNet (tensorflow_hub). Compatible with processor's predict() API."""

    def __init__(self, cfg: dict, logger: logging.Logger):
        self._logger = logger
        audio_cfg = cfg.get("audio", {})
        self._threshold = float(audio_cfg.get("yamnet_threshold", 0.3))
        self._class_names: list[str] = list(audio_cfg.get("cat_sound_classes") or ["Cat", "Meow", "Purr", "Hiss"])
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        root = project_root()
        models_dir = root / "models"
        class_map_path = models_dir / "yamnet_class_map.csv"
        _ensure_class_map_csv(class_map_path)
        name_to_idx = _load_display_name_to_index(class_map_path)
        self._indices: list[int] = []
        missing: list[str] = []
        for name in self._class_names:
            if name in name_to_idx:
                self._indices.append(name_to_idx[name])
            else:
                missing.append(name)
        if missing:
            logger.warning("YAMNet: class name(s) not in class map: %s", missing)

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        self._model = hub.load(YAMNET_HUB)
        self._last_predict_failure: str | None = None

        classes_s = ", ".join(self._class_names)
        print("  🔊 Audio filter: YAMNet (tensorflow_hub)")
        print(f"  🔊 Cat sound classes: {classes_s}")
        print(f"  🔊 Threshold: {self._threshold:.2f}")
        logger.info("Audio filter: YAMNet (tensorflow_hub); classes=%s; threshold=%s", classes_s, self._threshold)

    def predict(self, audio_chunk: AudioSegment) -> tuple[bool, float]:
        """Return (accept_chunk, max_cat_class_score)."""
        try:
            wav = _pydub_to_float_mono_16k(audio_chunk, 16000)
            if wav.size == 0:
                self._last_predict_failure = "no_features"
                return False, 0.0
            wav_tf = tf.convert_to_tensor(wav, dtype=tf.float32)
            scores, _embeddings, _spectro = self._model(wav_tf)
            scores_np = scores.numpy()
            mean_scores = np.mean(scores_np, axis=0)
            if self._indices:
                sub = mean_scores[self._indices]
                max_cat = float(np.max(sub))
            else:
                max_cat = float(np.max(mean_scores))
            self._last_predict_failure = None if max_cat >= self._threshold else "below_threshold"
            ok = max_cat >= self._threshold
            return ok, max_cat
        except Exception:
            self._logger.debug("YAMNet predict failed on chunk", exc_info=True)
            self._last_predict_failure = "no_features"
            return False, 0.0

    @property
    def model_name(self) -> str:
        return "yamnet_tfhub"
