"""Audio preclassifier: YAMNet v3 or sklearn v1/v2 via inference.predict_one (v2 default)."""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
import types
import warnings
from pathlib import Path
from typing import Any, Callable

from sklearn.exceptions import InconsistentVersionWarning

import joblib
import numpy as np
from pydub import AudioSegment

from src.esc_class_names import pre_classifier_class_names

_REPO_ROOT = Path(__file__).resolve().parents[4]

_AP3_PKG = "tiktok_ap3_internal"


def _load_audio_preclassification_v3() -> tuple[type, Callable[[Any], Any]]:
    ap3 = _REPO_ROOT / "audio" / "audio-pre-classifier"
    src = ap3 / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"audio/audio-pre-classifier/src not found at {src}")

    if _AP3_PKG not in sys.modules:
        pkg = types.ModuleType(_AP3_PKG)
        pkg.__path__ = [str(src)]
        sys.modules[_AP3_PKG] = pkg

    load_order = ("config", "audio_utils", "yamnet_runner", "video_chunk_pipeline")
    for name in load_order:
        full = f"{_AP3_PKG}.{name}"
        if full in sys.modules:
            continue
        path = src / f"{name}.py"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ap3 module: {path}")
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _AP3_PKG
        mod.__name__ = full
        sys.modules[full] = mod
        spec.loader.exec_module(mod)

    ymod = sys.modules[f"{_AP3_PKG}.yamnet_runner"]
    vmod = sys.modules[f"{_AP3_PKG}.video_chunk_pipeline"]
    return ymod.YamNetRunner, vmod.pydub_segment_to_waveform_16k_mono


def _proba_for_binary_classes(model: Any, proba_row: np.ndarray) -> tuple[float, float]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        d = {int(c): float(proba_row[i]) for i, c in enumerate(classes)}
        return d.get(0, 0.0), d.get(1, 0.0)
    if len(proba_row) >= 2:
        return float(proba_row[0]), float(proba_row[1])
    return 1.0 - float(proba_row[0]), float(proba_row[0])


def _ensure_ap2_on_path() -> None:
    ap2 = _REPO_ROOT / "_archive" / "audio_preclassifier_v2"
    if ap2.is_dir() and str(ap2) not in sys.path:
        sys.path.insert(0, str(ap2))


def extract_features_pydub(
    audio_segment: AudioSegment,
    *,
    extract_kw: dict[str, Any],
    sample_rate: int,
    n_mfcc: int,
) -> np.ndarray | None:
    _ensure_ap2_on_path()
    from features import (  # noqa: E402
        audiosegment_to_float_array,
        extract_features_from_array,
    )

    try:
        audio_float = audiosegment_to_float_array(audio_segment, sample_rate)
        return extract_features_from_array(
            audio_float,
            sample_rate,
            n_mfcc,
            **extract_kw,
        )
    except Exception:
        return None


def _resolve_v2_pickle(repo_root: Path, v2_path_cfg: str) -> Path | None:
    if (v2_path_cfg or "").strip():
        p = Path(v2_path_cfg)
        return repo_root / p if not p.is_absolute() else p
    runs_dir = repo_root / "_archive" / "audio_preclassifier_v2" / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = list(runs_dir.glob("run_*/best_model.pkl"))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


class AudioClassifier:
    def __init__(self, cfg: dict, logger: logging.Logger):
        self._logger = logger
        audio_cfg = cfg.get("audio", {})
        backend = str(audio_cfg.get("backend", "sklearn")).strip().lower()
        self._threshold = float(audio_cfg.get("cat_prob_threshold", 0.5))
        self._yamnet: Any = None
        self._pydub_to_wf: Callable[[AudioSegment], Any] | None = None
        self._yamnet_sr = 16_000
        self._last_predict_failure: str | None = None
        self._v2_artifact: Any = None
        self._use_predict_one = False
        self._predict_tmp_dir: Path | None = None
        pt = (audio_cfg.get("predict_temp_dir") or "").strip()
        if pt:
            p = Path(pt)
            self._predict_tmp_dir = p if p.is_absolute() else _REPO_ROOT / p
            self._predict_tmp_dir.mkdir(parents=True, exist_ok=True)

        root = _REPO_ROOT

        if backend == "yamnet":
            ap3 = root / "audio" / "audio-pre-classifier"
            if not ap3.is_dir():
                raise FileNotFoundError(
                    f"audio_preclassification_v3 not found at {ap3}; set audio.backend: sklearn"
                )
            YamNetRunner, pydub_segment_to_waveform_16k_mono = _load_audio_preclassification_v3()

            patch_bs = int(audio_cfg.get("yamnet_patch_batch_size", 32))
            agg = str(audio_cfg.get("yamnet_aggregate_cat_classes", "sum")).strip().lower()
            if agg not in ("sum", "max"):
                raise ValueError("audio.yamnet_aggregate_cat_classes must be 'sum' or 'max'")
            include_roar = bool(audio_cfg.get("yamnet_include_roaring_cats", False))
            self._yamnet = YamNetRunner(
                patch_batch_size=patch_bs,
                aggregate_cat_classes=agg,
                include_roaring_cats=include_roar,
            )
            self._pydub_to_wf = pydub_segment_to_waveform_16k_mono
            self._version = "v3"
            self._model = None
            self._extract_segment = lambda _seg: np.array([])
            self._cat_idx = 0
            logger.info("🔊 Audio classifier: v3 YAMNet (%s)", ap3)
            logger.info("🔊 Cat probability threshold: %.4f", self._threshold)
            return

        _ensure_ap2_on_path()
        from features import (  # noqa: E402
            N_MFCC,
            SAMPLE_RATE,
            WAVEFORM_NORMALIZE_EPS_DEFAULT,
            merge_feature_extract_kwargs,
        )
        v1_path = audio_cfg.get("model_path", "models/audio_preclassifier/voting_classifier_with-add-data.pkl")
        v2_path_cfg = (audio_cfg.get("v2_model_path") or "").strip()
        prefer_v2 = bool(audio_cfg.get("prefer_v2", True))

        yaml_fc = {
            "waveform_normalize": str(audio_cfg.get("waveform_normalize", "none")),
            "waveform_normalize_eps": float(
                audio_cfg.get("waveform_normalize_eps", WAVEFORM_NORMALIZE_EPS_DEFAULT)
            ),
            "waveform_normalize_noise_floor": float(
                audio_cfg.get("waveform_normalize_noise_floor", 0.01)
            ),
            "waveform_normalize_rms_target": float(
                audio_cfg.get("waveform_normalize_rms_target", 0.1)
            ),
        }
        v1_abs = root / v1_path if not Path(v1_path).is_absolute() else Path(v1_path)

        self._model: Any = None
        self._version = "v1"
        self._cat_idx = pre_classifier_class_names.index("cat")
        v1_kw = merge_feature_extract_kwargs({**yaml_fc, "waveform_normalize": "none"})
        self._extract_segment = lambda seg: extract_features_pydub(
            seg,
            extract_kw=v1_kw,
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
        )

        if prefer_v2:
            v2_abs = _resolve_v2_pickle(root, v2_path_cfg)
            if v2_abs is not None and v2_abs.is_file():
                try:
                    _ensure_ap2_on_path()
                    from inference import load_model as inf_load_model  # noqa: E402

                    # v2 artifact is joblib/sklearn; unpickle version warnings are about the bundle, not v1 fallback.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InconsistentVersionWarning)
                        self._v2_artifact = inf_load_model(v2_abs)
                    self._use_predict_one = True
                    self._version = "v2"
                    logger.info("🔊 Audio classifier: v2 via predict_one (%s)", v2_abs)
                except Exception as e:
                    logger.warning("v2 load failed (%s); falling back to v1 sklearn path", e)
                    self._v2_artifact = None
                    self._use_predict_one = False
            elif prefer_v2:
                logger.warning(
                    "prefer_v2 is true but no v2 model found; falling back to v1 sklearn path"
                )

        if self._use_predict_one and self._v2_artifact is not None:
            logger.info("🔊 Cat probability threshold: %.2f (predict_one)", self._threshold)
            return

        if self._model is None:
            self._model = joblib.load(v1_abs)
            self._version = "v1"
            self._extract_segment = lambda seg: extract_features_pydub(
                seg,
                extract_kw=v1_kw,
                sample_rate=SAMPLE_RATE,
                n_mfcc=N_MFCC,
            )
            logger.info("🔊 Audio classifier: v1 (%s)", v1_abs)

        logger.info("🔊 Cat probability threshold: %.2f", self._threshold)

    def extract_features(self, audio_chunk: AudioSegment) -> np.ndarray | None:
        if self._version == "v3":
            return None
        return self._extract_segment(audio_chunk)

    def predict_from_features(self, features: np.ndarray) -> tuple[bool, float]:
        if self._version == "v3":
            raise RuntimeError("predict_from_features is not used for audio.backend: yamnet")
        X = features.reshape(1, -1)
        proba_row = self._model.predict_proba(X)[0]
        if self._version == "v2" and not self._use_predict_one:
            _, p_cat = _proba_for_binary_classes(self._model, proba_row)
            return p_cat >= self._threshold, float(p_cat)
        proba = float(proba_row[self._cat_idx])
        return proba >= self._threshold, proba

    def predict(self, audio_chunk: AudioSegment) -> tuple[bool, float]:
        if self._version == "v3":
            assert self._pydub_to_wf is not None and self._yamnet is not None
            try:
                wf = self._pydub_to_wf(audio_chunk)
                p_cat = float(self._yamnet.predict_p_cat_from_waveform(wf, self._yamnet_sr))
                ok = p_cat >= self._threshold
                self._last_predict_failure = None if ok else "below_threshold"
                return ok, p_cat
            except Exception:
                self._logger.debug("YAMNet predict failed on chunk", exc_info=True)
                self._last_predict_failure = "no_features"
                return False, 0.0

        if self._use_predict_one and self._v2_artifact is not None:
            _ensure_ap2_on_path()
            from inference import predict_one  # noqa: E402

            tmp_kw: dict[str, Any] = {"suffix": ".mp3", "delete": False}
            if self._predict_tmp_dir is not None:
                tmp_kw["dir"] = str(self._predict_tmp_dir)
            with tempfile.NamedTemporaryFile(**tmp_kw) as tmp:
                tmp_path = Path(tmp.name)
            try:
                audio_chunk.export(str(tmp_path), format="mp3")
                r = predict_one(
                    self._v2_artifact,
                    tmp_path,
                    threshold=self._threshold,
                )
                if not r.get("ok"):
                    self._last_predict_failure = "no_features"
                    return False, 0.0
                p_cat = float(r.get("proba_cat", 0.0))
                ok = bool(r.get("label") == 1) if "label" in r else p_cat >= self._threshold
                self._last_predict_failure = None if ok else "below_threshold"
                return ok, p_cat
            except Exception:
                self._logger.debug("predict_one failed on chunk", exc_info=True)
                self._last_predict_failure = "no_features"
                return False, 0.0
            finally:
                try:
                    if tmp_path.is_file():
                        tmp_path.unlink()
                except OSError:
                    pass

        features = self.extract_features(audio_chunk)
        if features is None:
            self._last_predict_failure = "no_features"
            return False, 0.0
        ok, proba = self.predict_from_features(features)
        self._last_predict_failure = None if ok else "below_threshold"
        return ok, proba

    @property
    def model_name(self) -> str:
        return self._version
