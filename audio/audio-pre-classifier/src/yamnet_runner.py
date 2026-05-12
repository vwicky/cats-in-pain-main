"""
PyTorch YAMNet inference via ``torch_audioset``: MPS/CPU, 521-class sigmoid outputs,
and aggregation of cat-related AudioSet classes into a single P(cat) per clip.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch_audioset.data.torch_input_processing import WaveformToInput
from torch_audioset.params import YAMNetParams
from torch_audioset.yamnet import yamnet

from . import config

logger = logging.getLogger(__name__)

# Official YAMNet frame hop (seconds); must be set before building mel patches
YAMNetParams.PATCH_HOP_SECONDS = 0.48


def load_category_rows() -> list[dict]:
    """Return the 521 YAMNet class entries in model output order (from bundled YAML)."""
    from torch_audioset.yamnet import yamnet_category_metadata

    meta = yamnet_category_metadata()
    if not isinstance(meta, list):
        raise TypeError("yamnet_category_metadata() expected a list")
    return meta


def discover_cat_class_indices(
    rows: Sequence[dict],
    keywords: Iterable[str] = config.CAT_KEYWORDS,
    *,
    include_roaring_cats: bool = config.INCLUDE_ROARING_CATS,
) -> list[int]:
    """
    Return output indices for classes that count toward P(cat).

    Matching rules:
    - Exclude ``Roaring cats (lions, tigers)`` unless ``include_roaring_cats`` is True.
    - Include if the display ``name`` contains a whole word ``cat`` (``\\bcat\\b``), or
      any of the lowercase ``keywords`` as a substring.
    """
    kw = tuple(k.lower() for k in keywords)
    cat_word = re.compile(r"\bcat\b", re.IGNORECASE)
    indices: list[int] = []

    for idx, row in enumerate(rows):
        name = str(row.get("name", ""))
        lower = name.lower()
        if not include_roaring_cats and "roaring cats" in lower:
            continue
        if cat_word.search(name):
            indices.append(idx)
            continue
        if any(k in lower for k in kw):
            indices.append(idx)

    indices.sort()
    logger.info(
        "Cat-related YAMNet class indices (%d): %s",
        len(indices),
        indices[:20] + (["..."] if len(indices) > 20 else []),
    )
    return indices


class YamNetRunner:
    """
    Load pretrained YAMNet, build log-mel patches (0.48 s hop), run inference,
    and aggregate cat-related class probabilities over time (max over frames).
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        patch_batch_size: int = config.DEFAULT_PATCH_BATCH_SIZE,
        cat_indices: list[int] | None = None,
        aggregate_cat_classes: str = config.AGGREGATE_CAT_CLASSES,
        include_roaring_cats: bool = config.INCLUDE_ROARING_CATS,
    ) -> None:
        self.device = device or config.device
        self.patch_batch_size = patch_batch_size
        self._transform = WaveformToInput()
        self._model = yamnet(pretrained=True).to(self.device).eval()

        rows = load_category_rows()
        if len(rows) != 521:
            logger.warning("Expected 521 YAMNet classes, got %d", len(rows))

        if cat_indices is not None:
            self._cat_indices = cat_indices
        else:
            self._cat_indices = discover_cat_class_indices(
                rows, include_roaring_cats=include_roaring_cats
            )
        self._cat_idx_tensor = (
            torch.tensor(self._cat_indices, dtype=torch.long, device=self.device)
            if self._cat_indices
            else None
        )
        if aggregate_cat_classes not in ("sum", "max"):
            raise ValueError("aggregate_cat_classes must be 'sum' or 'max'")
        self._agg = aggregate_cat_classes

    @torch.inference_mode()
    def predict_p_cat_from_waveform(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """
        Run YAMNet on a waveform tensor ``[1, T]`` at ``sample_rate`` (typically 16000).

        Returns aggregated P(cat) in [0, 1] (max over frames of per-frame cat score).
        """
        if waveform.dim() != 2 or waveform.shape[0] != 1:
            raise ValueError(f"Expected waveform [1, T], got {tuple(waveform.shape)}")

        # MelSpectrogram / STFT in torch_audioset live on CPU (window buffers). MPS waveform
        # triggers "stft input and window must be on the same device".
        from .audio_utils import pad_waveform_min_yamnet

        wf_cpu = pad_waveform_min_yamnet(waveform.detach().cpu().to(torch.float32))
        patches, _ = self._transform.wavform_to_log_mel(wf_cpu, sample_rate)
        if patches.numel() == 0 or patches.shape[0] == 0:
            logger.warning("No YAMNet patches produced; returning 0.0")
            return 0.0

        patches = patches.to(self.device)
        n = patches.shape[0]
        frame_scores: list[torch.Tensor] = []

        for start in range(0, n, self.patch_batch_size):
            batch = patches[start : start + self.patch_batch_size]
            logits = self._model(batch, to_prob=False)
            probs = torch.sigmoid(logits)
            frame_scores.append(self._per_frame_cat_score(probs))

        per_frame = torch.cat(frame_scores, dim=0)
        # Inner "sum" over cat classes can exceed 1; clamp to a scalar score in [0, 1]
        clip_score = float(torch.clamp(per_frame.max(), 0.0, 1.0).item()) if per_frame.numel() else 0.0
        return clip_score

    def _per_frame_cat_score(self, probs: torch.Tensor) -> torch.Tensor:
        """ probs: [B, 521] -> [B] """
        if self._cat_idx_tensor is None:
            return torch.zeros(probs.shape[0], device=probs.device)

        sel = probs.index_select(-1, self._cat_idx_tensor)
        if self._agg == "sum":
            return sel.sum(dim=-1)
        return sel.max(dim=-1).values

    def predict_p_cat_path(self, path: Path) -> float:
        """
        Load audio with :func:`audio_utils.load_waveform_16k_mono` and return P(cat).
        """
        from .audio_utils import load_waveform_16k_mono

        # Keep decoded audio on CPU; mel/STFT runs on CPU, YAMNet on MPS/CPU.
        wf, sr = load_waveform_16k_mono(path, device=torch.device("cpu"))
        return self.predict_p_cat_from_waveform(wf, sr)
