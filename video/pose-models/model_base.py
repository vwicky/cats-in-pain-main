"""
Abstract base class for pose-sequence classification models.
Concrete implementations live in model_training_v2/models/.

Track 2 in models_training.py defines six pose architectures: P0, P1, P1b, P2, P3b, P4
(all implemented here plus an optional ``dummy`` smoke-test model).
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class PoseModelBase(ABC, nn.Module):
    """
    Input contract:
      pose:  FloatTensor (B, T, J, C)  — batch, frames, joints, channels
      mask:  BoolTensor  (B, T)        — True=real, False=padded (may be ignored by encoders)

    Output:
      "logits_5": (B, 5), "logits_binary": (B, 2)

    Loss = focal_5class + 0.3 * focal_binary
    """

    def __init__(
        self,
        n_frames: int,
        n_keypoints: int,
        n_channels: int,
        n_classes: int = 5,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_keypoints = n_keypoints
        self.n_channels = n_channels
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, pose, mask) -> dict:
        """Returns {"logits_5": (B,5), "logits_binary": (B,2)}."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """e.g. 'P1'."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name."""

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
