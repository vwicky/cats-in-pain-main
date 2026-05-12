"""
Dummy flatten + MLP baseline for pipeline smoke tests.

Not in the original models_training registry; kept for quick integration checks.
Same dual-head layout as production models (5-class + binary).
"""

from __future__ import annotations

import torch.nn as nn

from model_base import  PoseModelBase


class DummyPoseModel(PoseModelBase):
    """Placeholder: flatten pose → linear → ReLU → Dropout → two linear heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        in_dim = n_frames * n_keypoints * n_channels
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.head_5 = nn.Linear(128, n_classes)
        self.head_binary = nn.Linear(128, 2)

    @property
    def model_id(self) -> str:
        return "dummy"

    @property
    def model_name(self) -> str:
        return "Dummy-MLP"

    def forward(self, pose, mask):
        del mask
        z = self.encoder(pose)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
