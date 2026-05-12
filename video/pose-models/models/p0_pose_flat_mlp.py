"""
Track 2 — P0: pose flatten MLP ablation (original `P0_Pose_Flat_MLP`).

Flattens the full tensor (T×J×C) into one vector and runs a wide MLP with two
output heads. No temporal structure (destroys motion prior by design in the legacy).

Maps to ``P0`` / ``Pose-Flat-MLP``.
"""

from __future__ import annotations

from model_base import  PoseModelBase
from models.blocks import  MLPBlock


class P0PoseFlatMLP(PoseModelBase):
    """Flatten (B,T,J,C) → MLP with dual heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        in_features = n_frames * n_keypoints * n_channels
        self.head_5 = MLPBlock(in_features, 512, n_classes)
        self.head_binary = MLPBlock(in_features, 512, 2)

    @property
    def model_id(self) -> str:
        return "P0"

    @property
    def model_name(self) -> str:
        return "Pose-Flat-MLP"

    def forward(self, pose, mask):
        del mask
        z = pose.view(pose.size(0), -1)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
