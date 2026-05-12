"""
Track 2 — P1: BiLSTM over pose sequences (original `P1_Pose_BiLSTM`).

Frames are flattened per timestep (J×C); a bidirectional LSTM reads the sequence
and the final forward/backward hidden states are concatenated, then passed to
dual MLP heads.

Maps to ``P1`` / ``Pose-BiLSTM``.
"""

from __future__ import annotations

from model_base import  PoseModelBase
from models.blocks import  BiLSTM_Encoder, MLPBlock


class P1PoseBiLSTM(PoseModelBase):
    """BiLSTM pose encoder + dual heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        self.encoder = BiLSTM_Encoder(num_joints=n_keypoints, dims=n_channels)
        self.head_5 = MLPBlock(self.encoder.out_features, 64, n_classes)
        self.head_binary = MLPBlock(self.encoder.out_features, 64, 2)

    @property
    def model_id(self) -> str:
        return "P1"

    @property
    def model_name(self) -> str:
        return "Pose-BiLSTM"

    def forward(self, pose, mask):
        del mask
        z = self.encoder(pose)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
