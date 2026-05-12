"""
Track 2 — P2: Temporal Convolutional Network (original `P2_Pose_TCN`).

Pose is reshaped to (B, C_in, T); two dilated TCN blocks with residuals aggregate
to channel-wise temporal features, then global average pooling over time. Dual
MLP heads on the pooled vector.

Maps to ``P2`` / ``Pose-TCN``.
"""

from __future__ import annotations

from model_base import  PoseModelBase
from models.blocks import  MLPBlock, TCN_Encoder


class P2PoseTCN(PoseModelBase):
    """TCN encoder + dual heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        self.encoder = TCN_Encoder(num_joints=n_keypoints, dims=n_channels)
        self.head_5 = MLPBlock(self.encoder.out_features, 64, n_classes)
        self.head_binary = MLPBlock(self.encoder.out_features, 64, 2)

    @property
    def model_id(self) -> str:
        return "P2"

    @property
    def model_name(self) -> str:
        return "Pose-TCN"

    def forward(self, pose, mask):
        del mask
        z = self.encoder(pose)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
