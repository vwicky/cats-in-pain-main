"""
Track 2 — P3b: Transformer encoder with CLS token (original `P3b_Pose_Transformer_CLS`).

Per-frame linear embedding, learnable [CLS] prepended, sinusoid-free learned positional
embeddings (length n_frames+1), TransformerEncoder, readout from CLS. Dual heads.

Maps to ``P3b`` / ``Pose-Transformer-CLS``.
"""

from __future__ import annotations

from model_base import  PoseModelBase
from models.blocks import  MLPBlock, Transformer_Encoder


class P3bPoseTransformerCLS(PoseModelBase):
    """Transformer + CLS + dual heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        self.encoder = Transformer_Encoder(
            num_frames=n_frames,
            num_joints=n_keypoints,
            dims=n_channels,
        )
        self.head_5 = MLPBlock(self.encoder.out_features, 64, n_classes)
        self.head_binary = MLPBlock(self.encoder.out_features, 64, 2)

    @property
    def model_id(self) -> str:
        return "P3b"

    @property
    def model_name(self) -> str:
        return "Pose-Transformer-CLS"

    def forward(self, pose, mask):
        del mask
        z = self.encoder(pose)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
