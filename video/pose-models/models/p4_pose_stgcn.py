"""
Track 2 — P4: Spatiotemporal graph convolutional network (original `P4_Pose_ST_GCN`).

Three ST-GCN blocks with a fixed normalized 17-joint feline adjacency (see `blocks.py`).
**Requires `n_keypoints == 17`** — the graph edges are defined for the standard 17-keypoint
layout. Temporal then spatial mean pooling → dual MLP heads (hidden 128 for 5-class head
path to mirror the original wider head).

Maps to ``P4`` / ``Pose-ST-GCN``.
"""

from __future__ import annotations

from model_base import  PoseModelBase
from models.blocks import  MLPBlock, STGCN_Encoder


class P4PoseSTGCN(PoseModelBase):
    """ST-GCN encoder + dual heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        if n_keypoints != 17:
            raise ValueError(f"P4 (ST-GCN) expects n_keypoints=17, got {n_keypoints}")
        self.encoder = STGCN_Encoder(num_frames=n_frames, num_joints=n_keypoints, dims=n_channels)
        self.head_5 = MLPBlock(self.encoder.out_features, 128, n_classes)
        self.head_binary = MLPBlock(self.encoder.out_features, 128, 2)

    @property
    def model_id(self) -> str:
        return "P4"

    @property
    def model_name(self) -> str:
        return "Pose-ST-GCN"

    def forward(self, pose, mask):
        del mask
        z = self.encoder(pose)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
