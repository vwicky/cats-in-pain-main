"""
ST-GCN on DeepLabCut SuperAnimal quadruped poses (39 keypoints).

Separate from P4 (17-keypoint ViT). Reuses ``STGCNBlock`` only; adjacency from
``superanimal_quadruped_stgcn_graph``.
"""

from __future__ import annotations

import torch.nn as nn

from model_base import  PoseModelBase
from models.blocks import  MLPBlock, STGCNBlock
from models.superanimal_quadruped_stgcn_graph import  NUM_KEYPOINTS, normalized_adjacency_tensor


class STGCN_EncoderDeepLabCut(nn.Module):
    """Three ST-GCN blocks with fixed 39-joint quadruped adjacency."""

    def __init__(self, dims: int):
        super().__init__()
        A_norm = normalized_adjacency_tensor()
        self.block1 = STGCNBlock(dims, 32, A_norm)
        self.block2 = STGCNBlock(32, 64, A_norm)
        self.block3 = STGCNBlock(64, 64, A_norm)
        self.out_features = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=1)
        x = x.mean(dim=1)
        return x


class P6PoseSTGCNDeepLabCut(PoseModelBase):
    """ST-GCN–DeepLabCut: 39-keypoint SuperAnimal skeleton + binary and/or multiclass heads."""

    def __init__(
        self,
        n_frames: int,
        n_keypoints: int,
        n_channels: int,
        n_classes: int = 5,
        *,
        binary_only: bool = False,
    ):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        if n_keypoints != NUM_KEYPOINTS:
            raise ValueError(f"P6 (ST-GCN–DeepLabCut) expects n_keypoints={NUM_KEYPOINTS}, got {n_keypoints}")
        self.binary_only = bool(binary_only)
        self.encoder = STGCN_EncoderDeepLabCut(dims=n_channels)
        if not self.binary_only:
            self.head_5 = MLPBlock(self.encoder.out_features, 128, n_classes)
        self.head_binary = MLPBlock(self.encoder.out_features, 128, 2)

    @property
    def model_id(self) -> str:
        return "P6"

    @property
    def model_name(self) -> str:
        return "ST-GCN-DeepLabCut"

    def forward(self, pose, mask):
        del mask
        z = self.encoder(pose)
        out = {"logits_binary": self.head_binary(z)}
        if not self.binary_only:
            out["logits_5"] = self.head_5(z)
        return out
