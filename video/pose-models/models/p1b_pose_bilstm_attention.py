"""
Track 2 — P1b: BiLSTM + temporal attention (original `P1b_Pose_BiLSTM_Attention`).

Same input layout as P1, but uses all timestep outputs with a learned attention
pooling vector instead of taking only final LSTM states.

Maps to ``P1b`` / ``Pose-BiLSTM-Attention``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_base import  PoseModelBase
from models.blocks import  MLPBlock


class P1bPoseBiLSTMAttention(PoseModelBase):
    """BiLSTM + attention pool + dual heads."""

    def __init__(self, n_frames: int, n_keypoints: int, n_channels: int, n_classes: int = 5):
        super().__init__(n_frames, n_keypoints, n_channels, n_classes)
        input_size = n_keypoints * n_channels
        hidden_size = 64
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )
        self.head_5 = MLPBlock(hidden_size * 2, 64, n_classes)
        self.head_binary = MLPBlock(hidden_size * 2, 64, 2)

    @property
    def model_id(self) -> str:
        return "P1b"

    @property
    def model_name(self) -> str:
        return "Pose-BiLSTM-Attention"

    def forward(self, pose, mask):
        del mask
        B, T, V, C = pose.shape
        x = pose.view(B, T, V * C)
        output, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(output), dim=1)
        z = torch.sum(attn_weights * output, dim=1)
        return {"logits_5": self.head_5(z), "logits_binary": self.head_binary(z)}
