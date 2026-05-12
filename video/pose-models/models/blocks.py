"""
Shared neural building blocks ported from `models_training.py` (legacy 08_models).

These encoders match the shapes used in SSL pretraining (`models_pretraining.py`) so
checkpoints remain compatible if you load pretrained weights. Widths follow the
original codebase: BiLSTM hidden 64, TCN channels (64,128), Transformer embed_dim 64,
ST-GCN channels 32→64→64 with a fixed 17-joint feline skeleton adjacency.

All modules take `dims` as the per-joint channel count (3 without kinematics, 6 with).
`num_joints` / `num_frames` are parameterized for flexibility with `config.yaml`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """MLP with LayerNorm (stable on small batches) and Dropout."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.58):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiLSTM_Encoder(nn.Module):
    def __init__(self, num_joints: int = 17, dims: int = 3, hidden_size: int = 64):
        super().__init__()
        input_size = num_joints * dims
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_features = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)
        _, (h_n, _) = self.lstm(x)
        return torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)


class TemporalBlock(nn.Module):
    """TCN block with dilated convolutions and residual connection."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding="same", dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding="same", dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_Encoder(nn.Module):
    def __init__(self, num_joints: int = 17, dims: int = 3, num_channels: tuple[int, ...] = (64, 128)):
        super().__init__()
        input_size = num_joints * dims
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size=3, dilation=dilation_size))

        self.network = nn.Sequential(*layers)
        self.out_features = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape
        x = x.view(B, T, V * C).permute(0, 2, 1)
        out = self.network(x)
        return out.mean(dim=2)


class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        num_frames: int = 35,
        num_joints: int = 17,
        dims: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        input_size = num_joints * dims
        self.num_frames = num_frames
        self.embedding = nn.Linear(input_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + 1, embed_dim) * 0.02)

        self.pos_dropout = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape
        x = x.view(B, T, V * C)
        x = self.embedding(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        out = self.transformer(x)
        return out[:, 0, :]


def _cat_adjacency(num_joints: int) -> torch.Tensor:
    """Normalized adjacency for COCO-style 17-keypoint cat skeleton (same edges as original)."""
    if num_joints != 17:
        raise ValueError(f"STGCN_Encoder in v2 expects num_joints==17, got {num_joints}")
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    A = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(num_joints)
    D = torch.sum(A, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, dropout: float = 0.3):
        super().__init__()
        self.spatial = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer("A", A)
        self.temporal = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding="same")
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spatial(x)
        out = torch.einsum("btvc,vw->btwc", out, self.A)

        out = out.permute(0, 3, 1, 2)
        res = x.permute(0, 3, 1, 2)

        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.temporal(out)

        res = self.residual(res)
        out = F.relu(out + res)
        return out.permute(0, 2, 3, 1)


class STGCN_Encoder(nn.Module):
    """Spatiotemporal GCN stack; outputs global average pooled (B, 64)."""

    def __init__(self, num_frames: int = 35, num_joints: int = 17, dims: int = 3):
        super().__init__()
        del num_frames  # unused; T comes from input
        A_norm = _cat_adjacency(num_joints)

        self.block1 = STGCNBlock(dims, 32, A_norm)
        self.block2 = STGCNBlock(32, 64, A_norm)
        self.block3 = STGCNBlock(64, 64, A_norm)

        self.out_features = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.mean(x, dim=1)
        x = torch.mean(x, dim=1)
        return x
