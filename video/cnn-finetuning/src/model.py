"""EfficientNet-B3 backbone with attention pooling and dual classification heads."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

logger = logging.getLogger("cat_cnn")


class AttentionPooling(nn.Module):
    """Learned per-frame attention pooling.

    Given frame_features (B, N_frames, feat_dim), produces a weighted
    average (B, feat_dim) where weights are learned via a linear layer.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(x).squeeze(-1)          # (B, N)
        weights = F.softmax(scores, dim=1)          # (B, N)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, feat_dim)
        return pooled, weights


class MeanPooling(nn.Module):
    """Simple mean pooling over the frame dimension."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        weights = torch.full((B, N), 1.0 / N, device=x.device)
        pooled = x.mean(dim=1)
        return pooled, weights


class CatPainCNN(nn.Module):
    """EfficientNet-B3 backbone with dual 5-class and binary heads.

    Architecture:
      - EfficientNet-B3 feature extractor (1536-d per frame)
      - Applied independently to each of the N frames
      - AttentionPooling (or MeanPooling) over frames -> (B, 1536)
      - head_5class: Linear -> LayerNorm -> ReLU -> Dropout -> Linear(5)
      - head_binary: Linear(64) -> ReLU -> Linear(2)
    """

    FEAT_DIM = 1536  # EfficientNet-B3 output channels

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        logger.info(
            "\U0001f4e5 Loading EfficientNet-B3 (IMAGENET1K_V1) "
            "\u2014 will auto-download if needed (~49MB)"
        )
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        base = efficientnet_b3(weights=weights)

        self.backbone = base.features
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)

        pooling_type = cfg.get("pooling", "attention")
        if pooling_type == "attention":
            self.frame_pool = AttentionPooling(self.FEAT_DIM)
        else:
            self.frame_pool = MeanPooling()

        hidden = cfg.get("head_hidden_dim", 256)
        dropout = cfg.get("head_dropout", 0.50)
        num_classes = len(cfg.get("classes_5", [""] * 5))

        self.head_5class = nn.Sequential(
            nn.Linear(self.FEAT_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        self.head_binary = nn.Sequential(
            nn.Linear(self.FEAT_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1536-d features from a single image (B, C, H, W)."""
        feat = self.backbone(x)                     # (B, 1536, h, w)
        feat = self.backbone_pool(feat)             # (B, 1536, 1, 1)
        return feat.flatten(1)                      # (B, 1536)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass on multi-frame input.

        Args:
            frames: (B, N, C, H, W)

        Returns:
            dict with logits_5 (B, 5), logits_binary (B, 2),
            frame_weights (B, N).
        """
        B, N, C, H, W = frames.shape
        flat = frames.reshape(B * N, C, H, W)
        feats = self._extract_features(flat)        # (B*N, 1536)
        feats = feats.reshape(B, N, -1)             # (B, N, 1536)

        pooled, weights = self.frame_pool(feats)    # (B, 1536), (B, N)

        logits_5 = self.head_5class(pooled)         # (B, 5)
        logits_binary = self.head_binary(pooled)    # (B, 2)

        return {
            "logits_5": logits_5,
            "logits_binary": logits_binary,
            "frame_weights": weights,
        }

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for head-only training."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("\U0001f512 Backbone frozen (head only training)")

    def unfreeze_last_n_blocks(self, n: int, lr_backbone: float = 0.0) -> None:
        """Unfreeze the last n MBConv block groups of EfficientNet-B3.

        EfficientNet-B3 features has indices 0-8:
          0: stem conv
          1-7: MBConv block groups
          8: final conv
        This unfreezes features[9-n:] (the last n groups + final conv).
        """
        num_children = len(list(self.backbone.children()))
        start = max(0, num_children - n)
        for i, child in enumerate(self.backbone.children()):
            if i >= start:
                for p in child.parameters():
                    p.requires_grad = True
        # Always unfreeze final conv (index 8)
        if num_children > 8:
            for p in list(self.backbone.children())[8].parameters():
                p.requires_grad = True
        logger.info(
            f"\U0001f513 Unfroze last {n} backbone blocks "
            f"(lr_backbone={lr_backbone:.1e})"
        )

    def count_trainable_params(self) -> dict[str, int]:
        """Count trainable parameters per component."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        backbone = _count(self.backbone)
        head5 = _count(self.head_5class)
        head_bin = _count(self.head_binary)
        pool = _count(self.frame_pool) if hasattr(self.frame_pool, "parameters") else 0
        total = backbone + head5 + head_bin + pool
        return {
            "backbone": backbone,
            "head_5": head5,
            "head_binary": head_bin,
            "pooling": pool,
            "total": total,
        }
