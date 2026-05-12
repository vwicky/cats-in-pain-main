"""
SuperAnimal quadruped (39 keypoints) skeleton for ST-GCN–DeepLabCut.

Adjacency matches ``_cat_adjacency`` in ``blocks.py``: undirected edges + self-loops,
then symmetric normalization D^{-1/2} A D^{-1/2}.
"""

from __future__ import annotations

from typing import Any

from quadruped_skeleton_spec import  (
    GROUPS,
    KEYPOINTS,
    NUM_KEYPOINTS,
    SKELETON_EDGES,
)

# Left–right swaps for horizontal flip after x → 1 - x (same convention as 17-kp path).
FLIP_PAIRS: list[tuple[int, int]] = [
    (3, 4),
    (5, 10),
    (6, 11),
    (7, 12),
    (8, 13),
    (9, 14),
    (24, 27),
    (25, 28),
    (26, 29),
    (32, 34),
    (31, 33),
    (30, 35),
]

def normalized_adjacency_tensor(*, device: Any = None) -> Any:
    """Symmetric normalized adjacency (NUM_KEYPOINTS, NUM_KEYPOINTS), float32."""
    import torch

    n = NUM_KEYPOINTS
    A = torch.zeros(n, n, dtype=torch.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(n, dtype=torch.float32)
    d = A.sum(dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    out = D_inv_sqrt @ A @ D_inv_sqrt
    if device is not None:
        out = out.to(device)
    return out
