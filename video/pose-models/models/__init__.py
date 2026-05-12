"""
Pose models for model_training_v2 (Track 2 from models_training.py).

Registry includes all six legacy pose IDs — P0, P1, P1b, P2, P3b, P4 — plus ``dummy``.
There is no separate P3 in the source repo (only P3b Transformer-CLS). CLIP-only (C1–C2)
and fusion (F1–F5) models are intentionally not included here.
"""

from __future__ import annotations

from models.dummy_pose_model import  DummyPoseModel
from models.p0_pose_flat_mlp import  P0PoseFlatMLP
from models.p1_pose_bilstm import  P1PoseBiLSTM
from models.p1b_pose_bilstm_attention import  P1bPoseBiLSTMAttention
from models.p2_pose_tcn import  P2PoseTCN
from models.p3b_pose_transformer import  P3bPoseTransformerCLS
from models.p4_pose_stgcn import  P4PoseSTGCN
from models.stgcn_deeplabcut_model import  P6PoseSTGCNDeepLabCut

MODEL_REGISTRY: dict[str, tuple[type, dict]] = {
    "P0": (P0PoseFlatMLP, {}),
    "P1": (P1PoseBiLSTM, {}),
    "P1b": (P1bPoseBiLSTMAttention, {}),
    "P2": (P2PoseTCN, {}),
    "P3b": (P3bPoseTransformerCLS, {}),
    "P4": (P4PoseSTGCN, {}),
    "P6": (P6PoseSTGCNDeepLabCut, {}),
    "dummy": (DummyPoseModel, {}),
}

DEFAULT_MODEL_IDS: tuple[str, ...] = tuple(k for k in MODEL_REGISTRY if k not in ("dummy", "P6"))

__all__ = [
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_IDS",
    "DummyPoseModel",
    "P0PoseFlatMLP",
    "P1PoseBiLSTM",
    "P1bPoseBiLSTMAttention",
    "P2PoseTCN",
    "P3bPoseTransformerCLS",
    "P4PoseSTGCN",
    "P6PoseSTGCNDeepLabCut",
]
