"""
SuperAnimal quadruped 39-keypoint names and ST-GCN skeleton edges.

Single source of truth (no torch). Imported by ``superanimal_quadruped_stgcn_graph``
and by visualization notebooks/scripts so edge lists cannot drift.
"""

from __future__ import annotations

KEYPOINTS: list[str] = [
    "nose",
    "upper_jaw",
    "lower_jaw",
    "mouth_end_right",
    "mouth_end_left",
    "right_eye",
    "right_earbase",
    "right_earend",
    "right_antler_base",
    "right_antler_end",
    "left_eye",
    "left_earbase",
    "left_earend",
    "left_antler_base",
    "left_antler_end",
    "neck_base",
    "neck_end",
    "throat_base",
    "throat_end",
    "back_base",
    "back_end",
    "back_middle",
    "back_middle_2",
    "tail_base",
    "front_left_thigh",
    "front_left_knee",
    "front_left_paw",
    "front_right_thigh",
    "front_right_knee",
    "front_right_paw",
    "back_left_paw",
    "back_left_knee",
    "back_left_thigh",
    "back_right_knee",
    "back_right_thigh",
    "back_right_paw",
    "belly_top",
    "belly_bottom",
    "belly_bottom_2",
]

NUM_KEYPOINTS: int = len(KEYPOINTS)

# Same edges as ST-GCN adjacency (undirected; each pair listed once).
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (1, 4),
    (0, 5),
    (0, 10),
    (5, 6),
    (6, 7),
    (6, 8),
    (8, 9),
    (10, 11),
    (11, 12),
    (11, 13),
    (13, 14),
    (5, 15),
    (10, 15),
    (15, 16),
    (15, 17),
    (17, 18),
    (15, 19),
    (19, 21),
    (21, 22),
    (22, 20),
    (20, 23),
    (19, 36),
    (36, 37),
    (37, 38),
    (23, 38),
    (15, 24),
    (24, 25),
    (25, 26),
    (15, 27),
    (27, 28),
    (28, 29),
    (23, 32),
    (32, 31),
    (31, 30),
    (23, 34),
    (34, 33),
    (33, 35),
]

# Optional edges for *drawing* only (not in ST-GCN graph): close obvious gaps in the mesh.
VIZ_SUPPLEMENT_EDGES: list[tuple[int, int]] = [
    (3, 4),  # mouth corners (both attach to upper_jaw but not to each other in ST-GCN)
]

# Semantic groups (same as former ``superanimal_quadruped_stgcn_graph.GROUPS``).
GROUPS: dict[str, list[int]] = {
    "face": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "neck_spine": [15, 16, 17, 18, 19, 20, 21, 22, 23],
    "front_legs": [24, 25, 26, 27, 28, 29],
    "back_legs": [30, 31, 32, 33, 34, 35],
    "belly": [36, 37, 38],
}

JOINT_REGION: dict[int, str] = {}
for _region_name, _idxs in GROUPS.items():
    for _ji in _idxs:
        JOINT_REGION[_ji] = _region_name

# BGR on black canvas (high contrast, distinct hues).
REGION_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "face": (102, 180, 255),  # light orange
    "neck_spine": (220, 220, 120),  # yellow-cyan
    "front_legs": (90, 140, 255),  # coral
    "back_legs": (220, 110, 255),  # magenta / pink
    "belly": (120, 220, 100),  # mint / green
}

# Edges whose endpoints fall in different ``GROUPS`` buckets.
INTER_REGION_EDGE_BGR: tuple[int, int, int] = (160, 160, 160)
