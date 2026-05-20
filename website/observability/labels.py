"""Canonical low-cardinality label values and StepTimer step → stage mapping."""

from __future__ import annotations

# Inference histogram / failure label
STAGE_AUDIO_PREPROCESS = "audio_preprocess"
STAGE_AUDIO_MODEL = "audio_model"
STAGE_POSE_DETECTION = "pose_detection"
STAGE_POSE_ESTIMATION = "pose_estimation"
STAGE_GRAPH_INFERENCE = "graph_inference"
STAGE_PIPELINE_TOTAL = "pipeline_total"

INFERENCE_STAGES = frozenset(
    {
        STAGE_AUDIO_PREPROCESS,
        STAGE_AUDIO_MODEL,
        STAGE_POSE_DETECTION,
        STAGE_POSE_ESTIMATION,
        STAGE_GRAPH_INFERENCE,
        STAGE_PIPELINE_TOTAL,
    }
)

BRANCH_AUDIO = "audio"
BRANCH_VIDEO = "video"
BRANCH_MULTICAT = "multicat"

# Exact StepTimer step name → stage
STEP_TO_STAGE: dict[str, str] = {
    "save_original_video": STAGE_PIPELINE_TOTAL,
    "audio_extraction": STAGE_AUDIO_PREPROCESS,
    "yamnet_preclassifier": STAGE_AUDIO_PREPROCESS,
    "audiosep_model_load": STAGE_AUDIO_PREPROCESS,
    "audiosep_separation": STAGE_AUDIO_PREPROCESS,
    "emotion_model_load": STAGE_AUDIO_MODEL,
    "emotion_classification": STAGE_AUDIO_MODEL,
    "pairwise_models_load": STAGE_GRAPH_INFERENCE,
    "meta_model_load": STAGE_GRAPH_INFERENCE,
    "yolo_inference": STAGE_POSE_DETECTION,
    "vitpose_inference": STAGE_POSE_ESTIMATION,
    "pairwise_stgcn_inference": STAGE_GRAPH_INFERENCE,
    "meta_learner_inference": STAGE_GRAPH_INFERENCE,
    "multicat_pass1_detect": STAGE_POSE_DETECTION,
    "pipeline_total": STAGE_PIPELINE_TOTAL,
}

_PREFIX_TO_STAGE: tuple[tuple[str, str], ...] = (
    ("multicat_pass2_track_", STAGE_POSE_ESTIMATION),
    ("pairwise_stgcn_track_", STAGE_GRAPH_INFERENCE),
    ("meta_learner_track_", STAGE_GRAPH_INFERENCE),
)


def step_to_stage(step_name: str) -> str | None:
    """Map a StepTimer step name to an inference stage label, or None if unmapped."""
    if step_name in STEP_TO_STAGE:
        return STEP_TO_STAGE[step_name]
    for prefix, stage in _PREFIX_TO_STAGE:
        if step_name.startswith(prefix):
            return stage
    return None
