# Cat-in-Pain Inference Pipeline

End-to-end inference on a single video clip. The pipeline runs from the **repo root**.

## Pipeline flow

```
Input video
    │
    ├─ ffmpeg → extracted_audio.wav (16 kHz mono)
    │
    └─ YAMNet pre-classifier → P(cat)
           │
           ├─ P(cat) ≥ threshold ──► AUDIO BRANCH
           │       │
           │       ├─ AudioSep → separated_audio.wav ("cat sounds")
           │       └─ CatEmotionModel → softmax [10 classes]
           │
           └─ P(cat) < threshold ──► VIDEO BRANCH
                   │
                   ├─ VitInference (ViTPose + YOLO8x)
                   │       ├─ pose_video.mp4         (sampled temporal overlay @ 5 fps)
                   │       ├─ raw_poses.npy          (35 × 17 × 3 @ 5 Hz, matches training)
                   │       └─ raw_poses_mask.npy   (35 bool, False = padded timestep)
                   │
                   ├─ 9 × P4PoseSTGCN → P(Paining|pair) per pair
                   └─ LogReg L2 meta → P(pain) / P(non_pain) + decision
```

## Quick start

```bash
# From repo root
python src/inference/pipeline.py --video path/to/clip.mp4
```

The **audio branch** (AudioSep / CLAP) needs extra Python packages (`lightning`, `transformers`, `huggingface_hub`, `h5py`, …) that are **not** in the root `requirements.txt`:

```bash
pip install -r src/inference/requirements-audiosep.txt
```

```bash
# With explicit settings
python src/inference/pipeline.py \
    --video path/to/clip.mp4 \
    --device mps \
    --cat-threshold 0.5 \
    --output-dir runs/inference \
    --stack-run runs/pose-models/stgcn_dlc_stack_20260504_235843/run_05 \
    --verbose
```

```bash
# Sliding-window split inference (example: 6s window, 3s step)
python src/inference/pipeline.py \
    --video path/to/clip.mp4 \
    --split-window-sec 6 \
    --split-step-sec 3
```

### Multicat video-only mode (`--multicat-video-only`)

When set, the pipeline **always** runs the **video branch** for pain scoring (YAMNet still
runs; `p_cat` is stored for debugging). SORT is enabled in ViTPose (`single_pose=False`);
each qualifying track gets `cats/<local_track_id>/raw_poses.npy` + mask and its own ST-GCN +
meta stack. Artifact paths in JSON are **relative to `run_dir`**.

| Flag | Default | Meaning |
|------|---------|---------|
| `--multicat-video-only` | off | Enable multicat forced-video mode |
| `--multicat-max-cats` | 8 | Cap scored tracks after coverage filter |
| `--multicat-min-track-coverage` | 0.15 | Min fraction of sampled frames track must appear |
| `--multicat-decision-threshold` | 0.5 | `p_pain` cutoff for prevalence (not YAMNet `cat_threshold`) |
| `--multicat-summary-strategy` | `coverage_weighted_mean` | `max` \| `mean` \| `majority_above_threshold` \| `coverage_weighted_mean` |

Each `cats[]` entry includes `local_track_id`, `window_index` (split windows), and optional
future fields `stable_cat_index` / `stitch_confidence` (reserved for cross-window identity).

### Split (sliding-window) mode

For each window the pipeline **cuts a clip** under `clips/`, then runs the full
flow on **that clip**: extract audio → YAMNet **on the clip’s audio** → audio or
video branch for that clip only. Routing is **clip-wise**, not a single decision
for the whole video.

Each per-window `result` in `pipeline_result.json` may include `clip_audio_probe`:
`duration_sec`, `rms` (normalized RMS of extracted 16-bit mono WAV), and
`likely_silent` (very low energy — distinct from “audio present but P(cat) below
threshold”, which still routes to the video branch).

## Output artifacts

Each run produces a timestamped directory `runs/inference/<timestamp>_<video_stem>/`:

| File | Description |
|------|-------------|
| `original_video.*` | Copy of the input video |
| `extracted_audio.wav` | 16 kHz mono audio extracted from video |
| `separated_audio.wav` | *(audio branch)* AudioSep-separated cat sounds |
| `pose_video.mp4` | *(video branch)* ViTPose overlay visualisation |
| `raw_poses.npy` | *(video branch)* Keypoints `(35, 17, 3)` sampled at **5 Hz** over the first ≤**7** s (`dataset_construction` pose_extraction) — feeds ST-GCN |
| `raw_poses_mask.npy` | *(video branch)* Boolean mask `(35,)`: `True` = real sample, `False` = padded (short clips) |
| `pipeline_result.json` | Full result: branch, all probs, emotion softmax or pairwise probs + meta decision |
| `timing.json` | Wall-clock seconds per pipeline step |
| `pipeline.log` | Full debug log |

When split mode is enabled, `pipeline_result.json` at the top-level run dir
contains:
- `windows`: one nested result per sliding clip
- `summary`: video-level aggregation (`video_level_p_pain_max`, `video_level_p_pain_mean`,
  and thresholded `video_level_decision_max` / `video_level_decision_mean`)

### `pipeline_result.json` structure

**Audio branch:**
```json
{
  "branch": "audio",
  "p_cat": 0.87,
  "emotion": {
    "predicted_class": "Paining",
    "confidence": 0.62,
    "softmax": {"Angry": 0.03, "Paining": 0.62, ...}
  },
  "timing_seconds": {"audio_extraction": 1.2, "audiosep_separation": 8.4, ...}
}
```

**Video branch:**
```json
{
  "branch": "video",
  "p_cat": 0.12,
  "pose_meta": {
    "n_frames_source_video": 224,
    "n_frames_sampled_inference": 35,
    "n_frames_total": 35,
    "pose_npy_shape": [35, 17, 3],
    "training_pose_sampling": {"training_target_fps": 5.0, "training_pose_n_frames": 35, ...},
    ...
  },
  "pairwise_probs": {
    "Angry|Paining": 0.73,
    "Defence|Paining": 0.41,
    ...
  },
  "meta_result": {
    "decision": "pain",
    "p_pain": 0.68,
    "meta_class_probs": {"Paining": 0.68, "non_pain": 0.32}
  },
  "timing_seconds": {"pose_extraction_and_render": 45.1, "pairwise_stgcn_inference": 12.3, ...}
}
```

## Required model weights

| Model | Default path | Notes |
|-------|-------------|-------|
| ViTPose-H (apt36k) | `models/pose_est/vitpose-h-apt36k.pth` | Same checkpoint family as `dataset_construction` pose extraction; `--vitpose-model` / `--vitpose-dataset` / `--vitpose-arch` override for other weights |
| YOLO8x | `models/yolo/yolov8x.pt` | `pip install ultralytics` auto-downloads |
| AudioSep | `audio/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt` | HF Space: Audio-AGI/AudioSep |
| CLAP audio (HTSAT) | `audio/AudioSep/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt` | Bundled with AudioSep vendor tree; used as default in `clap_encoder.py` |
| CatEmotionModel | `models/audio_emotions/best_model_final.pth` | Trained in this repo |
| PANNs Cnn14 | `audio/audio-emotion-classifier/src/audio_classifier_utils/pretrained_weights/Cnn14_16k_mAP=0.438.pth` | Required by CatEmotionModel |
| 9 × P4PoseSTGCN | `runs/pose-models/p4_pairwise_ensemble_bundle_20260428/…/training/best_weights.pth` | Trained pairwise models |
| LogReg meta | `runs/pose-models/stgcn_dlc_stack_20260504_235843/run_05/model.pkl` | Stacking run_05 |
| ffmpeg | on `PATH` | **Recommended:** `pose_assembler` re-encodes the overlay to H.264 after OpenCV writes mp4v so IDE/Quick Look previews work; without ffmpeg the file is still valid in VLC |

## Module overview

| File | Role |
|------|------|
| `pipeline.py` | CLI entry point + orchestrator |
| `audio_clip_probe.py` | RMS / duration / `likely_silent` for extracted clip WAV (UI + diagnostics) |
| `audio_branch.py` | AudioSep separation + CatEmotionModel classification |
| `video_branch.py` | ViTPose + ST-GCN stacking orchestration |
| `pose_assembler.py` | Frame-level VitInference → `(T, 17, 3)` numpy + pose-overlay video |
| `stgcn_loader.py` | Load 9 pairwise ST-GCN models + LogReg meta-model; run pairwise + meta inference |
| `timer.py` | `StepTimer` context manager |
| `artifact_io.py` | Run-dir creation, file copy, JSON serialisation |
