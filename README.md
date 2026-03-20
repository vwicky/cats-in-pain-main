# Cats in Pain - Dataset Preparation Pipeline

This repository contains scripts and notebooks for building a cat pain behavior dataset from YouTube videos.  
The main workflow filters candidate videos, downloads selected items, extracts short snippets, and stores structured metadata in JSON/JSONL files.

## What this repository contains

- `streaming_video_pipeline.py` - main end-to-end pipeline (download -> detect/process -> save snippets -> write logs/metadata).
- `video_downloader.py` - earlier downloader/processor script for local directory processing.
- `metadata.jsonl`, `metadata_cleaned.jsonl`, `gemini_labeled_videos.jsonl` - core metadata and labeling artifacts.
- `logs/` - pipeline and crawler logs retained for reproducibility.
- Notebooks (`*.ipynb`) - experiments, analysis, labeling, and model work.

## Requirements

- Python 3.10+
- `ffmpeg` available in your system PATH (used by audio/video operations)
- Python dependencies from `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Model and data prerequisites

- Place YOLO weights at `yolo11m.pt` (or update defaults in `streaming_video_pipeline.py`).
- Ensure the audio preclassifier model exists at:
  - `models/audio_preclassifier/voting_classifier_with-add-data.pkl`
- Ensure input metadata files exist (default paths):
  - `gemini_labeled_videos.jsonl`
  - `logs/crawled_cat_candidates.jsonl`

## Run the main streaming pipeline

Example command:

```bash
python streaming_video_pipeline.py \
  --cookies-from-browser chrome \
  --video-workers 2 \
  --frame-stride 4 \
  --yolo-imgsz 512
```

Useful optional flags:

- `--limit N` to run only the first N items (smoke testing).
- `--metadata-file PATH` to write metadata somewhere else.
- `--pipeline-log-file PATH` to write run logs somewhere else.
- `--snippets-output-dir PATH` to change snippet output location.

## Data policy for GitHub publishing

This repository intentionally keeps:

- `logs/`
- `*.jsonl`
- `*.json`

This repository ignores generated/downloaded media artifacts (for example `downloads/`, `downloaded_snippets/`, `dataset_snippets/`, `crawled_downloads/`, and `crawled_snippets/`, as well as video/audio files like `*.mp4` and `*.mp3`).

