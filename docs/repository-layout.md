# Repository layout

High-level directory map for the Cats-in-Pain codebase. Heavy data and checkpoints live under `data/` and `models/` locally (see [`.gitignore`](../.gitignore)); those folders are mostly empty in a fresh clone.

```
.
├── audio/
│   ├── audio-pre-classifier/        YAMNet (PyTorch) cat / non-cat gating (active)
│   └── audio-emotion-classifier/    10-class CatEmotionModel (PANN Cnn14 backbone)
├── video/
│   ├── gpt-fgs/                     GPT vision evaluation against the Feline Grimace Scale
│   ├── cnn-finetuning/              EfficientNet-B3 cropped-cat finetune
│   ├── easyViTPose/                 ViTPose pose extraction (incl. 06_pose_extraction.py)
│   ├── deep-lab-cut/                DeepLabCut SuperAnimal-Quadruped extraction & inference
│   └── pose-models/                 Pose-based classifiers (P0..P4 + ST-GCN; ST-GCN headline)
├── src/
│   ├── scrapers/
│   │   ├── tiktok/                  TikTok candidate scraper + filter + downloader
│   │   ├── youtube/                 YouTube scraper
│   │   ├── dailymotion/             DailyMotion scraper
│   │   └── data_pipeline_v2/        Multi-source orchestrator (default audio backend: YAMNet v3)
│   ├── dataset_construction/        Cleaning + labeling pipeline (numbered steps + manifests)
│   ├── human_validation/            Gradio app + GPT description scripts for human-in-the-loop
│   ├── inference/                   End-to-end CLI inference on a single video
│   ├── utils/
│   │   └── pipeline_helpers/        Shared class-name + persistence helpers
│   └── scripts/                     Cross-cutting utility scripts (frame extraction, crops, …)
├── website/                         FastAPI + worker + React/Vite MVP
├── runs/                            Consolidated experiment outputs grouped by subproject
│   ├── pose-models/
│   ├── cnn-finetuning/
│   ├── audio-pre-classifier/
│   ├── data-pipeline-v2/
│   ├── tiktok-pipeline/
│   ├── dailymotion/
│   └── gpt-fgs/                     LLM eval predictions + reports
├── models/                          Checkpoints (git-ignored except README)
│   ├── yolo/                        yolo11m, yolov8l/m/x.pt
│   ├── audio_emotions/              CatEmotionModel best_model_final.pth + intermediates
│   └── pose_est/                    ViTPose-H (apt36k) weights
├── data/                            All training data (git-ignored except README)
│   ├── dataset/                     Snippets, downsampled videos, full frames, manifests, DLC h5
│   ├── audio_cat-classification/    NAYA / Catmood raw audio
│   ├── audio_preclassification/     ESC-50, AudioSet, etc.
│   ├── video-data/                  Misc raw video collections
│   └── img_examples/
├── docs/                            Guides (quickstart, script index, research overview, …)
├── logs/                            Pipeline runtime logs (kept in git)
└── _archive/                        Deprecated code & data (kept for review, never published)
    ├── top_level_scripts/           V1 training scripts (binary_classifier, supcon, …)
    ├── top_level_notebooks/         Old analysis & training notebooks
    ├── audio_preclassifier_v2/      Legacy sklearn audio preclassifier
    ├── audio_progress/              Old training intermediate dumps
    ├── audio_progress_3/            (same)
    ├── audio_separation_checkpoints/ Old crm/u-net checkpoints + voting classifier
    ├── old_runs/                    V1 training run dirs
    ├── old_metadata_files/          Legacy *.jsonl / *.json metadata
    ├── plots/                       Stand-alone PNGs / SVGs
    ├── separated/                   Old separated-audio dump
    └── misc/                        cloud/, deploy/, tools/, snippet_analytics, downloads/, …
```

Return to the [root README](../README.md) or [research overview](research-overview.md) for how these pieces fit together.
