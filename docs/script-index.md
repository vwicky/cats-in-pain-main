# Script index

Task-oriented entrypoints. Run **Python commands from the repository root** unless noted.

## Data collection

| Task | Command | Doc |
|------|---------|-----|
| Multi-source orchestrator | `python src/scrapers/data_pipeline_v2/src/pipeline.py --config src/scrapers/data_pipeline_v2/config/pipeline.yaml` | [data_pipeline_v2 README](../src/scrapers/data_pipeline_v2/README.md) |
| TikTok | `python src/scrapers/tiktok/src/pipeline.py --config src/scrapers/tiktok/config/pipeline.yaml` | [src/scrapers/tiktok/](../src/scrapers/tiktok/) |
| YouTube | `python src/scrapers/youtube/src/pipeline.py --config src/scrapers/youtube/config/pipeline.yaml` | [youtube README](../src/scrapers/youtube/README.md) |
| Dailymotion | `python src/scrapers/dailymotion/main.py --config src/scrapers/dailymotion/config/pipeline.yaml` | [src/scrapers/dailymotion/](../src/scrapers/dailymotion/) |

## Dataset construction and labeling

| Task | Command | Doc |
|------|---------|-----|
| Merge metadata | `python src/dataset_construction/00_merge_metadata.py` | [dataset_construction README](../src/dataset_construction/README.md) |
| Static detection | `python src/dataset_construction/01_static_detection.py` | same |
| GPT snippet descriptions | `python src/dataset_construction/02_gpt_description.py` | same |
| Cat ID clustering | `python src/dataset_construction/03_cat_id.py` | same |
| Audio emotion (separated audios) | `python audio/audio-emotion-classifier/inference/04_audio_classification.py` | [audio-emotion-classifier README](../audio/audio-emotion-classifier/README.md) |
| Final manifest | `python src/dataset_construction/05_final_dataset.py` | [dataset_construction README](../src/dataset_construction/README.md) |
| Human validation UI | `python src/human_validation/app.py` | [src/human_validation/](../src/human_validation/) |

## Inference

| Task | Command | Doc |
|------|---------|-----|
| End-to-end video inference | `python src/inference/pipeline.py --video path/to/clip.mp4` | [inference README](../src/inference/README.md) |
| Sliding-window inference | `python src/inference/pipeline.py --video path/to/clip.mp4 --split-window-sec 6 --split-step-sec 3` | same |

## Web MVP

| Task | Command | Doc |
|------|---------|-----|
| One-shot local stack (from repo root) | `bash website/launch.sh` | [website README](../website/README.md) |
| Install web + frontend deps | `bash website/launch.sh --install-deps` | same |
| Docker Compose | `cd website && docker compose up --build` | same |

## Training and evaluation (selected)

| Task | Command | Doc |
|------|---------|-----|
| YAMNet evaluation | `python audio/audio-pre-classifier/src/evaluate_v2_data.py --help` | [audio-pre-classifier README](../audio/audio-pre-classifier/README.md) |
| EfficientNet finetune | `python video/cnn-finetuning/src/train.py --config video/cnn-finetuning/config/default.yaml` | [cnn-finetuning README](../video/cnn-finetuning/README.md) |
| ViTPose extraction | `python video/easyViTPose/extraction/06_pose_extraction.py --help` | [easyViTPose README](../video/easyViTPose/README.md) |
| DeepLabCut extraction | `python video/deep-lab-cut/extraction/06_pose_extraction_superanimal.py --help` | [video/deep-lab-cut/](../video/deep-lab-cut/) |
| ST-GCN training | `python video/pose-models/run_stgcn_deeplabcut_train.py --config video/pose-models/config_stgcn_dlc.yaml` | [video/pose-models/](../video/pose-models/) |
| GPT FGS notebook | `jupyter notebook video/gpt-fgs/cat_pain_llm_eval.ipynb` | [video/gpt-fgs/](../video/gpt-fgs/) |

## Requirements layers

| Scope | File |
|------|------|
| Shared Python deps | [requirements.txt](../requirements.txt) |
| Docker-oriented top-level | [requirements-docker.txt](../requirements-docker.txt) (if used) |
| Web API + worker | [website/requirements.txt](../website/requirements.txt) |
| Inference audio branch | [src/inference/requirements-audiosep.txt](../src/inference/requirements-audiosep.txt) |
| Data pipeline v2 | [src/scrapers/data_pipeline_v2/requirements.txt](../src/scrapers/data_pipeline_v2/requirements.txt) |

Return to the [root README](../README.md) for repository layout and **models / data** expectations.
