# Quickstart

All **Python CLIs** below assume the current working directory is the **repository root** (the folder that contains `src/`, `audio/`, `video/`, and `website/`).

## Python and tooling

- **Inference / scraping / most training:** Python **3.10+**
- **Website stack:** Python **3.11+** recommended; **Node 20+**; **PostgreSQL 16+**
- **`ffmpeg` / `ffprobe`** on `PATH`

## 1. Minimal install (inference + common deps)

```bash
pip install -r requirements.txt
pip install -r audio/audio-pre-classifier/requirements.txt
```

**CLI inference on one file** (video branch; see inference README for model paths):

```bash
python src/inference/pipeline.py --video path/to/clip.mp4
```

**Optional — audio branch** (AudioSep + related deps):

```bash
pip install -r src/inference/requirements-audiosep.txt
```

Details, flags, and artifact layout: [src/inference/README.md](../src/inference/README.md)

## 2. Multi-source data pipeline (collect + filter + download)

```bash
pip install -r src/scrapers/data_pipeline_v2/requirements.txt
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml
```

Env vars (e.g. `OPENAI_API_KEY`, `YOUTUBE_API_KEY`): [src/scrapers/data_pipeline_v2/README.md](../src/scrapers/data_pipeline_v2/README.md)

## 3. Dataset construction (manifests)

Start here: [src/dataset_construction/README.md](../src/dataset_construction/README.md)

Typical sequence from repo root:

```bash
python src/dataset_construction/00_merge_metadata.py
python src/dataset_construction/01_static_detection.py
python src/dataset_construction/02_gpt_description.py
python src/dataset_construction/03_cat_id.py
# Step 04: audio labeling on separated clips (see dataset README)
python audio/audio-emotion-classifier/inference/04_audio_classification.py
# Ensure metadata_clean_04.jsonl exists before step 05 (see dataset README)
python src/dataset_construction/05_final_dataset.py
```

## 4. Web MVP (API + worker + UI)

From **repository root**:

```bash
# First time only
bash website/launch.sh --install-deps

# Configure DB and app (see website README)
cp website/.env.example website/.env

# Start API (:8000), worker, and Vite (:5173)
bash website/launch.sh
```

Full prerequisites, manual three-process run, Postgres troubleshooting, and Docker: [website/README.md](../website/README.md)

## 5. Where next

- [Script index](script-index.md) — task → command → owning README
- [Root README](../README.md) — layout, models/data layout, archive policy
