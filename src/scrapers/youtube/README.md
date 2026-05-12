# Cat Behavior YouTube Scraper

Standalone YouTube scraping pipeline for the Cats-in-Pain dataset.
Runs independently on any machine with Python 3.10+ and ffmpeg.

## Quick start

```bash
git clone https://github.com/your-username/youtube_scraper
cd youtube_scraper
bash setup.sh
# .env is created from .env.example — edit API keys
source .venv/bin/activate
python src/pipeline.py --config config/pipeline.yaml
```

## Cross-machine deduplication

`global_video_registry.jsonl` tracks every video seen across all
machines. Before starting a run:

```bash
git pull origin main   # get latest registry
```

After a run completes:

```bash
git add global_video_registry.jsonl
git commit -m "registry: add N videos from pc_gtx1650"
git push origin main
```

To merge registries from two machines manually:

```bash
python src/dedup.py merge \
  --registry-a global_video_registry.jsonl \
  --registry-b /path/to/other/global_video_registry.jsonl \
  --output global_video_registry.jsonl
```

Or:

```bash
python merge_registry.py \
  --registry-a global_video_registry.jsonl \
  --registry-b /path/to/other/global_video_registry.jsonl \
  --output global_video_registry.jsonl
```

## Configuration

All settings in `config/pipeline.yaml`.
Set `machine_id` to identify which machine produced each record.
API keys via environment variables or `.env` file (`YOUTUBE_API_KEY`, `OPENAI_API_KEY`, `COOKIE_FILE`).

## Resume a run

```bash
python src/pipeline.py \
  --config config/pipeline.yaml \
  --start-from download \
  --resume-run-dir runs/pipeline_run_20260415_120000
```

## Docker (optional)

Build and run (GPU optional; mount API keys via env):

```bash
docker build -t youtube-scraper .
docker run --rm -e YOUTUBE_API_KEY -e OPENAI_API_KEY -v $(pwd)/global_video_registry.jsonl:/app/global_video_registry.jsonl youtube-scraper
```

TensorFlow may use CPU unless you install a GPU-enabled build; Ultralytics YOLO uses PyTorch with CUDA when available.
