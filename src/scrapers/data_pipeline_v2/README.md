# Cat Behavior Data Pipeline v2

Self-contained pipeline under `src/scrapers/data_pipeline_v2/`: multilingual query generation, rule + GPT metadata filters, `yt-dlp` download, YOLO single-cat tracking with subsampled inference (`frame_skip`), and **audio gating** with **YAMNet** from [`audio_preclassification_v3`](../audio/audio-pre-classifier/) by default (`audio.backend: yamnet`, aggregated P(cat) vs `audio.cat_prob_threshold`, default **0.65** in `pipeline.yaml`).

## What is new vs v1

- GPT-4o-mini generates diverse multilingual search queries (seeds × languages) instead of only hardcoded English queries.
- Explicit GPT metadata stage; rules live in `config/gpt_filter_rules.md` (loaded verbatim as the system prompt).
- YOLO weights default to `yolov8x.pt`; `yolo.frame_skip` speeds up tracking on MPS/GPU while keeping sequential `cap.read()`. Optional `yolo.processing_workers` uses a process pool (initializer loads YOLO+audio once per worker). Set `yolo.parallel_yolo_device` to `cpu`, `mps`, `cuda`, or `auto` (same priority as sequential: CUDA → MPS → CPU—not “safe CPU” by default). **MPS + multiple workers** on one GPU is untested—prefer `processing_workers: 1` for a single Apple GPU unless you know what you are doing. With `processing_workers > 1`, metadata JSONL append order follows **task completion**, not video order; downstream steps should sort or key by `video_id` / timestamps if order matters.
- Audio defaults to **YAMNet** (`audio.backend: yamnet`). Set `audio.backend: sklearn` to use the legacy stack: v2 binary pickle when `audio.prefer_v2: true` (`audio.v2_model_path` under `_archive/audio_preclassifier_v2/runs/.../best_model.pkl`), else v1 ESC multi-class (`audio.model_path`). YAMNet needs `torch`, `torchaudio`, `soundfile`, and the `torch-audioset` git dependency from `requirements.txt` (same pin as `audio_preclassification_v3`).
- Per-stage JSONL logs, plots, and `final_report.txt` under each run directory.
- Resume any stage with `--resume-run-dir` (required when `--start-from` is not `search`).
- Query cache at `logs/v2_generated_queries.jsonl` with `logs/v2_generated_queries.meta.json` hash; regenerates when `search:` config changes.

## Quick start

From the **repository root**:

```bash
pip install -r src/scrapers/data_pipeline_v2/requirements.txt
```

Full pipeline:

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml
```

Resume from download (after search + filters finished in a previous run):

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml \
  --start-from download \
  --resume-run-dir src/scrapers/data_pipeline_v2/runs/pipeline_v2_run_20260408_120000
```

Run only through GPT filter (no download):

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml \
  --stop-after gpt_filter
```

If `--start-from` is not `search`, you **must** pass `--resume-run-dir` pointing at an existing run folder; otherwise the process exits with an error and lists the three most recent run directories.

On resume, configuration is **`merge(run_dir/config_used.yaml, your --config)`** — nested keys from your current `pipeline.yaml` (and env-based API keys from `load_config`) **override** the frozen snapshot so edits to YAML take effect without starting a new run.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `YOUTUBE_API_KEY` | Optional. YouTube Data API v3; if unset, search uses `yt-dlp` (warning logged). |
| `OPENAI_API_KEY` | Query expansion + GPT filter; if unset, hardcoded English seeds are used and GPT filter passes through all tag-filtered rows with low confidence. |
| `COOKIE_FILE` | Netscape cookies for `yt-dlp` (optional). |

## Editing GPT filter rules

Edit `src/scrapers/data_pipeline_v2/config/gpt_filter_rules.md`. It is the **only** source for filtering criteria (loaded verbatim). No code change required for rule tweaks.

## Charts caveat

Plots that break down by `behavioral_category` are labeled: **Category = first matching query, not verified behavior.**

## Manual stage experiments

See `notebooks/try_stages.ipynb`: loads `REPO/.env` via `python-dotenv`, runs real `run_gpt_filter`, and can exercise sklearn v2 inference or the pipeline `AudioClassifier` (defaults to YAMNet v3 when `audio.backend: yamnet` in config).

## Outputs (default paths, repo-relative)

- Snippets: `data/dataset/snippets_v2/`
- Video metadata: `data/dataset/metadata_v2.jsonl`
- Global pipeline log: `logs/v2_pipeline_log.jsonl`
- Runs: `src/scrapers/data_pipeline_v2/runs/<run_name>_<timestamp>/` with `stage_*` folders, `final_report.txt`, `pipeline.log`, `pipeline_funnel.png`

YOLO `.pt` paths are repo-root-relative in `config/pipeline.yaml`. For `audio.backend: sklearn`, audio pickles are loaded as documented above. For `audio.backend: yamnet`, the pretrained YAMNet weights are pulled via `torch-audioset` on first use.
