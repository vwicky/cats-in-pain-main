# YouTube pipeline (data_pipeline_v2) — second run

## What changed (aligned with TikTok second run)

1. **300 GPT seed queries**  
   `seed_queries_per_category: 60` → 5 categories × 60 = **300** English seed queries before multilingual expansion.

2. **More languages per seed**  
   `languages_per_query: 15` (was 5): each seed is translated into up to **15** languages from `target_languages` (first 15 entries in the list).

3. **Broader language list**  
   Added: Polish, Turkish, Hindi, Vietnamese, Dutch, Indonesian, Czech, Romanian (20 entries total; the first **15** are used with current `languages_per_query`).

4. **Less duplicate / generic wording from GPT**  
   - Stronger system/user prompts (distinct queries, varied angles).  
   - `temperature` **0.55** for seed generation.  
   - **Case-insensitive dedupe** per behavioral category, capped at **n** queries per category.

5. **No duplicate videos vs earlier runs (search output)**  
   Before collecting new candidates, `seen_ids` is seeded from **`collect_skip_video_ids()`**: IDs in `data/dataset/metadata_v2.jsonl`, `logs/v2_pipeline_log.jsonl` (successful / already_exists downloads), and optional `data/dataset/final_dataset.jsonl`. Those videos are not written to `stage_1_search/candidates.jsonl`.  
   Download still enforces the same skip set.

6. **Query cache**  
   Any change under `search:` in `pipeline.yaml` changes the config hash; the pipeline regenerates `logs/v2_generated_queries.jsonl` (and meta) instead of using a stale cache.

7. **Random subsample of expanded queries (optional)**  
   If `search.random_sample_queries` is a positive integer and the expanded list is longer, the pipeline randomly picks that many queries **after** loading from cache / generation and **before** calling YouTube. Set `search.random_sample_seed` to an integer for a reproducible draw.  
   CLI: `--search-random-sample N` and `--search-random-sample-seed SEED` override YAML.  
   Artifacts: `stage_1_search/search_query_sample.json`, `stage_1_search/sampled_queries.jsonl`.

## Cost & time

- **Many more** OpenAI calls: 300 seeds + one translation call per seed (each asks for 15 languages). Expect higher API cost and longer search stage than before.

## Launch — full new run

From the **repository root**:

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml
```

## Resume

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml \
  --start-from search \
  --resume-run-dir src/scrapers/data_pipeline_v2/runs/<run_folder_name>
```

### Resume search with only 300 random queries (from full expanded cache)

Uses the cached expanded list in `logs/v2_generated_queries.jsonl` (no regeneration unless you change `search:` hash), then samples 300 queries. Reproducible with a fixed seed:

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml \
  --start-from search \
  --resume-run-dir src/scrapers/data_pipeline_v2/runs/<run_folder_name> \
  --search-random-sample 300 \
  --search-random-sample-seed 42
```

Stages: `search` → `tag_filter` → `gpt_filter` → `download` → `process`.

If `--start-from` is not `search`, `--resume-run-dir` is **required**.

### New run: reuse a large candidate pool, sample another 300, continue from tag filter

You need a **single JSONL** that still contains **all** discovered candidates (e.g. from a run where search wrote the full list, or a copy you saved). If you only kept the post–search-sample file, that pool is only as large as that step produced.

1. Pick a **new** run directory name and sample 300 rows into `stage_1_search/candidates.jsonl`:

```bash
python src/scrapers/data_pipeline_v2/scripts/sample_candidates_for_new_run.py \
  src/scrapers/data_pipeline_v2/runs/<SOURCE_RUN>/stage_1_search/candidates.jsonl \
  src/scrapers/data_pipeline_v2/runs/<NEW_RUN_NAME> \
  -n 300 --seed 43
```

Use a **different `--seed`** than last time if you want a disjoint random subset (overlap is still possible by chance). Omit `--seed` for a different draw every time.

2. Run the pipeline **starting at tag filter** (search is skipped; it loads your sampled file):

```bash
python src/scrapers/data_pipeline_v2/src/pipeline.py \
  --config src/scrapers/data_pipeline_v2/config/pipeline.yaml \
  --start-from tag_filter \
  --resume-run-dir src/scrapers/data_pipeline_v2/runs/<NEW_RUN_NAME>
```

Optional: copy `config_used.yaml` from an old run into `<NEW_RUN_NAME>/` if you want the merge snapshot to match; otherwise your `--config` alone applies.

## Environment

- `OPENAI_API_KEY` — seed + translation + GPT filter.  
- `YOUTUBE_API_KEY` — optional; without it search uses yt-dlp fallback (warning logged).
