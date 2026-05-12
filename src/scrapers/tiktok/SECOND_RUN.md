# TikTok pipeline — second run (config bump + dedup)

## What changed

1. **GPT hashtag volume: 300 “searches”**  
   `seed_queries_per_category` is **60** (5 categories × 60 = **300** distinct hashtag strings). The previous default was 6 per category (30 total).

2. **More languages**  
   `languages_per_query` is **15** (was 5), so `query_language` cycles across more entries from `target_languages`. **Eight** languages were added: Polish, Turkish, Hindi, Vietnamese, Dutch, Indonesian, Czech, Romanian (plus the original 12).

3. **Fewer duplicate hashtags from GPT**  
   - Stronger system/user prompts (distinct tags, no trivial variants, niche/regional variety).  
   - `temperature` **0.55** (was 0.4).  
   - Case-insensitive **dedupe** inside each category’s hashtag list, then trim to `n` per category.

4. **No duplicate *videos* vs earlier runs**  
   - **Search stage:** `seen_ids` is pre-filled with `collect_skip_video_ids()` — same IDs as in `data/dataset/tiktok_metadata.jsonl`, `src/scrapers/tiktok/logs/pipeline_log.jsonl` (successful downloads), and optional `dataset/tiktok_final_dataset.jsonl`. Those videos are **not** added to `candidates.jsonl` again.  
   - **Download stage:** unchanged — still skips/reuses if a `video_id` is already known.

5. **Query cache**  
   Changing `search:` in YAML changes the config hash; the pipeline will **regenerate** `src/scrapers/tiktok/cache/tiktok_generated_queries.jsonl` instead of reusing the old cache.

## Launch (full new run)

From the **repository root**, with `OPENAI_API_KEY` set (and cookies as you already use):

```bash
python src/scrapers/tiktok/src/pipeline.py \
  --config src/scrapers/tiktok/config/pipeline.yaml
```

This creates a **new** timestamped folder under `src/scrapers/tiktok/runs/`.

## Resume an interrupted run

Use the same run directory and the stage where you left off:

```bash
python src/scrapers/tiktok/src/pipeline.py \
  --config src/scrapers/tiktok/config/pipeline.yaml \
  --start-from <stage> \
  --resume-run-dir src/scrapers/tiktok/runs/<your_run_folder>
```

Stages (in order): `search` → `enrich` → `tag_filter` → `gpt_filter` → `download` → `process`.

## Notes

- Search is **heavy** now (300 hashtag URLs × Playwright). Expect long wall time.  
- If TikTok rate-limits, keep `download.sleep_interval` as configured or raise it slightly.  
- To force **regeneration** of GPT hashtags after tweaking only prompts, any `search:` change invalidates the cache; or delete `src/scrapers/tiktok/cache/tiktok_generated_queries.*` manually.
