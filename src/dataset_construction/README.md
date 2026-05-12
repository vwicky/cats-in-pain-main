# Dataset Construction Pipeline

Sequential cleaning and labeling steps for the Cats-in-Pain dataset.
Each step produces a new manifest in manifests/ that the next step
consumes. Original data files are never modified.

## Steps

| Script | Input manifest | Output manifest | Purpose |
|--------|---------------|-----------------|---------|
| 00_merge_metadata.py | metadata_v2.jsonl + tiktok_metadata.jsonl | metadata_merged.jsonl | Merge sources; adds `source` per record |
| 01_static_detection.py | metadata_merged.jsonl | metadata_clean_01.jsonl | Remove **strictly** static clips (anchor BGR vs frame 0; see script) |
| restore_manifest_keep_static.py | metadata_merged + static_scan_results | metadata_clean_01.jsonl | **Recovery:** keep “static” snippets; drop only missing files |
| 02_gpt_description.py | metadata_clean_01.jsonl | metadata_clean_02.jsonl | GPT quality + description per snippet |
| 03_cat_id.py | metadata_clean_02.jsonl | metadata_clean_03.jsonl | Assign cat identity across videos |
| audio/audio-emotion-classifier/inference/04_audio_classification.py | — | — | Cat emotion labels on WAV/MP3 under `separeted_audios/` (`CatEmotionModel`, `models/audio_emotions/best_model_final.pth`). Writes `reports/separated_audios_emotion_predictions.jsonl`, `emotion_labeling_report.txt`, plots; logs under `src/dataset_construction/logs/`. **Manifest merge** (`metadata_clean_03` → `metadata_clean_04`) not implemented yet. |
| 05_final_dataset.py | metadata_clean_04.jsonl | final_dataset_v2.jsonl | Merge, validate, export |

## Running

From REPO_ROOT, merge manifests once (or after upstream metadata changes), then run step 01:

```bash
python src/dataset_construction/00_merge_metadata.py
python src/dataset_construction/01_static_detection.py
python src/dataset_construction/01_static_detection.py --threshold 150.0
python src/dataset_construction/01_static_detection.py --dry-run
python src/dataset_construction/01_static_detection.py --workers 4
```

Review reports/static_flagged.txt before proceeding to step 02.
**Strict detection:** every decoded frame is compared to the **first** frame in **BGR**;
the run stops as soon as mean abs diff exceeds `static_detection.variance_threshold`
(≈0.1–0.5 for lossy H.264; not 0). Tune using `static_detection_histogram.png`.

**Project rule:** do not delete raw dataset files; manifests and reports only ([`.cursor/rules/never-delete-data.mdc`](../.cursor/rules/never-delete-data.mdc)).

If automatic static removal was too aggressive, rebuild the clean manifest while **keeping**
snippets flagged as static (still drop only **missing** files):

```bash
python src/dataset_construction/restore_manifest_keep_static.py
```

This overwrites `manifests/metadata_clean_01.jsonl` with `cleaning_stage: "01_missing_only"`.

`metadata_merged.jsonl` lists TikTok rows from `data/dataset/tiktok_metadata.jsonl` with
`"source": "tiktok_metadata"` and pipeline-v2 rows with `"source": "metadata_v2"`.

TikTok snippet files live under `data/dataset/tiktok_snippets/` (`{snippet_id}.mp4`).
`sources.snippets_dirs` in `config.yaml` must include that directory (TikTok files are
not under `snippets_v2/`).

### Step 04: audio classification (separated audios)

Step 04 lives in `audio/audio-emotion-classifier/` (the home of `CatEmotionModel`).
See [`audio/audio-emotion-classifier/README.md`](../../audio/audio-emotion-classifier/README.md).

Fine-tuned weights default to `models/audio_emotions/best_model_final.pth`. PANNs
backbone weights are expected at
`audio/audio-emotion-classifier/src/audio_classifier_utils/pretrained_weights/Cnn14_16k_mAP=0.438.pth`.

From the **repository root**:

```bash
python audio/audio-emotion-classifier/inference/04_audio_classification.py
python audio/audio-emotion-classifier/inference/04_audio_classification.py --dry-run
python audio/audio-emotion-classifier/inference/04_audio_classification.py --limit 100
```

Scans `src/dataset_construction/separeted_audios/` recursively for `.wav` / `.mp3` (and `.flac` / `.m4a`).
