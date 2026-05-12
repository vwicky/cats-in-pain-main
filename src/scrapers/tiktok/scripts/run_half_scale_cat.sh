#!/usr/bin/env bash
# New TikTok pipeline run: ~half the default search breadth, GPT hashtags must include
# "cat" (or gato/kot/neko/…), prior hashtag hints + video_id skip reduce duplicates.
#
# Usage (from repo root, with venv activated):
#   ./tiktok_pipeline/scripts/run_half_scale_cat.sh
#   ./tiktok_pipeline/scripts/run_half_scale_cat.sh --stop-after search
#   ./tiktok_pipeline/scripts/run_half_scale_cat.sh --start-from enrich --resume-run-dir tiktok_pipeline/runs/...
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH=tiktok_pipeline

exec python tiktok_pipeline/src/pipeline.py \
  --config tiktok_pipeline/config/pipeline_half_scale_cat.yaml \
  "$@"
