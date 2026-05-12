#!/usr/bin/env bash
# TikTok pipeline: search only Pain/Vet hashtags (sick/injured/vet/pain/paining cats).
# Other behavioral categories are skipped — see search.behavior_categories in the config.
#
# Usage (repo root, venv on):
#   ./tiktok_pipeline/scripts/run_pain_vet_only.sh
#   ./tiktok_pipeline/scripts/run_pain_vet_only.sh --stop-after search
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH=tiktok_pipeline

exec python tiktok_pipeline/src/pipeline.py \
  --config tiktok_pipeline/config/pipeline_pain_vet_only.yaml \
  "$@"
