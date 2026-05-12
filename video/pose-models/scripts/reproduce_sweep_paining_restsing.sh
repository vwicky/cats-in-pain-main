#!/usr/bin/env bash
# Reproduces the approach from
#   model_training_v2/runs/p4_sweep_paining_restsing_20260423_235721
# Single pair Paining,Resting; default 4×4×3×3×2 = 288 hparam cells;
# each cell: up to 1000 epochs, early stop patience 100, grad clip 1.0 (sweep script defaults),
# use_class_weights, ViT-pose QC filter, RAM pose cache, cat_id group stratified split.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
NAME="p4_sweep_paining_restsing_$(date +%Y%m%d_%H%M%S)"
exec python3 model_training_v2/scripts/sweep_p4_pairwise_hparams.py \
  --config "model_training_v2/config_p4_hparam_sweep_baseline.yaml" \
  --pair "Paining,Resting" \
  --experiment-name "$NAME" \
  --epochs 1000 \
  --early-stop-patience 100 \
  "$@"
