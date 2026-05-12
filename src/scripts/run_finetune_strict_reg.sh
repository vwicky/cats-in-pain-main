#!/usr/bin/env bash
# Fusion fine-tuning (F1–F5) with stricter regularization than defaults.
# Default training is focal loss + no WeightedRandomSampler (see finetuning_after_ssl.py).
# --label-smoothing only applies if you pass --loss ce.
# Edit SSL_RUN_DIR to match your ssl_sweep_* folder containing P*_*/pretrained_encoder.pth
set -euo pipefail
cd "$(dirname "$0")/.."

SSL_RUN_DIR="${SSL_RUN_DIR:-runs/ssl_sweep_20260331_212713}"

exec python finetuning_after_ssl.py \
  --mode finetune \
  --ssl-run-dir "$SSL_RUN_DIR" \
  --num-workers 0 \
  --weight-decay 0.05 \
  --label-smoothing 0.12 \
  --lr-encoder 4e-5 \
  --lr-head 8e-5 \
  "$@"
