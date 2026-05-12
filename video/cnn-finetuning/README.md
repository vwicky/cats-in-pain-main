# Cat Pain CNN Fine-tuning

## Quick start

1. Inspect dataset first (recommended):
   ```
   jupyter notebook video/cnn-finetuning/notebooks/inspect_dataset.ipynb
   ```

2. Run training with default config:
   ```
   python video/cnn-finetuning/src/train.py --config video/cnn-finetuning/config/default.yaml
   ```

3. Override any config key at runtime:
   ```
   python video/cnn-finetuning/src/train.py \
     --config video/cnn-finetuning/config/default.yaml \
     --run_name my_experiment \
     --batch_size 32 \
     --lr_head 5e-4 \
     --freeze_backbone_epochs 10
   ```

4. EfficientNet-B3 weights (~49MB) download automatically from
   PyTorch model zoo on first run. No manual download needed.

## Output structure
Each run saves to `video/cnn-finetuning/runs/{run_name}_{timestamp}/`.
See `final_report.txt` for a human-readable summary.
