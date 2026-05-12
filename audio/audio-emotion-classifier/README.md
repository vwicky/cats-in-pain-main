# Cat Audio Emotion Classifier (10-class CNN)

A 10-class fine-tuned `CatEmotionModel` built on top of a PANNs `Cnn14` backbone
(`Cnn14_16k_mAP=0.438.pth`). Predicts one of:
`Angry`, `Defence`, `Fighting`, `Happy`, `HuntingMind`, `Mating`, `MotherCall`,
`Paining`, `Resting`, `Warning`.

This is the **audio side** of the labeling pipeline (Step 04 of dataset
construction). The 4-class **YAMNet** cat / non-cat **pre-classifier** lives in
[`../audio-pre-classifier/`](../audio-pre-classifier/).

## Layout

```
audio-emotion-classifier/
├── src/
│   └── audio_classifier_utils/      # PANNs Cnn14 backbone + CatEmotionModel + AudioConfig
│       ├── audio_config.py
│       ├── data_utils.py
│       ├── dataset.py
│       ├── models/                  # cnn14.py, panns.py, pytorch_utils.py
│       └── pretrained_weights/      # Cnn14_16k_mAP=0.438.pth (PANNs base weights)
├── inference/
│   └── 04_audio_classification.py   # batch inference on separated_audios/
└── scripts/
    └── audio_emotion_inference_embeddings_pca.py   # standalone PCA / embedding export
```

## Checkpoints

`models/audio_emotions/best_model_final.pth` (top-level `models/` dir).

## Usage

Run from the **repository root** so `sys.path` injection picks up the package:

```bash
python audio/audio-emotion-classifier/inference/04_audio_classification.py
python audio/audio-emotion-classifier/inference/04_audio_classification.py --dry-run
python audio/audio-emotion-classifier/inference/04_audio_classification.py --limit 100
```

Inputs default to `src/src/dataset_construction/separeted_audios/` (recursive scan
for `.wav` / `.mp3` / `.flac` / `.m4a`). Outputs go to
`src/src/dataset_construction/reports/separated_audios_emotion_predictions.jsonl`
plus distribution / confidence plots and a `emotion_labeling_report.txt`.

## Training

Training was performed in (now archived) Jupyter notebooks
`_archive/top_level_notebooks/audio_classifier.ipynb` and
`_archive/top_level_notebooks/cat_audio_extraction_model.ipynb`. The fine-tuned
weights at `models/audio_emotions/best_model_final.pth` are the artifact reused
here.
