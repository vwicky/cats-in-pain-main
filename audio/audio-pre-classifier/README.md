# Audio preclassification v3 (PyTorch YAMNet + MPS)

This subproject evaluates a **PyTorch port of YAMNet** ([`torch_audioset`](https://github.com/w-hc/torch_audioset)) on cat vs. non-cat audio. The model outputs **521 AudioSet** class probabilities per short time window; we aggregate **cat-related** classes (e.g. Cat, Meow, Purr) and pool over time to produce a single **P(cat)** score per file.

## Requirements

- Python 3.10+
- PyTorch / torchaudio (install a build that matches your platform from [pytorch.org](https://pytorch.org))
- macOS: **Metal (MPS)** is used automatically when `torch.backends.mps.is_available()` is true; otherwise inference runs on **CPU**
- **FFmpeg** on PATH is recommended for MP3/M4A (used by librosa/audioread). Decoding avoids TorchCodec so you should not see long FFmpeg dylib errors from `torchaudio.load` on macOS.

```bash
cd audio_preclassification_v3
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `torch_audioset` package is installed from GitHub via that file. If you see `ModuleNotFoundError: No module named 'torch_audioset'`, activate your venv and run `pip install -r requirements.txt` again from `audio/audio-pre-classifier/`.

## Data layout vs. full V2 training

The V2 pipeline (`audio_preclassifier_v2`) merges **Kaggle datasets**, **human-labeled** clips, and **local** folders. The default path here points at `../data/audio_cat-classification/raw/`, which in many setups is **mostly cat audio** (e.g. NAYA emotion folders). To stress-test with **non-cat** samples, either:

- Pass extra roots: `--labeled-root cat=/path/to/cats --labeled-root non-cat=/path/to/non-cats`, or
- Provide a **manifest CSV** with columns `filepath,label` (`label` = `cat` / `non-cat` or `1` / `0`)

Environment override for the default data root:

```bash
export AUDIO_PRECLASS_V3_DATA_ROOT=/absolute/path/to/raw
```

## Evaluation CLI

From the repository root (so `src` imports resolve):

```bash
cd audio_preclassification_v3
PYTHONPATH=. python src/evaluate_v2_data.py --help
```

**Examples**

```bash
# Directory layout: data-root/cat/... and data-root/non-cat/...
PYTHONPATH=. python src/evaluate_v2_data.py --data-root /path/to/labeled_tree

# Multiple labeled roots (same as V2-style local_dirs)
PYTHONPATH=. python src/evaluate_v2_data.py \
  --labeled-root cat=../data/audio_cat-classification/raw/NAYA_DATA_AUG1X \
  --labeled-root non-cat=../src/human_validation/video_audio_human_validation/non-cat

# Manifest CSV
PYTHONPATH=. python src/evaluate_v2_data.py --manifest ./my_manifest.csv

# NAYA-style tree (all files are cats; emotion subfolders are not "non-cat")
PYTHONPATH=. python src/evaluate_v2_data.py \
  --data-root ../data/audio_cat-classification/raw/NAYA_DATA_AUG1X \
  --label-all cat
```

Each run writes a timestamped folder under `audio/audio-pre-classifier/runs/run_YYYYMMDD_HHMMSS/`:

| Artifact | Description |
|----------|-------------|
| `execution.log` | INFO logs (device, progress, errors) |
| `predictions.csv` | `filename`, `true_label`, `p_cat_score`, `predicted_label` |
| `metrics.json` | Precision, recall, F1, ROC-AUC (when both classes exist) |
| `confusion_matrix.png` | Seaborn heatmap |
| `score_distribution.png` | P(cat) histograms for cat vs. non-cat |

## P(cat) definition

- Audio is **16 kHz mono float32** in **[-1, 1]** (see `src/audio_utils.py`).
- YAMNet uses **overlapping 0.96 s** patches with **0.48 s hop** (see `src/yamnet_runner.py`).
- For each frame, cat-related class probabilities (sigmoid outputs) are combined with **`sum` or `max`** (`--aggregate-cat-classes`).
- The clip score is the **maximum** over frames (strongest cat evidence in the clip).
- **`--threshold`** (default 0.5) maps P(cat) to predicted `cat` / `non-cat` for metrics.

## `different/` MP4s — chunk, YAMNet, Gradio verification

Place `.mp4` files in [`different/`](different/) (or point `--video-dir` elsewhere, e.g. `../_archive/audio_preclassifier_v2/different`). The script extracts **16 kHz mono** WAV with ffmpeg, splits into **6 s** chunks (same idea as [`_archive/audio_preclassifier_v2/notebooks/inference_verify.ipynb`](../_archive/audio_preclassifier_v2/notebooks/inference_verify.ipynb)), scores each chunk with YAMNet, writes `inference_results.jsonl`, then opens a **Gradio** UI to label chunks (cat / not a cat / skip). Verdicts append to `verification_results.jsonl` under the same results directory (resume supported).

```bash
cd audio_preclassification_v3
pip install -r requirements.txt   # pydub, gradio
PYTHONPATH=. python src/run_different_videos.py --help
PYTHONPATH=. python src/run_different_videos.py
# Other folder:
PYTHONPATH=. python src/run_different_videos.py --video-dir ../_archive/audio_preclassifier_v2/different
```

Use `--no-gradio` for batch scoring only. Outputs default to `audio/audio-pre-classifier/video_inference/`.

## Module map

| Module | Role |
|--------|------|
| `src/config.py` | Paths, sample rate, device, defaults |
| `src/audio_utils.py` | Load, mono, resample, padding |
| `src/yamnet_runner.py` | YAMNet load, MPS inference, cat aggregation |
| `src/evaluate_v2_data.py` | CLI, manifests, metrics, plots |
| `src/video_chunk_pipeline.py` | ffmpeg extract, pydub chunking |
| `src/gradio_verify_yamnet.py` | Gradio verification UI |
| `src/run_different_videos.py` | End-to-end: `different/*.mp4` → YAMNet → Gradio |

## License note

YAMNet weights and class metadata follow the original **TensorFlow / AudioSet** terms bundled with `torch_audioset`; see that repository for details.
