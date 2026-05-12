#!/usr/bin/env bash
# Install DeepLabCut + SuperAnimal stack in the *current* conda env (Python 3.11 recommended).
# PyTables from conda-forge avoids broken ``pip install tables==3.8.0`` builds; DeepLabCut is
# installed with ``--no-deps`` so pip does not try to replace PyTables.
#
# Official install guide (conda, ``deeplabcut[tf]``, model zoo, TF notes):
#   https://deeplabcut.github.io/DeepLabCut/docs/installation.html
#
# Apple Silicon (arm64): ``tensorflow-macos`` 2.13, then ``keras==2.12.0`` with ``pip install
# --no-deps`` (cannot be in the same ``pip install`` as TF: metadata requires keras>=2.13.1).
# Keras 2.13+ wheels dropped top-level ``keras.legacy_tf_layers`` that TF's v1 layer shims
# lazy-import (fixes ``No module named 'keras.legacy_tf_layers'`` / similar ``tf_keras`` paths).
#
# Usage (from repo root):
#   conda create -n dlc-superanimal python=3.11 -y
#   conda activate dlc-superanimal
#   bash dataset_construction/install_superanimal_deps.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$ROOT/dataset_construction/requirements-superanimal.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; install Miniconda/Mambaforge first." >&2
  exit 1
fi

# PyTables + NumPy in one conda solve. Broad ``numpy<2`` + default (libmamba) solver has
# triggered ``Segmentation fault: 11`` during "Solving environment" on some Macs — prefer
# ``mamba`` or ``--solver=classic`` and pin NumPy 1.26.x (``<2`` for TF wheels; ``scipy`` needs ``>=1.26``).
echo "==> conda-forge: pytables + numpy=1.26.4 (binary stack for DLC / tensorflow-macos)"
CF=( -y -c conda-forge "numpy=1.26.4" pytables )
if command -v mamba >/dev/null 2>&1; then
  echo "    (using mamba)"
  mamba install "${CF[@]}"
elif conda install --help 2>&1 | grep -q -- '--solver'; then
  echo "    (using conda --solver=classic)"
  conda install --solver=classic "${CF[@]}"
else
  conda install "${CF[@]}"
fi

echo "==> pip: TensorFlow (platform-specific; see script header + DLC installation docs)"
OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  pip uninstall -y tensorflow tf-keras 2>/dev/null || true
  pip install "tensorflow-macos==2.13.1" "tensorflow-metal==1.0.1"
  # Override Keras in a second step (resolver rejects keras==2.12 with TF in one transaction).
  pip install --no-deps --force-reinstall "keras==2.12.0"
else
  # Align with DLC docs (last TF tested 2.12 on non-Windows). Adjust for your CUDA stack.
  pip install "tensorflow==2.12.1"
  pip install --no-deps --force-reinstall "keras==2.12.0"
fi

echo "==> pip: base packages (see requirements-superanimal.txt)"
pip install -r "$REQ"

echo ""
echo "NOTE (Apple arm64 + DLC 2.3): Pip may still report tensorflow-macos vs keras/typing_extensions."
echo "      Keras 2.12.0 is intentional (legacy_tf_layers). typing_extensions>=4.10 is for PyTorch;"
echo "      TF's wheel metadata wants <4.6 — ignore if the smoke test below passes."
echo ""

echo "==> pip: DeepLabCut (no deps — PyTables already from conda)"
pip install "deeplabcut==2.3.11" --no-deps

echo "==> pip: DeepLabCut runtime dependencies except tables"
# One line avoids broken line-continuations (a missing trailing \\ after imgaug makes the
# shell run ``imageio-ffmpeg`` as a command: "command not found").
pip install "dlclibrary>=0.0.6" "filterpy>=1.4.4" "ruamel.yaml>=0.15.0" "imgaug>=0.4.0" "imageio-ffmpeg" "numba>=0.54" "matplotlib>=3.3,!=3.7.0,!=3.7.1,<3.9" "networkx>=2.6" "pandas!=1.5.0,>=1.0.1,<2.3" "scikit-image>=0.17" "scikit-learn>=1.0" "scipy>=1.9" "statsmodels>=0.11" "torch" "tqdm" "pyyaml" "Pillow>=7.1" "tensorpack" "tf_slim"

echo "==> pip: re-pin NumPy<2 (scipy/matplotlib/torch often upgrade to NumPy 2.x; TF 2.13 then crashes)"
pip install "numpy==1.26.4" --force-reinstall --no-deps

echo ""
echo "NOTE: Pip may warn that deeplabcut wants tables==3.8.0 but PyTables from conda is newer."
echo "      That is intentional: conda-forge ships working binaries; pip's pinned tables often"
echo "      fails to build. HDF5 reading/writing for DLC still uses the conda PyTables stack."
echo ""

echo "==> smoke test (first import can take ~1 minute — TensorFlow)"
python -u -c "import numpy as np; print('numpy', np.__version__); import cv2, pandas; print('cv2, pandas: ok'); import tensorflow as tf; print('tensorflow', tf.__version__); from tensorflow.python.layers import normalization; _ = normalization.BatchNormalization; print('tensorflow.python.layers.normalization: ok'); import deeplabcut; print('deeplabcut', deeplabcut.__version__); from deeplabcut.modelzoo.api import superanimal_inference; print('deeplabcut.modelzoo.superanimal_inference: ok')"

echo "Done. Run: python dataset_construction/06_pose_extraction_superanimal.py --limit 3"
