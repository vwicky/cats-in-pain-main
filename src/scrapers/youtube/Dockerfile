# Optional: Ubuntu + CUDA runtime for PyTorch YOLO; YAMNet runs via TensorFlow (CPU or install GPU TF separately).
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt setup.sh ./
RUN chmod +x setup.sh

# Non-interactive: skip sudo in setup; ffmpeg already installed
RUN python3 -m venv .venv && . .venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && mkdir -p logs runs tmp_downloads dataset/snippets models

COPY . .
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "src/pipeline.py", "--config", "config/pipeline.yaml"]
