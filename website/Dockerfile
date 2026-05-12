# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Expected build context: repository root (parent of `website/`).
COPY website/requirements.txt /tmp/website-requirements.txt
RUN pip install --no-cache-dir -r /tmp/website-requirements.txt

# Default working directory for API/worker when repo is bind-mounted to /workspace
WORKDIR /workspace/website
ENV PYTHONPATH=/workspace/website

# Override in compose: `uvicorn ...` or `python worker/runner.py`
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
