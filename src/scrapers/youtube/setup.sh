#!/bin/bash
set -e

echo "=== Cat Behavior YouTube Scraper — Setup ==="

# Python version check
python3 --version | grep -E "3\.(10|11|12)" || {
    echo "ERROR: Python 3.10-3.12 required"; exit 1; }

# ffmpeg check
command -v ffmpeg >/dev/null 2>&1 || {
    echo "Installing ffmpeg..."
    sudo apt-get install -y ffmpeg; }

# CUDA check (warn only, not required)
nvidia-smi >/dev/null 2>&1 && echo "✅ GPU detected" || \
    echo "⚠ No GPU detected — YOLO will run on CPU (slower)"

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
mkdir -p logs runs tmp_downloads dataset/snippets models

# Environment file template
if [ ! -f .env ]; then
    cp -n .env.example .env 2>/dev/null || cat > .env << 'EOF'
YOUTUBE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
COOKIE_FILE=
EOF
    echo "⚠ Created .env — fill in your API keys before running"
fi

echo ""
echo "✅ Setup complete. To run:"
echo "   source .venv/bin/activate"
echo "   python src/pipeline.py --config config/pipeline.yaml"
