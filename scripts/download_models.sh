#!/usr/bin/env bash
set -e

echo "=== Child Monitoring System — Model Downloader ==="

# 1. Create directory structure
mkdir -p models/grounding_dino
mkdir -p models/sam2
mkdir -p models/moondream
mkdir -p configs/sam2.1

# 2. Grounding DINO Swin-T Checkpoint
echo "Downloading Grounding DINO Swin-T weights..."
if [ ! -f "models/grounding_dino/groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress -O models/grounding_dino/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "Grounding DINO weights already present."
fi

# 3. SAM 2.1 Hiera Small Checkpoint
echo "Downloading SAM 2.1 Hiera Small weights..."
if [ ! -f "models/sam2/sam2.1_hiera_small.pt" ]; then
    wget -q --show-progress -O models/sam2/sam2.1_hiera_small.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
else
    echo "SAM 2.1 weights already present."
fi

# 4. Optional Pre-download for Qwen3-VL-4B-Instruct
echo "Note: Qwen3-VL-4B-Instruct will automatically download from Hugging Face on first run,"
echo "or you can pre-cache it using: huggingface-cli download Qwen/Qwen3-VL-4B-Instruct"

echo "=== Model download complete! ==="
