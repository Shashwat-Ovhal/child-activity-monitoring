#!/usr/bin/env bash
set -e

echo "=== Child Monitoring System — Model Downloader ==="

# 1. Create directory structure
mkdir -p models/grounding_dino
mkdir -p models/sam2
mkdir -p models/moondream
mkdir -p configs/sam2.1

# 2. Grounding DINO Swin-T Checkpoint & Config
echo "Downloading Grounding DINO Swin-T weights and config..."
if [ ! -f "models/grounding_dino/groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress -O models/grounding_dino/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "Grounding DINO weights already present."
fi

if [ ! -f "configs/GroundingDINO_SwinT_OGC.py" ]; then
    wget -q --show-progress -O configs/GroundingDINO_SwinT_OGC.py \
        https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
else
    echo "Grounding DINO config already present."
fi

# 3. SAM 2.1 Hiera Small Checkpoint & Config
echo "Downloading SAM 2.1 Hiera Small weights and config..."
if [ ! -f "models/sam2/sam2.1_hiera_small.pt" ]; then
    wget -q --show-progress -O models/sam2/sam2.1_hiera_small.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
else
    echo "SAM 2.1 weights already present."
fi

if [ ! -f "configs/sam2.1/sam2.1_hiera_s.yaml" ]; then
    wget -q --show-progress -O configs/sam2.1/sam2.1_hiera_s.yaml \
        https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
else
    echo "SAM 2.1 config already present."
fi

# 4. Moondream 3 Notes
echo "Note: Moondream 3 (moondream/moondream3-preview) will automatically download from Hugging Face on first run,"
echo "or you can pre-cache it using: python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('moondream/moondream3-preview', trust_remote_code=True)\""

echo "=== Model download complete! ==="
