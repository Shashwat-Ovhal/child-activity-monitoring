#!/usr/bin/env bash
set -e

echo "=== Setting up Child Activity Monitoring System Environments ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'"

# 1. Setup Vision Environment (Grounding DINO + SAM 2)
echo "Setting up env_vision..."
python3 -m venv env_vision
source env_vision/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements_vision.txt

mkdir -p third_party

# Install SAM 2
if [ ! -d "third_party/sam2" ]; then
    echo "Cloning and installing SAM 2..."
    git clone https://github.com/facebookresearch/sam2.git third_party/sam2
    cd third_party/sam2 && pip install -e . && cd ../..
fi

# Install Grounding DINO
if [ ! -d "third_party/GroundingDINO" ]; then
    echo "Cloning and building Grounding DINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
    cd third_party/GroundingDINO && pip install -e . && cd ../..
fi

deactivate

# 2. Setup VLM Environment (Moondream 3)
echo "Setting up env_vlm..."
python3 -m venv env_vlm
source env_vlm/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements_vlm.txt
deactivate

echo "=== Environments successfully configured! ==="
