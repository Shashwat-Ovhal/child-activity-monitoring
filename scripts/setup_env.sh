#!/usr/bin/env bash
set -e

echo "=== Setting up Child Activity Monitoring System Environment ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'"

# 1. Create Unified Virtual Environment
ENV_NAME="${1:-.venv}"
echo "Creating unified Python virtual environment at '${ENV_NAME}'..."
python3 -m venv "${ENV_NAME}"
source "${ENV_NAME}/bin/activate"
pip install --upgrade pip wheel setuptools

# 2. Install Unified Dependencies
echo "Installing all dependencies (Core, Vision, VLM)..."
pip install -r requirements.txt

mkdir -p third_party

# 3. Install SAM 2 (Segment Anything Model 2)
if [ ! -d "third_party/sam2" ]; then
    echo "Cloning and installing SAM 2..."
    git clone https://github.com/facebookresearch/sam2.git third_party/sam2
    cd third_party/sam2 && pip install -e . && cd ../..
fi

# 4. Install Grounding DINO
if [ ! -d "third_party/GroundingDINO" ]; then
    echo "Cloning and building Grounding DINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
    cd third_party/GroundingDINO && pip install -e . && cd ../..
fi

echo ""
echo "=== Environment successfully configured! ==="
echo "To activate:"
echo "    source ${ENV_NAME}/bin/activate"
echo "To run tests:"
echo "    python -m pytest tests/"
echo "To launch system:"
echo "    python src/app/main.py"
echo "============================================="
