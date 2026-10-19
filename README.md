# Child Real-Time Activity Monitoring System (V1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0 / BSL-1.1](https://img.shields.io/badge/license-Apache--2.0%20%2F%20BSL--1.1-green.svg)](THIRD_PARTY_LICENSES.md)
[![Hardware: CPU / GTX / RTX](https://img.shields.io/badge/hardware-CPU%20%7C%20GTX%20%7C%20RTX-orange.svg)](NOTES.md)

An open-vocabulary visual perception platform that continuously monitors a room, detects a child, persistently tracks them across time (even amidst adults), and generates stable, real-time natural language descriptions of their ongoing physical activity (e.g., *"The child is walking toward the table."*, *"The child is sitting on the floor."*).

---

## 1. System Architecture

```text
Camera (30 FPS)
   │
   ▼
Grounding DINO ("child") ───────► Initial Child Bounding Box
   │
   ▼
SAM 2.1 Video Tracker ──────────► Persistent Mask & Child ROI (Track ID)
   │
   ▼
Child ROI Cropper (20% margin)
   │
   ▼
Temporal Frame Buffer (2–5s @ 8 FPS)
   │
   ▼
Representative Keyframe Sampler (5–12 frames)
   │
   ▼
Motion Change Gating ───────────► Trigger VLM only on movement change / heartbeat
   │
   ▼
Moondream 3 (MoE VLM) ──────────► Natural Language Reasoning
   │
   ▼
Temporal Anti-Flicker Smoother ─► Hysteresis & State Transition Filter
   │
   ▼
Event Engine ───────────────────► Structured Event Logging & Inactivity Alerts
   │
   ▼
Live UI Overlay (20–30 FPS)
```

---

## 2. Key Highlights

- **Open-Ended Vocabulary:** Avoids closed-set action classifiers. The Vision-Language Model dynamically describes unscripted, real-world behaviors.
- **Moondream 3 MoE Architecture:** Uses Moondream 3 (~9B total parameters, ~2B active parameters) for lightweight inference, fast latency, and low VRAM requirements compared to dense 4B/7B models.
- **Representative Keyframe Sampling:** Instead of overloading the VLM with every camera frame, a rolling buffer extracts 5–12 representative keyframes, querying the VLM asynchronously.
- **Hardware Agility:** Runs on CPU (with zero-resource `MockVLMEngine` for rapid UI & logic development) and switches seamlessly to a college or rented GPU for full deep learning inference.
- **Strictly Permissive Compliance:** Completely free of AGPL (YOLO, BoxMOT) and non-commercial (OpenPose) dependencies.

---

## 3. Directory Layout

```text
child-monitoring/
├── configs/
│   └── settings.yaml          # All operational parameters & thresholds
├── models/                    # Downloaded model weights (gitignored)
├── src/
│   ├── camera/                # Webcam / video stream capture
│   ├── detection/             # Grounding DINO child detector
│   ├── tracking/              # SAM 2.1 persistent video tracker
│   ├── temporal/              # Rolling buffer & keyframe sampler
│   ├── vlm/                   # Pluggable VLM engines (Moondream, Mock)
│   ├── activity/              # Temporal reasoning & anti-flicker smoothing
│   ├── events/                # State transition & inactivity alerts
│   └── app/                   # UI overlay & main execution loop
├── tests/                     # Unit test suite & ground truth evaluation
├── scripts/                   # Download & environment setup automation
├── requirements/              # Core, vision, and VLM dependency manifests
├── THIRD_PARTY_LICENSES.md    # Upstream license register
└── README.md
```

---

## 4. Quickstart Guide

The complete pipeline utilizes deep learning models for detection (Grounding DINO), persistent segmentation (SAM 2), and temporal visual-language reasoning (Moondream 3). Follow the guide below based on your hardware setup:
- **Part 1:** [Running on Local PC with NVIDIA GPU (Ubuntu / WSL2 Recommended)](#part-1-running-on-local-pc-with-nvidia-gpu-recommended)
- **Part 2:** [Running via Rented Cloud GPU on Vast.ai](#part-2-running-via-rented-cloud-gpu-on-vastai)

---

### Part 1: Running on Local PC with NVIDIA GPU (Recommended)

This is the primary setup for running the complete system locally with real GPU-accelerated inference.

> [!TIP]
> **OS Recommendation:** **Linux (Ubuntu 20.04/22.04/24.04)** or **Windows with WSL2 (Ubuntu)** is strongly recommended. Grounding DINO and SAM 2 build C++/CUDA extensions (`MultiScaleDeformableAttention`) which compile natively and smoothly in a Linux environment.

#### Step 1: Verify Hardware & Prerequisites
Ensure you have an NVIDIA GPU with the proper drivers installed, along with Python 3.10+:
```bash
nvidia-smi
python3 --version  # Must be 3.10 or higher
```

#### Step 2: Clone Repository & Create Virtual Environment
Isolating dependencies inside a dedicated virtual environment ensures that the project packages and CUDA extensions do not conflict with system-level packages.

```bash
git clone https://github.com/Shashwat-Ovhal/child-activity-monitoring.git
cd child-activity-monitoring

# Create isolated virtual environment
python3 -m venv .venv

# Activate the virtual environment:
# On Linux / WSL2 / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
```

#### Step 3: Install Dependencies & CUDA Extensions
Upgrade pip and run the automated setup script, which installs PyTorch with CUDA, project libraries, and builds Grounding DINO & SAM 2:

```bash
# Automated setup (installs requirements.txt and compiles SAM 2 & Grounding DINO)
bash scripts/setup_env.sh
```

*(Manual alternative)*:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# Install SAM 2
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
cd third_party/sam2 && pip install -e . && cd ../..
# Install Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
cd third_party/GroundingDINO && pip install -e . && cd ../..
```

#### Step 4: Download Model Weights & Configs
Run the model downloader script to fetch the pre-trained checkpoints for Grounding DINO and SAM 2.1 Hiera:
```bash
bash scripts/download_models.sh
```
This downloads:
- Grounding DINO Swin-T weights (`models/grounding_dino/groundingdino_swint_ogc.pth`) & config (`configs/GroundingDINO_SwinT_OGC.py`)
- SAM 2.1 Hiera Small weights (`models/sam2/sam2.1_hiera_small.pt`) & config (`configs/sam2.1/sam2.1_hiera_s.yaml`)
- Moondream 3 loads automatically from Hugging Face on its first run (or you can pre-cache it).

#### Step 5: Configure `configs/settings.yaml`
Open `configs/settings.yaml` in your favorite text editor (VS Code, Nano, Vim, etc.). This single file controls all operational parameters.

```bash
# Open in VS Code or nano:
nano configs/settings.yaml
```

**Where and how to configure key parameters for GPU execution:**

```yaml
camera:
  # 0 or 1 for live USB webcam, or string path to a test video:
  source: 0                         # e.g., "tests/videos/sample.mp4" if no webcam is connected
  width: 1280
  height: 720
  fps: 30
  loop_video: true

detection:
  prompt: "child"                   # Open-vocabulary prompt to locate child
  box_threshold: 0.35
  text_threshold: 0.25
  config_path: "configs/GroundingDINO_SwinT_OGC.py"
  weights_path: "models/grounding_dino/groundingdino_swint_ogc.pth"
  device: "cuda"                    # <-- SET TO "cuda" for GPU acceleration

tracking:
  model_cfg: "configs/sam2.1/sam2.1_hiera_s.yaml"
  checkpoint_path: "models/sam2/sam2.1_hiera_small.pt"
  segment_length_seconds: 3.0
  device: "cuda"                    # <-- SET TO "cuda" for GPU acceleration

temporal:
  buffer_duration_seconds: 3.0      # Rolling buffer duration
  buffer_target_fps: 8              # FPS at which child crops are captured
  crop_margin: 0.20                 # 20% spatial padding around child
  sample_keyframes_count: 8         # Frames extracted per VLM inference
  motion_change_threshold: 0.15     # Motion threshold to trigger VLM call
  heartbeat_interval_seconds: 8.0   # Heartbeat refresh interval

vlm:
  engine_type: "moondream"          # <-- "moondream" for real VLM, or "mock" for dummy test
  model_id: "moondream/moondream3-preview"
  device: "cuda"                    # <-- SET TO "cuda"
  precision: "float16"              # <-- "float16" (optimal for GTX 1650/1660/1080 and RTX series)

activity:
  smoothing_window_size: 4
  min_state_streak: 2
  inactivity_timeout_seconds: 180   # Alert if child is inactive/static > 3 minutes

display:
  show_bounding_box: true
  show_mask: true
  show_activity_banner: true
```

#### Step 6: Verify GPU Detection & Run Tests
Verify that PyTorch recognizes your NVIDIA GPU:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Run the unit test suite:
```bash
python -m pytest tests/
```

#### Step 7: Launch the Application
Run the complete real-time pipeline:
```bash
python src/app/main.py
```
- A window titled **"Child Activity Monitoring System (V1)"** will open showing the camera feed.
- The system will detect the child, track them with persistent SAM 2 segmentation, continuously sample keyframes, query Moondream 3, smooth predictions to eliminate flicker, and display real-time activity banners.
- Press `q` while focused on the video window to cleanly exit.

---

### Part 2: Running via Rented Cloud GPU on Vast.ai

If you do not have a dedicated NVIDIA GPU locally or have limited VRAM, you can rent an affordable GPU on **[Vast.ai](https://vast.ai)** (or RunPod / Lambda) for pennies per hour.

#### Step 1: Rent an Instance on Vast.ai
1. Log into your [Vast.ai Console](https://cloud.vast.ai).
2. Choose a template:
   - **Recommended Template:** `PyTorch` (Ubuntu 22.04 with CUDA 12.1 or higher).
3. Select GPU specs:
   - **Minimum VRAM:** 8 GB to 16 GB (e.g., RTX 3060, RTX 3080, RTX 4090, or A4000).
   - **Disk Allocation:** Allocate at least **30 GB** to comfortably store CUDA packages and model checkpoints.
4. Click **Rent** and wait for the instance to transition to `Running`.

#### Step 2: Connect via SSH
Copy the SSH connection command provided in your Vast.ai instance dashboard and run it in your local terminal:
```bash
ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080
```

#### Step 3: Clone Repository & Setup Environment
Once connected to the remote instance shell:
```bash
# 1. Install OS dependencies for OpenCV and video handling
apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 ffmpeg

# 2. Clone repository
git clone https://github.com/Shashwat-Ovhal/child-activity-monitoring.git
cd child-activity-monitoring

# 3. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Run automated environment setup
bash scripts/setup_env.sh

# 5. Download model weights
bash scripts/download_models.sh
```

#### Step 4: Configure for Cloud / Headless Execution
Since a cloud GPU instance has no physical webcam and no desktop monitor attached:

1. **Provide a test video:** Upload a test MP4 video (or place one at `tests/videos/sample.mp4`).
2. **Update `configs/settings.yaml`:**
   ```yaml
   camera:
     source: "tests/videos/sample.mp4"   # <-- Point to video file instead of webcam index
     loop_video: true

   detection:
     device: "cuda"

   tracking:
     device: "cuda"

   vlm:
     engine_type: "moondream"
     device: "cuda"
     precision: "float16"
   ```

#### Step 5: Run Verification & Headless Execution
1. Run tests to confirm the entire environment is intact:
   ```bash
   python -m pytest tests/
   ```

2. Launch the pipeline in **headless mode** (no GUI window required):
   ```bash
   python src/app/main.py --headless
   ```
   *Tip:* If you want to use virtual framebuffers for rendering visuals remotely, you can also use `xvfb-run`:
   ```bash
   apt-get install -y xvfb
   xvfb-run -a python src/app/main.py --headless
   ```
   The system will process the video, log child detection, run SAM 2 tracking, execute Moondream 3 reasoning, and print real-time activity events to stdout and log files.

---

## 5. Milestone Verification Matrix

| Milestone | Target | Status |
|---|---|---|
| **M1** | Camera Stream & File Looping | ✅ Implemented (`src/camera/camera_stream.py`) |
| **M2** | Grounding DINO Child Detection | ✅ Implemented (`src/detection/child_detector.py`) |
| **M3** | SAM 2 Persistent Tracking | ✅ Implemented (`src/tracking/child_tracker.py`) |
| **M4** | 2–5s Rolling Buffer & Frame Sampling | ✅ Implemented (`src/temporal/`) |
| **M5** | Moondream 3 Integration | ✅ Implemented (`src/vlm/moondream_engine.py`) |
| **M6** | Multi-Frame Temporal Activity Fusion | ✅ Implemented (`src/activity/activity_fusion.py`) |
| **M7** | Temporal Anti-Flicker Smoothing | ✅ Implemented (`src/activity/temporal_smoothing.py`) |
| **M8** | Event Engine & Inactivity Tracking | ✅ Implemented (`src/events/event_engine.py`) |

---

## 6. Disclaimer
This project is an assistive monitoring and behavioral observation system. It does not provide medical diagnostic conclusions.
