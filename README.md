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

The complete pipeline utilizes deep learning models for detection (Grounding DINO), persistent segmentation (SAM 2), and temporal visual-language reasoning (Moondream 2 / 3). Follow the guide below based on your hardware setup:
- **Part 1:** [Running Locally on PC / Workstation with NVIDIA GPU (Live Webcam & Video)](#part-1-running-on-local-pc-with-nvidia-gpu-recommended)
- **Part 2:** [Running via Rented Cloud GPU on Vast.ai (Fast-Track Rental Guide)](#part-2-running-via-rented-cloud-gpu-on-vastai)

---

### Part 1: Running on Local PC with NVIDIA GPU (Recommended)

This is the primary setup for running the complete system locally with real GPU acceleration and a **live visual display window**.

> [!TIP]
> **OS Recommendation:** **Linux (Ubuntu 20.04/22.04/24.04)** or **Windows with WSL2 (Ubuntu)** is strongly recommended. Grounding DINO and SAM 2 compile and link CUDA extensions natively in Linux environments.

#### Step 1: Verify Hardware & Prerequisites
Ensure you have an NVIDIA GPU (GTX 1650/1660/1080 or RTX series with 6 GB+ VRAM) and Python 3.10+:
```bash
nvidia-smi
python3 --version  # Must be 3.10 or higher
```

#### Step 2: Clone Repository & Create Virtual Environment
Isolating dependencies inside a dedicated virtual environment ensures that the project packages and CUDA extensions do not conflict with system-level packages:

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
Upgrade pip and run the automated setup script:

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
cd third_party/sam2 && pip install --no-build-isolation -e . && cd ../..

# Install Grounding DINO
# Note: Use BUILD_CUDA_EXT=0 to prevent C++ compiler/Ninja mismatches on newer Python versions
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
cd third_party/GroundingDINO && BUILD_CUDA_EXT=0 pip install --no-build-isolation -e . && cd ../..
# Alternatively, install the pre-built wheel directly: pip install groundingdino-py
```
> [!NOTE]
> `src/detection/child_detector.py` includes an automatic backward-compatibility patch for `BertModel.get_extended_attention_mask`, so newer versions of Hugging Face `transformers` (v4.40+) load Grounding DINO without error.

#### Step 4: Download Model Weights & Configs
Run the model downloader script to fetch the pre-trained checkpoints for Grounding DINO and SAM 2.1 Hiera:
```bash
bash scripts/download_models.sh
```
This downloads:
- Grounding DINO Swin-T weights (`models/grounding_dino/groundingdino_swint_ogc.pth`) & config (`configs/GroundingDINO_SwinT_OGC.py`)
- SAM 2.1 Hiera Small weights (`models/sam2/sam2.1_hiera_small.pt`) & config (`configs/sam2.1/sam2.1_hiera_s.yaml`)
- Moondream 2 open weights (`vikhyatk/moondream2`) load automatically via Hugging Face Transformers on first run with zero cloud authentication.

#### Step 5: Configure `configs/settings.yaml`
Open `configs/settings.yaml` to choose between a **Live Webcam** or a **Local Video File**:

```yaml
camera:
  # Option A (Live USB Webcam): set source to 0 or 1
  # Option B (Recorded Video): set source to video path, e.g., "tests/videos/sample.mp4"
  source: 0
  width: 1280
  height: 720
  fps: 30
  loop_video: true                  # Loops recorded video indefinitely

detection:
  prompt: "child"                   # Open-vocabulary prompt to locate child
  box_threshold: 0.35
  text_threshold: 0.25
  config_path: "configs/GroundingDINO_SwinT_OGC.py"
  weights_path: "models/grounding_dino/groundingdino_swint_ogc.pth"
  device: "cuda"                    # <-- "cuda" for GPU acceleration

tracking:
  model_cfg: "configs/sam2.1/sam2.1_hiera_s.yaml"
  checkpoint_path: "models/sam2/sam2.1_hiera_small.pt"
  segment_length_seconds: 3.0
  device: "cuda"                    # <-- "cuda" for GPU acceleration

temporal:
  buffer_duration_seconds: 3.0      # Rolling buffer duration
  buffer_target_fps: 8              # FPS at which child crops are captured
  crop_margin: 0.20                 # 20% spatial padding around child
  sample_keyframes_count: 8         # Frames extracted per VLM inference
  motion_change_threshold: 0.15     # Motion threshold to trigger VLM call
  heartbeat_interval_seconds: 8.0   # Heartbeat refresh interval

vlm:
  engine_type: "moondream"
  model_id: "vikhyatk/moondream2"   # 100% open weights, no API key needed
  device: "cuda"                    # <-- "cuda"
  precision: "float16"              # <-- "float16" (optimal for GTX & RTX cards)

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

#### Step 7: Launch the Application (Live Visual Display)
Run the complete real-time pipeline:
```bash
python src/app/main.py
```
- A window titled **"Child Activity Monitoring System (V1)"** will open showing your webcam or video stream at 30 FPS.
- The system will detect the child, track them with persistent SAM 2 segmentation, continuously sample keyframes, query Moondream on GPU, smooth predictions to eliminate flicker, and display a real-time activity banner at the top (e.g., *"The child is sitting on the floor"*).
- Press `q` while focused on the video window to cleanly exit.

**Optional: Record Annotated Video to MP4:**
You can also record the visual output with all overlays directly to a video file:
```bash
python src/app/main.py --save-video my_child_activity_recording.mp4
```

---

### Part 2: Running via Rented Cloud GPU on Vast.ai

If you do not have a dedicated NVIDIA GPU locally or want an on-demand cloud machine, follow this fast-track guide.

#### Should you rent it?
**Yes, definitely.** 
An **RTX 3090 (24 GB VRAM)** is an ideal GPU for this system. The entire pipeline (Grounding DINO + SAM 2 + Moondream 2/3 FP16) consumes roughly **7–9 GB VRAM**, so the 3090 has plenty of headroom and will run inference at maximum speed.

**Is 1.5 hours enough?**
**Yes, easily**—on a cloud datacenter connection (typically 500 Mbps – 1 Gbps):
- Environment setup & building CUDA extensions: **~5–8 minutes**
- Downloading model checkpoints: **~2–4 minutes**
- That leaves you with **over 1 hour and 15 minutes** of uninterrupted testing and execution.

---

#### Critical Warnings Before Starting (To Save Time & Credits)
1. **Pick the Right Docker Image on Vast.ai:** 
   Choose the **PyTorch** template (`pytorch/pytorch` with CUDA 12.1 or 12.4). **Do not** start from a blank Ubuntu image, or you will waste 15–20 minutes just installing CUDA and PyTorch.
2. **Cloud Has No Webcam:**
   A cloud GPU does not have a physical camera attached (`/dev/video0`). You **must** provide an MP4 video or use a sample video clip, and run in `--headless` mode.
3. **Download Results Before 1.5 Hours Expire:**
   When credits run out, Vast.ai will stop the instance. Make sure you copy any generated logs or saved video clips to your local PC before the timer runs out.

---

#### Step-by-Step 15-Minute Fast-Track Guide

##### 1. Rent the Instance on Vast.ai
1. In the Vast.ai search filters:
   - Select **1x RTX 3090** (or RTX 3080/4090/A4000).
   - Under **Template / Image**, select **PyTorch** (e.g. `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` or latest PyTorch 2.x devel image).
   - Disk space: set to **30 GB** or more.
2. Click **Rent** and wait ~1–2 minutes until the status shows **Running**.

##### 2. Connect via SSH
Copy the SSH command from your Vast.ai dashboard and paste it into your local terminal:
```bash
ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080
```

##### 3. Install Required System Libraries (1 minute)
Cloud containers often lack basic video and OpenGL libraries needed by OpenCV:
```bash
apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 ffmpeg
```

##### 4. Clone Repo & Set Up Environment (5 minutes)
```bash
# Clone the repository
git clone https://github.com/Shashwat-Ovhal/child-activity-monitoring.git
cd child-activity-monitoring

# Create and activate virtual environment
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# Upgrade pip & install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run automated setup for SAM 2 & Grounding DINO
bash scripts/setup_env.sh
```

##### 5. Download Model Weights (2–3 minutes)
```bash
bash scripts/download_models.sh
```

##### 6. Prepare a Test Video & Configure `configs/settings.yaml`
Since a cloud GPU has no webcam, upload an MP4 video or use a test video clip:
```bash
mkdir -p tests/videos
wget -q --show-progress -O tests/videos/sample.mp4 "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

# Configure settings.yaml for RTX 3090 & video file automatically
python3 -c "
import yaml
with open('configs/settings.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['camera']['source'] = 'tests/videos/sample.mp4'
cfg['camera']['loop_video'] = True
cfg['detection']['device'] = 'cuda'
cfg['tracking']['device'] = 'cuda'
cfg['vlm']['model_id'] = 'vikhyatk/moondream2'
cfg['vlm']['device'] = 'cuda'
cfg['vlm']['precision'] = 'float16'
with open('configs/settings.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('settings.yaml configured!')
"
```

##### 7. Verify GPU & Launch Application
```bash
# Verify GPU detection
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

# Run tests
python -m pytest tests/

# Launch pipeline and record annotated video with bounding boxes & activity banner
python src/app/main.py --headless --save-video /workspace/child-activity-monitoring/annotated_output.mp4 2>&1 | tee full_monitoring_log.txt
```
*(Let it run for 15–30 seconds, then press `Ctrl + C`)*.

##### 8. Download Video & Logs to Local PC Before Credits Expire
Open a **new PowerShell / Terminal window on your local PC** and run:
```powershell
# Save commands history on remote server first:
# ssh -p <PORT> root@<IP> "history > /workspace/child-activity-monitoring/commands_history.txt"

# Download video and logs to your local Desktop:
scp -P <PORT> -i "$env:USERPROFILE\.ssh\id_ed25519" root@<IP>:/workspace/child-activity-monitoring/annotated_output.mp4 "$env:USERPROFILE\Desktop\"
scp -P <PORT> -i "$env:USERPROFILE\.ssh\id_ed25519" root@<IP>:/workspace/child-activity-monitoring/full_monitoring_log.txt "$env:USERPROFILE\Desktop\"
```

##### 9. Destroy Instance
Go to your **[Vast.ai Console](https://cloud.vast.ai)** and click **Destroy Instance** to immediately stop credit consumption.

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
