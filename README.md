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

### Option A: Local CPU Development (Immediate, Zero-Download)
You can run and test the entire pipeline locally without downloading any model weights using the built-in synthetic stream and `MockVLMEngine`:

1. Install core dependencies:
   ```bash
   pip install -r requirements/requirements_core.txt
   ```
2. Run unit tests:
   ```bash
   pytest tests/
   ```
3. Run the application:
   ```bash
   python src/app/main.py
   ```

---

### Option B: GPU Execution (College PC / Cloud GPU)
When running on an NVIDIA GPU machine (Ubuntu or WSL2 recommended):

1. Run the environment setup script:
   ```bash
   bash scripts/setup_env.sh
   ```
2. Download model checkpoints:
   ```bash
   bash scripts/download_models.sh
   ```
3. In `configs/settings.yaml`, set:
   ```yaml
   detection:
     device: "cuda"
   tracking:
     device: "cuda"
   vlm:
     engine_type: "moondream"
     device: "cuda"
     precision: "float16" # Crucial for GTX cards (no bfloat16)
   ```
4. Launch the application:
   ```bash
   source env_vision/bin/activate
   python src/app/main.py
   ```

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
