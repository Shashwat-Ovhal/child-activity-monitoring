# Child Activity Monitoring System – Project Handover Guide

_Part 1 of our final-year capstone. Read the whole guide once before starting, then follow the phases in order._

# Part A – What this project is (the short version)

We are building a system that **watches a room through a camera, finds the child, and keeps writing in plain English what the child is doing right now**, for example:

- "The child is walking toward the table."
- "The child is sitting on the floor."
- "The child is eating."
- "The child is playing with a toy."

That sentence is shown live on the screen, next to the child.

**Key points:**

- It must describe **any** action, not only actions from a fixed list or dataset. That is why we use a Vision-Language Model (VLM) instead of a normal action classifier.
- It must run **locally on our own PC**, using **free, open-source models** only. No paid cloud APIs.
- It is the first building block of a bigger capstone: a monitoring system for children with autism and similar conditions (later: parent-voice commands, dashboards, etc.). **Those later parts are NOT in scope now.** Do not build them.
- We describe the product as **assistive monitoring / behaviour observation**. We never claim it diagnoses autism.

# Part B – How we will build it (in a nutshell)

The big idea: **a VLM is smart but slow, so we never run it on every camera frame.** We use fast, light models to _find and follow_ the child, and call the VLM only every 1–2 seconds on a short clip of that child to get the sentence.

Camera (30 FPS)  
↓  
Grounding DINO → finds the "child" (text prompt: "child")  
↓  
SAM 2.1 → follows that same child frame after frame (persistent ID)  
↓  
Child crop → cut out only the child's region  
↓  
2–5 second buffer of the child's crops  
↓  
Qwen3-VL-4B → "What is the child doing?" → one short sentence  
↓  
Smoothing → remove flickering between answers  
↓  
Overlay text on live video

**Why each piece:**

| Job                                   | Tool                                                      | Why                                                     |
| ------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| Find the child (not just "a person")  | **Grounding DINO**                                        | Detects objects from a text prompt like "child"         |
| Keep following the child              | **SAM 2.1 (Hiera Small)**                                 | Video tracker, keeps a stable child ID and mask         |
| Understand the action and describe it | **Qwen3-VL-4B-Instruct**                                  | Open-vocabulary video understanding, Apache-2.0 licence |
| Camera, frames, display               | **OpenCV**                                                | Standard and free                                       |
| Reference code for detect → track     | **Grounded-SAM-2** (GitHub: IDEA-Research/Grounded-SAM-2) | We copy its logic, we do not install it as our app      |

**Hardware:** NVIDIA GTX GPU with 16 GB VRAM. This is enough for all three models together if we are careful (details in Phase 1).

**Tools we are deliberately NOT using:** YOLO, BoxMOT (AGPL licences), OpenPose (non-commercial), old OpenPose + LSTM projects, and paid cloud VLM APIs.

**Later (V2, only after V1 works):** RTMPose for body keypoints, RTMDet as a faster detector, MMAction2 for specialised action recognition. Research references only: ASDMotion, CV-child-action-recognition, autism-related-behavior, VideoChat3.

# Part C – What "done" looks like for this part

V1 is complete when all of these are true:

1. A live webcam feed opens on screen.
2. When a child is in view, a box/mask appears on the child and stays on the **same** child while they move, even if adults are also in the room.
3. A text line updates about every 1–2 seconds with a sentence describing the child's current activity.
4. The text is stable (does not flicker between different answers every second).
5. Everything runs locally, offline, on our GPU.
6. We have measured and written down FPS, latency, VRAM use and accuracy (Phase 8).
7. The code is organised in the folder structure below, with a README and a licence file.

# Part D – Step-by-step guide

## Ground rules (read first)

- **Reuse, don't rewrite.** Clone the GitHub repos and borrow their code. Write only the glue code between them.
- **One environment per heavy stack.** Never install all libraries into one Python environment. They need different PyTorch/dependency versions and will break each other. We use two: env_vlm and env_vision.
- **Finish each milestone and test it before starting the next.** Save a short screen recording or screenshot as proof of each milestone.
- **Keep a log** (a simple NOTES.md): what you installed, versions, errors and how you fixed them.
- **Track licences from day one** in THIRD_PARTY_LICENSES.md. For every repo and every model checkpoint, record name, source URL and licence.
- **Do not claim diagnosis.** Wording everywhere (UI, README, report) is "activity and behaviour monitoring".
- **Privacy:** use only test videos of consenting people or public/sample videos. If a real child is filmed, a parent/guardian must consent. Keep all footage local.

## Target folder structure

child-monitoring/  
│  
├── env_vlm/ ← Python env for Qwen3-VL (not committed to git)  
├── env_vision/ ← Python env for Grounding DINO + SAM 2  
│  
├── third_party/ ← cloned repos (read-only reference + installs)  
│ ├── Qwen3-VL/  
│ ├── GroundingDINO/  
│ ├── sam2/  
│ └── Grounded-SAM-2/ ← reference only  
│  
├── models/ ← downloaded checkpoints  
│ ├── groundingdino_swint_ogc.pth  
│ └── sam2.1_hiera_small.pt  
│  
├── configs/  
│ └── settings.yaml ← camera index, prompts, buffer length, intervals  
│  
├── src/  
│ ├── camera/ ← capture.py (OpenCV frame stream)  
│ ├── detection/ ← child_detector.py (Grounding DINO wrapper)  
│ ├── tracking/ ← child_tracker.py (SAM 2 wrapper)  
│ ├── temporal/ ← buffer.py, smoothing.py  
│ ├── vlm/ ← vlm_engine.py, vlm_server.py (Qwen3-VL)  
│ ├── app/ ← main.py, overlay.py (display)  
│ └── utils/  
│  
├── tests/ ← test videos + small test scripts per milestone  
├── results/ ← benchmark numbers, screenshots, recordings  
├── THIRD_PARTY_LICENSES.md  
├── NOTES.md  
└── README.md

## Phase 0 – Preparation

**Goal:** know exactly what machine we are building on, and have a clean workspace.

1. **Check the GPU.** Open a terminal and run nvidia-smi. Note the exact GPU name, total VRAM, driver version and the CUDA version shown. Send this to me. It decides the exact PyTorch/CUDA install.
2. **Important GTX note:** GTX cards (Pascal/Turing generation) do not handle bfloat16 well and do not support FlashAttention-2. When loading models, use float16 and the standard (sdpa) attention. If you see dtype or attention errors, this is the first thing to check.
3. **Use Linux or WSL2.** SAM 2's own documentation recommends Linux (or WSL on Windows). If the PC runs Windows, install **WSL2 with Ubuntu** and the NVIDIA driver for WSL, then work inside Ubuntu for everything. Confirm the GPU is visible inside WSL with nvidia-smi.
4. **Install basics:** git, Python 3.10+, ffmpeg, a C++ build toolchain (build-essential) and the CUDA toolkit matching your PyTorch build (needed to compile extensions for Grounding DINO and SAM 2).
5. **Create the folder structure** above and run git init. Add env_\*, models/ and large videos to .gitignore.
6. **Create THIRD_PARTY_LICENSES.md** with an empty table: Component | Source | Version | Licence.
7. **Collect test material:** 5–10 short video clips (10–30 s each) showing a child (or a person acting as one, or public sample videos) doing different things: walking, sitting, eating, playing, running, standing. Include at least 2 clips with an adult also in the frame. Put them in tests/videos/. Write what actually happens in each clip in tests/ground_truth.csv (clip name, time range, true activity). You need this for Phase 8.

**Checkpoint:** nvidia-smi works inside the environment you will use, and test videos are ready.

## Phase 1 – Set up and test the VLM on its own (env_vlm)

**Goal:** prove Qwen3-VL-4B can describe a child's activity from a short video, before adding anything else.

1. Create env_vlm (Python 3.10+). Install a PyTorch build matching your CUDA version, then transformers **version 4.57.0 or newer** (Qwen3-VL requires it), plus accelerate and qwen-vl-utils.
2. Clone <https://github.com/QwenLM/Qwen3-VL> into third_party/. Read its README section on **video inference** with Hugging Face Transformers.
3. Download the model **Qwen/Qwen3-VL-4B-Instruct** from Hugging Face. Record its licence (Apache-2.0) in THIRD_PARTY_LICENSES.md.
4. Write a small script tests/test_qwen_video.py that loads the model with AutoModelForImageTextToText and AutoProcessor, in float16, on the GPU. Follow the README's example.
5. Feed it one of your test videos (3–5 s) with this prompt:

Look at the child in this video.  
Describe ONLY the child's current physical activity.  
Answer in one short sentence, starting with "The child is".  
Do not describe the room, clothes, or anything else.

1. Experiment with the visual-budget settings the Qwen3-VL docs describe: **video FPS, number of frames, and maximum visual tokens**. Lower values = faster and less VRAM, higher = more accurate. Try e.g. 2–4 FPS and 8–16 frames.
2. Record for each run: output sentence, time taken, peak GPU memory (torch.cuda.max_memory_allocated()).
3. If the 4B model does not fit or is too slow, fall back to **Qwen/Qwen3-VL-2B-Instruct** and note it. With 16 GB we expect 4B to work.

**Checkpoint:** for all your test clips, the model gives sensible one-sentence answers, and you have a table of latency and VRAM. Save it in results/phase1_vlm.md.

## Phase 2 – Set up the vision stack (env_vision)

**Goal:** get Grounding DINO and SAM 2 installed and each working on its own.

1. Create env_vision (Python 3.10+). SAM 2 needs **PyTorch ≥ 2.5.1 and torchvision ≥ 0.20.1**. Install a matching CUDA build of PyTorch first.
2. **Install SAM 2:** clone <https://github.com/facebookresearch/sam2> into third_party/, follow its INSTALL.md, and pip install -e .. Download the checkpoint **sam2.1_hiera_small.pt** into models/. (Use Small, not Large. It is the right size for our GPU.)
3. **Install Grounding DINO:** clone <https://github.com/IDEA-Research/GroundingDINO> into third_party/ and follow its README. It compiles a CUDA extension, so the CUDA toolkit version must match your PyTorch build. Download groundingdino_swint_ogc.pth into models/. Use the config groundingdino/config/GroundingDINO_SwinT_OGC.py.
4. **Test Grounding DINO alone** with from groundingdino.util.inference import load_model, predict. Load one test image, use the text prompt child and print the boxes and confidence scores. Try an image with an adult and a child: does it pick the child? Try prompts child, young child, kid and record which works best.
5. **Test SAM 2 alone** using build_sam2_video_predictor: give it a video (or a folder of frames), click/box the child on frame 0 using add_new_points_or_box, then run propagate_in_video and check the mask follows the child.
6. If a compile error appears for either repo, it is almost always a CUDA/PyTorch version mismatch. Fix the mismatch, don't hack the repo. Write down the fix in NOTES.md.
7. Fill in the licence table for Grounding DINO, SAM 2 and their checkpoints.

**Checkpoint:** both models run on their own, on your GPU, inside env_vision.

## Phase 3 – Milestone M1: Camera → find the child

**Goal:** live camera feed with a box around the child.

1. Write src/camera/capture.py: a small class that opens the webcam (or a video file for testing) with OpenCV and yields frames. It must also support reading from a video file so you can test repeatably without a live person.
2. Write src/detection/child_detector.py: a class that wraps Grounding DINO. It loads the model once and exposes detect(frame) -> list of (box, score) using the prompt child. Put the prompt and thresholds (box threshold, text threshold) in configs/settings.yaml.
3. Write a simple src/app/main.py that reads frames, runs the detector, and draws the box on the screen.
4. Test on your videos. Check: child found? Adult wrongly boxed? Empty room, does it correctly show nothing? Tune the thresholds. If an empty room produces false boxes, raise the thresholds.
5. Note the speed in FPS. Grounding DINO is not meant to run on every frame. That is fine, because the next phase only needs it occasionally.

**Checkpoint (M1):** recording showing the box on the child in live/test video, and a note on detection accuracy.

## Phase 4 – Milestone M2: Detect → track the same child

**Goal:** once the child is found, follow the same child continuously.

1. Open the reference repo Grounded-SAM-2 (third_party/Grounded-SAM-2). Study these scripts, which already do "detect with Grounding DINO → prompt SAM 2 → propagate → keep IDs":
   - grounded_sam2_tracking_demo.py
   - grounded_sam2_tracking_demo_with_continuous_id.py
   - grounded_sam2_tracking_demo_with_continuous_id_plus.py
2. Run one of them on your test videos first, as-is, to see the expected behaviour.
3. Write src/tracking/child_tracker.py by adapting that logic into a clean class:
   - start(frame, child_box): initialise SAM 2 state and add the box as the prompt
   - update(frame): return the child's current mask and bounding box
   - is_lost(): true if the mask disappears or becomes tiny
   - reset(): clear state
4. **Important design note:** SAM 2's standard video predictor is built for videos/frame sets, not an endless live stream. For a live camera, work in **short segments** (for example 2–4 seconds of frames at a time). At the start of each segment, run Grounding DINO to re-find the child and re-prompt SAM 2. This also fixes drift. Also look at SAM 2 streaming/live community variants if segments cause visible jumps, but get the segment method working first.
5. Add a rule: if tracking is lost (child left the frame or got hidden), run Grounding DINO again until the child reappears, then restart tracking.
6. Add a rule for multiple people: keep following the **same** child ID. Do not switch to another person. Test with a clip where an adult walks past the child.
7. Draw the mask/box and the track ID on the live video.

**Checkpoint (M2):** the box stays on the same child while they walk, sit and turn, including when an adult crosses in front. Record a video as proof.

## Phase 5 – Milestones M3 and M4: Child crop and temporal buffer

**Goal:** produce a clean short clip of just the child for the VLM.

1. **Crop (M3):** in src/temporal/buffer.py, each frame, take the child's bounding box, add margin (about 15–25%) so arms and legs are not cut off, and crop that region from the original frame. Resize to a consistent size (for example, shorter side 336–448 px). Keep the aspect ratio.
2. **Buffer (M4):** keep a rolling buffer (collections.deque) of the most recent crops covering **2–5 seconds**. Store frames at a reduced rate (e.g. 4–8 FPS), not the full 30. Buffer length and sampling rate come from settings.yaml.
3. Add a method get_clip() that returns the buffer as a list of frames or saves it to a temporary .mp4, in whichever format Qwen3-VL's video input accepts in your Phase 1 script.
4. Add a debug option to save clips to results/clips/ so you can watch exactly what the VLM sees. If the clips look bad (child tiny, cut off, blurry), fix that here, because the VLM can only describe what it sees.

**Checkpoint (M3 + M4):** saved clips clearly show only the child, centred, with full body visible.

## Phase 6 – Milestones M5 and M6: Connect the VLM and get stable sentences

**Goal:** a clean activity sentence from the buffer, shown on screen.

1. **Two environments, two programs.** The vision code lives in env_vision, Qwen3-VL in env_vlm. Do not merge them. Connect them with a small local service:
   - src/vlm/vlm_server.py (runs in env_vlm): a FastAPI app on localhost that loads Qwen3-VL once at start-up and has an endpoint like POST /describe that accepts a clip and returns the sentence.
   - src/vlm/vlm_engine.py (runs in env_vision): a tiny client that sends the clip to the server and returns the text.
   - Start the VLM server first, then the main app.
2. **Prompt.** Use the tested prompt from Phase 1. Keep it strict: one sentence, only the child's current physical action. Store it in settings.yaml.
3. **Run it periodically.** In main.py, every 1–2 seconds (configurable), take the latest clip and request a description. Run this in a **background thread**, so the camera display never freezes while the VLM thinks. The screen always shows the latest finished answer.
4. **Clean the output.** Write a small parser that trims whitespace, keeps only the first sentence, and rejects empty or obviously wrong output (e.g. text that does not mention a child action).
5. **Smoothing** (src/temporal/smoothing.py): VLMs may say "walking", "standing", "walking" in quick succession. Keep the last N answers (e.g. 3–5) and only change the displayed text when the new answer has repeated, or when it clearly differs for 2 consecutive runs. Also show "Child not in view" when tracking is lost, instead of repeating the last activity.
6. **Display** (src/app/overlay.py): draw the child's box/mask and a text line below or at the top of the frame with cv2.putText, with a dark background strip so text is readable.

**Checkpoint (M5 + M6):** running the app on your test videos shows live, readable, stable sentences matching what the child actually does most of the time. Record a demo video.

## Phase 7 – Milestone M7: Make it real-time and efficient

**Goal:** smooth live performance on our GPU.

1. **Measure first.** Log per-stage timing: capture, Grounding DINO, SAM 2, crop, VLM request, total. Log GPU memory during a 10-minute run. Save in results/phase7_perf.md.
2. **Rules of thumb:** the camera display must stay near 20–30 FPS; the VLM may answer only every 1–2 s. That is acceptable and expected.
3. **Optimisations to try, in this order:**
   - Call Grounding DINO only at segment starts or when tracking is lost, not every frame.
   - Lower the VLM's FPS, frame count and visual-token budget.
   - Use the smaller crop size.
   - Use float16 everywhere.
   - Only call the VLM when needed: add a simple **motion/change detector** (for example, difference between the child's crops or movement of the mask). If the child has not changed much, skip the VLM call and keep the previous sentence. Call the VLM again when significant change appears, plus a periodic refresh (e.g. every 5–10 s).
4. **Stability:** run the app for 30+ minutes. Check for memory leaks, crashes when the child leaves or re-enters the frame, and camera disconnects. Add error handling and clean shutdown.

**Checkpoint (M7):** 30-minute run without crash, with VRAM staying under 16 GB and the display staying smooth.

## Phase 8 – Testing and evaluation

**Goal:** prove it works with numbers, not just a demo. This is what makes the capstone industry-grade.

1. Run the full pipeline on every clip in tests/videos/ and log the sentences with timestamps.
2. Compare against tests/ground_truth.csv. For each time window, mark the output **correct / partly correct / wrong / hallucinated** (describes something not happening).
3. Calculate and tabulate:
   - **Activity accuracy** (% correct)
   - **Latency** (time from action happening to correct sentence displayed)
   - **FPS** of the display loop
   - **VRAM** peak
   - **Temporal consistency** (how often the text flickers)
   - **Hallucination rate**
   - **Child-tracking success** (% of time the right child was followed)
4. **Test hard cases on purpose:** adult plus child, child partly hidden behind furniture, low light, fast movement (running/jumping), child leaving and re-entering the frame, empty room.
5. **Ablation (optional, strong for the report).** Compare:
   - A: VLM only on full frames
   - B: Detector + VLM
   - C: Detector + SAM 2 tracker + VLM (our design)

This shows why each component is needed.

1. Write results in results/final_evaluation.md with tables and screenshots.

## Phase 9 – Documentation and handover

1. README.md: what it does, hardware needed, installation steps for both environments (exact commands that worked), how to download the checkpoints, how to start the VLM server and the app, and known limitations.
2. Complete THIRD_PARTY_LICENSES.md.
3. Add requirements_vlm.txt and requirements_vision.txt with pinned versions.
4. Record a 2–3 minute demo video of the full system working.
5. Make sure the project can be started from scratch following only the README (test this on a clean folder if possible).
6. Tag the code in git as v1.0.

## Phase 10 – V2 upgrades (ONLY after V1 is signed off)

Do not start these until I confirm V1 is complete.

- **RTMPose (MMPose, Apache-2.0):** body keypoints of the child, feeding movement information such as arm flapping, rocking, jumping.
- **RTMDet (MMDetection, Apache-2.0):** a faster person detector as a fallback or companion to Grounding DINO.
- **MMAction2 (Apache-2.0):** specialised action classifier, for example PoseC3D, to compare against the VLM.
- **Event detection and memory:** "child has been sitting for 8 minutes", "child has fallen".
- **Research references to study (not dependencies):** ASDMotion, CV-child-action-recognition, autism-related-behavior, VideoChat3.

# Quick reference

## Do

- Test every stage alone before joining it to the next.
- Keep two separate environments and a small local server between them.
- Run the VLM in the background, every 1–2 s, on a child-only clip.
- Save proof (video/screenshot) at every milestone.
- Record versions, fixes and licences as you go.

## Don't

- Don't use YOLO, BoxMOT, OpenPose or paid APIs.
- Don't install everything into one Python environment.
- Don't send every frame to the VLM.
- Don't copy whole repositories into our code. Install them from third_party/ and write thin wrapper classes in src/.
- Don't add pose, action classifiers, audio, dashboards or voice features in V1.
- Don't describe the system as diagnosing autism.

## Milestone summary

| Milestone | Phase | What it proves                             |
| --------- | ----- | ------------------------------------------ |
| M1        | 3     | Camera finds the child with Grounding DINO |
| M2        | 4     | SAM 2 follows the same child continuously  |
| M3        | 5     | Clean child-only crops                     |
| M4        | 5     | 2–5 second rolling buffer                  |
| M5        | 6     | Qwen3-VL returns a sentence for the buffer |
| M6        | 6     | Stable, smoothed text shown live           |
| M7        | 7     | Real-time performance on our GPU           |
| M8        | 10    | Pose and action recognition (V2 only)      |