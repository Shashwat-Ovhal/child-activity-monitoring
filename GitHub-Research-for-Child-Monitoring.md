> From: https://chatgpt.com/c/6ac4a0df-970c-83ec-a37b-aca2dae69884

# you asked

message time: 2026-10-06 12:50:32

I want to create a project, an engineering project. Just to give you an idea of the scale of the project I need, I am making this project as a capstone project for my final year engineering semester, final year engineering project. So I want it to be industry-grade. At the end, or the conclusion of the project, what I am trying to build is something like a child monitoring system, or a child monitoring system for children suffering from autism and autism-like neurological diseases or disorders, monitoring them, even using live labs to give them command in voice clones of their parents and something like that, just monitoring them, running it locally, and all that. This is the complete end goal. Don't keep this end goal in the mind for now, just for your idea. What I want you to do now, or what I want to build now, is the part of the project. In this part of the project, I want something like a visual-based model. I want a system or a model or an architecture that will use VLM, and based on VLMs, or based on using VLMs, it will only identify a child in a room, and once it identifies a child in a room, it will constantly monitor its action and name it. Just there will be a text line on the screen or anywhere. It will give just a text output of what exactly the child is doing in real time at that particular moment, for example, is walking, is sitting, is eating, or anything. Whatever the child might be doing must be analyzed and must be named in a word or in a sentence, described in a sentence. And I want something like this system to be built. I want VLMs to be used, included in this system, and use the free-tier VLMs and free-tier options most probably wherever possible. If not possible at all, then we can shift towards the paid or some cheaper alternatives. So this is what I want to build. As you have the complete idea of the part of what I want to build of this project, I want you to go through GitHub and do a detailed search, analysis, and detailed research on the GitHub repositories or GitHub projects readily available, as I feel or I believe I want to avoid completely writing code from scratch, and I believe if possible, I want to pick something like this from GitHub itself. I want to copy the project from GitHub, and if a complete project is not possible to copy, then I need all the parts of the various GitHub repositories. So do a detailed research analysis on projects like this or components of these projects, which I may need, all in a detailed research manner. Go through GitHub and find all the repositories and their use cases for me. repositories.

---

# chatgpt response

I’ll treat this as a GitHub-focused architecture study: first identify the strongest existing child/action-recognition projects, then map VLM, tracking, temporal-action, and local-inference components into one buildable capstone stack.

A key finding already is that a pure VLM loop is not the best engineering design: use a fast detector/tracker continuously, and invoke the VLM selectively to produce the natural-language activity label. This gives much better real-time feasibility.

Yes. I did a fairly broad GitHub search specifically around **real-time child detection → persistent tracking → activity/action understanding → VLM/video-MLLM description**, including autism-related repositories. The good news is that you **do not need to build this from scratch**.

The important finding is that I would **not** recommend taking one giant repository and trying to force it to do everything. The strongest engineering solution is to combine a few mature open-source components.

## 1. What you are actually trying to build

Your immediate subsystem can be defined as:

```text
                    LIVE CAMERA
                         │
                         ▼
              ┌─────────────────────┐
              │ Child Detection     │
              │ "Is there a child?" │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Person/Child        │
              │ Tracking             │
              │ Persistent ID        │
              └──────────┬──────────┘
                         │
                Child ROI / Track
                         │
                         ▼
              ┌─────────────────────┐
              │ Temporal Buffer     │
              │ 2–5 sec of video    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ VLM / Video MLLM    │
              │                     │
              │ What is child doing?│
              └──────────┬──────────┘
                         │
                         ▼
             "The child is walking."
             "The child is sitting."
             "The child is eating."
             "The child is playing."
                         │
                         ▼
              ┌─────────────────────┐
              │ Temporal Smoothing  │
              │ + Confidence        │
              └─────────────────────┘
```

This architecture is considerably better than:

```text
Camera → VLM → description → VLM → description → VLM...
```

because a large VLM is too expensive computationally to run on every camera frame.

---

# 2. The most important GitHub discovery: there is already a child-specific foundation

One repository is particularly interesting for your project:

### 

This is **not just generic human action recognition**.

It is specifically:

> Automated Analysis of Stereotypical Movements in Videos of Children With Autism Spectrum Disorder

The repository contains the code, dataset/model resources and training pipeline associated with that work. More importantly, its README explicitly points to a **Child Detector** for cases where there are multiple adults and one child. ([GitHub](https://github.com/Dinstein-Lab/ASDMotion?utm_source=chatgpt.com))

That is extremely relevant to your eventual system.

Its architecture is roughly:

```text
Video
  │
  ├── Child Detector
  │
  ▼
Child localization
  │
  ▼
Pose estimation
  │
  ▼
Temporal action analysis
  │
  ▼
Autism-related movement recognition
```

This is much closer to your eventual capstone than a generic "video captioning" repository.

The repository is also listed by the organization's GitHub as MIT licensed. ([GitHub](https://github.com/Dinstein-Lab?utm_source=chatgpt.com))

### My assessment

**Very high relevance for your final capstone.**

But I would **not use ASDMotion as your complete current system**, because your current objective is broader:

> "What is the child doing right now?"

rather than:

> "Is this stereotypical/autism-related movement?"

We can borrow the child-detection/pose/temporal-analysis concepts while using a modern VLM for general activity descriptions.

---

# 3. Another extremely relevant child-action repository

### 

This one is almost literally named for your problem:

**Child Action and Gender Recognition**

It contains notebooks using pretrained action-recognition architectures including:

- CSN
- I3D
- TPN
- R(2+1)D

and has models trained specifically using a **children's action dataset**. The repository reports very high classification results on its datasets. ([GitHub](https://github.com/androbaza/CV-child-action-recognition?utm_source=chatgpt.com))

Architecture:

```text
Child Video
     │
     ▼
Video frames
     │
     ▼
3D CNN / Video model
     │
     ▼
Action classification
     │
     ├── walking
     ├── sitting
     ├── etc.
```

### Why I like it for your project

It gives you something that VLM repositories generally don't:

**child-specific action-recognition knowledge.**

However, its limitation is important:

It is a **closed-set classifier**.

For example:

```text
walking
sitting
running
...
```

A VLM can potentially say:

> "The child is sitting on the floor while playing with a toy."

So I would use this repository as a **specialized action-recognition reference**, not as the final natural-language description engine.

---

# 4. The strongest VLM candidate: Qwen3-VL

### 

This is one of the most important repositories I found for your project.

Qwen3-VL is the newer Qwen multimodal model family, with capabilities specifically covering:

- image understanding
- visual reasoning
- spatial understanding
- video understanding
- temporal dynamics
- long-context video
- agent interaction

and has dense and MoE models ranging from smaller models to very large models. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))

The repository is Apache-2.0. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))

### This fits your requirement extremely well.

Instead of asking:

```text
What class does this video belong to?
```

you can ask:

```text
Observe the child in this video.

Describe ONLY the child's current physical activity.
Return one concise sentence.

Examples:
"The child is walking across the room."
"The child is sitting on the floor."
"The child is eating."
"The child is playing with a toy."
```

The model can then generate the textual activity.

---

# 5. Even more interesting: VideoChat3

There is a very interesting **2026** project:

### 

This one deserves serious attention.

VideoChat3 is a **4B video MLLM** specifically designed for:

- fine-grained motion
- long-video reasoning
- temporal grounding
- online/streaming video understanding

The authors describe it as a model intended to understand video across time, including subtle motion and live streams. ([GitHub](https://github.com/MCG-NJU/VideoChat3?utm_source=chatgpt.com))

Its architecture uses:

```text
Video
 ↓
I3D-ViT
 ↓
spatiotemporal compression
 ↓
Video MLLM
 ↓
language response
```

and is specifically designed to reduce the enormous token cost associated with feeding every video frame into a VLM. ([GitHub](https://github.com/MCG-NJU/VideoChat3?utm_source=chatgpt.com))

The repository's training code is Apache-2.0. ([GitHub](https://github.com/MCG-NJU/VideoChat3/blob/main/xtuner-videochat3/README.md?utm_source=chatgpt.com))

### This could eventually be one of your strongest candidates.

Especially because your requirement isn't simply:

> "Analyze this image."

It is:

> **"Understand what the child is doing over time."**

That's a video-understanding problem.

---

# 6. Qwen2.5-VL is also extremely useful

### 

This repository is particularly useful because it already contains an **experimental streaming video demo**.

It supports:

```text
webcam
screen capture
video
images
```

and has a `web_demo_streaming` implementation. ([GitHub](https://github.com/QwenLM-corp/Qwen2.5-VL?utm_source=chatgpt.com))

The project also documents deployment through vLLM for faster inference. ([GitHub](https://github.com/anglerchen/Qwen2.5-VL?utm_source=chatgpt.com))

So a very practical prototype could be:

```text
Webcam
  ↓
Qwen2.5-VL
  ↓
"What is the child doing?"
  ↓
"The child is walking."
```

before we even add the sophisticated tracking layer.

---

# 7. Video-LLaVA — useful, but I would not choose it first

### 

This is a well-established open-source video-language model.

It supports:

- video understanding
- image understanding
- video question answering
- video-language reasoning
- local inference
- 4-bit loading

The official repository provides CLI and Gradio inference. ([GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA?utm_source=chatgpt.com))

It's a very good research baseline.

However, for **your new project**, I would rank it below Qwen3-VL / VideoChat3.

Why?

Because your problem is strongly oriented toward:

```text
live/streaming
+
temporal understanding
+
real-time
+
local inference
```

and newer models are more attractive for that.

---

# 8. Video-LLaVA is still valuable as a baseline

For your final-year report, having a baseline is actually useful.

You could experimentally compare:

```text
Model A: Video-LLaVA
Model B: Qwen2.5-VL
Model C: Qwen3-VL
Model D: VideoChat3
```

and measure:

| Metric | What you measure |
|---|---|
| Activity accuracy | Was "walking" correct? |
| Latency | Time to produce description |
| FPS | Processing speed |
| VRAM | GPU requirement |
| Temporal consistency | Does it keep saying the same thing? |
| Hallucination | Does it describe something not happening? |
| Description quality | How useful is the sentence? |

That would make your capstone significantly more research-oriented.

---

# 9. But we need a proper child detector

This is another critical point.

A generic object detector gives:

```text
person
```

but your eventual application wants:

```text
CHILD
```

That distinction matters.

One excellent repository for this is:

### 

It combines:

```text
Grounding DINO
       ↓
SAM 2
       ↓
video tracking
```

and can accept text prompts.

The repository explicitly demonstrates:

- object grounding
- video tracking
- custom video inputs
- continuous IDs
- Grounding DINO
- Florence-2
- SAM 2. ([GitHub](https://github.com/ydjayling/G-SAM2?utm_source=chatgpt.com))

So theoretically:

```text
prompt = "child"
```

could be used to locate the child.

Then:

```text
Child bounding box
        ↓
SAM 2
        ↓
child mask
        ↓
persistent tracking
```

---

# 10. Grounding DINO itself

### 

This is another important building block.

Unlike a normal YOLO model that has predefined categories, Grounding DINO supports **open-set / language-guided object detection**.

So instead of:

```python
classes = ["person", "car", "dog"]
```

you can conceptually ask for:

```text
"child"
```

or:

```text
"young child"
```

or potentially:

```text
"child wearing a red shirt"
```

The official repository describes it as an open-set object detector and is Apache-2.0 licensed. ([GitHub](https://github.com/idea-research/groundingdino?utm_source=chatgpt.com))

This makes it very interesting for your system.

---

# 11. SAM 2 is probably the best tracking/segmentation component

### 

SAM 2 is designed for image **and video segmentation**.

More importantly, it has a video predictor with streaming memory and supports tracking objects through video. ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

The repository provides:

```text
SAM2VideoPredictor
```

and supports:

```text
point prompts
box prompts
mask prompts
multiple objects
```

with persistent tracking through video. ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

And importantly for your industry-grade goal:

**SAM 2 is Apache-2.0.** ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

---

# 12. There is actually a ready-made combination of these

This is one of the most useful findings.

### 

It already combines:

```text
Grounding DINO
       +
Florence-2
       +
SAM 2
```

for:

```text
detect
→ ground
→ segment
→ track
```

in videos. ([GitHub](https://github.com/ydjayling/G-SAM2?utm_source=chatgpt.com))

That means you could potentially avoid writing the entire tracking subsystem yourself.

---

# 13. YOLO is another option

### 

YOLO11 supports:

- object detection
- segmentation
- classification
- pose
- tracking

and its tracking framework supports persistent object IDs. ([GitHub](https://github.com/ultralytics/yolo11?utm_source=chatgpt.com))

For your system:

```text
Camera
 ↓
YOLO
 ↓
person
 ↓
tracker
 ↓
ID = 7
 ↓
crop child
```

is extremely fast.

### But there is a catch.

Ultralytics' current licensing is **AGPL-3.0**, with a separate enterprise licensing route for proprietary/commercial products. ([GitHub](https://github.com/ultralytics/yolo11?utm_source=chatgpt.com))

For a college research prototype this may be fine, but if your eventual goal is:

> commercial product

you need to take the license seriously.

---

# 14. BoxMOT

### 

This is another very useful repository.

It provides multiple multi-object trackers:

- ByteTrack
- BoT-SORT
- Deep OC-SORT
- StrongSORT
- etc.

and can work with detection systems such as YOLO. ([GitHub](https://github.com/mikel-brostrom/boxmot?utm_source=chatgpt.com))

This is useful if you want:

```text
Detector
    ↓
Tracker
    ↓
Child ID #1
```

rather than constantly re-detecting the child.

However, BoxMOT is also AGPL-3.0. ([GitHub](https://github.com/mikel-brostrom/boxmot?utm_source=chatgpt.com))

---

# 15. Pose estimation is another layer worth keeping

For your eventual autism-monitoring system, pose could become extremely important.

### 

MMPose provides:

- human pose
- multi-person pose
- hand pose
- face landmarks
- whole-body pose
- 3D human mesh
- RTMPose

and is Apache-2.0. ([GitHub](https://github.com/open-mmlab/mmpose?utm_source=chatgpt.com))

RTMPose is specifically intended for **real-time multi-person pose estimation**. ([GitHub](https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.yml?utm_source=chatgpt.com))

So later your architecture could become:

```text
Child
 │
 ├── RGB appearance
 │
 ├── VLM
 │
 └── Pose
       │
       ├── arms
       ├── hands
       ├── head
       ├── torso
       └── legs
```

This is particularly valuable for:

```text
hand flapping
rocking
spinning
head movement
repetitive movements
jumping
falling
abnormal posture
```

---

# 16. MMAction2 is a major building block

### 

This is one of the most comprehensive repositories I found.

It supports:

- action recognition
- action localization
- spatio-temporal action detection
- skeleton action recognition
- video retrieval

and includes models such as:

- I3D
- SlowFast
- VideoMAE
- TSM
- ST-GCN
- PoseC3D
- CTR-GCN
- etc. ([GitHub](https://github.com/open-mmlab/mmaction2?utm_source=chatgpt.com))

This could become your **specialized action-recognition engine**.

For example:

```text
Child video
     │
     ▼
Pose
     │
     ▼
PoseC3D / ST-GCN
     │
     ▼
walking / sitting / standing / ...
```

MMAction2 even has a skeleton-based action-recognition demo combining:

```text
human detection
+
pose estimation
+
action recognition
```

directly. ([GitHub](https://github.com/open-mmlab/mmaction2/blob/main/demo/README.md?utm_source=chatgpt.com))

---

# 17. There is an older but useful complete real-time architecture

### 

This repository is conceptually very close to what you want.

Its architecture is:

```text
Webcam
 ↓
OpenPose
 ↓
human pose
 ↓
DeepSORT
 ↓
person tracking
 ↓
action recognition
```

It explicitly supports multi-person tracking and real-time webcam operation. ([GitHub](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose?utm_source=chatgpt.com))

It even acknowledges an important problem:

> classifying single frames is not really action recognition because actions are temporal sequences.

That observation is exactly why your system needs a temporal/VLM layer instead of simply classifying individual frames. ([GitHub](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose?utm_source=chatgpt.com))

The downside is that this repository is old and uses a dated TensorFlow/OpenPose stack.

So:

**architecture = useful**

**codebase = probably don't build your new project directly on it.**

---

# 18. A modern version of that architecture already exists

I also found:

### 

It combines:

```text
YOLOv8n
     ↓
Pose
     ↓
Skeleton analysis
     ↓
Action recognition
     ↓
LSTM
     ↓
Memory
```

The repository implements real-time webcam/video operation and even a three-layer memory architecture. ([GitHub](https://github.com/itz-youssef/APEX?utm_source=chatgpt.com))

That's interesting for your eventual project because you're ultimately talking about:

> continuous monitoring

rather than a single prediction.

The memory concept could eventually become:

```text
Current activity:
walking

Previous 10 sec:
walking → standing → sitting

Previous 5 min:
walking → playing → eating → sitting
```

That becomes much more useful for parental monitoring.

---

# 19. Autism-specific behavior recognition repositories

This area has some surprisingly useful research code.

## Autism-related behavior recognition

### 

This project specifically addresses:

> Activity Recognition in Children with Autism-Related Behaviors.

It uses temporal models and reports experiments using:

```text
I3D
EfficientNet
Temporal Convolutional Networks
Multi-Stage TCN
```

The authors report a best weighted F1 of 0.83 for their three autism-related actions and also provide a lightweight model reaching 0.71 weighted F1. ([GitHub](https://github.com/Samwei1/autism-related-behavior?utm_source=chatgpt.com))

This is very relevant later.

---

## Autism pose behavior detection

### 

This repository uses:

```text
RTMPose
 ↓
pose
 ↓
ML classification
```

for autism-related behaviors including:

- arm flapping
- spinning
- headbanging

and related repetitive behavior. ([GitHub](https://github.com/HossamFakher1/Autism-Behavior-Detection-Using-Pose-Estimation?utm_source=chatgpt.com))

That is a very good candidate for a **future specialized behavior module**.

---

# 20. Another particularly important autism project: Autism-CLIP

### 

This is particularly interesting academically because it uses a **pretrained vision-language model** for autism-related analysis.

It analyzes caregiver-child interaction videos and combines video features with structured clinical information to predict ASD outcomes. ([GitHub](https://github.com/kkd-1021/Autism-CLIP_public?utm_source=chatgpt.com))

However:

**I would NOT use this as the core of your monitoring system.**

It is aimed at ASD prediction/assessment rather than general real-time activity description.

But it could become useful in your eventual research component.

---

# 21. AV-ASD

### 

This is an audio-visual autism behavior recognition project.

It provides code for:

```text
video
+
audio
↓
autism behavior recognition
```

and includes preprocessing scripts for extracting video and audio clips. ([GitHub](https://github.com/ShijianDeng/AV-ASD?utm_source=chatgpt.com))

This is potentially useful later when you introduce:

```text
child movement
+
voice
+
environmental audio
```

into your monitoring system.

---

# 22. OpenPose

### 

OpenPose remains a major reference implementation for:

- body keypoints
- face
- hands
- feet
- multiple people

with 135 keypoints. ([GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose?utm_source=chatgpt.com))

But I would **not start a new project with OpenPose**.

RTMPose/MMPose is a much more attractive modern foundation.

Also, OpenPose's licensing is explicitly oriented around free **non-commercial** use, which matters for your eventual product direction. ([GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose?utm_source=chatgpt.com))

---

# 23. VideoChat-Flash / VideoChat3 is particularly interesting for your future

### 

This deserves attention because the authors specifically discuss:

**streaming video understanding.**

The project reports that the newer VideoChat3 model is a 4B model designed for:

- general video
- long-form video
- streaming video
- fine-grained motion

and reports improvements over Qwen3-VL-4B on several offline and streaming metrics. ([GitHub](https://github.com/OpenGVLab/VideoChat-Flash?utm_source=chatgpt.com))

There is also a dedicated VideoChat3 repository with model/training resources. ([GitHub](https://github.com/MCG-NJU/VideoChat3?utm_source=chatgpt.com))

### This is probably the most interesting "research" alternative to Qwen3-VL.

---

# 24. LLaVA-NeXT-Video

### 

Another mature video model.

It supports models such as:

```text
LLaVA-NeXT-Video-7B
LLaVA-NeXT-Video-32B
```

and uses sampled video frames for video understanding. ([GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA-NeXT-Video.md?utm_source=chatgpt.com))

It's useful as a baseline but, again, I would prefer Qwen3-VL or VideoChat3 for your actual prototype.

---

# 25. My ranking of the repositories for YOUR project

If I were building this capstone, I'd rank them like this:

| Rank | Repository | Role | Relevance |
|---|---|---|---|
| 🥇 | **Qwen3-VL** | Main VLM | ⭐⭐⭐⭐⭐ |
| 🥇 | **VideoChat3** | Video/streaming VLM alternative | ⭐⭐⭐⭐⭐ |
| 🥇 | **ASDMotion** | Child + ASD motion research | ⭐⭐⭐⭐⭐ |
| 🥇 | **Grounded SAM 2** | Child grounding + tracking | ⭐⭐⭐⭐⭐ |
| 🥈 | **SAM 2** | Video tracking/segmentation | ⭐⭐⭐⭐⭐ |
| 🥈 | **MMAction2** | Specialized action recognition | ⭐⭐⭐⭐⭐ |
| 🥈 | **MMPose/RTMPose** | Pose estimation | ⭐⭐⭐⭐⭐ |
| 🥈 | **Qwen2.5-VL** | Practical VLM prototype | ⭐⭐⭐⭐⭐ |
| 🥈 | **CV-child-action-recognition** | Child action models | ⭐⭐⭐⭐ |
| 🥈 | **Autism-related-behavior** | ASD behavior classifier | ⭐⭐⭐⭐ |
| 🥉 | **Grounding DINO** | Child/open-vocabulary detection | ⭐⭐⭐⭐ |
| 🥉 | **YOLO11** | Fast detector/tracker | ⭐⭐⭐⭐ |
| 🥉 | **BoxMOT** | Tracking | ⭐⭐⭐⭐ |
| 🥉 | **LLaVA-NeXT-Video** | VLM baseline | ⭐⭐⭐ |
| 🥉 | **Video-LLaVA** | VLM baseline | ⭐⭐⭐ |
| — | OpenPose | Legacy pose pipeline | ⭐⭐ |

---

# 26. What I would actually build

I would **not clone one repository**.

I would create a new project by combining the best pieces.

### Version 1 — simplest working prototype

```text
                         WEBCAM
                           │
                           ▼
                    OpenCV / GStreamer
                           │
                           ▼
                    Child Detection
                           │
                           ▼
                   Child Bounding Box
                           │
                           ▼
                   3–5 second buffer
                           │
                           ▼
                       Qwen3-VL
                           │
                    "What is happening?"
                           │
                           ▼
               ┌─────────────────────────┐
               │ "The child is walking." │
               └─────────────────────────┘
```

This gets you your first demonstrable system.

---

# 27. Version 2 — proper engineering architecture

Then:

```text
                         CAMERA
                           │
                           ▼
                     Frame Capture
                           │
                           ▼
                ┌────────────────────┐
                │ Child Detector     │
                │ Grounding DINO     │
                └─────────┬──────────┘
                          │
                    Child bbox
                          │
                          ▼
                ┌────────────────────┐
                │ Tracker            │
                │ SAM2 / ByteTrack   │
                └─────────┬──────────┘
                          │
                    Child ID = 01
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        RGB crop                    Pose
             │                         │
             │                    RTMPose
             │                         │
             ▼                         ▼
       Temporal buffer            Skeleton buffer
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                 Video VLM / MLLM
                  Qwen3-VL /
                  VideoChat3
                          │
                          ▼
                   Activity Engine
                          │
                          ▼
                 Temporal Smoothing
                          │
                          ▼
                 Current Activity
```

---

# 28. Version 3 — your eventual capstone-grade architecture

This is where your original idea becomes much more interesting.

```text
                         ┌─────────────┐
                         │   CAMERA    │
                         └──────┬──────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Video Ingestion │
                       └────────┬────────┘
                                │
                ┌───────────────┼────────────────┐
                │               │                │
                ▼               ▼                ▼
          Child Detection     Pose          Audio
                │               │                │
                ▼               ▼                ▼
            Tracking        RTMPose          Whisper
                │               │                │
                └───────────────┼────────────────┘
                                │
                                ▼
                       Temporal Memory
                                │
                                ▼
                       Video MLLM / VLM
                                │
                                ▼
                     Activity Understanding
                                │
             ┌──────────────────┼─────────────────┐
             │                  │                 │
             ▼                  ▼                 ▼
         Activity          Risk/Event        Behavior
         "walking"          detection        analysis
             │                  │                 │
             └──────────────────┼─────────────────┘
                                │
                                ▼
                         Event Database
                                │
                                ▼
                       Parent Dashboard
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              Live monitoring          Historical
                                      behaviour
```

That is much closer to an **industry-grade architecture**.

---

# 29. A particularly important design decision

I strongly recommend separating:

### Perception

```text
Where is the child?
What is the child doing physically?
```

from:

### Reasoning

```text
What does this sequence of actions mean?
Is this unusual?
Has this been happening repeatedly?
Should the parent be notified?
```

and from:

### Communication

```text
"The child has been sitting quietly for 8 minutes."
"The child is currently playing with a toy."
"The child has fallen."
```

Do **not** make the VLM responsible for every single part.

---

# 30. Why this matters for real-time operation

Suppose your camera produces:

```text
30 FPS
```

You don't want:

```text
30 VLM calls/sec
```

Instead:

```text
Camera
30 FPS
 │
 ▼
Detector/Tracker
20–30 FPS
 │
 ▼
Temporal buffer
 │
 ├──── every 2 sec ────► VLM
 │
 ├──── every 2 sec ────► VLM
 │
 └──── every 2 sec ────► VLM
```

So the VLM might operate at:

**0.5–2 predictions/sec**

while the rest of the computer-vision pipeline remains real-time.

That is much more realistic.

---

# 31. An even better approach: event-triggered VLM

Eventually we can make it smarter.

Instead of:

```text
VLM every 2 seconds
```

do:

```text
Tracker
   │
   ▼
Motion/change detector
   │
   ├── no meaningful change
   │       ↓
   │    DON'T call VLM
   │
   └── significant change
           ↓
          VLM
```

Example:

```text
12:00:00
child sitting
       ↓
12:00:01
still sitting
       ↓
12:00:02
still sitting
       ↓
12:00:04
child stands
       ↓
      VLM
       ↓
"The child is standing up."
```

This dramatically reduces computational load.

---

# 32. What about "identify only the child"?

This is actually one of the harder parts.

Suppose the camera sees:

```text
        Adult
          │

 Child ───┼─── Adult

       Toy
```

A normal YOLO model says:

```text
person #1
person #2
person #3
```

It doesn't inherently know which one is the child.

We have several possible approaches:

### Option A — Grounding DINO

Prompt:

```text
"child"
```

Very flexible.

### Option B — dedicated child detector

The ASDMotion project explicitly references a child detector for scenes containing multiple adults and one child. ([GitHub](https://github.com/Dinstein-Lab/ASDMotion?utm_source=chatgpt.com))

### Option C — face/age estimation

```text
person
 ↓
face
 ↓
age estimation
 ↓
child
```

But I would **not** make age estimation the primary mechanism.

### Option D — enrollment

Eventually, your application could allow:

```text
Parent setup
      ↓
"Register child"
      ↓
camera observes child
      ↓
child appearance embedding
      ↓
persistent identity
```

That is much better for a real monitoring product.

---

# 33. The project can therefore have two identity levels

### Level 1

```text
Child
```

The system knows:

> this is the child we're monitoring.

### Level 2

```text
Child #01
```

The tracker maintains:

```text
track_id = 01
```

### Eventually

```text
Child = Aarav
```

through an enrolled identity.

That will integrate beautifully with your future parent voice-command system.

---

# 34. Free/local strategy

Your requirement of avoiding paid APIs is very realistic.

You can make the core system:

```text
OpenCV              → free
Qwen3-VL            → local/open model
VideoChat3          → local/open model
SAM2                → local/open model
Grounding DINO      → local/open model
MMPose/RTMPose      → local/open model
MMAction2           → local/open model
PyTorch             → free
FFmpeg              → free
SQLite              → free
FastAPI             → free
React/Next.js       → free
```

So you don't need:

```text
GPT-4 API
Gemini API
Claude API
```

for the core system.

---

# 35. Hardware is going to determine which VLM we choose

This is the one thing I **don't want to guess yet**.

For example, there is a huge difference between:

```text
RTX 3050 6 GB
```

and:

```text
RTX 4070 12 GB
```

and:

```text
RTX 4090 24 GB
```

and:

```text
CPU only
```

The best model changes dramatically.

For example:

### Low-end GPU

We might use:

```text
small VLM
+
quantization
+
YOLO/RTMPose
```

### 12 GB GPU

We can explore:

```text
Qwen3-VL 4B-class
```

or similar efficient models.

### 24 GB+

We have considerably more flexibility.

---

# 36. One project I would NOT simply copy

I would avoid blindly copying old repositories such as:

```text
OpenPose + TensorFlow 1.x
DeepSORT
old LSTM
```

even though they work.

They are useful as **architectural references**, but not as your primary foundation.

Your capstone should look like:

```text
Modern detector
+
modern tracker
+
modern pose
+
modern video MLLM
+
temporal memory
```

rather than a 2018-era computer vision stack.

---

# 37. My recommended repository stack

If you told me:

> "Shashwat has to start this project tomorrow. Choose the repositories for him."

I would initially choose:

### Core VLM

**Qwen3-VL**



---

### Video VLM alternative / research comparison

**VideoChat3**



---

### Child localization / tracking

**Grounded SAM 2**



---

### Video segmentation/tracking foundation

**SAM 2**



---

### Pose

**MMPose / RTMPose**



---

### Classical action-recognition backup

**MMAction2**



---

### Child-specific research

**ASDMotion**



---

### Child action dataset/model reference

**CV-child-action-recognition**



---

### Autism-specific action recognition

**autism-related-behavior**



---

# 38. The architecture I'd recommend you actually implement

For **Phase 1**, don't build the whole autism-monitoring platform.

Build this:

```text
                    ┌──────────────┐
                    │ USB CAMERA   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ OpenCV       │
                    │ Frame Stream  │
                    └──────┬───────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Child Localization  │
                │ Grounding DINO      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Persistent Tracking │
                │ SAM2                 │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Child ROI Buffer │
                 │ ~3–5 seconds     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Qwen3-VL /       │
                 │ VideoChat3       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Activity Parser  │
                 └────────┬─────────┘
                          │
                          ▼
               ┌────────────────────────┐
               │ CURRENT ACTIVITY       │
               │                        │
               │ "Child is walking."   │
               └────────────────────────┘
```

Then add RTMPose.

Then add specialized action recognition.

Then add event detection.

Then add memory.

Then add audio.

Then add the parent-facing application.

---

# 39. One major research opportunity for your capstone

The project becomes much more interesting if you don't simply say:

> "I integrated Qwen3-VL."

Instead, your research question could be:

> **Can a lightweight hybrid perception architecture combining child-specific detection, persistent tracking, pose estimation, and a local video-language model provide accurate real-time child activity recognition on edge hardware?**

Then experimentally compare:

```text
Architecture A
VLM only

vs.

Architecture B
Detector + VLM

vs.

Architecture C
Detector + Tracker + VLM

vs.

Architecture D
Detector + Tracker + Pose + VLM
```

Measure:

```text
FPS
latency
VRAM
activity accuracy
temporal consistency
false descriptions
```

That gives you an actual **engineering/research contribution**, rather than simply assembling a GitHub demo.

---

# 40. One important warning about the medical/autism part

For the eventual system, I would frame it as:

**assistive monitoring / behavioral observation**

rather than:

**autism diagnosis**

unless you build and clinically validate a completely separate medical system.

Repositories such as Autism-CLIP and ASDMotion are research systems, but their existence doesn't mean a student-built monitoring system can safely diagnose ASD.

For your capstone, the safer technical claim is:

> **real-time child activity and behavior monitoring**

and later:

> **research-oriented detection of selected repetitive behaviors**

rather than:

> **diagnosis of autism.**

---

## Bottom line

**You were right that you should not start from a blank Python file.**

There is already an unusually strong ecosystem available.

The most promising combination I found is:

**Qwen3-VL / VideoChat3**  
→ natural-language video understanding

**Grounded SAM 2 + SAM 2**  
→ find and continuously track the child

**MMPose / RTMPose**  
→ body/pose/movement information

**MMAction2**  
→ specialized temporal action recognition

**ASDMotion + autism-related-behavior + child-action-recognition**  
→ child/ASD-specific research and datasets

That gives us a very credible path from a simple prototype:

> **"Child is walking."**

to the eventual industry-grade system:

> **"Child #01 has been walking around the room for 24 seconds, then stopped near the table and is now interacting with an object."**

And importantly, most of the **core perception/ML stack can remain local and free**, rather than depending on paid cloud VLM APIs.

### One thing I would do before writing any code

The next step should be a **repository-level implementation study**, not coding yet. We should take the ~8 most relevant repositories above and inspect their **actual folder structures, inference scripts, model sizes, dependencies, licenses, GPU requirements, input/output formats, and exactly which files we can reuse**. Then we can produce a concrete:

```text
REPO A
  └── copy these files
REPO B
  └── copy this module
REPO C
  └── use this model
REPO D
  └── use this dataset
        ↓
YOUR NEW PROJECT
        ↓
module-by-module architecture
```

That would let us get very close to a **"clone + integrate + modify" project instead of a from-scratch project**, which is exactly the direction you want.

---

# you asked

message time: 2026-10-06 13:28:28

1] as YOLO and BoxMOT are AGPL, and OpenPose is non-commercial. They are fine for college but risky if you ever make a product. so can we find good alternatives for them if needed? 2] In GPU I have GTX not RTX is it ok? 3] With this info, select or finalize repos. I suggested next step is to inspect the selected repos in detail (folders, dependencies, GPU needs, licenses) and decide exactly which files and modules to reuse.

---

# chatgpt response

Yes—GTX is workable, but it changes which VLMs are practical. I’ll now narrow the stack to permissive-license components and inspect the selected repositories’ actual setup, model requirements, and reusable modules.

I’m changing one earlier recommendation: we do **not** need YOLO or BoxMOT. An Apache-2.0 OpenMMLab detector/tracker stack can cover those roles, while SAM 2 remains the higher-level video tracker.

Yes. With the new constraints, I would **change the architecture from my previous recommendation**. We can avoid YOLO, BoxMOT, and OpenPose entirely, and build the core stack around **Apache-2.0 / permissively licensed components wherever practical**.

The other important point is your **GTX GPU**: it is not automatically a problem. The exact GTX model and VRAM matter a lot, but we can design the system so that the detector/tracker/pose stages are lightweight and the VLM is called only periodically.

## 1. First: replace YOLO and BoxMOT

### Detector: use MMDetection + RTMDet

I would now choose:



MMDetection is Apache-2.0, and its RTMDet implementation is also explicitly documented as Apache-2.0. ([GitHub](https://github.com/open-mmlab/mmdetection?utm_source=chatgpt.com))

This gives us:

```text
Camera
   ↓
RTMDet
   ↓
person bounding boxes
```

instead of:

```text
YOLO
```

### Why this is better for your requirement

We aren't actually required to use YOLO.

What we need is:

> a fast person detector.

RTMDet can fill that role.

And because MMDetection is modular, we can use its detector without taking the entire framework into our application. Its repository is designed around configurable detection components. ([GitHub](https://github.com/open-mmlab/mmdetection/blob/main/setup.py?utm_source=chatgpt.com))

### One caveat

A standard RTMDet model detects **person**, not necessarily **child**.

So I would use:

```text
RTMDet
   ↓
person candidates
   ↓
child-selection module
```

or, when needed:

```text
Grounding DINO
   ↓
"text prompt: child"
```

Grounding DINO itself is Apache-2.0. ([GitHub](https://github.com/idea-research/groundingdino?utm_source=chatgpt.com))

---

# 2. Tracker: we can avoid BoxMOT too

We have two good choices.

### Choice A — MMTracking



MMTracking is Apache-2.0 and supports multiple-object tracking, including ByteTrack and OC-SORT. ([GitHub](https://github.com/open-mmlab/mmtracking?utm_source=chatgpt.com))

So:

```text
RTMDet
   ↓
bounding boxes
   ↓
MMTracking
   ↓
Child ID = 1
```

This is my preferred choice if we want a clean OpenMMLab stack.

### Choice B — standalone ByteTrack implementation

There are MIT-licensed ByteTrack implementations available, so we don't actually need BoxMOT.

For example, this maintained implementation explicitly identifies ByteTrack as MIT licensed. ([GitHub](https://github.com/artefactory/bytetrack?utm_source=chatgpt.com))

So another architecture is:

```text
MMDetection / RTMDet
       ↓
   ByteTrack
       ↓
Track ID
```

### My decision

**For our first implementation: MMTracking.**

We don't need another tracker dependency unless MMTracking becomes unnecessarily heavy.

---

# 3. And yes — remove OpenPose

I agree with your concern.

We don't need OpenPose.

In fact, the ASDMotion repository itself currently requires OpenPose, Python 3.9, PyTorch 1.13 and CUDA 11.7, which is already a fairly old dependency stack. ([GitHub](https://github.com/Dinstein-Lab/ASDMotion))

Instead:

## MMPose / RTMPose



MMPose is Apache-2.0 and includes modern real-time pose models such as RTMPose. ([GitHub](https://github.com/open-mmlab/mmpose))

So:

```text
Child bounding box
       ↓
RTMPose
       ↓
17 / 133 keypoints
       ↓
movement information
```

This is substantially cleaner for our future autism-behavior module.

---

# 4. So our basic perception stack becomes

Instead of:

```text
YOLO
+
BoxMOT
+
OpenPose
```

we use:

```text
MMDetection / RTMDet
          ↓
MMTracking / ByteTrack
          ↓
MMPose / RTMPose
```

All three can be based on Apache-2.0 OpenMMLab components. MMDetection and MMTracking explicitly state Apache-2.0, while MMPose does as well. ([GitHub](https://github.com/open-mmlab/mmdetection?utm_source=chatgpt.com))

That's a **much better foundation for an eventual product**.

---

# 5. But there is an even better option for child tracking

This is where **Grounded SAM 2** becomes useful.



It combines:

```text
Grounding DINO
      +
SAM 2
```

and explicitly provides video grounding, segmentation and tracking. ([GitHub](https://github.com/ydjayling/G-SAM2))

Most importantly, the repository already contains ready-made tracking demos including:

```text
grounded_sam2_tracking_demo.py

grounded_sam2_tracking_demo_custom_video_input_gd1.0_hf_model.py

grounded_sam2_tracking_demo_with_continuous_id.py

grounded_sam2_tracking_demo_with_continuous_id_plus.py
```

([GitHub](https://github.com/ydjayling/G-SAM2))

That's very attractive for our prototype.

You could conceptually do:

```text
"child"
   ↓
Grounding DINO
   ↓
child bbox
   ↓
SAM 2
   ↓
persistent mask/track
```

No YOLO.

No BoxMOT.

No OpenPose.

---

# 6. SAM 2 is also permissively licensed

The official SAM 2 repository states that the **model checkpoints, demo code and training code are Apache 2.0**. ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

It provides:

```text
sam2.1_hiera_tiny
sam2.1_hiera_small
sam2.1_hiera_base_plus
sam2.1_hiera_large
```

checkpoints. ([GitHub](https://github.com/facebookresearch/sam2))

And its video predictor is explicitly designed to propagate object masks through video while maintaining an inference state. ([GitHub](https://github.com/facebookresearch/sam2))

That is exactly the sort of persistent child tracking we need.

---

# 7. GTX vs RTX — this needs clarification

**GTX is absolutely usable.**

RTX is not a mandatory requirement.

But there are three questions:

1. **Which GTX?**
2. **How much VRAM?**
3. **Which CUDA capability does it have?**

For example, these are very different:

```text
GTX 1050 Ti 4 GB
GTX 1650 4 GB
GTX 1660 Super 6 GB
GTX 1070 8 GB
GTX 1080 Ti 11 GB
GTX 1080 8 GB
GTX 1060 6 GB
```

The fact that it says **GTX instead of RTX** doesn't tell us enough.

### The good news

The perception models are fine on many GTX cards:

```text
RTMDet
RTMPose
SAM2 tiny/small
Grounding DINO
```

The **VLM is the difficult component**.

---

# 8. For your GTX, I would not start with VideoChat3

This is an important change from my previous answer.

VideoChat3 is extremely interesting scientifically.

It is a **4B video MLLM** specifically designed for fine-grained motion, long video reasoning, temporal grounding and online response. ([GitHub](https://github.com/MCG-NJU/VideoChat3?utm_source=chatgpt.com))

But its current training environment expects:

```text
Linux
Python >= 3.10
PyTorch >= 2.6
NVIDIA GPU
```

and its training examples use multi-GPU configurations. ([GitHub](https://github.com/MCG-NJU/VideoChat3/blob/main/xtuner-videochat3/README.md?utm_source=chatgpt.com))

That's not what I want to make our first implementation depend on.

### So:

**VideoChat3 = research benchmark / future experiment**

not

**our initial production pipeline.**

---

# 9. Qwen3-VL is much more attractive for us



This is currently my **primary VLM choice**.

The important part is that Qwen3-VL now has:

```text
2B
4B
8B
30B-A3B
32B
235B-A22B
```

scale options. ([GitHub](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md?_bhlid=f1a48c2c52e9e82e3cbaabbb05437b7c821fffa8))

For your machine, we start at the bottom.

### Qwen3-VL-2B

The official Hugging Face model is Apache-2.0. ([Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/main?utm_source=chatgpt.com))

### Qwen3-VL-4B

Also Apache-2.0. ([Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main?utm_source=chatgpt.com))

That is excellent for your commercial-license concern.

---

# 10. Qwen3-VL is particularly appropriate for your task

The official implementation supports direct **video input**, not merely individual images. ([GitHub](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md?_bhlid=f1a48c2c52e9e82e3cbaabbb05437b7c821fffa8))

It also allows controlling:

```text
video FPS
number of frames
visual-token budget
```

during video processing. ([GitHub](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md?_bhlid=f1a48c2c52e9e82e3cbaabbb05437b7c821fffa8))

That is very important.

We can tell it:

```text
Analyze this 3-second child video.

Describe only the child's current physical activity.

Return exactly one short sentence.
```

rather than sending the entire 30-FPS stream.

---

# 11. This is where your GTX becomes manageable

Suppose you have a modest GTX.

We don't do:

```text
30 FPS
 ×
VLM
```

Instead:

```text
30 FPS CAMERA
      │
      ▼
RTMDet
      │
      ▼
Tracking
      │
      ▼
Child ROI
      │
      ▼
3-second buffer
      │
      ▼
Qwen3-VL-2B
      │
      ▼
"The child is walking."
```

Then wait.

Maybe 1–2 seconds later:

```text
new temporal window
       ↓
Qwen3-VL
       ↓
"The child is sitting."
```

The VLM becomes the **semantic brain**, not the frame-rate engine.

---

# 12. Our finalized repositories

I'm now narrowing the project to **7 core repositories**.

## CORE — actually used

### 1. Qwen3-VL



**Purpose:** video/image understanding and natural-language activity description.

**Use:** Qwen3-VL-2B first → Qwen3-VL-4B if hardware permits.

**License:** Apache-2.0 for the selected 2B/4B model repositories. ([Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main?utm_source=chatgpt.com))

---

### 2. Grounding DINO



**Purpose:** open-vocabulary child detection.

Input:

```text
"child"
```

Output:

```text
bounding box
confidence
```

**License:** Apache-2.0. ([GitHub](https://github.com/idea-research/groundingdino?utm_source=chatgpt.com))

---

### 3. SAM 2



**Purpose:** persistent video segmentation/tracking.

**License:** Apache-2.0. ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

Start with:

```text
SAM2.1 Hiera Tiny
```

or

```text
SAM2.1 Hiera Small
```

depending on your GTX.

---

### 4. Grounded SAM 2



**Purpose:** integration reference.

This is **not necessarily something we'll install as the entire application**.

We'll reuse its architecture/code patterns for:

```text
Grounding DINO
       ↓
SAM 2
       ↓
tracking
```

The repository already has continuous-ID tracking demos. ([GitHub](https://github.com/ydjayling/G-SAM2))

---

### 5. MMDetection



**Purpose:** fast conventional object detection fallback.

Use:

```text
RTMDet
```

rather than YOLO.

**License:** Apache-2.0. ([GitHub](https://github.com/open-mmlab/mmdetection?utm_source=chatgpt.com))

---

### 6. MMPose



**Purpose:** pose estimation.

Use:

```text
RTMPose
```

for real-time pose.

**License:** Apache-2.0. ([GitHub](https://github.com/open-mmlab/mmpose))

---

### 7. MMAction2



**Purpose:** specialized action recognition.

**License:** Apache-2.0. ([GitHub](https://github.com/open-mmlab/mmaction2))

We won't necessarily need it in V1.

But it becomes extremely useful when we start evaluating:

```text
walking
running
sitting
standing
falling
jumping
playing
```

as actual classified activities rather than VLM-generated descriptions.

---

# 13. What gets demoted

These repositories are now **research references**, not dependencies.

### ASDMotion



Extremely valuable because it is genuinely about children with ASD.

But it depends on:

```text
OpenPose
+
forked MMAction2
+
YOLOv5 child detector
```

and its documented environment is Python 3.9/PyTorch 1.13/CUDA 11.7. ([GitHub](https://github.com/Dinstein-Lab/ASDMotion))

So we will study its:

```text
dataset
model
labels
PoseC3D configuration
evaluation methodology
```

but **not inherit its outdated perception stack**.

---

### CV-child-action-recognition



Useful for:

```text
child action datasets
CSN
I3D
TPN
R(2+1)D
```

and its reported child-action experiments. ([GitHub](https://github.com/androbaza/CV-child-action-recognition?utm_source=chatgpt.com))

But again:

**research reference, not core dependency.**

---

### autism-related-behavior



Useful later for:

```text
ASD-related activity labels
TCN
MS-TCN
I3D feature extraction
```

but its code has its own legacy dependencies and even requires manually obtaining certain feature-extractor components because of licensing. ([GitHub](https://github.com/Samwei1/autism-related-behavior?utm_source=chatgpt.com))

So:

**research module later.**

---

# 14. Final architecture I now recommend

```text
                         ┌──────────────────┐
                         │    USB CAMERA     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ OpenCV / FFmpeg  │
                         │ Frame Capture    │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Child Localization      │
                    │                         │
                    │ Grounding DINO          │
                    │ "child"                 │
                    └────────────┬────────────┘
                                 │
                          Child BBox
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ SAM 2                   │
                    │ Persistent Tracking     │
                    └────────────┬────────────┘
                                 │
                         Child Track #1
                                 │
                   ┌─────────────┴──────────────┐
                   │                            │
                   ▼                            ▼
             Child RGB ROI                 RTMPose
                   │                            │
                   │                     Body keypoints
                   │                            │
                   └─────────────┬──────────────┘
                                 │
                                 ▼
                         Temporal Buffer
                           2–5 seconds
                                 │
                                 ▼
                       ┌─────────────────┐
                       │ Qwen3-VL 2B/4B │
                       │                 │
                       │ Video reasoning │
                       └────────┬────────┘
                                │
                                ▼
                       Activity Description
                                │
                                ▼
                ┌─────────────────────────────┐
                │ "The child is walking."    │
                └─────────────────────────────┘
```

---

# 15. Then the advanced version

Once V1 works:

```text
                     CHILD VIDEO
                          │
           ┌──────────────┼───────────────┐
           │              │               │
           ▼              ▼               ▼
       Appearance        Pose          Motion
           │              │               │
           ▼              ▼               ▼
      Qwen3-VL        RTMPose        MMAction2
           │              │               │
           └──────────────┼───────────────┘
                          │
                          ▼
                   Activity Fusion
                          │
                          ▼
                   Temporal Memory
                          │
                          ▼
                 Behaviour Engine
```

This is where your capstone becomes genuinely interesting.

---

# 16. Exactly what we should reuse from each repository

This is the part I think you were asking for when you said:

> "decide exactly which files and modules to reuse."

### Qwen3-VL

We should **not copy the entire repository**.

We need primarily:

```text
Qwen3-VL/
├── qwen-vl-utils/
│   └── video processing utilities
│
└── Transformers model interface
    └── AutoModelForImageTextToText
```

The official repository already provides direct video inference and frame/FPS control. ([GitHub](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md?_bhlid=f1a48c2c52e9e82e3cbaabbb05437b7c821fffa8))

Our application will call it through our own:

```text
vlm_engine.py
```

---

### Grounding DINO

We need:

```text
groundingdino/
```

plus:

```text
configuration
checkpoint
inference interface
```

We do **not** need:

```text
training
evaluation
demo notebooks
```

The repository itself has a relatively clean structure with `groundingdino/`, `demo/`, requirements and setup files. ([GitHub](https://github.com/idea-research/groundingdino?utm_source=chatgpt.com))

---

### SAM 2

We need:

```text
sam2/
configs/
checkpoints/
```

particularly:

```text
build_sam2_video_predictor
```

because that's the component responsible for video tracking. ([GitHub](https://github.com/facebookresearch/sam2))

We don't need:

```text
training/
dataset/
web demo
```

for V1.

---

### Grounded SAM 2

This is where we can save substantial development time.

We should study/reuse the logic in:

```text
grounded_sam2_tracking_demo.py

grounded_sam2_tracking_demo_with_continuous_id.py

grounded_sam2_tracking_demo_with_continuous_id_plus.py
```

Those are already specifically aimed at tracking objects through video with continuous IDs. ([GitHub](https://github.com/ydjayling/G-SAM2))

This could save us from writing the initial:

```text
detect → prompt SAM2 → propagate → maintain ID
```

logic ourselves.

---

### MMPose

We only need:

```text
RTMPose model
inference API
```

not the entire training ecosystem.

MMPose is explicitly modular, which makes this kind of extraction practical. ([GitHub](https://github.com/open-mmlab/mmpose))

---

### MMAction2

Initially:

**don't integrate it.**

Later:

```text
MMAction2
└── PoseC3D / other temporal model
```

can become our specialized activity classifier.

---

# 17. What I would NOT install yet

Don't install all seven repositories into one Python environment.

That will create a dependency disaster.

Especially:

```text
Qwen3-VL
+
MMDetection
+
MMPose
+
MMTracking
+
SAM2
+
GroundingDINO
+
MMAction2
```

all have their own PyTorch/version requirements.

Instead we'll create a **modular service architecture**.

For example:

```text
project/
│
├── app/
│
├── camera/
│
├── detection/
│
├── tracking/
│
├── pose/
│
├── vlm/
│
├── activity/
│
├── temporal/
│
└── ui/
```

and isolate troublesome dependencies where necessary.

---

# 18. The GTX issue: what I need from you

Before I give you the **exact model size and installation commands**, I need the exact GPU.

On Windows, run:

```powershell
nvidia-smi
```

and send me the output.

Or simply tell me something like:

> GTX 1650 4 GB

or:

> GTX 1660 Super 6 GB

or:

> GTX 1080 Ti 11 GB

The VRAM is particularly important.

### Roughly:

| GTX VRAM | Our strategy |
|---:|---|
| 4 GB | Very lightweight VLM / aggressive quantization |
| 6 GB | Qwen3-VL-2B becomes the primary target |
| 8 GB | Qwen3-VL-2B comfortably worth testing; 4B may be possible with optimization |
| 11 GB+ | Much more flexibility |
| 16 GB+ | We can explore 4B-class models seriously |

These are **planning ranges, not guarantees**—actual memory usage depends heavily on quantization, visual-token budget, video frames and CUDA/PyTorch compatibility.

Qwen3-VL specifically lets us control video FPS, frame count and visual-token budget, which is exactly what we'll use to adapt it to your GTX. ([GitHub](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md?_bhlid=f1a48c2c52e9e82e3cbaabbb05437b7c821fffa8))

---

# 19. One more important licensing distinction

You raised the right concern, but there's a subtle point:

**"Repository is Apache-2.0" ≠ "everything associated with the project is automatically commercially unrestricted."**

We need to check separately:

```text
Source code license
        +
Model-weight license
        +
Dataset license
        +
Third-party dependencies
```

For example, Grounded SAM 2 is commendably explicit about separating licenses for its bundled components: `LICENSE_groundingdino`, `LICENSE_sam2`, and `LICENSE_cctorch`. ([GitHub](https://github.com/ydjayling/G-SAM2))

For Qwen3-VL, the **2B and 4B model repositories currently show Apache-2.0**, which makes them much more attractive for our chosen stack. ([Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main?utm_source=chatgpt.com))

We'll maintain a `THIRD_PARTY_LICENSES.md` in your project from day one.

---

# 20. Final selection

So, **this is now my finalized stack**:

### Mandatory V1

| Component | Selected repo/model | License |
|---|---|---|
| Camera | OpenCV | Apache-2.0 |
| Child detection | Grounding DINO | Apache-2.0 |
| Tracking | SAM 2 | Apache-2.0 |
| VLM | **Qwen3-VL-2B-Instruct** initially | Apache-2.0 |
| Video processing | Qwen VL Utils / FFmpeg | permissive components |
| UI | Our own lightweight application | Our code |

### V2

| Component | Selected |
|---|---|
| Fast detector | MMDetection / RTMDet |
| Pose | MMPose / RTMPose |
| Temporal action | MMAction2 |
| VLM upgrade | Qwen3-VL-4B |
| Specialized ASD behavior | ASDMotion research/model concepts |

### Research comparison

```text
VideoChat3-4B
CV-child-action-recognition
autism-related-behavior
ASDMotion
```

### Explicitly rejected

```text
❌ YOLO as core dependency
❌ BoxMOT
❌ OpenPose
❌ Old OpenPose + LSTM pipelines
❌ Cloud-only VLM APIs
```

---

## The next step is now very concrete

Send me the output of:

```powershell
nvidia-smi
```

**especially the GPU name + VRAM.**

Then I can do the next repository audit at the level you're asking for: **exact repo folders → exact files/classes/scripts worth reusing → dependency versions → CUDA/PyTorch compatibility → which models/checkpoints to download → which repositories should remain isolated → final project folder architecture → first milestone implementation order.**

That is where we'll turn this from a collection of GitHub projects into **one coherent capstone system**.

---

# you asked

message time: 2026-10-06 13:42:52

Answer in very short. We have 16 GB plus GTX. Which all repos are we using (when you have choice select the best or better option) Then do the next repository audit at the level you're asking for: **exact repo folders → exact files/classes/scripts worth reusing → dependency versions → CUDA/PyTorch compatibility → which models/checkpoints to download → which repositories should remain isolated → final project folder architecture → first milestone implementation order.**

---

# chatgpt response

Yes. With **16 GB VRAM + GTX**, I would keep the core stack lean and avoid the AGPL/non-commercial repos.

## Final repos we will use

| Role | Repo | Decision |
|---|---|---|
| **VLM / activity understanding** |  | **Core — Qwen3-VL-4B-Instruct** |
| **Child detection** |  | **Core** |
| **Tracking + segmentation** |  | **Core — SAM 2.1 Hiera Small** |
| **Integration reference** |  | **Reference only** |
| **Pose — later** |  | **V2** |
| **Fast detector — later** |  | **V2** |
| **Action recognition — later** |  | **V2/research** |

**Not using:** YOLO, BoxMOT, OpenPose, MMYOLO.

---

# Repository audit

### 1. Qwen3-VL — semantic brain

**Reuse:**
- `Qwen3-VL/`
- `qwen-vl-utils/`
- Transformers integration from `README.md`
- `AutoModelForImageTextToText`
- `AutoProcessor`

Qwen3-VL officially supports video understanding and requires `transformers >=4.57.0`. ([GitHub](https://github.com/QwenLM/Qwen3-VL?utm_source=chatgpt.com))

**Download:**
- `Qwen/Qwen3-VL-4B-Instruct`

**Purpose:**  
Input = 2–5 sec child video/ROI → output = `"The child is walking toward the table."`

---

### 2. Grounding DINO — child localization

**Important files:**
```text
groundingdino/
├── models/
├── datasets/
├── util/
│   └── inference.py       ← REUSE
├── config/
│   └── GroundingDINO_SwinT_OGC.py
```

The important API is:

```python
from groundingdino.util.inference import load_model, predict
```

The repo already provides `Model`, `load_model()`, `predict()` etc. ([GitHub](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py?utm_source=chatgpt.com))

**Checkpoint:**
```text
groundingdino_swint_ogc.pth
```

**Prompt:**
```text
"child"
```

**Dependency concern:** it builds a CUDA deformable-attention extension, so CUDA/PyTorch must be aligned. ([GitHub](https://github.com/idea-research/groundingdino?utm_source=chatgpt.com))

---

### 3. SAM 2.1 — persistent child tracking

**Reuse:**
```text
sam2/
├── build_sam.py
├── sam2_video_predictor.py
├── sam2_image_predictor.py
└── configs/
    └── sam2.1/
```

Main API:

```python
build_sam2_video_predictor()
```

then:

```python
init_state()
add_new_points_or_box()
propagate_in_video()
```

SAM2 officially provides this video tracking workflow. ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

**Checkpoint:**
```text
sam2.1_hiera_small.pt
```

I prefer **Small**, not Large, for your GTX/16-GB setup.

SAM2 currently requires **Python ≥3.10, PyTorch ≥2.5.1, torchvision ≥0.20.1** and recommends WSL/Linux for Windows. ([GitHub](https://github.com/facebookresearch/sam2?utm_source=chatgpt.com))

---

### 4. Grounded-SAM2 — don't install wholesale

We only borrow its integration logic, especially:

```text
grounded_sam2_tracking_demo.py
grounded_sam2_tracking_demo_with_continuous_id.py
grounded_sam2_tracking_demo_with_continuous_id_plus.py
```

These demonstrate exactly the **Grounding DINO → SAM2 → continuous tracking** pipeline we need. ([GitHub](https://github.com/ydjayling/G-SAM2?utm_source=chatgpt.com))

---

# Environment strategy

**Do NOT put every repository into one environment.**

Recommended:

```text
child-monitoring/
│
├── env_vlm/          ← Qwen3-VL
├── env_vision/       ← GroundingDINO + SAM2
├── models/
├── src/
└── configs/
```

Initially, however, we'll try to make **Grounding DINO + SAM2 one environment** and Qwen its own environment.

For SAM2, PyTorch/CUDA compatibility is especially important because its CUDA extension is compiled against the installed PyTorch/CUDA combination. ([GitHub](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md?utm_source=chatgpt.com))

---

# Final project architecture

```text
Camera
   ↓
Grounding DINO
   ↓
"CHILD" bounding box
   ↓
SAM 2.1
   ↓
Persistent Child ID + Mask
   ↓
Child ROI
   ↓
2–5 sec Temporal Buffer
   ↓
Qwen3-VL 4B
   ↓
Activity Description
   ↓
Temporal Smoothing
   ↓
"Child is walking"
"Child is sitting"
"Child is eating"
"Child is playing"
```

Later:

```text
             ┌── RTMPose
             ├── MMAction2
Camera → Detection → SAM2 → Temporal Fusion → Qwen3-VL
                                      ↓
                              Behavior Engine
```

---

# First milestone order

**M1:** Camera → Grounding DINO detects `"child"`.

**M2:** Grounding DINO → SAM2 → child remains tracked while moving.

**M3:** Extract only the child's ROI.

**M4:** Store a 2–5 second temporal buffer.

**M5:** Send buffer to Qwen3-VL-4B.

**M6:** Produce stable activity labels/sentences.

**M7:** Real-time optimization/event-triggering.

**M8:** Add RTMPose/MMAction2 only after V1 works.

### One thing I need before giving the exact installation matrix

Send the output of:

```powershell
nvidia-smi
```

Then I can give you the **exact PyTorch + CUDA + Python versions and exact checkpoint sizes/VRAM plan for your particular GTX**, rather than guessing.

---

# you asked

message time: 2026-10-06 13:47:52

Answer in one sentence: Will this project, when complete, be able to tell any action the child is doing in real time and not only the actions with a specific dataset.

---

# chatgpt response

Yes—**with Qwen3-VL as the semantic VLM, the completed system can describe previously unseen/unscripted actions in real time, rather than being limited to a fixed dataset of predefined actions**.