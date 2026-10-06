# Third-Party Licenses & Component Ledger

This project adheres strictly to open-source compliance. AGPL components (such as YOLOv8/v9/v11 and BoxMOT) and non-commercial components (such as OpenPose) are strictly excluded from our codebase.

| Component / Subsystem | Source / Upstream Repository | Version / Release | License | Commercial / Use Notes |
|---|---|---|---|---|
| **Grounding DINO** | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | SwinT_OGC | Apache-2.0 | Permissive open-source license. |
| **SAM 2.1 (Segment Anything 2.1)** | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Hiera Small / 2.1 | Apache-2.0 | Permissive open-source license (weights and code). |
| **Grounded-SAM-2 (Reference)** | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Reference | Apache-2.0 | Used solely for architectural integration reference. |
| **Moondream 3 (MoE VLM)** | [m87-labs/moondream-python](https://github.com/m87-labs/moondream-python) & [moondream3-preview](https://huggingface.co/moondream/moondream3-preview) | 3-preview | Business Source License 1.1 (BSL 1.1) with Additional Use Grant | Permitted for research, educational, personal, and non-competing applications. Cannot be offered as a competing managed/hosted VLM API. |
| **OpenCV Python** | [opencv/opencv-python](https://github.com/opencv/opencv-python) | >= 4.8.0 | Apache-2.0 | Frame capture and image manipulation. |
| **PyTorch & Torchvision** | [pytorch/pytorch](https://github.com/pytorch/pytorch) | >= 2.5.1 | BSD-3-Clause | Core deep learning runtime. |
| **Hugging Face Transformers** | [huggingface/transformers](https://github.com/huggingface/transformers) | >= 4.45.0 | Apache-2.0 | Model loading and tokenization. |
| **PyYAML** | [yaml/pyyaml](https://github.com/yaml/pyyaml) | >= 6.0 | MIT | Configuration file parsing. |

---

### Important Notes on Excluded Technologies
- **YOLO & BoxMOT:** Excluded due to AGPL-3.0 viral licensing terms.
- **OpenPose:** Excluded due to restrictive non-commercial research-only license.
