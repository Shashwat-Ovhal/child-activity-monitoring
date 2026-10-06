# Engineering Log & Notes (NOTES.md)

## Hardware & Environment Observations

### GTX Compatibility
- GTX GPUs (e.g. GTX 1650, 1660 Ti, 1080) do **not** natively support `bfloat16` arithmetic efficiently and lack hardware Tensor Core support for `FlashAttention-2`.
- **Mitigation:** Ensure all PyTorch inference calls use `torch.float16` (or `torch.float32` on CPU) and standard `sdpa` (Scaled Dot-Product Attention).

### WSL2 / Linux Requirement for CUDA Extensions
- Grounding DINO compiles custom CUDA deformable attention extensions (`MultiScaleDeformableAttention`).
- SAM 2 compiles custom CUDA extensions.
- When running on a remote/college PC under Windows, install **WSL2 with Ubuntu 22.04 / 24.04** and install the NVIDIA CUDA Toolkit inside WSL matching the PyTorch build to compile extensions without MSVC build errors.

### Moondream 3 MoE Optimization
- Moondream 3 has ~9B total parameters with ~2B active parameters.
- It can be initialized via `moondream-python` or Hugging Face Transformers (`AutoModelForCausalLM` with `trust_remote_code=True`).
- If GPU is unavailable, `MockVLMEngine` allows instant pipeline testing on any CPU without downloading 9B weights.

---

## Log Entries

- **2026-10-06:** Project initialized. Architecture transitioned from dense Qwen3-VL-4B to Moondream 3 (MoE) for deployment efficiency across CPU, college GPUs, and rented GPU instances.
