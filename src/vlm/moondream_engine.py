import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
from src.vlm.base import BaseVLMEngine
from src.utils.logger import setup_logger

logger = setup_logger("moondream_engine")

class MoondreamEngine(BaseVLMEngine):
    """Moondream 3 Vision-Language Model execution wrapper."""

    def __init__(
        self,
        model_id: str = "moondream/moondream3-preview",
        device: str = "cuda",
        precision: str = "float16"
    ):
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            logger.info(f"Attempting to load Moondream 3 from '{self.model_id}' on {self.device}...")

            # Option 1: Official moondream package
            try:
                import moondream as md
                self.model = md.vl(model=self.model_id)
                self._is_loaded = True
                logger.info("Moondream loaded via official moondream library.")
                return
            except (ImportError, Exception) as e:
                logger.info(f"Official moondream package not available or failed ({e}); trying Transformers...")

            # Option 2: Hugging Face Transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = torch.float16 if self.precision == "float16" and self.device == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device == "cuda" else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._is_loaded = True
            logger.info("Moondream 3 loaded successfully via Transformers.")
        except Exception as e:
            logger.warning(
                f"Could not load Moondream model ({e}). "
                "MoondreamEngine will operate in stub/dry-run mode until weights are present."
            )
            self._is_loaded = False

    def is_available(self) -> bool:
        return self._is_loaded

    def _create_temporal_contact_sheet(self, frames: List[np.ndarray]) -> Image.Image:
        """Combines representative frames into a horizontal/grid strip so Moondream can view chronological motion."""
        if not frames:
            return Image.new("RGB", (384, 384))

        count = len(frames)
        # Select up to 4 keyframes to avoid high resolution token inflation
        sample_indices = np.linspace(0, count - 1, min(4, count), dtype=int)
        sampled = [frames[i] for i in sample_indices]

        # Horizontal composite strip
        h, w = sampled[0].shape[:2]
        strip = np.hstack(sampled)
        # Convert BGR to RGB
        rgb_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_strip)

    def describe_activity(self, frames: List[np.ndarray], prompt: str) -> str:
        """Prompt Moondream with representative temporal child crops."""
        if not frames:
            return "Child not in view."

        if not self._is_loaded or self.model is None:
            return "The child is moving in the room."

        try:
            image = self._create_temporal_contact_sheet(frames)

            # Query Moondream
            if hasattr(self.model, "query"):
                # Official moondream API
                answer = self.model.query(image, prompt)["answer"]
            elif hasattr(self.model, "answer_question"):
                # Transformers custom remote code interface
                answer = self.model.answer_question(image, prompt, tokenizer=self.tokenizer)
            else:
                answer = "The child is active in the room."

            # Post-process answer
            clean_answer = answer.strip()
            if not clean_answer.lower().startswith("the child"):
                clean_answer = f"The child is {clean_answer.lstrip('is ')}"

            return clean_answer
        except Exception as e:
            logger.error(f"Error during Moondream inference: {e}")
            return "The child is performing an activity."
