import os
import cv2
import numpy as np
from typing import List, Optional
from src.vlm.base import BaseVLMEngine
from src.utils.logger import setup_logger

logger = setup_logger("qwen_engine")

class QwenVLEngine(BaseVLMEngine):
    """Qwen3-VL / Qwen2.5-VL Vision-Language Model execution engine for video reasoning."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        precision: str = "float16",
        server_url: Optional[str] = None
    ):
        self.model_id = model_id
        self.device = device
        self.precision = precision
        self.server_url = server_url
        self.model = None
        self.processor = None
        self._is_loaded = False

        if not self.server_url:
            self._load_local_model()
        else:
            logger.info(f"QwenVLEngine configured to query microservice at: {self.server_url}")
            self._is_loaded = True

    def _load_local_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            logger.info(f"Loading Qwen-VL model '{self.model_id}' on {self.device} ({self.precision})...")
            torch_dtype = torch.float16 if self.precision == "float16" and self.device == "cuda" else torch.float32

            # GTX GPUs use sdpa attention, NOT flash_attention_2
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device if self.device == "cuda" else None,
                attn_implementation="sdpa"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self._is_loaded = True
            logger.info("Qwen-VL loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load local Qwen-VL model ({e}). "
                "Engine will operate in fallback mode or use remote VLM server."
            )
            self._is_loaded = False

    def is_available(self) -> bool:
        return self._is_loaded

    def describe_activity(self, frames: List[np.ndarray], prompt: str) -> str:
        """Process chronological frames through Qwen3-VL video reasoning."""
        if not frames:
            return "Child not in view."

        # If microservice URL configured, call via REST
        if self.server_url:
            return self._query_server(frames, prompt)

        if not self._is_loaded or self.model is None:
            return "The child is moving in the room."

        try:
            import torch
            from PIL import Image

            # Convert BGR frames to PIL RGB images
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": 2.0},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                videos=[pil_frames],
                padding=True,
                return_tensors="pt"
            )
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=48)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            sentence = output_text.strip().split("\n")[0].split(".")[0] + "."
            if not sentence.lower().startswith("the child"):
                sentence = f"The child is {sentence.lstrip('is ')}"

            return sentence
        except Exception as e:
            logger.error(f"Error during Qwen-VL video inference: {e}")
            return "The child is performing an activity."

    def _query_server(self, frames: List[np.ndarray], prompt: str) -> str:
        import requests
        try:
            # Encode frames as JPEG bytes
            encoded_frames = []
            for f in frames:
                _, buf = cv2.imencode(".jpg", f)
                encoded_frames.append(buf.tobytes().hex())

            payload = {"frames_hex": encoded_frames, "prompt": prompt}
            res = requests.post(f"{self.server_url}/describe", json=payload, timeout=10)
            if res.status_code == 200:
                return res.json().get("description", "The child is moving.")
        except Exception as e:
            logger.error(f"Microservice call failed: {e}")
        return "The child is active."
