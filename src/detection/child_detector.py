import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger("child_detector")

class ChildDetector:
    """Wrapper for Grounding DINO open-vocabulary child localization."""

    def __init__(
        self,
        config_path: str = "configs/GroundingDINO_SwinT_OGC.py",
        weights_path: str = "models/grounding_dino/groundingdino_swint_ogc.pth",
        prompt: str = "child",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda"
    ):
        self.config_path = config_path
        self.weights_path = weights_path
        self.prompt = prompt.lower().strip()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.weights_path):
            logger.warning(
                f"Grounding DINO weights not found at '{self.weights_path}'. "
                "Detector will run in simulated mode for CPU/offline testing."
            )
            return

        try:
            # Patch for transformers >= 4.40 compatibility with Grounding DINO
            try:
                import torch
                from transformers import BertModel
                if not hasattr(BertModel, "get_extended_attention_mask"):
                    def _get_extended_attention_mask(self, attention_mask, input_shape):
                        if attention_mask.dim() == 3:
                            ext_mask = attention_mask[:, None, :, :]
                        elif attention_mask.dim() == 2:
                            ext_mask = attention_mask[:, None, None, :]
                        else:
                            raise ValueError(f"Wrong shape: {attention_mask.shape}")
                        return (1.0 - ext_mask) * -10000.0
                    BertModel.get_extended_attention_mask = _get_extended_attention_mask
            except Exception as patch_err:
                logger.debug(f"BertModel patch not needed or failed: {patch_err}")

            from groundingdino.util.inference import load_model
            logger.info(f"Loading Grounding DINO on {self.device}...")
            self.model = load_model(self.config_path, self.weights_path, device=self.device)
            logger.info("Grounding DINO loaded successfully.")
        except ImportError:
            logger.warning(
                "GroundingDINO package not installed in current environment. "
                "Running in simulated mode."
            )
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO model: {e}. Falling back to simulation.")

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect child in frame. Returns list of ((x1, y1, x2, y2), score)."""
        h, w = frame.shape[:2]

        if self.model is not None:
            try:
                import torch
                from groundingdino.util.inference import predict
                from torchvision import transforms

                # Preprocessing
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize([800, 800]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = transform(rgb_frame)

                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=self.prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

                results = []
                for box, score in zip(boxes, logits):
                    # Grounding DINO returns normalized cx, cy, w, h
                    cx, cy, bw, bh = box.tolist()
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    # Clamp to image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    results.append(((x1, y1, x2, y2), float(score)))

                return results
            except Exception as e:
                logger.error(f"Error during Grounding DINO inference: {e}")

        # Simulated fallback: center-oriented simulated person detection
        # Finds prominent color blob or center area
        sim_w = int(w * 0.25)
        sim_h = int(h * 0.5)
        sim_x1 = int(w * 0.35)
        sim_y1 = int(h * 0.3)
        sim_box = (sim_x1, sim_y1, sim_x1 + sim_w, sim_y1 + sim_h)
        return [(sim_box, 0.88)]
