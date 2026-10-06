from typing import List
import numpy as np
import random
from src.vlm.base import BaseVLMEngine
from src.utils.logger import setup_logger

logger = setup_logger("mock_vlm")

class MockVLMEngine(BaseVLMEngine):
    """Zero-resource mock engine for rapid local development, UI debugging, and CI tests."""

    def __init__(self):
        self.actions = [
            "The child is sitting on the floor.",
            "The child is walking across the room.",
            "The child is playing with a toy.",
            "The child is standing near the table.",
            "The child is reaching for an object."
        ]
        self._current_idx = 0
        self._call_count = 0
        logger.info("MockVLMEngine initialized for CPU/offline simulation.")

    def is_available(self) -> bool:
        return True

    def describe_activity(self, frames: List[np.ndarray], prompt: str) -> str:
        """Returns plausible simulated action string, holding state for several calls before transitioning."""
        self._call_count += 1
        # Change simulated activity every 4 calls
        if self._call_count % 4 == 0:
            self._current_idx = (self._current_idx + 1) % len(self.actions)

        action = self.actions[self._current_idx]
        return action
