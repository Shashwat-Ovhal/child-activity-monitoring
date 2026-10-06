from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseVLMEngine(ABC):
    """Abstract interface for Vision-Language Models."""

    @abstractmethod
    def describe_activity(self, frames: List[np.ndarray], prompt: str) -> str:
        """Analyze a sequence of child crop frames and return a concise natural language description."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model weights and execution backend are ready."""
        pass
