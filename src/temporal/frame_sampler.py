import cv2
import numpy as np
from typing import List, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("frame_sampler")

class FrameSampler:
    """Selects representative keyframes from a buffer and measures temporal motion dynamics."""

    def __init__(self, target_sample_count: int = 8, motion_threshold: float = 0.15):
        self.target_sample_count = target_sample_count
        self.motion_threshold = motion_threshold
        self.last_sampled_motion_score = 0.0

    def sample_representative_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Selects N evenly spaced representative frames across the temporal window."""
        n = len(frames)
        if n == 0:
            return []
        if n <= self.target_sample_count:
            return list(frames)

        indices = np.linspace(0, n - 1, self.target_sample_count, dtype=int)
        return [frames[i] for i in indices]

    def compute_motion_magnitude(self, frames: List[np.ndarray]) -> float:
        """Measures inter-frame pixel change across the temporal buffer."""
        if len(frames) < 2:
            return 0.0

        diffs = []
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            # Normalized difference percentage
            score = float(np.mean(diff) / 255.0)
            diffs.append(score)

        avg_motion = float(np.mean(diffs)) if diffs else 0.0
        self.last_sampled_motion_score = avg_motion
        return avg_motion

    def has_motion_changed(self, frames: List[np.ndarray]) -> bool:
        """Determines if the child has moved significantly enough to warrant a VLM query."""
        motion = self.compute_motion_magnitude(frames)
        return motion >= self.motion_threshold
