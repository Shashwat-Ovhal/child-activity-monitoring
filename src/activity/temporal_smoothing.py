from collections import deque, Counter
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger("temporal_smoothing")

class TemporalSmoother:
    """Anti-flicker filter ensuring stable, jitter-free activity descriptions."""

    def __init__(self, window_size: int = 4, min_streak: int = 2):
        self.window_size = window_size
        self.min_streak = min_streak
        self.history = deque(maxlen=window_size)
        self.current_stable_text = "Observing room..."
        self.last_candidate = None
        self.candidate_streak = 0

    def update(self, raw_sentence: str, child_present: bool = True) -> str:
        """Applies hysteresis smoothing on newly received sentence."""
        if not child_present:
            self.history.clear()
            self.current_stable_text = "Child not in view."
            self.last_candidate = None
            self.candidate_streak = 0
            return self.current_stable_text

        sentence = raw_sentence.strip()
        if not sentence:
            return self.current_stable_text

        self.history.append(sentence)

        # Track candidate streak
        if sentence == self.last_candidate:
            self.candidate_streak += 1
        else:
            self.last_candidate = sentence
            self.candidate_streak = 1

        # Check if candidate met streak threshold
        if self.candidate_streak >= self.min_streak:
            self.current_stable_text = sentence
        else:
            # Check majority consensus in history window
            counts = Counter(self.history)
            most_common, freq = counts.most_common(1)[0]
            if freq >= (self.window_size // 2 + 1):
                self.current_stable_text = most_common

        return self.current_stable_text

    def get_current(self) -> str:
        return self.current_stable_text

    def reset(self) -> None:
        self.history.clear()
        self.current_stable_text = "Observing room..."
        self.last_candidate = None
        self.candidate_streak = 0
