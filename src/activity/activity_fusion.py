import time
import numpy as np
from typing import List, Optional
from src.temporal.frame_buffer import FrameBuffer
from src.temporal.frame_sampler import FrameSampler
from src.vlm.base import BaseVLMEngine
from src.activity.temporal_smoothing import TemporalSmoother
from src.utils.logger import setup_logger

logger = setup_logger("activity_fusion")

class ActivityFusionEngine:
    """Coordinates temporal sampling, motion change detection, VLM reasoning, and smoothing."""

    def __init__(
        self,
        vlm_engine: BaseVLMEngine,
        prompt: str,
        sample_count: int = 8,
        motion_threshold: float = 0.15,
        heartbeat_interval: float = 8.0,
        smoothing_window: int = 4
    ):
        self.vlm = vlm_engine
        self.prompt = prompt
        self.sampler = FrameSampler(target_sample_count=sample_count, motion_threshold=motion_threshold)
        self.smoother = TemporalSmoother(window_size=smoothing_window)
        self.heartbeat_interval = heartbeat_interval

        self.last_vlm_call_time = 0.0
        self.last_raw_description = "Observing..."
        self.is_busy = False

    def process(self, frame_buffer: FrameBuffer, child_present: bool) -> str:
        """Evaluates buffer and produces smoothed activity sentence."""
        if not child_present or not frame_buffer.is_ready():
            return self.smoother.update("", child_present=child_present)

        frames = frame_buffer.get_frames()
        now = time.time()
        elapsed_since_vlm = now - self.last_vlm_call_time

        motion_changed = self.sampler.has_motion_changed(frames)
        needs_heartbeat = elapsed_since_vlm >= self.heartbeat_interval

        # Trigger VLM only if motion shifted significantly or heartbeat expired
        if (motion_changed or needs_heartbeat) and not self.is_busy:
            self.is_busy = True
            sampled_frames = self.sampler.sample_representative_frames(frames)

            # Query VLM
            raw_sentence = self.vlm.describe_activity(sampled_frames, self.prompt)
            self.last_raw_description = raw_sentence
            self.last_vlm_call_time = now
            self.is_busy = False

            logger.info(f"VLM inferred: '{raw_sentence}' (motion={motion_changed}, heartbeat={needs_heartbeat})")

        return self.smoother.update(self.last_raw_description, child_present=True)
