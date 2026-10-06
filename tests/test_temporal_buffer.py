import pytest
import numpy as np
from src.temporal.frame_buffer import FrameBuffer
from src.temporal.frame_sampler import FrameSampler

def test_frame_buffer_cropping_and_capacity():
    buffer = FrameBuffer(duration_seconds=1.0, target_fps=5, crop_margin=0.20, target_crop_size=(384, 384))
    assert buffer.max_capacity == 5

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    bbox = (200, 150, 400, 500)

    # Add frames
    added = buffer.add_frame(dummy_frame, bbox, timestamp=0.0)
    assert added is True
    assert len(buffer.get_frames()) == 1

    crop = buffer.get_frames()[0]
    assert crop.shape == (384, 384, 3)

    # Throttling check (same timestamp should be rejected)
    rejected = buffer.add_frame(dummy_frame, bbox, timestamp=0.05)
    assert rejected is False

    # Fill capacity
    for t in [0.25, 0.5, 0.75, 1.0, 1.25]:
        buffer.add_frame(dummy_frame, bbox, timestamp=t)

    assert len(buffer.get_frames()) == 5
    assert buffer.is_ready() is True

def test_frame_sampler():
    sampler = FrameSampler(target_sample_count=4, motion_threshold=0.10)
    frames = [np.full((100, 100, 3), i * 20, dtype=np.uint8) for i in range(10)]

    sampled = sampler.sample_representative_frames(frames)
    assert len(sampled) == 4

    # Motion computation
    motion = sampler.compute_motion_magnitude(sampled)
    assert motion > 0.0
    assert sampler.has_motion_changed(sampled) is True
