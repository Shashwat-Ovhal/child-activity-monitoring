import pytest
import numpy as np
from src.vlm.mock_engine import MockVLMEngine

def test_mock_vlm_engine():
    engine = MockVLMEngine()
    assert engine.is_available() is True

    dummy_frames = [np.zeros((384, 384, 3), dtype=np.uint8) for _ in range(4)]
    prompt = "Describe activity"

    desc1 = engine.describe_activity(dummy_frames, prompt)
    assert isinstance(desc1, str)
    assert desc1.startswith("The child is")

def test_qwen_vlm_engine_fallback():
    from src.vlm.qwen_engine import QwenVLEngine
    # When initialized on CPU without weights, engine gracefully returns fallback text
    engine = QwenVLEngine(model_id="non_existent_model", device="cpu")
    dummy_frames = [np.zeros((384, 384, 3), dtype=np.uint8) for _ in range(2)]
    desc = engine.describe_activity(dummy_frames, "What is the child doing?")
    assert isinstance(desc, str)
    assert "child" in desc.lower()
