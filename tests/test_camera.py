import pytest
import numpy as np
from src.camera.camera_stream import CameraStream

def test_camera_synthetic_stream():
    # Pass non-existent path to verify synthetic fallback
    cam = CameraStream(source="non_existent_file.mp4", width=640, height=360, fps=10)

    gen = cam.frames()
    success, frame = next(gen)

    assert success is True
    assert frame is not None
    assert frame.shape == (360, 640, 3)

    cam.release()
