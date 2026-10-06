import os
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger("frame_buffer")

class FrameBuffer:
    """Maintains a rolling temporal buffer of normalized child ROI crops."""

    def __init__(
        self,
        duration_seconds: float = 3.0,
        target_fps: int = 8,
        crop_margin: float = 0.20,
        target_crop_size: Tuple[int, int] = (384, 384)
    ):
        self.duration_seconds = duration_seconds
        self.target_fps = target_fps
        self.crop_margin = crop_margin
        self.target_crop_size = target_crop_size  # (height, width)
        self.max_capacity = max(5, int(duration_seconds * target_fps))
        self.buffer = deque(maxlen=self.max_capacity)
        self.last_capture_time = 0.0
        self.interval = 1.0 / target_fps

    def add_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], timestamp: float) -> bool:
        """Crops child ROI with margin, normalizes size, and appends to buffer if interval met."""
        if (timestamp - self.last_capture_time) < self.interval:
            return False

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        # Add margin
        mx = int(bw * self.crop_margin)
        my = int(bh * self.crop_margin)

        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return False

        # Resize with aspect ratio preserved
        target_h, target_w = self.target_crop_size
        ch, cw = crop.shape[:2]
        scale = min(target_w / cw, target_h / ch)
        nw, nh = int(cw * scale), int(ch * scale)
        resized_crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        # Place onto canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top = (target_h - nh) // 2
        left = (target_w - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized_crop

        self.buffer.append(canvas)
        self.last_capture_time = timestamp
        return True

    def get_frames(self) -> List[np.ndarray]:
        """Returns copy of all current frames in buffer."""
        return list(self.buffer)

    def is_ready(self) -> bool:
        """Returns true when buffer has filled at least 50% capacity."""
        return len(self.buffer) >= (self.max_capacity // 2)

    def clear(self) -> None:
        self.buffer.clear()

    def save_debug_clip(self, output_path: str, fps: int = 8) -> bool:
        """Saves current buffer as an mp4 file for visual inspection."""
        if not self.buffer:
            return False
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        h, w = self.target_crop_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for f in self.buffer:
            out.write(f)
        out.release()
        logger.info(f"Debug clip written to {output_path}")
        return True
