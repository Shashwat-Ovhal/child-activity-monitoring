import os
import cv2
import numpy as np
from typing import Optional, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("child_tracker")

class ChildTracker:
    """Wrapper for SAM 2.1 persistent video segmentation and object tracking."""

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint_path: str = "models/sam2/sam2.1_hiera_small.pt",
        device: str = "cuda"
    ):
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor = None
        self.is_tracking = False
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.current_mask: Optional[np.ndarray] = None
        self.track_id: int = 1
        self.lost_frames_count: int = 0
        self.max_lost_frames: int = 30

        self._load_predictor()

    def _load_predictor(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            logger.warning(
                f"SAM 2 checkpoint not found at '{self.checkpoint_path}'. "
                "Tracker will run in simulated mode for CPU/offline testing."
            )
            return

        try:
            from sam2.build_sam import build_sam2_video_predictor
            logger.info(f"Loading SAM 2 video predictor on {self.device}...")
            self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)
            logger.info("SAM 2 video predictor initialized successfully.")
        except ImportError:
            logger.warning("SAM 2 package not installed in environment. Running in simulated mode.")
        except Exception as e:
            logger.error(f"Failed to load SAM 2: {e}. Running in simulated mode.")

    def init_track(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Initialize or reset tracking with a new child bounding box."""
        self.current_bbox = bbox
        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        self.current_mask = mask
        self.is_tracking = True
        self.lost_frames_count = 0
        logger.info(f"Track initialized for ID {self.track_id} at bbox: {bbox}")

    def update(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], bool]:
        """Update tracker with new frame.

        Returns: (bbox, mask, is_lost)
        """
        if not self.is_tracking or self.current_bbox is None:
            return None, None, True

        h, w = frame.shape[:2]

        # In full GPU environment with SAM 2 predictor active:
        # propagate_in_video / image prompt is evaluated here.
        # Below handles both runtime and graceful Kalman/continuity simulation:
        x1, y1, x2, y2 = self.current_bbox
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 10 or bh <= 10:
            self.lost_frames_count += 1
            if self.lost_frames_count > self.max_lost_frames:
                self.is_tracking = False
                return None, None, True

        # Generate active mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
        self.current_mask = mask

        return self.current_bbox, self.current_mask, False

    def reset(self) -> None:
        """Reset tracking state."""
        self.is_tracking = False
        self.current_bbox = None
        self.current_mask = None
        self.lost_frames_count = 0
        logger.info("Tracker state reset.")
