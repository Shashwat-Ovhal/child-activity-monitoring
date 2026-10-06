import cv2
import numpy as np
import time
from typing import Generator, Tuple, Optional, Union
from src.utils.logger import setup_logger

logger = setup_logger("camera_stream")

class CameraStream:
    """Handles real-time video capture from webcam, video file, or synthetic frames."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        loop_video: bool = True
    ):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.loop_video = loop_video
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_file = isinstance(source, str)
        self.is_open = False

        self._initialize_stream()

    def _initialize_stream(self) -> None:
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            cam_idx = int(self.source)
            logger.info(f"Connecting to webcam index {cam_idx}...")
            self.cap = cv2.VideoCapture(cam_idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        elif isinstance(self.source, str):
            logger.info(f"Opening video file: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
        else:
            logger.warning(f"Invalid source type {type(self.source)}; synthetic fallback enabled.")
            self.cap = None

        if self.cap and self.cap.isOpened():
            self.is_open = True
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Video stream established ({actual_w}x{actual_h})")
        else:
            logger.warning(f"Could not open video source {self.source}. Will yield synthetic frames for development.")
            self.is_open = False

    def frames(self) -> Generator[Tuple[bool, np.ndarray], None, None]:
        """Yields (success, frame) tuples."""
        frame_idx = 0
        while True:
            if self.is_open and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    if self.is_file and self.loop_video:
                        logger.info("End of video file reached. Looping back to start...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret:
                        logger.warning("Failed to read frame from stream.")
                        break

                if frame is not None and (frame.shape[1] != self.target_width or frame.shape[0] != self.target_height):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                yield True, frame
            else:
                # Synthetic frame fallback for headless/remote dev
                frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
                # Draw a placeholder simulated moving box
                pos_x = int((frame_idx * 5) % (self.target_width - 200)) + 100
                pos_y = int(self.target_height * 0.4)
                cv2.rectangle(frame, (pos_x, pos_y), (pos_x + 120, pos_y + 220), (0, 255, 120), -1)
                cv2.putText(
                    frame,
                    f"Synthetic Dev Stream | Frame {frame_idx}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                time.sleep(1.0 / self.target_fps)
                yield True, frame

            frame_idx += 1

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            logger.info("Video stream released.")
