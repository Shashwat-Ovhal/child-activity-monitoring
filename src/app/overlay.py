import cv2
import numpy as np
from typing import Tuple, Optional

class VisualOverlay:
    """Renders sleek bounding boxes, masks, track IDs, and activity banners."""

    def __init__(
        self,
        font_scale: float = 0.75,
        banner_height: int = 55,
        box_color: Tuple[int, int, int] = (0, 220, 110),
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ):
        self.font_scale = font_scale
        self.banner_height = banner_height
        self.box_color = box_color
        self.text_color = text_color
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        mask: Optional[np.ndarray],
        track_id: int,
        activity_text: str,
        fps: float
    ) -> np.ndarray:
        """Annotates frame with tracker information and sleek HUD text."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw semi-transparent segmentation mask if available
        if mask is not None and np.any(mask):
            colored_mask = np.zeros_like(annotated)
            colored_mask[mask > 0] = self.box_color
            annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.25, 0)

        # Draw Bounding Box and Track ID
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.box_color, 2)

            id_tag = f"Child #{track_id}"
            (tw, th), _ = cv2.getTextSize(id_tag, self.font, 0.55, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 24)), (x1 + tw + 10, y1), self.box_color, -1)
            cv2.putText(annotated, id_tag, (x1 + 5, y1 - 6), self.font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw Top HUD Banner (Dark background strip)
        cv2.rectangle(annotated, (0, 0), (w, self.banner_height), (20, 20, 25), -1)
        cv2.line(annotated, (0, self.banner_height), (w, self.banner_height), (60, 60, 70), 1)

        # Display Activity Text
        display_text = f"Activity: {activity_text}"
        cv2.putText(
            annotated,
            display_text,
            (20, int(self.banner_height * 0.65)),
            self.font,
            self.font_scale,
            self.text_color,
            2,
            cv2.LINE_AA
        )

        # Display FPS in corner
        fps_text = f"{fps:.1f} FPS"
        (ftw, _), _ = cv2.getTextSize(fps_text, self.font, 0.55, 1)
        cv2.putText(
            annotated,
            fps_text,
            (w - ftw - 20, int(self.banner_height * 0.65)),
            self.font,
            0.55,
            (180, 180, 190),
            1,
            cv2.LINE_AA
        )

        return annotated
