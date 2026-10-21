import os
import sys
import time
import yaml
import cv2
import threading
from typing import Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import setup_logger
from src.camera.camera_stream import CameraStream
from src.detection.child_detector import ChildDetector
from src.tracking.child_tracker import ChildTracker
from src.temporal.frame_buffer import FrameBuffer
from src.vlm.mock_engine import MockVLMEngine
from src.vlm.moondream_engine import MoondreamEngine
from src.activity.activity_fusion import ActivityFusionEngine
from src.events.event_engine import EventEngine
from src.app.overlay import VisualOverlay

logger = setup_logger("main_app")

def load_config(config_path: str = "configs/settings.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config {config_path} not found. Using defaults.")
    return {}

def main(config_file: str = "configs/settings.yaml", headless: bool = False, save_video: Optional[str] = None):
    config = load_config(config_file)

    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    tmp_cfg = config.get("temporal", {})
    vlm_cfg = config.get("vlm", {})
    act_cfg = config.get("activity", {})
    dsp_cfg = config.get("display", {})

    logger.info("Initializing Child Activity Monitoring System...")

    # 1. Camera
    camera = CameraStream(
        source=cam_cfg.get("source", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps=cam_cfg.get("fps", 30),
        loop_video=cam_cfg.get("loop_video", True)
    )

    # 2. Detector
    detector = ChildDetector(
        config_path=det_cfg.get("config_path", "configs/GroundingDINO_SwinT_OGC.py"),
        weights_path=det_cfg.get("weights_path", "models/grounding_dino/groundingdino_swint_ogc.pth"),
        prompt=det_cfg.get("prompt", "child"),
        box_threshold=det_cfg.get("box_threshold", 0.35),
        text_threshold=det_cfg.get("text_threshold", 0.25),
        device=det_cfg.get("device", "cuda")
    )

    # 3. Tracker
    tracker = ChildTracker(
        model_cfg=trk_cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        checkpoint_path=trk_cfg.get("checkpoint_path", "models/sam2/sam2.1_hiera_small.pt"),
        device=trk_cfg.get("device", "cuda")
    )

    # 4. Temporal Buffer
    buffer = FrameBuffer(
        duration_seconds=tmp_cfg.get("buffer_duration_seconds", 3.0),
        target_fps=tmp_cfg.get("buffer_target_fps", 8),
        crop_margin=tmp_cfg.get("crop_margin", 0.20),
        target_crop_size=tuple(tmp_cfg.get("target_crop_size", [384, 384]))
    )

    # 5. Pluggable VLM Engine
    engine_type = vlm_cfg.get("engine_type", "mock").lower()
    if engine_type == "qwen":
        from src.vlm.qwen_engine import QwenVLEngine
        vlm_engine = QwenVLEngine(
            model_id=vlm_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
            device=vlm_cfg.get("device", "cuda"),
            precision=vlm_cfg.get("precision", "float16"),
            server_url=vlm_cfg.get("server_url", None)
        )
    elif engine_type == "moondream":
        vlm_engine = MoondreamEngine(
            model_id=vlm_cfg.get("model_id", "moondream/moondream3-preview"),
            device=vlm_cfg.get("device", "cuda"),
            precision=vlm_cfg.get("precision", "float16")
        )
    else:
        vlm_engine = MockVLMEngine()

    # 6. Activity Fusion & Smoothing
    fusion = ActivityFusionEngine(
        vlm_engine=vlm_engine,
        prompt=vlm_cfg.get("prompt_template", "Describe ONLY the child's current physical activity in one short sentence starting with 'The child is'."),
        sample_count=tmp_cfg.get("sample_keyframes_count", 8),
        motion_threshold=tmp_cfg.get("motion_change_threshold", 0.15),
        heartbeat_interval=tmp_cfg.get("heartbeat_interval_seconds", 8.0),
        smoothing_window=act_cfg.get("smoothing_window_size", 4)
    )

    # 7. Event Engine
    events = EventEngine(
        inactivity_timeout_seconds=act_cfg.get("inactivity_timeout_seconds", 180.0)
    )

    # 8. Overlay
    overlay = VisualOverlay(
        font_scale=dsp_cfg.get("font_scale", 0.75),
        banner_height=dsp_cfg.get("banner_height", 55)
    )

    logger.info("Pipeline initialized. Starting video stream loop...")

    prev_time = time.time()
    fps = 30.0
    latest_activity_text = "Observing room..."
    segment_duration = trk_cfg.get("segment_length_seconds", 3.0)
    last_detect_time = 0.0

    # Threading lock for VLM query
    vlm_lock = threading.Lock()

    def async_vlm_worker():
        nonlocal latest_activity_text
        with vlm_lock:
            text = fusion.process(buffer, child_present=tracker.is_tracking)
            latest_activity_text = text
            events.update(text)

    try:
        for success, frame in camera.frames():
            if not success or frame is None:
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, now - prev_time))
            prev_time = now

            # Periodic Detection / Grounding
            need_detection = (not tracker.is_tracking) or ((now - last_detect_time) > segment_duration)

            if need_detection:
                detections = detector.detect(frame)
                if detections:
                    best_box, score = detections[0]
                    tracker.init_track(frame, best_box)
                    last_detect_time = now
                else:
                    tracker.reset()

            # Tracking update
            bbox, mask, is_lost = tracker.update(frame)

            if not is_lost and bbox is not None:
                buffer.add_frame(frame, bbox, now)

            # Spawn asynchronous VLM inference thread if not already running
            if not vlm_lock.locked():
                t = threading.Thread(target=async_vlm_worker, daemon=True)
                t.start()

            # Render UI
            display_frame = overlay.draw(
                frame=frame,
                bbox=bbox,
                mask=mask,
                track_id=tracker.track_id,
                activity_text=latest_activity_text,
                fps=fps
            )

            if save_video:
                if video_writer is None:
                    h, w = display_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(save_video, fourcc, cam_cfg.get("fps", 30), (w, h))
                video_writer.write(display_frame)

            if not headless:
                cv2.imshow("Child Activity Monitoring System (V1)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested termination.")
                    break
            else:
                # In headless mode, run for sample time or break on signal
                pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        camera.release()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Saved annotated video with live activity banner to: {save_video}")
        if not headless:
            cv2.destroyAllWindows()

        # Save structured activity readings & event transitions
        report_path = "activity_events_report.json"
        try:
            import json
            with open(report_path, "w") as f:
                json.dump(events.get_events(), f, indent=2)
            logger.info(f"Exported {len(events.get_events())} activity events to {report_path}")
        except Exception as e:
            logger.warning(f"Could not export event logs: {e}")

        logger.info("System gracefully shut down.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Child Activity Monitoring System")
    parser.add_argument("--config", type=str, default="configs/settings.yaml", help="Path to YAML settings file")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window (for cloud/remote servers)")
    parser.add_argument("--save-video", type=str, default=None, help="Save annotated video with bounding boxes and activity overlay")
    args = parser.parse_args()
    main(config_file=args.config, headless=args.headless, save_video=args.save_video)
