import time
from typing import List, Dict, Optional, Any
from src.utils.logger import setup_logger

logger = setup_logger("event_engine")

class EventEngine:
    """Tracks state transitions, inactivity alerts, and structured activity events."""

    def __init__(self, inactivity_timeout_seconds: float = 180.0):
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self.current_state: Optional[str] = None
        self.state_start_time: float = time.time()
        self.events_log: List[Dict[str, Any]] = []
        self.last_inactivity_alert_time: float = 0.0

    def update(self, smoothed_activity: str) -> Optional[Dict[str, Any]]:
        """Updates current state, checks for transitions, and returns event dict if triggered."""
        now = time.time()
        new_event = None

        if self.current_state is None:
            self.current_state = smoothed_activity
            self.state_start_time = now
            return None

        # Check for state transition
        if smoothed_activity != self.current_state and smoothed_activity not in ("Child not in view.", "Observing room..."):
            duration = now - self.state_start_time
            new_event = {
                "timestamp": now,
                "event_type": "STATE_TRANSITION",
                "previous_state": self.current_state,
                "new_state": smoothed_activity,
                "duration_seconds": round(duration, 1),
                "summary": f"State transition: '{self.current_state}' -> '{smoothed_activity}' (lasted {duration:.1f}s)"
            }
            logger.info(new_event["summary"])
            self.events_log.append(new_event)
            self.current_state = smoothed_activity
            self.state_start_time = now

        # Check for prolonged inactivity
        duration_in_current = now - self.state_start_time
        if duration_in_current >= self.inactivity_timeout_seconds:
            if (now - self.last_inactivity_alert_time) >= 60.0:  # Avoid spamming every frame
                inactivity_event = {
                    "timestamp": now,
                    "event_type": "INACTIVITY_ALERT",
                    "state": self.current_state,
                    "duration_seconds": round(duration_in_current, 1),
                    "summary": f"Child has remained in state '{self.current_state}' for {duration_in_current/60.0:.1f} minutes."
                }
                logger.warning(inactivity_event["summary"])
                self.events_log.append(inactivity_event)
                self.last_inactivity_alert_time = now
                if new_event is None:
                    new_event = inactivity_event

        return new_event

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self.events_log)
