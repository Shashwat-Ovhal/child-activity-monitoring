import pytest
from src.activity.temporal_smoothing import TemporalSmoother

def test_temporal_smoothing_hysteresis():
    smoother = TemporalSmoother(window_size=4, min_streak=2)

    # First prediction does not immediately shift default until streak or consensus
    assert smoother.update("The child is walking.") == "Observing room..."

    # Second identical prediction triggers streak
    assert smoother.update("The child is walking.") == "The child is walking."

    # Single isolated jitter does not immediately switch active state
    assert smoother.update("The child is jumping.") == "The child is walking."

    # Two consecutive updates to new action transitions state
    assert smoother.update("The child is sitting.") == "The child is walking."
    assert smoother.update("The child is sitting.") == "The child is sitting."

def test_temporal_smoothing_lost_child():
    smoother = TemporalSmoother(window_size=4, min_streak=2)
    smoother.update("The child is walking.")
    smoother.update("The child is walking.")
    assert smoother.get_current() == "The child is walking."

    # When child lost
    result = smoother.update("", child_present=False)
    assert result == "Child not in view."
    assert smoother.get_current() == "Child not in view."
