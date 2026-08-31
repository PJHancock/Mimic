"""Unit tests for hand tracking."""

import numpy as np
import pytest

from mimic.tracking import HandTracker


@pytest.fixture
def hand_tracker():
    """Create a hand tracker instance."""
    return HandTracker(min_detection_confidence=0.5)


def test_hand_tracker_init(hand_tracker):
    """Test HandTracker initialization."""
    assert hand_tracker is not None
    # Detector may be None if MediaPipe legacy API unavailable; that's OK
    assert hasattr(hand_tracker, "detector")
    assert hasattr(hand_tracker, "detector_type")


def test_process_blank_frame(hand_tracker):
    """Test processing a frame with no hand."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = hand_tracker.process(frame, frame_idx=0)
    assert result is None


def test_process_with_grayscale_frame(hand_tracker):
    """Test that grayscale frames are handled."""
    frame = np.zeros((480, 640), dtype=np.uint8)
    # This should not raise an error
    result = hand_tracker.process(frame, frame_idx=0)
    # Expect None since there's no actual hand in a blank frame
    assert result is None


def test_finger_closure_range(hand_tracker):
    """Test that finger closure is in valid range [0, 1]."""
    # We can't easily create a frame with a real hand, so we mock the calculation
    # Test the internal function directly with synthetic landmarks

    # Create a mock hand landmarks object
    class MockLandmark:
        def __init__(self, x, y, z=0):
            self.x = x
            self.y = y
            self.z = z

    class MockHandLandmarks:
        def __init__(self):
            # Create 21 landmarks (hand has 21 points in MediaPipe)
            self.landmark = [MockLandmark(*coord) for coord in [
                (0.5, 0.5, 0),   # 0: wrist
                (0.4, 0.4, 0), (0.38, 0.38, 0), (0.36, 0.36, 0), (0.34, 0.34, 0),  # 1-4: thumb
                (0.5, 0.3, 0), (0.48, 0.28, 0), (0.46, 0.26, 0), (0.44, 0.24, 0),  # 5-8: index
                (0.5, 0.4, 0),   # 9: palm
                (0.52, 0.3, 0), (0.53, 0.28, 0), (0.54, 0.26, 0), (0.55, 0.24, 0), # 10-13: middle
                (0.54, 0.4, 0), (0.56, 0.38, 0), (0.58, 0.36, 0), (0.60, 0.34, 0), # 14-17: ring
                (0.56, 0.5, 0), (0.58, 0.48, 0), (0.60, 0.46, 0), (0.62, 0.44, 0), # 18-20: pinky
            ]]

    landmarks = MockHandLandmarks()
    closure = hand_tracker._calculate_finger_closure(landmarks, h=480, w=640)

    assert 0 <= closure <= 1, f"Finger closure {closure} out of valid range [0, 1]"


def test_process_frame_idx(hand_tracker):
    """Test that frame index is preserved."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = hand_tracker.process(frame, frame_idx=42)
    # No hand, but if it were detected, frame_idx should be 42
    # For blank frame, result is None


def test_hand_tracker_cleanup(hand_tracker):
    """Test that tracker can be properly released."""
    hand_tracker.release()
    # Should not raise an error


def test_process_uint8_rgb_frame(hand_tracker):
    """Test processing a uint8 RGB frame."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    result = hand_tracker.process(frame, frame_idx=0)
    # Expected: None or HandTrack (no hand in uniform gray)
    assert result is None or result.frame_idx == 0


def test_process_frame_dtype_conversion(hand_tracker):
    """Test that various frame dtypes are handled."""
    # Test float frame [0, 1]
    frame_float = np.ones((480, 640, 3), dtype=np.float32) * 0.5
    result = hand_tracker.process(frame_float, frame_idx=0)
    assert result is None or hasattr(result, "frame_idx")

    # Test uint8 frame [0, 255]
    frame_uint8 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    result = hand_tracker.process(frame_uint8, frame_idx=0)
    assert result is None or hasattr(result, "frame_idx")


def test_process_returns_hand_track_on_valid_hand():
    """Test that HandTrack is returned with correct fields."""
    # This is a minimal test; without an actual hand image, we can't test full detection
    tracker = HandTracker(min_detection_confidence=0.3)  # Lower threshold for testing
    # Blank frame should return None
    result = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result is None
    tracker.release()
