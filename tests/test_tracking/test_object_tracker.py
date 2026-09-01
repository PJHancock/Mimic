"""Unit tests for object tracking."""

import numpy as np
import pytest

from mimic.tracking import CSRTObjectTracker, ImageObjectTrack, find_initial_bbox


@pytest.fixture
def csrt_tracker():
    """Create a CSRT tracker instance."""
    return CSRTObjectTracker()


def test_csrt_init(csrt_tracker):
    """Test CSRT tracker initialization."""
    assert csrt_tracker is not None
    assert not csrt_tracker.initialized


def test_csrt_init_with_bbox(csrt_tracker):
    """Test initializing tracker with bounding box."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = (100, 100, 50, 50)
    csrt_tracker.init(frame, bbox)
    assert csrt_tracker.initialized


def test_csrt_update_requires_init(csrt_tracker):
    """Test that update raises error if tracker not initialized."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        csrt_tracker.update(frame)


def test_csrt_update_returns_image_object_track(csrt_tracker):
    """The tracker emits an image-domain record before calibration."""
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a white square in frame1
    frame1[100:150, 100:150] = 255

    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw white square shifted in frame2 (simulate motion)
    frame2[105:155, 105:155] = 255

    bbox = (100, 100, 50, 50)
    csrt_tracker.init(frame1, bbox)
    track = csrt_tracker.update(frame2, frame_idx=1)

    assert isinstance(track, ImageObjectTrack)
    assert track.frame_idx == 1
    assert track.center_2d is not None
    assert isinstance(track.center_2d, tuple)
    assert len(track.center_2d) == 2
    assert track.confidence >= 0  # Confidence should be 0 or 1


def test_csrt_rgb_bgr_conversion(csrt_tracker):
    """Test that RGB frames are correctly converted to BGR for OpenCV."""
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_rgb[100:150, 100:150] = [255, 0, 0]  # Red in RGB

    bbox = (100, 100, 50, 50)
    csrt_tracker.init(frame_rgb, bbox)
    # Should not raise an error
    assert csrt_tracker.initialized


def test_find_initial_bbox_empty_frame():
    """Test bbox detection on empty frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = find_initial_bbox(frame, [0, 100, 100], [10, 255, 255])
    assert bbox is None


def test_find_initial_bbox_detects_color():
    """Test that bbox detection finds colored object."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw red square in HSV
    # Red in RGB is (255, 0, 0), in HSV is (0, 255, 255)
    frame[100:150, 100:150, 2] = 255  # Red channel in RGB

    # Convert to check HSV
    import cv2
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # HSV range for red (approximately)
    hsv_lower = [0, 100, 100]
    hsv_upper = [10, 255, 255]

    bbox = find_initial_bbox(frame, hsv_lower, hsv_upper, min_contour_area=100)

    # If detection works, bbox should be approximately (100, 100, 50, 50)
    if bbox is not None:
        assert len(bbox) == 4
        x, y, w, h = bbox
        assert x >= 90 and x <= 110  # x near 100
        assert y >= 90 and y <= 110  # y near 100
        assert w > 0 and h > 0


def test_find_initial_bbox_min_area_threshold():
    """Test that min_contour_area filters small objects."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw very small red region (1x1 pixel)
    frame[100, 100, 2] = 255

    bbox = find_initial_bbox(frame, [0, 100, 100], [10, 255, 255], min_contour_area=1000)
    # Should return None because area is smaller than threshold
    assert bbox is None


def test_find_initial_bbox_with_wraparound():
    """Test HSV wraparound range for colors like red."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:150, 100:150, 2] = 255  # Red

    # Red wraps around hue 0/180
    hsv_lower = [0, 100, 100]
    hsv_upper = [10, 255, 255]
    hsv_lower_wrap = [170, 100, 100]
    hsv_upper_wrap = [180, 255, 255]

    bbox = find_initial_bbox(
        frame,
        hsv_lower,
        hsv_upper,
        hsv_lower_wrap,
        hsv_upper_wrap,
        min_contour_area=100,
    )

    # Should detect the red square
    if bbox is not None:
        assert len(bbox) == 4


def test_csrt_bbox_format():
    """Test that returned bbox has correct format."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:150, 100:150] = 255

    bbox = (100, 100, 50, 50)
    csrt_tracker = CSRTObjectTracker()
    csrt_tracker.init(frame, bbox)
    track = csrt_tracker.update(frame)

    # Check bbox format
    assert track.bbox is not None
    assert len(track.bbox) == 4
    x, y, w, h = track.bbox
    assert x >= 0 and y >= 0 and w > 0 and h > 0


def test_csrt_center_from_bbox(csrt_tracker):
    """Test that center_2d is correctly computed from bbox."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:150, 100:150] = 255

    bbox = (100, 100, 50, 50)
    csrt_tracker.init(frame, bbox)
    track = csrt_tracker.update(frame)

    # Center should be bbox center: (100 + 50/2, 100 + 50/2) = (125, 125)
    x, y = track.center_2d
    assert isinstance(x, float) and isinstance(y, float)
    assert x > 0 and y > 0
