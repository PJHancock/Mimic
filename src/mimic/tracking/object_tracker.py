"""Object tracking using OpenCV CSRT with color-based initialization."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mimic.common.types import ObjectTrack


class ObjectTracker(ABC):
    """Abstract base for object tracking."""

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Initialize tracker with a bounding box.

        Args:
            frame: RGB frame.
            bbox: (x, y, width, height) bounding box in pixels.
        """

    @abstractmethod
    def update(self, frame: np.ndarray, frame_idx: int = 0) -> ObjectTrack:
        """Update tracker and return object position.

        Args:
            frame: RGB frame.
            frame_idx: Frame index for tracking.

        Returns:
            ObjectTrack with current bounding box and confidence.
        """


class CSRTObjectTracker(ObjectTracker):
    """CSRT-based object tracker.

    Stateful across frames: initialize once with a bbox, then update per frame.
    """

    def __init__(self):
        """Initialize CSRT tracker."""
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Initialize tracker with bounding box.

        Args:
            frame: RGB or BGR frame.
            bbox: (x, y, width, height).
        """
        # Convert RGB to BGR if needed (OpenCV expects BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        self.tracker.init(frame_bgr, bbox)
        self.initialized = True

    def update(self, frame: np.ndarray, frame_idx: int = 0) -> ObjectTrack:
        """Update tracker with new frame.

        Args:
            frame: RGB or BGR frame.
            frame_idx: Frame index.

        Returns:
            ObjectTrack with bbox and confidence.
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")

        # Convert RGB to BGR if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        success, bbox = self.tracker.update(frame_bgr)
        x, y, w, h = bbox
        center_2d = (x + w / 2, y + h / 2)
        confidence = 1.0 if success else 0.0

        return ObjectTrack(
            frame_idx=frame_idx,
            center_2d=center_2d,
            bbox=(x, y, w, h),
            confidence=confidence,
        )


def find_initial_bbox(
    frame: np.ndarray,
    hsv_lower: List[int],
    hsv_upper: List[int],
    hsv_lower_wrap: Optional[List[int]] = None,
    hsv_upper_wrap: Optional[List[int]] = None,
    min_contour_area: int = 500,
) -> Optional[Tuple[int, int, int, int]]:
    """Detect object bounding box via HSV color thresholding.

    Used to auto-detect red Solo cup (or similar) in frame 0.

    Args:
        frame: RGB frame.
        hsv_lower: [H_min, S_min, V_min] for primary color range.
        hsv_upper: [H_max, S_max, V_max] for primary color range.
        hsv_lower_wrap: [H_min, S_min, V_min] for wrap-around range (e.g., red at 170-180).
        hsv_upper_wrap: [H_max, S_max, V_max] for wrap-around range.
        min_contour_area: Minimum area to consider valid detection.

    Returns:
        Bounding box (x, y, w, h) if object found, None otherwise.
    """
    # Convert RGB to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Create mask for primary range
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lower, upper)

    # Add wrap-around range if provided (red wraps at hue boundary)
    if hsv_lower_wrap is not None and hsv_upper_wrap is not None:
        lower_wrap = np.array(hsv_lower_wrap, dtype=np.uint8)
        upper_wrap = np.array(hsv_upper_wrap, dtype=np.uint8)
        mask_wrap = cv2.inRange(frame_hsv, lower_wrap, upper_wrap)
        mask = cv2.bitwise_or(mask, mask_wrap)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_contour_area:
        return None

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)
