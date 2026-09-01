"""Object tracking and camera calibration for tabletop demonstrations."""

from .coordinate_mapping import CoordinateMapper
from .object_tracker import CSRTObjectTracker, ObjectTracker, find_initial_bbox
from .types import ImageObjectTrack

__all__ = [
    "CoordinateMapper",
    "CSRTObjectTracker",
    "ImageObjectTrack",
    "ObjectTracker",
    "find_initial_bbox",
]
