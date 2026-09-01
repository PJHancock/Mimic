"""Tracking module: hand and object tracking with coordinate mapping."""

from typing import List, Tuple

from mimic.common.types import CalibrationData, HandTrack, ObjectTrack

from .coordinate_mapping import CoordinateMapper
from .hand_tracker import HandTracker
from .object_tracker import CSRTObjectTracker, ObjectTracker, find_initial_bbox
from .types import ImageObjectTrack
from .trajectory import (
    interpolate_gaps,
    process_trajectory,
    resample_trajectory,
    smooth_trajectory,
)

__all__ = [
    "HandTracker",
    "ObjectTracker",
    "CSRTObjectTracker",
    "ImageObjectTrack",
    "find_initial_bbox",
    "CoordinateMapper",
    "interpolate_gaps",
    "smooth_trajectory",
    "resample_trajectory",
    "process_trajectory",
    "track_demonstration",
]


def track_demonstration(
    video_path: str,
    calibration: CalibrationData,
    hand_confidence_threshold: float = 0.5,
    object_confidence_threshold: float = 0.5,
    num_trajectory_waypoints: int = 30,
    hsv_lower: List[int] = None,
    hsv_upper: List[int] = None,
    hsv_lower_wrap: List[int] = None,
    hsv_upper_wrap: List[int] = None,
    min_contour_area: int = 500,
) -> Tuple[List[HandTrack], List[ObjectTrack], List[Tuple[float, float]]]:
    """End-to-end demonstration tracking.

    1. Open video file
    2. For each frame: detect hand + object
    3. Map pixels to calibrated table coordinates
    4. Process object trajectory
    5. Return tracks and trajectory

    Args:
        video_path: Path to demo video.
        calibration: CalibrationData with homography.
        hand_confidence_threshold: MediaPipe confidence threshold.
        object_confidence_threshold: Tracker confidence threshold.
        num_trajectory_waypoints: Number of waypoints in output trajectory.
        hsv_lower: HSV lower bounds for object detection.
        hsv_upper: HSV upper bounds for object detection.
        hsv_lower_wrap: HSV lower bounds for wrap-around range.
        hsv_upper_wrap: HSV upper bounds for wrap-around range.
        min_contour_area: Minimum contour area for valid detection.

    Returns:
        ``object_tracks`` use one-based source frames and calibrated table
        centimeters. ``trajectory`` contains processed table-centimeter points.
    """
    import cv2

    from mimic.config import get_config

    # Use default config values if not provided
    if hsv_lower is None:
        cfg = get_config()
        tracking_cfg = cfg.get("tracking", {})
        hsv_lower = tracking_cfg.get("object_color_hsv_lower", [0, 100, 100])
        hsv_upper = tracking_cfg.get("object_color_hsv_upper", [10, 255, 255])
        hsv_lower_wrap = tracking_cfg.get("object_color_hsv_lower_wrap", [170, 100, 100])
        hsv_upper_wrap = tracking_cfg.get("object_color_hsv_upper_wrap", [180, 255, 255])
        min_contour_area = tracking_cfg.get("min_contour_area", 500)
        num_trajectory_waypoints = tracking_cfg.get("num_trajectory_waypoints", 30)

    # Initialize trackers
    hand_tracker = HandTracker(min_detection_confidence=hand_confidence_threshold)
    object_tracker = CSRTObjectTracker()
    coordinate_mapper = CoordinateMapper(
        table_width_m=calibration.table_corners_world[1][0]
        - calibration.table_corners_world[0][0],
        table_height_m=calibration.table_corners_world[2][1]
        - calibration.table_corners_world[0][1],
    )
    coordinate_mapper.homography = calibration.camera_matrix  # This should be the homography
    coordinate_mapper.is_calibrated = True

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    hand_tracks = []
    object_tracks_raw = []
    frame_idx = 0
    object_tracker_init = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        hand_track = hand_tracker.process(frame_rgb, frame_idx)
        if hand_track:
            hand_tracks.append(hand_track)

        # Detect/track object
        if not object_tracker_init:
            # Initialize with color detection on frame 0
            bbox = find_initial_bbox(
                frame_rgb,
                hsv_lower,
                hsv_upper,
                hsv_lower_wrap,
                hsv_upper_wrap,
                min_contour_area,
            )
            if bbox:
                object_tracker.init(frame_rgb, bbox)
                object_tracker_init = True

        if object_tracker_init:
            obj_track = object_tracker.update(frame_rgb, frame_idx)
            object_tracks_raw.append(obj_track)

        frame_idx += 1

    cap.release()
    hand_tracker.release()

    # Map coordinates and process trajectory
    object_tracks_mapped = []
    for track in object_tracks_raw:
        table_xy_cm = coordinate_mapper.pixel_to_table_xy_cm(track.center_2d)

        mapped_track = ObjectTrack(
            frame_idx=track.frame_idx + 1,
            table_xy_cm=table_xy_cm,
            bbox=track.bbox,
            confidence=track.confidence,
        )
        object_tracks_mapped.append(mapped_track)

    # Process trajectory (interpolate, smooth, resample)
    trajectory_processed = process_trajectory(
        object_tracks_raw, num_trajectory_waypoints
    )

    trajectory_table_xy_cm = [
        coordinate_mapper.pixel_to_table_xy_cm(point)
        for point in trajectory_processed
    ]

    return hand_tracks, object_tracks_mapped, trajectory_table_xy_cm
