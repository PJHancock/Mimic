"""Shared data types used across all modules."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ActionPhase(str, Enum):
    """Manipulation action phases."""

    APPROACH = "APPROACH"
    GRASP = "GRASP"
    MOVE = "MOVE"
    RELEASE = "RELEASE"


@dataclass
class ObjectTrack:
    """Tracked object position and properties over time."""

    frame_idx: int
    center_2d: Tuple[float, float]  # (x, y) in image coordinates
    center_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z) in world coords
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)
    confidence: float = 1.0
    object_id: Optional[str] = None


@dataclass
class HandTrack:
    """Tracked hand position and properties over time."""

    frame_idx: int
    wrist_2d: Tuple[float, float]  # (x, y) in image coordinates
    wrist_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z) in world coords
    fingertips_2d: Optional[List[Tuple[float, float]]] = None  # 5 fingertips
    finger_closure: Optional[float] = None  # 0.0 = open, 1.0 = closed
    confidence: float = 1.0


@dataclass
class VideoClip:
    """A segment of video with temporal boundaries."""

    video_path: str
    start_frame: int
    end_frame: int
    fps: float


@dataclass
class ActionPrediction:
    """Temporal model prediction for a video frame."""

    frame_idx: int
    phase: ActionPhase
    confidence: float
    timestamp: float  # seconds


@dataclass
class TaskRepresentation:
    """Symbolic task representation independent of robot embodiment."""

    actions: List[str]  # ["APPROACH", "GRASP", "MOVE", "RELEASE", ...]
    object_start: Tuple[float, float]  # normalized workspace coords
    object_end: Tuple[float, float]
    trajectory: List[Tuple[float, float]]  # waypoints in normalized workspace
    approach_height: float = 0.1  # meters above object
    grasp_height: float = 0.0
    transport_height: float = 0.1
    release_height: float = 0.0


@dataclass
class RobotCommand:
    """Robot control command (Panda-specific)."""

    phase: ActionPhase
    target_position: Tuple[float, float, float]  # Cartesian coordinates
    target_orientation: Optional[Tuple[float, float, float, float]] = None  # quaternion
    gripper_open: bool = True
    duration: float = 1.0  # seconds
    trajectory_points: Optional[List[Tuple[float, float, float]]] = None


@dataclass
class CalibrationData:
    """Camera to table coordinate mapping."""

    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    table_corners_image: List[Tuple[float, float]]  # 4 corners in image space
    table_corners_world: List[Tuple[float, float]]  # 4 corners in world space
    table_height: float  # z offset from world origin
