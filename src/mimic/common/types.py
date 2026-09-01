"""Shared data types used across all modules."""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import List, Mapping, Optional, Tuple

import numpy as np


class ActionPhase(str, Enum):
    """Deployment-default composite skill labels."""

    IDLE = "IDLE"
    HOVER = "HOVER"
    GRASP = "GRASP"
    CARRY = "CARRY"
    RELEASE = "RELEASE"


@dataclass
class ObjectTrack:
    """One object observation at a one-based source-video frame.

    table_xy_m is already image-to-table calibrated: meters, top-left
    origin, +X right and +Y down. Pixel coordinates are not accepted here.
    """

    frame_idx: int
    table_xy_m: Tuple[float, float]
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
    """Phase prediction keyed by the same one-based source frame as tracking.

    timestamp, when available, remains video time in seconds, never a frame
    counter. Offline task extraction uses frame_idx and does not require FPS.
    """

    frame_idx: int
    phase: ActionPhase
    confidence: float
    timestamp: Optional[float] = None  # seconds, not inferred from frame_idx


def _source_frame(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("frame_idx must be a positive, one-based source-frame integer")
    return int(value)


def _finite_xy(value: Tuple[float, float]) -> Tuple[float, float]:
    if np.shape(value) != (2,) or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value
    ):
        raise ValueError("coordinates must contain exactly two finite real numbers")
    if not np.all(np.isfinite(value)):
        raise ValueError("coordinates must contain exactly two finite real numbers")
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class PhaseBoundary:
    """Inclusive phase onset in the original video; the next onset ends it."""

    phase: ActionPhase
    frame_idx: int
    timestamp_s: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "phase", ActionPhase(self.phase))
        object.__setattr__(self, "frame_idx", _source_frame(self.frame_idx))
        if self.timestamp_s is not None:
            if (
                isinstance(self.timestamp_s, (bool, np.bool_))
                or not isinstance(self.timestamp_s, Real)
                or not np.isfinite(self.timestamp_s)
                or self.timestamp_s < 0
            ):
                raise ValueError("timestamp_s must be finite, nonnegative seconds or None")


@dataclass(frozen=True)
class TablePathSample:
    """Detached table-space observation; phase follows the supplied boundaries."""

    frame_idx: int
    table_xy_m: Tuple[float, float]
    phase: ActionPhase
    confidence: float

    def __post_init__(self):
        object.__setattr__(self, "frame_idx", _source_frame(self.frame_idx))
        object.__setattr__(self, "table_xy_m", _finite_xy(self.table_xy_m))
        object.__setattr__(self, "phase", ActionPhase(self.phase))
        if (
            isinstance(self.confidence, (bool, np.bool_))
            or not isinstance(self.confidence, Real)
            or not np.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("tracker confidence must be finite and between 0 and 1")


@dataclass(frozen=True)
class ExtractedTask:
    """One complete pick/place in table meters; no robot assumptions.

    demonstration includes both GRASP and RELEASE onset observations. Samples
    may be irregularly spaced; no missing observations are interpolated.
    Endpoints are derived from the preserved samples, never stored separately.
    """

    phase_boundaries: Tuple[PhaseBoundary, ...]
    demonstrated_path: Tuple[TablePathSample, ...]
    object_id: Optional[str] = None

    def __post_init__(self):
        boundaries = tuple(self.phase_boundaries)
        samples = tuple(self.demonstrated_path)
        expected = (
            ActionPhase.IDLE,
            ActionPhase.HOVER,
            ActionPhase.GRASP,
            ActionPhase.CARRY,
            ActionPhase.RELEASE,
            ActionPhase.HOVER,
            ActionPhase.IDLE,
        )
        if tuple(b.phase for b in boundaries) != expected:
            raise ValueError(
                "Task requires IDLE -> HOVER -> GRASP -> CARRY -> RELEASE -> HOVER -> IDLE "
                "boundaries"
            )
        if any(a.frame_idx >= b.frame_idx for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError("Phase boundaries must have strictly increasing frame IDs")
        if len(samples) < 2 or any(
            a.frame_idx >= b.frame_idx for a, b in zip(samples, samples[1:])
        ):
            raise ValueError("Path requires strictly increasing frames and both endpoint samples")
        if samples[0].frame_idx != boundaries[2].frame_idx or (
            samples[-1].frame_idx != boundaries[4].frame_idx
        ):
            raise ValueError("Path endpoints must exactly match GRASP and RELEASE onset frames")
        for sample in samples:
            expected_phase = next(
                b.phase for b in reversed(boundaries) if b.frame_idx <= sample.frame_idx
            )
            if sample.phase != expected_phase:
                raise ValueError("Path sample phase does not match the phase boundaries")
        if self.object_id is not None and (
            not isinstance(self.object_id, str) or not self.object_id.strip()
        ):
            raise ValueError("object_id must be a nonempty string or None")
        object.__setattr__(self, "phase_boundaries", boundaries)
        object.__setattr__(self, "demonstrated_path", samples)

    @property
    def coordinate_frame(self) -> str:
        return "table"

    @property
    def start_xy_m(self) -> Tuple[float, float]:
        return self.demonstrated_path[0].table_xy_m

    @property
    def goal_xy_m(self) -> Tuple[float, float]:
        return self.demonstrated_path[-1].table_xy_m

    @property
    def grasp_frame(self) -> int:
        return self.phase_boundaries[2].frame_idx

    @property
    def release_frame(self) -> int:
        return self.phase_boundaries[4].frame_idx

    @property
    def carry_trajectory_xy_m(self) -> Tuple[Tuple[float, float], ...]:
        """CARRY-only observations; may be empty when tracking is sparse."""
        return tuple(s.table_xy_m for s in self.demonstrated_path if s.phase == ActionPhase.CARRY)

    @property
    def path_xy_m(self) -> Tuple[Tuple[float, float], ...]:
        """Every retained observation; path selection belongs downstream."""
        return tuple(s.table_xy_m for s in self.demonstrated_path)


@dataclass(frozen=True)
class RetargetedTask:
    """Target-frame XY meters, paired one-for-one with immutable source samples.

    This is object-path geometry, not tool poses or executable robot commands.
    Frame IDs, phases, confidence and original coordinates remain in source_task.
    """

    source_task: ExtractedTask
    target_frame: str
    demonstrated_path_xy_m: Tuple[Tuple[float, float], ...]

    def __post_init__(self):
        if not isinstance(self.target_frame, str) or not self.target_frame.strip():
            raise ValueError("target_frame must be a nonempty name")
        points = tuple(_finite_xy(p) for p in self.demonstrated_path_xy_m)
        if len(points) != len(self.source_task.demonstrated_path):
            raise ValueError("Retargeting must preserve every source-path sample")
        object.__setattr__(self, "demonstrated_path_xy_m", points)

    @property
    def start_xy_m(self) -> Tuple[float, float]:
        return self.demonstrated_path_xy_m[0]

    @property
    def goal_xy_m(self) -> Tuple[float, float]:
        return self.demonstrated_path_xy_m[-1]

    @property
    def carry_trajectory_xy_m(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(
            point
            for point, sample in zip(
                self.demonstrated_path_xy_m, self.source_task.demonstrated_path
            )
            if sample.phase == ActionPhase.CARRY
        )

    @property
    def path_xy_m(self) -> Tuple[Tuple[float, float], ...]:
        """Every mapped observation; path selection belongs to PathProcessor."""
        return self.demonstrated_path_xy_m


@dataclass
class TaskRepresentation:
    """Symbolic task representation independent of robot embodiment."""

    actions: List[str]  # Composite skill labels from the active SkillCatalog.
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

    homography: np.ndarray  # 3x3 image-pixel to table-meter transform
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix
    table_corners_image: List[Tuple[float, float]]  # 4 corners in image space
    table_corners_world: List[Tuple[float, float]]  # 4 corners in world space
    table_height: float  # z offset from world origin


@dataclass(frozen=True)
class ToolPose:
    """Tool pose in MuJoCo world: meters and unit quaternion (w, x, y, z)."""

    position: Tuple[float, float, float]
    quaternion_wxyz: Tuple[float, float, float, float]

    def __post_init__(self):
        if np.shape(self.position) != (3,) or not np.all(np.isfinite(self.position)):
            raise ValueError("position must contain three finite meters")
        if np.shape(self.quaternion_wxyz) != (4,) or not np.all(np.isfinite(self.quaternion_wxyz)):
            raise ValueError("quaternion must contain four finite values in wxyz order")
        if not np.isclose(np.linalg.norm(self.quaternion_wxyz), 1.0, rtol=0, atol=1e-8):
            raise ValueError("quaternion must be unit length; no implicit normalization")
        object.__setattr__(self, "position", tuple(map(float, self.position)))
        object.__setattr__(self, "quaternion_wxyz", tuple(map(float, self.quaternion_wxyz)))


@dataclass(frozen=True)
class GripperFeedback:
    """Measured nominal opening (m), speed (m/s), and target-object contact forces (N)."""

    width_m: float
    speed_m_s: float
    finger_contact_forces_n: Tuple[float, ...]


@dataclass(frozen=True)
class RobotState:
    """Detached observation; named joint coordinates include non-arm joints.

    Hinge/slide coordinates have length one; ball/free joints retain their model
    coordinate representation. No live simulation arrays are shared.
    """

    timestamp_s: float
    joint_positions: Mapping[str, Tuple[float, ...]]
    tool_pose: ToolPose
    gripper: GripperFeedback
    object_position: Optional[Tuple[float, float, float]] = None


class IKStatus(str, Enum):
    VALID_STEP = "VALID_STEP"
    AT_TARGET = "AT_TARGET"
    INVALID_INPUT = "INVALID_INPUT"
    LIMIT_VIOLATION = "LIMIT_VIOLATION"
    SOLVER_FAILED = "SOLVER_FAILED"


@dataclass(frozen=True)
class IKResult:
    """One bounded control step; AT_TARGET refers to the measured configuration."""

    status: IKStatus
    joint_targets: Mapping[str, float]
    position_error_m: Optional[float]
    orientation_error_rad: Optional[float]
    solve_time_s: float
    detail: str = ""

    @property
    def valid(self) -> bool:
        return self.status in (IKStatus.VALID_STEP, IKStatus.AT_TARGET)


@dataclass(frozen=True)
class PickPlaceWaypoints:
    """Already retargeted/processed tool poses; geometry is supplied, never inferred.

    Task extraction and source-coordinate conventions remain upstream.
    goal_position is the intended object center, not the tool position.
    """

    approach: ToolPose
    grasp: ToolPose
    lift: ToolPose
    path: Tuple[ToolPose, ...]
    lower: ToolPose
    retreat: ToolPose
    goal_position: Tuple[float, float, float]

    def __post_init__(self):
        if not self.path:
            raise ValueError("CARRY requires a nonempty processed path")
        if np.shape(self.goal_position) != (3,) or not np.all(np.isfinite(self.goal_position)):
            raise ValueError("Object goal must contain three finite world coordinates")
        poses = (self.approach, self.grasp, self.lift, *self.path, self.lower, self.retreat)
        reference = np.array(self.approach.quaternion_wxyz)
        if any(
            not np.isclose(abs(np.dot(reference, p.quaternion_wxyz)), 1, rtol=0, atol=1e-8)
            for p in poses
        ):
            raise ValueError("The MVP requires one fixed tool orientation")
