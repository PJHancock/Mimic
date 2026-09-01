"""Camera to table to robot workspace coordinate mapping via perspective transform."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mimic.common.constants import PANDA_WORKSPACE_X_MAX, PANDA_WORKSPACE_X_MIN, PANDA_WORKSPACE_Y_MAX, PANDA_WORKSPACE_Y_MIN
from mimic.common.types import CalibrationData


class CoordinateMapper:
    """Maps image pixels → table coordinates → normalized robot workspace.

    One-time calibration per physical setup.
    """

    def __init__(self, table_width_m: float, table_height_m: float):
        """Initialize mapper with table dimensions.

        Args:
            table_width_m: Table width in meters.
            table_height_m: Table depth/height in meters.
        """
        self.table_width_m = table_width_m
        self.table_height_m = table_height_m
        self.homography: Optional[np.ndarray] = None
        self.is_calibrated = False

    def calibrate(
        self,
        frame: np.ndarray,
        table_corners_image: List[Tuple[float, float]],
        table_corners_world: List[Tuple[float, float]],
        camera_matrix: Optional[np.ndarray] = None,
    ) -> CalibrationData:
        """Compute homography from image corners to world coordinates.

        Args:
            frame: Calibration frame (for intrinsics storage).
            table_corners_image: 4 corners in image space [(x1, y1), ..., (x4, y4)].
            table_corners_world: 4 corners in world/table space [(x1, y1), ..., (x4, y4)].
            camera_matrix: Optional camera intrinsics (3x3).

        Returns:
            CalibrationData with homography and corner mappings.
        """
        pts_img = np.float32(table_corners_image)
        pts_world = np.float32(table_corners_world)

        self.homography = cv2.getPerspectiveTransform(pts_img, pts_world)
        self.is_calibrated = True

        if camera_matrix is None:
            # Identity intrinsics if not provided
            camera_matrix = np.eye(3)

        calib = CalibrationData(
            camera_matrix=camera_matrix,
            table_corners_image=list(table_corners_image),
            table_corners_world=list(table_corners_world),
            table_height=0.0,  # Z offset from world origin (not used for tabletop)
        )

        return calib

    def load(self, path: str) -> None:
        """Load calibration from JSON file.

        Args:
            path: Path to calibration JSON.
        """
        with open(path) as f:
            data = json.load(f)

        self.homography = np.array(data["homography"])
        self.table_width_m = data["table_width_m"]
        self.table_height_m = data["table_height_m"]
        self.is_calibrated = True

    def save(self, path: str, calib: CalibrationData) -> None:
        """Save calibration to JSON file.

        Args:
            path: Path to save calibration.
            calib: CalibrationData with homography.
        """
        if self.homography is None:
            raise RuntimeError("No calibration to save. Call calibrate() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "homography": self.homography.tolist(),
            "camera_matrix": calib.camera_matrix.tolist(),
            "table_corners_image": calib.table_corners_image,
            "table_corners_world": calib.table_corners_world,
            "table_width_m": self.table_width_m,
            "table_height_m": self.table_height_m,
            "table_height": calib.table_height,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def pixel_to_workspace(
        self, pixel_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized robot workspace [0,1].

        Pipeline:
        1. pixel → table meters (via homography)
        2. table meters → normalized [0,1]
        3. normalized → robot workspace

        Args:
            pixel_point: (x, y) in image pixels.

        Returns:
            (x, y) in normalized robot workspace [0, 1] × [0, 1].
        """
        if not self.is_calibrated:
            raise RuntimeError("Mapper not calibrated. Call calibrate() or load() first.")

        # Apply homography: pixel → table meters
        pt_img = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_img, self.homography)[0][0]

        # Normalize to [0, 1] by table dimensions
        normalized_x = np.clip(pt_world[0] / self.table_width_m, 0, 1)
        normalized_y = np.clip(pt_world[1] / self.table_height_m, 0, 1)

        return (float(normalized_x), float(normalized_y))

    def pixel_to_table_xy_cm(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Map image pixels to calibrated table centimeters without clipping."""
        if not self.is_calibrated:
            raise RuntimeError("Mapper not calibrated. Call calibrate() or load() first.")
        pt_img = np.array(
            [[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32
        )
        point_m = cv2.perspectiveTransform(pt_img, self.homography)[0][0]
        return (float(point_m[0] * 100), float(point_m[1] * 100))

    def pixels_to_workspace_batch(
        self, pixel_points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Batch convert pixel coordinates to workspace.

        Args:
            pixel_points: List of (x, y) in image pixels.

        Returns:
            List of (x, y) in normalized workspace.
        """
        return [self.pixel_to_workspace(pt) for pt in pixel_points]

    def workspace_to_panda(
        self, workspace_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Convert normalized workspace [0,1] to Panda arm workspace.

        Args:
            workspace_point: (x, y) in [0, 1] × [0, 1].

        Returns:
            (x, y) in Panda workspace meters.
        """
        x_panda = PANDA_WORKSPACE_X_MIN + workspace_point[0] * (
            PANDA_WORKSPACE_X_MAX - PANDA_WORKSPACE_X_MIN
        )
        y_panda = PANDA_WORKSPACE_Y_MIN + workspace_point[1] * (
            PANDA_WORKSPACE_Y_MAX - PANDA_WORKSPACE_Y_MIN
        )

        return (float(x_panda), float(y_panda))
