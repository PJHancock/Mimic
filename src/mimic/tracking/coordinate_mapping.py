"""Camera-pixel to physical-table coordinate mapping via perspective transform."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mimic.common.types import CalibrationData


class CoordinateMapper:
    """Maps image pixels to metric or normalized physical-table coordinates.

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
        self.image_width_px: Optional[int] = None
        self.image_height_px: Optional[int] = None
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
        self.image_height_px, self.image_width_px = frame.shape[:2]
        self.is_calibrated = True

        if camera_matrix is None:
            # Identity intrinsics if not provided
            camera_matrix = np.eye(3)

        calib = CalibrationData(
            homography=self.homography.copy(),
            camera_matrix=camera_matrix,
            image_width_px=self.image_width_px,
            image_height_px=self.image_height_px,
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
        self.image_width_px = data.get("image_width_px")
        self.image_height_px = data.get("image_height_px")
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
            "image_width_px": calib.image_width_px,
            "image_height_px": calib.image_height_px,
            "table_corners_image": calib.table_corners_image,
            "table_corners_world": calib.table_corners_world,
            "table_width_m": self.table_width_m,
            "table_height_m": self.table_height_m,
            "table_height": calib.table_height,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def pixel_to_normalized_table(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized physical-table coordinates.

        Pipeline:
        1. pixel → table meters (via homography)
        2. table meters → normalized [0,1]

        Args:
            pixel_point: (x, y) in image pixels.

        Returns:
            (x, y) in clipped normalized table coordinates [0, 1] × [0, 1].
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

    def pixel_to_table_xy_m(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """Map image pixels to calibrated table meters without clipping."""
        if not self.is_calibrated:
            raise RuntimeError("Mapper not calibrated. Call calibrate() or load() first.")
        pt_img = np.array([[[float(pixel_point[0]), float(pixel_point[1])]]], dtype=np.float32)
        point_m = cv2.perspectiveTransform(pt_img, self.homography)[0][0]
        return (float(point_m[0]), float(point_m[1]))

    def pixels_to_normalized_table_batch(
        self, pixel_points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Batch convert pixel coordinates to normalized physical-table coordinates.

        Args:
            pixel_points: List of (x, y) in image pixels.

        Returns:
            List of (x, y) in normalized table coordinates.
        """
        return [self.pixel_to_normalized_table(pt) for pt in pixel_points]
