#!/usr/bin/env python3
"""Interactive camera calibration tool for coordinate mapping.

Records table corner points in image space and maps them to world coordinates
to compute a perspective transformation. Saves calibration for later use.

Usage:
    uv run python scripts/calibrate_camera.py \\
        --image data/raw/calibration_frame.png \\
        --width 0.6 \\
        --height 0.4 \\
        --output data/annotations/calibration.json
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from mimic.tracking import CoordinateMapper


class CalibrationUI:
    """Interactive UI for clicking table corners."""

    def __init__(self, image_path: str):
        """Initialize calibration UI with an image.

        Args:
            image_path: Path to calibration frame (image or video).
        """
        self.image_path = image_path
        self.image = None
        self.display = None
        self.corners_image = []
        self.window_name = "Calibration: Click the 4 table corners in order (TL, TR, BL, BR)"

    def load_image(self) -> bool:
        """Load image from file or extract first frame from video.

        Returns:
            True if successful, False otherwise.
        """
        path = Path(self.image_path)

        if path.suffix.lower() in [".mov", ".mp4", ".avi", ".mkv"]:
            # Video file: extract first frame
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"Error: Could not open video {path}")
                return False
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Error: Could not read first frame from {path}")
                return False
            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Image file
            self.image = cv2.imread(str(path))
            if self.image is None:
                print(f"Error: Could not load image {path}")
                return False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Create display copy
        self.display = self.image.copy()
        return True

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for corner clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ignore clicks after 4 corners are recorded
            if len(self.corners_image) >= 4:
                return

            self.corners_image.append((float(x), float(y)))
            # Draw circle at clicked point
            cv2.circle(self.display, (x, y), 5, (0, 255, 0), -1)
            # Draw label
            label = ["TL", "TR", "BL", "BR"][len(self.corners_image) - 1]
            cv2.putText(
                self.display,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(self.window_name, self.display)
            print(f"Clicked corner {len(self.corners_image)}: ({x}, {y})")

            if len(self.corners_image) == 4:
                print("\n✓ All 4 corners recorded! Proceeding to calibration...")
                import time
                time.sleep(1)
                cv2.destroyAllWindows()
                return True

    def run(self) -> bool:
        """Run interactive corner selection.

        Returns:
            True if calibration completed, False if cancelled.
        """
        if not self.load_image():
            return False

        print(f"\nImage size: {self.image.shape[1]} x {self.image.shape[0]} pixels")
        print("Click the 4 table corners in this order:")
        print("  1. Top-left (TL)")
        print("  2. Top-right (TR)")
        print("  3. Bottom-left (BL)")
        print("  4. Bottom-right (BR)")
        print("\nPress ESC to cancel or 'r' to restart.\n")

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display)

        while len(self.corners_image) < 4:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                print("Cancelled.")
                cv2.destroyAllWindows()
                return False
            elif key == ord("r"):  # Reset
                print("Restarting...")
                self.corners_image = []
                self.display = self.image.copy()
                cv2.imshow(self.window_name, self.display)


def main():
    """Main calibration workflow."""
    parser = argparse.ArgumentParser(
        description="Interactive camera calibration for table coordinate mapping"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to calibration frame (image or video file)",
    )
    parser.add_argument(
        "--width",
        type=float,
        required=True,
        help="Table width in meters (e.g., 0.6)",
    )
    parser.add_argument(
        "--height",
        type=float,
        required=True,
        help="Table height/depth in meters (e.g., 0.4)",
    )
    parser.add_argument(
        "--output",
        default="data/annotations/calibration.json",
        help="Output path for calibration JSON",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return 1

    if args.width <= 0 or args.height <= 0:
        print("Error: Table dimensions must be positive")
        return 1

    # Run calibration UI
    print("\n" + "=" * 60)
    print("CAMERA CALIBRATION TOOL")
    print("=" * 60)
    print(f"\nTable dimensions: {args.width}m × {args.height}m")
    print(f"Calibration file: {args.output}\n")

    ui = CalibrationUI(args.image)
    if not ui.run():
        return 1

    # Compute homography
    print("\nCalibrating...")
    mapper = CoordinateMapper(table_width_m=args.width, table_height_m=args.height)

    # Table corners in world space (meters)
    table_corners_world = [
        (0.0, 0.0),  # TL
        (args.width, 0.0),  # TR
        (0.0, args.height),  # BL
        (args.width, args.height),  # BR
    ]

    frame = ui.image
    if frame.shape[2] == 3 and frame.dtype == np.uint8:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame

    calib = mapper.calibrate(frame_bgr, ui.corners_image, table_corners_world)

    # Save calibration
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mapper.save(args.output, calib)

    print(f"\n✓ Calibration saved to: {args.output}")
    print("\nCalibration details:")
    print(f"  Image corners: {ui.corners_image}")
    print(f"  World corners: {table_corners_world}")
    print(f"  Homography shape: {mapper.homography.shape}")

    # Test: map corners back to verify
    print("\nVerification (mapping image corners back to world):")
    for i, corner_img in enumerate(ui.corners_image):
        corner_world = mapper.pixel_to_workspace(corner_img)
        expected_world = table_corners_world[i]
        error = np.sqrt((corner_world[0] - expected_world[0]) ** 2 +
                       (corner_world[1] - expected_world[1]) ** 2)
        print(f"  Corner {i}: {corner_img} → {corner_world} (error: {error:.4f}m)")

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)
    print("\nNext step: Extract tracks from your demo videos:")
    print(f"  uv run python scripts/extract_tracks.py \\")
    print(f"    --calibration {args.output} \\")
    print(f"    --output-dir data/tracks/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
