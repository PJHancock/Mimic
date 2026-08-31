"""MediaPipe-based hand tracking for manipulation tasks."""

from typing import Optional

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

from mimic.common.types import HandTrack


class HandTracker:
    """Detects hand landmarks using MediaPipe HandLandmarker.

    Per frame, outputs wrist position, fingertips, and finger closure.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipe hand landmark detector.

        Args:
            min_detection_confidence: Threshold for hand detection (0-1).
        """
        options = HandLandmarkerOptions(
            base_options=vision.BaseOptions(model_asset_path=""),  # Use bundled model
            running_mode=RunningMode.IMAGE,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, frame_idx: int = 0) -> Optional[HandTrack]:
        """Detect hand landmarks in a frame.

        Args:
            frame: RGB frame (H, W, 3) with values in [0, 255].
            frame_idx: Frame index for tracking.

        Returns:
            HandTrack if hand detected, None otherwise.
        """
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            # Already RGB, uint8 — good
            pass
        else:
            # Convert if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        # Create MediaPipe Image from numpy array
        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame)

        # Detect hand landmarks
        results = self.detector.detect(mp_image)
        if not results.hand_landmarks:
            return None

        landmarks = results.hand_landmarks[0]
        h, w = frame.shape[:2]

        # Extract wrist (landmark 0)
        wrist_landmark = landmarks[0]
        wrist_2d = (wrist_landmark.x * w, wrist_landmark.y * h)

        # Extract fingertips (landmarks 4, 8, 12, 16, 20)
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertips_2d = [
            (landmarks[i].x * w, landmarks[i].y * h)
            for i in fingertip_indices
        ]

        # Calculate finger closure (0 = closed fist, 1 = open hand)
        finger_closure = self._calculate_finger_closure(landmarks, h, w)

        # Confidence from MediaPipe (handedness confidence)
        confidence = results.handedness[0][0].score if results.handedness else 0.5

        return HandTrack(
            frame_idx=frame_idx,
            wrist_2d=wrist_2d,
            fingertips_2d=fingertips_2d,
            finger_closure=finger_closure,
            confidence=float(confidence),
        )

    def _calculate_finger_closure(self, landmarks, h: int, w: int) -> float:
        """Calculate normalized finger closure metric.

        Returns 0 when fist is closed, 1 when hand is open.
        Based on average distance from fingertips to palm center.
        """
        # Palm center approximation: wrist + middle MCP
        wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y])
        middle_mcp = np.array([landmarks.landmark[9].x, landmarks.landmark[9].y])
        palm_center = (wrist + middle_mcp) / 2

        # Hand size: distance from wrist to middle MCP (scale invariant)
        hand_size = np.linalg.norm(middle_mcp - wrist)
        if hand_size < 1e-5:
            return 0.0

        # Average distance from fingertips to palm center, normalized by hand size
        fingertip_indices = [4, 8, 12, 16, 20]
        distances = []
        for i in fingertip_indices:
            fingertip = np.array(
                [landmarks.landmark[i].x, landmarks.landmark[i].y]
            )
            dist = np.linalg.norm(fingertip - palm_center)
            distances.append(dist)

        avg_distance = np.mean(distances)
        # Normalize: closed fist ~0.05*hand_size, open hand ~0.35*hand_size
        finger_closure = np.clip((avg_distance - 0.05 * hand_size) / (0.3 * hand_size), 0, 1)

        return float(finger_closure)

    def release(self) -> None:
        """Clean up MediaPipe resources."""
        self.detector.close() if hasattr(self.detector, "close") else None
