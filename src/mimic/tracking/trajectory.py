"""Trajectory processing: interpolation, smoothing, and resampling."""

from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from mimic.tracking.types import ImageObjectTrack


def interpolate_gaps(
    tracks: List[ImageObjectTrack],
) -> List[ImageObjectTrack]:
    """Fill tracking gaps via linear interpolation.

    When CSRT loses track (confidence=0), interpolate between nearest confident frames.

    Args:
        tracks: List of ImageObjectTrack from CSRT tracker.

    Returns:
        List of ImageObjectTrack with gaps filled.
    """
    if not tracks:
        return []

    # Find confidence gaps
    confidences = [t.confidence for t in tracks]
    frames = [t.frame_idx for t in tracks]

    # Convert to numpy for easier manipulation
    x_coords = np.array([t.center_2d[0] for t in tracks], dtype=float)
    y_coords = np.array([t.center_2d[1] for t in tracks], dtype=float)

    # Find indices where tracking is confident (non-zero)
    confident_idx = [i for i, c in enumerate(confidences) if c > 0]

    if len(confident_idx) < 2:
        # Not enough confident frames to interpolate
        return tracks

    # Interpolate missing values
    confident_frames = [frames[i] for i in confident_idx]
    confident_x = [x_coords[i] for i in confident_idx]
    confident_y = [y_coords[i] for i in confident_idx]

    f_x = interp1d(confident_frames, confident_x, kind="linear", fill_value="extrapolate")
    f_y = interp1d(confident_frames, confident_y, kind="linear", fill_value="extrapolate")

    x_interp = f_x(frames)
    y_interp = f_y(frames)

    # Rebuild tracks with interpolated positions
    interpolated = []
    for i, track in enumerate(tracks):
        new_track = ImageObjectTrack(
            frame_idx=track.frame_idx,
            center_2d=(float(x_interp[i]), float(y_interp[i])),
            bbox=track.bbox,
            confidence=track.confidence,
        )
        interpolated.append(new_track)

    return interpolated


def smooth_trajectory(
    tracks: List[ImageObjectTrack],
    window_length: int = 5,
    polyorder: int = 2,
) -> List[Tuple[float, float]]:
    """Apply Savitzky-Golay filter to smooth trajectory.

    Args:
        tracks: List of ImageObjectTrack (already interpolated).
        window_length: Window size for filter (must be odd).
        polyorder: Polynomial order for filter.

    Returns:
        Smoothed trajectory as [(x, y), ...].
    """
    if len(tracks) < window_length:
        # Not enough points; return as-is
        return [t.center_2d for t in tracks]

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    x_coords = np.array([t.center_2d[0] for t in tracks])
    y_coords = np.array([t.center_2d[1] for t in tracks])

    x_smooth = savgol_filter(x_coords, window_length, polyorder)
    y_smooth = savgol_filter(y_coords, window_length, polyorder)

    return list(zip(x_smooth, y_smooth))


def resample_trajectory(
    trajectory: List[Tuple[float, float]],
    num_waypoints: int = 30,
) -> List[Tuple[float, float]]:
    """Resample trajectory to fixed number of waypoints.

    Decouples trajectory length from video frame count.

    Args:
        trajectory: List of (x, y) waypoints.
        num_waypoints: Target number of waypoints.

    Returns:
        Resampled trajectory.
    """
    if len(trajectory) <= 1:
        return trajectory

    if len(trajectory) == num_waypoints:
        return trajectory

    # Parameterize by arc length (or uniform parameter if very short)
    t = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_waypoints)

    x_coords = np.array([p[0] for p in trajectory])
    y_coords = np.array([p[1] for p in trajectory])

    f_x = interp1d(t, x_coords, kind="cubic" if len(trajectory) >= 4 else "linear")
    f_y = interp1d(t, y_coords, kind="cubic" if len(trajectory) >= 4 else "linear")

    x_resampled = f_x(t_new)
    y_resampled = f_y(t_new)

    return list(zip(x_resampled, y_resampled))


def process_trajectory(
    tracks: List[ImageObjectTrack],
    num_waypoints: int = 30,
    smooth_window: int = 5,
    smooth_order: int = 2,
) -> List[Tuple[float, float]]:
    """Full trajectory processing pipeline.

    1. Interpolate gaps (occlusions)
    2. Smooth jitter (Savitzky-Golay)
    3. Resample to fixed waypoints

    Args:
        tracks: Raw object tracks from CSRT.
        num_waypoints: Target waypoint count.
        smooth_window: Smoothing window size (must be odd).
        smooth_order: Smoothing polynomial order.

    Returns:
        Processed trajectory as [(x, y), ...].
    """
    # 1. Fill gaps
    tracks_filled = interpolate_gaps(tracks)

    # 2. Smooth
    trajectory = smooth_trajectory(tracks_filled, smooth_window, smooth_order)

    # 3. Resample
    trajectory_final = resample_trajectory(trajectory, num_waypoints)

    return trajectory_final
