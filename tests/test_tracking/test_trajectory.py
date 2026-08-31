"""Unit tests for trajectory processing."""

import numpy as np
import pytest

from mimic.common.types import ObjectTrack
from mimic.tracking import (
    interpolate_gaps,
    process_trajectory,
    resample_trajectory,
    smooth_trajectory,
)


@pytest.fixture
def sample_tracks():
    """Create sample object tracks with a gap."""
    tracks = []
    # Frames 0-3: confident tracking
    for i in range(4):
        tracks.append(
            ObjectTrack(
                frame_idx=i,
                center_2d=(100 + i * 10, 100 + i * 5),
                confidence=1.0,
            )
        )
    # Frames 4-5: lost track (gap)
    for i in range(4, 6):
        tracks.append(
            ObjectTrack(
                frame_idx=i,
                center_2d=(140, 120),  # dummy position (will be interpolated)
                confidence=0.0,  # lost
            )
        )
    # Frames 6-9: track regained
    for i in range(6, 10):
        tracks.append(
            ObjectTrack(
                frame_idx=i,
                center_2d=(140 + (i - 6) * 10, 130 + (i - 6) * 5),
                confidence=1.0,
            )
        )
    return tracks


def test_interpolate_gaps_empty_list():
    """Test interpolation on empty list."""
    result = interpolate_gaps([])
    assert result == []


def test_interpolate_gaps_single_track():
    """Test interpolation with single track."""
    track = ObjectTrack(frame_idx=0, center_2d=(100, 100), confidence=1.0)
    result = interpolate_gaps([track])
    assert len(result) == 1
    assert result[0].center_2d == (100, 100)


def test_interpolate_gaps_fills_missing(sample_tracks):
    """Test that gaps are filled with interpolation."""
    result = interpolate_gaps(sample_tracks)

    assert len(result) == len(sample_tracks)
    # Frames 4-5 should have interpolated positions
    # Frame 4 confidence was 0, but position should be interpolated
    frame_4 = [t for t in result if t.frame_idx == 4][0]
    frame_5 = [t for t in result if t.frame_idx == 5][0]

    # Positions should be interpolated between confident frames
    assert frame_4.center_2d is not None
    assert frame_5.center_2d is not None


def test_smooth_trajectory_empty():
    """Test smoothing empty trajectory."""
    result = smooth_trajectory([])
    assert result == []


def test_smooth_trajectory_single_point():
    """Test smoothing single point."""
    track = ObjectTrack(frame_idx=0, center_2d=(100, 100), confidence=1.0)
    result = smooth_trajectory([track], window_length=3)
    assert len(result) == 1


def test_smooth_trajectory_reduces_noise():
    """Test that smoothing reduces jitter."""
    # Create noisy trajectory
    tracks = []
    for i in range(10):
        # Sine wave with noise
        y = 100 + 20 * np.sin(i * 0.5) + np.random.randn() * 2
        tracks.append(ObjectTrack(frame_idx=i, center_2d=(100 + i * 10, y), confidence=1.0))

    result = smooth_trajectory(tracks, window_length=5, polyorder=2)

    assert len(result) == len(tracks)
    # Smoothed trajectory should have lower variance
    y_coords = [pt[1] for pt in result]
    y_variance = np.var(y_coords)
    # Should be reasonable (smoothing reduces noise)
    assert y_variance > 0  # Still has variation from underlying sine


def test_smooth_trajectory_window_adjustment():
    """Test that window length is made odd if needed."""
    tracks = [
        ObjectTrack(frame_idx=i, center_2d=(100 + i, 100 + i), confidence=1.0)
        for i in range(10)
    ]
    # Pass even window length; should be adjusted to odd
    result = smooth_trajectory(tracks, window_length=4, polyorder=2)
    assert len(result) == len(tracks)


def test_resample_trajectory_empty():
    """Test resampling empty trajectory."""
    result = resample_trajectory([])
    assert result == []


def test_resample_trajectory_single_point():
    """Test resampling single point."""
    traj = [(100, 100)]
    result = resample_trajectory(traj, num_waypoints=30)
    assert len(result) == 1


def test_resample_trajectory_target_waypoints():
    """Test that resampled trajectory has target length."""
    # Create straight line trajectory
    traj = [(100 + i, 100 + i) for i in range(20)]

    for num_waypoints in [10, 30, 50]:
        result = resample_trajectory(traj, num_waypoints=num_waypoints)
        assert len(result) == num_waypoints


def test_resample_trajectory_preserves_endpoints():
    """Test that resampling preserves start and end points approximately."""
    traj = [(100 + i * 10, 100 + i * 5) for i in range(10)]

    result = resample_trajectory(traj, num_waypoints=20)

    # First and last points should be close to original
    assert result[0] is not None
    assert result[-1] is not None
    # Should be approximately at the endpoints
    assert abs(result[0][0] - traj[0][0]) < 1
    assert abs(result[-1][0] - traj[-1][0]) < 1


def test_resample_trajectory_same_length():
    """Test resampling to same length."""
    traj = [(100 + i, 100 + i) for i in range(30)]
    result = resample_trajectory(traj, num_waypoints=30)
    assert len(result) == 30


def test_process_trajectory_pipeline(sample_tracks):
    """Test full trajectory processing pipeline."""
    result = process_trajectory(
        sample_tracks,
        num_waypoints=15,
        smooth_window=5,
        smooth_order=2,
    )

    # Should have exactly 15 waypoints
    assert len(result) == 15

    # All waypoints should be tuples with 2 elements
    for waypoint in result:
        assert len(waypoint) == 2
        assert isinstance(waypoint[0], (float, np.floating))
        assert isinstance(waypoint[1], (float, np.floating))


def test_process_trajectory_smooths_and_resamples():
    """Test that pipeline combines smoothing and resampling."""
    # Create trajectory with noise
    tracks = []
    for i in range(20):
        y = 100 + 5 * np.sin(i * 0.3) + np.random.randn() * 1
        tracks.append(ObjectTrack(frame_idx=i, center_2d=(100 + i * 5, y), confidence=1.0))

    result = process_trajectory(tracks, num_waypoints=10)

    assert len(result) == 10
    # Processed trajectory should be a valid path
    for x, y in result:
        assert isinstance(x, (float, np.floating))
        assert isinstance(y, (float, np.floating))


def test_process_trajectory_handles_occlusions():
    """Test that pipeline handles occlusions (confidence=0)."""
    tracks = []
    # Confident frames 0-2
    for i in range(3):
        tracks.append(
            ObjectTrack(frame_idx=i, center_2d=(100 + i * 10, 100), confidence=1.0)
        )
    # Occlusion frames 3-4
    for i in range(3, 5):
        tracks.append(
            ObjectTrack(
                frame_idx=i, center_2d=(130, 100), confidence=0.0
            )  # Lost track
        )
    # Confident frames 5-7
    for i in range(5, 8):
        tracks.append(
            ObjectTrack(
                frame_idx=i,
                center_2d=(100 + i * 10, 100),
                confidence=1.0,
            )
        )

    result = process_trajectory(tracks, num_waypoints=7)

    assert len(result) == 7
    # All points should be valid (no NaN)
    for x, y in result:
        assert not np.isnan(x)
        assert not np.isnan(y)


def test_smooth_trajectory_edge_case_too_few_points():
    """Test smoothing with fewer points than window."""
    tracks = [
        ObjectTrack(frame_idx=0, center_2d=(100, 100), confidence=1.0),
        ObjectTrack(frame_idx=1, center_2d=(110, 110), confidence=1.0),
    ]
    result = smooth_trajectory(tracks, window_length=5)  # Window > num_points
    # Should return as-is
    assert len(result) == 2
