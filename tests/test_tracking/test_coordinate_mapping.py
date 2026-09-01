"""Unit tests for coordinate mapping."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mimic.tracking import CoordinateMapper


@pytest.fixture
def mapper():
    """Create a coordinate mapper instance."""
    return CoordinateMapper(table_width_m=0.6, table_height_m=0.4)


def test_mapper_init(mapper):
    """Test CoordinateMapper initialization."""
    assert mapper is not None
    assert mapper.table_width_m == 0.6
    assert mapper.table_height_m == 0.4
    assert not mapper.is_calibrated
    assert not hasattr(mapper, "workspace_to_panda")


def test_mapper_calibrate(mapper):
    """Test calibration with corner points."""
    # Image corners
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    # World corners (in meters)
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    calib = mapper.calibrate(frame, corners_img, corners_world)

    assert mapper.is_calibrated
    assert calib is not None
    assert mapper.homography is not None
    assert mapper.homography.shape == (3, 3)
    np.testing.assert_allclose(calib.homography, mapper.homography)


def test_mapper_pixel_to_normalized_table(mapper):
    """Test pixel to normalized physical-table conversion."""
    # Set up simple identity homography
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mapper.calibrate(frame, corners_img, corners_world)

    # Test: pixel (0, 0) should map to normalized (0, 0)
    norm_coord = mapper.pixel_to_normalized_table((0, 0))
    assert isinstance(norm_coord, tuple)
    assert len(norm_coord) == 2
    assert 0 <= norm_coord[0] <= 1
    assert 0 <= norm_coord[1] <= 1


def test_mapper_pixel_to_table_meters_without_clipping(mapper):
    """Calibrated robot inputs retain table units and out-of-bounds evidence."""
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mapper.calibrate(frame, corners_img, corners_world)

    assert mapper.pixel_to_table_xy_m((320, 240)) == pytest.approx((0.3, 0.2))
    outside = mapper.pixel_to_table_xy_m((800, 600))
    assert outside[0] > 0.6
    assert outside[1] > 0.4


def test_mapper_not_calibrated_error(mapper):
    """Test that error is raised if mapping without calibration."""
    with pytest.raises(RuntimeError):
        mapper.pixel_to_normalized_table((100, 100))


def test_mapper_save_load(mapper):
    """Test save and load calibration."""
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    calib = mapper.calibrate(frame, corners_img, corners_world)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_path = Path(tmpdir) / "calibration.json"
        mapper.save(str(calib_path), calib)

        # Verify file exists
        assert calib_path.exists()

        # Load in new mapper
        mapper2 = CoordinateMapper(table_width_m=0.6, table_height_m=0.4)
        mapper2.load(str(calib_path))

        assert mapper2.is_calibrated
        assert mapper2.homography is not None


def test_mapper_batch_conversion(mapper):
    """Test batch pixel to normalized-table conversion."""
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mapper.calibrate(frame, corners_img, corners_world)

    # Batch conversion
    pixels = [(0, 0), (320, 240), (640, 480)]
    workspace_coords = mapper.pixels_to_normalized_table_batch(pixels)

    assert len(workspace_coords) == len(pixels)
    for coord in workspace_coords:
        assert len(coord) == 2
        assert 0 <= coord[0] <= 1
        assert 0 <= coord[1] <= 1


def test_mapper_save_creates_directory(mapper):
    """Test that save creates directories if needed."""
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    calib = mapper.calibrate(frame, corners_img, corners_world)

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "deep" / "nested" / "path" / "calibration.json"
        mapper.save(str(nested_path), calib)

        assert nested_path.exists()
        # Verify JSON is valid
        with open(nested_path) as f:
            data = json.load(f)
        assert "homography" in data


def test_mapper_load_invalid_path():
    """Test that load raises error for missing file."""
    mapper = CoordinateMapper(table_width_m=0.6, table_height_m=0.4)
    with pytest.raises(FileNotFoundError):
        mapper.load("/nonexistent/path/calibration.json")


def test_mapper_normalization_bounds(mapper):
    """Test that normalized coordinates are clipped to [0, 1]."""
    corners_img = [(0, 0), (640, 0), (0, 480), (640, 480)]
    corners_world = [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mapper.calibrate(frame, corners_img, corners_world)

    # Test points outside bounds
    coord_oob_neg = mapper.pixel_to_normalized_table((-100, -100))
    coord_oob_pos = mapper.pixel_to_normalized_table((1000, 1000))

    # All should be clipped to [0, 1]
    for coord in [coord_oob_neg, coord_oob_pos]:
        assert 0 <= coord[0] <= 1
        assert 0 <= coord[1] <= 1
