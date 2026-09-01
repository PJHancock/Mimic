"""Synthetic task geometry; these mappings are not deployment calibration."""

import pytest

from mimic.common.types import ActionPhase, ActionPrediction, ObjectTrack
from mimic.robot import extract_task


@pytest.fixture
def task_predictions():
    return [
        ActionPrediction(frame, phase, 0.9)
        for frame, phase in (
            (1, ActionPhase.IDLE),
            (2, ActionPhase.HOVER),
            (3, ActionPhase.GRASP),
            (5, ActionPhase.GRASP),
            (7, ActionPhase.CARRY),
            (9, ActionPhase.CARRY),
            (11, ActionPhase.RELEASE),
            (13, ActionPhase.RELEASE),
            (14, ActionPhase.HOVER),
            (15, ActionPhase.IDLE),
        )
    ]


@pytest.fixture
def table_tracks():
    return [
        ObjectTrack(frame, xy, confidence=0.8, object_id="object-1")
        for frame, xy in (
            (1, (0.0, 0.0)),
            (3, (0.10, 0.20)),
            (4, (0.12, 0.24)),
            (6, (0.20, 0.40)),
            (7, (0.30, 0.50)),
            (8, (0.40, 0.60)),
            (10, (0.50, 0.40)),
            (11, (0.60, 0.30)),
            (13, (0.80, 0.80)),
        )
    ]


@pytest.fixture
def extracted_task(task_predictions, table_tracks):
    return extract_task(task_predictions, table_tracks)


@pytest.fixture
def mapping_config_values():
    return {
        "source_frame": "table",
        "target_frame": "synthetic_world",
        "table_origin_target_xy_m": [1.0, -2.0],
        "table_x_axis_target_xy": [0.0, 1.0],
        "table_y_axis_target_xy": [-1.0, 0.0],
    }
