"""Extraction contract: exact boundaries, frame alignment, and lossless paths."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mimic.common.types import ActionPhase, ActionPrediction, ObjectTrack
from mimic.robot import TaskExtractionError, TaskExtractor, extract_task, extract_tasks


def test_boundary_positions_and_inclusive_path(extracted_task):
    assert extracted_task.grasp_frame == 3
    assert extracted_task.release_frame == 11
    assert extracted_task.start_xy_m == (0.10, 0.20)
    assert extracted_task.goal_xy_m == (0.60, 0.30)
    assert extracted_task.object_id == "object-1"
    assert extracted_task.coordinate_frame == "table"
    assert [s.frame_idx for s in extracted_task.demonstrated_path] == [3, 4, 6, 7, 8, 10, 11]
    assert [b.frame_idx for b in extracted_task.phase_boundaries] == [1, 2, 3, 7, 11, 14, 15]
    assert all(b.timestamp_s is None for b in extracted_task.phase_boundaries)


def test_phase_intervals_do_not_require_prediction_for_each_track(extracted_task):
    assert [s.phase for s in extracted_task.demonstrated_path] == [
        ActionPhase.GRASP,
        ActionPhase.GRASP,
        ActionPhase.GRASP,
        ActionPhase.CARRY,
        ActionPhase.CARRY,
        ActionPhase.CARRY,
        ActionPhase.RELEASE,
    ]
    assert extracted_task.carry_trajectory_xy_m == (
        (0.30, 0.50),
        (0.40, 0.60),
        (0.50, 0.40),
    )
    assert all(s.confidence == 0.8 for s in extracted_task.demonstrated_path)


def test_extracted_task_exposes_the_complete_path_without_selecting_it(extracted_task):
    assert extracted_task.path_xy_m == (
        (0.10, 0.20),
        (0.12, 0.24),
        (0.20, 0.40),
        (0.30, 0.50),
        (0.40, 0.60),
        (0.50, 0.40),
        (0.60, 0.30),
    )
    assert not hasattr(extracted_task, "get_path")
    with pytest.raises(FrozenInstanceError):
        extracted_task.object_id = "replacement"


@pytest.mark.parametrize("missing_frame,phase", [(3, "GRASP"), (11, "RELEASE")])
def test_missing_endpoint_rejects_even_with_nearby_track(
    task_predictions, table_tracks, missing_frame, phase
):
    tracks = [t for t in table_tracks if t.frame_idx != missing_frame]
    with pytest.raises(TaskExtractionError, match=f"{phase} onset frame {missing_frame}"):
        extract_task(task_predictions, tracks)


@pytest.mark.parametrize("which", ["predictions", "tracks"])
def test_empty_input_rejected(task_predictions, table_tracks, which):
    with pytest.raises(TaskExtractionError, match="nonempty"):
        extract_task(
            [] if which == "predictions" else task_predictions,
            [] if which == "tracks" else table_tracks,
        )


@pytest.mark.parametrize("which", ["predictions", "tracks"])
@pytest.mark.parametrize("invalid_frame", [0, -1, True, 2.5, "3", None, np.nan])
def test_invalid_frame_ids_rejected(task_predictions, table_tracks, which, invalid_frame):
    sequence = task_predictions if which == "predictions" else table_tracks
    sequence[0].frame_idx = invalid_frame
    with pytest.raises(TaskExtractionError):
        extract_task(task_predictions, table_tracks)


@pytest.mark.parametrize("which", ["predictions", "tracks"])
@pytest.mark.parametrize("disorder", ["duplicate", "reversed"])
def test_duplicate_or_unsorted_frames_rejected(task_predictions, table_tracks, which, disorder):
    sequence = task_predictions if which == "predictions" else table_tracks
    if disorder == "duplicate":
        sequence.insert(1, sequence[0])
    else:
        sequence.reverse()
    with pytest.raises(TaskExtractionError, match="unique and increasing"):
        extract_task(task_predictions, table_tracks)


@pytest.mark.parametrize(
    "phases",
    [
        ["HOVER", "GRASP", "CARRY", "RELEASE", "HOVER", "IDLE"],
        ["IDLE", "GRASP", "CARRY", "RELEASE", "HOVER", "IDLE"],
        ["IDLE", "HOVER", "GRASP", "RELEASE", "HOVER", "IDLE"],
        ["IDLE", "HOVER", "GRASP", "CARRY", "RELEASE", "IDLE"],
        ["IDLE", "HOVER", "GRASP", "HOVER", "CARRY", "RELEASE", "HOVER", "IDLE"],
        ["IDLE", "HOVER", "GRASP", "CARRY", "RELEASE", "HOVER"],
    ],
)
def test_invalid_phase_sequence_is_not_repaired(table_tracks, phases):
    predictions = [ActionPrediction(i, p, 0.9) for i, p in enumerate(phases, 1)]
    with pytest.raises(TaskExtractionError, match="one or more complete"):
        extract_task(predictions, table_tracks)


def test_unknown_phase_rejected(task_predictions, table_tracks):
    task_predictions[0].phase = "UNKNOWN"
    with pytest.raises(TaskExtractionError, match="Invalid prediction"):
        extract_task(task_predictions, table_tracks)


@pytest.mark.parametrize(
    "xy", [(1,), (1, 2, 3), (np.nan, 0), (0, np.inf), ("1", 2), (True, 2), (1j, 2), None]
)
def test_invalid_coordinates_rejected(task_predictions, table_tracks, xy):
    table_tracks[1].table_xy_m = xy
    with pytest.raises(TaskExtractionError, match="Invalid track"):
        extract_task(task_predictions, table_tracks)


@pytest.mark.parametrize("identity", ["other-object", None])
def test_mixed_identity_rejected(task_predictions, table_tracks, identity):
    table_tracks[2].object_id = identity
    with pytest.raises(TaskExtractionError, match="identity"):
        extract_task(task_predictions, table_tracks)


def test_consistently_absent_identity_allowed(task_predictions, table_tracks):
    for track in table_tracks:
        track.object_id = None
    assert extract_task(task_predictions, table_tracks).object_id is None


@pytest.mark.parametrize("confidence", [np.nan, np.inf, -0.1, 1.1, "0.5", True])
def test_invalid_tracker_confidence_rejected(task_predictions, table_tracks, confidence):
    table_tracks[1].confidence = confidence
    with pytest.raises(TaskExtractionError, match="confidence"):
        extract_task(task_predictions, table_tracks)


def test_zero_confidence_is_retained_without_inventing_a_threshold(task_predictions, table_tracks):
    table_tracks[1].confidence = 0.0
    task = extract_task(task_predictions, table_tracks)
    assert task.demonstrated_path[0].confidence == 0.0


def test_sparse_interior_is_preserved_without_interpolation(task_predictions, table_tracks):
    task = extract_task(task_predictions, [t for t in table_tracks if t.frame_idx in (3, 11)])
    assert task.path_xy_m == ((0.10, 0.20), (0.60, 0.30))
    assert task.carry_trajectory_xy_m == ()


def test_mutating_inputs_cannot_change_task(task_predictions, table_tracks):
    mutable_xy = np.array([0.10, 0.20])
    table_tracks[1].table_xy_m = mutable_xy
    task = extract_task(task_predictions, table_tracks)
    mutable_xy[:] = 999
    table_tracks[1].confidence = 0.1
    task_predictions[1].frame_idx = 100
    table_tracks.clear()
    assert task.start_xy_m == (0.10, 0.20)
    assert task.grasp_frame == 3
    assert task.demonstrated_path[0].confidence == 0.8


def test_existing_seconds_are_preserved_not_recomputed(task_predictions, table_tracks):
    for p in task_predictions:
        p.timestamp = p.frame_idx * 0.123
    task = extract_task(task_predictions, table_tracks)
    assert task.phase_boundaries[2].timestamp_s == pytest.approx(0.369)
    assert task.phase_boundaries[4].timestamp_s == pytest.approx(1.353)


@pytest.mark.parametrize("timestamp", [np.nan, -1, "1.0", True])
def test_invalid_supplied_seconds_rejected(task_predictions, table_tracks, timestamp):
    task_predictions[1].timestamp = timestamp
    with pytest.raises(TaskExtractionError, match="timestamp_s"):
        extract_task(task_predictions, table_tracks)


def test_task_record_rejects_inconsistent_manual_construction(extracted_task):
    with pytest.raises(ValueError, match="endpoints"):
        replace(extracted_task, demonstrated_path=extracted_task.demonstrated_path[1:])
    altered = replace(extracted_task.demonstrated_path[1], phase=ActionPhase.CARRY)
    with pytest.raises(ValueError, match="phase"):
        replace(
            extracted_task,
            demonstrated_path=(
                extracted_task.demonstrated_path[0],
                altered,
                *extracted_task.demonstrated_path[2:],
            ),
        )


def test_extractor_instance_does_not_retain_previous_demonstration(task_predictions, table_tracks):
    extractor = TaskExtractor()
    first = extractor.extract(task_predictions, table_tracks)
    assert extractor.extract(task_predictions, table_tracks) == first


def test_multiple_episodes_are_extracted_from_one_timeline(task_predictions, table_tracks):
    second_predictions = [
        ActionPrediction(
            prediction.frame_idx + 14,
            prediction.phase,
            prediction.confidence,
            None if prediction.timestamp is None else prediction.timestamp + 14,
        )
        for prediction in task_predictions
        if prediction.phase != ActionPhase.IDLE or prediction.frame_idx != 1
    ]
    predictions = [*task_predictions, *second_predictions]
    second_tracks = [
        ObjectTrack(
            track.frame_idx + 14,
            track.table_xy_m,
            confidence=track.confidence,
            object_id=track.object_id,
        )
        for track in table_tracks
        if track.frame_idx >= 3
    ]
    tasks = extract_tasks(predictions, [*table_tracks, *second_tracks])
    assert len(tasks) == 2
    assert [task.grasp_frame for task in tasks] == [3, 17]
    with pytest.raises(TaskExtractionError, match="use extract_tasks"):
        extract_task(predictions, [*table_tracks, *second_tracks])


def test_old_pixel_keyword_cannot_silently_enter_table_contract():
    with pytest.raises(TypeError):
        ObjectTrack(frame_idx=1, center_2d=(10, 20))
