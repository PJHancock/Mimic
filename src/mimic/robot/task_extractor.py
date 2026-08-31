"""Offline extraction from already labeled predictions and table-space tracks."""

from typing import List, Sequence

from mimic.common.types import (
    ActionPhase,
    ActionPrediction,
    ExtractedTask,
    ObjectTrack,
    PhaseBoundary,
    TablePathSample,
)


class TaskExtractionError(ValueError):
    """The demonstration cannot satisfy the agreed extraction contract."""


class TaskExtractor:
    """Validate one complete action; never smooth, relabel, or repair input."""

    def extract(
        self,
        action_predictions: Sequence[ActionPrediction],
        object_tracks: Sequence[ObjectTrack],
    ) -> ExtractedTask:
        predictions = tuple(action_predictions)
        tracks = tuple(object_tracks)
        if not predictions or not tracks:
            raise TaskExtractionError("Predictions and object tracks must both be nonempty")

        boundaries: List[PhaseBoundary] = []
        previous_frame = 0
        for prediction in predictions:
            try:
                onset = PhaseBoundary(prediction.phase, prediction.frame_idx, prediction.timestamp)
            except (ValueError, TypeError) as exc:
                raise TaskExtractionError(f"Invalid prediction: {exc}") from exc
            if onset.frame_idx <= previous_frame:
                raise TaskExtractionError("Prediction frame IDs must be unique and increasing")
            previous_frame = onset.frame_idx
            if not boundaries or onset.phase != boundaries[-1].phase:
                boundaries.append(onset)

        if tuple(b.phase for b in boundaries) != tuple(ActionPhase):
            raise TaskExtractionError(
                "Expected one APPROACH -> GRASP -> MOVE -> RELEASE sequence; "
                "state post-processing is not performed by the extractor"
            )

        grasp_frame, release_frame = boundaries[1].frame_idx, boundaries[3].frame_idx
        samples: List[TablePathSample] = []
        previous_frame = 0
        object_id = tracks[0].object_id
        for track in tracks:
            if track.object_id != object_id:
                raise TaskExtractionError("Object identity must be consistent across all tracks")
            # Phase intervals are [onset, next_onset), using source-video frames.
            # An absent prediction at a tracking frame does not require interpolation.
            try:
                phase = ActionPhase.APPROACH
                for boundary in boundaries:
                    if boundary.frame_idx <= track.frame_idx:
                        phase = boundary.phase
                sample = TablePathSample(
                    track.frame_idx, track.table_xy_cm, phase, track.confidence
                )
            except (ValueError, TypeError) as exc:
                raise TaskExtractionError(f"Invalid track: {exc}") from exc
            if sample.frame_idx <= previous_frame:
                raise TaskExtractionError("Tracking frame IDs must be unique and increasing")
            previous_frame = sample.frame_idx
            if grasp_frame <= sample.frame_idx <= release_frame:
                samples.append(sample)

        observed = {sample.frame_idx for sample in samples}
        for phase_name, frame in (("GRASP", grasp_frame), ("RELEASE", release_frame)):
            if frame not in observed:
                raise TaskExtractionError(
                    f"Missing exact object observation at {phase_name} onset frame {frame}; "
                    "no nearest-frame fallback or interpolation"
                )

        try:
            return ExtractedTask(tuple(boundaries), tuple(samples), object_id)
        except (ValueError, TypeError) as exc:
            raise TaskExtractionError(str(exc)) from exc


def extract_task(
    action_predictions: Sequence[ActionPrediction], object_tracks: Sequence[ObjectTrack]
) -> ExtractedTask:
    """Convenience entry point for extracting a single demonstration."""
    return TaskExtractor().extract(action_predictions, object_tracks)
