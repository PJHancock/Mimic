"""Offline extraction of complete episodes from resolved skill labels and tracks."""

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
    """Validate complete actions; never smooth, relabel, or repair input."""

    EPISODE_WITH_RETREAT = (
        ActionPhase.IDLE,
        ActionPhase.HOVER,
        ActionPhase.GRASP,
        ActionPhase.CARRY,
        ActionPhase.RELEASE,
        ActionPhase.HOVER,
        ActionPhase.IDLE,
    )
    EPISODE_DIRECT_TO_IDLE = (
        ActionPhase.IDLE,
        ActionPhase.HOVER,
        ActionPhase.GRASP,
        ActionPhase.CARRY,
        ActionPhase.RELEASE,
        ActionPhase.IDLE,
    )
    EPISODES = (EPISODE_WITH_RETREAT, EPISODE_DIRECT_TO_IDLE)

    def extract(
        self,
        action_predictions: Sequence[ActionPrediction],
        object_tracks: Sequence[ObjectTrack],
    ) -> ExtractedTask:
        tasks = self.extract_tasks(action_predictions, object_tracks)
        if len(tasks) != 1:
            raise TaskExtractionError(
                f"Expected exactly one complete episode, received {len(tasks)}; use extract_tasks"
            )
        return tasks[0]

    def extract_tasks(
        self,
        action_predictions: Sequence[ActionPrediction],
        object_tracks: Sequence[ObjectTrack],
    ) -> tuple[ExtractedTask, ...]:
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

        episodes: List[tuple[PhaseBoundary, ...]] = []
        cursor = 0
        while cursor < len(boundaries) - 1:
            episode = None
            for expected in self.EPISODES:
                candidate = tuple(boundaries[cursor : cursor + len(expected)])
                if tuple(boundary.phase for boundary in candidate) == expected:
                    episode = candidate
                    break
            if episode is None:
                break
            episodes.append(episode)
            cursor += len(episode) - 1  # Adjacent episodes share their IDLE boundary.
        if not episodes or cursor != len(boundaries) - 1:
            expected = " -> ".join(phase.value for phase in self.EPISODE_WITH_RETREAT)
            raise TaskExtractionError(
                f"Expected one or more complete {expected} episodes (terminal HOVER is "
                "optional); "
                "state post-processing is not performed by the extractor"
            )

        detached_tracks: List[TablePathSample] = []
        previous_frame = 0
        object_id = tracks[0].object_id
        for track in tracks:
            if track.object_id != object_id:
                raise TaskExtractionError("Object identity must be consistent across all tracks")
            # Phase intervals are [onset, next_onset), using source-video frames.
            # An absent prediction at a tracking frame does not require interpolation.
            try:
                phase = ActionPhase.IDLE
                for boundary in boundaries:
                    if boundary.frame_idx <= track.frame_idx:
                        phase = boundary.phase
                sample = TablePathSample(track.frame_idx, track.table_xy_m, phase, track.confidence)
            except (ValueError, TypeError) as exc:
                raise TaskExtractionError(f"Invalid track: {exc}") from exc
            if sample.frame_idx <= previous_frame:
                raise TaskExtractionError("Tracking frame IDs must be unique and increasing")
            previous_frame = sample.frame_idx
            detached_tracks.append(sample)

        tasks: List[ExtractedTask] = []
        for episode in episodes:
            grasp_frame, release_frame = episode[2].frame_idx, episode[4].frame_idx
            samples = tuple(
                sample
                for sample in detached_tracks
                if grasp_frame <= sample.frame_idx <= release_frame
            )
            observed = {sample.frame_idx for sample in samples}
            for phase_name, frame in (("GRASP", grasp_frame), ("RELEASE", release_frame)):
                if frame not in observed:
                    raise TaskExtractionError(
                        f"Missing exact object observation at {phase_name} onset frame {frame}; "
                        "no nearest-frame fallback or interpolation"
                    )
            try:
                tasks.append(ExtractedTask(episode, samples, object_id))
            except (ValueError, TypeError) as exc:
                raise TaskExtractionError(str(exc)) from exc
        return tuple(tasks)


def extract_task(
    action_predictions: Sequence[ActionPrediction], object_tracks: Sequence[ObjectTrack]
) -> ExtractedTask:
    """Convenience entry point for extracting a single demonstration."""
    return TaskExtractor().extract(action_predictions, object_tracks)


def extract_tasks(
    action_predictions: Sequence[ActionPrediction], object_tracks: Sequence[ObjectTrack]
) -> tuple[ExtractedTask, ...]:
    """Convenience entry point for extracting every complete episode in a timeline."""
    return TaskExtractor().extract_tasks(action_predictions, object_tracks)
