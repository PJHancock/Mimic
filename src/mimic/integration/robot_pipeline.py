"""Convert saved perception artifacts into explicit robot-world waypoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from mimic.common.types import ExtractedTask, ObjectTrack, PickPlaceWaypoints, RetargetedTask
from mimic.integration.action_results import RobotActionResults, load_robot_actions
from mimic.robot import build_waypoints, extract_tasks, process_path, retarget_task
from mimic.robot.path_processing import ProcessedPath
from mimic.tracking.coordinate_mapping import CoordinateMapper

_DEMO_RESULTS_SCHEMA = "mimic.demo_results.v2"
_JSON_SOURCE = Union[str, Path, Mapping[str, Any]]
_YAML_SOURCE = Union[str, Path, Mapping[str, Any]]


@dataclass(frozen=True)
class RobotPipelineArtifacts:
    """Traceable intermediate records for one explicitly selected episode."""

    episode_count: int
    selected_episode: int
    table_tracks: Tuple[ObjectTrack, ...]
    task: ExtractedTask
    retargeted_task: RetargetedTask
    processed_path: ProcessedPath
    waypoints: PickPlaceWaypoints


def _load_json(source: _JSON_SOURCE, description: str) -> dict:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = json.loads(Path(source).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _load_yaml(source: _YAML_SOURCE, description: str) -> dict:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = yaml.safe_load(Path(source).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a YAML mapping")
    return payload


def _strict_positive_frame(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("tracking frame_idx must be a positive one-based integer")
    return int(value)


def _strict_probability(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("tracking confidence must be a finite probability")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("tracking confidence must be a finite probability")
    return result


def _strict_positive_length(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number of meters")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number of meters")
    return result


def _coordinate_mapper(calibration_source: _JSON_SOURCE) -> CoordinateMapper:
    payload = _load_json(calibration_source, "Calibration")
    width_m = _strict_positive_length(payload.get("table_width_m"), "table_width_m")
    height_m = _strict_positive_length(payload.get("table_height_m"), "table_height_m")
    homography = np.asarray(payload.get("homography"))
    if homography.shape != (3, 3) or not np.issubdtype(homography.dtype, np.number):
        raise ValueError("Calibration homography must be a numeric 3x3 matrix")
    homography = homography.astype(float)
    if not np.all(np.isfinite(homography)) or np.linalg.matrix_rank(homography) != 3:
        raise ValueError("Calibration homography must be finite and nonsingular")
    mapper = CoordinateMapper(width_m, height_m)
    mapper.homography = homography
    mapper.is_calibrated = True
    return mapper


def _demo_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != _DEMO_RESULTS_SCHEMA:
        raise ValueError(f"Tracking input requires schema {_DEMO_RESULTS_SCHEMA}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Demo results metadata must be a mapping")
    if metadata.get("tracking_coordinate_frame") != "image_pixels":
        raise ValueError("Demo results must explicitly declare image-pixel tracking")
    return metadata


def load_calibrated_object_tracks(
    results_source: _JSON_SOURCE,
    calibration_source: _JSON_SOURCE,
) -> Tuple[ObjectTrack, ...]:
    """Load saved image-pixel observations and map them into table meters."""

    payload = _load_json(results_source, "Demo results")
    _demo_metadata(payload)
    mapper = _coordinate_mapper(calibration_source)
    return _calibrated_tracks_from_payload(payload, mapper)


def _calibrated_tracks_from_payload(
    payload: Mapping[str, Any], mapper: CoordinateMapper
) -> Tuple[ObjectTrack, ...]:
    # New results preserve the complete tracker-native stream separately from
    # potentially sparse classifier frames. Fall back for existing v2 artifacts.
    frames = payload.get("object_tracks", payload.get("per_frame"))
    if not isinstance(frames, list) or not frames:
        raise ValueError("Demo results object_tracks/per_frame must be a nonempty list")

    tracks = []
    previous_frame = 0
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("Each demo result frame must be a mapping")
        frame_idx = _strict_positive_frame(frame.get("frame_idx"))
        if frame_idx <= previous_frame:
            raise ValueError("Tracking frame_idx values must be unique and increasing")
        previous_frame = frame_idx
        position = frame.get("position")
        if position is None:
            continue
        if not isinstance(position, Mapping):
            raise ValueError("Tracking position must be a mapping or null")
        pixel_x, pixel_y = position.get("x"), position.get("y")
        if pixel_x is None and pixel_y is None:
            continue
        if (
            pixel_x is None
            or pixel_y is None
            or isinstance(pixel_x, (bool, np.bool_))
            or isinstance(pixel_y, (bool, np.bool_))
            or not isinstance(pixel_x, Real)
            or not isinstance(pixel_y, Real)
        ):
            raise ValueError("Tracking x and y must both be finite pixel coordinates or null")
        pixel_xy = (float(pixel_x), float(pixel_y))
        if not np.all(np.isfinite(pixel_xy)):
            raise ValueError("Tracking x and y must both be finite pixel coordinates or null")
        confidence = _strict_probability(position.get("confidence"))
        tracks.append(
            ObjectTrack(
                frame_idx=frame_idx,
                table_xy_m=mapper.pixel_to_table_xy_m(pixel_xy),
                confidence=confidence,
            )
        )
    if not tracks:
        raise ValueError("Demo results contain no valid object observations")
    return tuple(tracks)


def _validate_table_clone(mapper: CoordinateMapper, retargeting_payload: Mapping[str, Any]) -> None:
    clone = retargeting_payload.get("tabletop_clone")
    if not isinstance(clone, Mapping):
        raise ValueError("Retargeting config requires tabletop_clone")
    clone_width = _strict_positive_length(clone.get("width_m"), "tabletop_clone.width_m")
    clone_depth = _strict_positive_length(clone.get("depth_m"), "tabletop_clone.depth_m")
    if not np.isclose(mapper.table_width_m, clone_width, rtol=0, atol=1e-12) or not np.isclose(
        mapper.table_height_m, clone_depth, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Calibration table dimensions must match the configured MuJoCo tabletop clone"
        )


def build_robot_pipeline(
    *,
    actions_source: _JSON_SOURCE,
    results_source: _JSON_SOURCE,
    calibration_source: _JSON_SOURCE,
    retargeting_source: _YAML_SOURCE,
    pipeline_config_source: _YAML_SOURCE,
    episode: Optional[int] = None,
) -> RobotPipelineArtifacts:
    """Run saved artifacts through calibration, extraction, retargeting, and waypoints."""

    results_payload = _load_json(results_source, "Demo results")
    metadata = _demo_metadata(results_payload)
    expected_catalog = metadata.get("catalog_fingerprint")
    if not isinstance(expected_catalog, str):
        raise ValueError("Demo results require catalog_fingerprint provenance")

    actions_payload = _load_json(actions_source, "Robot actions")
    action_results = RobotActionResults.model_validate(actions_payload)
    predictions = load_robot_actions(
        actions_payload,
        expected_catalog_fingerprint=expected_catalog,
    )
    expected_postprocessing = metadata.get("postprocessing_fingerprint")
    if expected_postprocessing != action_results.postprocessing.fingerprint:
        raise ValueError("Demo results and robot actions use different post-processing settings")

    mapper = _coordinate_mapper(calibration_source)
    retargeting_payload = _load_yaml(retargeting_source, "Retargeting config")
    _validate_table_clone(mapper, retargeting_payload)
    mapping = retargeting_payload.get("retargeting")
    if not isinstance(mapping, Mapping):
        raise ValueError("Retargeting config requires retargeting")

    tracks = _calibrated_tracks_from_payload(results_payload, mapper)
    tasks = extract_tasks(predictions, tracks)
    if episode is None:
        if len(tasks) != 1:
            raise ValueError(
                f"Found {len(tasks)} complete episodes; select one with --episode (one-based)"
            )
        selected_episode = 1
    else:
        if isinstance(episode, bool) or not isinstance(episode, Integral) or episode < 1:
            raise ValueError("episode must be a positive one-based integer")
        selected_episode = int(episode)
        if selected_episode > len(tasks):
            raise ValueError(
                f"Episode {selected_episode} is unavailable; found {len(tasks)} complete episodes"
            )

    pipeline_payload = _load_yaml(pipeline_config_source, "Robot pipeline config")
    if set(pipeline_payload) != {"robot_pipeline"} or not isinstance(
        pipeline_payload["robot_pipeline"], Mapping
    ):
        raise ValueError("Robot pipeline config must contain exactly the robot_pipeline root")
    pipeline_config = pipeline_payload["robot_pipeline"]
    if set(pipeline_config) != {"path_processing", "waypoint_construction"}:
        raise ValueError(
            "robot_pipeline must contain exactly path_processing and waypoint_construction"
        )

    task = tasks[selected_episode - 1]
    target_task = retarget_task(task, mapping)
    path = process_path(target_task, pipeline_config["path_processing"])
    waypoints = build_waypoints(path, pipeline_config["waypoint_construction"])
    return RobotPipelineArtifacts(
        episode_count=len(tasks),
        selected_episode=selected_episode,
        table_tracks=tracks,
        task=task,
        retargeted_task=target_task,
        processed_path=path,
        waypoints=waypoints,
    )


def waypoint_payload(waypoints: PickPlaceWaypoints) -> dict:
    """Serialize the existing executor contract without adding another schema layer."""

    return asdict(waypoints)


def write_world_waypoints(
    path: Union[str, Path], waypoints: PickPlaceWaypoints, *, overwrite: bool = False
) -> None:
    """Write simulator-ready waypoints, refusing replacement unless explicitly requested."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode) as stream:
        json.dump(waypoint_payload(waypoints), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    root = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True, help="mimic.robot_actions.v1 JSON")
    parser.add_argument("--results", type=Path, required=True, help="mimic.demo_results.v2 JSON")
    parser.add_argument("--calibration", type=Path, required=True, help="Camera homography JSON")
    parser.add_argument(
        "--retargeting-config",
        type=Path,
        default=root / "configs" / "retargeting.yaml",
        help="Table-to-world mapping and tabletop clone YAML",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        required=True,
        help="Explicit path-processing and waypoint-construction YAML",
    )
    parser.add_argument("--episode", type=int, help="One-based episode to execute")
    parser.add_argument("--waypoints", type=Path, required=True, help="New world-waypoint JSON")
    parser.add_argument("--overwrite", action="store_true", help="Replace --waypoints if it exists")
    parser.add_argument("--robot-config", type=Path, help="Also run this robot simulation config")
    parser.add_argument("--log", type=Path, help="New simulator JSONL log; requires --robot-config")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.robot_config is None) != (args.log is None):
        parser.error("--robot-config and --log must be supplied together")
    for path, description in (
        (args.actions, "Robot actions"),
        (args.results, "Demo results"),
        (args.calibration, "Calibration"),
        (args.retargeting_config, "Retargeting config"),
        (args.pipeline_config, "Robot pipeline config"),
    ):
        if not path.is_file():
            parser.error(f"{description} not found: {path}")
    if args.robot_config is not None and not args.robot_config.is_file():
        parser.error(f"Robot config not found: {args.robot_config}")

    try:
        artifacts = build_robot_pipeline(
            actions_source=args.actions,
            results_source=args.results,
            calibration_source=args.calibration,
            retargeting_source=args.retargeting_config,
            pipeline_config_source=args.pipeline_config,
            episode=args.episode,
        )
        write_world_waypoints(args.waypoints, artifacts.waypoints, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote episode {artifacts.selected_episode}/{artifacts.episode_count} "
        f"with {len(artifacts.processed_path.xy_m)} path points to {args.waypoints}"
    )
    if args.robot_config is None:
        return 0
    command = (
        sys.executable,
        str(_repository_root() / "scripts" / "simulate_robot.py"),
        "--config",
        str(args.robot_config),
        "--waypoints",
        str(args.waypoints),
        "--log",
        str(args.log),
    )
    return subprocess.run(command, cwd=_repository_root(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
