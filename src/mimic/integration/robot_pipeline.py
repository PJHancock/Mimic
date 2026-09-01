"""Convert saved perception artifacts into explicit robot-world waypoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from mimic.common.types import ExtractedTask, ObjectTrack, PickPlaceWaypoints, RetargetedTask
from mimic.integration.task_input import (
    DemoTaskInput,
    DemoVideoMetadata,
    load_demo_task_input,
    load_task_actions,
)
from mimic.robot import build_waypoints, extract_tasks, process_path, retarget_task
from mimic.robot.path_processing import ProcessedPath
from mimic.tracking.coordinate_mapping import CoordinateMapper

_JSON_SOURCE = Union[str, Path, Mapping[str, Any], DemoTaskInput]
_YAML_SOURCE = Union[str, Path, Mapping[str, Any]]
_TIMESTAMPED_VIDEO = object()


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


def _strict_positive_length(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number of meters")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number of meters")
    return result


def _strict_positive_pixel_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer pixel count")
    return int(value)


def _coordinate_mapper(calibration_source: _JSON_SOURCE) -> CoordinateMapper:
    payload = _load_json(calibration_source, "Calibration")
    width_m = _strict_positive_length(payload.get("table_width_m"), "table_width_m")
    height_m = _strict_positive_length(payload.get("table_height_m"), "table_height_m")
    image_width_px = _strict_positive_pixel_count(payload.get("image_width_px"), "image_width_px")
    image_height_px = _strict_positive_pixel_count(
        payload.get("image_height_px"), "image_height_px"
    )
    homography = np.asarray(payload.get("homography"))
    if homography.shape != (3, 3) or not np.issubdtype(homography.dtype, np.number):
        raise ValueError("Calibration homography must be a numeric 3x3 matrix")
    homography = homography.astype(float)
    if not np.all(np.isfinite(homography)) or np.linalg.matrix_rank(homography) != 3:
        raise ValueError("Calibration homography must be finite and nonsingular")
    mapper = CoordinateMapper(width_m, height_m)
    mapper.image_width_px = image_width_px
    mapper.image_height_px = image_height_px
    mapper.homography = homography
    mapper.is_calibrated = True
    return mapper


def _validate_calibration_frame(metadata: DemoVideoMetadata, mapper: CoordinateMapper) -> None:
    tracking_size = (
        metadata.image_width_px,
        metadata.image_height_px,
    )
    calibration_size = (mapper.image_width_px, mapper.image_height_px)
    if tracking_size != calibration_size:
        raise ValueError(
            "Tracking and calibration image dimensions differ; check video rotation "
            "and resolution"
        )


def load_calibrated_object_tracks(
    task_input_source: _JSON_SOURCE,
    calibration_source: _JSON_SOURCE,
) -> Tuple[ObjectTrack, ...]:
    """Load saved image-pixel observations and map them into table meters."""

    task_input = load_demo_task_input(task_input_source)
    mapper = _coordinate_mapper(calibration_source)
    _validate_calibration_frame(task_input.video, mapper)
    return _calibrated_tracks_from_task_input(task_input, mapper)


def _calibrated_tracks_from_task_input(
    task_input: DemoTaskInput, mapper: CoordinateMapper
) -> Tuple[ObjectTrack, ...]:
    if mapper.image_width_px is None or mapper.image_height_px is None:
        raise ValueError("Calibration requires its decoded image width and height")
    image_width_px, image_height_px = mapper.image_width_px, mapper.image_height_px
    tracks = []
    for frame in task_input.object_tracks:
        position = frame.position
        if position is None:
            continue
        pixel_xy = (position.x, position.y)
        if not (0 <= pixel_xy[0] < image_width_px and 0 <= pixel_xy[1] < image_height_px):
            raise ValueError(
                "Tracking pixel coordinate is outside the calibrated image frame; "
                "check video rotation and resolution"
            )
        tracks.append(
            ObjectTrack(
                frame_idx=frame.frame_idx,
                table_xy_m=mapper.pixel_to_table_xy_m(pixel_xy),
                confidence=position.confidence,
            )
        )
    if not tracks:
        raise ValueError("Demo task input contains no valid object observations")
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
    task_input_source: _JSON_SOURCE,
    calibration_source: _JSON_SOURCE,
    retargeting_source: _YAML_SOURCE,
    pipeline_config_source: _YAML_SOURCE,
    episode: Optional[int] = None,
) -> RobotPipelineArtifacts:
    """Run saved artifacts through calibration, extraction, retargeting, and waypoints."""

    task_input = load_demo_task_input(task_input_source)
    predictions = load_task_actions(
        task_input,
        expected_catalog_fingerprint=task_input.catalog.fingerprint,
    )

    mapper = _coordinate_mapper(calibration_source)
    _validate_calibration_frame(task_input.video, mapper)
    retargeting_payload = _load_yaml(retargeting_source, "Retargeting config")
    _validate_table_clone(mapper, retargeting_payload)
    mapping = retargeting_payload.get("retargeting")
    if not isinstance(mapping, Mapping):
        raise ValueError("Retargeting config requires retargeting")

    tracks = _calibrated_tracks_from_task_input(task_input, mapper)
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
    parser.add_argument(
        "--task-input",
        type=Path,
        required=True,
        help="mimic.demo_task_input.v1 JSON",
    )
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
    parser.add_argument(
        "--log",
        type=Path,
        help="Simulator JSONL log, replacing existing contents; requires --robot-config",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show live robot simulation; requires --robot-config and --log",
    )
    parser.add_argument(
        "--video-out",
        nargs="?",
        type=Path,
        const=_TIMESTAMPED_VIDEO,
        help=(
            "Write a simulation MP4 to PATH, or to a timestamped file when PATH is omitted; "
            "requires --robot-config and --log"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.robot_config is None) != (args.log is None):
        parser.error("--robot-config and --log must be supplied together")
    if args.viewer and args.robot_config is None:
        parser.error("--viewer requires --robot-config and --log")
    if args.video_out is not None and args.robot_config is None:
        parser.error("--video-out requires --robot-config and --log")
    for path, description in (
        (args.task_input, "Demo task input"),
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
            task_input_source=args.task_input,
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
    executable = sys.executable
    if (args.viewer or args.video_out is not None) and sys.platform == "darwin":
        executable = shutil.which("mjpython")
        if executable is None:
            print(
                "ERROR: mjpython is required for MuJoCo viewing or video output on macOS",
                file=sys.stderr,
            )
            return 2
    command = (
        executable,
        str(_repository_root() / "scripts" / "simulate_robot.py"),
        "--config",
        str(args.robot_config),
        "--waypoints",
        str(args.waypoints),
        "--log",
        str(args.log),
    )
    if args.video_out is _TIMESTAMPED_VIDEO:
        command += ("--video-out",)
    elif args.video_out is not None:
        command += ("--video-out", str(args.video_out))
    if args.viewer:
        command += ("--viewer",)
    return subprocess.run(command, cwd=_repository_root(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
