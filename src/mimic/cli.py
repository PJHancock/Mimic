"""Run a demonstration video through inference and a configured robot simulation."""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from mimic.integration.robot_pipeline import main as run_robot_pipeline
from mimic.integration.run_video_pipeline import main as run_video_pipeline


@dataclass(frozen=True)
class RobotDefaults:
    """Repository-relative inputs selected by one public robot name."""

    execution_config: Path
    calibration: Path
    retargeting_config: Path
    pipeline_config: Path
    skill_config: Path

    def resolve(self, repository_root: Path) -> "RobotDefaults":
        return RobotDefaults(
            execution_config=repository_root / self.execution_config,
            calibration=repository_root / self.calibration,
            retargeting_config=repository_root / self.retargeting_config,
            pipeline_config=repository_root / self.pipeline_config,
            skill_config=repository_root / self.skill_config,
        )


ROBOT_DEFAULTS: Mapping[str, RobotDefaults] = {
    "panda": RobotDefaults(
        execution_config=Path("configs/robots/panda_complete.yaml"),
        calibration=Path("data/annotations/calibrations.json"),
        retargeting_config=Path("configs/retargeting.yaml"),
        pipeline_config=Path("configs/robot_pipeline.yaml"),
        skill_config=Path("configs/skills/pick_place.yaml"),
    )
}

_DEFAULT_VIDEO_OUTPUT = object()
_DEFAULT_CONTEXT_WINDOW = 32


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input demonstration video")
    parser.add_argument(
        "--robot",
        required=True,
        choices=tuple(sorted(ROBOT_DEFAULTS)),
        help="Robot defaults and MuJoCo execution target",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Artifact directory; defaults to results/<video-name>",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Classifier checkpoint; defaults to models/action_classifier_lstm.pt",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Torch inference device; does not affect MuJoCo control",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show and real-time-pace the MuJoCo simulation",
    )
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument(
        "--video-out",
        nargs="?",
        type=Path,
        const=_DEFAULT_VIDEO_OUTPUT,
        default=_DEFAULT_VIDEO_OUTPUT,
        help="Override the default results/<video>/<video>.mimic.mp4 path",
    )
    video_group.add_argument(
        "--no-video-out",
        action="store_const",
        const=None,
        dest="video_out",
        help="Disable simulation video recording",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print both pipeline stages without running them",
    )
    return parser


def _resolve_path(path: Path, caller_directory: Path) -> Path:
    return path.resolve() if path.is_absolute() else (caller_directory / path).resolve()


def _timestamped_collision_path(path: Path, now: Optional[datetime] = None) -> Path:
    """Retain the requested stem while preserving an existing default video."""

    current = now or datetime.now().astimezone()
    timestamp = current.strftime("%Y%m%d-%H%M%S-%f")
    return path.with_name(f"{path.stem}.{timestamp}{path.suffix}")


def _default_video_path(
    output_directory: Path,
    video_stem: str,
    *,
    now: Optional[datetime] = None,
) -> Path:
    requested = output_directory / f"{video_stem}.mimic.mp4"
    if not requested.exists():
        return requested
    timestamped = _timestamped_collision_path(requested, now)
    candidate = timestamped
    index = 2
    while candidate.exists():
        candidate = timestamped.with_name(f"{timestamped.stem}.{index}{timestamped.suffix}")
        index += 1
    return candidate


def _explicit_video_path(path: Path, caller_directory: Path) -> Path:
    output = _resolve_path(path, caller_directory)
    if not output.suffix:
        output = output.with_suffix(".mp4")
    elif output.suffix.lower() != ".mp4":
        raise ValueError("--video-out must use the .mp4 extension")
    if output.exists():
        raise ValueError(f"Explicit video output already exists: {output}")
    return output


def _missing_inputs(paths: Sequence[Tuple[Path, str]]) -> Tuple[str, ...]:
    return tuple(
        f"{description} not found: {path}" for path, description in paths if not path.is_file()
    )


def _print_dry_run(
    *,
    video_arguments: Sequence[str],
    robot_arguments: Sequence[str],
    output_directory: Path,
    video_output: Optional[Path],
) -> None:
    print(f"Output directory: {output_directory}")
    print(f"Video stage: mimic-video-pipeline {shlex.join(video_arguments)}")
    print(f"Robot stage: mimic-robot-pipeline {shlex.join(robot_arguments)}")
    if video_output is not None:
        print(f"Planned simulation video: {video_output}")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    repository_root: Optional[Path] = None,
    caller_directory: Optional[Path] = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = (repository_root or _repository_root()).resolve()
    caller = (caller_directory or Path.cwd()).resolve()

    video = _resolve_path(args.video, caller)
    output = (
        _resolve_path(args.output, caller)
        if args.output is not None
        else root / "results" / video.stem
    )
    model = (
        _resolve_path(args.model, caller)
        if args.model is not None
        else root / "models" / "action_classifier_lstm.pt"
    )
    robot = ROBOT_DEFAULTS[args.robot].resolve(root)

    try:
        if args.video_out is _DEFAULT_VIDEO_OUTPUT:
            video_output = _default_video_path(output, video.stem)
        elif args.video_out is None:
            video_output = None
        else:
            video_output = _explicit_video_path(args.video_out, caller)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    required_inputs = (
        (video, "Input video"),
        (model, "Classifier checkpoint"),
        (robot.skill_config, "Skill configuration"),
        (robot.execution_config, "Robot execution configuration"),
        (robot.calibration, "Camera calibration"),
        (robot.retargeting_config, "Retargeting configuration"),
        (robot.pipeline_config, "Robot pipeline configuration"),
    )
    missing = _missing_inputs(required_inputs)
    if missing:
        for error in missing:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    task_input = output / f"{video.stem}_task_input.json"
    scores = output / f"{video.stem}_scores.json"
    world_waypoints = output / f"{video.stem}_world_waypoints.json"
    execution_log = output / f"{video.stem}_execution.jsonl"

    video_arguments = (
        str(video),
        "--model",
        str(model),
        "--skill-config",
        str(robot.skill_config),
        "--output",
        str(output),
        "--device",
        args.device,
        "--context-window",
        str(_DEFAULT_CONTEXT_WINDOW),
    )
    robot_arguments = (
        "--task-input",
        str(task_input),
        "--calibration",
        str(robot.calibration),
        "--retargeting-config",
        str(robot.retargeting_config),
        "--pipeline-config",
        str(robot.pipeline_config),
        "--waypoints",
        str(world_waypoints),
        "--overwrite",
        "--robot-config",
        str(robot.execution_config),
        "--log",
        str(execution_log),
    )
    if video_output is not None:
        robot_arguments += ("--video-out", str(video_output))
    if args.viewer:
        robot_arguments += ("--viewer",)

    if args.dry_run:
        _print_dry_run(
            video_arguments=video_arguments,
            robot_arguments=robot_arguments,
            output_directory=output,
            video_output=video_output,
        )
        return 0

    print(f"Processing {video.name} for robot {args.robot}")
    print(f"Artifacts: {output}")
    video_status = run_video_pipeline(video_arguments, repository_root=root)
    if video_status != 0:
        return video_status
    missing_video_artifacts = tuple(path for path in (task_input, scores) if not path.is_file())
    if missing_video_artifacts:
        for path in missing_video_artifacts:
            print(f"ERROR: Video stage did not create expected artifact: {path}", file=sys.stderr)
        return 1

    robot_status = run_robot_pipeline(robot_arguments)
    required_robot_artifacts = (world_waypoints, execution_log)
    if video_output is not None:
        required_robot_artifacts += (video_output,)
    missing_robot_artifacts = tuple(path for path in required_robot_artifacts if not path.is_file())
    if robot_status == 0 and missing_robot_artifacts:
        for path in missing_robot_artifacts:
            print(f"ERROR: Robot stage did not create expected artifact: {path}", file=sys.stderr)
        return 1
    if video_output is not None:
        if video_output.is_file():
            print(f"Simulation video: {video_output.resolve()}")
    return robot_status


if __name__ == "__main__":
    raise SystemExit(main())
