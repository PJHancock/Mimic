#!/usr/bin/env python3
"""Run the complete video-to-postprocessed-actions pipeline from one entry point."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from mimic.integration.action_results import load_skill_system


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("context window must be nonnegative")
    return parsed


def build_pipeline_command(
    *,
    python_executable: str,
    pipeline_script: Path,
    video: Path,
    model: Path,
    skill_config: Path,
    output: Path,
    device: str,
    context_window: int,
    simulate_robot: bool = False,
    robot_config: Optional[Path] = None,
    calibration: Optional[Path] = None,
    retargeting_config: Optional[Path] = None,
    robot_pipeline_config: Optional[Path] = None,
    episode: Optional[int] = None,
) -> Tuple[str, ...]:
    """Build the subprocess argument vector without invoking a shell."""
    if context_window < 0:
        raise ValueError("context_window must be nonnegative")
    if simulate_robot and any(
        value is None
        for value in (robot_config, calibration, retargeting_config, robot_pipeline_config)
    ):
        raise ValueError(
            "robot_config, calibration, retargeting_config, and robot_pipeline_config "
            "are required when simulate_robot is enabled"
        )
    if episode is not None and episode < 1:
        raise ValueError("episode must be a positive one-based integer")

    command = [
        python_executable,
        str(pipeline_script),
        "--video",
        str(video),
        "--model",
        str(model),
        "--skill-config",
        str(skill_config),
        "--output",
        str(output),
        "--device",
        device,
        "--context-window",
        str(context_window),
    ]
    if simulate_robot:
        command.extend(
            (
                "--config",
                str(robot_config),
                "--calibration",
                str(calibration),
                "--retargeting-config",
                str(retargeting_config),
                "--robot-pipeline-config",
                str(robot_pipeline_config),
            )
        )
        if episode is not None:
            command.extend(("--episode", str(episode)))
        command.append("--simulate-robot")
    return tuple(command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input demonstration video")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Classifier checkpoint; defaults to models/action_classifier_lstm.pt",
    )
    parser.add_argument(
        "--skill-config",
        type=Path,
        default=None,
        help="Skill graph/settings YAML; defaults to configs/skills/pick_place.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to results/<video-name>",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Torch inference device",
    )
    parser.add_argument(
        "--context-window",
        type=_nonnegative_integer,
        default=32,
        help="LSTM frames before and after the current frame",
    )
    parser.add_argument(
        "--simulate-robot",
        action="store_true",
        help="Build calibrated world waypoints and invoke MuJoCo execution",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="Robot execution YAML required with --simulate-robot",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Camera homography JSON required with --simulate-robot",
    )
    parser.add_argument(
        "--retargeting-config",
        type=Path,
        help="Table-to-world YAML; defaults to configs/retargeting.yaml",
    )
    parser.add_argument(
        "--robot-pipeline-config",
        type=Path,
        help="Explicit path/waypoint YAML required with --simulate-robot",
    )
    parser.add_argument(
        "--episode",
        type=int,
        help="One-based complete episode to simulate when multiple are present",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the delegated command without running it",
    )
    return parser


def _existing_file(path: Path, description: str) -> Optional[str]:
    if not path.is_file():
        return f"{description} not found: {path}"
    return None


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    repository_root: Optional[Path] = None,
) -> int:
    args = _parser().parse_args(argv)
    root = (repository_root or _repository_root()).resolve()
    caller_directory = Path.cwd()

    video = (caller_directory / args.video).resolve()
    model = (
        (caller_directory / args.model).resolve()
        if args.model is not None
        else root / "models" / "action_classifier_lstm.pt"
    )
    skill_config = (
        (caller_directory / args.skill_config).resolve()
        if args.skill_config is not None
        else root / "configs" / "skills" / "pick_place.yaml"
    )
    output = (
        (caller_directory / args.output).resolve()
        if args.output is not None
        else root / "results" / video.stem
    )
    pipeline_script = root / "scripts" / "process_demo_video.py"
    robot_config = (
        (caller_directory / args.robot_config).resolve() if args.robot_config is not None else None
    )
    calibration = (
        (caller_directory / args.calibration).resolve() if args.calibration is not None else None
    )
    retargeting_config = (
        (caller_directory / args.retargeting_config).resolve()
        if args.retargeting_config is not None
        else root / "configs" / "retargeting.yaml"
    )
    robot_pipeline_config = (
        (caller_directory / args.robot_pipeline_config).resolve()
        if args.robot_pipeline_config is not None
        else None
    )

    required_files: Tuple[Tuple[Path, str], ...] = (
        (video, "Input video"),
        (model, "Classifier checkpoint"),
        (skill_config, "Skill configuration"),
        (pipeline_script, "Pipeline script"),
    )
    if args.simulate_robot:
        missing = [
            name
            for name, value in (
                ("--robot-config", robot_config),
                ("--calibration", calibration),
                ("--robot-pipeline-config", robot_pipeline_config),
            )
            if value is None
        ]
        if missing:
            print(
                f"ERROR: {', '.join(missing)} required with --simulate-robot",
                file=sys.stderr,
            )
            return 2
        assert robot_config is not None
        assert calibration is not None
        assert robot_pipeline_config is not None
        required_files = (
            *required_files,
            (robot_config, "Robot configuration"),
            (calibration, "Camera calibration"),
            (retargeting_config, "Retargeting configuration"),
            (robot_pipeline_config, "Robot pipeline configuration"),
        )

    for path, description in required_files:
        error = _existing_file(path, description)
        if error is not None:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2

    try:
        load_skill_system(skill_config).build_postprocessor()
    except (OSError, ValueError) as exc:
        print(f"ERROR: Invalid skill configuration: {exc}", file=sys.stderr)
        return 2

    command = build_pipeline_command(
        python_executable=sys.executable,
        pipeline_script=pipeline_script,
        video=video,
        model=model,
        skill_config=skill_config,
        output=output,
        device=args.device,
        context_window=args.context_window,
        simulate_robot=args.simulate_robot,
        robot_config=robot_config,
        calibration=calibration,
        retargeting_config=retargeting_config,
        robot_pipeline_config=robot_pipeline_config,
        episode=args.episode,
    )
    print(f"Running: {shlex.join(command)}")
    if args.dry_run:
        return 0
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
