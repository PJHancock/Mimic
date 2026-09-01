"""Run explicit processed world-space pick/place waypoints in MuJoCo."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import imageio.v2 as imageio
import mujoco

from mimic.robot.factory import build_executor, read_waypoint_sequence
from mimic.robot.simulation import MuJoCoAdapter

_TIMESTAMPED_VIDEO = object()
_VIDEO_FPS = 30
_VIDEO_WIDTH_PX = 640
_VIDEO_HEIGHT_PX = 480


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--waypoints",
        type=Path,
        required=True,
        help="JSON of processed world tool poses and object goal; not raw tracking",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="JSONL diagnostics file; replaces existing contents",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show and real-time-pace the live simulation; use mjpython on macOS",
    )
    parser.add_argument(
        "--video-out",
        nargs="?",
        type=Path,
        const=_TIMESTAMPED_VIDEO,
        help=(
            "Write a simulation MP4 to PATH, or a timestamped MP4 when PATH is omitted; "
            "use mjpython on macOS"
        ),
    )
    return parser


@contextmanager
def _viewer_session(
    executor,
    enabled: bool,
    advance_observer: Optional[Callable[[float], None]] = None,
):
    """Attach optional display/capture observers to the executor's adapter."""
    if not enabled and advance_observer is None:
        yield None
        return

    io = executor.controller.io
    if not isinstance(io, MuJoCoAdapter):
        raise TypeError("Viewer and video output require the MuJoCo simulation adapter")
    if not enabled:
        io.set_advance_observer(advance_observer)
        try:
            yield None
        finally:
            io.set_advance_observer(None)
        return

    from mujoco import viewer

    try:
        handle_context = viewer.launch_passive(
            io.model,
            io.data,
            show_left_ui=False,
            show_right_ui=False,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}. On macOS, run this command with `uv run --group robot mjpython`"
        ) from exc

    with handle_context as handle:
        simulation_started_s = None
        wall_started_s = None

        def synchronize(duration_s: float) -> None:
            nonlocal simulation_started_s, wall_started_s
            if not handle.is_running():
                raise RuntimeError("MuJoCo viewer closed during execution")
            if advance_observer is not None:
                advance_observer(duration_s)
            if simulation_started_s is None:
                simulation_started_s = float(io.data.time) - duration_s
                wall_started_s = time.monotonic()
            assert wall_started_s is not None
            handle.sync()
            target_wall_s = wall_started_s + float(io.data.time) - simulation_started_s
            delay_s = target_wall_s - time.monotonic()
            if delay_s > 0:
                time.sleep(delay_s)

        io.set_advance_observer(synchronize)
        handle.sync()
        try:
            yield handle
        finally:
            io.set_advance_observer(None)


def _timestamped_video_path(now: Optional[datetime] = None) -> Path:
    current = now or datetime.now().astimezone()
    return Path(f"mimic-simulation-{current:%Y%m%d-%H%M%S-%f}.mp4")


def _resolve_video_path(requested) -> Optional[Path]:
    if requested is None:
        return None
    output = _timestamped_video_path() if requested is _TIMESTAMPED_VIDEO else Path(requested)
    if not output.suffix:
        output = output.with_suffix(".mp4")
    elif output.suffix.lower() != ".mp4":
        raise ValueError("--video-out must use the .mp4 extension")
    if output.exists():
        raise ValueError(f"Video output already exists: {output}")
    return output


@contextmanager
def _video_session(executor, output: Optional[Path]):
    """Render completed simulation intervals without advancing physics."""
    if output is None:
        yield None
        return

    io = executor.controller.io
    if not isinstance(io, MuJoCoAdapter):
        raise TypeError("Video output requires the MuJoCo simulation adapter")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Video output already exists: {output}")

    try:
        renderer = mujoco.Renderer(io.model, height=_VIDEO_HEIGHT_PX, width=_VIDEO_WIDTH_PX)
    except Exception as exc:
        hint = (
            ". On macOS, run this command with `uv run --group robot mjpython`"
            if sys.platform == "darwin"
            else ""
        )
        raise RuntimeError(f"Unable to initialize MuJoCo video rendering: {exc}{hint}") from exc
    try:
        writer = imageio.get_writer(
            output,
            fps=_VIDEO_FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
    except Exception as exc:
        renderer.close()
        raise RuntimeError(f"Unable to initialize MP4 writer for {output}: {exc}") from exc

    next_frame_s = 0.0
    last_rendered_time_s = None

    def write_frame() -> None:
        nonlocal last_rendered_time_s
        renderer.update_scene(io.data)
        writer.append_data(renderer.render().copy())
        last_rendered_time_s = float(io.data.time)

    def capture(_duration_s: float) -> None:
        nonlocal next_frame_s
        current_time_s = float(io.data.time)
        while current_time_s + 1e-9 >= next_frame_s:
            write_frame()
            next_frame_s += 1 / _VIDEO_FPS

    try:
        yield capture
    finally:
        try:
            current_time_s = float(io.data.time)
            if last_rendered_time_s is None or current_time_s > last_rendered_time_s + 1e-9:
                write_frame()
        finally:
            try:
                writer.close()
            finally:
                renderer.close()


def _wait_for_viewer_close(handle) -> None:
    print("Execution finished; close the MuJoCo viewer to exit.")
    while handle.is_running():
        handle.sync()
        time.sleep(1 / 60)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        video_path = _resolve_video_path(args.video_out)
    except ValueError as exc:
        parser.error(str(exc))
    tasks = read_waypoint_sequence(json.loads(args.waypoints.read_text()))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as stream:

        def record(event):
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

        executor = build_executor(args.config, record)
        record(
            {
                "event": "metadata",
                "config": args.config.read_text(),
                "waypoints_sha256": hashlib.sha256(args.waypoints.read_bytes()).hexdigest(),
                "video_out": str(video_path.resolve()) if video_path is not None else None,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("mink", "mujoco", "qpsolvers", "daqp")
                },
            }
        )
        try:
            with _video_session(executor, video_path) as capture_frame:
                with _viewer_session(executor, args.viewer, capture_frame) as viewer_handle:
                    report = (
                        executor.run(tasks[0]) if len(tasks) == 1 else executor.run_sequence(tasks)
                    )
                    print(json.dumps(asdict(report), indent=2, allow_nan=False))
                    if viewer_handle is not None and viewer_handle.is_running():
                        _wait_for_viewer_close(viewer_handle)
        except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2
    if video_path is not None:
        print(f"Wrote simulation video to {video_path}")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
