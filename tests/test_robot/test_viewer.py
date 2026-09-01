"""Live viewing and recording observe simulation without owning physics."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.simulate_robot import (
    _parser,
    _resolve_video_path,
    _timestamped_video_path,
    _video_session,
    _viewer_session,
    main as simulate_main,
)
from mimic.robot.factory import read_waypoint_sequence

pytest_plugins = ("tests.test_robot.execution_fixtures",)


class FakeViewerHandle:
    def __init__(self):
        self.sync_calls = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def is_running(self):
        return not self.closed

    def sync(self):
        self.sync_calls += 1


def test_viewer_session_syncs_after_advance_and_detaches_after_close(
    small_robot, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, io, _ = small_robot
    executor = SimpleNamespace(controller=SimpleNamespace(io=io))
    handle = FakeViewerHandle()
    launches = []

    def launch(model, data, **options):
        launches.append((model, data, options))
        return handle

    monkeypatch.setattr("mujoco.viewer.launch_passive", launch)
    monkeypatch.setattr("scripts.simulate_robot.time.sleep", lambda _duration: None)

    with _viewer_session(executor, True) as active:
        assert active is handle
        io.advance(0.01)

    assert launches == [(io.model, io.data, {"show_left_ui": False, "show_right_ui": False})]
    assert handle.sync_calls == 2
    assert handle.closed

    io.advance(0.01)
    assert handle.sync_calls == 2


def test_headless_session_does_not_require_simulation_adapter() -> None:
    with _viewer_session(SimpleNamespace(), False) as handle:
        assert handle is None


def test_video_flag_accepts_explicit_or_timestamped_mp4_paths() -> None:
    parser = _parser()
    required = ("--config", "robot.yaml", "--waypoints", "task.json", "--log", "run.jsonl")

    explicit = parser.parse_args((*required, "--video-out", "clips/run"))
    assert _resolve_video_path(explicit.video_out) == Path("clips/run.mp4")

    timestamped = parser.parse_args((*required, "--video-out"))
    assert _resolve_video_path(timestamped.video_out).match(
        "mimic-simulation-????????-??????-??????.mp4"
    )
    assert _timestamped_video_path(
        datetime(2026, 9, 1, 7, 8, 9, 123456, tzinfo=timezone.utc)
    ) == Path("mimic-simulation-20260901-070809-123456.mp4")


def test_video_path_refuses_non_mp4_and_existing_outputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".mp4 extension"):
        _resolve_video_path(tmp_path / "simulation.mov")

    output = tmp_path / "simulation.mp4"
    output.touch()
    with pytest.raises(ValueError, match="already exists"):
        _resolve_video_path(output)


def test_video_session_samples_completed_steps_and_releases_resources(
    small_robot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, io, _ = small_robot
    executor = SimpleNamespace(controller=SimpleNamespace(io=io))
    rendered = []
    writers = []

    class FakeRenderer:
        def __init__(self, model, *, height, width):
            assert model is io.model
            self.height = height
            self.width = width
            self.closed = False

        def update_scene(self, data):
            assert data is io.data
            rendered.append(float(data.time))

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    class FakeWriter:
        def __init__(self, path, options):
            self.path = path
            self.options = options
            self.frames = []
            self.closed = False

        def append_data(self, frame):
            self.frames.append(frame)

        def close(self):
            self.closed = True

    def fake_get_writer(path, **options):
        writer = FakeWriter(path, options)
        writers.append(writer)
        return writer

    monkeypatch.setattr("scripts.simulate_robot.mujoco.Renderer", FakeRenderer)
    monkeypatch.setattr("scripts.simulate_robot.imageio.get_writer", fake_get_writer)

    output = tmp_path / "simulation.mp4"
    with _video_session(executor, output) as capture:
        with _viewer_session(executor, False, capture):
            io.advance(0.01)
            io.advance(0.03)

    assert rendered == pytest.approx([0.01, 0.04])
    assert writers[0].path == output
    assert writers[0].options["fps"] == 30
    assert len(writers[0].frames) == 2
    assert writers[0].closed

    io.advance(0.01)
    assert rendered == pytest.approx([0.01, 0.04])


def test_simulation_log_replaces_existing_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pose = {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}
    waypoints = {
        **{name: pose for name in ("approach", "grasp", "lift", "lower", "retreat")},
        "path": [pose],
        "goal_position": [0, 0, 0],
    }
    config = tmp_path / "robot.yaml"
    task = tmp_path / "waypoints.json"
    log = tmp_path / "execution.jsonl"
    config.write_text("robot_execution: {}\n")
    task.write_text(json.dumps(waypoints))
    log.write_text("stale log contents\n")

    @dataclass(frozen=True)
    class FakeReport:
        success: bool = True

    def fake_build_executor(_config, record):
        record({"event": "configuration"})
        return SimpleNamespace(run=lambda _task: FakeReport())

    monkeypatch.setattr("scripts.simulate_robot.build_executor", fake_build_executor)
    monkeypatch.setattr("scripts.simulate_robot.importlib.metadata.version", lambda _name: "test")

    result = simulate_main(("--config", str(config), "--waypoints", str(task), "--log", str(log)))

    assert result == 0
    assert "stale log contents" not in log.read_text()
    assert log.read_text().splitlines()[0] == '{"event": "configuration"}'


def test_waypoint_sequence_reader_preserves_episode_order() -> None:
    pose = {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}
    first = {
        **{name: pose for name in ("approach", "grasp", "lift", "lower", "retreat")},
        "path": [pose],
        "goal_position": [1, 0, 0],
    }
    second = {**first, "goal_position": [2, 0, 0]}

    tasks = read_waypoint_sequence(
        {
            "schema": "mimic.world_waypoint_sequence.v1",
            "episodes": [first, second],
        }
    )

    assert [task.goal_position for task in tasks] == [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


def test_simulation_main_dispatches_multiple_episodes_as_one_playback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pose = {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}
    episode = {
        **{name: pose for name in ("approach", "grasp", "lift", "lower", "retreat")},
        "path": [pose],
        "goal_position": [0, 0, 0],
    }
    config = tmp_path / "robot.yaml"
    task = tmp_path / "waypoints.json"
    log = tmp_path / "execution.jsonl"
    config.write_text("robot_execution: {}\n")
    task.write_text(
        json.dumps(
            {
                "schema": "mimic.world_waypoint_sequence.v1",
                "episodes": [episode, episode],
            }
        )
    )
    received = []

    @dataclass(frozen=True)
    class FakePlayback:
        success: bool = True

    def fake_build_executor(_config, record):
        record({"event": "configuration"})

        def run_sequence(tasks):
            received.extend(tasks)
            return FakePlayback()

        return SimpleNamespace(run_sequence=run_sequence)

    monkeypatch.setattr("scripts.simulate_robot.build_executor", fake_build_executor)
    monkeypatch.setattr("scripts.simulate_robot.importlib.metadata.version", lambda _name: "test")

    result = simulate_main(("--config", str(config), "--waypoints", str(task), "--log", str(log)))

    assert result == 0
    assert len(received) == 2
