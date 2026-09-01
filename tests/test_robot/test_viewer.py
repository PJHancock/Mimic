"""Live viewing observes completed MuJoCo steps without owning physics."""

from datetime import datetime, timezone
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
)

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
