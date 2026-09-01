"""Public one-command video-to-robot orchestration."""

from pathlib import Path

import pytest

from mimic.cli import ROBOT_DEFAULTS, _default_video_path, main


def _repository_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    required = (
        Path("models/action_classifier_lstm.pt"),
        *ROBOT_DEFAULTS["panda"].execution_configs.values(),
        ROBOT_DEFAULTS["panda"].calibration,
        ROBOT_DEFAULTS["panda"].retargeting_config,
        ROBOT_DEFAULTS["panda"].pipeline_config,
        ROBOT_DEFAULTS["panda"].skill_config,
    )
    for relative_path in required:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    return root


def _write_video_outputs(video_arguments: tuple[str, ...]) -> Path:
    output = Path(video_arguments[video_arguments.index("--output") + 1])
    video = Path(video_arguments[0])
    output.mkdir(parents=True, exist_ok=True)
    task_input = output / f"{video.stem}_task_input.json"
    task_input.write_text("{}\n")
    (output / f"{video.stem}_scores.json").write_text("{}\n")
    return task_input


def _write_robot_outputs(robot_arguments: tuple[str, ...]) -> None:
    for option in ("--waypoints", "--log"):
        path = Path(robot_arguments[robot_arguments.index(option) + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    if "--video-out" in robot_arguments:
        video_output = Path(robot_arguments[robot_arguments.index("--video-out") + 1])
        video_output.parent.mkdir(parents=True, exist_ok=True)
        video_output.write_bytes(b"mp4")


def test_cli_runs_both_stages_with_panda_defaults_and_prints_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "VIDEO_NAME.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_video_stage(arguments, *, repository_root):
        calls.append(("video", arguments, repository_root))
        _write_video_outputs(arguments)
        return 0

    def fake_robot_stage(arguments):
        calls.append(("robot", arguments))
        _write_robot_outputs(arguments)
        return 0

    monkeypatch.setattr("mimic.cli.run_video_pipeline", fake_video_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", fake_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 0
    output = root / "results" / "VIDEO_NAME"
    expected_video = output / "VIDEO_NAME.mimic.mp4"
    video_arguments = calls[0][1]
    robot_arguments = calls[1][1]
    assert calls[0][2] == root
    assert video_arguments[video_arguments.index("--device") + 1] == "cpu"
    assert video_arguments[video_arguments.index("--output") + 1] == str(output)
    assert robot_arguments[robot_arguments.index("--task-input") + 1] == str(
        output / "VIDEO_NAME_task_input.json"
    )
    assert robot_arguments[robot_arguments.index("--robot-config") + 1] == str(
        root / "configs/robots/panda/slow.yaml"
    )
    assert robot_arguments[robot_arguments.index("--calibration") + 1] == str(
        root / "data/annotations/calibrations.json"
    )
    assert robot_arguments[robot_arguments.index("--video-out") + 1] == str(expected_video)
    assert f"Simulation video: {expected_video}" in capsys.readouterr().out


def test_cli_config_selects_named_robot_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    observed: dict[str, object] = {}

    def fake_video_stage(arguments, *, repository_root):
        _write_video_outputs(arguments)
        return 0

    def fake_robot_stage(arguments):
        observed["robot_config"] = Path(arguments[arguments.index("--robot-config") + 1])
        _write_robot_outputs(arguments)
        return 0

    monkeypatch.setattr("mimic.cli.run_video_pipeline", fake_video_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", fake_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda", "--config", "fast"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 0
    assert observed["robot_config"] == root / "configs/robots/panda/fast.yaml"


def test_cli_config_rejects_unknown_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")

    def unexpected_stage(*args, **kwargs):
        raise AssertionError("unknown --config must stop before inference")

    monkeypatch.setattr("mimic.cli.run_video_pipeline", unexpected_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", unexpected_stage)

    result = main(
        ("--video", str(video), "--robot", "panda", "--config", "missing"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 2
    assert "--config must be one of" in capsys.readouterr().err


def test_default_video_keeps_canonical_path_when_a_recording_already_exists(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "demo.mimic.mp4"
    existing.touch()

    assert _default_video_path(tmp_path, "demo") == existing


def test_cli_replaces_existing_default_simulation_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "VIDEO_NAME.mp4"
    video.write_bytes(b"video")
    output = root / "results" / "VIDEO_NAME"
    output.mkdir(parents=True)
    canonical = output / "VIDEO_NAME.mimic.mp4"
    leftover = output / "VIDEO_NAME.mimic.20260901-080910-123456.mp4"
    canonical.write_bytes(b"old")
    leftover.write_bytes(b"older")
    observed: dict[str, object] = {}

    def fake_video_stage(arguments, *, repository_root):
        _write_video_outputs(arguments)
        return 0

    def fake_robot_stage(arguments):
        video_output = Path(arguments[arguments.index("--video-out") + 1])
        observed["video_out"] = video_output
        observed["canonical_exists_before_write"] = canonical.exists()
        observed["leftover_exists_before_write"] = leftover.exists()
        _write_robot_outputs(arguments)
        return 0

    monkeypatch.setattr("mimic.cli.run_video_pipeline", fake_video_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", fake_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 0
    assert observed["video_out"] == canonical
    assert observed["canonical_exists_before_write"] is False
    assert observed["leftover_exists_before_write"] is False
    assert canonical.read_bytes() == b"mp4"


def test_cli_can_disable_video_and_forward_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    calls = []

    def fake_video_stage(arguments, *, repository_root):
        calls.append(arguments)
        _write_video_outputs(arguments)
        return 0

    def fake_robot_stage(arguments):
        calls.append(arguments)
        _write_robot_outputs(arguments)
        return 0

    monkeypatch.setattr("mimic.cli.run_video_pipeline", fake_video_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", fake_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda", "--device", "mps", "--no-video-out"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 0
    assert calls[0][calls[0].index("--device") + 1] == "mps"
    assert "--video-out" not in calls[1]


def test_disabled_video_output_does_not_delete_an_existing_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    output = root / "results" / "demo"
    output.mkdir(parents=True)
    existing = output / "demo.mimic.mp4"
    existing.write_bytes(b"keep")

    def fake_video_stage(arguments, *, repository_root):
        _write_video_outputs(arguments)
        return 0

    def fake_robot_stage(arguments):
        _write_robot_outputs(arguments)
        return 0

    monkeypatch.setattr("mimic.cli.run_video_pipeline", fake_video_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", fake_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda", "--no-video-out"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 0
    assert existing.read_bytes() == b"keep"


def test_video_stage_failure_stops_before_robot_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr("mimic.cli.run_video_pipeline", lambda *args, **kwargs: 7)

    def unexpected_robot_stage(_arguments):
        raise AssertionError("robot stage must not run after video-stage failure")

    monkeypatch.setattr("mimic.cli.run_robot_pipeline", unexpected_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 7


def test_successful_video_stage_must_create_declared_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("mimic.cli.run_video_pipeline", lambda *args, **kwargs: 0)

    def unexpected_robot_stage(_arguments):
        raise AssertionError("robot stage must wait for complete video artifacts")

    monkeypatch.setattr("mimic.cli.run_robot_pipeline", unexpected_robot_stage)

    result = main(
        ("--video", str(video), "--robot", "panda"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 1
    assert "Video stage did not create expected artifact" in capsys.readouterr().err


def test_dry_run_preflights_and_prints_both_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")

    def unexpected_stage(*args, **kwargs):
        raise AssertionError("dry-run must not invoke either pipeline stage")

    monkeypatch.setattr("mimic.cli.run_video_pipeline", unexpected_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", unexpected_stage)

    result = main(
        ("--video", str(video), "--robot", "panda", "--dry-run"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Video stage: mimic-video-pipeline" in output
    assert "Robot stage: mimic-robot-pipeline" in output
    assert str(root / "results/demo/demo.mimic.mp4") in output


def test_preflight_reports_missing_default_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository_fixture(tmp_path)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    (root / "configs/robots/panda/slow.yaml").unlink()

    def unexpected_stage(*args, **kwargs):
        raise AssertionError("preflight failure must stop before inference")

    monkeypatch.setattr("mimic.cli.run_video_pipeline", unexpected_stage)
    monkeypatch.setattr("mimic.cli.run_robot_pipeline", unexpected_stage)

    result = main(
        ("--video", str(video), "--robot", "panda"),
        repository_root=root,
        caller_directory=tmp_path,
    )

    assert result == 2
    assert "Robot execution configuration not found" in capsys.readouterr().err
