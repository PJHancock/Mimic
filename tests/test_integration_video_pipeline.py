"""Tests for the one-command integration pipeline runner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mimic.integration.run_video_pipeline import build_pipeline_command, main


def test_build_command_delegates_every_required_input(tmp_path: Path) -> None:
    command = build_pipeline_command(
        python_executable="python",
        pipeline_script=tmp_path / "process.py",
        video=tmp_path / "demo.mov",
        model=tmp_path / "model.pt",
        skill_config=tmp_path / "skills.yaml",
        output=tmp_path / "results",
        device="mps",
        context_window=24,
    )

    assert command == (
        "python",
        str(tmp_path / "process.py"),
        "--video",
        str(tmp_path / "demo.mov"),
        "--model",
        str(tmp_path / "model.pt"),
        "--skill-config",
        str(tmp_path / "skills.yaml"),
        "--output",
        str(tmp_path / "results"),
        "--device",
        "mps",
        "--context-window",
        "24",
    )


def test_simulation_requires_and_forwards_robot_config(tmp_path: Path) -> None:
    base = {
        "python_executable": "python",
        "pipeline_script": tmp_path / "process.py",
        "video": tmp_path / "demo.mov",
        "model": tmp_path / "model.pt",
        "skill_config": tmp_path / "skills.yaml",
        "output": tmp_path / "results",
        "device": "cpu",
        "context_window": 32,
        "simulate_robot": True,
    }
    with pytest.raises(ValueError, match="robot_config"):
        build_pipeline_command(**base)

    robot_config = tmp_path / "robot.yaml"
    calibration = tmp_path / "calibration.json"
    retargeting_config = tmp_path / "retargeting.yaml"
    robot_pipeline_config = tmp_path / "robot_pipeline.yaml"
    command = build_pipeline_command(
        **base,
        robot_config=robot_config,
        calibration=calibration,
        retargeting_config=retargeting_config,
        robot_pipeline_config=robot_pipeline_config,
        episode=2,
    )
    assert command[-11:] == (
        "--config",
        str(robot_config),
        "--calibration",
        str(calibration),
        "--retargeting-config",
        str(retargeting_config),
        "--robot-pipeline-config",
        str(robot_pipeline_config),
        "--episode",
        "2",
        "--simulate-robot",
    )


def test_main_preflights_and_runs_existing_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    video = tmp_path / "demo.mov"
    model = tmp_path / "model.pt"
    video.write_bytes(b"video fixture")
    model.write_bytes(b"model fixture")
    output = tmp_path / "output"
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("mimic.integration.run_video_pipeline.subprocess.run", fake_run)
    result = main(
        (
            str(video),
            "--model",
            str(model),
            "--skill-config",
            str(root / "configs" / "skills" / "pick_place.yaml"),
            "--output",
            str(output),
        ),
        repository_root=root,
    )

    assert result == 7
    assert len(calls) == 1
    command, cwd, check = calls[0]
    assert cwd == root
    assert check is False
    assert command[2:4] == ("--video", str(video))
    assert "--simulate-robot" not in command


def test_main_rejects_missing_checkpoint_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).resolve().parents[1]
    video = tmp_path / "demo.mov"
    video.write_bytes(b"video fixture")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("subprocess must not run after failed preflight")

    monkeypatch.setattr("mimic.integration.run_video_pipeline.subprocess.run", unexpected_run)
    result = main(
        (str(video), "--model", str(tmp_path / "missing.pt")),
        repository_root=root,
    )

    assert result == 2
    assert "Classifier checkpoint not found" in capsys.readouterr().err
