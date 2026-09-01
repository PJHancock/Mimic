# Setup

## Requirements

- `uv`
- Python 3.9+ for the base package
- Python 3.10+ for the robot dependency group
- Git

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Install robot dependencies when running Panda simulation:

```bash
uv sync --group robot
uv run python scripts/fetch_panda_model.py
```

The model fetch is pinned and writes ignored Menagerie assets under `models/franka_emika_panda/upstream/`.

## Verify

```bash
uv run python -c "import mimic; print(mimic.__version__)"
uv run pytest tests/
```

Robot-focused checks require the robot group:

```bash
uv run --group robot pytest tests/test_robot/
```

## Run

Complete video-to-Panda simulation:

```bash
uv run --group robot mimic --video path/to/demo.mov --robot panda
```

Video inference only:

```bash
uv run mimic-video-pipeline path/to/demo.mov
```

Build world waypoints from an existing task input:

```bash
uv run mimic-robot-pipeline \
  --task-input results/demo/demo_task_input.json \
  --calibration data/annotations/calibrations.json \
  --pipeline-config configs/robot_pipeline.yaml \
  --waypoints results/demo/demo_world_waypoints.json
```

See [Robot execution](ROBOT_EXECUTION.md) for viewer, recording, and explicit simulation commands.

## Local configuration

`configs/default.yaml` is the baseline. Pass an explicit experiment YAML where supported or create ignored `config.local.yaml` for `mimic.config` overrides. Robot execution still requires complete explicit scene and safety values; it never fills null template fields.

Use `pyproject.toml` and `uv.lock` as the dependency sources. Do not maintain a parallel requirements file.
