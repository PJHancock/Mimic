# Contributing

Read `AGENTS.md` before changing the project. Preserve the distinction between implementation fixes and model/design changes.

## Repository map

- `src/mimic/common/` — shared records and constants.
- `src/mimic/data_pipeline/` — audio-derived label preparation.
- `src/mimic/vision/` — frame features and temporal classification.
- `src/mimic/tracking/` — object tracking and calibration
- `src/mimic/skills/` — catalog, transition graph, postprocessing, and handlers.
- `src/mimic/robot/` — task geometry and deterministic simulation execution.
- `src/mimic/integration/` — persisted schemas and pipeline orchestration.
- `tests/` — focused unit and integration tests.

Use public package interfaces when they express the required contract. Deep imports are acceptable for backend-specific implementation and tests when the ownership is explicit.

## Workflow

1. Recover the intended behavior from tests, configs, and authoritative docs.
2. Make one focused conceptual change.
3. Add or update a regression test when behavior changes.
4. Run the narrowest relevant tests, then expand verification in proportion to risk.
5. Report what was and was not verified.

Common commands:

```bash
uv sync
uv run pytest tests/test_integration_action_results.py
uv run pytest tests/test_tracking/
uv run --group robot pytest tests/test_robot/
uv run black --check src scripts tests
uv run mypy src/mimic
```

## Dependencies

Declare runtime dependencies in `pyproject.toml`, use dependency groups for scoped tooling, and commit the updated `uv.lock`. Do not add a parallel requirements file.

## Data and artifacts

- Raw videos, embeddings, fetched robot assets, and generated output belong in ignored paths.
- Keep committed result artifacts only when they provide active reproducibility or diagnostic evidence.
- Persist robot handoff data only as `mimic.demo_task_input.v1`; raw score distributions remain separate diagnostics.
- Do not invent new coordinate, timing, catalog, or robot-command semantics without explicit approval and documentation.

Authoritative subsystem documents are indexed in `docs/README.md`.
