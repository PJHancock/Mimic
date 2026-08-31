# Contributing Guide

## Project Organization

This project is structured for **parallel, independent development**. Each subsystem has its own folder with minimal coupling.

### Subsystems

| Subsystem | Path | Owner | Responsibility |
|-----------|------|-------|---|
| Data Pipeline | `src/mimic/data_pipeline/` | — | Video recording, transcription, annotation |
| Vision | `src/mimic/vision/` | — | V-JEPA inference, temporal model, training |
| Tracking | `src/mimic/tracking/` | — | Hand/object tracking, coordinate mapping |
| Robot | `src/mimic/robot/` | — | Task representation, state machine, control |
| Integration | `src/mimic/integration/` | — | End-to-end pipeline, visualization |

**Key principle**: Each subsystem exports a clean interface via `__init__.py`. Minimize cross-module dependencies.

## Shared Conventions

### Data Types
All modules use types defined in `src/mimic/common/types.py`. If you need a new type:
1. Add it to `types.py` with clear docstring
2. Use it consistently across modules
3. Never duplicate type definitions

### Configuration
- Global config: `src/mimic/config.py`
- Experiment configs: `configs/*.yaml`
- Local overrides: Create `config.local.yaml` (in `.gitignore`)

### Cached Data
Embeddings and tracking results should be cached:
```
data/
├── embeddings/       # V-JEPA embeddings (indexed by video hash)
├── tracks/           # Cached hand/object tracks
└── processed/        # Cleaned video frames
```

### File Naming
- Videos: `demo_NNN.mp4` (padded number)
- Embeddings: `demo_NNN_embeddings.pt` (PyTorch tensor)
- Annotations: `demo_NNN_annotations.json` (JSON with timestamps)

## Workflow

### Starting New Work

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Work in your subsystem folder** — minimize changes outside your area

3. **Write tests** in `tests/` mirroring your module structure
   ```
   src/mimic/vision/temporal_model.py
   tests/test_vision/test_temporal_model.py
   ```

4. **Run tests before committing**
   ```bash
   pytest tests/test_vision/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "vision: add GRU temporal model for action classification"
   ```

### Shared Data Access

If you need to use cached data from another subsystem:

```python
from mimic.common.types import EmbeddingCache
from mimic.vision import get_cached_embeddings

# Don't recompute; load cache
embeddings = get_cached_embeddings("demo_001")
```

**Never**:
- Assume intermediate files exist (check before using)
- Modify another subsystem's cache without their knowledge
- Skip cache creation (always save results for reuse)

### Integration Testing

When your subsystem feeds into another:

```python
# tests/test_integration.py
def test_vision_to_robot_pipeline():
    # Use actual output from vision module
    actions = temporal_model.infer(embeddings)
    
    # Feed into robot module
    commands = state_machine.translate(actions)
    
    # Verify correctness
    assert len(commands) > 0
```

## Running Experiments

1. **Create a config file** in `configs/experiment_name.yaml`
2. **Run the experiment** via script:
   ```bash
   python scripts/train_temporal_model.py --config configs/experiment_name.yaml
   ```
3. **Save results** in `experiments/experiment_name/`
   - Model weights
   - Evaluation metrics
   - Config file (copy)
   - Notes on results

## Adding Dependencies

Only add to `requirements.txt` if:
- Essential for your subsystem
- No redundant alternatives exist
- Lightweight or already used elsewhere

Run `pip freeze > requirements_frozen.txt` after changes so others can reproduce exactly.

## Common Pitfalls

❌ **Don't**:
- Import deeply between subsystems (use interfaces in `__init__.py`)
- Create new data formats (use existing types in `common/types.py`)
- Commit large data files (use `data/.gitignore`)
- Hardcode paths or configs
- Skip caching computation results

✅ **Do**:
- Export clean APIs from your `__init__.py`
- Use `src/mimic/config.py` for settings
- Cache expensive computations to `data/`
- Write idempotent code (same input → same output)
- Test your subsystem in isolation

## Questions?

Check existing subsystem patterns in:
- `src/mimic/vision/` (reference implementation)
- `tests/test_vision/` (test patterns)
- `scripts/train_temporal_model.py` (script patterns)

---

**Goal**: You should be able to work on your subsystem for days without worrying about breaking someone else's work.
