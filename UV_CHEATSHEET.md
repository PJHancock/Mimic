# UV Cheatsheet

Quick reference for common `uv` commands used in the Mimic project.

## Setup & Install

```bash
# First time setup
git clone <repo>
cd Mimic
uv sync                    # Install all dependencies

# Recreate environment from scratch
rm -rf .venv uv.lock
uv sync
```

## Running Commands

```bash
# Use uv run (automatic venv, no activation needed)
uv run pytest tests/
uv run python scripts/train_temporal_model.py
uv run black src/

# Or activate venv once per session
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pytest tests/
deactivate                      # Exit when done
```

## Dependency Management

```bash
# Add a runtime dependency
# 1. Edit pyproject.toml in [project] dependencies
# 2. uv sync

# Add a dev dependency
# 1. Edit pyproject.toml in [project.optional-dependencies] dev
# 2. uv sync

# Update dependencies
uv lock --upgrade              # Update all
uv lock --upgrade-package torch # Update one package

# View dependencies
uv pip list
uv pip tree
```

## Common Tasks

```bash
# Tests
uv run pytest tests/                # All tests
uv run pytest tests/test_vision/    # Specific module
uv run pytest -v tests/             # Verbose

# Code quality
uv run black src/ scripts/          # Format
uv run isort src/ scripts/          # Sort imports
uv run mypy src/mimic/              # Type check
uv run flake8 src/ scripts/         # Lint
uv run pre-commit run --all-files   # All checks

# Scripts
uv run python scripts/train_temporal_model.py
uv run python scripts/run_inference.py --video data/raw/demo.mp4
```

## Python Version

```bash
# List available Python versions
uv python list

# Install specific Python version
uv python install 3.11

# Use specific Python in project
uv venv --python 3.11
uv sync
```

## Lock File

```bash
# Generate/regenerate lock file
uv lock

# Update lock file and upgrade deps
uv lock --upgrade

# Update specific package in lock
uv lock --upgrade-package pytorch
```

## Cache

```bash
# Clear cache
uv cache clean

# Show cache directory
uv cache dir
```

## Troubleshooting

```bash
# Reinstall everything fresh
rm -rf .venv uv.lock
uv sync

# Force resync ignoring lock file
uv sync --no-lock

# Verbose output for debugging
uv sync --verbose

# Update specific package
uv sync --upgrade-package <package-name>
```

## Installation Methods Comparison

| Task | Command |
|------|---------|
| Full setup | `uv sync` |
| Run command | `uv run pytest tests/` |
| Add dependency | Edit `pyproject.toml`, then `uv sync` |
| Activate venv | `source .venv/bin/activate` |
| List packages | `uv pip list` |
| Clean cache | `uv cache clean` |

## When to Use Each

**Use `uv run` for:**
- One-off commands
- Scripts in CI/CD
- Running without venv activation
- Reproducible commands

**Use activated `.venv` for:**
- Interactive development
- Running multiple commands
- REPL/Jupyter notebooks
- Faster command execution

## Environment Variables

```bash
# Force specific Python version
UV_PYTHON=3.11 uv sync

# Skip lock file
UV_NO_LOCK=1 uv sync

# Use specific index
UV_INDEX_URL=https://index.example.com uv sync
```

## CI/CD Usage

```bash
# GitHub Actions example
- run: curl -LsSf https://astral.sh/uv/install.sh | sh
- run: uv sync
- run: uv run pytest tests/
- run: uv run black --check src/
```

---

For more details, see:
- [docs/SETUP.md](docs/SETUP.md) — Full setup guide
- [docs/UV_GUIDE.md](docs/UV_GUIDE.md) — Comprehensive uv guide
- [Official Docs](https://docs.astral.sh/uv/) — Complete uv documentation
