# Using UV with Mimic

Comprehensive guide to using `uv` as the package manager for the Mimic project.

## What is UV?

`uv` is an extremely fast Python package manager written in Rust. It's a drop-in replacement for `pip` with several advantages:

- **10-100x faster** than pip
- **Reproducible installs** with lock files
- **Automatic virtual environment** management
- **Built-in script execution**
- **Python installation** management

## Installation

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Package managers
```bash
brew install uv          # macOS
sudo apt install uv      # Ubuntu/Debian (if available)
pacman -S uv            # Arch
```

## Quick Start

### First Time Setup
```bash
# Clone and enter the project
git clone https://github.com/yourusername/Mimic.git
cd Mimic

# Sync dependencies (creates .venv automatically)
uv sync

# You're ready to go!
```

## Common Commands

### Installing & Managing Dependencies

```bash
# Sync all dependencies from pyproject.toml
uv sync

# Install with dev dependencies
uv sync --all-extras

# Install a specific extra (e.g., docs)
uv sync --extra docs

# Update all dependencies to latest compatible versions
uv lock --upgrade

# Lock specific package to new version
uv lock --upgrade-package torch
```

### Running Commands

**With `uv run` (recommended for CI/scripting):**
```bash
uv run python scripts/train_temporal_model.py
uv run pytest tests/
uv run black src/
```

**With activated `.venv` (recommended for interactive work):**
```bash
source .venv/bin/activate          # Linux/macOS
# or
.venv\Scripts\activate             # Windows

pytest tests/
python scripts/train_temporal_model.py
black src/

deactivate  # Exit when done
```

### Virtual Environment

```bash
# Create venv (usually automatic with uv sync)
uv venv

# Create venv with specific Python version
uv venv --python 3.11

# Remove venv
rm -rf .venv
```

### Python Version Management

```bash
# List installed Python versions
uv python list

# Install specific Python version
uv python install 3.11

# Use specific Python in project
uv venv --python 3.11
uv sync
```

### Adding Dependencies

**For runtime dependencies:**
Edit `pyproject.toml`:
```toml
dependencies = [
    # ... existing ...
    "new-package>=1.0.0",
]
```

Then sync:
```bash
uv sync
```

**For dev dependencies:**
```toml
[project.optional-dependencies]
dev = [
    # ... existing ...
    "new-dev-tool>=1.0.0",
]
```

Or in the `[tool.uv]` section:
```toml
[tool.uv]
dev-dependencies = [
    "new-dev-tool>=1.0.0",
]
```

Then sync:
```bash
uv sync
```

### Git Dependencies

```toml
dependencies = [
    "vjepa @ git+https://github.com/facebookresearch/vjepa.git",
    "sam2 @ git+https://github.com/facebookresearch/sam2.git@main",
]
```

Then:
```bash
uv sync
```

### Lock File

The `uv.lock` file is automatically created and maintained:
- Commit it to git for reproducible builds
- Everyone gets the exact same versions
- `uv sync` reads from the lock file

```bash
# Regenerate lock file
uv lock

# Upgrade dependencies and update lock
uv lock --upgrade

# Upgrade specific package in lock
uv lock --upgrade-package pytorch
```

## Development Workflow

### Initial Setup
```bash
git clone https://github.com/yourusername/Mimic.git
cd Mimic
uv sync  # This handles everything
```

### Daily Development
```bash
# Activate venv once per terminal session
source .venv/bin/activate

# Run tests
pytest tests/

# Format code
black src/ scripts/
isort src/ scripts/

# Type check
mypy src/mimic/

# Lint
flake8 src/ scripts/
```

### Adding a New Dependency
```bash
# Edit pyproject.toml
vim pyproject.toml

# Add your package to dependencies or [project.optional-dependencies]

# Sync to install
uv sync

# Verify it works
uv run python -c "import your_package"
```

### Running Scripts

All scripts support both modes:

**Via uv run:**
```bash
uv run python scripts/train_temporal_model.py --config configs/default.yaml
uv run pytest tests/ -v
```

**Via activated venv:**
```bash
source .venv/bin/activate
python scripts/train_temporal_model.py --config configs/default.yaml
pytest tests/ -v
```

## Project Structure for UV

### pyproject.toml
Main configuration file with:
- Project metadata (name, version, description)
- Dependencies (runtime)
- Optional dependencies (dev, docs)
- Tool configurations (black, pytest, mypy, etc.)
- UV-specific settings

### uv.lock
Auto-generated lock file:
- Exact versions of all dependencies
- Ensures reproducible builds
- Should be committed to git

### .venv/
Virtual environment directory:
- Automatically managed by uv
- Don't commit to git (.gitignore handles it)
- Recreate anytime with `uv sync`

## Tips & Tricks

### Skip the Lock File Check
If you need to install without checking the lock file:
```bash
uv sync --no-lock
```

### See What's Installed
```bash
# List all installed packages
uv pip list

# Show specific package info
uv pip show torch
```

### Export to Requirements Format
```bash
# Generate requirements.txt from lock file
uv export --output-file requirements.txt

# With hashes (for extra security)
uv export --hashes --output-file requirements.txt
```

### Tree View of Dependencies
```bash
uv pip tree
```

### Cache Management
```bash
# Clear uv cache
uv cache clean

# See cache size
uv cache dir
```

## Troubleshooting

### "uv: command not found"
Make sure uv is in your PATH:
```bash
# Verify installation
which uv
uv --version
```

### Stale Lock File
```bash
# Regenerate lock file
rm uv.lock
uv lock

# Or update it
uv lock --upgrade
```

### Virtual Environment Issues
```bash
# Recreate from scratch
rm -rf .venv
uv sync
```

### Python Version Mismatch
```bash
# Check Python version used
uv python list

# Use specific version
uv venv --python 3.11
uv sync

# Or use uv to install Python
uv python install 3.11
```

### Slow Installation
This shouldn't happen with uv! But if it does:
```bash
# Try clearing cache
uv cache clean
uv sync --refresh

# Or check network
uv sync --verbose
```

## Comparison with Other Tools

| Feature | uv | pip | poetry | pipenv |
|---------|----|----|--------|--------|
| Speed | ⚡⚡⚡ | 🐢 | ⚡ | 🐢 |
| Lock file | ✓ | ✗ | ✓ | ✓ |
| Venv management | Auto | Manual | Auto | Auto |
| Config file | pyproject.toml | requirements.txt | pyproject.toml | Pipfile |
| Ease of use | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## Resources

- **Official Docs**: https://docs.astral.sh/uv/
- **GitHub**: https://github.com/astral-sh/uv
- **Why UV?**: https://astral.sh/blog/uv

## FAQ

**Q: Do I still need to use `pip`?**
A: No. `uv` replaces pip entirely for this project.

**Q: Can I use `pip` and `uv` together?**
A: Not recommended. Use one or the other. Stick with `uv`.

**Q: What about `conda`?**
A: `uv` is an alternative to conda. For this project, use `uv`.

**Q: Does it work on all platforms?**
A: Yes. macOS (Intel/Apple Silicon), Linux, Windows all supported.

**Q: Can I lock Python version?**
A: Yes, in `pyproject.toml`:
```toml
requires-python = ">=3.9,<3.13"
```

**Q: How do I use `uv` in CI/CD?**
A: Same as local:
```yaml
- run: uv sync
- run: uv run pytest tests/
```

---

For more information, see [SETUP.md](./SETUP.md) or the official [uv documentation](https://docs.astral.sh/uv/).
