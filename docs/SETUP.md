# Setup and Installation

Guide for setting up the Mimic project on different systems using `uv`.

## Prerequisites

- Python 3.9+
- Git
- CUDA 11.8+ (for GPU acceleration on RTX 3090, optional)
- 16GB+ RAM (for development)

## Installation

### 1. Install uv

`uv` is a blazingly fast Python package manager written in Rust. Install it first:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or via your package manager:**
```bash
# macOS
brew install uv

# Ubuntu/Debian
sudo apt install uv  # if available in your distro

# Arch
pacman -S uv
```

Verify installation:
```bash
uv --version
```

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/Mimic.git
cd Mimic
```

### 3. Sync Dependencies
`uv` automatically creates and manages a virtual environment:

```bash
# Sync all dependencies (creates venv automatically)
uv sync
```

This command:
- Creates `.venv/` directory (automatically managed by uv)
- Installs all dependencies
- Creates `uv.lock` for reproducible builds
- Activates the environment automatically for subsequent `uv` commands

### 4. Verify Installation
```bash
# Test imports
uv run python -c "import mimic; print(mimic.__version__)"

# Run tests
uv run pytest tests/

# Alternative: activate the venv and use normally
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -c "import mimic; print(mimic.__version__)"
pytest tests/
```

## Hardware-Specific Setup

### RTX 3090 (V-JEPA inference)
1. Install CUDA 11.8:
   ```bash
   # Download from https://developer.nvidia.com/cuda-11-8-0-download-archive
   ```

2. Install cuDNN 8.6:
   ```bash
   # Download from https://developer.nvidia.com/cudnn
   ```

3. Sync with GPU-specific PyTorch (uv handles this):
   ```bash
   uv sync
   # uv automatically installs CPU PyTorch by default
   # For CUDA support, modify pyproject.toml to specify GPU wheels
   ```

4. Verify GPU availability:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

   If CUDA is not available, update `pyproject.toml` to use GPU wheels and resync.

### M4 Pro MacBook (Data & simulation)
1. Just use uv (handles ARM automatically):
   ```bash
   uv sync
   # uv automatically selects compatible wheels for Apple Silicon
   ```

2. Verify Metal acceleration:
   ```bash
   uv run python -c "import torch; print(torch.backends.mps.is_available())"
   ```

## Special Dependencies

### V-JEPA 2, SAM2, and Other Git Repos

Add git dependencies to `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "vjepa @ git+https://github.com/facebookresearch/vjepa.git",
    "sam2 @ git+https://github.com/facebookresearch/sam2.git",
]
```

Then resync:
```bash
uv sync
```

### MuJoCo (already in pyproject.toml)
Already included in the base dependencies. Verify with:
```bash
uv run python -c "import mujoco; print(mujoco.__version__)"
```

## Configuration

1. Copy default config:
   ```bash
   cp configs/default.yaml configs/experiment_1.yaml
   ```

2. Adjust for your hardware:
   ```yaml
   device: cuda  # or "cpu", "mps"
   batch_size: 32  # Reduce for limited GPU memory
   ```

3. Create local override (optional):
   ```bash
   cp configs/default.yaml config.local.yaml
   ```

## Data Setup

1. Create data directories:
   ```bash
   mkdir -p data/{raw,processed,embeddings,annotations,splits}
   ```

2. Add your demonstration videos to `data/raw/`

## Verify Complete Setup

```bash
# 1. Test imports
uv run python -c "from mimic import common, vision, tracking, robot, integration"

# 2. Test configuration
uv run python -c "from mimic.config import get_config; cfg = get_config(); print(cfg.to_dict())"

# 3. Run tests
uv run pytest tests/ -v

# 4. Check data paths
uv run python -c "from mimic.config import get_data_dir; print(get_data_dir())"
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python -c "from mimic import *"
pytest tests/ -v
```

## Troubleshooting

### CUDA/GPU Issues
```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.device_count())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
uv run python ...
```

### Memory Issues
- Reduce `batch_size` in config
- Use smaller model (`hidden_size: 128` instead of 256)
- Enable `cache_embeddings: true`

### Virtual Environment Issues
```bash
# Recreate the venv from scratch
rm -rf .venv
uv sync

# Or if uv doesn't auto-create, explicitly create it
uv venv
uv sync
```

### Module Import Errors
```bash
# Reinstall the package in development mode
uv sync --reinstall

# Or use pip in the venv
source .venv/bin/activate
pip install -e .
```

## Development Setup

For contributing to the codebase:

```bash
# Sync with dev dependencies (automatically included)
uv sync

# Format code with black and isort (configured in pyproject.toml)
uv run black src/ scripts/
uv run isort src/ scripts/

# Type checking with mypy
uv run mypy src/mimic/

# Linting with flake8
uv run flake8 src/ scripts/

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Working with the Virtual Environment

You have two options:

**Option 1: Use `uv run` (recommended)**
```bash
uv run pytest tests/
uv run black src/
uv run python scripts/train_temporal_model.py
```

**Option 2: Activate `.venv` and work normally**
```bash
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pytest tests/
black src/
python scripts/train_temporal_model.py
deactivate  # Exit venv when done
```

---

For issues, see [README.md](../README.md) or open an issue on GitHub.
