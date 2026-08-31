# Setup and Installation

Guide for setting up the Mimic project on different systems.

## Prerequisites

- Python 3.9+
- Git
- CUDA 11.8+ (for GPU acceleration on RTX 3090)
- 16GB+ RAM (for development)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Mimic.git
cd Mimic
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
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

3. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. Verify GPU availability:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### M4 Pro MacBook (Data & simulation)
1. Create environment with ARM support:
   ```bash
   conda create -n mimic python=3.10
   conda activate mimic
   ```

2. Install PyTorch for Apple Silicon:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. Verify Metal acceleration:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

## Special Dependencies

### V-JEPA 2
```bash
pip install git+https://github.com/facebookresearch/vjepa.git
```

### SAM2
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### MuJoCo
```bash
pip install mujoco>=3.0.0
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
python -c "from mimic import common, vision, tracking, robot, integration"

# 2. Test configuration
python -c "from mimic.config import get_config; cfg = get_config(); print(cfg.to_dict())"

# 3. Run tests
pytest tests/ -v

# 4. Check data paths
python -c "from mimic.config import get_data_dir; print(get_data_dir())"
```

## Troubleshooting

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.device_count())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### Memory Issues
- Reduce `batch_size` in config
- Use smaller model (`hidden_size: 128` instead of 256)
- Enable `cache_embeddings: true`

### Import Errors
```bash
# Reinstall package in development mode
pip install -e .
```

## Development Setup

For contributing to the codebase:

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Format code
black src/ scripts/
isort src/ scripts/

# Type checking
mypy src/mimic/

# Linting
flake8 src/ scripts/
```

---

For issues, see [README.md](../README.md) or open an issue on GitHub.
