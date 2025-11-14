# Installation and Setup

Complete installation guide for beats_trainer.

## Quick Install

### Option 1: Using uv (Recommended)

```bash
# Install from GitHub
uv add git+https://github.com/ninanor/beats_trainer.git
```

### Option 2: Using pip

```bash
# Install from GitHub
pip install git+https://github.com/ninanor/beats_trainer.git
```

## Optional Dependencies

### For Development

```bash
# Clone the repository
git clone https://github.com/ninanor/beats_trainer.git
cd beats_trainer

# Install in development mode with dev dependencies
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### For Notebooks

```bash
# Additional packages for Jupyter notebooks
uv add jupyter matplotlib seaborn plotly

# Or with pip
pip install jupyter matplotlib seaborn plotly
```

## System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.5.1+ (automatically installed)
- **CUDA**: Optional but recommended for GPU acceleration
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large datasets
- **Storage**: 2-10GB for model checkpoints and datasets

## GPU Setup (Optional but Recommended)

### CUDA Installation

Follow [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your system.

### Verify GPU Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name()}")
```

## Verification

Test your installation:

```python
# Test basic import
from beats_trainer import BEATsFeatureExtractor, BEATsTrainer

# Test feature extraction
extractor = BEATsFeatureExtractor()
print("✅ beats_trainer installed successfully!")

# Test model download (optional)
from beats_trainer import ensure_checkpoint
checkpoint_path = ensure_checkpoint()
print(f"✅ Default model downloaded to: {checkpoint_path}")
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'beats_trainer'**
- Make sure you installed with `uv add` or `pip install`
- Check you're using the correct Python environment

**CUDA out of memory**
- Reduce batch size in config: `config.data.batch_size = 16` 
- Use CPU instead: Set `CUDA_VISIBLE_DEVICES=""`

**Audio loading errors**
- Install additional audio backends: `pip install soundfile librosa`
- Check audio file format is supported

**Model download fails**
- Check internet connection
- Try manual download from GitHub releases

### Getting Help

- **Documentation**: Check the [docs/](../docs/) folder
- **Examples**: See [notebooks/](../notebooks/) and [example_scripts/](../example_scripts/)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/NINAnor/beats_trainer/issues)

## Development Setup

For contributing to beats_trainer:

```bash
# Clone and setup development environment
git clone https://github.com/ninanor/beats_trainer.git
cd beats_trainer

# Install with all dependencies
uv sync --all-extras

# Run tests
pytest

# Run with coverage
pytest --cov=beats_trainer

# Format code
black src/ test/
isort src/ test/
```