"""BEATs Trainer: A streamlined library for audio classification with BEATs transformers."""

__version__ = "0.1.0"
__author__ = "Benjamin Cretois"
__email__ = "benjamin.cretois@nina.no"

# Core feature extraction (always available)
from .feature_extractor import BEATsFeatureExtractor
from .checkpoint_utils import (
    ensure_checkpoint,
    list_available_models,
    download_beats_checkpoint,
    find_checkpoint,
    validate_checkpoint,
    get_model_info,
)

# Training components (optional - only if dependencies are available)
try:
    from .trainer import BEATsTrainer

    _TRAINING_AVAILABLE = True
except ImportError:
    BEATsTrainer = None  # Avoid unused import warning
    _TRAINING_AVAILABLE = False

# Package metadata
__title__ = "beats-trainer"
__summary__ = "A streamlined library for training and using BEATs models on audio classification tasks"
__uri__ = "https://github.com/benjamin-cretois/beats-trainer"
__license__ = "MIT"
__copyright__ = "Copyright 2024-2025 Benjamin Cretois"

# Public API - what users get when they import the package
__all__ = [
    # Version info
    "__version__",
    # Core feature extraction (always available)
    "BEATsFeatureExtractor",
    # Checkpoint management utilities
    "ensure_checkpoint",
    "list_available_models",
    "download_beats_checkpoint",
    "find_checkpoint",
    "validate_checkpoint",
    "get_model_info",
]

# Add training components if available
if _TRAINING_AVAILABLE:
    __all__.append("BEATsTrainer")
