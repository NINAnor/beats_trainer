"""BEATs Trainer: A streamlined library for audio classification with BEATs transformers."""

__version__ = "0.1.0"
__author__ = "Benjamin Cretois"
__email__ = "benjamin.cretois@nina.no"

# Core components - import the main classes and functions users need
from .feature_extractor import BEATsFeatureExtractor
from .checkpoint_utils import (
    ensure_checkpoint,
    list_available_models,
    download_beats_checkpoint,
    find_checkpoint,
    validate_checkpoint,
)

# Optional imports (only if dependencies are available)
try:
    from .trainer import BEATsTrainer

    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False

_CONFIG_AVAILABLE = True

# Public API - what users get when they do `from beats_trainer import *`
__all__ = [
    # Version info
    "__version__",
    # Core feature extraction (always available)
    "BEATsFeatureExtractor",
    # Checkpoint management
    "ensure_checkpoint",
    "list_available_models",
    "download_beats_checkpoint",
    "find_checkpoint",
    "validate_checkpoint",
]

# Add training components if available
if _TRAINING_AVAILABLE:
    __all__.extend(
        [
            "BEATsTrainer",
            "BEATsDataModule",
            "BEATsClassifier",
        ]
    )

if _CONFIG_AVAILABLE:
    __all__.append("TrainingConfig")

# Package metadata
__title__ = "beats-trainer"
__summary__ = "A streamlined library for training and using BEATs models on audio classification tasks"
__uri__ = "https://github.com/benjamin-cretois/beats-trainer"
__license__ = "MIT"
__copyright__ = "Copyright 2024-2025 Benjamin Cretois"
__all__ = [
    "BEATsTrainer",
    "BEATsFeatureExtractor",
    "download_beats_checkpoint",
    "ensure_checkpoint",
    "list_available_models",
    "find_checkpoint",
    "get_model_info",
]
