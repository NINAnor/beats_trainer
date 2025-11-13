"""Simplified BEATs Trainer class - focused on core training functionality."""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd
import torch
import pytorch_lightning as pl

from ..core.config import Config
from ..data.datasets import validate_dataset
from ..data.module import BEATsDataModule
from ..core.model import BEATsLightningModule
from ..core.feature_extractor import BEATsFeatureExtractor

# Import modular components
from .callbacks import setup_training_callbacks, setup_pytorch_lightning_trainer
from .factory import BEATsTrainerFactory
from .utils import (
    configure_deterministic_mode,
    setup_logging_directory,
    validate_training_setup,
    print_training_summary,
    get_checkpoint_path,
    format_training_results,
)


class BEATsTrainer:
    """
    Simplified BEATs Trainer focused on core training functionality.

    This class provides a clean, high-level API for training BEATs models
    on custom audio classification datasets. Complex setup logic has been
    moved to separate utility modules for better maintainability.

    Examples:
        # Train from directory structure
        trainer = BEATsTrainer.from_directory("/path/to/dataset")
        trainer.train()

        # Train from pre-split data
        trainer = BEATsTrainer.from_split_directories(
            train_dir="data/train", val_dir="data/val"
        )
        trainer.train()

        # Advanced configuration
        config = Config(learning_rate=1e-4, max_epochs=100)
        trainer = BEATsTrainer.from_esc50("./datasets", config=config)
        trainer.train()
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        data_module: Optional[pl.LightningDataModule] = None,
    ):
        """
        Initialize BEATs trainer.

        Args:
            dataset: DataFrame with audio files and labels
            data_dir: Directory containing audio files
            config: Training configuration
            experiment_name: Name for this experiment
            log_dir: Directory for logs and checkpoints
        """
        # Store configuration
        self.config = config or Config()
        if experiment_name:
            self.config.experiment_name = experiment_name

        # Store data
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self._provided_data_module = data_module  # Store provided data module

        # Setup logging directory
        self.log_dir = setup_logging_directory(log_dir, self.config.experiment_name)

        # Initialize components (will be set up in _setup_components)
        self.data_module = None
        self.model = None
        self.trainer = None
        self.callbacks = None

        # Setup all components
        self._setup_components()

    def _setup_components(self):
        """Setup all training components."""
        # Configure deterministic mode for reproducibility
        configure_deterministic_mode()

        # Validate dataset (skip if using provided data module)
        if self._provided_data_module is None:
            dataset_stats = validate_dataset(self.dataset, self.data_dir)

        # Setup data module
        if self._provided_data_module is not None:
            # Use provided data module (for pre-split scenarios)
            self.data_module = self._provided_data_module
        else:
            # Create standard data module
            self.data_module = BEATsDataModule(
                dataset=self.dataset,
                data_dir=self.data_dir,
                config=self.config.data,
            )

        # Setup data module to get number of classes
        self.data_module.setup()

        # Update config with number of classes
        self.config.model.num_classes = self.data_module.num_classes

        # Setup model
        self.model = BEATsLightningModule(self.config, self.data_module.num_classes)

        # Setup callbacks and trainer
        self.callbacks = setup_training_callbacks(self.config)
        self.trainer = setup_pytorch_lightning_trainer(
            self.config, self.callbacks, self.log_dir
        )

        # Validate setup and print summary
        validation_results = validate_training_setup(self.data_module, self.model)
        print_training_summary(self.config, validation_results)

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Dictionary with training results
        """
        print(f"🚀 Starting training for: {self.config.experiment_name}")

        # Train the model
        self.trainer.fit(
            self.model,
            datamodule=self.data_module,
            ckpt_path=resume_from_checkpoint,
        )

        # Get training results
        results = self.trainer.callback_metrics
        formatted_results = format_training_results(results)

        print(f"✅ Training completed for: {self.config.experiment_name}")
        return formatted_results

    def test(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Test the model.

        Args:
            checkpoint_path: Path to checkpoint to test

        Returns:
            Dictionary with test results
        """
        # Get checkpoint path
        ckpt_path = get_checkpoint_path(self.trainer, checkpoint_path)
        print(f"🧪 Testing model from: {ckpt_path}")

        # Test the model
        test_results = self.trainer.test(
            self.model, datamodule=self.data_module, ckpt_path=ckpt_path
        )

        formatted_results = format_training_results(test_results[0] if test_results else {})
        print(f"✅ Testing completed")
        return formatted_results

    def predict(
        self,
        audio_paths,
        checkpoint_path: Optional[str] = None,
        return_probabilities: bool = False,
    ):
        """
        Make predictions on new audio files.

        Args:
            audio_paths: List of paths to audio files
            checkpoint_path: Path to checkpoint to use
            return_probabilities: Whether to return class probabilities

        Returns:
            Predictions for each audio file
        """
        # Get checkpoint path
        ckpt_path = get_checkpoint_path(self.trainer, checkpoint_path)

        # Load model from checkpoint
        model = BEATsLightningModule.load_from_checkpoint(ckpt_path)
        model.eval()

        # Make predictions
        predictions = []
        with torch.no_grad():
            for audio_path in audio_paths:
                # This would need to be implemented based on your model's predict method
                # For now, placeholder
                predictions.append({"file": audio_path, "prediction": "placeholder"})

        return predictions

    def get_feature_extractor(
        self, checkpoint_path: Optional[str] = None, **kwargs
    ) -> BEATsFeatureExtractor:
        """
        Get a feature extractor from the trained model.

        Args:
            checkpoint_path: Path to specific checkpoint
            **kwargs: Additional arguments for BEATsFeatureExtractor

        Returns:
            BEATsFeatureExtractor instance
        """
        # Get checkpoint path
        ckpt_path = get_checkpoint_path(self.trainer, checkpoint_path)
        print(f"Creating feature extractor from: {ckpt_path}")

        return BEATsFeatureExtractor(model_path=ckpt_path, **kwargs)

    # Factory methods - delegate to BEATsTrainerFactory
    @classmethod
    def from_directory(cls, *args, **kwargs):
        """Create trainer from directory structure."""
        return BEATsTrainerFactory.from_directory(cls, *args, **kwargs)

    @classmethod
    def from_csv(cls, *args, **kwargs):
        """Create trainer from CSV metadata."""
        return BEATsTrainerFactory.from_csv(cls, *args, **kwargs)

    @classmethod
    def from_esc50(cls, *args, **kwargs):
        """Create trainer for ESC-50 dataset."""
        return BEATsTrainerFactory.from_esc50(cls, *args, **kwargs)

    @classmethod
    def from_split_directories(cls, *args, **kwargs):
        """Create trainer from pre-split directories."""
        return BEATsTrainerFactory.from_split_directories(cls, *args, **kwargs)

    @classmethod
    def from_split_csvs(cls, *args, **kwargs):
        """Create trainer from pre-split CSV files."""
        return BEATsTrainerFactory.from_split_csvs(cls, *args, **kwargs)