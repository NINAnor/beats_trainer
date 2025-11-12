"""Main BEATs Trainer class providing a simple API for training."""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from .config import Config
from .datasets import load_dataset, validate_dataset, PRESET_LOADERS
from .data_module import BEATsDataModule
from .model import BEATsLightningModule

# Import BEATsFeatureExtractor only for convenience method
from .feature_extractor import BEATsFeatureExtractor


class BEATsTrainer:
    """
    Main trainer class for BEATs audio classification.

    This class provides a simple, high-level API for training BEATs models
    on custom audio classification datasets. It handles the complete training
    pipeline including data loading, model setup, training, and evaluation.

    🎯 **Purpose**: Training and fine-tuning BEATs models
    🔗 **Relationship**: Can create BEATsFeatureExtractor instances for trained models
    ⚠️  **Note**: For feature extraction only, use BEATsFeatureExtractor directly

    Examples:
        # Train from directory structure
        trainer = BEATsTrainer.from_directory("/path/to/dataset")
        trainer.train()

        # Train from CSV
        trainer = BEATsTrainer.from_csv("/path/to/metadata.csv", data_dir="/path/to/audio")
        trainer.train()

        # Advanced configuration
        config = TrainingConfig(learning_rate=1e-4, max_epochs=100)
        trainer = BEATsTrainer.from_directory("/path/to/dataset", config=config)
        trainer.train()

        # Get feature extractor for trained model
        extractor = trainer.get_feature_extractor()
        features = extractor.extract_from_file("new_audio.wav")
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize BEATsTrainer.

        Args:
            dataset: DataFrame with 'filename' and 'category' columns
            data_dir: Root directory containing audio files
            config: Training configuration
            experiment_name: Name for the experiment
            log_dir: Directory to save logs and checkpoints
        """
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.config = config or Config()

        if experiment_name:
            self.config.experiment_name = experiment_name

        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate dataset
        self.dataset_stats = validate_dataset(self.dataset, self.data_dir)

        # Set number of classes
        if self.config.model.num_classes is None:
            self.config.model.num_classes = self.dataset_stats["num_classes"]

        # Initialize components
        self._setup_components()

    def _setup_components(self):
        """Setup PyTorch Lightning components."""
        # Set random seed
        pl.seed_everything(self.config.seed, workers=True)

        # Configure deterministic behavior for CUDA if needed
        self._configure_deterministic_mode()

        # Data module
        self.data_module = BEATsDataModule(
            dataset=self.dataset,
            data_dir=self.data_dir,
            config=self.config.data,
        )

        # Model
        self.model = BEATsLightningModule(
            config=self.config,
            num_classes=self.config.model.num_classes,
        )

        # Callbacks
        self.callbacks = self._setup_callbacks()

        # Logger
        self.logger = TensorBoardLogger(
            save_dir=str(self.log_dir),
            name=self.config.experiment_name,
            version=None,
        )

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            devices="auto" if torch.cuda.is_available() else 1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            precision=self.config.training.precision,
            callbacks=self.callbacks,
            logger=self.logger,
            log_every_n_steps=self.config.training.log_every_n_steps,
            default_root_dir=str(self.log_dir),
            deterministic=getattr(self.config.training, "deterministic", False),
            enable_progress_bar=True,
            enable_model_summary=True,
        )

    def _configure_deterministic_mode(self):
        """Configure deterministic behavior for reproducible results."""
        deterministic = getattr(self.config.training, "deterministic", False)

        if deterministic and torch.cuda.is_available():
            # Set CUBLAS workspace config for deterministic CuBLAS operations
            # This is required when using deterministic=True with CUDA >= 10.2
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                print(
                    "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA operations"
                )

    def _setup_callbacks(self):
        """Setup PyTorch Lightning callbacks."""
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.training.monitor_metric,
            mode="max" if "acc" in self.config.training.monitor_metric else "min",
            save_top_k=self.config.training.save_top_k,
            save_last=True,
            filename=f"{self.config.experiment_name}-{{epoch:02d}}-{{{self.config.training.monitor_metric}:.3f}}",
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor=self.config.training.monitor_metric,
            patience=self.config.training.patience,
            mode="max" if "acc" in self.config.training.monitor_metric else "min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        return callbacks

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        print(f"Starting training for experiment: {self.config.experiment_name}")
        print(
            f"Dataset: {self.dataset_stats['total_samples']} samples, {self.dataset_stats['num_classes']} classes"
        )
        print(f"Model: BEATs with {self.config.model.num_classes} output classes")
        print(
            f"Training config: {self.config.training.max_epochs} epochs, LR={self.config.training.learning_rate}"
        )

        # Configure deterministic mode for CUDA
        self._configure_deterministic_mode()

        # Save configuration
        config_path = self.log_dir / self.config.experiment_name / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(str(config_path))

        # Train
        self.trainer.fit(
            self.model,
            datamodule=self.data_module,
            ckpt_path=resume_from_checkpoint,
        )

        # Test on best model
        test_results = self.trainer.test(datamodule=self.data_module, ckpt_path="best")

        results = {
            "best_checkpoint": self.callbacks[0].best_model_path,
            "best_score": self.callbacks[0].best_model_score.item(),
            "test_results": test_results[0] if test_results else None,
            "dataset_stats": self.dataset_stats,
        }

        print(
            f"Training completed! Best {self.config.training.monitor_metric}: {results['best_score']:.4f}"
        )

        return results

    def test(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Test the model.

        Args:
            checkpoint_path: Path to checkpoint, uses best if None

        Returns:
            Test results dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = "best"

        test_results = self.trainer.test(
            datamodule=self.data_module,
            ckpt_path=checkpoint_path,
        )

        return test_results[0] if test_results else {}

    def predict(
        self, audio_paths: Union[str, list], checkpoint_path: Optional[str] = None
    ):
        """
        Make predictions on new audio files.

        Args:
            audio_paths: Path to audio file or list of paths
            checkpoint_path: Path to model checkpoint

        Returns:
            Predictions
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        # Load model from checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.callbacks[0].best_model_path

        model = BEATsLightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()

        # TODO: Implement prediction logic
        raise NotImplementedError("Prediction functionality coming soon!")

    def get_feature_extractor(
        self, checkpoint_path: Optional[str] = None, **kwargs
    ) -> BEATsFeatureExtractor:
        """
        Get a feature extractor using the trained model.

        This is a convenience method for users who want to extract features
        using their trained model checkpoint. For general feature extraction,
        create BEATsFeatureExtractor directly.

        Note: This method bridges BEATsTrainer (training) and BEATsFeatureExtractor
        (inference) by providing the trained model checkpoint to the extractor.

        Args:
            checkpoint_path: Path to trained model checkpoint (uses best if None)
            **kwargs: Additional arguments for BEATsFeatureExtractor

        Returns:
            BEATsFeatureExtractor instance configured with trained model

        Example:
            # Train a model
            trainer = BEATsTrainer.from_directory("/path/to/data")
            trainer.train()

            # Get feature extractor with trained model
            extractor = trainer.get_feature_extractor()
            features = extractor.extract_from_file("new_audio.wav")
        """
        if checkpoint_path is None:
            # Use best checkpoint from training
            checkpoint_path = getattr(self.callbacks[0], "best_model_path", None)

        # If still None, fall back to pretrained model
        if checkpoint_path is None:
            checkpoint_path = self.config.model.model_path

        return BEATsFeatureExtractor(model_path=checkpoint_path, **kwargs)

    @classmethod
    def from_directory(
        cls, data_dir: Union[str, Path], config: Optional[Config] = None, **kwargs
    ) -> "BEATsTrainer":


        dataset = load_dataset(data_dir, dataset_type="directory")
        return cls(dataset=dataset, data_dir=data_dir, config=config, **kwargs)

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        audio_column: str = "filename",
        label_column: str = "category",
        **kwargs,
    ) -> "BEATsTrainer":
        """
        Create trainer from CSV metadata.

        Args:
            csv_path: Path to CSV file
            data_dir: Directory containing audio files
            config: Training configuration
            audio_column: Name of audio filename column
            label_column: Name of label column
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance
        """
        dataset = load_dataset(
            csv_path,
            dataset_type="csv",
            audio_column=audio_column,
            label_column=label_column,
        )
        return cls(dataset=dataset, data_dir=data_dir, config=config, **kwargs)

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        **kwargs,
    ) -> "BEATsTrainer":
        """
        Create trainer using preset dataset loader.

        Args:
            preset_name: Name of preset ("esc50", "urbansound8k")
            data_dir: Directory containing dataset
            config: Training configuration
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance
        """
        if preset_name not in PRESET_LOADERS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(PRESET_LOADERS.keys())}"
            )

        dataset = PRESET_LOADERS[preset_name](data_dir)
        return cls(dataset=dataset, data_dir=data_dir, config=config, **kwargs)

    @classmethod
    def from_esc50(
        cls,
        data_dir: Union[str, Path] = "./datasets",
        config: Optional[Config] = None,
        auto_download: bool = True,
        force_download: bool = False,
        **kwargs,
    ) -> "BEATsTrainer":
        """
        Create trainer with ESC-50 dataset (auto-download and organize).

        This is a convenience method that automatically downloads, extracts,
        and organizes the ESC-50 dataset for training with .from_directory().

        Args:
            data_dir: Directory where to store/find ESC-50 dataset
            config: Training configuration
            auto_download: Automatically download if dataset not found
            force_download: Re-download even if dataset exists
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance ready for training

        Example:
            # Download and train on ESC-50
            trainer = BEATsTrainer.from_esc50()
            trainer.train()

            # Custom data directory
            trainer = BEATsTrainer.from_esc50(data_dir="./my_datasets")
            trainer.train()
        """
        from .datasets import download_and_organize_esc50

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if auto_download or force_download:
            # Download and organize ESC-50
            organized_dir = download_and_organize_esc50(data_dir, force_download)
        else:
            # Look for existing organized dataset
            organized_dir = data_dir / "ESC50_organized"
            if not organized_dir.exists():
                raise FileNotFoundError(
                    f"ESC-50 dataset not found at {organized_dir}. "
                    f"Set auto_download=True to download automatically."
                )

        # Create trainer from organized directory
        return cls.from_directory(organized_dir, config=config, **kwargs)

    @classmethod
    def from_split_directories(
        cls,
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        **kwargs
    ) -> "BEATsTrainer":
        """
        Create trainer from pre-split train/val/test directories.

        Expected directory structure:
        data_dir/
        ├── train/
        │   ├── class1/
        │   │   └── audio_files...
        │   └── class2/
        │       └── audio_files...
        ├── val/
        │   └── class_dirs...
        └── test/
            └── class_dirs...

        Args:
            data_dir: Root directory containing train/val/test splits
            config: Training configuration
            train_dir: Name of training directory (default: "train")
            val_dir: Name of validation directory (default: "val")
            test_dir: Name of test directory (default: "test")
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance with pre-split data

        Example:
            # Standard directory structure
            trainer = BEATsTrainer.from_split_directories("./my_dataset")
            trainer.train()

            # Custom directory names
            trainer = BEATsTrainer.from_split_directories(
                "./my_dataset", 
                train_dir="training",
                val_dir="validation", 
                test_dir="testing"
            )
        """
        from .datasets import load_split_directories

        splits = load_split_directories(
            data_dir, train_dir=train_dir, val_dir=val_dir, test_dir=test_dir
        )

        # Create trainer with pre-split data
        return cls._from_splits(
            splits=splits,
            data_dir=data_dir,
            config=config,
            **kwargs
        )

    @classmethod
    def from_split_csvs(
        cls,
        data_dir: Union[str, Path],
        train_csv: Union[str, Path],
        config: Optional[Config] = None,
        val_csv: Union[str, Path] = None,
        test_csv: Union[str, Path] = None,
        audio_column: str = "filename",
        label_column: str = "category",
        **kwargs
    ) -> "BEATsTrainer":
        """
        Create trainer from pre-split CSV files.

        Args:
            data_dir: Directory containing audio files
            train_csv: Path to training CSV file
            config: Training configuration
            val_csv: Path to validation CSV file (optional)
            test_csv: Path to test CSV file (optional)
            audio_column: Name of column containing audio filenames
            label_column: Name of column containing labels
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance with pre-split data

        Example:
            # Basic usage
            trainer = BEATsTrainer.from_split_csvs(
                data_dir="./audio_files",
                train_csv="train.csv",
                val_csv="val.csv",
                test_csv="test.csv"
            )
            trainer.train()

            # Custom column names
            trainer = BEATsTrainer.from_split_csvs(
                data_dir="./audio_files",
                train_csv="train.csv",
                audio_column="audio_path",
                label_column="class_name"
            )
        """
        from .datasets import load_split_csvs

        splits = load_split_csvs(
            data_dir=data_dir,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            audio_column=audio_column,
            label_column=label_column
        )

        # Create trainer with pre-split data
        return cls._from_splits(
            splits=splits,
            data_dir=data_dir,
            config=config,
            **kwargs
        )

    @classmethod
    def _from_splits(
        cls,
        splits: Dict[str, pd.DataFrame],
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        **kwargs
    ) -> "BEATsTrainer":
        """
        Internal method to create trainer from pre-split dataframes.

        Args:
            splits: Dictionary containing 'train', 'val', 'test' DataFrames
            data_dir: Directory containing audio files
            config: Training configuration
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance with pre-split data
        """
        # Use dummy dataset for main dataset parameter (not used with pre_split=True)
        dummy_dataset = splits["train"].copy() if not splits["train"].empty else pd.DataFrame()

        # Create trainer instance
        trainer = cls(dataset=dummy_dataset, data_dir=data_dir, config=config, **kwargs)

        # Override data module with pre-split version
        trainer.data_module = BEATsDataModule(
            dataset=dummy_dataset,
            data_dir=Path(data_dir),
            config=trainer.config.data,
            pre_split=True,
            train_df=splits["train"],
            val_df=splits["val"],
            test_df=splits["test"]
        )

        return trainer
