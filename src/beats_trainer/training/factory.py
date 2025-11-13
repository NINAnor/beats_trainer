"""Factory methods for creating BEATsTrainer instances from different data sources."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from ..core.config import Config
from ..data.datasets import load_dataset, load_esc50, load_split_csvs, scan_directory_dataset
from ..data.module import PreSplitDataModule


class BEATsTrainerFactory:
    """Factory class for creating BEATsTrainer instances from various data sources."""

    @classmethod
    def from_directory(
        cls, 
        trainer_class,
        data_dir: Union[str, Path], 
        config: Optional[Config] = None, 
        **kwargs
    ):
        """
        Create trainer from directory structure.
        
        Expected structure:
            data_dir/
            ├── class1/
            │   ├── audio1.wav
            │   └── audio2.wav
            └── class2/
                ├── audio3.wav
                └── audio4.wav
        """
        dataset = load_dataset(data_dir, dataset_type="directory")
        return trainer_class(dataset=dataset, data_dir=data_dir, config=config, **kwargs)

    @classmethod
    def from_csv(
        cls,
        trainer_class,
        csv_path: Union[str, Path],
        data_dir: Union[str, Path],
        config: Optional[Config] = None,
        audio_column: str = "filename",
        label_column: str = "category",
        **kwargs,
    ):
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
            data_dir,
            dataset_type="csv",
            csv_path=csv_path,
            audio_column=audio_column,
            label_column=label_column,
        )
        return trainer_class(dataset=dataset, data_dir=data_dir, config=config, **kwargs)

    @classmethod
    def from_esc50(
        cls,
        trainer_class,
        data_dir: Union[str, Path] = "./datasets",
        config: Optional[Config] = None,
        **kwargs,
    ):
        """
        Create trainer for ESC-50 dataset with automatic download.

        Args:
            data_dir: Directory to download/find ESC-50 dataset
            config: Training configuration
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance configured for ESC-50
        """
        # Load ESC-50 dataset (auto-download if needed)
        dataset = load_esc50(data_dir, auto_download=True)
        
        # Use organized directory
        organized_dir = Path(data_dir) / "ESC50_organized"
        
        return trainer_class(
            dataset=dataset, data_dir=organized_dir, config=config, **kwargs
        )

    @classmethod
    def from_split_directories(
        cls,
        trainer_class,
        train_dir: Union[str, Path],
        val_dir: Union[str, Path],
        test_dir: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None,
        **kwargs,
    ):
        """
        Create trainer from pre-split directory structure.

        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data
            test_dir: Directory with test data (optional)
            config: Training configuration
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance with pre-split data
        """
        # Load the split datasets
        train_df = scan_directory_dataset(train_dir)
        val_df = scan_directory_dataset(val_dir)
        test_df = scan_directory_dataset(test_dir) if test_dir else None

        # Create pre-split data module with explicit data directories
        data_module = PreSplitDataModule(
            train_data=train_df,
            val_data=val_df,
            test_data=test_df if test_df is not None and not test_df.empty else None,
            train_data_dir=train_dir,
            val_data_dir=val_dir,
            test_data_dir=test_dir,
            batch_size=config.data.batch_size if config else 16,
            num_workers=config.data.num_workers if config else 4,
            sample_rate=config.data.sample_rate if config else 16000,
        )

        # Create trainer with pre-split data
        return cls._create_trainer_with_data_module(
            trainer_class, data_module, config, **kwargs
        )

    @classmethod
    def from_split_csvs(
        cls,
        trainer_class,
        train_csv: Union[str, Path],
        val_csv: Union[str, Path],
        test_csv: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        audio_column: str = "filename",
        label_column: str = "category",
        config: Optional[Config] = None,
        **kwargs,
    ):
        """
        Create trainer from pre-split CSV files.

        Args:
            train_csv: Path to training CSV
            val_csv: Path to validation CSV
            test_csv: Path to test CSV (optional)
            data_dir: Base directory for audio files
            audio_column: Name of audio filename column
            label_column: Name of label column
            config: Training configuration
            **kwargs: Additional arguments for BEATsTrainer

        Returns:
            BEATsTrainer instance with pre-split data
        """
        # Load the split datasets
        train_df, val_df, test_df = load_split_csvs(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            data_dir=data_dir,
            audio_column=audio_column,
            label_column=label_column,
        )

        # Create pre-split data module
        data_module = PreSplitDataModule(
            train_data=train_df,
            val_data=val_df,
            test_data=test_df if test_df is not None and not test_df.empty else None,
            batch_size=config.data.batch_size if config else 16,
            num_workers=config.data.num_workers if config else 4,
            sample_rate=config.data.sample_rate if config else 16000,
        )

        # Create trainer with pre-split data
        return cls._create_trainer_with_data_module(
            trainer_class, data_module, config, **kwargs
        )

    @staticmethod
    def _create_trainer_with_data_module(trainer_class, data_module, config, **kwargs):
        """Helper method to create trainer with a pre-configured data module."""
        # Create a dummy dataset for compatibility (data module will be used instead)
        dummy_dataset = pd.DataFrame({"filename": [], "category": []})
        
        # Create trainer instance with provided data module
        trainer = trainer_class(
            dataset=dummy_dataset,
            data_dir="",  # Not used with pre-split data
            config=config,
            data_module=data_module,  # Pass the pre-configured data module
            **kwargs
        )
        
        return trainer