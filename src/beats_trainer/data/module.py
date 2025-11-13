"""PyTorch Lightning data module for BEATs training."""

import librosa
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

from ..core.config import DataConfig


def collate_audio_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length audio data.
    
    Args:
        batch: List of (audio_tensor, padding_mask, label) tuples
        
    Returns:
        Tuple of (padded_audio, padding_mask, labels) tensors
    """
    # Separate the components
    audio_tensors, padding_masks, labels = zip(*batch)
    
    # Find the maximum length in the batch
    max_length = max(audio.shape[0] for audio in audio_tensors)
    
    # Pad all audio tensors to the same length
    padded_audio = []
    batch_padding_masks = []
    
    for audio, old_mask in zip(audio_tensors, padding_masks):
        current_length = audio.shape[0]
        
        if current_length < max_length:
            # Pad with zeros
            padding_needed = max_length - current_length
            padded = torch.nn.functional.pad(audio, (0, padding_needed), value=0.0)
            
            # Update padding mask (True = padded/masked, False = real audio)
            new_mask = torch.cat([
                torch.zeros(current_length, dtype=torch.bool),  # Real audio = False
                torch.ones(padding_needed, dtype=torch.bool)    # Padded = True
            ])
        else:
            padded = audio
            new_mask = torch.zeros(current_length, dtype=torch.bool)
            
        padded_audio.append(padded)
        batch_padding_masks.append(new_mask)
    
    # Stack into batches
    audio_batch = torch.stack(padded_audio)
    padding_batch = torch.stack(batch_padding_masks)
    label_batch = torch.tensor(labels, dtype=torch.long)
    
    return audio_batch, padding_batch, label_batch


class AudioDataset(Dataset):
    """Dataset for loading audio files."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_dir: Path,
        sample_rate: int = 16000,
        transform=None,
    ):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.transform = transform

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.dataframe["category"])
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Load audio
        audio_path = self.data_dir / row["filename"]
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Create padding mask
        padding_mask = torch.zeros(1, audio_tensor.shape[0], dtype=torch.bool).squeeze(
            0
        )

        # Apply transform if any
        if self.transform:
            audio_tensor = self.transform(audio_tensor)

        # Encode label
        label = self.label_encoder.transform([row["category"]])[0]

        return audio_tensor, padding_mask, label


class BEATsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: pd.DataFrame,
        data_dir: Path,
        config: DataConfig,
        pre_split: bool = False,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.config = config
        self.pre_split = pre_split
        
        # For pre-split datasets
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        if self.pre_split:
            # Use pre-provided splits
            self._setup_pre_split()
        else:
            # Perform automatic splitting
            self._setup_auto_split()

        # Store number of classes
        self.num_classes = self.train_dataset.num_classes

    def _setup_pre_split(self):
        """Setup datasets from pre-split dataframes."""
        if self.train_df is not None:
            self.train_dataset = AudioDataset(
                self.train_df, self.data_dir, self.config.sample_rate
            )

        if self.val_df is not None and len(self.val_df) > 0:
            self.val_dataset = AudioDataset(
                self.val_df, self.data_dir, self.config.sample_rate
            )

        if self.test_df is not None and len(self.test_df) > 0:
            self.test_dataset = AudioDataset(
                self.test_df, self.data_dir, self.config.sample_rate
            )

    def _setup_auto_split(self):
        """Setup datasets with automatic splitting."""
        # Shuffle dataset
        dataset_shuffled = self.dataset.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        # Split dataset
        if self.config.test_split > 0:
            train_val, test = train_test_split(
                dataset_shuffled,
                test_size=self.config.test_split,
                random_state=42,
                stratify=dataset_shuffled["category"],
            )
        else:
            train_val = dataset_shuffled
            test = pd.DataFrame()

        if self.config.val_split > 0:
            train, val = train_test_split(
                train_val,
                test_size=self.config.val_split / (1 - self.config.test_split),
                random_state=42,
                stratify=train_val["category"],
            )
        else:
            train = train_val
            val = pd.DataFrame()

        # Create datasets
        self.train_dataset = AudioDataset(train, self.data_dir, self.config.sample_rate)

        if len(val) > 0:
            self.val_dataset = AudioDataset(val, self.data_dir, self.config.sample_rate)

        if len(test) > 0:
            self.test_dataset = AudioDataset(
                test, self.data_dir, self.config.sample_rate
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )


class PreSplitDataModule(pl.LightningDataModule):
    """Data module for pre-split datasets."""

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        train_data_dir: Optional[str] = None,
        val_data_dir: Optional[str] = None, 
        test_data_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_data_dir = Path(train_data_dir) if train_data_dir else None
        self.val_data_dir = Path(val_data_dir) if val_data_dir else None
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        # Use provided data directories or try to infer from first file
        if self.train_data_dir:
            train_data_dir = self.train_data_dir
        else:
            # Fallback: extract data directory from the first training file
            first_file = self.train_data.iloc[0]["filename"]
            if "/" in str(first_file):
                # If it's a full path, extract directory
                train_data_dir = Path(first_file).parent.parent
            else:
                # If it's just a filename, assume current directory
                train_data_dir = Path(".")

        # Create datasets with their respective data directories
        self.train_dataset = AudioDataset(self.train_data, train_data_dir, self.sample_rate)
        
        if self.val_data is not None and len(self.val_data) > 0:
            val_data_dir = self.val_data_dir if self.val_data_dir else train_data_dir
            self.val_dataset = AudioDataset(self.val_data, val_data_dir, self.sample_rate)
        
        if self.test_data is not None and len(self.test_data) > 0:
            test_data_dir = self.test_data_dir if self.test_data_dir else train_data_dir
            self.test_dataset = AudioDataset(self.test_data, test_data_dir, self.sample_rate)

        # Store number of classes
        self.num_classes = self.train_dataset.num_classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )
