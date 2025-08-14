"""Dataset utilities and loaders for BEATs Trainer."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
import warnings


def scan_directory_dataset(
    data_dir: Union[str, Path], audio_extensions: List[str] = None
) -> pd.DataFrame:
    """
    Scan a directory structure to create a dataset DataFrame.

    Expected structure:
    data_dir/
    ├── class1/
    │   ├── audio1.wav
    │   └── audio2.wav
    ├── class2/
    │   └── audio3.wav
    └── ...

    Args:
        data_dir: Root directory containing class folders
        audio_extensions: List of valid audio extensions

    Returns:
        DataFrame with 'filename' and 'category' columns
    """
    if audio_extensions is None:
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data = []
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(class_dir.glob(f"*{ext}"))
            audio_files.extend(class_dir.glob(f"*{ext.upper()}"))

        if not audio_files:
            warnings.warn(f"No audio files found in {class_dir}")
            continue

        for audio_file in audio_files:
            relative_path = audio_file.relative_to(data_dir)
            data.append({"filename": str(relative_path), "category": class_name})

    if not data:
        raise ValueError(f"No audio files found in {data_dir}")

    df = pd.DataFrame(data)
    print(f"Found {len(df)} audio files across {df['category'].nunique()} classes")
    return df


def load_csv_dataset(
    csv_path: Union[str, Path],
    audio_column: str = "filename",
    label_column: str = "category",
) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        csv_path: Path to CSV file
        audio_column: Name of column containing audio file paths
        label_column: Name of column containing labels

    Returns:
        DataFrame with standardized 'filename' and 'category' columns
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if audio_column not in df.columns:
        raise ValueError(f"Audio column '{audio_column}' not found in CSV")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in CSV")

    # Standardize column names
    df = df[[audio_column, label_column]].copy()
    df.columns = ["filename", "category"]

    print(f"Loaded {len(df)} samples from CSV with {df['category'].nunique()} classes")
    return df


def load_dataset(
    data_source: Union[str, Path], dataset_type: str = "auto", **kwargs
) -> pd.DataFrame:
    """
    Load dataset from various sources.

    Args:
        data_source: Path to data directory or CSV file
        dataset_type: Type of dataset ("directory", "csv", or "auto")
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        DataFrame with 'filename' and 'category' columns
    """
    data_source = Path(data_source)

    if dataset_type == "auto":
        if data_source.is_dir():
            dataset_type = "directory"
        elif data_source.suffix.lower() == ".csv":
            dataset_type = "csv"
        else:
            raise ValueError(
                f"Cannot auto-detect dataset type for: {data_source}. "
                "Please specify dataset_type explicitly."
            )

    if dataset_type == "directory":
        return scan_directory_dataset(data_source, **kwargs)
    elif dataset_type == "csv":
        return load_csv_dataset(data_source, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def validate_dataset(df: pd.DataFrame, data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate dataset and return statistics.

    Args:
        df: Dataset DataFrame
        data_dir: Root directory containing audio files

    Returns:
        Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)

    # Check for missing files
    missing_files = []
    existing_files = []

    for _, row in df.iterrows():
        file_path = data_dir / row["filename"]
        if file_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)

    # Class distribution
    class_counts = df["category"].value_counts()

    stats = {
        "total_samples": len(df),
        "num_classes": df["category"].nunique(),
        "class_names": sorted(df["category"].unique()),
        "class_distribution": class_counts.to_dict(),
        "missing_files": len(missing_files),
        "existing_files": len(existing_files),
    }

    if missing_files:
        warnings.warn(f"Found {len(missing_files)} missing audio files")
        stats["missing_file_list"] = [
            str(f) for f in missing_files[:10]
        ]  # Show first 10

    # Check for class imbalance
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float("inf")

    if imbalance_ratio > 10:
        warnings.warn(
            f"Dataset is highly imbalanced (ratio: {imbalance_ratio:.1f}). "
            "Consider using balanced sampling or class weights."
        )

    stats["imbalance_ratio"] = imbalance_ratio

    return stats


# Preset dataset loaders for common datasets
def load_esc50(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load ESC-50 dataset."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "meta" / "esc50.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[["filename", "category"]].copy()

    # For ESC-50, we need to add the 'audio' subdirectory to filenames
    # since the data_dir points to the ESC-50-master root, not the audio folder
    df["filename"] = "audio/" + df["filename"]

    return df


def load_urbansound8k(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Load UrbanSound8K dataset."""
    data_dir = Path(data_dir)
    csv_path = data_dir / "metadata" / "UrbanSound8K.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"UrbanSound8K metadata not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["filename"] = df.apply(
        lambda x: f"fold{x['fold']}/{x['slice_file_name']}", axis=1
    )
    df["category"] = df["class"]
    return df[["filename", "category"]]


# Registry of preset loaders
PRESET_LOADERS = {
    "esc50": load_esc50,
    "urbansound8k": load_urbansound8k,
}
