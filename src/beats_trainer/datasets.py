"""Dataset utilities and loaders for BEATs Trainer."""

import pandas as pd
import zipfile
import urllib.request
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union
import warnings
from tqdm import tqdm


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
def download_with_progress(url: str, destination: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading {url}")

    def progress_hook(block_num, block_size, total_size):
        if hasattr(progress_hook, "pbar"):
            progress_hook.pbar.update(block_size)
        else:
            progress_hook.pbar = tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            )

    urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
    if hasattr(progress_hook, "pbar"):
        progress_hook.pbar.close()
    print(f"Downloaded to {destination}")


def download_esc50(data_dir: Union[str, Path], force_download: bool = False) -> Path:
    """
    Download and extract ESC-50 dataset.

    Args:
        data_dir: Directory where to download and extract the dataset
        force_download: If True, re-download even if dataset exists

    Returns:
        Path to the extracted ESC-50 directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ESC-50 download URL
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = data_dir / "ESC-50-master.zip"
    extracted_dir = data_dir / "ESC-50-master"

    # Check if already exists
    if extracted_dir.exists() and not force_download:
        print(f"ESC-50 dataset already exists at {extracted_dir}")
        return extracted_dir

    # Download if needed
    if not zip_path.exists() or force_download:
        download_with_progress(esc50_url, zip_path)

    # Extract
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip file
    zip_path.unlink()

    print(f"ESC-50 dataset extracted to {extracted_dir}")
    return extracted_dir


def organize_esc50_for_training(
    esc50_dir: Union[str, Path], output_dir: Union[str, Path] = None
) -> Path:
    """
    Organize ESC-50 dataset into class folders for .from_directory() usage.

    Args:
        esc50_dir: Path to ESC-50-master directory
        output_dir: Output directory for organized dataset (default: esc50_dir_parent/ESC50_organized)

    Returns:
        Path to organized dataset directory
    """
    esc50_dir = Path(esc50_dir)

    if output_dir is None:
        output_dir = esc50_dir.parent / "ESC50_organized"
    else:
        output_dir = Path(output_dir)

    # Read metadata
    meta_file = esc50_dir / "meta" / "esc50.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {meta_file}")

    df = pd.read_csv(meta_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Organizing ESC-50 dataset to {output_dir}")

    # Group by category and copy files
    for category, group in tqdm(df.groupby("category"), desc="Organizing classes"):
        # Create category directory
        category_dir = output_dir / category.replace(
            " ", "_"
        )  # Replace spaces with underscores
        category_dir.mkdir(exist_ok=True)

        # Copy audio files
        for _, row in group.iterrows():
            src_file = esc50_dir / "audio" / row["filename"]
            dst_file = category_dir / row["filename"]

            if src_file.exists():
                if not dst_file.exists():  # Don't overwrite existing files
                    shutil.copy2(src_file, dst_file)
            else:
                warnings.warn(f"Source file not found: {src_file}")

    # Print statistics
    class_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    total_files = sum(len(list(d.glob("*.wav"))) for d in class_dirs)

    print("✅ ESC-50 organized successfully!")
    print(f"   📁 Classes: {len(class_dirs)}")
    print(f"   🎵 Audio files: {total_files}")
    print(f"   📍 Location: {output_dir}")

    return output_dir


def download_and_organize_esc50(
    data_dir: Union[str, Path], force_download: bool = False
) -> Path:
    """
    Download and organize ESC-50 dataset in one step.

    Args:
        data_dir: Directory where to download and organize the dataset
        force_download: If True, re-download even if dataset exists

    Returns:
        Path to organized dataset directory ready for .from_directory()
    """
    data_dir = Path(data_dir)

    # Check if organized dataset already exists
    organized_dir = data_dir / "ESC50_organized"
    if organized_dir.exists() and not force_download:
        class_dirs = [d for d in organized_dir.iterdir() if d.is_dir()]
        if len(class_dirs) == 50:  # ESC-50 has 50 classes
            print(f"ESC-50 organized dataset already exists at {organized_dir}")
            return organized_dir

    # Download raw dataset
    raw_dir = download_esc50(data_dir, force_download)

    # Organize for training
    organized_dir = organize_esc50_for_training(raw_dir, organized_dir)

    return organized_dir


def load_esc50(data_dir: Union[str, Path], auto_download: bool = True) -> pd.DataFrame:
    """
    Load ESC-50 dataset.

    Args:
        data_dir: Root directory containing or where to download ESC-50
        auto_download: If True, automatically download and organize if not found

    Returns:
        DataFrame for organized dataset ready for .from_directory()
    """
    data_dir = Path(data_dir)

    # Try to find organized dataset first
    organized_dir = data_dir / "ESC50_organized"
    if organized_dir.exists():
        return scan_directory_dataset(organized_dir)

    # Try to find raw ESC-50 dataset
    raw_dir = data_dir / "ESC-50-master"
    if raw_dir.exists():
        csv_path = raw_dir / "meta" / "esc50.csv"
        if csv_path.exists():
            # Organize the raw dataset
            organized_dir = organize_esc50_for_training(raw_dir)
            return scan_directory_dataset(organized_dir)

    # Auto-download if requested
    if auto_download:
        print("ESC-50 dataset not found. Downloading and organizing...")
        organized_dir = download_and_organize_esc50(data_dir)
        return scan_directory_dataset(organized_dir)
    else:
        raise FileNotFoundError(
            f"ESC-50 dataset not found in {data_dir}. "
            f"Set auto_download=True to download automatically, or download manually from: "
            f"https://github.com/karoldvl/ESC-50/archive/master.zip"
        )


def download_urbansound8k(data_dir: Union[str, Path]) -> None:
    """
    UrbanSound8K requires manual download due to licensing.
    Provide instructions for manual download.
    """
    print("UrbanSound8K Dataset Download Instructions:")
    print("=" * 50)
    print("UrbanSound8K requires manual download due to licensing requirements.")
    print("Please follow these steps:")
    print("1. Visit: https://urbansounddataset.weebly.com/urbansound8k.html")
    print("2. Fill out the form to request download access")
    print("3. Download the dataset and extract to:", data_dir)
    print("4. The directory structure should be:")
    print("   UrbanSound8K/")
    print("   ├── metadata/")
    print("   │   └── UrbanSound8K.csv")
    print("   └── audio/")
    print("       ├── fold1/")
    print("       ├── fold2/")
    print("       └── ...")
    raise FileNotFoundError("UrbanSound8K requires manual download")


def organize_urbansound8k_for_training(
    us8k_dir: Union[str, Path], output_dir: Union[str, Path] = None
) -> Path:
    """
    Organize UrbanSound8K dataset into class folders for .from_directory() usage.

    Args:
        us8k_dir: Path to UrbanSound8K directory
        output_dir: Output directory for organized dataset (default: us8k_dir_parent/UrbanSound8K_organized)

    Returns:
        Path to organized dataset directory
    """
    us8k_dir = Path(us8k_dir)

    if output_dir is None:
        output_dir = us8k_dir.parent / "UrbanSound8K_organized"
    else:
        output_dir = Path(output_dir)

    # Read metadata
    meta_file = us8k_dir / "metadata" / "UrbanSound8K.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"UrbanSound8K metadata not found: {meta_file}")

    df = pd.read_csv(meta_file)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Organizing UrbanSound8K dataset to {output_dir}")

    # Group by class and copy files
    for class_name, group in tqdm(df.groupby("class"), desc="Organizing classes"):
        # Create class directory
        class_dir = output_dir / class_name.replace(
            " ", "_"
        )  # Replace spaces with underscores
        class_dir.mkdir(exist_ok=True)

        # Copy audio files
        for _, row in group.iterrows():
            src_file = (
                us8k_dir / "audio" / f"fold{row['fold']}" / row["slice_file_name"]
            )
            dst_file = class_dir / row["slice_file_name"]

            if src_file.exists():
                if not dst_file.exists():  # Don't overwrite existing files
                    shutil.copy2(src_file, dst_file)
            else:
                warnings.warn(f"Source file not found: {src_file}")

    # Print statistics
    class_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    total_files = sum(len(list(d.glob("*.wav"))) for d in class_dirs)

    print("✅ UrbanSound8K organized successfully!")
    print(f"   📁 Classes: {len(class_dirs)}")
    print(f"   🎵 Audio files: {total_files}")
    print(f"   📍 Location: {output_dir}")

    return output_dir


def load_urbansound8k(
    data_dir: Union[str, Path], auto_organize: bool = True
) -> pd.DataFrame:
    """
    Load UrbanSound8K dataset.

    Args:
        data_dir: Root directory containing UrbanSound8K
        auto_organize: If True, automatically organize if raw dataset found

    Returns:
        DataFrame for organized dataset ready for .from_directory()
    """
    data_dir = Path(data_dir)

    # Try to find organized dataset first
    organized_dir = data_dir / "UrbanSound8K_organized"
    if organized_dir.exists():
        return scan_directory_dataset(organized_dir)

    # Try to find raw UrbanSound8K dataset
    raw_dir = data_dir / "UrbanSound8K"
    if raw_dir.exists():
        csv_path = raw_dir / "metadata" / "UrbanSound8K.csv"
        if csv_path.exists() and auto_organize:
            # Organize the raw dataset
            organized_dir = organize_urbansound8k_for_training(raw_dir)
            return scan_directory_dataset(organized_dir)
        elif csv_path.exists():
            # Return the original CSV-based dataset
            df = pd.read_csv(csv_path)
            df["filename"] = df.apply(
                lambda x: f"fold{x['fold']}/{x['slice_file_name']}", axis=1
            )
            df["category"] = df["class"]
            return df[["filename", "category"]]

    # Dataset not found, provide download instructions
    download_urbansound8k(data_dir)


# Registry of preset loaders
PRESET_LOADERS = {
    "esc50": load_esc50,
    "urbansound8k": load_urbansound8k,
}

# Dataset download functions
DATASET_DOWNLOADERS = {
    "esc50": download_and_organize_esc50,
    "urbansound8k": download_urbansound8k,  # Manual download only
}


def load_split_directories(
    data_dir: Union[str, Path],
    train_dir: str = "train",
    val_dir: str = "val", 
    test_dir: str = "test",
    audio_extensions: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load pre-split dataset from train/val/test directories.
    
    Expected structure:
    data_dir/
    ├── train/
    │   ├── class1/
    │   │   ├── audio1.wav
    │   │   └── audio2.wav
    │   └── class2/
    │       └── audio3.wav
    ├── val/
    │   ├── class1/
    │   │   └── audio4.wav
    │   └── class2/
    │       └── audio5.wav
    └── test/
        ├── class1/
        │   └── audio6.wav
        └── class2/
            └── audio7.wav
            
    Args:
        data_dir: Root directory containing train/val/test splits
        train_dir: Name of training directory (default: "train")
        val_dir: Name of validation directory (default: "val") 
        test_dir: Name of test directory (default: "test")
        audio_extensions: List of valid audio extensions
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataFrames.
        Missing splits will have empty DataFrames.
    """
    if audio_extensions is None:
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a"]
        
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    splits = {}
    
    # Load each split directory
    for split_name, split_dir_name in [
        ("train", train_dir), 
        ("val", val_dir), 
        ("test", test_dir)
    ]:
        split_path = data_dir / split_dir_name
        
        if split_path.exists() and split_path.is_dir():
            try:
                # Scan the split directory
                df = scan_directory_dataset(split_path, audio_extensions)
                # Update file paths to include split directory
                df["filename"] = df["filename"].apply(
                    lambda x: f"{split_dir_name}/{x}"
                )
                splits[split_name] = df
                print(f"✓ Loaded {split_name} split: {len(df)} files")
            except (FileNotFoundError, ValueError) as e:
                print(f"⚠️  {split_name} split not found or empty: {e}")
                splits[split_name] = pd.DataFrame(columns=["filename", "category"])
        else:
            print(f"⚠️  {split_name} directory not found: {split_path}")
            splits[split_name] = pd.DataFrame(columns=["filename", "category"])
    
    # Validate that at least training split exists
    if splits["train"].empty:
        raise ValueError(f"Training split is required but not found in {data_dir / train_dir}")
    
    return splits


def load_split_csvs(
    data_dir: Union[str, Path],
    train_csv: Union[str, Path],
    val_csv: Union[str, Path] = None,
    test_csv: Union[str, Path] = None,
    audio_column: str = "filename",
    label_column: str = "category"
) -> Dict[str, pd.DataFrame]:
    """
    Load pre-split dataset from separate CSV files.
    
    Args:
        data_dir: Directory containing audio files
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file (optional)
        test_csv: Path to test CSV file (optional)
        audio_column: Name of column containing audio filenames
        label_column: Name of column containing labels
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataFrames.
        Missing splits will have empty DataFrames.
    """
    data_dir = Path(data_dir)
    splits = {}
    
    # Load training CSV (required)
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    train_df = pd.read_csv(train_csv)
    if audio_column not in train_df.columns or label_column not in train_df.columns:
        raise ValueError(f"Required columns [{audio_column}, {label_column}] not found in {train_csv}")
    
    splits["train"] = train_df[[audio_column, label_column]].rename(
        columns={audio_column: "filename", label_column: "category"}
    )
    print(f"✓ Loaded train split: {len(splits['train'])} files")
    
    # Load validation CSV (optional)
    if val_csv:
        val_csv = Path(val_csv)
        if val_csv.exists():
            val_df = pd.read_csv(val_csv)
            splits["val"] = val_df[[audio_column, label_column]].rename(
                columns={audio_column: "filename", label_column: "category"}
            )
            print(f"✓ Loaded val split: {len(splits['val'])} files")
        else:
            print(f"⚠️  Validation CSV not found: {val_csv}")
            splits["val"] = pd.DataFrame(columns=["filename", "category"])
    else:
        splits["val"] = pd.DataFrame(columns=["filename", "category"])
    
    # Load test CSV (optional)  
    if test_csv:
        test_csv = Path(test_csv)
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
            splits["test"] = test_df[[audio_column, label_column]].rename(
                columns={audio_column: "filename", label_column: "category"}
            )
            print(f"✓ Loaded test split: {len(splits['test'])} files")
        else:
            print(f"⚠️  Test CSV not found: {test_csv}")
            splits["test"] = pd.DataFrame(columns=["filename", "category"])
    else:
        splits["test"] = pd.DataFrame(columns=["filename", "category"])
    
    return splits
