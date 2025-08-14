"""Utilities for downloading and managing BEATs model checkpoints."""

import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil


# Default directories to search for checkpoints
CHECKPOINT_DIRS = [
    Path("checkpoints"),
    Path("./checkpoints"),
    Path("../checkpoints"),
    Path.home() / "checkpoints",
    Path.home() / ".cache" / "beats_trainer",
]

# Known BEATs model checkpoints with metadata
BEATS_MODELS = {
    "BEATs_iter3_plus_AS2M": {
        "url": "https://my.microsoftpersonalcontent.com/personal/6b83b49411ca81a7/_layouts/15/download.aspx?UniqueId=11ca81a7-b494-2083-806b-646500000000&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiJlNmU2YThhZi0yOWNmLTRhYjMtYmM2Zi05ODFmNTRlZmVhOWYiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvbXkubWljcm9zb2Z0cGVyc29uYWxjb250ZW50LmNvbUA5MTg4MDQwZC02YzY3LTRjNWItYjExMi0zNmEzMDRiNjZkYWQiLCJleHAiOiIxNzU1MDg0NTUyIn0.WAxNc-b3F6dQ_zEaeEUt_5FNtDvVWLuo64xH6j4xsElz9HiP2hoD4xdBtqJ-BBbdqIFTRRQi5zKtoUi41L-pQpGoZjmJmEE42hpfQuVFll81Oi9h63Rp0rXBYh30p4vI_VEglDSpdJNZSxKT7vya9QVwZZgCjcy-ju-BwR0IhelP7SZ7ALkfC6hfJr4_wVYGNN0gZCC6fdYYGoNWBgYSf_mA-aQDU-ZRvICruMhpkM2pu6nDuUuwZTQeOZB1wsIL8Dgpapqgimd44S-61sK9s6ByuFaKuyyvKw8TVi3khp2anUq2RXH3EBY9ihIVFkHK1ILkTpZa1d43OejEYebMIuKW7XnVSC4WDy-3FaqYqrZdkt55YB4wE1PhnCXd_IHBYD9lesX0Hpofsk3v_3_qEgzsKoopWm3-f6DL5ep16CmqC3yYIDEbAKY_PUVojEsthTaZyFR0LGur_wX3E7lvTrHXxCG1B83bZV6XMM6pA0g8xN3yVQdQt6oED4hOGbsw_3g5v0pwhRvwUAI4d2zqNA.-KyMwY6C11eJoSFIHtoAi_GuvmAJkL5t62Ij9MndxJs&ApiVersion=2.0",
        "filename": "BEATs_iter3_plus_AS2M.pt",
        "description": "BEATs model trained on AudioSet with 2M iterations",
        "size_mb": 110,  # Approximate size
    }
}


def download_file_with_progress(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file with progress indication."""
    print(f"Downloading {filepath.name}...")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as f:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(
                            f"\rProgress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)",
                            end="",
                        )
                    else:
                        print(
                            f"\rDownloaded: {downloaded / 1024 / 1024:.1f} MB", end=""
                        )

        print()  # New line after progress
        print(f"✓ Download complete: {filepath}")

    except Exception as e:
        if filepath.exists():
            filepath.unlink()  # Remove incomplete file
        raise Exception(f"Download failed: {e}")


def download_beats_checkpoint(
    model_name: str = "BEATs_iter3_plus_AS2M",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """
    Download a BEATs model checkpoint.

    Args:
        model_name: Name of the model to download
        cache_dir: Directory to store downloaded models (default: ./checkpoints)
        force_download: Whether to re-download if file already exists

    Returns:
        Path to the downloaded checkpoint file

    Raises:
        ValueError: If model_name is not recognized
        Exception: If download fails
    """

    # Validate model name
    if model_name not in BEATS_MODELS:
        available = list(BEATS_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    # Set up paths
    if cache_dir is None:
        cache_dir = Path.cwd() / "checkpoints"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    model_info = BEATS_MODELS[model_name]
    filepath = cache_dir / model_info["filename"]

    # Check if file already exists
    if filepath.exists() and not force_download:
        print(f"✓ Model already exists: {filepath}")
        return filepath

    # Download the model
    print(f"📥 Downloading {model_name}...")
    print(f"Description: {model_info['description']}")
    print(f"Expected size: ~{model_info['size_mb']} MB")
    print(f"Saving to: {filepath}")
    print()

    try:
        # Download to temporary file first, then move to final location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_path = Path(tmp_file.name)

        download_file_with_progress(model_info["url"], tmp_path)

        # Move to final location
        shutil.move(str(tmp_path), str(filepath))

        print(f"🎉 Successfully downloaded {model_name}!")
        return filepath

    except Exception as e:
        # Clean up temporary file if it exists
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()
        raise e


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available BEATs models.

    Returns:
        Dictionary mapping model names to their metadata
    """
    return BEATS_MODELS.copy()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model metadata

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in BEATS_MODELS:
        available = list(BEATS_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    return BEATS_MODELS[model_name].copy()


def find_checkpoint(
    model_name: str = None, search_paths: list = None
) -> Optional[Path]:
    """
    Find an existing BEATs checkpoint in common locations.

    Args:
        model_name: Specific model name to search for (optional)
        search_paths: List of paths to search (default: common checkpoint locations)

    Returns:
        Path to found checkpoint or None if not found
    """
    if search_paths is None:
        search_paths = CHECKPOINT_DIRS

    for search_path in search_paths:
        search_path = Path(search_path)  # Ensure it's a Path object
        if not search_path.exists():
            continue

        # If specific model name provided, look for that
        if model_name:
            filename = f"{model_name}.pt"
            filepath = search_path / filename
            if filepath.exists():
                return filepath

        # Look for specific filenames first
        for filename in [info["filename"] for info in BEATS_MODELS.values()]:
            filepath = search_path / filename
            if filepath.exists():
                return filepath

        # Then look for any .pt or .ckpt files
        for pattern in ["*.pt", "*.ckpt"]:
            matches = list(search_path.glob(pattern))
            if matches:
                return matches[0]  # Return first match

    return None


def ensure_checkpoint(
    model_name: str = "BEATs_iter3_plus_AS2M",
    cache_dir: Optional[str] = None,
    search_first: bool = True,
) -> Path:
    """
    Ensure a BEATs checkpoint is available, downloading if necessary.

    Args:
        model_name: Name of the model
        cache_dir: Directory to store downloaded models
        search_first: Whether to search for existing checkpoints first

    Returns:
        Path to the checkpoint file
    """

    # First try to find existing checkpoint
    if search_first:
        existing_path = find_checkpoint()
        if existing_path:
            print(f"✓ Found existing checkpoint: {existing_path}")
            return existing_path

    # Download if not found
    print(f"🔍 No existing checkpoint found, downloading {model_name}...")
    return download_beats_checkpoint(model_name, cache_dir)


def validate_checkpoint(checkpoint_path: Path) -> bool:
    """
    Validate that a checkpoint file is a valid PyTorch checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        True if the checkpoint is valid, False otherwise
    """
    try:
        # Check if file exists
        if not checkpoint_path.exists():
            return False

        # Try to load the checkpoint
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Basic validation - checkpoint should be a dictionary
        if not isinstance(checkpoint, dict):
            return False

        return True

    except Exception:
        # Any exception during loading means invalid checkpoint
        return False


if __name__ == "__main__":
    # CLI interface for downloading models
    import sys

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        print("Available BEATs models:")
        for name, info in list_available_models().items():
            print(f"  {name}: {info['description']}")
        print(f"\nUsage: python {sys.argv[0]} <model_name>")
        sys.exit(1)

    try:
        checkpoint_path = download_beats_checkpoint(model_name)
        print(f"\n✅ Model ready at: {checkpoint_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
