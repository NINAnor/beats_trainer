"""Utilities for downloading and managing BEATs model checkpoints."""

import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


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
        "hf_repo": "Bencr/beats-checkpoints",
        "hf_filename": "BEATs_iter3_plus_AS2M.pt",
        "filename": "BEATs_iter3_plus_AS2M.pt",
        "description": "BEATs model trained on AudioSet with 2M iterations",
        "size_mb": 110,  # Approximate size
    },
    "openbeats": {
        "hf_repo": "Bencr/beats-checkpoints",
        "hf_filename": "openbeats.pt",
        "filename": "openbeats.pt",
        "description": "OpenBEATs checkpoint",
        "size_mb": 110,  # Approximate size
    },
}


def download_from_huggingface(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Download a file from Hugging Face Hub."""
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for downloading checkpoints. "
            "Install with: pip install huggingface-hub"
        )

    try:
        print(f"Downloading {filename} from {repo_id}...")

        # Use HF Hub to download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            force_download=force_download,
        )

        print(f"✓ Download complete: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        raise Exception(f"Download from Hugging Face Hub failed: {e}")


def download_file_with_progress(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file with progress indication (fallback method)."""
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
        # Try downloading from Hugging Face Hub first
        if "hf_repo" in model_info and "hf_filename" in model_info:
            downloaded_path = download_from_huggingface(
                repo_id=model_info["hf_repo"],
                filename=model_info["hf_filename"],
                cache_dir=str(cache_dir),
                force_download=force_download,
            )

            # Copy to expected location if different
            if downloaded_path != filepath:
                shutil.copy2(downloaded_path, filepath)

        else:
            # Fallback to direct URL download (if url is available)
            if "url" in model_info:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".tmp"
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)

                download_file_with_progress(model_info["url"], tmp_path)
                shutil.move(str(tmp_path), str(filepath))
            else:
                raise ValueError(f"No download method available for {model_name}")

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
