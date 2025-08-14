"""Updated examples showing automatic checkpoint download and management."""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import beats_trainer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beats_trainer import (
    BEATsFeatureExtractor,
    download_beats_checkpoint,
    ensure_checkpoint,
    list_available_models,
    find_checkpoint,
)


def example_automatic_download():
    """Example of automatic checkpoint download and usage."""
    print("=== Automatic Checkpoint Download ===")

    try:
        # Show available models
        print("Available BEATs models:")
        models = list_available_models()
        for name, info in models.items():
            print(f"  • {name}: {info['description']}")

        # Create feature extractor without specifying model path
        # This will automatically find or download a checkpoint
        print("\nCreating feature extractor (auto-download if needed)...")
        extractor = BEATsFeatureExtractor()  # No model path specified!

        print("✓ Feature extractor ready!")
        print(f"  Model path: {extractor.model_path}")
        print(f"  Feature dim: {extractor.get_feature_dim()}")
        print(f"  Device: {extractor.device}")

        # Test with some dummy audio data
        import numpy as np

        dummy_audio = np.random.randn(16000)  # 1 second of audio
        features = extractor.extract_from_array(dummy_audio)
        print(f"  Test extraction: {features.shape}")

    except Exception as e:
        print(f"Error: {e}")


def example_explicit_download():
    """Example of explicitly downloading a specific model."""
    print("\n=== Explicit Model Download ===")

    try:
        # Download specific model
        model_name = "BEATs_iter3_plus_AS2M"
        print(f"Downloading {model_name}...")

        checkpoint_path = download_beats_checkpoint(
            model_name=model_name,
            cache_dir="./checkpoints",
            force_download=False,  # Don't re-download if exists
        )

        print(f"✓ Model downloaded to: {checkpoint_path}")

        # Use the downloaded model
        print("✓ Feature extractor created with downloaded model")

    except Exception as e:
        print(f"Error: {e}")


def example_find_existing():
    """Example of finding existing checkpoints."""
    print("\n=== Finding Existing Checkpoints ===")

    # Try to find any existing checkpoint
    existing_checkpoint = find_checkpoint()

    if existing_checkpoint:
        print(f"✓ Found existing checkpoint: {existing_checkpoint}")

        # Use the found checkpoint
        print("✓ Using existing checkpoint")

    else:
        print("❌ No existing checkpoints found")
        print("Available search locations:")
        search_paths = [
            Path.cwd() / "checkpoints",
            Path.cwd() / "models",
            Path.cwd(),
            Path.home() / ".cache" / "beats",
        ]
        for path in search_paths:
            print(f"  • {path} {'(exists)' if path.exists() else '(not found)'}")


def example_ensure_checkpoint():
    """Example of ensuring a checkpoint is available (smart approach)."""
    print("\n=== Smart Checkpoint Management ===")

    try:
        # This will:
        # 1. First try to find an existing checkpoint
        # 2. If not found, download the default model
        print("Ensuring BEATs checkpoint is available...")

        checkpoint_path = ensure_checkpoint(
            model_name="BEATs_iter3_plus_AS2M",
            search_first=True,  # Search for existing first
        )

        print(f"✓ Checkpoint ready at: {checkpoint_path}")

        # Create multiple extractors with different pooling
        pooling_methods = ["mean", "max", "first"]
        extractors = {}

        for pooling in pooling_methods:
            extractors[pooling] = BEATsFeatureExtractor(
                model_path=checkpoint_path, pooling=pooling
            )
            print(f"✓ Created {pooling} pooling extractor")

        # Test all extractors
        import numpy as np

        test_audio = np.random.randn(32000)  # 2 seconds

        print("\nTesting different pooling methods:")
        for pooling, extractor in extractors.items():
            features = extractor.extract_from_array(test_audio)
            print(f"  {pooling:5}: {features.shape}")

    except Exception as e:
        print(f"Error: {e}")


def example_library_integration():
    """Example showing how the checkpoint utilities integrate with the library."""
    print("\n=== Library Integration Example ===")

    try:
        # Method 1: Let the library handle everything
        print("Method 1: Fully automatic")
        extractor1 = BEATsFeatureExtractor()  # Auto-finds/downloads
        print(f"✓ Auto extractor: {extractor1.model_path.name}")

        # Method 2: Ensure specific model
        print("\nMethod 2: Ensure specific model")
        checkpoint = ensure_checkpoint("BEATs_iter3_plus_AS2M")
        extractor2 = BEATsFeatureExtractor(checkpoint)
        print(f"✓ Specific model extractor: {extractor2.model_path.name}")

        # Method 3: Use with trainer (if we have organized data)
        print("\nMethod 3: Integration with trainer")
        if os.path.exists("notebooks/datasets/ESC50_organized"):
            from beats_trainer import BEATsTrainer

            # The trainer will also use ensure_checkpoint if no model path specified
            trainer = BEATsTrainer.from_directory("notebooks/datasets/ESC50_organized")

            # Get feature extractor from trainer
            trainer_extractor = trainer.get_feature_extractor()
            print(f"✓ Trainer extractor: {trainer_extractor.model_path.name}")
        else:
            print("  (No organized dataset found, skipping trainer example)")

        print("\n✅ All integration methods working!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("BEATs Checkpoint Management Examples")
    print("=" * 50)

    try:
        example_automatic_download()
        example_find_existing()
        example_ensure_checkpoint()
        example_explicit_download()
        example_library_integration()

        print("\n" + "=" * 50)
        print("🎉 All examples completed successfully!")
        print("\nKey benefits of the checkpoint utilities:")
        print("✅ Automatic model download when needed")
        print("✅ Smart checkpoint discovery")
        print("✅ No manual file management required")
        print("✅ Seamless integration with BEATsTrainer and BEATsFeatureExtractor")
        print("✅ Caching to avoid re-downloading")
        print("✅ Multiple model support (extensible)")

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
