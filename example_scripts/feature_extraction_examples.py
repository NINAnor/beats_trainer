"""Example usage of BEATs feature extractor."""

import numpy as np
import os
from pathlib import Path

# Add the parent directory to the path to import beats_trainer
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from beats_trainer import BEATsFeatureExtractor, BEATsTrainer


def example_standalone_feature_extraction():
    """Example of using BEATsFeatureExtractor standalone."""
    print("=== Standalone Feature Extraction ===")

    # Path to pretrained BEATs model
    model_path = "checkpoints/BEATs_iter3_plus_AS2M.pt"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please download the BEATs model checkpoint first")
        return

    # Create feature extractor
    extractor = BEATsFeatureExtractor(
        model_path=model_path,
        pooling="mean",  # Options: "mean", "max", "first", "last", "none"
    )

    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print(f"Model info: {extractor.get_model_info()}")

    # Check if we have audio files to test
    if os.path.exists("notebooks/datasets/ESC-50-master/audio"):
        audio_dir = Path("notebooks/datasets/ESC-50-master/audio")
        audio_files = list(audio_dir.glob("*.wav"))[:5]  # Test with first 5 files

        if audio_files:
            print(f"\nExtracting features from {len(audio_files)} files...")

            # Extract from single file
            single_features = extractor.extract_from_file(audio_files[0])
            print(f"Single file features shape: {single_features.shape}")

            # Extract from multiple files
            batch_features = extractor.extract_from_files(audio_files, batch_size=2)
            print(f"Batch features shape: {batch_features.shape}")

            # You can now use these features for downstream tasks
            # e.g., clustering, similarity search, etc.

        else:
            print("No audio files found for testing")
    else:
        print(
            "ESC-50 dataset not found. Please run the notebook first to download data."
        )


def example_trainer_feature_extraction():
    """Example of using feature extraction through BEATsTrainer."""
    print("\n=== Feature Extraction via Trainer ===")

    # Check if we have the organized dataset
    if os.path.exists("notebooks/datasets/ESC50_organized"):
        try:
            # Create trainer (this will use the pretrained model)
            trainer = BEATsTrainer.from_directory("notebooks/datasets/ESC50_organized")

            # Get some sample files
            sample_files = []
            for class_dir in Path("notebooks/datasets/ESC50_organized").iterdir():
                if class_dir.is_dir():
                    audio_files = list(class_dir.glob("*.wav"))
                    if audio_files:
                        sample_files.extend(audio_files[:2])  # 2 files per class
                if len(sample_files) >= 10:  # Limit to 10 files total
                    break

            if sample_files:
                print(
                    f"Extracting features from {len(sample_files)} files via trainer..."
                )

                # Extract features using trainer
                features = trainer.extract_features(
                    sample_files, pooling="mean", normalize=True
                )

                print(f"Extracted features shape: {features.shape}")

                # Get standalone extractor from trainer
                extractor = trainer.get_feature_extractor(pooling="max")
                print(f"Got standalone extractor with {extractor.pooling} pooling")

            else:
                print("No sample files found")

        except Exception as e:
            print(f"Error creating trainer: {e}")
            print("Make sure the dataset is properly organized")
    else:
        print("Organized dataset not found. Please run the notebook first.")


def example_numpy_array_extraction():
    """Example of extracting features from numpy arrays."""
    print("\n=== Feature Extraction from NumPy Arrays ===")

    model_path = "checkpoints/BEATs_iter3_plus_AS2M.pt"

    if not os.path.exists(model_path):
        print("Model checkpoint not found")
        return

    try:
        extractor = BEATsFeatureExtractor(model_path)

        # Create dummy audio data (normally you'd load real audio)
        dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
        print(f"Dummy audio shape: {dummy_audio.shape}")

        # Extract features
        features = extractor.extract_from_array(dummy_audio, sample_rate=16000)
        print(f"Extracted features shape: {features.shape}")

        # Batch processing with numpy arrays
        batch_audio = np.random.randn(3, 16000)  # 3 samples
        batch_features = extractor.extract_from_array(batch_audio)
        print(f"Batch features shape: {batch_features.shape}")

    except Exception as e:
        print(f"Error in numpy extraction: {e}")


def example_different_pooling_methods():
    """Example showing different pooling methods."""
    print("\n=== Different Pooling Methods ===")

    model_path = "checkpoints/BEATs_iter3_plus_AS2M.pt"

    if not os.path.exists(model_path):
        print("Model checkpoint not found")
        return

    # Test different pooling methods
    pooling_methods = ["mean", "max", "first", "last"]

    # Create dummy audio
    dummy_audio = np.random.randn(16000)

    for pooling in pooling_methods:
        try:
            extractor = BEATsFeatureExtractor(model_path, pooling=pooling)
            features = extractor.extract_from_array(dummy_audio)
            print(f"Pooling '{pooling}': shape {features.shape}")
        except Exception as e:
            print(f"Error with pooling '{pooling}': {e}")


if __name__ == "__main__":
    print("BEATs Feature Extraction Examples")
    print("=" * 40)

    try:
        example_standalone_feature_extraction()
        example_trainer_feature_extraction()
        example_numpy_array_extraction()
        example_different_pooling_methods()

        print("\n=== Summary ===")
        print("The BEATs feature extractor provides several ways to extract features:")
        print("1. Standalone BEATsFeatureExtractor class")
        print("2. Through BEATsTrainer.extract_features() method")
        print("3. From files, numpy arrays, or batches")
        print("4. Different pooling strategies for sequence features")
        print("5. Option to normalize features")
        print("\nThese features can be used for:")
        print("- Similarity search and retrieval")
        print("- Clustering audio samples")
        print("- Training downstream classifiers")
        print("- Audio recommendation systems")
        print("- Unsupervised analysis and visualization")

    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
