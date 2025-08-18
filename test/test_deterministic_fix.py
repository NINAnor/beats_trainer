#!/usr/bin/env python3
"""
Test script to verify the deterministic CUDA fix.
"""

import tempfile
import os
from pathlib import Path
from beats_trainer import BEATsTrainer
from beats_trainer.config import Config


def create_test_dataset():
    """Create a minimal test dataset."""
    import numpy as np
    import soundfile as sf

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    # Create two classes with minimal audio files
    for class_name in ["class1", "class2"]:
        class_dir = temp_dir / class_name
        class_dir.mkdir()

        # Create 2 short audio files per class
        for i in range(2):
            # Generate 1 second of random audio at 16kHz
            audio = np.random.randn(16000).astype(np.float32) * 0.1
            audio_path = class_dir / f"sample_{i}.wav"
            sf.write(str(audio_path), audio, 16000)

    return str(temp_dir)


def test_deterministic_training():
    """Test that training works with deterministic mode."""
    print("Creating test dataset...")
    data_dir = create_test_dataset()

    try:
        print("Testing BEATs trainer with deterministic mode...")

        # Create config with very short training for testing
        config = Config()
        config.training.max_epochs = 1
        config.training.deterministic = True
        config.data.batch_size = 1

        # Create trainer
        trainer = BEATsTrainer.from_directory(data_dir, config=config)

        print("Starting training (1 epoch)...")
        results = trainer.train()

        print("✅ Training completed successfully!")
        print(f"Best score: {results['best_score']:.4f}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

    finally:
        # Clean up
        import shutil

        shutil.rmtree(data_dir)
        print("Test dataset cleaned up")


if __name__ == "__main__":
    print("🧪 Testing deterministic CUDA fix...")

    # Check CUDA availability using a subprocess command
    cuda_check_cmd = 'python -c "import torch; print(torch.cuda.is_available())"'
    cuda_available = os.popen(cuda_check_cmd).read().strip()
    print(f"CUDA available: {cuda_available}")

    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "Not set")
    print(f"Current CUBLAS_WORKSPACE_CONFIG: {cublas_config}")

    success = test_deterministic_training()

    if success:
        print("\n🎉 Fix verified! The deterministic CUDA issue has been resolved.")
    else:
        print("\n⚠️  Test failed. Please check the error messages above.")
