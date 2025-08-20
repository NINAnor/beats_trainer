#!/usr/bin/env python3
"""
Example: Download and train on ESC-50 dataset

This example demonstrates how to:
1. Automatically download and organize ESC-50 dataset
2. Train a BEATs model with backbone fine-tuning
3. Achieve ~95% accuracy as demonstrated in the notebook

Run with:
    python examples/esc50_download_and_train.py
"""

from beats_trainer import BEATsTrainer
from beats_trainer.config import Config


def main():
    """Download ESC-50 and train BEATs model."""

    print("🎵 ESC-50 Download and Training Example")
    print("=" * 50)

    # Create optimized configuration for ESC-50
    config = Config()

    # Model configuration - use standard implementation (not ESPnet)
    config.model.model_path = (
        "checkpoints/openbeats.pt"  # Will auto-download if not found
    )
    config.model.freeze_backbone = (
        False  # Fine-tune entire model (key for 95% performance)
    )
    config.model.fine_tune_backbone = True

    # Training parameters optimized for ESC-50
    config.training.max_epochs = 50
    config.training.batch_size = 32
    config.training.learning_rate = 5e-5
    config.training.weight_decay = 0.01
    config.training.scheduler = "cosine"
    config.training.patience = 15
    config.training.val_check_interval = 0.25

    print("📋 Configuration:")
    print("  🧠 Model: Standard OpenBEATs")
    print("  🔓 Backbone: Fine-tuning enabled")
    print(f"  📈 Epochs: {config.training.max_epochs}")
    print(f"  📦 Batch size: {config.training.batch_size}")
    print(f"  🎚️ Learning rate: {config.training.learning_rate}")

    # Create trainer with auto-download ESC-50
    print("\n📥 Setting up ESC-50 dataset...")
    trainer = BEATsTrainer.from_esc50(
        data_dir="./datasets",  # Will create this directory
        config=config,
        auto_download=True,  # Download if not found
        force_download=False,  # Don't re-download if exists
    )

    # Train the model
    trainer.train()

    print("\n✅ Training completed!")
    print("Check the logs directory for detailed metrics and model checkpoints.")

    # Get the best model path
    checkpoint_path = trainer.get_best_checkpoint()
    if checkpoint_path:
        print(f"\n💾 Best model saved at: {checkpoint_path}")
        print("You can now use this model for feature extraction or inference.")


if __name__ == "__main__":
    main()
