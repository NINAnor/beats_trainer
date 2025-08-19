#!/usr/bin/env python3
"""
Example script showing how to train BEATs from scratch without pre-trained weights.
This is useful when you want to train the model completely from random initialization.
"""

import sys
import os

# Add the source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from beats_trainer.config import Config, ModelConfig, DataConfig, TrainingConfig


def create_scratch_training_config():
    """Create a configuration for training BEATs from scratch."""

    # Data configuration
    data_config = DataConfig(
        data_dir="path/to/your/audio/data",
        batch_size=16,  # Smaller batch size for training from scratch
        num_workers=4,
        sample_rate=16000,
    )

    # Model configuration for training from scratch
    model_config = ModelConfig(
        train_from_scratch=True,  # Key parameter: enables training from scratch
        # Model architecture (you can customize these)
        encoder_layers=12,  # Number of transformer layers
        encoder_embed_dim=768,  # Hidden dimension
        encoder_ffn_embed_dim=3072,  # Feed-forward dimension
        encoder_attention_heads=12,  # Number of attention heads
        input_patch_size=16,  # Patch size for audio tokenization
        embed_dim=512,  # Patch embedding dimension
        # Training strategy (automatically set when train_from_scratch=True)
        fine_tune_backbone=True,  # Will train the entire backbone
        freeze_backbone=False,  # Won't freeze any layers
        # Classification head
        use_custom_head=False,  # Use simple linear classifier
        dropout_rate=0.1,
    )

    # Training configuration (more aggressive for training from scratch)
    training_config = TrainingConfig(
        learning_rate=1e-3,  # Higher learning rate for training from scratch
        weight_decay=0.01,
        max_epochs=100,  # More epochs needed for training from scratch
        patience=15,  # More patience for convergence
        optimizer="adamw",
        scheduler="cosine",
    )

    # Combined configuration
    config = Config(
        experiment_name="beats_from_scratch",
        data=data_config,
        model=model_config,
        training=training_config,
        seed=42,
    )

    return config


def main():
    """Demonstrate how to create and use a from-scratch configuration."""

    print("🎵 BEATs Training from Scratch Example")
    print("=" * 50)

    # Create configuration
    config = create_scratch_training_config()

    print("Configuration created:")
    print(f"  Train from scratch: {config.model.train_from_scratch}")
    print("  Model architecture:")
    print(f"    - Layers: {config.model.encoder_layers}")
    print(f"    - Embed dim: {config.model.encoder_embed_dim}")
    print(f"    - Attention heads: {config.model.encoder_attention_heads}")
    print(f"    - Patch size: {config.model.input_patch_size}")
    print("  Training settings:")
    print(f"    - Learning rate: {config.training.learning_rate}")
    print(f"    - Max epochs: {config.training.max_epochs}")
    print(f"    - Fine-tune backbone: {config.model.fine_tune_backbone}")
    print(f"    - Freeze backbone: {config.model.freeze_backbone}")

    # Save configuration to file
    config_path = "config_train_from_scratch.yaml"
    config.to_yaml(config_path)
    print(f"\n✅ Configuration saved to: {config_path}")

    print("\n📝 Usage:")
    print("""
To use this configuration for training:

from beats_trainer import BEATsTrainer
from beats_trainer.config import Config

# Load the configuration
config = Config.from_yaml("config_train_from_scratch.yaml")

# Create trainer
trainer = BEATsTrainer(config)

# Train the model (will initialize from scratch, no pre-trained weights)
trainer.train()
""")

    print("\n💡 Key Differences from Pre-trained Training:")
    print("  • No checkpoint file is loaded")
    print("  • Model weights are randomly initialized")
    print("  • Entire backbone is trained (not frozen)")
    print("  • Typically requires more epochs and data")
    print("  • Higher learning rates are often beneficial")
    print("  • More aggressive data augmentation may help")


if __name__ == "__main__":
    main()
