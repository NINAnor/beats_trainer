#!/usr/bin/env python3
"""
Demo script showing the difference between pre-trained and from-scratch training.
This script creates example configurations for both approaches.
"""

import sys
import os

# Add the source directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from beats_trainer.config import Config, ModelConfig, DataConfig, TrainingConfig


def create_pretrained_config():
    """Create a configuration for fine-tuning a pre-trained BEATs model."""
    return Config(
        experiment_name="beats_pretrained_finetuning",
        data=DataConfig(
            data_dir="path/to/your/audio/data",
            batch_size=32,
            num_workers=4,
            sample_rate=16000,
        ),
        model=ModelConfig(
            # Pre-trained model settings (default behavior)
            train_from_scratch=False,  # Use pre-trained weights
            model_path=None,  # Will auto-download default model
            # Training strategy for fine-tuning
            freeze_backbone=True,  # Freeze pre-trained features
            fine_tune_backbone=False,  # Only train classifier head
            # Classification head
            use_custom_head=False,
            dropout_rate=0.1,
        ),
        training=TrainingConfig(
            learning_rate=1e-4,  # Lower LR for fine-tuning
            weight_decay=0.01,
            max_epochs=20,  # Fewer epochs needed
            patience=5,
            optimizer="adamw",
            scheduler="cosine",
        ),
    )


def create_from_scratch_config():
    """Create a configuration for training BEATs from scratch."""
    return Config(
        experiment_name="beats_from_scratch_training",
        data=DataConfig(
            data_dir="path/to/your/audio/data",
            batch_size=16,  # Smaller batch size for stability
            num_workers=4,
            sample_rate=16000,
        ),
        model=ModelConfig(
            # From-scratch model settings
            train_from_scratch=True,  # No pre-trained weights
            # Model architecture (fully customizable)
            encoder_layers=12,  # Standard BEATs architecture
            encoder_embed_dim=768,
            encoder_ffn_embed_dim=3072,
            encoder_attention_heads=12,
            input_patch_size=16,
            embed_dim=512,
            # Training strategy (automatically set for from-scratch)
            freeze_backbone=False,  # Train entire model
            fine_tune_backbone=True,  # Train backbone + classifier
            # Classification head
            use_custom_head=True,
            hidden_dims=[512, 256],  # Multi-layer head for better learning
            dropout_rate=0.1,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,  # Higher LR for from-scratch training
            weight_decay=0.01,
            max_epochs=100,  # More epochs needed
            patience=15,  # More patience for convergence
            optimizer="adamw",
            scheduler="cosine",
        ),
    )


def compare_configs():
    """Compare the two training approaches."""
    pretrained_config = create_pretrained_config()
    scratch_config = create_from_scratch_config()

    print("🎵 BEATs Training Approaches Comparison")
    print("=" * 60)

    print("\n📦 PRE-TRAINED FINE-TUNING:")
    print(
        f"  • Uses pre-trained weights: {not pretrained_config.model.train_from_scratch}"
    )
    print(
        f"  • Model path required: {pretrained_config.model.model_path is not None or 'auto-download'}"
    )
    print(f"  • Freeze backbone: {pretrained_config.model.freeze_backbone}")
    print(f"  • Learning rate: {pretrained_config.training.learning_rate}")
    print(f"  • Max epochs: {pretrained_config.training.max_epochs}")
    print("  • Best for: Transfer learning, limited data, quick training")

    print("\n🏗️ FROM-SCRATCH TRAINING:")
    print(
        f"  • Uses pre-trained weights: {not scratch_config.model.train_from_scratch}"
    )
    print("  • Model path required: No")
    print(f"  • Freeze backbone: {scratch_config.model.freeze_backbone}")
    print(f"  • Learning rate: {scratch_config.training.learning_rate}")
    print(f"  • Max epochs: {scratch_config.training.max_epochs}")
    print("  • Best for: Domain-specific data, large datasets, custom architectures")

    # Save configurations
    pretrained_config.to_yaml("config_pretrained_example.yaml")
    scratch_config.to_yaml("config_from_scratch_example.yaml")

    print("\n✅ Configuration files saved:")
    print("  • config_pretrained_example.yaml")
    print("  • config_from_scratch_example.yaml")

    print("\n💡 Usage:")
    print("""
# For pre-trained fine-tuning:
from beats_trainer import BEATsTrainer
from beats_trainer.config import Config

config = Config.from_yaml("config_pretrained_example.yaml")
trainer = BEATsTrainer(config)
results = trainer.train()

# For from-scratch training:
config = Config.from_yaml("config_from_scratch_example.yaml")
trainer = BEATsTrainer(config)
results = trainer.train()
""")

    print("\n🔍 When to choose each approach:")
    print("""
PRE-TRAINED FINE-TUNING:
✅ Small to medium datasets (< 10k samples)
✅ General audio classification tasks
✅ Quick experimentation and prototyping
✅ Limited computational resources
✅ Similar domain to pre-training data

FROM-SCRATCH TRAINING:
✅ Large datasets (> 50k samples)
✅ Highly domain-specific audio (medical, industrial, etc.)
✅ Custom model architectures needed
✅ Sufficient computational resources
✅ Maximum performance potential
""")


if __name__ == "__main__":
    compare_configs()
