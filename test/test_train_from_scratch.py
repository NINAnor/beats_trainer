"""
Test the training from scratch functionality for BEATs models.
"""

import torch
import tempfile
import os

# Import the module under test
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from beats_trainer.core.config import Config, ModelConfig, DataConfig, TrainingConfig
from beats_trainer.core.model import BEATsLightningModule


class TestTrainFromScratch:
    """Test cases for training BEATs from scratch functionality."""

    def test_config_from_scratch_setup(self):
        """Test that configuration properly handles train_from_scratch option."""
        # Test training from scratch configuration
        model_config = ModelConfig(
            train_from_scratch=True,
            encoder_layers=6,  # Smaller model for testing
            encoder_embed_dim=384,
            num_classes=10,
        )

        # Verify settings are correctly applied
        assert model_config.train_from_scratch is True
        assert model_config.fine_tune_backbone is True  # Should be auto-set
        assert model_config.freeze_backbone is False  # Should be auto-set
        assert model_config.model_path is None  # Should not set default path

        # Test pre-trained configuration
        model_config_pretrained = ModelConfig(train_from_scratch=False, num_classes=10)

        assert model_config_pretrained.train_from_scratch is False
        assert model_config_pretrained.model_path is not None  # Should set default path

    def test_model_initialization_from_scratch(self):
        """Test that model properly initializes from scratch."""
        # Create configuration for small model
        config = Config(
            data=DataConfig(batch_size=2, num_workers=0),
            model=ModelConfig(
                train_from_scratch=True,
                encoder_layers=2,  # Very small for testing
                encoder_embed_dim=128,
                encoder_ffn_embed_dim=256,
                encoder_attention_heads=4,
                num_classes=5,
            ),
            training=TrainingConfig(max_epochs=1),
        )

        # Initialize model
        model = BEATsLightningModule(config, num_classes=5)

        # Verify model structure
        assert model.backbone is not None
        assert model.classifier is not None
        assert model.num_classes == 5

        # Verify backbone is trainable (not frozen)
        # Note: Some layers like padding layers might be frozen for OpenBEATs compatibility
        backbone_params = list(model.backbone.parameters())
        assert len(backbone_params) > 0

        # Check that most parameters are trainable (allow some to be frozen like padding layers)
        trainable_params = [p for p in backbone_params if p.requires_grad]
        frozen_params = [p for p in backbone_params if not p.requires_grad]

        assert len(trainable_params) > len(frozen_params), (
            "Most backbone parameters should be trainable when training from scratch"
        )

        # Test forward pass
        batch_size = 2
        waveform_length = 16000  # 1 second at 16kHz

        # Create dummy waveform input (raw audio)
        x = torch.randn(batch_size, waveform_length)
        padding_mask = None  # BEATs will handle padding internally

        # Forward pass
        with torch.no_grad():
            logits = model.forward(x, padding_mask)

        assert logits.shape == (batch_size, 5)

    def test_model_weights_are_random(self):
        """Test that model weights are properly randomized when training from scratch."""
        config = Config(
            data=DataConfig(),
            model=ModelConfig(
                train_from_scratch=True,
                encoder_layers=2,
                encoder_embed_dim=768,  # Divisible by 12 heads
                encoder_attention_heads=12,
                num_classes=3,
            ),
        )

        # Create two models with same config
        model1 = BEATsLightningModule(config, num_classes=3)
        model2 = BEATsLightningModule(config, num_classes=3)

        # Get parameters from both models
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())

        # Check that weights are different (random initialization)
        weights_are_different = False
        for p1, p2 in zip(params1, params2):
            if not torch.allclose(p1, p2, atol=1e-6):
                weights_are_different = True
                break

        assert weights_are_different, "Models should have different random weights"

    def test_config_yaml_serialization(self):
        """Test that from-scratch config can be saved and loaded from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config
            config = Config(
                experiment_name="test_from_scratch",
                data=DataConfig(batch_size=8),
                model=ModelConfig(
                    train_from_scratch=True,
                    encoder_layers=4,
                    encoder_embed_dim=256,
                    num_classes=7,
                ),
                training=TrainingConfig(learning_rate=0.001, max_epochs=10),
            )

            # Save to YAML
            yaml_path = os.path.join(temp_dir, "test_config.yaml")
            config.to_yaml(yaml_path)

            # Load from YAML
            loaded_config = Config.from_yaml(yaml_path)

            # Verify key settings are preserved
            assert loaded_config.model.train_from_scratch is True
            assert loaded_config.model.encoder_layers == 4
            assert loaded_config.model.encoder_embed_dim == 256
            assert loaded_config.model.fine_tune_backbone is True
            assert loaded_config.model.freeze_backbone is False
            assert loaded_config.experiment_name == "test_from_scratch"

    def test_classifier_head_dimensions(self):
        """Test that classifier head has correct dimensions for from-scratch models."""
        # Test simple classifier
        config = Config(
            model=ModelConfig(
                train_from_scratch=True,
                encoder_embed_dim=768,  # Divisible by 12 heads
                encoder_attention_heads=12,
                use_custom_head=False,
                num_classes=20,
            )
        )

        model = BEATsLightningModule(config, num_classes=20)

        # Check classifier dimensions
        assert isinstance(model.classifier, torch.nn.Linear)
        assert model.classifier.in_features == 768
        assert model.classifier.out_features == 20

        # Test custom multi-layer head
        config_custom = Config(
            model=ModelConfig(
                train_from_scratch=True,
                encoder_embed_dim=768,  # Divisible by 12 heads
                encoder_attention_heads=12,
                use_custom_head=True,
                hidden_dims=[256, 128],
                num_classes=20,
            )
        )

        model_custom = BEATsLightningModule(config_custom, num_classes=20)

        # Check that custom head is a Sequential module
        assert isinstance(model_custom.classifier, torch.nn.Sequential)

        # Test forward pass through custom head
        features = torch.randn(4, 768)  # batch_size=4, embed_dim=768
        with torch.no_grad():
            logits = model_custom.classifier(features)
        assert logits.shape == (4, 20)


def test_integration_example():
    """Integration test showing complete workflow."""
    print("\n🧪 Running integration test for training from scratch...")

    # Create minimal config
    config = Config(
        experiment_name="integration_test",
        data=DataConfig(batch_size=2, num_workers=0, sample_rate=16000),
        model=ModelConfig(
            train_from_scratch=True,
            encoder_layers=2,  # Minimal size for testing
            encoder_embed_dim=128,
            encoder_ffn_embed_dim=256,
            encoder_attention_heads=4,
            input_patch_size=16,
            embed_dim=64,
            num_classes=3,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            max_epochs=1,  # Just test one epoch
            patience=1,
        ),
    )

    # Initialize model
    model = BEATsLightningModule(config, num_classes=3)

    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )
    print(
        f"✅ Training mode: {'From scratch' if config.model.train_from_scratch else 'Pre-trained'}"
    )
    print(
        f"✅ Backbone trainable: {any(p.requires_grad for p in model.backbone.parameters())}"
    )

    # Test forward pass
    batch_size = 2
    waveform_length = 16000  # 1 second of audio at 16kHz

    x = torch.randn(batch_size, waveform_length)  # Raw waveform
    padding_mask = None  # BEATs handles padding internally
    targets = torch.randint(0, 3, (batch_size,))

    # Forward pass
    logits = model.forward(x, padding_mask)
    loss = model.criterion(logits, targets)

    print(
        f"✅ Forward pass successful: logits shape {logits.shape}, loss {loss.item():.4f}"
    )

    # Test training step
    batch = (x, padding_mask, targets)
    train_loss = model.training_step(batch, 0)

    print(f"✅ Training step successful: loss {train_loss.item():.4f}")
    print("🎉 Integration test passed!")


if __name__ == "__main__":
    test_integration_example()
