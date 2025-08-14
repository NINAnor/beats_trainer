"""
Test suite for BEATs model functionality.
"""

import pytest
import torch

from .conftest import (
    TestConfig,
    skip_if_no_model,
    skip_if_no_gpu,
    AudioTestCase,
    create_synthetic_audio,
)


class TestBEATsModel(AudioTestCase):
    """Test cases for BEATs model functionality."""

    def setUp(self):
        super().setUp()

    @skip_if_no_model()
    def test_model_loading(self):
        """Test BEATs model can be loaded successfully."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        assert checkpoint_path is not None, "Checkpoint should be available"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model config (using defaults)
        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )

        # Initialize model
        model = BEATs(cfg)

        # Load state dict
        model.load_state_dict(checkpoint)
        model.eval()

        assert model is not None
        assert len(list(model.parameters())) > 0

    @skip_if_no_model()
    def test_model_inference(self):
        """Test BEATs model inference."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model
        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg)
        model.load_state_dict(checkpoint)
        model.eval()

        # Create synthetic audio
        audio = create_synthetic_audio()

        # Convert to tensor
        audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            # Forward pass
            output = model(audio_tensor)

            assert output is not None
            assert isinstance(output, torch.Tensor)
            assert output.shape[0] == 1  # Batch size
            assert output.shape[1] > 0  # Feature dimension

    @skip_if_no_model()
    def test_model_different_input_lengths(self):
        """Test model with different input lengths."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg)
        model.load_state_dict(checkpoint)
        model.eval()

        # Test different length inputs
        lengths = [
            TestConfig.SAMPLE_RATE * 1,  # 1 second
            TestConfig.SAMPLE_RATE * 2,  # 2 seconds
            TestConfig.SAMPLE_RATE * 5,
        ]  # 5 seconds

        outputs = []

        with torch.no_grad():
            for length in lengths:
                audio = create_synthetic_audio(duration=length / TestConfig.SAMPLE_RATE)
                audio_tensor = torch.tensor(audio).unsqueeze(0)

                output = model(audio_tensor)
                outputs.append(output)

                # Check output properties
                assert output.shape[0] == 1
                assert output.shape[1] == cfg.embed_dim or output.shape[1] > 0

        # All outputs should have same feature dimension
        feature_dims = [output.shape[1] for output in outputs]
        assert len(set(feature_dims)) == 1, (
            "All outputs should have same feature dimension"
        )

    @skip_if_no_model()
    def test_model_batch_processing(self):
        """Test model with batch processing."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg)
        model.load_state_dict(checkpoint)
        model.eval()

        # Create batch of audio
        batch_size = 3
        audio_batch = []

        for i in range(batch_size):
            audio = create_synthetic_audio(seed=42 + i)
            audio_batch.append(audio)

        # Convert to tensor (assuming equal length)
        audio_tensor = torch.stack([torch.tensor(audio) for audio in audio_batch])

        with torch.no_grad():
            output = model(audio_tensor)

            assert output.shape[0] == batch_size
            assert output.shape[1] > 0

    @skip_if_no_gpu()
    @skip_if_no_model()
    def test_model_gpu_inference(self):
        """Test model inference on GPU."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg).cuda()
        model.load_state_dict(checkpoint)
        model.eval()

        # Create synthetic audio
        audio = create_synthetic_audio()
        audio_tensor = torch.tensor(audio).cuda().unsqueeze(0)

        with torch.no_grad():
            output = model(audio_tensor)

            assert output.device.type == "cuda"
            assert output.shape[0] == 1
            assert output.shape[1] > 0

    @skip_if_no_model()
    def test_model_deterministic_output(self):
        """Test that model produces deterministic outputs."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg)
        model.load_state_dict(checkpoint)
        model.eval()

        # Create synthetic audio
        audio = create_synthetic_audio()
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        # Run inference multiple times
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = model(audio_tensor)
                outputs.append(output.clone())

        # All outputs should be identical
        for i in range(1, len(outputs)):
            torch.testing.assert_close(outputs[0], outputs[i], rtol=1e-6, atol=1e-6)

    @skip_if_no_model()
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg)
        model.load_state_dict(checkpoint)
        model.train()  # Set to training mode

        # Create synthetic audio with gradient tracking
        audio = create_synthetic_audio()
        audio_tensor = torch.tensor(audio, requires_grad=True).unsqueeze(0)

        # Forward pass
        output = model(audio_tensor)

        # Create dummy loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert audio_tensor.grad is not None
        assert torch.any(audio_tensor.grad != 0), "Gradients should be non-zero"

        # Check that model parameters have gradients
        param_has_grad = any(param.grad is not None for param in model.parameters())
        assert param_has_grad, "Some model parameters should have gradients"


class TestBEATsConfig:
    """Test cases for BEATs configuration."""

    def test_config_initialization(self):
        """Test BEATsConfig initialization."""
        from BEATs import BEATsConfig

        # Test default config
        cfg = BEATsConfig()
        assert hasattr(cfg, "input_size")
        assert hasattr(cfg, "embed_dim")

        # Test custom config
        custom_cfg = BEATsConfig(
            {"input_size": 512, "embed_dim": 384, "conv_bias": False}
        )

        assert custom_cfg.input_size == 512
        assert custom_cfg.embed_dim == 384
        assert custom_cfg.conv_bias is False

    def test_config_validation(self):
        """Test BEATsConfig parameter validation."""
        from BEATs import BEATsConfig

        # Test valid configurations
        valid_configs = [
            {"input_size": 512, "embed_dim": 384},
            {"input_size": 1024, "embed_dim": 768},
            {"input_size": 2048, "embed_dim": 1024},
        ]

        for config_dict in valid_configs:
            cfg = BEATsConfig(config_dict)
            assert cfg.input_size == config_dict["input_size"]
            assert cfg.embed_dim == config_dict["embed_dim"]


class TestModelIntegration:
    """Integration tests for the complete model pipeline."""

    @skip_if_no_model()
    def test_model_feature_extractor_integration(self):
        """Test integration between raw model and feature extractor."""
        from beats_trainer.feature_extractor import BEATsFeatureExtractor
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        # Test both approaches produce similar results
        checkpoint_path = find_checkpoint()

        # Method 1: Direct model usage
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model = BEATs(cfg)
        model.load_state_dict(checkpoint)
        model.eval()

        # Method 2: Feature extractor
        extractor = BEATsFeatureExtractor(
            model_path=checkpoint_path, pooling="mean", device="cpu"
        )

        # Create test audio
        audio = create_synthetic_audio()

        # Extract features using both methods
        with torch.no_grad():
            # Direct model
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            model_output = model(audio_tensor)

            # Feature extractor
            extractor_output = extractor.extract_features(audio)

        # Both should produce valid outputs
        assert model_output.shape[0] == 1
        assert extractor_output.shape[0] == 1

        # Feature dimensions might differ due to pooling
        assert model_output.shape[1] > 0
        assert extractor_output.shape[1] > 0

    @skip_if_no_model()
    def test_model_consistency_across_devices(self):
        """Test model produces consistent results across different devices."""
        from beats_trainer.checkpoint_utils import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()

        # Test on CPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(
            {
                "input_size": 1024,
                "embed_dim": 768,
                "conv_bias": True,
            }
        )
        model_cpu = BEATs(cfg)
        model_cpu.load_state_dict(checkpoint)
        model_cpu.eval()

        audio = create_synthetic_audio()
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        with torch.no_grad():
            output_cpu = model_cpu(audio_tensor)

        # Test on GPU if available
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model_gpu = BEATs(cfg).cuda()
            model_gpu.load_state_dict(checkpoint)
            model_gpu.eval()

            audio_tensor_gpu = audio_tensor.cuda()

            with torch.no_grad():
                output_gpu = model_gpu(audio_tensor_gpu)

            # Compare outputs (move GPU output to CPU for comparison)
            output_gpu_cpu = output_gpu.cpu()

            # Outputs should be very similar (allowing for minor numerical differences)
            torch.testing.assert_close(output_cpu, output_gpu_cpu, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
