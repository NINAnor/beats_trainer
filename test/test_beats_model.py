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
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        assert checkpoint_path is not None, "Checkpoint should be available"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model config (using checkpoint config)
        cfg = BEATsConfig(checkpoint["cfg"])

        # Initialize model
        model = BEATs(cfg)

        # Load state dict
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        assert model is not None
        assert len(list(model.parameters())) > 0

    @skip_if_no_model()
    def test_checkpoint_parameter_loading(self):
        """Test that model parameters match checkpoint parameters after loading."""
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model and load checkpoint
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)

        if "model" in checkpoint:
            checkpoint_state_dict = checkpoint["model"]
        else:
            checkpoint_state_dict = checkpoint

        # Load using reload_pretrained_parameters (our enhanced method)
        model.reload_pretrained_parameters(state_dict=checkpoint_state_dict)
        model.eval()

        # Get model state dict
        model_state_dict = model.state_dict()

        # Compare parameters that should match
        matching_params = 0
        total_checkpoint_params = 0
        mismatched_params = []

        for param_name, checkpoint_param in checkpoint_state_dict.items():
            total_checkpoint_params += 1

            if param_name in model_state_dict:
                model_param = model_state_dict[param_name]

                # Check if tensors are equal
                if torch.equal(checkpoint_param, model_param):
                    matching_params += 1
                else:
                    # Check if they're approximately equal (for floating point precision)
                    try:
                        torch.testing.assert_close(
                            checkpoint_param, model_param, rtol=1e-6, atol=1e-8
                        )
                        matching_params += 1
                    except AssertionError:
                        mismatched_params.append(
                            {
                                "name": param_name,
                                "checkpoint_shape": checkpoint_param.shape,
                                "model_shape": model_param.shape,
                                "max_diff": torch.max(
                                    torch.abs(checkpoint_param - model_param)
                                ).item(),
                            }
                        )

        # Report results
        print("\nParameter loading verification:")
        print(f"  Total checkpoint parameters: {total_checkpoint_params}")
        print(f"  Matching parameters: {matching_params}")
        print(f"  Match rate: {matching_params / total_checkpoint_params * 100:.1f}%")

        if mismatched_params:
            print(f"  Mismatched parameters ({len(mismatched_params)}):")
            for mismatch in mismatched_params[:5]:  # Show first 5
                print(
                    f"    {mismatch['name']}: shapes {mismatch['checkpoint_shape']} vs {mismatch['model_shape']}, max_diff: {mismatch['max_diff']:.2e}"
                )

        # Assert that we have a high match rate (allowing for some parameters that might not load due to architecture differences)
        match_rate = matching_params / total_checkpoint_params
        assert match_rate > 0.8, (
            f"Parameter match rate too low: {match_rate * 100:.1f}% (expected > 80%)"
        )

        # Assert that we loaded a substantial number of parameters
        assert matching_params > 50, (
            f"Too few parameters loaded: {matching_params} (expected > 50)"
        )

    @skip_if_no_model()
    def test_model_inference(self):
        """Test BEATs model inference."""
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
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
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
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
                assert output.shape[0] == 1  # batch size
                assert len(output.shape) == 3  # (batch, time, features)
                assert output.shape[2] > 0  # feature dimension

        # All outputs should have same feature dimension (last dimension)
        feature_dims = [output.shape[2] for output in outputs]
        assert len(set(feature_dims)) == 1, (
            "All outputs should have same feature dimension"
        )

    @skip_if_no_model()
    def test_model_batch_processing(self):
        """Test model with batch processing."""
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
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
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cuda")

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg).cuda()
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
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
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
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
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.train()  # Set to training mode

        # Create synthetic audio with gradient tracking
        audio = create_synthetic_audio()
        audio_tensor = torch.tensor(audio, requires_grad=True).unsqueeze(0)
        audio_tensor.retain_grad()  # Retain gradients for non-leaf tensor

        # Forward pass
        output = model(audio_tensor)

        # Create dummy loss
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert audio_tensor.grad is not None
        assert torch.any(audio_tensor.grad != 0), "Gradients should be non-zero"

        # Also check that model parameters have gradients
        has_param_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_param_grad = True
                break
        assert has_param_grad, "Model parameters should have gradients"

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
        from beats_trainer.core.feature_extractor import BEATsFeatureExtractor
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        # Test both approaches produce similar results
        checkpoint_path = find_checkpoint()

        # Method 1: Direct model usage
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
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
        from beats_trainer.utils.checkpoints import find_checkpoint
        from BEATs import BEATs, BEATsConfig

        checkpoint_path = find_checkpoint()

        # Test on CPU
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        model_cpu = BEATs(cfg)
        if "model" in checkpoint:
            model_cpu.load_state_dict(checkpoint["model"], strict=False)
        else:
            model_cpu.load_state_dict(checkpoint, strict=False)
        model_cpu.eval()

        audio = create_synthetic_audio()
        audio_tensor = torch.tensor(audio).unsqueeze(0)

        with torch.no_grad():
            output_cpu = model_cpu(audio_tensor)

        # Test on GPU if available
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model_gpu = BEATs(cfg).cuda()
            if "model" in checkpoint:
                model_gpu.load_state_dict(checkpoint["model"], strict=False)
            else:
                model_gpu.load_state_dict(checkpoint, strict=False)
            model_gpu.eval()

            audio_tensor_gpu = audio_tensor.cuda()

            with torch.no_grad():
                output_gpu = model_gpu(audio_tensor_gpu)

            # Compare outputs (move GPU output to CPU for comparison)
            output_gpu_cpu = output_gpu.cpu()

            # Outputs should be very similar (allowing for minor numerical differences)
            torch.testing.assert_close(output_cpu, output_gpu_cpu, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
