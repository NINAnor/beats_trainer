"""
Test suite for BEATs feature extraction functionality.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

from beats_trainer.core.feature_extractor import BEATsFeatureExtractor
from .conftest import (
    TestConfig,
    skip_if_no_model,
    AudioTestCase,
    create_synthetic_audio,
)


class TestBEATsFeatureExtractor(AudioTestCase):
    """Test cases for BEATsFeatureExtractor class."""

    def setUp(self):
        super().setUp()

    @skip_if_no_model()
    def test_extractor_initialization_with_auto_checkpoint(self):
        """Test extractor initialization with automatic checkpoint detection."""
        extractor = BEATsFeatureExtractor(
            model_path=None,  # Auto-detect
            pooling="mean",
            device="auto",
        )

        assert extractor.model is not None
        assert extractor.model_path is not None
        assert extractor.model_path.exists()
        assert extractor.pooling == "mean"

    def test_extractor_initialization_with_invalid_path(self):
        """Test extractor initialization with invalid checkpoint path."""
        with pytest.raises(FileNotFoundError):
            BEATsFeatureExtractor(
                model_path=Path("/nonexistent/model.pt"), pooling="mean"
            )

    def test_extractor_initialization_with_invalid_pooling(self):
        """Test extractor initialization with invalid pooling method."""
        with pytest.raises(ValueError):
            BEATsFeatureExtractor(model_path=None, pooling="invalid_pooling_method")

    @skip_if_no_model()
    def test_different_pooling_methods(self):
        """Test different pooling methods."""
        pooling_methods = ["mean", "max", "cls"]

        for pooling in pooling_methods:
            extractor = BEATsFeatureExtractor(
                model_path=None,
                pooling=pooling,
                device="cpu",  # Use CPU for consistent testing
            )
            assert extractor.pooling == pooling

    @skip_if_no_model()
    def test_extract_features_single_audio(self, synthetic_audio):
        """Test feature extraction from single audio array."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        features = extractor.extract_features(synthetic_audio)

        # Verify output shape and properties
        self.assert_features_shape(features, expected_batch_size=1)
        assert not np.isnan(features).any(), "Features should not contain NaN values"
        assert np.isfinite(features).all(), "Features should be finite"

    @skip_if_no_model()
    def test_extract_features_batch_audio(self, batch_synthetic_audio):
        """Test feature extraction from batch of audio arrays."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        features = extractor.extract_from_batch(batch_synthetic_audio)

        # Verify output shape and properties
        self.assert_features_shape(
            features, expected_batch_size=len(batch_synthetic_audio)
        )
        assert not np.isnan(features).any(), "Features should not contain NaN values"
        assert np.isfinite(features).all(), "Features should be finite"

    @skip_if_no_model()
    def test_extract_features_different_lengths(self):
        """Test feature extraction from audio of different lengths."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        # Create audio of different lengths
        short_audio = create_synthetic_audio(duration=1.0)  # 1 second
        long_audio = create_synthetic_audio(duration=5.0)  # 5 seconds

        short_features = extractor.extract_features(short_audio)
        long_features = extractor.extract_features(long_audio)

        # Both should produce same feature dimension
        assert short_features.shape[1] == long_features.shape[1]
        assert short_features.shape[0] == 1
        assert long_features.shape[0] == 1

    @skip_if_no_model()
    def test_extract_from_files_mock(self, temp_dir):
        """Test extract_from_files method with mock audio files."""
        # Create mock audio files
        audio_files = []
        for i in range(3):
            file_path = temp_dir / f"test_{i}.wav"
            audio = create_synthetic_audio(frequency=440 * (i + 1))

            # Mock file by saving audio data
            np.save(file_path.with_suffix(".npy"), audio)
            audio_files.append(file_path.with_suffix(".npy"))

        # Mock the extractor's file loading
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        # Mock the file loading to use our numpy arrays
        def mock_load_audio(file_path):
            return np.load(file_path)

        with patch.object(extractor, "_load_audio_file", side_effect=mock_load_audio):
            features = extractor.extract_from_files(audio_files, batch_size=2)

        self.assert_features_shape(features, expected_batch_size=3)

    @skip_if_no_model()
    def test_extract_features_normalization(self, synthetic_audio):
        """Test feature normalization option."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        features_normalized = extractor.extract_features(
            synthetic_audio, normalize=True
        )
        features_raw = extractor.extract_features(synthetic_audio, normalize=False)

        # Normalized features should have unit norm (approximately)
        norm = np.linalg.norm(features_normalized, axis=1)
        assert np.allclose(norm, 1.0, atol=1e-6), (
            "Normalized features should have unit norm"
        )

        # Raw features might not have unit norm
        raw_norm = np.linalg.norm(features_raw, axis=1)
        assert not np.allclose(raw_norm, 1.0, atol=1e-2), (
            "Raw features should not necessarily have unit norm"
        )

    def test_device_selection(self):
        """Test device selection logic."""
        # Test auto device selection
        extractor_auto = BEATsFeatureExtractor(
            model_path=None, pooling="mean", device="auto"
        )

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(extractor_auto.device) == expected_device

        # Test explicit CPU device
        extractor_cpu = BEATsFeatureExtractor(
            model_path=None, pooling="mean", device="cpu"
        )
        assert str(extractor_cpu.device) == "cpu"

    @skip_if_no_model()
    def test_extract_features_consistency(self, synthetic_audio):
        """Test that feature extraction is consistent across multiple runs."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        # Extract features multiple times
        features1 = extractor.extract_features(synthetic_audio)
        features2 = extractor.extract_features(synthetic_audio)
        features3 = extractor.extract_features(synthetic_audio)

        # Features should be identical (deterministic)
        np.testing.assert_array_almost_equal(features1, features2, decimal=6)
        np.testing.assert_array_almost_equal(features2, features3, decimal=6)

    @skip_if_no_model()
    def test_pooling_methods_consistency(self, synthetic_audio):
        """Test that different pooling methods produce different features."""
        pooling_methods = ["mean", "max", "cls"]
        features_by_pooling = {}

        for pooling in pooling_methods:
            extractor = BEATsFeatureExtractor(
                model_path=None, pooling=pooling, device="cpu"
            )
            features = extractor.extract_features(synthetic_audio)
            features_by_pooling[pooling] = features

        # Different pooling methods should produce different features
        for i, method1 in enumerate(pooling_methods):
            for method2 in pooling_methods[i + 1 :]:
                features1 = features_by_pooling[method1]
                features2 = features_by_pooling[method2]

                # Features should be different
                assert not np.allclose(features1, features2, atol=1e-4), (
                    f"{method1} and {method2} pooling should produce different features"
                )

    def test_empty_audio_handling(self):
        """Test handling of empty or invalid audio input."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        # Test empty audio array
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            empty_audio = np.array([])
            extractor.extract_features(empty_audio)

        # Test None input
        with pytest.raises((ValueError, TypeError)):
            extractor.extract_features(None)

    @skip_if_no_model()
    def test_memory_cleanup(self, batch_synthetic_audio):
        """Test that memory is properly cleaned up after feature extraction."""
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        # Get initial memory usage if on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Extract features from large batch
        large_batch = batch_synthetic_audio * 10  # 40 samples
        features = extractor.extract_from_batch(large_batch)

        # Verify features were extracted
        self.assert_features_shape(features, expected_batch_size=len(large_batch))

        # Check memory cleanup on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()

            # Memory should not have grown significantly
            memory_growth = final_memory - initial_memory
            assert memory_growth < 100_000_000, (
                f"Memory grew by {memory_growth} bytes, possible memory leak"
            )


class TestFeatureExtractorIntegration:
    """Integration tests for feature extractor."""

    @skip_if_no_model()
    def test_end_to_end_feature_pipeline(self, temp_dir):
        """Test complete feature extraction pipeline."""
        # Create test audio files
        n_files = 5
        audio_files = []

        for i in range(n_files):
            audio = create_synthetic_audio(
                frequency=440 * (2 ** (i % 3)),  # Different frequencies
                seed=42 + i,
            )
            file_path = temp_dir / f"audio_{i}.npy"
            np.save(file_path, audio)
            audio_files.append(file_path)

        # Initialize extractor
        extractor = BEATsFeatureExtractor(model_path=None, pooling="mean", device="cpu")

        # Mock file loading
        def mock_load_audio(file_path):
            return np.load(file_path)

        with patch.object(extractor, "_load_audio_file", side_effect=mock_load_audio):
            # Extract features
            features = extractor.extract_from_files(audio_files, batch_size=2)

        # Verify results
        assert features.shape == (n_files, TestConfig.FEATURE_DIM)
        assert not np.isnan(features).any()
        assert np.isfinite(features).all()

        # Features from different frequencies should be different
        feature_similarities = np.corrcoef(features)
        np.fill_diagonal(feature_similarities, 0)  # Ignore self-similarity

        # Not all features should be identical
        max_similarity = np.max(feature_similarities)
        assert max_similarity < 0.99, (
            "Features should not be too similar for different audio"
        )


if __name__ == "__main__":
    pytest.main([__file__])
