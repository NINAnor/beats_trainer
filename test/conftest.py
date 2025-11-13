"""
Test configuration and utilities for BEATs library tests.
"""

import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

# Add src directory to Python path for test imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test configuration
TEST_SAMPLE_RATE = 16000
TEST_DURATION = 2.0  # seconds
TEST_FEATURE_DIM = 768  # BEATs feature dimension
TEST_BATCH_SIZE = 4


class TestConfig:
    """Configuration for tests."""

    # Audio test parameters
    SAMPLE_RATE = TEST_SAMPLE_RATE
    DURATION = TEST_DURATION
    N_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Model test parameters
    FEATURE_DIM = TEST_FEATURE_DIM
    BATCH_SIZE = TEST_BATCH_SIZE

    # Test data paths (relative to test directory)
    TEST_DATA_DIR = Path(__file__).parent / "data"
    TEMP_DIR = Path(tempfile.gettempdir()) / "beats_tests"


def create_synthetic_audio(
    duration: float = TEST_DURATION,
    sample_rate: int = TEST_SAMPLE_RATE,
    frequency: float = 440.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Create synthetic audio signal for testing.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequency: Primary frequency in Hz
        noise_level: Amplitude of added noise (0-1)
        seed: Random seed for reproducibility

    Returns:
        Synthetic audio array
    """
    np.random.seed(seed)

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create base signal (sine wave)
    signal = np.sin(2 * np.pi * frequency * t)

    # Add harmonics for more realistic sound
    signal += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    signal += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)

    # Add noise
    noise = np.random.randn(len(signal)) * noise_level
    signal += noise

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    return signal.astype(np.float32)


def create_test_audio_files(
    output_dir: Path, n_files: int = 5, different_classes: bool = True
) -> List[Path]:
    """
    Create multiple test audio files.

    Args:
        output_dir: Directory to save test files
        n_files: Number of files to create
        different_classes: Whether to create different types of sounds

    Returns:
        List of created file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []

    for i in range(n_files):
        # Create different frequencies for different "classes"
        if different_classes:
            frequency = 220 * (2 ** (i % 5))  # Different octaves
        else:
            frequency = 440.0  # Same frequency

        audio = create_synthetic_audio(
            frequency=frequency,
            seed=42 + i,  # Different random seed for each file
        )

        file_path = output_dir / f"test_audio_{i:03d}.wav"

        # Save using librosa (requires soundfile)
        try:
            import soundfile as sf

            sf.write(file_path, audio, TEST_SAMPLE_RATE)
        except ImportError:
            # Fallback to scipy if soundfile not available
            try:
                from scipy.io import wavfile

                wavfile.write(file_path, TEST_SAMPLE_RATE, audio)
            except ImportError:
                # Create a dummy file for testing structure
                file_path.write_text(f"dummy_audio_file_{i}")

        file_paths.append(file_path)

    return file_paths


@pytest.fixture
def temp_dir():
    """Fixture to provide a temporary directory for tests."""
    temp_path = TestConfig.TEMP_DIR
    temp_path.mkdir(parents=True, exist_ok=True)
    yield temp_path

    # Cleanup after test
    import shutil

    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def synthetic_audio():
    """Fixture to provide synthetic audio data."""
    return create_synthetic_audio()


@pytest.fixture
def batch_synthetic_audio():
    """Fixture to provide a batch of synthetic audio data."""
    batch = []
    for i in range(TEST_BATCH_SIZE):
        audio = create_synthetic_audio(
            frequency=440 * (2 ** (i % 3)),  # Different frequencies
            seed=42 + i,
        )
        batch.append(audio)
    return batch


@pytest.fixture
def test_audio_files(temp_dir):
    """Fixture to provide test audio files."""
    return create_test_audio_files(temp_dir)


def skip_if_no_model():
    """Decorator to skip tests if no BEATs model is available."""
    return pytest.mark.skipif(
        not _model_available(), reason="BEATs model not available for testing"
    )


def _model_available() -> bool:
    """Check if BEATs model is available for testing."""
    try:
        from beats_trainer.utils.checkpoints import find_checkpoint

        checkpoint = find_checkpoint()
        return checkpoint is not None
    except Exception:
        return False


def skip_if_no_gpu():
    """Decorator to skip tests if GPU is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="GPU not available for testing"
    )


class AudioTestCase:
    """Base class for audio-related test cases."""

    def setUp(self):
        """Set up test case."""
        self.temp_dir = TestConfig.TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test case."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def assert_audio_shape(self, audio: np.ndarray, expected_length: int = None):
        """Assert audio has correct shape."""
        assert isinstance(audio, np.ndarray), "Audio should be numpy array"
        assert audio.ndim == 1, "Audio should be 1D array"

        if expected_length:
            assert len(audio) == expected_length, (
                f"Expected length {expected_length}, got {len(audio)}"
            )

    def assert_features_shape(
        self, features: np.ndarray, expected_batch_size: int = None
    ):
        """Assert features have correct shape."""
        assert isinstance(features, np.ndarray), "Features should be numpy array"
        assert features.ndim == 2, (
            "Features should be 2D array (batch_size, feature_dim)"
        )
        assert features.shape[1] == TEST_FEATURE_DIM, (
            f"Expected feature dim {TEST_FEATURE_DIM}, got {features.shape[1]}"
        )

        if expected_batch_size:
            assert features.shape[0] == expected_batch_size, (
                f"Expected batch size {expected_batch_size}, got {features.shape[0]}"
            )
