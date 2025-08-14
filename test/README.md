# BEATs Library Test Suite

This folder contains comprehensive tests for the BEATs feature extraction library.

## 🧪 Test Structure

### Test Files

- **`test_feature_extractor.py`** - Tests for the `BEATsFeatureExtractor` class
- **`test_checkpoint_utils.py`** - Tests for checkpoint management utilities
- **`test_beats_model.py`** - Tests for the core BEATs model functionality
- **`test_cli_and_tools.py`** - Tests for CLI tools and usage patterns
- **`test_integration.py`** - End-to-end integration tests and real-world scenarios
- **`conftest.py`** - Test configuration, fixtures, and utilities
- **`run_tests.py`** - Test runner with different execution modes

### Test Categories

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Tests for complete workflows and component interaction
- **CLI Tests**: Tests for command-line tools and scripts
- **Performance Tests**: Memory usage and timing tests (marked as `slow`)
- **GPU Tests**: CUDA-specific tests (marked as `gpu`, auto-skipped if no GPU)

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
uv run python test/run_tests.py

# Run only fast unit tests
uv run python test/run_tests.py unit

# Run integration tests
uv run python test/run_tests.py integration

# Run quick development tests (stop on first failure)
uv run python test/run_tests.py quick
```

### Using pytest directly

```bash
# Run all tests with verbose output
uv run pytest test/ -v

# Run specific test file
uv run pytest test/test_feature_extractor.py -v

# Run tests matching pattern
uv run pytest test/ -k "checkpoint" -v

# Run with coverage
uv run pytest test/ --cov=beats_trainer --cov-report=html
```

### Test Markers

- `@skip_if_no_model` - Skip if BEATs model not available
- `@skip_if_no_gpu` - Skip if CUDA GPU not available
- `@pytest.mark.slow` - Mark as slow test (excluded in quick runs)

## 📋 Test Coverage

### Feature Extractor Tests

- ✅ Initialization with automatic checkpoint detection
- ✅ Different pooling methods (mean, max, cls)
- ✅ Single audio and batch feature extraction
- ✅ File-based feature extraction
- ✅ Feature normalization options
- ✅ Device selection (CPU/GPU/auto)
- ✅ Error handling for invalid inputs
- ✅ Memory management and cleanup

### Checkpoint Management Tests

- ✅ Listing available models
- ✅ Finding existing checkpoints
- ✅ Downloading models from Microsoft Research
- ✅ Checkpoint validation
- ✅ Automatic checkpoint ensuring
- ✅ Error handling for network issues
- ✅ Directory priority handling

### Model Tests

- ✅ Model loading and initialization
- ✅ Forward inference
- ✅ Different input lengths
- ✅ Batch processing
- ✅ GPU inference (if available)
- ✅ Deterministic outputs
- ✅ Gradient flow for training

### Integration Tests

- ✅ Complete feature extraction workflows
- ✅ Multi-model comparisons
- ✅ Batch processing pipelines
- ✅ Save/load feature workflows
- ✅ Error recovery scenarios
- ✅ Real-world application scenarios

### CLI and Tools Tests

- ✅ Command-line interface functionality
- ✅ Library import structure
- ✅ Error handling patterns
- ✅ Configuration validation
- ✅ Documentation consistency
- ✅ Performance characteristics

## 🛠️ Test Configuration

### Test Data

Tests use synthetic audio data by default to ensure reproducibility:

- **Synthetic Audio**: Generated sine waves with harmonics and noise
- **Multiple Frequencies**: Different classes have different frequency profiles
- **Batch Audio**: Arrays of synthetic audio for batch testing
- **Mock Files**: Temporary audio files for file-based testing

### Fixtures

- `temp_dir` - Temporary directory for test files
- `synthetic_audio` - Single synthetic audio array
- `batch_synthetic_audio` - Batch of synthetic audio arrays
- `test_audio_files` - Mock audio files for testing

### Configuration

```python
# Test parameters (conftest.py)
TEST_SAMPLE_RATE = 16000
TEST_DURATION = 2.0
TEST_FEATURE_DIM = 768
TEST_BATCH_SIZE = 4
```

## ⚡ Performance Testing

### Memory Tests

- Import memory usage monitoring
- Feature extraction memory tracking
- Memory leak detection
- GPU memory management (if available)

### Timing Tests

- Library import timing
- Feature extraction speed
- Batch processing efficiency
- Model loading performance

### CPU Usage Tests

- CPU utilization monitoring
- Resource usage patterns
- Background process detection

## 🎯 Real-World Scenarios

### Audio Similarity Search

Tests complete workflow for building similarity search:
- Feature database creation
- Query processing
- Similarity computation
- Top-K retrieval

### Audio Classification

Tests transfer learning scenario:
- Feature extraction from pretrained model
- Classifier training on features
- Performance evaluation
- Feature importance analysis

### Audio Clustering

Tests unsupervised learning:
- Feature-based clustering
- Cluster quality evaluation
- Cluster analysis and interpretation

## 🔧 Development Workflow

### Running Tests During Development

```bash
# Quick feedback during development
uv run python test/run_tests.py quick

# Test specific component you're working on
uv run pytest test/test_feature_extractor.py::TestBEATsFeatureExtractor::test_extract_features_single_audio -v

# Run with debugger on failure
uv run pytest test/test_feature_extractor.py --pdb
```

### Adding New Tests

1. **Unit Tests**: Add to appropriate `test_*.py` file
2. **Use Fixtures**: Leverage existing fixtures from `conftest.py`
3. **Mock External Dependencies**: Use mocks for file I/O, downloads, etc.
4. **Test Error Cases**: Include both success and failure scenarios
5. **Add Markers**: Use `@skip_if_no_model` for tests requiring models

### Test Writing Guidelines

- **Descriptive Names**: Test names should clearly describe what's being tested
- **Single Responsibility**: Each test should test one specific behavior
- **Mock External Resources**: Don't depend on internet, real files, etc.
- **Cleanup**: Use fixtures and context managers for proper cleanup
- **Assertions**: Include descriptive assertion messages

## 📊 Coverage Goals

- **Feature Extractor**: > 90% coverage
- **Checkpoint Utils**: > 85% coverage
- **Model Interface**: > 80% coverage
- **CLI Tools**: > 70% coverage
- **Integration**: > 60% coverage

## 🚨 Continuous Integration

Tests are designed to run in CI environments:

- **No External Dependencies**: All tests use mocks or synthetic data
- **Timeout Protection**: Long-running tests have timeouts
- **Platform Independence**: Tests work on Linux, macOS, Windows
- **Resource Awareness**: Graceful degradation when GPU/models unavailable

Run the test suite to ensure your BEATs library is working correctly! 🎵✅
