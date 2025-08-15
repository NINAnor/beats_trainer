# BEATs on ESC-50: A Simple Audio Classification Library

[BEATs (Bidirectional Encoder representation from Audio Transformers)](https://github.com/microsoft/unilm/tree/master/beats) is a very powerful audio classification model and perform state-of-the-art results. However it has proven difficult to train due to model complexity as well as a lack of documentation.

We attempt to solve this by building a streamlined library for training and using BEATs (Bidirectional Encoder representation from Audio Transformers) models on audio classification tasks, with automatic checkpoint management. The library also aims at providing utilities for using BEATs as a feature extractor.

## 🚀 Quick Start

### Install and Test

```bash
# Install from GitHub
pip install git+https://github.com/ninanor/beats-trainer.git

# Test the installation
python -c "
from beats_trainer import BEATsFeatureExtractor
extractor = BEATsFeatureExtractor()
print('✅ BEATs Trainer installed successfully!')
print(f'Feature dimension: {extractor.get_feature_dim()}')
"
```

## ✨ Key Features

- **🤖 Automatic Checkpoint Management**: Download BEATs models automatically
- **📊 Feature Extraction**: Extract high-quality audio embeddings
- **🔧 Simple Training API**: Fine-tune BEATs with just a few lines of code
- **📚 Comprehensive Notebooks**: Step-by-step tutorials and examples
- **⚡ GPU Support**: Automatic CUDA detection and optimization

## 🎯 Usage Examples

### Feature Extraction (No Training Required)

```python
from beats_trainer import BEATsFeatureExtractor

# Automatically downloads model if needed
extractor = BEATsFeatureExtractor()

# Extract features from audio file
features = extractor.extract_from_file("audio.wav")
print(f"Features shape: {features.shape}")  # (768,)

# Batch processing
features = extractor.extract_from_files(["audio1.wav", "audio2.wav"])
print(f"Batch features: {features.shape}")  # (2, 768)
```

### Checkpoint Management

```python
from beats_trainer import ensure_checkpoint, list_available_models

# List available models
models = list_available_models()
print(models.keys())  # ['BEATs_iter3_plus_AS2M', 'openbeats']

# Ensure checkpoint is available (automatically downloads from Hugging Face Hub)
checkpoint_path = ensure_checkpoint()
print(f"Checkpoint ready at: {checkpoint_path}")
```

**🚀 Automatic Downloads**: Checkpoints are automatically downloaded from [Hugging Face Hub](https://huggingface.co/datasets/ninanor/beats-checkpoints) - no OneDrive tokens required!

### Fine-tuning on Custom Data

```python
from beats_trainer import BEATsTrainer

# Train from directory structure
trainer = BEATsTrainer.from_directory("/path/to/dataset")
results = trainer.train()

# Extract features with the trained model
features = trainer.extract_features(["new_audio1.wav", "new_audio2.wav"])
```

## 📁 Data Formatting for Training

### Option 1: Directory Structure (Recommended)

Organize your audio files into class directories:

```
your_dataset/
├── class1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── audio3.mp3
├── class2/
│   ├── audio4.wav
│   ├── audio5.flac
│   └── audio6.wav
└── class3/
    ├── audio7.wav
    └── audio8.wav
```

**Usage:**
```python
from beats_trainer import BEATsTrainer

# Automatically detects classes from directory names
trainer = BEATsTrainer.from_directory("path/to/your_dataset")
results = trainer.train()
```

### Option 2: CSV Metadata File

Create a CSV file with audio paths and labels:

**dataset.csv:**
```csv
filename,category
audio/sample1.wav,bird
audio/sample2.mp3,dog
audio/sample3.flac,cat
audio/sample4.wav,bird
```

**Usage:**
```python
from beats_trainer import BEATsTrainer

# Load from CSV metadata
trainer = BEATsTrainer.from_csv(
    csv_path="dataset.csv",
    audio_column="filename",      # Column with audio file paths
    label_column="category"       # Column with class labels
)
results = trainer.train()
```

### Option 3: Custom Configuration

For more control over data loading:

```python
from beats_trainer import BEATsTrainer
from beats_trainer.config import DataConfig, TrainingConfig

# Configure data loading
data_config = DataConfig(
    data_dir="path/to/audio/files",
    metadata_file="labels.csv",
    audio_column="file_path",
    label_column="class_name",
    train_split=0.7,              # 70% for training
    val_split=0.2,                # 20% for validation
    test_split=0.1,               # 10% for testing
    batch_size=16,
    sample_rate=16000             # BEATs requires 16kHz
)

# Configure training
training_config = TrainingConfig(
    learning_rate=1e-4,
    max_epochs=30,
    patience=5
)

trainer = BEATsTrainer(data_config=data_config, training_config=training_config)
results = trainer.train()
```

### 🗂️ Preset Dataset Loaders

For common datasets, use built-in loaders:

```python
from beats_trainer.datasets import load_esc50, load_urbansound8k

# ESC-50 Environmental Sound Classification
df = load_esc50("path/to/ESC-50-master")

# UrbanSound8K
df = load_urbansound8k("path/to/UrbanSound8K")

# Use with trainer
trainer = BEATsTrainer.from_dataframe(df, audio_column="filename", label_column="category")
results = trainer.train()
```

### 🎵 Supported Audio Formats

- **WAV** (`.wav`) - Recommended for best quality
- **MP3** (`.mp3`) - Compressed format, widely supported
- **FLAC** (`.flac`) - Lossless compression
- **M4A** (`.m4a`) - Apple audio format
- **OGG** (`.ogg`) - Open source format

### 📊 Data Requirements

- **Sample Rate**: Audio will be automatically resampled to 16kHz (BEATs requirement)
- **Duration**: Any length supported (longer files will be more memory intensive)
- **File Size**: No strict limits, but consider memory usage for very large files
- **Classes**: Minimum 2 classes required for classification

### 💡 Data Preparation Tips

**For Best Results:**
```python
# Load and inspect your data
from beats_trainer.datasets import scan_directory_dataset, load_csv_dataset

# For directory structure
df = scan_directory_dataset("path/to/your_dataset")
print(f"Classes: {df['category'].nunique()}")
print(f"Total files: {len(df)}")
print(f"Class distribution:\n{df['category'].value_counts()}")

# For CSV data
df = load_csv_dataset("your_data.csv", audio_col="filename", label_col="category")
```

**Quality Guidelines:**
- ✅ **Balanced classes**: Similar number of samples per class
- ✅ **Clean audio**: Remove or minimize background noise
- ✅ **Consistent format**: Same sample rate and bit depth when possible
- ✅ **Sufficient data**: At least 20-50 examples per class for fine-tuning
- ✅ **Representative samples**: Cover the variation you expect in real-world usage

**Example Data Validation:**
```python
from beats_trainer.datasets import validate_dataset

# Validate your dataset (works with DataFrame + data directory)
df = scan_directory_dataset("path/to/dataset")
validation_report = validate_dataset(df, "path/to/dataset")

if validation_report["warnings"]:
    print("⚠️ Warnings found:")
    for warning in validation_report["warnings"]:
        print(f"  - {warning}")

if validation_report["errors"]:
    print("❌ Errors found:")
    for error in validation_report["errors"]:
        print(f"  - {error}")
else:
    print("✅ Dataset validation passed!")
```

## Ideas for usage

### 🔍 Audio Search & Similarity
```python
# Build an audio search engine
extractor = BEATsFeatureExtractor()
database_features = extractor.extract_from_files(audio_database)

# Find similar sounds
query_features = extractor.extract_from_file("query.wav")
similarities = cosine_similarity([query_features], database_features)
```

### 🎵 Content-Based Recommendation
```python
# Music recommendation based on audio features
user_liked_songs = ["song1.mp3", "song2.mp3"]
user_profile = extractor.extract_from_files(user_liked_songs).mean(axis=0)

# Find similar songs
candidate_features = extractor.extract_from_files(music_catalog)
recommendations = find_closest_songs(user_profile, candidate_features)
```

### 🔬 Audio Analysis & Clustering
```python
# Analyze audio patterns without labels
features = extractor.extract_from_files(unlabeled_audio)
clusters = KMeans(n_clusters=10).fit_predict(features)

# Visualize with UMAP
embedding_2d = umap.UMAP().fit_transform(features)
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters)
```

## 🔧 Advanced Configuration

### Custom Pooling Strategies
```python
# Different ways to aggregate sequence features
extractors = {
    'mean': BEATsFeatureExtractor(pooling='mean'),      # Average over time
    'max': BEATsFeatureExtractor(pooling='max'),        # Max over time
    'first': BEATsFeatureExtractor(pooling='first'),    # First token (CLS-like)
    'last': BEATsFeatureExtractor(pooling='last')       # Last token
}
```

### Training Configuration
```python
from beats_trainer.config import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    max_epochs=50,
    batch_size=32,
    optimizer="adamw"
)

trainer = BEATsTrainer.from_directory("/path/to/data", config=config)
```

## 📜 License

This project is licensed under the MIT License. The BEATs model weights are provided by Microsoft Research under their respective license.

## 🙏 Acknowledgments

- **BEATs**: [Microsoft Research](https://github.com/microsoft/unilm/tree/master/beats)

---
