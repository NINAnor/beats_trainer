# beats_trainer: A high level API for training BEATs

[BEATs (Bidirectional Encoder representation from Audio Transformers)](https://github.com/microsoft/unilm/tree/master/beats) is a very powerful audio classification model and perform state-of-the-art results. However it has proven difficult to train due to model complexity as well as a lack of documentation.

We attempt to solve this by building a streamlined library for training and using BEATs (Bidirectional Encoder representation from Audio Transformers) models on audio classification tasks, with automatic checkpoint management. The library also aims at providing utilities for using BEATs as a feature extractor.

## 🚀 Quick Start

## 🚀 Installation

### Option 1: Using uv (Recommended)

```bash
# Install from GitHub
uv add git+https://github.com/ninanor/beats_trainer.git
```

### Option 2: Using pip

```bash
# Install from GitHub
pip install git+https://github.com/ninanor/beats_trainer.git
```

## ✨ Key Features

- **🤖 Automatic Checkpoint Management**: Download BEATs models automatically
- **� Dataset Auto-Download**: ESC-50 and UrbanSound8K with automatic organization
- **�📊 Feature Extraction**: Extract high-quality audio embeddings
- **🔧 Simple Training API**: Fine-tune BEATs with just a few lines of code
- **🏗️ Train from Scratch**: Initialize and train BEATs without pre-trained weights
- **📚 Comprehensive Notebooks**: Step-by-step tutorials and examples
- **⚡ GPU Support**: Automatic CUDA detection and optimization

## 🎯 Usage Examples

### Feature Extraction (No Training Required)

```python
from beats_trainer import BEATsFeatureExtractor

# Use default model (BEATs_iter3_plus_AS2M - recommended)
extractor = BEATsFeatureExtractor()

# OR download a specific model first:
from beats_trainer import download_beats_checkpoint
checkpoint_path = download_beats_checkpoint("openbeats")  # Alternative model
extractor = BEATsFeatureExtractor(model_path=checkpoint_path)

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
print(models.keys())  # ['BEATs_iter3_plus_AS2M', 'openbeats', 'openbeats_i1', 'openbeats_i2', 'openbeats_i3']

# Ensure checkpoint is available (automatically downloads from Hugging Face Hub)
checkpoint_path = ensure_checkpoint()
print(f"Checkpoint ready at: {checkpoint_path}")
```

### Fine-tuning on Custom Data

For training only the classification head:

```python
from beats_trainer import BEATsTrainer

# Train from directory structure
trainer = BEATsTrainer.from_directory("/path/to/dataset")
results = trainer.train()

# Extract features with the trained model
features = trainer.extract_features(["new_audio1.wav", "new_audio2.wav"])
```

For fine-tuning the entire model architecture:

```python
from beats_trainer import BEATsTrainer
from beats_trainer.config import Config

config.model.freeze_backbone = False  # Fine-tune the entire model
config.model.fine_tune_backbone = True

trainer = BEATsTrainer.from_directory("/path/to/dataset", config=config)
results = trainer.train()
```

### ESC-50 Dataset (Auto-Download & Train)

Get started immediately with the ESC-50 environmental sound classification dataset:

```python
from beats_trainer import BEATsTrainer
from beats_trainer.config import Config

# Create optimized config for 95% ESC-50 performance
config = Config()
config.model.freeze_backbone = False  # Key: fine-tune entire model
config.model.fine_tune_backbone = True
config.training.max_epochs = 50
config.training.learning_rate = 5e-5

# Auto-download, organize, and train (achieves ~95% accuracy)
trainer = BEATsTrainer.from_esc50(data_dir="./datasets", config=config)
trainer.train()
```

**What this does:**
- 📥 **Downloads** ESC-50 dataset (50 environmental sound classes)
- 📁 **Organizes** into proper folder structure for training
- 🚀 **Trains** with optimal settings for 95% accuracy
- 💾 **Saves** best model checkpoint automatically

The dataset will be organized as:
```
datasets/
└── ESC50_organized/
    ├── airplane/
    ├── breathing/
    ├── car_horn/
    └── ... (50 classes total)
```


### Training From Scratch (No Pre-trained Weights)

For scenarios where you want to train BEATs completely from scratch without using pre-trained weights:

```python
from beats_trainer.config import Config, ModelConfig, DataConfig, TrainingConfig

# Create configuration for training from scratch
config = Config(
    experiment_name="beats_from_scratch",
    data=DataConfig(
        data_dir="path/to/your/audio/data",
        batch_size=16,  # Smaller batch size often works better
        sample_rate=16000
    ),
    model=ModelConfig(
        train_from_scratch=True,    # Key parameter!

        # Model architecture (customizable)
        encoder_layers=12,          # Number of transformer layers
        encoder_embed_dim=768,      # Hidden dimension
        encoder_attention_heads=12, # Number of attention heads
        input_patch_size=16,        # Audio patch size

        # Training strategy (automatically configured)
        fine_tune_backbone=True,    # Train entire backbone
        freeze_backbone=False,      # Don't freeze any layers
    ),
    training=TrainingConfig(
        learning_rate=1e-3,         # Higher LR for training from scratch
        max_epochs=100,             # More epochs needed
        patience=15,                # More patience for convergence
        optimizer="adamw"
    )
)

# Train the model
trainer = BEATsTrainer(config)
results = trainer.train()
```

**Key Differences from Pre-trained Training:**
- No checkpoint file is required
- Model weights are randomly initialized
- Entire backbone is trained (not frozen)
- Typically requires more epochs and data
- Higher learning rates are often beneficial

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

## 📜 License

This project is licensed under the MIT License.

BEATs_iter3_plus_AS2M.pt model weights are provided by Microsoft Research under their respective license terms (MIT). These can be found in their [GitHub repository](https://github.com/microsoft/unilm/tree/master/beats). The BEATs paper detailing its architecture is [available on Arxiv](https://arxiv.org/pdf/2507.14129)

OpenBEATs-Base-i3.pt model weights are provided by the OpenBEATs project under their respective license terms. More details can be found in the [OpenBEATs paper](https://arxiv.org/pdf/2507.14129)

---
