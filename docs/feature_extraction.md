# Feature Extraction with BEATs

Extract high-quality audio embeddings using pre-trained BEATs models.

## Quick Start

```python
from beats_trainer import BEATsFeatureExtractor

# Use default model (BEATs_iter3_plus_AS2M - recommended)
extractor = BEATsFeatureExtractor()

# Extract features from audio file
features = extractor.extract_from_file("audio.wav")
print(f"Features shape: {features.shape}")  # (768,)

# Batch processing
features = extractor.extract_from_files(["audio1.wav", "audio2.wav"])
print(f"Batch features: {features.shape}")  # (2, 768)
```

## Checkpoint Management

### Available Models

```python
from beats_trainer import list_available_models

models = list_available_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")
```

### Automatic Download

```python
from beats_trainer import ensure_checkpoint

# Download default model (best quality)
checkpoint_path = ensure_checkpoint()

# Download specific model
checkpoint_path = ensure_checkpoint("openbeats")  # Faster, smaller model
```

### Using Different Models

```python
# Method 1: Use model name (auto-downloads)
extractor = BEATsFeatureExtractor(model_path="openbeats")

# Method 2: Local file path
extractor = BEATsFeatureExtractor(model_path="/path/to/model.pt")

# Method 3: Custom trained model
extractor = BEATsFeatureExtractor(model_path="/path/to/my_model.ckpt")
```

## Advanced Configuration

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

## Use Cases

### Audio Search & Similarity

```python
# Build an audio search engine
extractor = BEATsFeatureExtractor()
database_features = extractor.extract_from_files(audio_database)

# Find similar sounds
from sklearn.metrics.pairwise import cosine_similarity
query_features = extractor.extract_from_file("query.wav")
similarities = cosine_similarity([query_features], database_features)
```

### Content-Based Recommendation

```python
# Music recommendation based on audio features
user_liked_songs = ["song1.mp3", "song2.mp3"]
user_profile = extractor.extract_from_files(user_liked_songs).mean(axis=0)

# Find similar songs
candidate_features = extractor.extract_from_files(music_catalog)
recommendations = find_closest_songs(user_profile, candidate_features)
```

### Audio Clustering

```python
# Analyze audio patterns without labels
from sklearn.cluster import KMeans
import umap

features = extractor.extract_from_files(unlabeled_audio)
clusters = KMeans(n_clusters=10).fit_predict(features)

# Visualize with UMAP
embedding_2d = umap.UMAP().fit_transform(features)
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters)
```

## Supported Audio Formats

- **WAV** (`.wav`) - Recommended for best quality
- **MP3** (`.mp3`) - Compressed format, widely supported
- **FLAC** (`.flac`) - Lossless compression
- **M4A** (`.m4a`) - Apple audio format
- **OGG** (`.ogg`) - Open source format

Audio is automatically resampled to 16kHz (BEATs requirement).