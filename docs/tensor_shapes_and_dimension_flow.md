# BEATs Tensor Shape and Dimension Flow - Complete Guide

## Overview

This document provides a comprehensive understanding of how audio data flows through the BEATs (Bidirectional Encoder representation from Audio Transformers) model, with detailed explanations of tensor shapes and dimension transformations at each step.

## Table of Contents

1. [Model Architecture Overview](#model-architecture-overview)
2. [Configuration Parameters](#configuration-parameters)
3. [Complete Dimension Flow](#complete-dimension-flow)
4. [Patch Embedding Deep Dive](#patch-embedding-deep-dive)
5. [Variable Length Audio Handling](#variable-length-audio-handling)
6. [Divisibility Requirements](#divisibility-requirements)
7. [Memory and Computational Analysis](#memory-and-computational-analysis)
8. [Practical Examples](#practical-examples)
9. [Key Insights and Gotchas](#key-insights-and-gotchas)

---

## Model Architecture Overview

BEATs follows this high-level processing pipeline:

```
Raw Audio → Filterbank Features → Patch Embedding → Transformer Encoder → Pooling → Final Features
```

The model transforms variable-length audio into fixed-size feature representations through a sophisticated patch-based approach similar to Vision Transformers (ViTs).

---

## Configuration Parameters

From the actual BEATs checkpoint configuration:

```python
input_patch_size: 16        # Square patch size: 16×16
embed_dim: 512             # Patch embedding dimension
encoder_embed_dim: 768     # Transformer encoder dimension
encoder_layers: 12         # Number of transformer layers
encoder_attention_heads: 12 # Multi-head attention heads
```

**Key Audio Processing Parameters:**
- **Sample Rate**: 16 kHz (fixed)
- **Frame Length**: 25ms
- **Frame Shift**: 10ms (15ms overlap)
- **Mel Bins**: 128

---

## Complete Dimension Flow

### Example: 10-Second Audio File

| Step | Description | Input Shape | Output Shape | Notes |
|------|-------------|-------------|--------------|-------|
| 1 | Raw Audio | `(160000,)` | `(1, 160000)` | Add batch dimension |
| 2 | Filterbank | `(1, 160000)` | `(1, 1000, 128)` | 1000 frames × 128 mel bins |
| 3 | Add Channel | `(1, 1000, 128)` | `(1, 1, 1000, 128)` | For Conv2D processing |
| 4 | Patch Embed | `(1, 1, 1000, 128)` | `(1, 512, 62, 8)` | **CRITICAL STEP** |
| 5 | Flatten Spatial | `(1, 512, 62, 8)` | `(1, 512, 496)` | 62×8=496 patches |
| 6 | Transpose | `(1, 512, 496)` | `(1, 496, 512)` | Sequence format |
| 7 | Layer Norm | `(1, 496, 512)` | `(1, 496, 512)` | Normalize features |
| 8 | Projection | `(1, 496, 512)` | `(1, 496, 768)` | Linear: 512→768 |
| 9 | Transformer | `(1, 496, 768)` | `(1, 496, 768)` | 12 layers of attention |
| 10 | Pooling | `(1, 496, 768)` | `(1, 768)` | Sequence → Fixed size |
| 11 | Normalize | `(1, 768)` | `(1, 768)` | L2 normalization |

---

## Patch Embedding Deep Dive

### The Critical Transformation

The patch embedding is the most important step that determines the final sequence length:

```python
# Conv2D Parameters
kernel_size = 16    # 16×16 patches
stride = 16         # Non-overlapping patches
in_channels = 1     # Mel spectrogram
out_channels = 512  # Embedding dimension
```

### Dimension Calculation

For filterbank features with shape `(batch, 1, time_frames, 128)`:

```python
# Time dimension patches
time_patches = time_frames // 16    # Floor division!
# Frequency dimension patches
freq_patches = 128 // 16 = 8        # Exactly 8

# Output shape: (batch, 512, time_patches, freq_patches)
```

### What Each Patch Represents

- **Temporal coverage**: 16 × 10ms = **160ms of audio**
- **Frequency coverage**: 16 mel bins (varies by frequency, ~125Hz to 4kHz range)
- **Each patch** becomes a **512-dimensional embedding vector**

### Patch Visualization

```
Filterbank (1000 × 128):
┌─────────────────────────────────────┐
│ ████████████████████████████████████ │ ← High frequencies
│ ████████████████████████████████████ │
│ ████████████████████████████████████ │   128 mel bins
│ ████████████████████████████████████ │
│ ████████████████████████████████████ │ ← Low frequencies
└─────────────────────────────────────┘
  ← 1000 time frames (10 seconds) →

After 16×16 Patching (62 × 8):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ P │ P │ P │ P │ P │ P │ P │ P │ ← 8 frequency patches
├───┼───┼───┼───┼───┼───┼───┼───┤
│ P │ P │ P │ P │ P │ P │ P │ P │
├───┼───┼───┼───┼───┼───┼───┼───┤
│ P │ P │ P │ P │ P │ P │ P │ P │
└───┴───┴───┴───┴───┴───┴───┴───┘
       ← 62 time patches →
```

---

## Variable Length Audio Handling

### Batch Processing Strategy

BEATs handles variable-length audio through **batch-level padding**:

```python
def extract_from_files(audio_paths, batch_size=16):
    for batch in batches:
        # 1. Load all audio files in batch
        audio_list = [load_audio(path) for path in batch_paths]

        # 2. Find maximum length in THIS batch
        max_length = max(len(audio) for audio in audio_list)

        # 3. Pad shorter files with zeros
        padded_batch = []
        for audio in audio_list:
            if len(audio) < max_length:
                padded = pad_with_zeros(audio, max_length)
            padded_batch.append(padded)

        # 4. Process uniform batch
        features = extract_features(stack(padded_batch))
```

### Intelligent Pooling

The pooling mechanisms handle padding masks to avoid contamination:

```python
# Mean pooling - excludes padded positions
if padding_mask is not None:
    valid_lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
    pooled = features.sum(dim=1) / valid_lengths.clamp(min=1)

# Max pooling - sets padded positions to -∞
features.masked_fill(padding_mask.unsqueeze(-1), -1e9)

# Last token - finds actual last non-padded position
valid_lengths = (~padding_mask).sum(dim=1) - 1
pooled = features[torch.arange(batch_size), valid_lengths]
```

---

## Divisibility Requirements

### The Core Problem

❌ **Issue**: Conv2D with `stride=kernel_size` performs **floor division**, truncating remainder frames.

### Impact Analysis

| Audio Duration | Time Frames | Patches | Lost Frames | Lost Audio |
|---------------|-------------|---------|-------------|------------|
| 0.5s | ~50 | 3 | 2 | ~20ms |
| 1.0s | ~100 | 6 | 4 | ~40ms |
| 2.5s | ~250 | 15 | 10 | ~100ms |
| 10.0s | ~1000 | 62 | 8 | ~80ms |

### Truncation Examples

```python
# Example calculations
def calculate_truncation(duration_seconds):
    frames = int(duration_seconds / 0.01)  # 10ms hop
    patches = frames // 16
    lost_frames = frames % 16
    lost_audio_ms = lost_frames * 10

    return {
        'frames': frames,
        'patches': patches,
        'lost_frames': lost_frames,
        'lost_audio_ms': lost_audio_ms
    }

# 10-second audio
result = calculate_truncation(10.0)
# {'frames': 1000, 'patches': 62, 'lost_frames': 8, 'lost_audio_ms': 80}
```

### Solutions

**Option 1: Accept Natural Truncation (Recommended)**
- Loss is typically <1% of total audio
- Lost frames usually contain silence or fade-out
- Model was trained with this behavior

**Option 2: Pre-pad to Ensure Divisibility**
```python
def pad_for_perfect_patches(audio, target_sr=16000):
    duration = len(audio) / target_sr
    current_frames = int(duration / 0.01)
    needed_frames = ((current_frames + 15) // 16) * 16
    needed_samples = int((needed_frames * 0.01) * target_sr)

    if needed_samples > len(audio):
        padding = needed_samples - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    return audio
```

---

## Memory and Computational Analysis

### Memory Usage (10-second audio)

| Stage | Shape | Memory | Notes |
|-------|--------|---------|-------|
| Raw Audio | `(1, 160000)` | 640 KB | Float32 |
| Filterbank | `(1, 1000, 128)` | 512 KB | Mel features |
| Patch Embed | `(1, 512, 62, 8)` | 1.01 MB | After convolution |
| Transformer | `(1, 496, 768)` | 1.52 MB | Sequence processing |
| Final Features | `(1, 768)` | 3 KB | After pooling |

### Computational Complexity

```python
# Attention complexity per layer
seq_len = 496
embed_dim = 768
attention_ops = seq_len² × embed_dim = 496² × 768 ≈ 188M ops

# Total transformer computation
total_layers = 12
total_attention = 12 × 188M ≈ 2.26B operations

# Feed-forward networks (4× expansion)
ffn_ops = 12 × 496 × 768 × 3072 ≈ 14.1B operations
```

---

## Practical Examples

### Single File Processing

```python
extractor = BEATsFeatureExtractor()

# 10-second audio file
features = extractor.extract_from_file("10sec_audio.wav")
print(f"Features shape: {features.shape}")  # (768,)

# Dimension flow:
# Raw: 160,000 samples → Filterbank: 1000×128 → Patches: 62×8 → Features: 768
```

### Batch Processing with Mixed Lengths

```python
files = [
    "short_2sec.wav",    # → 32,000 samples
    "medium_5sec.wav",   # → 80,000 samples
    "long_10sec.wav",    # → 160,000 samples
]

features = extractor.extract_from_files(files)
print(f"Batch features: {features.shape}")  # (3, 768)

# Internal processing:
# 1. All padded to 160,000 samples (longest in batch)
# 2. Filterbank: all become 1000×128
# 3. Patches: all become 62×8 = 496 sequence length
# 4. Features: all become 768-dimensional
```

### Different Pooling Methods

```python
# Mean pooling (default) - averages sequence
extractor = BEATsFeatureExtractor(pooling="mean")
mean_features = extractor.extract_from_file("audio.wav")

# Max pooling - takes maximum across sequence
extractor = BEATsFeatureExtractor(pooling="max")
max_features = extractor.extract_from_file("audio.wav")

# First token (CLS-like) - uses first patch
extractor = BEATsFeatureExtractor(pooling="first")
cls_features = extractor.extract_from_file("audio.wav")

# No pooling - returns full sequence
extractor = BEATsFeatureExtractor(pooling="none")
sequence_features = extractor.extract_features(audio)  # Shape: (1, 496, 768)
```

---

## Key Insights and Gotchas

### ✅ **Key Insights**

1. **Patch Size Determines Everything**: The 16×16 patch size creates the fundamental time-frequency trade-off
2. **Sequence Length Formula**: `seq_len = (time_frames // 16) × (128 // 16) = (time_frames // 16) × 8`
3. **Temporal Resolution**: Each token represents 160ms of audio across 16 mel bins
4. **Batch Adaptation**: Each batch adapts padding to its longest sample (memory efficient)
5. **Information Preservation**: Despite truncation, rich spectro-temporal patterns are preserved

### ⚠️ **Important Gotchas**

1. **Silent Truncation**: Audio is truncated without warning if not divisible by 16 frames
2. **Batch Memory Scaling**: Batch memory scales with the longest audio in that batch
3. **Fixed Sample Rate**: Must be 16kHz - other rates require resampling
4. **Pooling Choice Matters**: Different pooling methods can give very different features
5. **Padding Contamination**: Improper pooling can be contaminated by padding zeros

### 🔧 **Best Practices**

1. **Monitor Truncation**: Check if significant audio is being lost for your use case
2. **Batch Homogeneous Lengths**: Group similar-length audio for memory efficiency
3. **Choose Pooling Wisely**:
   - `mean`: General-purpose, robust
   - `max`: Emphasizes prominent features
   - `first`: Fast, but may miss temporal context
   - `last`: Good for audio with important endings
4. **Validate Feature Quality**: Test with known audio samples to verify expected behavior
5. **Consider Pre-padding**: For critical applications where every millisecond matters

---

## Conclusion

BEATs transforms audio through a sophisticated patch-based approach that balances computational efficiency with information preservation. Understanding the dimension flow is crucial for:

- **Memory planning** in batch processing
- **Choosing appropriate audio lengths** for your use case
- **Selecting optimal pooling strategies**
- **Debugging unexpected behavior**
- **Optimizing processing pipelines**

The patch-based architecture enables BEATs to capture both local acoustic patterns (within patches) and global temporal relationships (via transformer attention), making it highly effective for a wide range of audio understanding tasks.
