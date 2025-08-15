# DVC + Hugging Face Hub for BEATs Checkpoints

This document explains how to use DVC (Data Version Control) with Hugging Face Hub to manage and distribute BEATs model checkpoints.

## Overview

Instead of relying on OneDrive links that require access tokens, we use:
- **DVC**: For versioning and tracking large model files
- **Hugging Face Hub**: As a public storage backend for the checkpoints
- **Automatic Downloads**: Users can download checkpoints seamlessly

## Setup

### 1. Install Dependencies

```bash
pip install "dvc[webdav]>=3.0.0" huggingface-hub>=0.17.0
```

### 2. Run Setup Script

```bash
./scripts/setup_dvc.sh
```

### 3. Create Hugging Face Dataset Repository

1. Go to https://huggingface.co/new-dataset
2. Create a public dataset repository named `beats-checkpoints`
3. Repository should be: `ninanor/beats-checkpoints`

### 4. Authenticate with Hugging Face

```bash
# Option 1: Using environment variable
export HF_TOKEN=your_huggingface_token

# Option 2: Using HF CLI
pip install huggingface_hub[cli]
huggingface-cli login
```

## Uploading Checkpoints

### 1. Add Checkpoints to DVC

```bash
# Place your .pt files in the checkpoints/ directory
# Then track with DVC
dvc add checkpoints/

# Commit the .dvc files
git add checkpoints.dvc .gitignore
git commit -m "Add BEATs checkpoints to DVC tracking"
```

### 2. Push to Hugging Face Hub

```bash
# Push data to Hugging Face Hub
dvc push
```

### 3. Update Repository

```bash
git push origin main
```

## For Users: Automatic Download

Users don't need to know about DVC! The checkpoints are automatically downloaded:

```python
from beats_trainer import BEATsFeatureExtractor

# This will automatically download checkpoints from HF Hub
extractor = BEATsFeatureExtractor()
```

## Available Checkpoints

Current checkpoints available through this system:

- **BEATs_iter3_plus_AS2M.pt**: Main BEATs model trained on AudioSet (110MB)
- **openbeats.pt**: OpenBEATs checkpoint (110MB)

## Manual DVC Operations

### Pull Latest Checkpoints

```bash
dvc pull
```

### Check Status

```bash
dvc status
dvc data_status
```

### Remove from Cache

```bash
dvc cache dir
dvc gc -w  # Remove unused cache files
```

## Repository Structure

```
beats-trainer/
├── checkpoints/           # Local checkpoint files (gitignored)
├── checkpoints.dvc        # DVC tracking file (committed)
├── .dvc/
│   ├── config            # DVC configuration
│   └── cache/            # Local DVC cache (gitignored)  
└── scripts/
    └── setup_dvc.sh      # Setup script
```

## Hugging Face Hub Structure

The Hugging Face dataset repository contains:
```
ninanor/beats-checkpoints/
├── BEATs_iter3_plus_AS2M.pt
├── openbeats.pt
└── README.md
```

## Benefits

✅ **No OneDrive tokens required**: Public access via HF Hub  
✅ **Version control**: DVC tracks changes to model files  
✅ **Automatic downloads**: Seamless user experience  
✅ **Bandwidth efficient**: HF Hub CDN for fast downloads  
✅ **Free**: Both DVC and HF Hub are free for public repos  
✅ **Reliable**: No expired download links  

## Troubleshooting

### DVC Push Fails

```bash
# Check remote configuration
dvc remote list -v

# Test HF Hub access
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

### Download Issues

```bash
# Clear local cache
dvc cache dir  # Shows cache location
rm -rf .dvc/cache/*

# Re-download
dvc pull
```

### Authentication Issues

```bash
# Check HF token
huggingface-cli whoami

# Re-login
huggingface-cli login --token your_token
```
