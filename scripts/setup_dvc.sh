#!/bin/bash
# Setup script for DVC with Hugging Face Hub

set -e

echo "🚀 Setting up DVC with Hugging Face Hub for BEATs checkpoints..."

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "Installing DVC..."
    uv add "dvc[webdav]>=3.0.0" huggingface-hub>=0.17.0
fi

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init --no-scm
fi

# Add Hugging Face Hub remote
echo "Configuring Hugging Face Hub remote..."
dvc remote add -d huggingface https://huggingface.co/datasets/Bencr/beats-checkpoints
dvc remote modify huggingface url https://huggingface.co/datasets/Bencr/beats-checkpoints

# Check if checkpoints directory exists and has files
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
    echo "Found checkpoints directory with files. Adding to DVC tracking..."
    
    # Track the checkpoints directory
    dvc add checkpoints/
    
    echo "Checkpoints are now tracked by DVC!"
    echo ""
    echo "Next steps:"
    echo "1. Create a Hugging Face dataset repository: https://huggingface.co/new-dataset"
    echo "2. Set your Hugging Face token: export HF_TOKEN=your_token"
    echo "3. Push checkpoints to Hugging Face Hub: dvc push"
    echo ""
else
    echo "⚠️  No checkpoints found in ./checkpoints/ directory"
    echo "Place your BEATs checkpoint files in the checkpoints/ directory first"
fi

echo "✅ DVC setup complete!"
