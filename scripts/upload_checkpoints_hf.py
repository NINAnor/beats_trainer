#!/usr/bin/env python3
"""
Script to upload BEATs checkpoints to Hugging Face Hub
"""

from pathlib import Path
from huggingface_hub import HfApi


def upload_checkpoints():
    """Upload checkpoints to Hugging Face Hub dataset."""

    # Configuration
    repo_id = "Bencr/beats-checkpoints"
    checkpoint_dir = Path("checkpoints")

    # Check if checkpoints exist
    if not checkpoint_dir.exists():
        print("❌ No checkpoints directory found!")
        print("Please place your checkpoint files in ./checkpoints/")
        return

    # Get list of checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        print("❌ No .pt files found in checkpoints directory!")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # Initialize HF API
    api = HfApi()

    # Check authentication
    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception:
        print("❌ Authentication failed!")
        print("Please set your HF token: export HF_TOKEN=your_token")
        print("Or login with: huggingface-cli login")
        return

    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"✅ Repository {repo_id} ready")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return

    # Upload each checkpoint file
    for checkpoint_file in checkpoint_files:
        try:
            print(f"⬆️  Uploading {checkpoint_file.name}...")

            api.upload_file(
                path_or_fileobj=str(checkpoint_file),
                path_in_repo=checkpoint_file.name,
                repo_id=repo_id,
                repo_type="dataset",
            )

            print(f"✅ Uploaded {checkpoint_file.name}")

        except Exception as e:
            print(f"❌ Failed to upload {checkpoint_file.name}: {e}")


if __name__ == "__main__":
    upload_checkpoints()
