#!/usr/bin/env python3
"""
Script to upload the README to Hugging Face Hub
"""

from huggingface_hub import HfApi


def upload_readme_to_hf():
    repo_id = "Bencr/beats-checkpoints"
    api = HfApi()

    readme_content = """# BEATs Checkpoints

This dataset contains pre-trained BEATs (Bidirectional Encoder representation from Audio Transformers) model checkpoints.

## Usage

These checkpoints are automatically downloaded by the `beats-trainer` Python package:

```python
from beats_trainer import BEATsFeatureExtractor

# Automatically downloads and uses checkpoints
extractor = BEATsFeatureExtractor()
```

## Source

These checkpoints are distributed as part of the [beats-trainer](https://github.com/Bencr/beats-trainer) project.

## License

BEATs_iter3_plus_AS2M.pt model weights are provided by Microsoft Research under their respective license terms (MIT). These can be found in their [GitHub repository](https://github.com/microsoft/unilm/tree/master/beats)

OpenBEATs-Base-i3.pt model weights are provided by the OpenBEATs project under their respective license terms. More details can be found in the [OpenBEATs paper](https://arxiv.org/pdf/2507.14129))
"""

    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✅ Uploaded README.md")
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")


if __name__ == "__main__":
    upload_readme_to_hf()
