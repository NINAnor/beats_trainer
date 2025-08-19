"""BEATs Feature Extractor for extracting embeddings from audio."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
import librosa

# Import BEATs
try:
    # Try relative import for installed package
    from ..BEATs.BEATs import BEATs, BEATsConfig
except ImportError:
    # Fallback for development/direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from BEATs.BEATs import BEATs, BEATsConfig

# Import checkpoint utilities
from .checkpoint_utils import ensure_checkpoint


class BEATsFeatureExtractor:
    """
    Standalone feature extractor for BEATs model.

    This class provides functionality to extract high-quality audio features using
    pretrained or custom-trained BEATs models. It operates independently and does
    not require the training pipeline.

    🎯 **Purpose**: Audio feature extraction and embedding generation
    🔗 **Independence**: Fully standalone - no dependency on BEATsTrainer
    📦 **Models**: Automatically downloads pretrained models from Hugging Face Hub

    Examples:
        # Basic feature extraction (auto-downloads pretrained model)
        extractor = BEATsFeatureExtractor()
        features = extractor.extract_from_file("audio.wav")

        # Specify custom model path
        extractor = BEATsFeatureExtractor("path/to/custom_model.pt")
        features = extractor.extract_from_file("audio.wav")

        # Batch processing
        features = extractor.extract_from_files(["audio1.wav", "audio2.wav"])

        # Extract from numpy array
        features = extractor.extract_from_array(audio_array, sample_rate=16000)

        # Use with trained model from BEATsTrainer
        trainer = BEATsTrainer.from_directory("/path/to/data")
        trainer.train()
        extractor = trainer.get_feature_extractor()  # Convenience method
    """

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        device: Optional[str] = None,
        layer: int = -1,
        pooling: str = "mean",
    ):
        """
        Initialize BEATs feature extractor.

        Args:
            model_path: Path to pretrained BEATs model. If None, will download default model.
            device: Device to run inference on ("cpu", "cuda", "auto")
            layer: Which transformer layer to extract features from (-1 for last layer)
            pooling: How to pool sequence features ("mean", "max", "first", "last", "none")
        """
        # Handle automatic checkpoint download/finding
        if model_path is None:
            print(
                "No model path specified, attempting to find or download BEATs checkpoint..."
            )
            model_path = ensure_checkpoint()

        self.model_path = Path(model_path)
        self.layer = layer
        self.pooling = pooling.lower()

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the pretrained BEATs model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")

        # Handle incomplete checkpoint configurations (like OpenBEATs)
        checkpoint_cfg = checkpoint["cfg"]

        # Define default configuration values for missing parameters
        default_config = {
            "encoder_layers": 12,
            "encoder_embed_dim": 768,
            "encoder_ffn_embed_dim": 3072,
            "encoder_attention_heads": 12,
            "activation_fn": "gelu",
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
            "encoder_layerdrop": 0.0,
            "dropout_input": 0.1,
            "layer_norm_first": False,
            "conv_bias": False,
            "conv_pos": 128,
            "conv_pos_groups": 16,
            "relative_position_embedding": True,
            "num_buckets": 320,
            "max_distance": 800,
            "gru_rel_pos": True,
            "deep_norm": True,
            "input_patch_size": 16,  # Critical for OpenBEATs compatibility
            "layer_wise_gradient_decay_ratio": 1.0,
            "embed_dim": 512,
        }

        # Merge default config with checkpoint config (checkpoint values take precedence)
        complete_cfg = {**default_config, **checkpoint_cfg}

        # Create config for feature extraction (no classifier)
        cfg = BEATsConfig(
            {
                **complete_cfg,
                "finetuned_model": False,  # Remove classifier head
            }
        )

        # Initialize model
        self.model = BEATs(cfg)

        # Use the improved loading method
        try:
            self.model.reload_pretrained_parameters(state_dict=checkpoint["model"])
        except Exception as e:
            print(f"Warning: Failed to use reload_pretrained_parameters: {e}")
            print("Falling back to standard loading...")
            self.model.load_state_dict(checkpoint["model"], strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Store config for reference
        self.config = cfg

        print(f"Loaded BEATs model with {cfg.encoder_layers} layers")
        print(f"Feature dimension: {cfg.encoder_embed_dim}")
        print(f"Device: {self.device}")

    def _load_audio(
        self, audio_path: Union[str, Path], target_sr: int = 16000
    ) -> torch.Tensor:
        """Load audio file and convert to tensor."""
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return torch.tensor(audio, dtype=torch.float32)

    # Alias for test compatibility
    def _load_audio_file(
        self, audio_path: Union[str, Path], target_sr: int = 16000
    ) -> torch.Tensor:
        """Alias for _load_audio for test compatibility."""
        return self._load_audio(audio_path, target_sr)

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocess audio tensor for BEATs input."""
        # Convert numpy array to tensor if needed
        if not isinstance(audio, torch.Tensor):
            import numpy as np

            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            else:
                audio = torch.tensor(audio, dtype=torch.float32)

        # Ensure audio is 2D (batch, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio.to(self.device)

    def _pool_features(
        self, features: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply pooling to sequence features."""
        if self.pooling == "none":
            return features

        # Handle padding mask for proper pooling
        if padding_mask is not None:
            # Set padded positions to large negative values for max pooling,
            # or zero for mean pooling
            if self.pooling == "max":
                features = features.masked_fill(padding_mask.unsqueeze(-1), -1e9)
            elif self.pooling == "mean":
                features = features.masked_fill(padding_mask.unsqueeze(-1), 0)

        if self.pooling == "mean":
            if padding_mask is not None:
                # Proper average excluding padded positions
                valid_lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
                pooled = features.sum(dim=1) / valid_lengths.clamp(min=1)
            else:
                pooled = features.mean(dim=1)
        elif self.pooling == "max":
            pooled, _ = features.max(dim=1)
        elif self.pooling in ("first", "cls"):  # cls is alias for first
            pooled = features[:, 0]  # CLS token equivalent
        elif self.pooling == "last":
            if padding_mask is not None:
                # Get last non-padded position for each sequence
                valid_lengths = (~padding_mask).sum(dim=1) - 1
                pooled = features[torch.arange(features.size(0)), valid_lengths]
            else:
                pooled = features[:, -1]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled

    @torch.no_grad()
    def extract_features(
        self,
        audio: torch.Tensor,
        return_all_layers: bool = False,
        normalize: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from audio tensor.

        Args:
            audio: Input audio tensor (batch_size, samples) or (samples,)
            return_all_layers: Whether to return features from all transformer layers
            normalize: Whether to normalize features

        Returns:
            Feature tensor or dict of features from all layers
        """
        # Preprocess
        audio = self._preprocess_audio(audio)

        # Extract features through BEATs
        if hasattr(self.model, "encoder") and hasattr(
            self.model.encoder, "extract_layer_results"
        ):
            # Modified forward pass to get intermediate layers
            features, padding_mask = self.model.extract_features(audio)
            # For now, just return final layer features
            # TODO: Implement layer-wise extraction if needed
        else:
            features, padding_mask = self.model.extract_features(audio)

        if return_all_layers:
            # TODO: Implement extraction from all layers
            return {"final_layer": self._pool_features(features, padding_mask)}

        # Apply pooling
        pooled_features = self._pool_features(features, padding_mask)

        # Normalize if requested
        if normalize:
            pooled_features = nn.functional.normalize(pooled_features, p=2, dim=-1)

        # Convert to numpy array for consistency with the API
        return pooled_features.cpu().numpy()

    def extract_from_file(self, audio_path: Union[str, Path], **kwargs) -> np.ndarray:
        """
        Extract features from a single audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for extract_features

        Returns:
            Feature array (feature_dim,)
        """
        audio = self._load_audio(audio_path)
        features = self.extract_features(audio, **kwargs)
        return features.squeeze()  # Already numpy array, just squeeze

    def extract_from_files(
        self, audio_paths: List[Union[str, Path]], batch_size: int = 16, **kwargs
    ) -> np.ndarray:
        """
        Extract features from multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            batch_size: Batch size for processing
            **kwargs: Additional arguments for extract_features

        Returns:
            Feature array (num_files, feature_dim)
        """
        all_features = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i : i + batch_size]
            batch_audio = []

            # Load batch of audio files
            max_length = 0
            for path in batch_paths:
                audio = self._load_audio(path)
                batch_audio.append(audio)
                max_length = max(max_length, len(audio))

            # Pad to same length
            padded_batch = []
            for audio in batch_audio:
                if len(audio) < max_length:
                    padded = torch.cat([audio, torch.zeros(max_length - len(audio))])
                else:
                    padded = audio
                padded_batch.append(padded)

            batch_tensor = torch.stack(padded_batch)
            batch_features = self.extract_features(batch_tensor, **kwargs)
            all_features.append(batch_features)  # Already numpy array

        return np.concatenate(all_features, axis=0)

    def extract_from_array(
        self, audio: np.ndarray, sample_rate: Optional[int] = None, **kwargs
    ) -> np.ndarray:
        """
        Extract features from numpy audio array.

        Args:
            audio: Audio array (samples,) or (batch_size, samples)
            sample_rate: Sample rate of audio (for resampling if needed)
            **kwargs: Additional arguments for extract_features

        Returns:
            Feature array
        """
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Resample if necessary
        if sample_rate is not None and sample_rate != 16000:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            if audio_tensor.dim() == 1:
                audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
            else:
                audio_tensor = resampler(audio_tensor)

        features = self.extract_features(audio_tensor, **kwargs)
        return features  # Already numpy array from extract_features

    def extract_from_batch(
        self, batch_audio: Union[torch.Tensor, np.ndarray, list], **kwargs
    ) -> np.ndarray:
        """
        Extract features from a batch of audio arrays.

        Args:
            batch_audio: Batch of audio data (batch_size, samples) or list of arrays
            **kwargs: Additional arguments passed to extract_features

        Returns:
            Batch of extracted features (batch_size, feature_dim)
        """
        if isinstance(batch_audio, list):
            # Handle list of audio arrays
            batch_features = []
            for audio in batch_audio:
                features = self.extract_from_array(audio, **kwargs)
                # Ensure features are squeezed to remove singleton dimensions
                # If shape is (1, feature_dim), squeeze to (feature_dim,)
                if features.ndim == 2 and features.shape[0] == 1:
                    features = features.squeeze(0)
                batch_features.append(features)
            return np.stack(batch_features)

        # Handle tensor or numpy array
        if isinstance(batch_audio, np.ndarray):
            batch_audio = torch.from_numpy(batch_audio).float()

        features = self.extract_features(batch_audio, **kwargs)
        # Remove extra dimensions if present
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.squeeze(1)
        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        return self.config.encoder_embed_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "feature_dim": self.get_feature_dim(),
            "num_layers": self.config.encoder_layers,
            "attention_heads": self.config.encoder_attention_heads,
            "pooling_method": self.pooling,
            "device": str(self.device),
        }
