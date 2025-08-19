"""Configuration classes for BEATs training and inference."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Data paths
    data_dir: Optional[str] = None
    metadata_file: Optional[str] = None

    # Data format
    audio_column: str = "filename"
    label_column: str = "category"
    audio_extensions: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac"]
    )

    # Audio processing
    sample_rate: int = 16000

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4

    def __post_init__(self):
        """Validate configuration."""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test splits must sum to 1.0")


@dataclass
class ModelConfig:
    """Configuration for the BEATs model."""

    # Model paths and setup
    model_path: Optional[str] = None
    num_classes: Optional[int] = None  # Auto-inferred if None

    # Training from scratch vs pre-trained
    train_from_scratch: bool = False  # If True, don't load pre-trained weights

    # Model architecture (used when training from scratch)
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    input_patch_size: int = 16
    embed_dim: int = 512

    # Training strategy
    freeze_backbone: bool = True
    fine_tune_backbone: bool = False

    # Classification head
    use_custom_head: bool = False
    hidden_dims: Optional[List[int]] = None
    dropout_rate: float = 0.1
    activation: str = "relu"

    def __post_init__(self):
        """Set default model path if none provided and not training from scratch."""
        if not self.train_from_scratch and self.model_path is None:
            self.model_path = "checkpoints/BEATs_iter3_plus_AS2M.pt"

        # When training from scratch, fine_tune_backbone should be True
        if self.train_from_scratch:
            self.fine_tune_backbone = True
            self.freeze_backbone = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Basic training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    patience: int = 10

    # Optimizer and scheduler
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # Loss and metrics
    loss_function: str = "cross_entropy"
    monitor_metric: str = "val_accuracy"

    # Hardware
    gpus: int = 1 if __import__("torch").cuda.is_available() else 0
    precision: int = 32

    # Training reproducibility
    deterministic: bool = False

    # Logging
    log_every_n_steps: int = 50
    save_top_k: int = 3


@dataclass
class Config:
    """Main configuration class combining all components."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment settings
    experiment_name: str = "beats_experiment"
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        main_config = {
            k: v
            for k, v in config_dict.items()
            if k not in ["data", "model", "training"]
        }

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            **main_config,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
