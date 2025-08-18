"""PyTorch Lightning model wrapper for BEATs."""

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchmetrics import Accuracy, F1Score
import pytorch_lightning as pl

from .config import Config

# Import BEATs - this should point to your BEATs implementation
try:
    # Try relative import for installed package
    from ..BEATs.BEATs import BEATs, BEATsConfig
except ImportError:
    # Fallback for development/direct execution
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from BEATs.BEATs import BEATs, BEATsConfig


class BEATsLightningModule(pl.LightningModule):
    """PyTorch Lightning module for BEATs classification."""

    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.num_classes = num_classes

        # Load BEATs model
        self._setup_model()

        # Setup metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _setup_model(self):
        """Setup BEATs model and classifier."""
        # Load pretrained BEATs
        model_path = self.config.model.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BEATs model not found at: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

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
            "input_patch_size": 16,
            "layer_wise_gradient_decay_ratio": 1.0,
            "embed_dim": 512,
        }

        # Merge default config with checkpoint config (checkpoint values take precedence)
        complete_cfg = {**default_config, **checkpoint_cfg}

        # Add training-specific parameters
        cfg = BEATsConfig(
            {
                **complete_cfg,
                "predictor_class": self.num_classes,
                "finetuned_model": False,
            }
        )

        print(f"Loading BEATs model from: {model_path}")
        print(
            f"Model config: embed_dim={cfg.encoder_embed_dim}, patch_size={cfg.input_patch_size}"
        )

        # Initialize BEATs backbone
        self.backbone = BEATs(cfg)

        # Load state dict with error handling for missing/extra keys
        try:
            self.backbone.load_state_dict(checkpoint["model"], strict=True)
            print("✅ Loaded checkpoint with exact match")
        except RuntimeError as e:
            if "Unexpected key(s)" in str(e) or "Missing key(s)" in str(e):
                print(f"⚠️ Checkpoint has architecture differences: {e}")
                print("Attempting to load with strict=False...")

                # Load with strict=False to ignore extra/missing keys
                missing_keys, unexpected_keys = self.backbone.load_state_dict(
                    checkpoint["model"], strict=False
                )

                if missing_keys:
                    print(
                        f"Missing keys (will use random initialization): {missing_keys}"
                    )
                if unexpected_keys:
                    print(f"Unexpected keys (will be ignored): {unexpected_keys}")

                print("✅ Loaded checkpoint with partial match")
            else:
                # Re-raise other types of errors
                raise e

        # Freeze backbone if requested
        if self.config.model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create classifier head
        if self.config.model.use_custom_head and self.config.model.hidden_dims:
            # Multi-layer head
            layers = []
            input_dim = cfg.encoder_embed_dim

            for hidden_dim in self.config.model.hidden_dims:
                layers.extend(
                    [
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU()
                        if self.config.model.activation == "relu"
                        else nn.GELU(),
                        nn.Dropout(self.config.model.dropout_rate),
                    ]
                )
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, self.num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            # Simple linear classifier
            self.classifier = nn.Linear(cfg.encoder_embed_dim, self.num_classes)

    def forward(self, x, padding_mask=None):
        """Forward pass."""
        # Extract features from BEATs
        if padding_mask is not None:
            features, _ = self.backbone.extract_features(x, padding_mask)
        else:
            features, _ = self.backbone.extract_features(x)

        # Global average pooling
        features = features.mean(dim=1)

        # Classification
        logits = self.classifier(features)

        return logits

    def _shared_step(self, batch, stage: str):
        """Shared step for train/val/test."""
        x, padding_mask, targets = batch
        logits = self.forward(x, padding_mask)
        loss = self.criterion(logits, targets)

        # Get metrics
        if stage == "train":
            acc = self.train_accuracy(logits, targets)
            f1 = self.train_f1(logits, targets)
        elif stage == "val":
            acc = self.val_accuracy(logits, targets)
            f1 = self.val_f1(logits, targets)
        else:  # test
            acc = self.test_accuracy(logits, targets)
            f1 = self.test_f1(logits, targets)

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_accuracy", acc, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Determine which parameters to optimize
        if self.config.model.fine_tune_backbone:
            params = self.parameters()
        else:
            params = self.classifier.parameters()

        # Setup optimizer
        if self.config.training.optimizer.lower() == "adamw":
            optimizer = AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer.lower() == "adam":
            optimizer = Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer.lower() == "sgd":
            optimizer = SGD(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )

        # Setup scheduler
        if self.config.training.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.config.training.max_epochs
            )
            return [optimizer], [scheduler]
        elif self.config.training.scheduler.lower() == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            return [optimizer], [scheduler]
        else:
            return optimizer
