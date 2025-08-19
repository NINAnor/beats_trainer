# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi

from .backbone import (
    TransformerEncoder,
)

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # patch size of patch embedding
        self.input_size: int = (
            16  # alias for input_patch_size for backward compatibility
        )
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = (
            1.0  # ratio for layer-wise gradient decay
        )
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # relative position embedding
        self.relative_position_embedding: bool = (
            False  # apply relative position embedding
        )
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = (
            1280  # maximum distance for relative position embedding
        )
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)
        # Keep input_size and input_patch_size synchronized
        if "input_size" in cfg and "input_patch_size" not in cfg:
            self.input_patch_size = cfg["input_size"]
        elif "input_patch_size" in cfg and "input_size" not in cfg:
            self.input_size = cfg["input_patch_size"]


class BEATs(nn.Module):
    def __init__(
        self,
        cfg: BEATsConfig,
    ) -> None:
        super().__init__()
        logger.info(f"BEATs Config: {cfg.__dict__}")

        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias,
        )

        # OpenBEATs-specific padding layers for proper padding mask computation
        self.patch_embedding_pad = nn.Conv2d(
            1,
            1,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=False,
        )
        self.raw2fbank_pad = nn.Conv1d(
            1,
            1,
            kernel_size=400,
            stride=160,
            bias=False,
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

        # Initialize model weights
        self.initialize()

    def initialize(self):
        """Initialize model weights using OpenBEATs-compatible initialization."""
        logger.info("BEATs Initialization function called.")

        # Initialize post-extract projection
        if self.post_extract_proj:
            torch.nn.init.xavier_normal_(self.post_extract_proj.weight)
            if self.post_extract_proj.bias is not None:
                torch.nn.init.constant_(self.post_extract_proj.bias, 0)

        # Initialize patch embedding
        torch.nn.init.xavier_normal_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            torch.nn.init.constant_(self.patch_embedding.bias, 0)

        # Initialize OpenBEATs padding layers (these are typically frozen)
        torch.nn.init.xavier_normal_(self.patch_embedding_pad.weight)
        torch.nn.init.xavier_normal_(self.raw2fbank_pad.weight)

        # Freeze the padding layers (as done in the reference implementation)
        for param in self.patch_embedding_pad.parameters():
            param.requires_grad = False
        for param in self.raw2fbank_pad.parameters():
            param.requires_grad = False

    def reload_pretrained_parameters(
        self, checkpoint_path: str = None, state_dict: dict = None
    ):
        """Load pretrained parameters from checkpoint.

        This method properly handles OpenBEATs and standard BEATs checkpoints
        with different architectures and missing/unexpected keys.
        """
        if checkpoint_path is not None:
            logger.info(f"Loading BEATs checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)

        if state_dict is None:
            logger.warning("No state dict provided for loading pretrained parameters")
            return

        # Load state dict with proper error handling
        try:
            load_info = self.load_state_dict(state_dict, strict=False)

            if load_info.missing_keys:
                logger.info(
                    f"Missing keys (will use random initialization): {load_info.missing_keys}"
                )
            if load_info.unexpected_keys:
                logger.info(
                    f"Unexpected keys (ignored from checkpoint): {load_info.unexpected_keys}"
                )

            logger.info("Successfully loaded BEATs pretrained parameters")

        except Exception as e:
            logger.error(f"Error loading pretrained parameters: {e}")
            raise e

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ):
        """Forward pass through the BEATs model."""
        features, _ = self.extract_features(
            source=source,
            padding_mask=padding_mask,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
        )
        return features

    def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2**15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(
                    logits
                )
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)

            return lprobs, padding_mask
        else:
            return x, padding_mask
