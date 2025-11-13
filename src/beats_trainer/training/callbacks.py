"""PyTorch Lightning callbacks setup for BEATs training."""

from typing import List

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from ..core.config import Config


def setup_training_callbacks(config: Config) -> List:
    """
    Setup PyTorch Lightning callbacks for training.
    
    Args:
        config: Training configuration
        
    Returns:
        List of PyTorch Lightning callbacks
    """
    callbacks = []

    # Model checkpoint - save best models
    checkpoint_callback = ModelCheckpoint(
        monitor=config.training.monitor_metric,
        mode="max" if "acc" in config.training.monitor_metric else "min",
        save_top_k=config.training.save_top_k,
        save_last=True,
        filename=f"{config.experiment_name}-{{epoch:02d}}-{{{config.training.monitor_metric}:.3f}}",
    )
    callbacks.append(checkpoint_callback)

    # Early stopping - prevent overfitting
    early_stop_callback = EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=config.training.patience,
        mode="max" if "acc" in config.training.monitor_metric else "min",
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitor - track LR changes
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def setup_pytorch_lightning_trainer(config: Config, callbacks: List, log_dir) -> object:
    """
    Setup PyTorch Lightning trainer with proper configuration.
    
    Args:
        config: Training configuration
        callbacks: List of callbacks
        log_dir: Directory for logging
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.loggers import TensorBoardLogger

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=config.experiment_name,
        version=None,  # Auto-increment version
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        devices="auto" if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=config.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.training.log_every_n_steps,
        default_root_dir=str(log_dir),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    return trainer