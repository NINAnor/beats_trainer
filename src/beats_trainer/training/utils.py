"""Utility functions for BEATs trainer setup and configuration."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl


def configure_deterministic_mode():
    """Configure deterministic behavior for reproducible training."""
    # Set seeds for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Configure deterministic operations
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging_directory(log_dir: Optional[str], experiment_name: str) -> Path:
    """
    Setup and return the logging directory path.
    
    Args:
        log_dir: Base logging directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to the logging directory
    """
    if log_dir is None:
        log_dir = "./logs"
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    return log_path


def validate_training_setup(data_module, model) -> Dict[str, Any]:
    """
    Validate that training setup is correct.
    
    Args:
        data_module: PyTorch Lightning data module
        model: PyTorch Lightning model
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "data_module_valid": data_module is not None,
        "model_valid": model is not None,
        "num_classes": None,
        "dataset_size": None,
    }
    
    if data_module is not None:
        try:
            # Setup data module to get class information
            data_module.setup()
            validation_results["num_classes"] = data_module.num_classes
            
            # Get dataset sizes
            if hasattr(data_module, 'train_dataset'):
                validation_results["dataset_size"] = {
                    "train": len(data_module.train_dataset),
                    "val": len(data_module.val_dataset) if data_module.val_dataset else 0,
                    "test": len(data_module.test_dataset) if data_module.test_dataset else 0,
                }
        except Exception as e:
            validation_results["error"] = str(e)
    
    return validation_results


def print_training_summary(config, validation_results: Dict[str, Any]):
    """
    Print a summary of the training configuration.
    
    Args:
        config: Training configuration
        validation_results: Results from validate_training_setup
    """
    print(f"🚀 BEATs Training Setup")
    print("=" * 50)
    print(f"📋 Experiment: {config.experiment_name}")
    print(f"🎯 Model: {config.model.model_path}")
    print(f"📊 Classes: {validation_results.get('num_classes', 'Unknown')}")
    
    if validation_results.get('dataset_size'):
        sizes = validation_results['dataset_size']
        print(f"📁 Dataset: {sizes['train']} train, {sizes['val']} val, {sizes['test']} test")
    
    print(f"⚙️  Training:")
    print(f"   • Epochs: {config.training.max_epochs}")
    print(f"   • Learning Rate: {config.training.learning_rate}")
    print(f"   • Batch Size: {config.data.batch_size}")
    print(f"   • Freeze Backbone: {config.model.freeze_backbone}")
    print(f"   • Fine-tune Backbone: {config.model.fine_tune_backbone}")
    print()


def get_checkpoint_path(trainer, checkpoint_path: Optional[str] = None) -> str:
    """
    Get the path to the best checkpoint.
    
    Args:
        trainer: PyTorch Lightning trainer
        checkpoint_path: Optional specific checkpoint path
        
    Returns:
        Path to the checkpoint to use
    """
    if checkpoint_path:
        return checkpoint_path
    
    # Get best checkpoint from trainer
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path and os.path.exists(best_path):
            return best_path
    
    # Fallback to last checkpoint
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        last_path = trainer.checkpoint_callback.last_model_path
        if last_path and os.path.exists(last_path):
            return last_path
    
    raise FileNotFoundError("No valid checkpoint found. Train the model first.")


def format_training_results(trainer_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format and clean training results for better readability.
    
    Args:
        trainer_results: Raw results from PyTorch Lightning trainer
        
    Returns:
        Formatted results dictionary
    """
    if not trainer_results:
        return {"status": "no_results"}
    
    formatted = {}
    
    # Extract key metrics
    for key, value in trainer_results.items():
        if "val" in key or "test" in key or "train" in key:
            # Clean up metric names
            clean_key = key.replace("val_", "").replace("test_", "").replace("train_", "")
            formatted[clean_key] = float(value) if isinstance(value, (int, float, torch.Tensor)) else value
    
    return formatted