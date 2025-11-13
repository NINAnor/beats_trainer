#!/usr/bin/env python3
"""
Example demonstrating pre-split dataset support in BEATs Trainer.

This script shows how to use datasets that are already split into 
train/validation/test sets, avoiding automatic splitting.
"""

from pathlib import Path
from src.beats_trainer import BEATsTrainer
from src.beats_trainer.config import Config


def create_example_split_dataset():
    """Create a minimal example dataset for demonstration."""
    base_dir = Path("./example_split_dataset")
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        for class_name in ["class_a", "class_b"]:
            (base_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created example dataset structure at: {base_dir}")
    print("   Note: Add your own audio files to each class directory")
    print(f"   Structure:")
    print(f"   {base_dir}/")
    print(f"   ├── train/")
    print(f"   │   ├── class_a/  # Add training files here")
    print(f"   │   └── class_b/  # Add training files here") 
    print(f"   ├── val/")
    print(f"   │   ├── class_a/  # Add validation files here")
    print(f"   │   └── class_b/  # Add validation files here")
    print(f"   └── test/")
    print(f"       ├── class_a/  # Add test files here")
    print(f"       └── class_b/  # Add test files here")
    
    return base_dir


def demo_split_directories():
    """Demonstrate training with pre-split directories."""
    print("🚀 Demo: Training with Pre-Split Directories")
    print("=" * 60)
    
    # You can use any dataset with train/val/test directory structure
    data_dir = "./example_split_dataset"  # or path to your pre-split dataset
    
    # Configuration
    config = Config()
    config.model.model_path = "openbeats"
    config.model.freeze_backbone = False
    config.training.max_epochs = 5
    config.training.learning_rate = 5e-5
    
    try:
        # Create trainer from pre-split directories
        trainer = BEATsTrainer.from_split_directories(
            data_dir=data_dir,
            config=config,
            train_dir="train",    # Default: "train"
            val_dir="val",        # Default: "val"  
            test_dir="test"       # Default: "test"
        )
        
        print("✅ Trainer created successfully!")
        print("   - Uses pre-existing train/val/test splits")
        print("   - No automatic splitting performed")
        print("   - Ready to train!")
        
        # Start training
        # trainer.train()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "not found" in str(e):
            print("💡 Tip: Create the example dataset first")


def demo_split_csvs():
    """Demonstrate training with pre-split CSV files."""
    print("\n🚀 Demo: Training with Pre-Split CSV Files")
    print("=" * 60)
    
    # You would have separate CSV files for each split
    data_dir = "./audio_files"  # Directory containing all audio files
    
    config = Config()
    config.model.model_path = "openbeats"
    config.training.max_epochs = 5
    
    try:
        # Create trainer from pre-split CSV files
        trainer = BEATsTrainer.from_split_csvs(
            data_dir=data_dir,
            train_csv="train_split.csv",      # Required
            val_csv="validation_split.csv",  # Optional
            test_csv="test_split.csv",       # Optional
            audio_column="filename",         # Column with audio file names
            label_column="category"          # Column with class labels
        )
        
        print("✅ Trainer created successfully!")
        print("   - Uses pre-defined CSV splits")
        print("   - Each CSV defines a different split")
        print("   - Ready to train!")
        
        # Start training
        # trainer.train()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Tip: Make sure CSV files exist with the correct columns")


def demo_custom_directories():
    """Demonstrate custom directory names."""
    print("\n🚀 Demo: Custom Directory Names")
    print("=" * 60)
    
    # Some datasets use different naming conventions
    data_dir = "./my_dataset"
    
    try:
        trainer = BEATsTrainer.from_split_directories(
            data_dir=data_dir,
            train_dir="training",     # Custom name
            val_dir="validation",     # Custom name
            test_dir="testing"        # Custom name
        )
        
        print("✅ Supports custom directory names!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🎵 BEATs Trainer - Pre-Split Dataset Examples")
    print("=" * 60)
    
    # Create example dataset structure
    create_example_split_dataset()
    print()
    
    # Run demonstrations
    demo_split_directories()
    demo_split_csvs() 
    demo_custom_directories()
    
    print("\n📚 Benefits of Pre-Split Datasets:")
    print("   ✅ Reproducible experiments")
    print("   ✅ Consistent evaluation")  
    print("   ✅ No data leakage between splits")
    print("   ✅ Works with standard ML dataset formats")
    print("   ✅ Supports custom split ratios")
    
    print("\n💡 Use Cases:")
    print("   📊 Academic datasets with provided splits")
    print("   🔬 Research requiring exact train/test splits")
    print("   🏢 Production datasets with business logic splits")
    print("   🎯 Benchmarking against published results")