# Training with BEATs

Complete guide for training and fine-tuning BEATs models on custom datasets.

## Quick Start

### Fine-tuning (Recommended)

```python
from beats_trainer import BEATsTrainer

# Train from directory structure
trainer = BEATsTrainer.from_directory("/path/to/dataset")
results = trainer.train()
```

### ESC-50 Example (Auto-Download & Train)

```python
from beats_trainer import BEATsTrainer
from beats_trainer.core.config import Config

config = Config()
config.model.freeze_backbone = False  # Fine-tune entire model
config.training.max_epochs = 50
config.training.learning_rate = 5e-5

# Auto-download, organize, and train
trainer = BEATsTrainer.from_esc50(data_dir="./datasets", config=config)
trainer.train()
```

## Data Formats

### Directory Structure (Recommended)

```
your_dataset/
├── class1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── audio3.mp3
├── class2/
│   ├── audio4.wav
│   └── audio5.flac
└── class3/
    ├── audio6.wav
    └── audio7.wav
```

```python
trainer = BEATsTrainer.from_directory("your_dataset")
results = trainer.train()
```

### Pre-Split Datasets

```
dataset/
├── train/
│   ├── class1/
│   └── class2/
├── val/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

```python
trainer = BEATsTrainer.from_split_directories(
    train_data_dir="dataset/train",
    val_data_dir="dataset/val",
    test_data_dir="dataset/test"
)
trainer.train()
```

### CSV Metadata

**dataset.csv:**
```csv
filename,category
audio/sample1.wav,bird
audio/sample2.mp3,dog
audio/sample3.flac,cat
```

```python
trainer = BEATsTrainer.from_csv(
    csv_path="dataset.csv",
    audio_column="filename",
    label_column="category"
)
trainer.train()
```

### Split CSV Files

```python
trainer = BEATsTrainer.from_split_csvs(
    train_csv="train.csv",
    val_csv="val.csv", 
    test_csv="test.csv",
    audio_column="filename",
    label_column="category"
)
trainer.train()
```

## Training Strategies

### Classification Head Only (Fast)

```python
from beats_trainer.core.config import Config

config = Config()
config.model.freeze_backbone = True  # Only train classifier
config.model.fine_tune_backbone = False

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Full Fine-tuning (Best Performance)

```python
config = Config()
config.model.freeze_backbone = False  # Train entire model
config.model.fine_tune_backbone = True
config.training.learning_rate = 5e-5  # Lower learning rate

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Training From Scratch

```python
from beats_trainer.core.config import Config, ModelConfig, TrainingConfig

config = Config(
    model=ModelConfig(
        train_from_scratch=True,    # Key parameter!
        encoder_layers=12,          # Transformer layers
        encoder_embed_dim=768,      # Hidden dimension
        encoder_attention_heads=12, # Attention heads
        fine_tune_backbone=True,
        freeze_backbone=False
    ),
    training=TrainingConfig(
        learning_rate=1e-3,         # Higher LR for scratch training
        max_epochs=100,             # More epochs needed
        patience=15                 # More patience
    )
)

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

## Advanced Configuration

### Complete Configuration Example

Here's how to configure every aspect of training with a full config:

```python
from beats_trainer.core.config import Config, DataConfig, ModelConfig, TrainingConfig

# Complete configuration with all parameters
config = Config(
    # Experiment settings
    experiment_name="my_audio_classifier",
    seed=42,
    
    # Data configuration
    data=DataConfig(
        # Basic data settings
        batch_size=32,
        num_workers=4,
        sample_rate=16000,
        audio_max_length=10.0,    # Maximum audio length in seconds
        
        # Data splits (only used with from_directory or from_csv)
        train_split=0.7,          # 70% for training
        val_split=0.2,            # 20% for validation  
        test_split=0.1,           # 10% for testing
        
        # Data loading
        shuffle=True,
        drop_last=False,
        pin_memory=True           # Speed up GPU training
    ),
    
    # Model configuration
    model=ModelConfig(
        # Model selection
        model_name="BEATs_iter3_plus_AS2M",  # or "openbeats"
        model_path=None,          # Use None for auto-download
        
        # Architecture settings
        num_classes=10,           # Set automatically from data
        dropout=0.1,
        
        # Training strategy
        freeze_backbone=False,     # True = freeze BEATs, False = fine-tune
        fine_tune_backbone=True,   # Enable backbone training
        train_from_scratch=False,  # True = no pre-trained weights
        
        # For training from scratch only
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_attention_heads=12,
        input_patch_size=16
    ),
    
    # Training configuration  
    training=TrainingConfig(
        # Optimization
        learning_rate=5e-5,       # Lower for fine-tuning, higher for scratch
        optimizer="adamw",        # "adamw", "adam", or "sgd"
        weight_decay=1e-4,
        
        # Learning rate scheduling
        scheduler="cosine",       # "cosine", "step", or "plateau"
        scheduler_step_size=10,   # For step scheduler
        scheduler_gamma=0.1,      # For step scheduler
        
        # Training duration
        max_epochs=50,
        min_epochs=5,
        
        # Early stopping
        patience=10,              # Stop if no improvement for N epochs
        early_stopping_monitor="val_accuracy",
        early_stopping_mode="max", # "max" for accuracy, "min" for loss
        
        # Validation
        val_check_interval=1.0,   # Check validation every epoch
        
        # Hardware settings
        gpus=1,                   # Number of GPUs (0 for CPU)
        precision=32,             # 16 for mixed precision, 32 for full
        
        # Reproducibility
        deterministic=False,      # True for reproducible results (slower)
        
        # Logging
        log_every_n_steps=50,
        save_top_k=3              # Keep top 3 checkpoints
    )
)

# Use the complete configuration
trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Configuration for Different Scenarios

#### Small Dataset (< 1000 samples)

```python
config = Config(
    data=DataConfig(
        batch_size=16,            # Smaller batches
        train_split=0.8,          # More training data
        val_split=0.2,
        test_split=0.0            # No test split for small data
    ),
    model=ModelConfig(
        freeze_backbone=True,     # Only train classifier head
        dropout=0.3               # More dropout to prevent overfitting
    ),
    training=TrainingConfig(
        learning_rate=1e-4,       # Higher learning rate
        max_epochs=100,           # More epochs
        patience=20,              # More patience
        weight_decay=1e-3         # More regularization
    )
)
```

#### Large Dataset (> 10,000 samples)

```python
config = Config(
    data=DataConfig(
        batch_size=64,            # Larger batches
        num_workers=8,            # More data loading workers
        pin_memory=True
    ),
    model=ModelConfig(
        freeze_backbone=False,    # Fine-tune entire model
        dropout=0.1               # Less dropout needed
    ),
    training=TrainingConfig(
        learning_rate=1e-5,       # Lower learning rate
        max_epochs=30,            # Fewer epochs needed
        patience=5,               # Less patience needed
        weight_decay=1e-5,        # Less regularization
        precision=16              # Mixed precision for speed
    )
)
```

#### GPU Optimization

```python
config = Config(
    data=DataConfig(
        batch_size=64,            # Larger batch for GPU
        num_workers=8,            # Match CPU cores
        pin_memory=True,          # Faster GPU transfer
        drop_last=True            # Consistent batch sizes
    ),
    training=TrainingConfig(
        precision=16,             # Mixed precision training
        gpus=1,                   # Use GPU
        deterministic=False       # Faster training
    )
)
```

### YAML Configuration

You can also save configurations as YAML files:

**config.yaml:**
```yaml
experiment_name: "audio_classifier_v1"
seed: 42

data:
  batch_size: 32
  sample_rate: 16000
  audio_max_length: 10.0
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

model:
  model_name: "BEATs_iter3_plus_AS2M"
  freeze_backbone: false
  dropout: 0.1

training:
  learning_rate: 5e-5
  optimizer: "adamw"
  max_epochs: 50
  patience: 10
  scheduler: "cosine"
```

Then load it:

```python
config = Config.from_yaml("config.yaml")
trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

### Quick Parameter Changes

```python
### Quick Parameter Changes
# Start with default config and modify specific parameters
config = Config()

# Change just a few key parameters
config.training.learning_rate = 1e-4
config.training.max_epochs = 100
config.data.batch_size = 16
config.model.freeze_backbone = False

trainer = BEATsTrainer.from_directory("dataset", config=config)
results = trainer.train()
```

## Model Testing and Evaluation

### Test Trained Model

```python
# Test the model
test_results = trainer.test()
print(f"Test accuracy: {test_results['test_accuracy']:.3f}")
```

### Make Predictions

```python
# Predict on new audio files
predictions = trainer.predict(["new_audio1.wav", "new_audio2.wav"])
```

### Extract Features with Trained Model

```python
# Get feature extractor from trained model
feature_extractor = trainer.get_feature_extractor()
features = feature_extractor.extract_from_files(["audio1.wav", "audio2.wav"])
```