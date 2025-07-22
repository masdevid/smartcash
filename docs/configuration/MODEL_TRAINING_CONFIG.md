# SmartCash Model Training Configuration

This document provides detailed documentation of all training configuration parameters and their purposes in the SmartCash system.

## Table of Contents
1. [Training Configuration](#training-configuration)
2. [Data Configuration](#data-configuration)
3. [Checkpoint Configuration](#checkpoint-configuration)
4. [Device Configuration](#device-configuration)
5. [Logging Configuration](#logging-configuration)
6. [Complete Training Configuration](#complete-training-configuration)

## Training Configuration

### Training Schedule
```yaml
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0005
  
  # Early stopping
  early_stopping: true
  patience: 10
  
  # Optimization
  optimizer: 'adam'    # 'adam' | 'sgd'
  scheduler: 'cosine'  # 'cosine' | 'step' | 'plateau'
  
  # Validation
  val_interval: 1
  save_best: true
  save_interval: 10
```

**Implementation References:**
- `smartcash/trainer/trainer.py` - Main training loop
- `smartcash/trainer/optimizer.py` - Optimizer and scheduler setup
- `smartcash/trainer/early_stopping.py` - Early stopping implementation

## Data Configuration

```yaml
data:
  pretrained_dir: '/data/pretrained'
  dataset_dir: '/data/preprocessed'
  batch_size: 32
  num_workers: 4
  pin_memory: true
  
  splits:
    train: 'train'
    valid: 'valid'
    test: 'test'
```

**Implementation References:**
- `smartcash/data/` - Data loading and preprocessing
- `smartcash/trainer/data_loader.py` - Data loading implementation

## Checkpoint Configuration

```yaml
checkpoint:
  save_dir: '/data/checkpoints'
  format: 'best_{model_name}_{backbone}_{layer_mode}_{date:%Y%m%d}.pt'
  max_checkpoints: 5
  auto_cleanup: true
```

**Implementation References:**
- `smartcash/model/core/checkpoint_manager.py` - Handles checkpoint saving/loading
- `smartcash/trainer/callbacks/checkpoint.py` - Training checkpoint callbacks

## Device Configuration

```yaml
device:
  auto_detect: true
  preferred: 'cuda'  # 'cuda' | 'cpu' | 'mps'
  mixed_precision: true
```

**Implementation References:**
- `smartcash/model/utils/device_utils.py` - Device management
- `smartcash/trainer/mixed_precision.py` - Mixed precision training

## Logging Configuration

```yaml
logging:
  level: 'INFO'  # 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  progress_tracking: true
  metrics_tracking: true
  save_logs: true
  log_dir: '/data/logs'
```

**Implementation References:**
- `smartcash/common/logger.py` - Logging setup
- `smartcash/utils/progress.py` - Progress tracking
- `smartcash/trainer/metrics/` - Metrics tracking implementation

## Complete Training Configuration

```yaml
# ========================================
# SMART CASH - TRAINING CONFIGURATION
# ========================================

training:
  # Training schedule
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0005
  
  # Early stopping
  early_stopping: true
  patience: 10  # Epochs to wait before stopping
  
  # Optimization
  optimizer: 'adam'    # 'adam' | 'sgd'
  scheduler: 'cosine'  # 'cosine' | 'step' | 'plateau'
  
  # Validation
  val_interval: 1     # Validate every N epochs
  save_best: true     # Save best model based on validation
  save_interval: 10   # Save checkpoint every N epochs

# Data Configuration
data:
  # Directory structure
  pretrained_dir: '/data/pretrained'
  dataset_dir: '/data/preprocessed'
  
  # Data loading
  batch_size: 32
  num_workers: 4
  pin_memory: true
  
  # Dataset splits
  splits:
    train: 'train'
    valid: 'valid'
    test: 'test'

# Checkpoint Configuration
checkpoint:
  save_dir: '/data/checkpoints'
  format: 'best_{model_name}_{backbone}_{layer_mode}_{date:%Y%m%d}.pt'
  max_checkpoints: 5
  auto_cleanup: true

# Device Configuration
device:
  auto_detect: true
  preferred: 'cuda'  # 'cuda' | 'cpu' | 'mps'
  mixed_precision: true

# Logging Configuration
logging:
  level: 'INFO'  # 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  progress_tracking: true
  metrics_tracking: true
  save_logs: true
  log_dir: '/data/logs'
