# SmartCash Model Configuration Guide

This document provides detailed documentation of all model configuration parameters, their purposes, and their corresponding implementations in the codebase.

## Configuration Loading Mechanism

SmartCash uses a hierarchical configuration system with the following structure:

### 1. Configuration Files

#### Model Configuration (`model_config.yaml`)
- **Location**: `smartcash/configs/model_config.yaml`
- **Loaded by**: `SmartCashModelAPI._load_config()`
- **Purpose**: Controls model architecture and inference settings
- **Loading Code**:
  ```python
  default_config_path = Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml'
  ```

#### Training Configuration (`training_config.yaml`)
- **Location**: `smartcash/configs/training_config.yaml`
- **Loaded by**: `TrainingService._load_training_config()`
- **Purpose**: Manages training process and hyperparameters
- **Loading Code**:
  ```python
  config_path = Path('smartcash/configs/training_config.yaml')
  if config_path.exists():
      with open(config_path, 'r') as f:
          return yaml.safe_load(f)
  ```

### 2. Configuration Hierarchy

1. **Runtime Arguments**: Highest precedence, override all other settings
2. **Custom Config File**: Loaded via constructor parameters
3. **Default Config Files**: `model_config.yaml` and `training_config.yaml`
4. **Hardcoded Defaults**: Used as fallback if config files are missing

### 3. Configuration Access

Components access their configuration through:

```python
# In model components
self.config = config['model']  # For model-specific settings

# In training components
self.config = config['training']  # For training-specific settings
```

### 4. Configuration Validation

- Each component validates its required configuration
- Missing values are replaced with sensible defaults
- Invalid values raise descriptive errors

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Inference Configuration](#inference-configuration)
3. [Data Configuration](#data-configuration)
4. [Training Configuration](#training-configuration)
5. [Checkpoint Configuration](#checkpoint-configuration)
6. [Device Configuration](#device-configuration)
7. [Logging Configuration](#logging-configuration)

## Model Architecture

### Backbone Network
```yaml
model:
  backbone: 'efficientnet_b4'  # or 'cspdarknet'
  model_name: 'smartcash_yolov5'
```

**Implementation References:**
- `smartcash/model/utils/backbone_factory.py` - Backbone factory implementation
- `smartcash/model/core/model_builder.py` - Model building logic

### Detection Configuration
```yaml
detection_layers: ['banknote']  # Can include 'nominal', 'security'
layer_mode: 'single'  # 'single' | 'multilayer'
num_classes: 7  # Number of currency denominations
img_size: 640   # Input image size (height=width)
```

**Implementation References:**
- `smartcash/model/core/model_builder.py:ModelBuilder.build()` - Handles detection layer setup
- `smartcash/model/core/yolo_head.py` - Implements detection heads

### Feature Optimization
```yaml
feature_optimization:
  enabled: false
  attention: 'none'  # 'none' | 'cbam' | 'se'
  use_fpn: true
  use_pan: true
```

**Implementation References:**
- `smartcash/model/utils/backbone_factory.py:EfficientNetB4Backbone` - Implements attention mechanisms
- `smartcash/model/core/model_builder.py:_build_neck()` - Implements FPN/PAN

## Inference Configuration

```yaml
inference:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
```

**Implementation References:**
- `smartcash/model/core/yolo_head.py:YOLOHead.post_process()` - Applies confidence and NMS thresholds
- `smartcash/model/api/core.py:SmartCashModelAPI.predict()` - Handles inference pipeline

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

## Configuration Loading Flow

1. **Initialization**:
   - `SmartCashModelAPI.__init__()` loads the config file
   - Default values are merged with user-provided config

2. **Model Building**:
   - `ModelBuilder.build()` uses config to construct the model
   - Backbone, neck, and head are configured based on parameters

3. **Training/Inference**:
   - Config values are passed to respective components
   - Callbacks and utilities access config through the trainer

## Runtime Configuration Overrides

Many configuration parameters can be overridden at runtime through method arguments. Here are the key methods that accept overrides:

### 1. Model Building Overrides

```python
# Example: Override backbone and feature optimization during model building
model = model_api.build_model(
    backbone='efficientnet_b4',
    detection_layers=['banknote', 'security'],
    feature_optimization={
        'enabled': True,
        'attention': 'cbam'
    }
)
```

**Overrideable Parameters**:
- `backbone`: Backbone architecture ('cspdarknet' or 'efficientnet_b4')
- `detection_layers`: List of detection layers to enable
- `layer_mode`: 'single' or 'multilayer' detection
- `num_classes`: Number of output classes
- `img_size`: Input image size
- `feature_optimization`: Feature optimization settings

### 2. Inference Overrides

```python
# Example: Override inference parameters
predictions = model_api.predict(
    input_data,
    confidence_threshold=0.3,  # Override default 0.25
    iou_threshold=0.5,         # Override default 0.45
    max_detections=50          # Override default 100
)
```

**Overrideable Parameters**:
- `confidence_threshold`: Minimum confidence score for detections
- `iou_threshold`: Non-maximum suppression threshold
- `max_detections`: Maximum number of detections to return

### 3. Training Overrides

```python
# Example: Override training parameters
model_api.train(
    train_loader,
    val_loader,
    epochs=50,               # Override default 100
    learning_rate=0.0005,    # Override default 0.001
    batch_size=16,           # Override default 32
    checkpoint_dir='custom_checkpoints'  # Custom checkpoint directory
)
```

**Overrideable Parameters**:
- `epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `batch_size`: Training batch size
- `checkpoint_dir`: Directory to save checkpoints
- `early_stopping`: Whether to use early stopping
- `patience`: Epochs to wait before early stopping

### 4. Checkpoint Management

```python
# Example: Override checkpoint settings
model_api.save_checkpoint(
    save_path='custom_model.pt',  # Override default naming
    include_optimizer=True,       # Include optimizer state
    metadata={
        'trained_epochs': 50,
        'dataset': 'custom_dataset_v2'
    }
)
```

**Overrideable Parameters**:
- `save_path`: Custom save path
- `include_optimizer`: Whether to save optimizer state
- `metadata`: Additional metadata to include

### 5. Device Configuration

```python
# Example: Override device settings
model_api.set_device(
    device='cuda:1',       # Use specific GPU
    mixed_precision=False  # Disable mixed precision
)
```

**Overrideable Parameters**:
- `device`: Target device ('cuda', 'cpu', 'cuda:0', etc.)
- `mixed_precision`: Whether to use mixed precision training

### Best Practices for Overrides

1. **Precedence**: Runtime arguments take highest precedence, followed by config file values, then defaults
2. **Persistence**: Changes made via runtime overrides are not automatically saved to the config file
3. **Validation**: All overrides are validated against allowed values and types
4. **Thread Safety**: Overrides are not thread-safe - avoid modifying config during concurrent operations

## Best Practices

1. **For Development**:
   - Use smaller `batch_size` to save GPU memory
   - Set `logging.level: 'DEBUG'` for detailed logs
   - Disable `mixed_precision` if encountering numerical issues

2. **For Production**:
   - Enable `feature_optimization` for better accuracy
   - Use larger `batch_size` for better GPU utilization
   - Set appropriate `checkpoint.max_checkpoints` to manage disk space

3. **For Experimentation**:
   - Try different `backbone` architectures
   - Experiment with different `learning_rate` schedules
   - Adjust `confidence_threshold` and `iou_threshold` based on your precision/recall needs


## COMPLETE MODEL AND TRAINING CONFIGURATION
```yaml
# ========================================
# SMART CASH - COMPLETE CONFIGURATION
# ========================================

# Model Architecture Configuration
model:
  # Backbone network architecture
  backbone: 'efficientnet_b4'  # Options: 'cspdarknet' | 'efficientnet_b4'
  model_name: 'smartcash_yolov5'
  
  # Feature optimization
  feature_optimization:
    enabled: false
    attention: 'none'  # Options: 'none' | 'cbam' | 'se'
    use_fpn: true
    use_pan: true
  
  # Detection configuration
  detection_layers: ['banknote']  # Can include: 'banknote', 'nominal', 'security'
  layer_mode: 'single'  # Options: 'single' | 'multilayer'
  
  # Model parameters
  num_classes: 7  # Number of currency denominations
  img_size: 640   # Input image size (height=width)
  confidence_threshold: 0.25
  iou_threshold: 0.45

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

# Training Configuration
training:
  # Training schedule
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0005
  
  # Early stopping
  early_stopping: true
  patience: 10  # Epochs to wait before stopping
  
  # Optimization
  optimizer: 'adam'    # Options: 'adam' | 'sgd'
  scheduler: 'cosine'  # Options: 'cosine' | 'step' | 'plateau'
  
  # Validation
  val_interval: 1     # Validate every N epochs
  save_best: true     # Save best model only
  save_interval: 10   # Save checkpoint every N epochs

# Checkpoint Configuration
checkpoint:
  save_dir: '/data/checkpoints'
  format: 'best_{model_name}_{backbone}_{layer_mode}_{date}.pt'
  max_checkpoints: 5  # Keep last N checkpoints
  auto_cleanup: true  # Remove old checkpoints

# Device Configuration
device:
  auto_detect: true   # Auto-detect best device
  preferred: 'cuda'   # Options: 'cuda' | 'cpu' | 'mps'
  mixed_precision: true  # Use Automatic Mixed Precision (faster training)

# Logging Configuration
logging:
  level: 'INFO'  # Options: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  progress_tracking: true
  metrics_tracking: true
  save_logs: true
  log_dir: '/data/logs'

# Training Optimizer Configuration (from training_config.yaml)
optimizer:
  type: 'SGD'  # Options: 'SGD' | 'Adam'
  weight_decay: 0.0005
  momentum: 0.937
  nesterov: true

# Learning Rate Scheduler Configuration
scheduler:
  type: 'cosine'  # Options: 'cosine' | 'step' | 'plateau'
  warmup_epochs: 3
  min_lr: 0.0001
  T_max: 100  # For cosine scheduler

# Loss Function Configuration
loss:
  box_loss_gain: 0.05
  cls_loss_gain: 0.5
  obj_loss_gain: 1.0
  label_smoothing: 0.0

```


## EVALUATION CONFIGURATION

Configuration for research scenarios and model evaluation.

### Base Structure

```yaml
evaluation:
  # Base directories
  data:
    evaluation_dir: 'data/evaluation'  # Base directory for evaluation outputs
    test_dir: 'data/preprocessed/test'  # Directory containing test data
    
  # Scenario configurations  
  scenarios:
    position_variation:  # Scenario name
      enabled: true
      name: "Position Variation"  # Display name
      augmentation_config:
        num_variations: 5  # Number of variations to generate
        
        # Position augmentation parameters
        rotation_range: [-30, 30]  # Degrees
        translation_range: [-0.2, 0.2]  # Fraction of image size
        scale_range: [0.8, 1.2]  # Scaling factors
        perspective_range: 0.1  # Perspective distortion
        horizontal_flip: 0.5  # Probability of horizontal flip
    
    lighting_variation:  # Another scenario
      enabled: true
      name: "Lighting Variation"
      augmentation_config:
        num_variations: 5
        
        # Lighting augmentation parameters
        brightness_range: [-0.3, 0.3]  # Brightness adjustment
        contrast_range: [0.7, 1.3]  # Contrast adjustment
        gamma_range: [0.7, 1.3]  # Gamma correction
        hsv_hue: 15  # Hue shift in degrees
        hsv_saturation: 20  # Saturation adjustment
```

### Default Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `rotation_range` | `[-30, 30]` | Rotation range in degrees |
| `translation_range` | `[-0.2, 0.2]` | Translation as fraction of image size |
| `scale_range` | `[0.8, 1.2]` | Scaling factors |
| `perspective_range` | `0.1` | Perspective distortion amount |
| `horizontal_flip` | `0.5` | Probability of horizontal flip |
| `brightness_range` | `[-0.3, 0.3]` | Brightness adjustment range |
| `contrast_range` | `[0.7, 1.3]` | Contrast adjustment range |
| `gamma_range` | `[0.7, 1.3]` | Gamma correction range |
| `hsv_hue` | `15` | Maximum hue shift in degrees |
| `hsv_saturation` | `20` | Maximum saturation adjustment |

### Usage Example

```python
from smartcash.model.evaluation.scenario_manager import create_scenario_manager

config = {
    'evaluation': {
        'data': {
            'evaluation_dir': 'data/evaluation',
            'test_dir': 'data/preprocessed/test'
        },
        'scenarios': {
            'position_variation': {
                'enabled': True,
                'augmentation_config': {
                    'num_variations': 5,
                    'rotation_range': [-15, 15]  # Override default
                }
            }
        }
    }
}

# Initialize scenario manager
scenario_manager = create_scenario_manager(config)

# Generate position variations
scenario_manager.setup_position_scenario()
```
