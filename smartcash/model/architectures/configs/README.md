# SmartCash Model Architecture Configurations

This directory contains organized model architecture configurations for the SmartCash banknote detection system.

## Directory Structure

```
configs/
├── models/                    # Model-specific configurations
│   └── yolov5/               # YOLOv5 architecture configurations
│       ├── cspdarknet/       # CSPDarknet backbone variants
│       │   ├── smartcash_yolov5n_cspdarknet.yaml  # Nano (lightweight)
│       │   ├── smartcash_yolov5s_cspdarknet.yaml  # Small (default)
│       │   ├── smartcash_yolov5m_cspdarknet.yaml  # Medium
│       │   ├── smartcash_yolov5l_cspdarknet.yaml  # Large
│       │   └── smartcash_yolov5x_cspdarknet.yaml  # Extra Large
│       └── efficientnet/     # EfficientNet backbone variants
│           ├── smartcash_yolov5n_efficientnet.yaml  # Nano (EfficientNet-B0)
│           ├── smartcash_yolov5s_efficientnet.yaml  # Small (EfficientNet-B4)
│           ├── smartcash_yolov5m_efficientnet.yaml  # Medium (EfficientNet-B3)
│           ├── smartcash_yolov5l_efficientnet.yaml  # Large (EfficientNet-B5)
│           └── smartcash_yolov5x_efficientnet.yaml  # Extra Large (EfficientNet-B7)
├── backbones/                # Backbone-specific configurations (future)
├── heads/                    # Detection head configurations (future)
└── unused/                   # Deprecated/unused configurations
```

## Configuration Files

### CSPDarknet Backbone Variants

| Size | Config File | Model Characteristics | Use Case |
|------|-------------|----------------------|----------|
| **Nano (n)** | `smartcash_yolov5n_cspdarknet.yaml` | Lightweight, fast inference | Mobile/edge deployment |
| **Small (s)** | `smartcash_yolov5s_cspdarknet.yaml` | Balanced speed/accuracy | Development, testing |
| **Medium (m)** | `smartcash_yolov5m_cspdarknet.yaml` | Better accuracy | Production with good hardware |
| **Large (l)** | `smartcash_yolov5l_cspdarknet.yaml` | High accuracy | Server deployment |
| **Extra Large (x)** | `smartcash_yolov5x_cspdarknet.yaml` | Maximum accuracy | High-end server deployment |

### EfficientNet Backbone Variants

| Size | Config File | EfficientNet Version | Use Case |
|------|-------------|---------------------|----------|
| **Nano (n)** | `smartcash_yolov5n_efficientnet.yaml` | EfficientNet-B0 | Mobile/edge deployment |
| **Small (s)** | `smartcash_yolov5s_efficientnet.yaml` | EfficientNet-B4 | Development, testing |
| **Medium (m)** | `smartcash_yolov5m_efficientnet.yaml` | EfficientNet-B3 | Production deployment |
| **Large (l)** | `smartcash_yolov5l_efficientnet.yaml` | EfficientNet-B5 | Server deployment |
| **Extra Large (x)** | `smartcash_yolov5x_efficientnet.yaml` | EfficientNet-B7 | High-end server deployment |

## Configuration Structure

Each configuration file contains:

### 1. Model Parameters
- `nc`: Number of classes per layer
- `depth_multiple`: Model depth scaling factor
- `width_multiple`: Channel width scaling factor
- `anchors`: YOLO anchor boxes for different scales

### 2. Architecture Definition
- `backbone`: Backbone network configuration
- `head`: Detection head and neck configuration

### 3. Multi-layer Specifications
- `layer_specs`: Detailed specifications for each detection layer
  - `layer_1`: Full banknote detection (7 classes)
  - `layer_2`: Nominal-defining features (7 classes)  
  - `layer_3`: Common features (3 classes)

### 4. Training Configuration
- `training`: Training-specific parameters
  - Image size, batch size, epochs
  - Phase-specific training (two-phase training)
  - Optimizer settings (different learning rates for backbone/head)
  - **Loss function**: `total_loss = λ1 * loss_layer1 + λ2 * loss_layer2 + λ3 * loss_layer3`
  - **Loss weighting**: Uncertainty-based dynamic weighting (Kendall et al.)

### 5. Backbone-Specific Settings
- `backbone_config`: Backbone-specific configurations
  - Model name, pretrained weights
  - Feature extraction layers for FPN

## Usage

### Loading Configurations

The configurations are automatically loaded by the YOLOv5 integration:

```python
from smartcash.model.architectures.yolov5_integration import YOLOv5Integration

# Create model with CSPDarknet backbone (small)
integration = YOLOv5Integration()
model = integration.create_model(backbone_type="cspdarknet", model_size="s")

# Create model with EfficientNet backbone (medium)  
model = integration.create_model(backbone_type="efficientnet", model_size="m")
```

### Configuration Path Pattern

The configuration files follow this naming pattern:
```
smartcash_yolov5{size}_{backbone}.yaml
```

Where:
- `{size}`: n, s, m, l, x
- `{backbone}`: cspdarknet, efficientnet

## Multi-layer Detection

All configurations support SmartCash's multi-layer detection system:

1. **Layer 1**: Full banknote detection (main objects)
2. **Layer 2**: Nominal-defining features (unique visual cues)
3. **Layer 3**: Common features (shared security features)

This enables hierarchical feature learning and improved detection accuracy.

## Training Optimization

Each model size includes optimized training parameters:

- **Nano/Small**: Higher learning rates, more epochs, larger batch sizes
- **Medium**: Balanced parameters for general use
- **Large/X-Large**: Lower learning rates, fewer epochs, smaller batch sizes

All configurations use **uncertainty-based dynamic weighting** for multi-layer loss balancing, following the Kendall et al. methodology where learnable log-variance parameters σᵢ² automatically balance losses between layers during training.

## Performance Characteristics

| Model Size | Parameters | Speed | Accuracy | Memory Usage |
|------------|------------|-------|----------|--------------|
| Nano       | ~1.9M      | Fastest | Good | Lowest |
| Small      | ~7.2M      | Fast | Better | Low |
| Medium     | ~21.2M     | Medium | Good | Medium |
| Large      | ~46.5M     | Slow | Better | High |
| X-Large    | ~86.7M     | Slowest | Best | Highest |

## Contributing

When adding new configurations:

1. Follow the established naming convention
2. Include all required sections (parameters, architecture, layer_specs, training)
3. Optimize training parameters for the model size
4. Update this README with new configurations
5. Test configurations before committing

## Notes

- All configurations use the same anchor boxes optimized for banknote detection
- Training configurations assume two-phase training (freeze backbone, then fine-tune)
- EfficientNet configurations include backbone-specific feature extraction layers
- **Loss weighting**: Uncertainty-based dynamic weighting with learnable parameters (not static weights)
- **Loss implementation**: Based on Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"