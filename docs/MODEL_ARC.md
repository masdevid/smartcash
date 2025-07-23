# 💸 Custom YOLOv5 Architecture for Banknote Detection

This document outlines the architecture and rationale for a YOLOv5-based model using a **pretrained EfficientNet-B4 backbone**, designed to detect multiple object layers in paper currency images.

---

## 🧠 Objective
- Build a YOLOv5 model using EfficientNet-B4 backbone with multilayer detection head
- Comparing backbone performance between:
   - CSPDarknet (Small:640x640)
   - EfficientNet-B4 (Small:640x640)
- Prediction test using aumented data scenario:
   - Position preset
   - Lighting preset

### Model Architecture
```json
{
  "backbone": {
    "type": ["EfficientNet-B4", "CSPDarknet"],
    "pretrained": true,
    "source": "data/pretrained" | "timm" | "YOLOv5 Repository"
  },
  "neck": {
    "type": "PAN/FPN",
    "compatible_with": "YOLOv5",
    "pretrained": true
  },
  "heads": [
    {
      "name": "layer_1_head",
      "description": "Detects full note bounding boxes",
      "init": "from_scratch"
    },
    {
      "name": "layer_2_head",
      "description": "Detects denomination-specific visual markers",
      "init": "from_scratch"
    },
    {
      "name": "layer_3_head",
      "description": "Detects common features across all notes",
      "init": "from_scratch"
    }
  ]
}
```
### Pretrained Strategy
```json
{
  "backbone": "pretrained on ImageNet",
  "neck": "pretrained or reused from YOLOv5",
  "detection_heads": "trained from scratch"
}
```

### Class Layers
```json
{
  "layer_1": {
    "description": "Full banknote detection (main object)",
    "examples": ["100K IDR", "50K IDR"],
    "classes": ["001", "002", "005", "010", "020", "050", "100"]
  },
  "layer_2": {
    "description": "Nominal-defining features (unique visual cues)",
    "examples": ["Large printed number", "Portrait", "Watermark", "Braile"],
    "classes": ["l2_001", "l2_002", "l2_005", "l2_010", "l2_020", "l2_050", "l2_100"]
  },
  "layer_3": {
    "description": "Common features (shared among notes)",
    "examples": ["BI Logo", "Serial Number & Micro Text", "Security Thread"],
    "classes": ["l3_sign", "l3_text", "l3_thread"]
  }
}
```

### Dataset Format & Handling
```json 
{
  "annotation_format": "YOLOv5PyTorch format",
  "layers": {
    "layer_1": "bounding boxes of entire banknotes",
    "layer_2": "objects that define denomination class",
    "layer_3": "shared/common objects across notes"
  },
  "missing_layer_handling": "mask loss during training when layer data is missing"
}
```

### Training Configuration
```json
{
  "phase_1": "Freeze backbone, train detection heads only",
  "phase_2": "Unfreeze entire model for fine-tuning",
  "loss_function": "total_loss = λ1 * loss_layer1 + λ2 * loss_layer2 + λ3 * loss_layer3",
  "loss_weighting": "uncertainty-based dynamic weighting"
}
```
###
```python
optimizer = Adam([
    {'params': backbone.parameters(), 'lr': 1e-5},
    {'params': head.parameters(), 'lr': 1e-3}
])
```
### Uncerainty-based dynamic weighting
From the paper "Multi-Task Learning Using Uncertainty to Weigh Losses" by Kendall et al. (Google DeepMind).

#### 🧮 Formula (for regression-style or detection losses)
For a task i with loss Lᵢ, and learnable log-variance σᵢ²:
```python
Lᵢ_weighted = (1 / (2 * σᵢ²)) * Lᵢ + log(σᵢ)
```
The total loss becomes:

```python
total_loss = Σ [(1 / (2 * σᵢ²)) * Lᵢ + log(σᵢ)]
```
Where:

σᵢ is the task-specific uncertainty parameter (learned during training)

log(σᵢ) acts as a regularizer

#### PyTorch Implementation Snipet
```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss1, loss2, loss3):
        loss = (
            (1.0 / (2 * torch.exp(self.log_sigma1)**2)) * loss1 + self.log_sigma1 +
            (1.0 / (2 * torch.exp(self.log_sigma2)**2)) * loss2 + self.log_sigma2 +
            (1.0 / (2 * torch.exp(self.log_sigma3)**2)) * loss3 + self.log_sigma3
        )
        return loss
```

---

## 🚀 Implementation Status

This section provides references to the implemented code components that realize the architecture described above.

### 📁 Core Model Architecture

#### Backbone Implementations
- **EfficientNet-B4**: `smartcash/model/architectures/backbones/efficientnet.py`
  - Enhanced with multi-layer detection support
  - Methods: `build_for_yolo()`, `prepare_for_training()`
  - Compatible with YOLOv5 neck architecture

- **CSPDarknet**: `smartcash/model/architectures/backbones/cspdarknet.py`
  - Enhanced with multi-layer detection support
  - Fixed out_channels compatibility with BaseBackbone
  - Methods: `build_for_yolo()`, `prepare_for_training()`

#### Multi-Layer Detection Heads
- **Implementation**: `smartcash/model/architectures/heads/multi_layer_head.py`
  - `MultiLayerHead` class with channel attention mechanisms
  - Factory functions for banknote detection heads
  - Support for 3-layer detection system (layer_1, layer_2, layer_3)

#### Uncertainty-Based Multi-Task Loss
- **Implementation**: `smartcash/model/training/multi_task_loss.py`
  - `UncertaintyMultiTaskLoss` class implementing Kendall et al. methodology
  - Learnable log-variance parameters for dynamic weighting
  - `AdaptiveMultiTaskLoss` for performance-based weighting

### 🏗️ Model Building Pipeline

#### Complete Model Builder
- **Implementation**: `smartcash/model/core/yolo_model_builder.py`
  - `SmartCashYOLOv5` class integrating backbone, neck, and heads
  - `YOLOModelBuilder` with testing mode support
  - Comprehensive build pipeline with validation

#### Checkpoint Management
- **Implementation**: `smartcash/model/core/checkpoint_manager.py`
  - **Checkpoint Format**: `best_{model_name}_{backbone}_{date:%Y%m%d}.pt`
  - Automatic naming and cleanup
  - Metadata preservation for model info and metrics

### 🎯 Training System

#### Training Service
- **Implementation**: `smartcash/model/training/training_service.py`
  - Two-phase training strategy (freeze backbone → fine-tune)
  - Integration with uncertainty-based loss
  - Progress tracking and UI integration

#### Training Configuration
- **Default Config**: `smartcash/configs/training_config.yaml`
- **UI Components**: `smartcash/ui/model/training/components/`
  - **Training Form**: `training_form.py` - Single accordion with 3-column layouts
  - **Configuration Summary**: `training_config_summary.py` - 4-column card-based UI
  - **Metrics Display**: `training_metrics.py` - Enhanced display with Accuracy, Precision, Recall, F1, mAP, Loss

### 📊 Evaluation System

#### Evaluation Configuration
- **Config File**: `smartcash/configs/evaluation_config.yaml`
- **Scenarios**: Position Variation, Lighting Variation
- **Metrics**: mAP@0.5, mAP@0.75, Precision, Recall, F1-Score, Accuracy, Inference Time

#### Evaluation UI
- **Implementation**: `smartcash/ui/model/evaluation/components/evaluation_ui.py`
  - **Left Column**: Scenario and metrics checkboxes
  - **Right Column**: Available best model selection
  - **Model Discovery**: Automatic detection of checkpoints with proper naming format

#### Evaluation Service
- **Implementation**: `smartcash/model/evaluation/evaluation_service.py`
- **Scenario Manager**: `smartcash/model/evaluation/scenario_manager.py`
- **Metrics Computation**: `smartcash/model/evaluation/evaluation_metrics.py`

### 🧪 Testing & Validation

#### Unit Tests
- **Model Building Tests**: `tests/model/test_model_building.py`
  - Comprehensive test suite with dummy data
  - Tests for backbone, heads, loss, and builder components
  - 17/20 tests passing (3 failures due to dummy model layer ordering)

#### Test Results Summary
- **Architecture Implementation**: ✅ Complete
- **Multi-Layer Detection**: ✅ Implemented with 3-layer system
- **Uncertainty-Based Loss**: ✅ Implemented following Kendall et al.
- **Training Pipeline**: ✅ Two-phase strategy implemented
- **Evaluation System**: ✅ 2-scenario evaluation with 7 metrics
- **UI Components**: ✅ Modern card-based layouts implemented

### 🎨 User Interface

#### Training Module UI
- **4-Column Configuration Summary**: Color-coded cards for Model, Training, Data, and Advanced settings
- **Enhanced Metrics Display**: Prominent cards for Accuracy, Precision, Recall, F1, mAP, Loss
- **Single Accordion Forms**: 3-column layouts for optimal space utilization

#### Evaluation Module UI  
- **2-Column Layout**: Scenarios/Metrics selection on left, Model selection on right
- **Checkpoint Integration**: Automatic discovery of `best_*.pt` files
- **Interactive Selection**: Checkboxes for scenarios, metrics, and available models

### 📈 Performance Targets

Based on the implemented architecture:
- **Expected mAP@0.5**: >0.80 for EfficientNet-B4, >0.75 for CSPDarknet
- **Training Phases**: Phase 1 (freeze backbone, LR: 1e-3), Phase 2 (fine-tune, LR: 1e-5)
- **Evaluation Scenarios**: Position variation (5 augmentations), Lighting variation (5 augmentations)

---

## 📦 Project Structure

```
smartcash/
├── model/
│   ├── architectures/
│   │   ├── backbones/           # EfficientNet-B4, CSPDarknet implementations
│   │   └── heads/               # Multi-layer detection heads
│   ├── core/
│   │   ├── yolo_model_builder.py   # Complete model building pipeline
│   │   └── checkpoint_manager.py   # Checkpoint management with proper naming
│   ├── training/
│   │   ├── training_service.py     # Two-phase training orchestrator
│   │   └── multi_task_loss.py      # Uncertainty-based loss implementation
│   └── evaluation/
│       ├── evaluation_service.py   # Scenario-based evaluation
│       └── scenario_manager.py     # Position/Lighting variation tests
├── ui/model/
│   ├── training/components/        # Enhanced training UI components
│   └── evaluation/components/      # 2-column evaluation UI layout
├── configs/
│   ├── evaluation_config.yaml     # Evaluation scenarios and metrics
│   └── training_config.yaml       # Training parameters and phases
└── tests/
    └── model/
        └── test_model_building.py # Comprehensive test suite
```

