# 💸 Custom YOLOv5 Architecture for Banknote Detection

This document outlines the architecture and rationale for a YOLOv5-based model using a **pretrained EfficientNet-B4 backbone**, designed to detect multiple object layers in paper currency images.

---

## 🧠 Objective
- Build a YOLOv5 model using EfficientNet-B4 backbone with multilayer detection head
- Compare backbone performance between:
  - CSPDarknet (Small:640x640)
  - EfficientNet-B4 (Small:640x640)
- Prediction test using augmented data scenarios:
  - Position preset
  - Lighting preset

## Model Architecture
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
    "examples": ["Large printed number", "Nominal Text", "Braile"],
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

## Training Configuration
```json
{
  "phase_1": "Freeze backbone, train detection heads only",
  "phase_2": "Unfreeze entire model for fine-tuning",
  "loss_function": "total_loss = λ1 * loss_layer1 + λ2 * loss_layer2 + λ3 * loss_layer3",
  "loss_weighting": "uncertainty-based dynamic weighting"
}
```

```python
optimizer = Adam([
    {'params': backbone.parameters(), 'lr': 1e-5},
    {'params': head.parameters(), 'lr': 1e-3}
])
```

### Uncertainty-Based Dynamic Weighting
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
- σᵢ is the task-specific uncertainty parameter (learned during training)
- log(σᵢ) acts as a regularizer

#### PyTorch Implementation Snippet
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

## 🎯 Training Modes & Prediction Structure
1. **Single-Phase + Multi-Layer** (training_mode='single_phase', layer_mode='multi'):
   - Return 3 predictions: {'layer_1': pred, 'layer_2': pred, 'layer_3': pred}
   - Processes all layers in a single training phase
2. **Single-Phase + Single-Layer** (training_mode='single_phase', layer_mode='single'):
   - Return 1 prediction: {'layer_1': pred}
   - Processes only the primary layer
3. **Two-Phase Mode**:
   - Phase 1: Return 1 prediction: {'layer_1': pred}
   - Phase 2: Return 3 predictions: {'layer_1': pred, 'layer_2': pred, 'layer_3': pred}

## 🔧 Hierarchical Validation System

### Phase-Aware Validation Processing
- **Phase 1 (Frozen Backbone)**: Standard single-layer validation for classes 0-6
- **Phase 2 (Unfrozen Backbone)**: Hierarchical multi-layer validation with confidence modulation

### Layer Architecture & Class Distribution
```
Layer 1: Denomination Detection (Classes 0-6)
├── 001, 002, 005, 010, 020, 050, 100 (Indonesian Rupiah denominations)
├── Purpose: Primary task - banknote denomination identification
└── Metrics: Primary validation metrics (mAP, precision, recall, F1)

Layer 2: Confidence Features (Classes 7-13)  
├── Denomination-specific visual cues and features
├── Purpose: Enhanced denomination validation through visual features
└── Integration: Spatial overlap + confidence modulation with Layer 1

Layer 3: Money Validation (Classes 14-16)
├── Security features and authenticity markers
├── Purpose: General money validation (authentic banknote detection)
└── Integration: Money authenticity threshold + confidence boost/reduction
```

### Hierarchical Processing Flow (Phase 2 Only)
```
Input: All predictions (classes 0-16)
    ↓
Phase Detection: max_class >= 7 → Phase 2 hierarchical processing
    ↓
Layer 1 Filtering: Extract classes 0-6 for primary evaluation
    ↓
Confidence Modulation:
    • Layer 2 Match: Same denomination + spatial IoU > 0.1
    • Layer 3 Match: Money validation + spatial IoU > 0.1
    • Hierarchical Boost: conf × (1 + layer2_conf × layer3_conf) if layer3_conf > 0.1
    • Confidence Reduction: conf × 0.1 if layer3_conf ≤ 0.1 (not money)
    ↓
Metrics Calculation: mAP, precision, recall, F1 on enhanced Layer 1 predictions
```

### Key Benefits
- **Focused Evaluation**: Primary metrics measure denomination detection quality (main task)
- **Intelligent Filtering**: Layer 3 ensures predictions are validated as actual money
- **Research Insights**: Per-layer metrics preserved for detailed analysis
- **Phase Consistency**: Phase 1 establishes baseline, Phase 2 adds hierarchical enhancement
- **Memory Optimization**: Chunked processing for large prediction sets (>10K predictions)

## 📊 Loss Calculation Strategy

### Phase 1: Frozen Backbone Training
```
Description: Backbone frozen, only train Layer 1 head (coarse/global detection)
Backbone State: frozen
Active Layers: layer_1 only
Loss Weights:
├── layer_1: 1.0 (full weight)
├── layer_2: 0.0 (not trained)
└── layer_3: 0.0 (not trained)
Purpose: Stabilize initial learning and establish baseline detection
Optional: Small warmup weights (0.1) for layer_2 and layer_3 can be added
```

### Phase 2: Uncertainty-Weighted Multi-Task Learning
```
Description: Backbone unfrozen, fine-tune all layers with uncertainty-based weighting
Backbone State: unfrozen  
Active Layers: layer_1, layer_2, layer_3
Loss Calculation Method: uncertainty_weighted_loss
Mathematical Formulation:
L_total = Σ (1 / (2 * σ_i²)) * L_i + log(σ_i)
Where:
├── L_i: Loss for layer i (i = 1, 2, 3)
├── σ_i: Learnable uncertainty parameter for layer i
├── 1/(2*σ_i²): Automatic weight based on uncertainty
└── log(σ_i): Regularization term to prevent σ_i → 0
Layer-Specific Weights:
├── Layer 1: L1_weight = 1 / (2 * σ1²), learnable σ1
├── Layer 2: L2_weight = 1 / (2 * σ2²), learnable σ2  
└── Layer 3: L3_weight = 1 / (2 * σ3²), learnable σ3
```

### Key Benefits of Uncertainty Weighting
- **Adaptive Balancing**: Automatically adjusts contribution of each layer based on task difficulty
- **Prevents Domination**: No single layer can dominate the loss function
- **Multi-Task Optimization**: Improved multi-task learning through uncertainty-based weighting
- **Self-Regulating**: Learnable σ parameters adapt during training to optimal values
- **Mathematical Foundation**: Based on principled uncertainty estimation in multi-task learning

## 🔄 Critical Two-Phase Training Weight Transfer Flow

### Phase 1 → Phase 2 Transition Process
This is one of the most critical aspects of the two-phase training system, ensuring Phase 1 learning is preserved when transitioning to Phase 2.

#### Step-by-Step Weight Transfer Flow:
```
1. Phase 1 Training Completion
   ├── Model: Frozen backbone + trained detection head
   ├── Checkpoint: Save best Phase 1 model (frozen state)
   └── Architecture: Single task optimized (backbone frozen)
2. Phase 2 Transition Preparation
   ├── Load Phase 1 checkpoint data into memory
   ├── Extract model configuration from checkpoint
   ├── Validate config matches actual model architecture
   └── Detect/correct any config mismatches via inference
3. Phase 2 Model Rebuilding
   ├── Build new model: same architecture + unfrozen backbone
   ├── CRITICAL: Set pretrained=False (don't load YOLOv5s weights)
   ├── Ensure exact architectural match (classes, layers, backbone)
   └── Result: Fresh Phase 2 model ready for weight loading
4. Weight Transfer Execution
   ├── Load Phase 1 state_dict into Phase 2 model
   ├── Use strict=False to handle minor mismatches
   ├── Log missing/unexpected keys for debugging
   └── Fallback: Use fresh weights if transfer fails
5. Phase 2 Training Continuation
   ├── Model: Unfrozen backbone + Phase 1 trained weights
   ├── Training: Fine-tune entire model (backbone + heads)
   └── Optimization: All parameters now trainable
```

#### Critical Configuration Validation:
The system performs intelligent validation to ensure weight transfer compatibility:
```python
# Architecture Validation Logic
saved_config = checkpoint['model_config']
actual_output_channels = state_dict['detection_head'].shape[0]
# Example: 66 output channels = 17 classes (multi-layer)
# But saved config shows num_classes=7 → MISMATCH DETECTED
if actual_output_channels == 66 and saved_config['num_classes'] != 17:
    # Trigger intelligent config inference
    inferred_config = infer_from_architecture(checkpoint, checkpoint_path)
    # Use inferred config: backbone='efficientnet_b4', num_classes=17
```

#### Weight Transfer Scenarios & Handling:
**✅ Successful Transfer:**
```
Phase 1 Model (frozen) → Phase 2 Model (unfrozen)
├── Backbone weights: Transferred successfully  
├── Detection head: Transferred successfully
├── Training state: Preserved and continued
└── Result: Phase 1 learning preserved in Phase 2
```
**⚠️ Architecture Mismatch (Fixed):**
```
Problem: Saved config mismatch (7 vs 17 classes)
├── Detection: Config validation detects mismatch
├── Solution: Intelligent inference from architecture
├── Action: Rebuild with correct configuration
└── Result: Successful weight transfer with correct config
```
**❌ Transfer Failure (Fallback):**
```
Critical mismatch preventing weight loading
├── Fallback: Phase 2 starts with fresh initialization
├── Loss: Phase 1 training progress is lost
├── Mitigation: Improved config validation prevents this
└── Logs: Clear error messages for debugging
```

## 🚀 Implementation Status

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
#### Key Implementation Files
- `pipeline_executor.py`: `_transition_to_phase2_and_train()` - Main transition logic
- `pipeline_executor.py`: `_rebuild_model_for_phase2()` - Model rebuilding with config validation
- `pipeline_executor.py`: `_infer_model_config_from_checkpoint()` - Intelligent config inference
- `checkpoint_manager.py`: Enhanced config saving and extraction
#### Recent Critical Fixes (Aug 2024):
1. **Config Validation**: Detect saved config vs actual architecture mismatches
2. **Intelligent Inference**: Extract correct config from checkpoint structure and filename
3. **Pretrained Weight Prevention**: Disable pretrained weights during Phase 2 rebuild
4. **Enhanced Error Handling**: Better logging and fallback strategies

### 📊 Evaluation System
#### Evaluation Configuration
- **Config File**: `smartcash/configs/evaluation_config.yaml`
- **Scenarios**: Position Variation, Lighting Variation
- **Metrics**: mAP@0.5, mAP@0.75, Precision, Recall, F1-Score, Accuracy, Inference Time
#### Evaluation Service
- **Implementation**: `smartcash/model/evaluation/evaluation_service.py`
- **Scenario Manager**: `smartcash/model/evaluation/scenario_manager.py`
- **Metrics Computation**: `smartcash/model/evaluation/evaluation_metrics.py`
- **Additional Files**: 
  - `yolov5_map_calculator.py`: Hierarchical mAP calculation with confidence modulation
  - `validation_metrics_computer.py`: Hierarchical validation metrics alignment  
  - `hierarchical_processor.py`: Core hierarchical filtering and confidence modulation logic

### 📈 Performance Targets
Based on the implemented architecture:
- **Expected mAP@0.5**: >0.80 for EfficientNet-B4, >0.75 for CSPDarknet
- **Training Phases**: Phase 1 (freeze backbone, LR: 1e-3), Phase 2 (fine-tune, LR: 1e-5)
- **Evaluation Scenarios**: Position variation (5 augmentations), Lighting variation (5 augmentations)

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
│       ├── scenario_manager.py     # Position/Lighting variation tests
│       ├── yolov5_map_calculator.py # Hierarchical mAP calculation
│       ├── validation_metrics_computer.py # Hierarchical validation metrics
│       └── hierarchical_processor.py # Confidence modulation logic
├── configs/
│   ├── evaluation_config.yaml     # Evaluation scenarios and metrics
│   └── training_config.yaml       # Training parameters and phases
└── tests/
    └── model/
        └── test_model_building.py # Comprehensive test suite
```