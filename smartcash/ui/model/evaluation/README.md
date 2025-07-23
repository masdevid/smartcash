# SmartCash Evaluation Module

**Evaluation module for comprehensive model performance assessment across multiple scenarios**

## Architecture Overview

The evaluation module follows the BaseUIModule pattern with operation-based backend integration. It works seamlessly with backbone and training modules through standardized model naming conventions.

### Current Evaluation Form Structure

**Row 1: Execution Configuration**
- **Execution Options**: Scenario selection and evaluation parameters
- **Model Selection**: Active backbone and layer mode selection (integrates with training output)

**Row 2: Metrics & Display**  
- **Metrics Selection**: Performance metrics to evaluate (mAP, precision, recall, etc.)
- **Available Models Display**: Real-time discovery of trained models with refresh capability

### Model Naming Convention (Latest Implementation)

Models follow the standardized `{training_id}_{backbone_type}_{timestamp}` format:

**Checkpoint Pattern:**
```
smartcash_training_1732123456_efficientnet_b4_best.pt
smartcash_training_1732123456_efficientnet_b4_epoch_50.pt
smartcash_resume_1732234567_yolov5s_best.pt
```

**Model Discovery Pattern:**
- **Training ID**: `smartcash_training_` + timestamp or `smartcash_resume_` + timestamp
- **Backbone Type**: From backbone module (`efficientnet_b4`, `yolov5s`, `resnet50`, etc.)
- **Checkpoint Type**: `best.pt` (best model) or `epoch_{N}.pt` (specific epoch)

This follows the latest implementation used by backbone and training modules as of 2024.

### Workflow Integration

1. **Backbone Module** → Configures model architecture (EfficientNet-B4, YOLOv5s, etc.)
2. **Training Module** → Trains models with timestamp-based naming
3. **Evaluation Module** → Auto-discovers trained models using naming convention
4. **Backend Integration** → Real evaluation service (no simulation)

### Key Features

- **Real Backend Integration**: Uses `smartcash.model.evaluation.run_evaluation_pipeline`
- **Model Auto-Discovery**: Scans checkpoint directory for trained models
- **Multi-Scenario Evaluation**: Position, lighting, distance, rotation scenarios
- **Triple Progress Tracking**: Overall → Phase → Current step granular progress
- **Live Metrics Updates**: Real-time chart updates during evaluation
- **Comprehensive Metrics**: mAP@0.5, mAP@0.75, precision, recall, F1-score

### Model Selection Logic

The evaluation module uses **active model selection** where:
- User selects backbone type in the form (EfficientNet-B4, YOLOv5s, etc.)
- System automatically discovers corresponding trained checkpoints
- Evaluation runs on the best checkpoint for the selected backbone
- Results are displayed with model-specific performance breakdowns

### Backend Service Integration

```python
from smartcash.model.evaluation import run_evaluation_pipeline
from smartcash.model.api.core import create_model_api

# Real evaluation service integration
evaluation_result = run_evaluation_pipeline(
    scenarios=[backend_scenario], 
    checkpoints=None,
    model_api=model_api,
    config=eval_config,
    progress_callback=progress_callback,
    ui_components=ui_components
)
```

**No simulation code** - all evaluation operations use real backend services.

### Operation Pattern

- **Prerequisite Checks**: Validates trained models availability
- **Progress Tracking**: Triple-bar progress with phase-specific updates  
- **Error Handling**: Comprehensive error states with user feedback
- **Result Storage**: Evaluation results saved with timestamp and model metadata