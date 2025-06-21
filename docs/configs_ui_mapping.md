### Simplified Form Analysis  

#### **1. Hyperparameters Form (`smartcash/ui/hyperparameters/components/ui_form.py`)**  
**Purpose**: Manages core training hyperparameters for model optimization.  
**Key Sections & Widgets**:  
- **Training**:  
  - `epochs_slider`: Total training epochs.  
  - `batch_size_slider`: Batch size (multiples of 4).  
  - `learning_rate_slider`: Initial learning rate.  
  - `image_size_slider`: Input image resolution.  
  - `mixed_precision_checkbox`: Enable FP16 training.  
  - `gradient_accumulation_slider`: Steps for gradient accumulation.  
  - `gradient_clipping_slider`: Clip gradients to prevent explosion.  
- **Optimizer**:  
  - `optimizer_dropdown`: Optimizer type (SGD, Adam, etc.).  
  - `weight_decay_slider`: L2 regularization strength.  
  - `momentum_slider`: Momentum for SGD.  
- **Scheduler**:  
  - `scheduler_dropdown`: Learning rate scheduler type.  
  - `warmup_epochs_slider`: Warmup epochs before decay.  
- **Loss**:  
  - `box_loss_gain_slider`: Weight for bounding box loss.  
  - `cls_loss_gain_slider`: Weight for class prediction loss.  
  - `obj_loss_gain_slider`: Weight for objectness loss.  
- **Control**:  
  - `early_stopping_checkbox`: Enable early stopping.  
  - `patience_slider`: Epochs to wait before stopping.  
  - `min_delta_slider`: Minimum improvement threshold.  
  - `save_best_checkbox`: Save best model checkpoint.  
  - `checkpoint_metric_dropdown`: Metric for "best" model (e.g., mAP).  

---  

#### **2. Strategy Form (`smartcash/ui/strategy/components/ui_form.py`)**  
**Purpose**: Manages training strategies, utilities, and non-hyperparameter configurations.  
**Key Sections & Widgets**:  
- **Validation**:  
  - `val_frequency_slider`: Validation frequency (epochs).  
  - `iou_thres_slider`: IoU threshold for NMS.  
  - `conf_thres_slider`: Confidence threshold for predictions.  
  - `max_detections_slider`: Max detections per image.  
- **Training Utilities**:  
  - `experiment_name_text`: Custom experiment name.  
  - `checkpoint_dir_text`: Path to save checkpoints.  
  - `tensorboard_checkbox`: Enable TensorBoard logging.  
  - `log_metrics_slider`: Log metrics every *N* steps.  
  - `visualize_batch_slider`: Visualize batches every *N* steps.  
  - `gradient_clipping_slider`: Clip gradients (**overlap**).  
  - `layer_mode_dropdown`: Single/multi-layer training.  
- **Multi-scale Training**:  
  - `multi_scale_checkbox`: Enable dynamic image resizing.  
  - `img_size_min_slider`: Minimum input size.  
  - `img_size_max_slider`: Maximum input size.  

---  

### 🔑 **Key Overlaps & Conflicts**  
1. **`gradient_clipping_slider`**  
   - **Hyperparameters Form**: Stored under `training.gradient_clipping`.  
   - **Strategy Form**: Stored under `training_utils.gradient_clipping`.  

2. **`image_size_slider` (Hyperparameters) vs. `img_size_min/max_slider` (Strategy)**  
   - **Hyperparameters**: Fixed input size (`training.image_size`).  
   - **Strategy**: Dynamic size range (`multi_scale.img_size_min/max`).  

---  

### 🗺️ **Config Key Mappings**  
#### Hyperparameters Form  
| **Widget Key**               | **Config Path**                 |  
|------------------------------|---------------------------------|  
| `epochs_slider`              | `training.epochs`              |  
| `batch_size_slider`          | `training.batch_size`          |  
| `learning_rate_slider`       | `training.learning_rate`       |  
| `gradient_clipping_slider`   | `training.gradient_clipping`   |  
| `optimizer_dropdown`         | `optimizer.type`               |  
| `scheduler_dropdown`         | `scheduler.type`               |  
| `box_loss_gain_slider`       | `loss.box_loss_gain`           |  
| `early_stopping_checkbox`    | `early_stopping.enabled`       |  
| `save_best_checkbox`         | `checkpoint.save_best`         |  

#### Strategy Form  
| **Widget Key**               | **Config Path**                     |  
|------------------------------|-------------------------------------|  
| `val_frequency_slider`       | `validation.frequency`             |  
| `gradient_clipping_slider`   | `training_utils.gradient_clipping` |  
| `experiment_name_text`       | `training_utils.experiment_name`   |  
| `layer_mode_dropdown`        | `training_utils.layer_mode`        |  
| `multi_scale_checkbox`       | `multi_scale.enabled`              |  

---  

### 📊 **Form Coverage Summary**  
| **Aspect**               | **Hyperparameters Form** | **Strategy Form** |  
|---------------------------|--------------------------|-------------------|  
| Core Hyperparameters      | ✅                       | ❌                |  
| Optimizer/Scheduler       | ✅                       | ❌                |  
| Loss Parameters           | ✅                       | ❌                |  
| Validation Strategy       | ❌                       | ✅                |  
| Logging/Visualization     | ❌                       | ✅                |  
| Multi-scale Training      | ❌                       | ✅                |  
| Early Stopping/Checkpoint | ✅                       | ❌                |