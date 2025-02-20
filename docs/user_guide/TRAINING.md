# 🎯 Training Guide

## 📋 Overview

Dokumen ini menjelaskan proses training model SmartCash dari awal hingga akhir.

## 🔧 Preparation

### 1. Dataset
```bash
# Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/processed
```

### 2. Configuration
```yaml
# configs/train_config.yaml
model:
  backbone: efficientnet_b4
  pretrained: true
  num_classes: 7

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0005
```

## 🚀 Training Process

### 1. Single GPU Training
```bash
python train.py \
    --config configs/train_config.yaml \
    --data data/rupiah.yaml \
    --epochs 100 \
    --batch-size 16 \
    --workers 8
```

### 2. Multi-GPU Training
```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    train.py \
    --config configs/train_config.yaml \
    --data data/rupiah.yaml \
    --sync-bn
```

## 📊 Monitoring

### 1. TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir runs/

# View at http://localhost:6006
```

### 2. Weights & Biases
```bash
# Login to W&B
wandb login

# Enable logging
python train.py --wandb
```

## 🔄 Training Pipeline

1. **Data Loading**
   - Dataset class
   - DataLoader
   - Augmentation

2. **Model Setup**
   - Backbone
   - Neck
   - Head
   - Loss functions

3. **Optimization**
   - Learning rate schedule
   - Gradient clipping
   - Weight decay

4. **Validation**
   - mAP calculation
   - Loss monitoring
   - Model checkpointing

## 📈 Hyperparameter Tuning

### 1. Manual Tuning
```bash
# Try different learning rates
python train.py --lr 0.001
python train.py --lr 0.0001
```

### 2. Automated Tuning
```bash
# Using Optuna
python tune.py \
    --config configs/tune_config.yaml \
    --trials 100
```

## 🔍 Training Tips

### 1. Learning Rate
- Start with 1e-3
- Use cosine schedule
- Warm-up for 3 epochs

### 2. Batch Size
- Start with 16
- Increase if memory allows
- Use gradient accumulation

### 3. Augmentation
- RandomRotate90
- RandomBrightnessContrast
- HueSaturationValue
- GaussNoise

### 4. Regularization
- Weight decay: 5e-4
- Dropout: 0.1
- Label smoothing: 0.1

## 🛠️ Advanced Techniques

### 1. Mixed Precision Training
```bash
python train.py --amp
```

### 2. Knowledge Distillation
```bash
python train.py \
    --teacher models/teacher.pt \
    --distill
```

### 3. Ensemble Training
```bash
python train.py \
    --fold 0 \
    --k-fold 5
```

## 📊 Progress Tracking

### 1. Metrics
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall

### 2. Logging
```python
# Log metrics
logger.log_metrics({
    "train/loss": loss,
    "val/mAP": mAP,
    "val/precision": precision,
    "val/recall": recall
})
```

## 🔄 Checkpointing

### 1. Save Checkpoints
```python
# Save model
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss
}, "checkpoint.pt")
```

### 2. Resume Training
```bash
python train.py \
    --resume runs/exp/weights/last.pt
```

## 🐛 Troubleshooting

### 1. Memory Issues
- Reduce batch size
- Enable AMP
- Use gradient checkpointing

### 2. Training Issues
- Check learning rate
- Monitor gradients
- Validate data

## 📈 Next Steps

1. [Evaluate model](EVALUATION.md)
2. [Export model](../technical/EXPORT.md)
3. [Deploy model](../technical/DEPLOYMENT.md)
