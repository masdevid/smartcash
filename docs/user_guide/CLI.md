# 🖥️ CLI Interface

## 📋 Overview

SmartCash menyediakan Command Line Interface (CLI) untuk berbagai operasi seperti training, inference, dan evaluasi.

## 🚀 Basic Usage

### 1. Training

```bash
python train.py --config configs/base_config.yaml
```

Options:
- `--config`: Path ke file konfigurasi
- `--epochs`: Jumlah epochs
- `--batch-size`: Ukuran batch
- `--workers`: Jumlah workers
- `--device`: CPU atau GPU (cuda)

### 2. Inference

```bash
python detect.py --source images/test.jpg --weights models/best.pt
```

Options:
- `--source`: Path ke gambar/video
- `--weights`: Path ke model weights
- `--conf-thres`: Confidence threshold
- `--iou-thres`: NMS IoU threshold
- `--save-txt`: Simpan hasil dalam format txt

### 3. Evaluation

```bash
python evaluate.py --weights models/best.pt --data data/test
```

Options:
- `--weights`: Path ke model weights
- `--data`: Path ke data test
- `--batch-size`: Ukuran batch
- `--task`: test, study, atau speed

## 🛠️ Advanced Usage

### 1. Export Model

```bash
python export.py --weights models/best.pt --include onnx tflite
```

Supported formats:
- ONNX
- TensorRT
- TFLite
- CoreML
- OpenVINO

### 2. Hyperparameter Tuning

```bash
python tune.py --data data/rupiah.yaml --epochs 100
```

Parameters:
- Learning rate
- Batch size
- Momentum
- Weight decay

### 3. Deployment

```bash
# Start API server
python serve.py --port 8000 --model models/best.pt

# Test API
curl -X POST "http://localhost:8000/predict" -F "image=@test.jpg"
```

## 📊 Logging & Monitoring

### 1. Training Logs

```bash
# View training progress
tail -f logs/train.log

# Monitor metrics
tensorboard --logdir runs/
```

### 2. Experiment Tracking

```bash
# Track with MLflow
mlflow ui --backend-store-uri ./mlruns

# Track with Weights & Biases
python train.py --weights "" --wandb
```

## 🔧 Configuration

### 1. Dataset Config

```yaml
# data/rupiah.yaml
path: ../data
train: train/images
val: valid/images
test: test/images

nc: 7  # number of classes
names: ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
```

### 2. Model Config

```yaml
# configs/base_config.yaml
model:
  backbone: efficientnet_b4
  neck: fpn
  head: yolov5

training:
  epochs: 100
  batch_size: 16
  optimizer: AdamW
  lr: 0.001
```

## 🔍 Debugging

### 1. Debug Mode

```bash
# Enable debug logging
python train.py --debug

# Profile memory usage
python -m memory_profiler train.py
```

### 2. Validation

```bash
# Validate dataset
python tools/validate_dataset.py

# Validate model
python tools/validate_model.py --weights models/best.pt
```

## 📈 Visualization

### 1. Results

```bash
# Plot results
python tools/plot_results.py --weights models/best.pt

# Create confusion matrix
python tools/confusion_matrix.py --weights models/best.pt
```

### 2. Model Analysis

```bash
# Analyze model
python tools/analyze_model.py --weights models/best.pt

# Profile inference
python tools/profile_inference.py --weights models/best.pt
```

## 🚀 Best Practices

1. Use version control for configs
2. Log experiments properly
3. Monitor system resources
4. Backup model checkpoints
5. Document custom changes

## 🆘 Troubleshooting

Common issues:
1. CUDA out of memory
2. Dataset loading errors
3. Training instability
4. Inference performance

Solutions in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
