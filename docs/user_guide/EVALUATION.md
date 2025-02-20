# 📊 Model Evaluation Guide

## 📋 Overview

Panduan lengkap untuk mengevaluasi performa model SmartCash.

## 🎯 Evaluation Metrics

### 1. Detection Metrics
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1-Score

### 2. Speed Metrics
- Inference time
- FPS
- Model size
- Memory usage

## 🚀 Basic Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --weights models/best.pt \
    --data data/rupiah.yaml \
    --batch-size 32
```

## 📊 Detailed Analysis

### 1. Per-Class Performance
```bash
python tools/analyze_classes.py \
    --weights models/best.pt \
    --data data/test
```

### 2. Error Analysis
```bash
python tools/error_analysis.py \
    --weights models/best.pt \
    --data data/test
```

## 🔍 Validation Methods

### 1. K-Fold Cross-Validation
```bash
# Run k-fold validation
python validate.py \
    --weights models/best.pt \
    --k 5
```

### 2. Hold-out Validation
```bash
# Validate on separate set
python validate.py \
    --weights models/best.pt \
    --data data/valid
```

## 📈 Performance Analysis

### 1. Confusion Matrix
```bash
python tools/confusion_matrix.py \
    --weights models/best.pt \
    --data data/test
```

### 2. ROC Curve
```bash
python tools/plot_roc.py \
    --weights models/best.pt \
    --data data/test
```

## 🔄 Benchmark Tests

### 1. Speed Benchmark
```bash
# Test inference speed
python benchmark.py \
    --weights models/best.pt \
    --batch-size 1 \
    --img-size 640
```

### 2. Resource Usage
```bash
# Monitor GPU usage
python tools/profile_gpu.py \
    --weights models/best.pt
```

## 🛠️ Advanced Analysis

### 1. Feature Analysis
```bash
# Analyze feature maps
python tools/analyze_features.py \
    --weights models/best.pt \
    --layer backbone.layer4
```

### 2. Attention Maps
```bash
# Generate attention maps
python tools/attention_maps.py \
    --weights models/best.pt \
    --image test.jpg
```

## 📊 Results Visualization

### 1. Detection Results
```bash
# Visualize detections
python detect.py \
    --weights models/best.pt \
    --source data/test \
    --save-txt
```

### 2. Performance Plots
```bash
# Generate plots
python tools/plot_results.py \
    --weights models/best.pt \
    --task study
```

## 🔍 Error Cases

### 1. False Positives
```bash
# Analyze false positives
python tools/analyze_errors.py \
    --weights models/best.pt \
    --type fp
```

### 2. False Negatives
```bash
# Analyze false negatives
python tools/analyze_errors.py \
    --weights models/best.pt \
    --type fn
```

## 📈 Performance Reports

### 1. Generate Report
```bash
# Create evaluation report
python tools/generate_report.py \
    --weights models/best.pt \
    --data data/test
```

### 2. Export Metrics
```bash
# Export to CSV
python tools/export_metrics.py \
    --weights models/best.pt \
    --format csv
```

## 🔄 Continuous Evaluation

### 1. Automated Testing
```bash
# Run test suite
pytest tests/test_model.py
```

### 2. Regression Testing
```bash
# Check for regressions
python tools/regression_test.py \
    --weights models/best.pt \
    --baseline models/baseline.pt
```

## 📊 Metrics Tracking

### 1. MLflow
```bash
# Track with MLflow
mlflow run . \
    --entry-point evaluate \
    --param-list weights=models/best.pt
```

### 2. Weights & Biases
```bash
# Log to W&B
python evaluate.py \
    --weights models/best.pt \
    --wandb
```

## 🔍 Best Practices

1. Use consistent evaluation settings
2. Test on multiple datasets
3. Monitor long-term performance
4. Document all findings
5. Version control evaluation code

## 📈 Next Steps

1. [Model optimization](../technical/OPTIMIZATION.md)
2. [Deployment](../technical/DEPLOYMENT.md)
3. [Monitoring](../technical/MONITORING.md)
