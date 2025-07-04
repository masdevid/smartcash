# SmartCash Model Evaluation Configuration

This document provides detailed documentation of all model evaluation configuration parameters and their purposes in the SmartCash system.

## Table of Contents
1. [Evaluation Configuration](#evaluation-configuration)
2. [Inference Configuration](#inference-configuration)
3. [Complete Evaluation Configuration](#complete-evaluation-configuration)

## Evaluation Configuration

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
    - name: 'default'
      description: 'Standard evaluation scenario'
      metrics: ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall']
      iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
      
    - name: 'high_recall'
      description: 'Optimized for high recall use cases'
      metrics: ['recall@0.5', 'precision@0.5', 'f1_score']
      confidence_threshold: 0.1  # Lower threshold for higher recall
      
  # Output options
  output:
    save_predictions: true
    save_visualizations: true
    visualization_dir: 'evaluation/visualizations'
    save_metrics: true
    metrics_file: 'evaluation/metrics.json'
    
  # Performance options
  batch_size: 16  # May be different from training batch size
  num_workers: 4
  device: 'cuda'  # 'cuda' | 'cpu' | 'auto'
  
  # Advanced options
  use_tta: false  # Test-time augmentation
  tta_scales: [0.83, 1.0, 1.25]  # Scales for TTA
  tta_flip: true  # Horizontal flip for TTA
```

## Inference Configuration

```yaml
inference:
  # Detection thresholds
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
  
  # Input processing
  img_size: 640  # Should match training size
  normalize: true
  normalize_mean: [0.485, 0.456, 0.406]  # ImageNet mean
  normalize_std: [0.229, 0.224, 0.225]   # ImageNet std
  
  # Output processing
  output_format: 'coco'  # 'coco' | 'yolo' | 'pascal_voc'
  include_scores: true
  include_class_names: true
  
  # Performance
  half: true  # Use FP16 for inference
  fuse: true  # Fuse Conv+BN+ReLU layers
  nms: true   # Apply non-maximum suppression
  
  # Debugging
  debug: false
  debug_dir: 'debug/inference'
```

## Complete Evaluation Configuration

```yaml
# ========================================
# SMART CASH - EVALUATION CONFIGURATION
# ========================================

evaluation:
  # Base directories
  data:
    evaluation_dir: 'data/evaluation'
    test_dir: 'data/preprocessed/test'
    
  # Scenario configurations
  scenarios:
    - name: 'default'
      description: 'Standard evaluation scenario'
      metrics: ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall']
      iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
      
    - name: 'high_recall'
      description: 'Optimized for high recall use cases'
      metrics: ['recall@0.5', 'precision@0.5', 'f1_score']
      confidence_threshold: 0.1
  
  # Output options
  output:
    save_predictions: true
    save_visualizations: true
    visualization_dir: 'evaluation/visualizations'
    save_metrics: true
    metrics_file: 'evaluation/metrics.json'
  
  # Performance options
  batch_size: 16
  num_workers: 4
  device: 'cuda'
  
  # Advanced options
  use_tta: false
  tta_scales: [0.83, 1.0, 1.25]
  tta_flip: true

# Inference Configuration
inference:
  # Detection thresholds
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
  
  # Input processing
  img_size: 640
  normalize: true
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  
  # Output processing
  output_format: 'coco'
  include_scores: true
  include_class_names: true
  
  # Performance
  half: true
  fuse: true
  nms: true
  
  # Debugging
  debug: false
  debug_dir: 'debug/inference'
