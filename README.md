# SmartCash YOLOv5 Architecture

This document outlines the architecture of the SmartCash YOLOv5 model, designed for currency detection with 17-class training (7 denominations, 7 denomination-specific features, 3 authenticity features) and 7-class inference mapping. The model uses a modular structure with separate backbone, neck, and head components, supporting both YOLOv5 and EfficientNet-B4 backbones.

## Overview

The SmartCash YOLOv5 model is built on the Ultralytics YOLOv5 framework with a custom architecture tailored for currency detection. It employs a two-phase training strategy (head-only, then full fine-tuning) and maps 17 training classes to 7 denominations during inference with confidence adjustments. The architecture is split into:

- **Backbone**: Extracts multi-scale features from input images (640x640).
- **Neck**: Aggregates features using a Path Aggregation Network (PANet).
- **Head**: Performs detection, outputting bounding boxes, scores, and class probabilities.

The model supports two backbone variants: YOLOv5 (e.g., yolov5s, yolov5m, yolov5l, yolov5x) and EfficientNet-B4.

## Architecture Diagram

Below is a simplified representation of the architecture:

```
Input (3x640x640)
  ↓
[Backbone]
  ↓ P3, P4, P5 (feature maps)
[Neck - PANet]
  ↓ Aggregated features (P3_out, P4_final, P5_final)
[Head]
  ↓ Detections (boxes, scores, classes)
[Post-processing]
  ↓ 17-to-7 class mapping, confidence adjustment
Output: Denomination predictions
```

## 1. Backbone

The backbone extracts features at multiple scales (P3, P4, P5) from the input image.

### YOLOv5 Backbone

- **Variants**: yolov5s, yolov5m, yolov5l, yolov5x
- **Structure** (for yolov5s):
  - Input: 3x640x640
  - Layers:
    - Conv (3→32, k=6, s=2, p=2): 320x320
    - Conv (32→64, k=3, s=2): 160x160
    - C2f (64→64, n=1): 160x160 (P2)
    - Conv (64→128, k=3, s=2): 80x80
    - C2f (128→128, n=2): 80x80 (P3)
    - Conv (128→256, k=3, s=2): 40x40
    - C2f (256→256, n=3): 40x40 (P4)
    - Conv (256→512, k=3, s=2): 20x20
    - C2f (512→512, n=1): 20x20
    - SPPF (512→512, k=5): 20x20 (P5)
  - Output Feature Dimensions:
    - P3: 128x80x80
    - P4: 256x40x40
    - P5: 512x20x20
  - Notes: Channel counts scale with variant (e.g., yolov5m doubles channels).

### EfficientNet-B4 Backbone

- **Implementation**: Uses `timm` library for EfficientNet-B4.
- **Structure**:
  - Input: 3x640x640
  - Stages: Extracts features from stages 2, 3, and 4 (equivalent to P3, P4, P5).
  - Output Feature Dimensions:
    - P3: 56x80x80
    - P4: 160x40x40
    - P5: 448x20x20
  - Notes: Pretrained weights available, no SPPF in backbone (added in neck).

## 2. Neck (PANet)

The neck aggregates features from the backbone using a Path Aggregation Network (PANet), combining top-down and bottom-up paths for multi-scale feature fusion.

### YOLOv5 Neck

- **Input**: P3 (128x80x80), P4 (256x40x40), P5 (512x20x20)
- **Layers**:
  1. Conv (512→256, k=1, s=1)
  2. Upsample (256, scale=2): 40x40
  3. Concat (256 + 256): 512x40x40
  4. C2f (512→256, n=1): 256x40x40
  5. Conv (256→128, k=1, s=1)
  6. Upsample (128, scale=2): 80x80
  7. Concat (128 + 128): 256x80x80
  8. C2f (256→128, n=1): 128x80x80 (P3_out)
  9. Conv (128→128, k=3, s=2): 40x40
  10. Concat (128 + 256): 384x40x40
  11. C2f (384→256, n=1): 256x40x40 (P4_final)
  12. Conv (256→256, k=3, s=2): 20x20
  13. Concat (256 + 256): 512x20x20
  14. C2f (512→512, n=1): 512x20x20 (P5_final)
- **Output**: P3_out (128x80x80), P4_final (256x40x40), P5_final (512x20x20)

### EfficientNet Neck

- **Input**: P3 (56x80x80), P4 (160x40x40), P5 (448x20x20)
- **Layers**:
  1. SPPF (448→448, k=5)
  2. Upsample (448, scale=2): 40x40
  3. Concat (448 + 160): 608x40x40
  4. C2f (608→160, n=1): 160x40x40
  5. Upsample (160, scale=2): 80x80
  6. Concat (160 + 56): 216x80x80
  7. C2f (216→56, n=1): 56x80x80 (P3_out)
  8. Conv (56→56, k=3, s=2, p=1)
  9. BatchNorm2d (56)
  10. SiLU
  11. Concat (56 + 160): 216x40x40
  12. C2f (216→160, n=1): 160x40x40 (P4_final)
  13. Conv (160→160, k=3, s=2, p=1)
  14. BatchNorm2d (160)
  15. SiLU
  16. Concat (160 + 448): 608x20x20
  17. C2f (608→448, n=1): 448x20x20 (P5_final)
- **Output**: P3_out (56x80x80), P4_final (160x40x40), P5_final (448x20x20)

## 3. Head

The detection head processes the neck outputs to produce bounding boxes, confidence scores, and class probabilities for 17 classes.

### YOLOv5 Head (Used for Both Backbones)

- **Input**: P3_out, P4_final, P5_final (dimensions depend on backbone)
- **Structure**:
  - Detect module (from Ultralytics YOLOv5)
  - Outputs per scale: (x, y, w, h, confidence, 17 class probabilities)
  - Anchors: Predefined for each scale (P3, P4, P5)
- **Output**:
  - Raw detections: [batch, num_anchors, (5 + 17)] per scale
  - Post-processed: Bounding boxes, scores, and labels (17 classes during training, mapped to 7 during inference)

## 4. Post-Processing

- **Non-Max Suppression (NMS)**: Filters overlapping detections (conf_thres=0.25, iou_thres=0.45).
- **Class Mapping**:
  - Classes 0-6: Direct mapping to 7 denominations ($1, $2, $5, $10, $20, $50, $100).
  - Classes 7-13: Denomination-specific features, mapped to corresponding denominations (0-6) with confidence boost (0.1×score).
  - Classes 14-16: Authenticity features (sign, text, thread), used for confidence adjustment (0.15×avg_score).
  - Confidence penalty: 20% reduction if no authenticity features detected for high-confidence detections (>0.5).
- **Output**: List of dictionaries per image with boxes, scores, labels, and denomination_scores (7 values).

## Implementation Details

- **Modular Structure**:
  - **Backbone**: `BaseBackbone`, `YOLOv5Backbone`, `EfficientNetBackbone`
  - **Neck**: `BaseNeck`, `YOLOv5Neck`, `EfficientNetNeck`
  - **Head**: `BaseHead`, `YOLOv5Head`
  - **Model**: `SmartCashYOLOv5Model`
  - **Factory**: `SmartCashYOLO`
- **Training**:
  - Phase 1: Freeze backbone, train head (default: 50 epochs, lr=1e-3).
  - Phase 2: Unfreeze backbone, fine-tune all (default: 100 epochs, lr=1e-4).
- **Dependencies**: `torch`, `ultralytics`, `timm` (for EfficientNet), `smartcash.common.logger`.

## Key Features

- **Flexibility**: Supports multiple YOLOv5 variants and EfficientNet-B4 via inheritance.
- **Modularity**: Separate classes for backbone, neck, and head ensure maintainability.
- **Currency Detection**: Optimized for fine-grained detection with authenticity verification.
- **Two-Phase Training**: Balances localization accuracy and feature learning.

This architecture provides a robust, scalable solution for currency detection, leveraging the strengths of YOLOv5 and EfficientNet-B4 while maintaining a clean, modular design.