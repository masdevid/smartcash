# 🔄 Integrasi Roboflow

## 📋 Overview

Dokumen ini menjelaskan cara menggunakan Roboflow untuk manajemen dataset SmartCash.

## 🔑 Setup API Key

1. Buat akun di [Roboflow](https://roboflow.com)
2. Dapatkan API key dari dashboard
3. Set environment variable:
```bash
export ROBOFLOW_API_KEY="your_api_key"
```

## 📤 Upload Dataset

```python
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")

# Create project
project = rf.create_project("smartcash")

# Upload dataset
project.upload(
    "data/processed/train",
    annotation_format="yolov5",
    split_ratio={"train": 0.7, "valid": 0.15, "test": 0.15}
)
```

## 📥 Download Dataset

```python
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")

# Get project
project = rf.project("smartcash")
dataset = project.version(1).download("yolov5")
```

## 🔄 Versioning

Setiap versi dataset di Roboflow harus memiliki:
1. Version tag (e.g., "v1.0.0")
2. Release notes
3. Preprocessing steps
4. Augmentation steps

## 🛠️ Preprocessing

Aktifkan preprocessing berikut di Roboflow:
- Auto-Orient
- Resize (640x640)
- Auto-Adjust Contrast
- Grayscale Normalization

## 🔄 Augmentation

Setup augmentasi berikut:
- Rotation (±30°)
- Brightness (±20%)
- Blur (up to 1px)
- Noise (up to 1%)
- Cutout (1 box)

## 📊 Export Format

Supported formats:
- YOLO v5 PyTorch
- COCO JSON
- Pascal VOC XML
- Tensorflow TFRecord

## 🔍 Dataset Health Check

Roboflow menyediakan metrics:
- Class distribution
- Bounding box sizes
- Image quality
- Annotation consistency

## 🤝 Collaboration

1. Invite team members
2. Set roles & permissions
3. Review annotations
4. Track changes

## 🔄 Workflow Integration

```python
# Training workflow
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")

# Get latest version
project = rf.project("smartcash")
dataset = project.version("latest").download("yolov5")

# Train model
!python train.py --data {dataset.location}/data.yaml

# Upload metrics
project.version(1).upload_metrics({
    "mAP": 0.956,
    "precision": 0.934,
    "recall": 0.921
})
```

## 📈 Monitoring & Analytics

Monitor via dashboard:
- Dataset growth
- Annotation quality
- Model performance
- Usage metrics

## 🚀 Best Practices

1. Version control dataset
2. Document preprocessing
3. Validate annotations
4. Monitor metrics
5. Regular backups
