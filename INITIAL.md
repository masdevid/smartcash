# YOLOv5 with EfficientNet-B4 for Rupiah Currency Detection

## FEATURE:

Build Colab IPython widget interfaces for YOLOv5 object detection algorithm by integrating it with the EfficientNet-B4 architecture as a backbone to improve the detection of currency denomination in Rupiah.

## EXAMPLES:

1. **Model Construction**
   - `smartcash/model/core/model_builder.py`: Implements `ModelBuilder` for constructing SmartCash models with various backbones
   - `smartcash/model/core/yolo_head.py`: Contains `YOLOHead` implementation for currency detection
   - `smartcash/model/core/checkpoint_manager.py`: Handles model saving and loading

2. **Training Pipeline**
   - `smartcash/model/training/training_service.py`: Main training service implementation
   - `smartcash/model/training/optimizer_factory.py`: Factory for creating optimizers
   - `smartcash/model/training/loss_manager.py`: Manages loss computation and metrics

3. **Dataset Handling**
   - `smartcash/dataset/components/datasets/yolo_dataset.py`: Implements `YOLODataset` for standard object detection
   - `smartcash/dataset/components/datasets/multilayer_dataset.py`: Advanced dataset for multi-layer processing
   - `smartcash/dataset/preprocessor/service.py`: Main preprocessing service for dataset preparation
   - `smartcash/dataset/augmentor/`: Contains various data augmentation techniques
   - `smartcash/dataset/organizer/`: Tools for dataset organization and management

4. **Model Architecture**
   - `SmartCashYOLO` class in `model_builder.py`: Complete YOLO model with modular architecture
   - Supports various backbones through `smartcash/model/utils/backbone_factory.py`

5. **Data Processing**
   - `smartcash/dataset/downloader/`: Tools for downloading and managing dataset files
   - `smartcash/dataset/preprocessor/`: Comprehensive preprocessing pipeline
   - `smartcash/dataset/evaluation/`: Tools for dataset evaluation and analysis

## DOCUMENTATION:

### Core Documentation
1. **Project Overview** (`docs/README.md`)
   - Research objectives and methodology for Rupiah currency detection
   - Implementation of YOLOv5 and EfficientNet-B4 hybrid approach
   - Performance optimization strategies

2. **Dataset Preparation** (`docs/dataset/README.md`)
   - Image specifications: 640x640px, JPG/JPEG, RGB
   - Label format: YOLO txt format with class IDs for Rupiah denominations
   - Dataset structure and organization

3. **Preprocessing Pipeline** (`docs/dataset/PREPROCESSING.md`)
   - Image resizing and normalization
   - Data augmentation techniques
   - Dataset splitting (train/validation/test)

4. **Technical Architecture** (`docs/technical/`)
   - System architecture (`ARSITEKTUR.md`)
   - Model specifications (`MODEL.md`)
   - Evaluation methodology (`EVALUASI.md`)

### External References
1. **YOLOv5 Official Documentation**
   - [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/)
   - Comprehensive documentation on YOLOv5 architecture and usage

2. **EfficientNet Paper**
   - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
   - Original paper detailing the EfficientNet architecture


## OTHER CONSIDERATIONS:


