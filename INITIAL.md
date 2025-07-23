# YOLOv5 with EfficientNet-B4 for Rupiah Currency Detection

## FEATURE:

A modular deep learning pipeline for Rupiah currency detection, featuring a YOLOv5-based architecture with support for multiple backbones (CSPDarknet and EfficientNet-B4) through a flexible factory pattern. The implementation includes comprehensive data preprocessing, model training, and evaluation pipelines, all accessible through an intuitive Colab interface.

## ⚠️ Technical Constraints

- **Input Resolution**: Fixed at 640x640 pixels
- **Backbone Options**: CSPDarknet or EfficientNet-B4
- **Test Scenarios**:
  - Single currency detection
  - Multiple currency detection
  - Various lighting conditions
  - Different angles and orientations
- **Supported Formats**:
  - Images: JPG/PNG (RGB)
  - Annotations: YOLO format
- **Minimum Requirements**:
  - 8GB RAM, 2GB VRAM GPU
  - Python 3.8+, PyTorch 1.7.0+
  - CUDA 11.0+ for GPU acceleration

### Cells to Create:

1. **Setup & Configuration**
   - `cell_1_1_repo_clone.py`: Clone the repository and set up the environment (need no changes)
   - `cell_1_2_colab.py`: Configure Colab-specific settings and requirements
   - `cell_1_3_dependency.py`: Install and verify dependencies

2. **Data Processing**
   - `cell_2_1_downloader.py`: Download from Roboflow and organize the dataset
   - `cell_2_2_split.py`: Split data into training, validation, and test sets configuration cell
   - `cell_2_3_preprocess.py`: Preprocess images and annotations
   - `cell_2_4_augment.py`: Apply data augmentation techniques
   - `cell_2_5_visualize.py`: Visualize dataset samples and annotations

3. **Model Training**
   - `cell_3_1_pretrained.py`: Download -> Sync pretrained model for later use
   - `cell_3_2_backbone.py`: Set up EfficientNet-B4 backbone for YOLOv5
   - `cell_3_3_train.py`: Train the model with configurable parameters
   - `cell_3_4_evaluate.py`: Evaluate model performance on test set

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
   - System requirements and dependencies

2. **Backend Architecture** (`docs/backend/`)
   - Model Training Pipeline (`MODEL_TRAINING_PIPELINE.md`)
   - Model Evaluation Pipeline (`MODEL_EVALUATION_PIPELINE.md`)
   - Data Augmentation API (`AUGMENTATION_API.md`)
   - Data Downloader API (`DOWNLOADER_API.md`)
   - Preprocessing API (`PREPROCESSING_API.md`)
   - Model Analysis & Reporting (`MODEL_ANALYSIS_REPORTING_PIPELINE.md`)
   - Core Model Components (`MODEL_CORE.md`)

3. **Dataset Management** (`docs/dataset/`)
   - Dataset Preparation (`README.md`)
     - Image specifications: 640x640px, JPG/JPEG, RGB
     - Label format: YOLO txt format with class IDs for Rupiah denominations
     - Dataset structure and organization
   - Preprocessing Pipeline (`PREPROCESSING.md`)
     - Image resizing and normalization
     - Data augmentation techniques
     - Dataset splitting (train/validation/test)

4. **Technical Architecture** (`docs/technical/`)
   - System Architecture (`ARSITEKTUR.md`)
   - Model Specifications (`MODEL.md`)
   - Evaluation Methodology (`EVALUASI.md`)
   - Dataset Documentation (`DATASET.md`)

5. **Common Utilities** (`docs/common/`)
   - Environment & Configuration Management (`ENV_AND_CONFIG_MANAGER.md`)
   - Exception Handling (`EXCEPTIONS.md`)
   - Logging System (`LOGGER.md`)
   - Thread Pool Management (`THREADPOOLS.md`)
   - Worker Utilities (`WORKER_UTILS.md`)

6. **UI Components** (`docs/components/`)
   - Base Components (`BASE_COMPONENTS.md`)
   - Container System (`CONTAINERS.md`)
   - Dialog System (`DIALOG.md`)
   - Logging Components (`LOGGING.md`)
   - Progress Tracking (`PROGRESS_TRACKER.md`)
   - Widget Library (`WIDGETS.md`)

### External References
1. **YOLOv5 Official Documentation**
   - [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/)
   - Comprehensive documentation on YOLOv5 architecture and usage

2. **EfficientNet Paper**
   - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
   - Original paper detailing the EfficientNet architecture


## OTHER CONSIDERATIONS:
- On commit `~/smartcash/**` will copied to active directory `~/` so colab can imports using `from smartcash.ui.xxx` instead of `from smartcash.smartcash.ui.xxx` 
- Drive might limited in capacity, so no need excessive backups
