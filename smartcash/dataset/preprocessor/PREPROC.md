# 📋 SmartCash Preprocessing Service Refactor Plan

## 🎯 **Overview**
Service preprocessing untuk normalisasi dan validasi dataset YOLOv5 dengan arsitektur EfficientNet-B4 backbone.

## 📁 **Output Structure**
```
data/
├── raw/{split}/
│   ├── images/          # Raw .jpg files
│   └── labels/          # Raw .txt files
├── preprocessing/
│   ├── images/          # .npy files (normalized float32)
│   └── labels/          # .txt files (copy dari raw)
└── invalid/{split}/
    ├── images/          # Invalid image files
    └── labels/          # Invalid label files
```

## 🔄 **Pipeline Flow**
1. **Raw** → **Validation**: Cek integritas file dan format
2. **Valid** → **Preprocessing**: Normalisasi ke .npy
3. **Invalid** → **Invalid folder**: Move files bermasalah

## 📂 **Directory Structure**

### 🏗️ **Main Module Structure**
```
smartcash/dataset/preprocessor/
├── __init__.py                 # Main API exports
├── service.py                  # PreprocessingService (main orchestrator)
├── core/
│   ├── __init__.py
│   ├── engine.py              # PreprocessingEngine (core logic)
│   └── validator.py           # ValidationEngine (file validation)
├── utils/
│   ├── __init__.py
│   ├── config_validator.py    # Config validation & defaults
│   ├── progress_bridge.py     # Progress tracking bridge
│   ├── file_processor.py      # File I/O handling
│   ├── file_scanner.py        # File scanning & discovery
│   ├── filename_manager.py    # Filename parsing & generation
│   ├── path_resolver.py       # Path resolution
│   └── cleanup_manager.py     # Cleanup utilities
└── validators/
    ├── __init__.py
    ├── image_validator.py      # Image integrity validation
    ├── label_validator.py      # Label format validation
    └── pair_validator.py       # Image-label pair validation
```

### ⚙️ **Configuration Structure**
```
configs/
└── preprocessing_config.yaml  # Standalone preprocessing config
```

## 🚀 **Main API Functions**

### `preprocess_dataset(config, target_split, progress_tracker, progress_callback)`
**Deskripsi**: Pipeline lengkap preprocessing
```python
result = preprocess_dataset(
    config=config,
    target_split='train',
    progress_tracker=tracker,
    progress_callback=callback
)
```

**Returns**:
```python
{
    'status': 'success',
    'total_processed': 150,      # Files processed
    'processing_time': 45.2,
    'validation_summary': {...},
    'phases': {
        'validation': {...},
        'preprocessing': {...}
    }
}
```

### `get_preprocessing_samples(config, target_split, max_samples, progress_tracker)`
**Deskripsi**: Ambil random samples untuk evaluasi
```python
samples = get_preprocessing_samples(
    config=config,
    target_split='train',
    max_samples=5
)
```

**Returns**:
```python
{
    'status': 'success',
    'samples': [
        {
            'uuid': 'a1b2c3d4-e5f6-...',
            'filename': 'rp_001000_uuid_001',
            'raw_image': [...],           # uint8 array
            'preprocessed_npy': [...],    # float32 array (normalized)
            'raw_path': 'data/raw/train/images/...',
            'npy_path': 'data/preprocessing/images/...npy'
        }
    ],
    'total_samples': 5,
    'target_split': 'train'
}
```

### `validate_dataset(config, target_split, progress_tracker)`
**Deskripsi**: Validasi dataset tanpa preprocessing
```python
result = validate_dataset(
    config=config,
    target_split='train'
)
```

**Returns**:
```python
{
    'status': 'success',
    'total_files': 500,
    'valid_files': 485,
    'invalid_files': 15,
    'validation_details': {
        'image_issues': 5,
        'label_issues': 7,
        'pair_issues': 3
    },
    'invalid_moved': True
}
```

### `cleanup_preprocessed_data(config, target_split, progress_tracker)`
**Deskripsi**: Hapus semua file preprocessed
```python
result = cleanup_preprocessed_data(
    config=config,
    target_split='train'  # None untuk all splits
)
```

### `get_preprocessing_status(config, progress_tracker)`
**Deskripsi**: Status file preprocessed
```python
status = get_preprocessing_status(config)
```

**Returns**:
```python
{
    'service_ready': True,
    'train_preprocessed': 150,   # .npy files
    'train_visualized': 150,     # .jpg visualization files
    'valid_preprocessed': 50,
    'valid_visualized': 50,
    'validation_enabled': True,
    'config': {...}
}
```

## 🔧 **Core Components Detail**

### 1. **PreprocessingService** (`service.py`)
```python
class PreprocessingService:
    """🎯 Service preprocessing dengan validation dan visualization"""
    
    def __init__(self, config, progress_tracker=None)
    def preprocess_and_visualize(self, target_split, progress_callback=None)
    def get_sampling(self, target_split, max_samples=5)
    def validate_dataset_only(self, target_split)
    def cleanup_preprocessed_data(self, target_split=None)
    def get_preprocessing_status(self)
```

### 2. **PreprocessingEngine** (`core/engine.py`)
```python
class PreprocessingEngine:
    """🎨 Core engine untuk preprocessing dengan clean progress tracking"""
    
    def preprocess_split(self, target_split, progress_callback=None)
    def _execute_preprocessing(self, files, target_split, progress_callback)
    def _preprocess_single_file(self, file_path, pipeline, output_dir)
    def _save_preprocessed_pair(self, npy_image, vis_image, labels, filename, output_dir)
```

### 3. **ValidationEngine** (`core/validator.py`)
```python
class ValidationEngine:
    """✅ Engine validasi dengan comprehensive checks"""
    
    def validate_split(self, target_split, progress_callback=None)
    def _validate_single_file(self, file_path)
    def _move_invalid_files(self, invalid_files, target_split)
```

### 4. **VisualizationEngine** (`core/visualizer.py`)
```python
class VisualizationEngine:
    """📊 Engine untuk create visualizations preprocessing results"""
    
    def create_visualization(self, original_image, normalized_image, metadata)
    def _add_comparison_info(self, vis_image, metadata)
    def _create_side_by_side_comparison(self, original, normalized)
```

## ⚙️ **Configuration**

### Preprocessing Parameters
```yaml
preprocessing:
  enabled: true
  validation:
    enabled: true             # Enable validation
    move_invalid: true        # Move invalid files
    fix_issues: false         # Try to fix issues
    
  normalization:
    method: 'minmax'          # minmax|standard|imagenet|none
    target_size: [640, 640]   # Fixed size untuk YOLO
    preserve_aspect_ratio: false
    denormalize: false        # Save as normalized (default)
    
  output:
    create_npy: true          # Save .npy files
    organize_by_split: true   # Organize by train/valid/test
```

### File Naming Pattern
```yaml
file_naming:
  preprocessed_pattern: 'pre_{nominal}_{uuid}_{increment}'
  preserve_uuid: true        # Maintain UUID consistency
```

## 📊 **File Formats**

### .npy Files (Preprocessed)
- **Format**: NumPy float32 array
- **Shape**: (640, 640, 3) - BGR channels
- **Range**: [0.0, 1.0] untuk minmax normalization
- **Usage**: Direct loading untuk YOLO training
```python
import numpy as np
preprocessed_image = np.load('pre_001000_uuid_001.npy')
# Shape: (640, 640, 3), dtype: float32, range: [0.0, 1.0]
```

### Visualization Files (.jpg)
- **Format**: JPEG image
- **Content**: Side-by-side comparison (original vs preprocessed)
- **Metadata**: Processing info overlay
- **Usage**: Visual validation dan monitoring

## 🔄 **Validation Components**

### Image Validator
```python
class ImageValidator:
    def validate_image_integrity(self, image_path)
    def validate_image_format(self, image_path)
    def validate_image_size(self, image_path)
```

### Label Validator
```python
class LabelValidator:
    def validate_label_format(self, label_path)
    def validate_bbox_coordinates(self, label_path)
    def validate_class_ids(self, label_path)
```

### Pair Validator
```python
class PairValidator:
    def validate_image_label_pair(self, image_path, label_path)
    def validate_filename_consistency(self, image_path, label_path)
```

## 🎯 **Best Practices**

### Recommended Configuration
```python
config = {
    'preprocessing': {
        'validation': {
            'enabled': True,          # Always validate
            'move_invalid': True,     # Clean invalid files
            'fix_issues': False       # Manual review recommended
        },
        'normalization': {
            'method': 'minmax',       # YOLO compatible
            'target_size': [640, 640], # Standard YOLO input
            'preserve_aspect_ratio': False
        },
        'visualization': {
            'enabled': True,          # Monitor preprocessing
            'comparison_mode': True,  # Visual validation
            'add_metadata': True      # Debug information
        }
    }
}
```

### Performance Optimization
- **Threading**: 4 workers optimal untuk I/O operations
- **Batch size**: 100 files per batch
- **Memory usage**: Max 4GB untuk large datasets
- **Progress tracking**: Granular updates setiap 5%

### Quality Control
- **Validation**: Comprehensive file integrity checks
- **Error handling**: Move invalid files, continue processing
- **Logging**: Detailed validation dan preprocessing reports
- **Visualization**: Visual validation untuk monitoring results

## 🚨 **Migration Notes**

### 1. **From Current Structure**
```python
# Current
from smartcash.dataset.preprocessing import preprocess_dataset

# New
from smartcash.dataset.preprocessor import preprocess_and_visualize
```

### 2. **Configuration Migration**
```yaml
# Old format support via inheritance
_base_: 'base_config.yaml'
preprocessing:
  # Override specific settings
```

### 3. **API Compatibility**
```python
# Provide backward compatibility wrappers
def preprocess_dataset(config, split='train'):
    """Legacy wrapper untuk existing code"""
    return preprocess_and_visualize(config, split)
```

## 🔍 **Implementation Checklist**

### Phase 1: Core Structure
- [ ] Create directory structure
- [ ] Implement PreprocessingService
- [ ] Setup config validation
- [ ] Create progress bridge

### Phase 2: Core Engines
- [ ] PreprocessingEngine implementation
- [ ] ValidationEngine implementation
- [ ] VisualizationEngine implementation
- [ ] Progress tracking integration

### Phase 3: Utilities
- [ ] File processors dan scanners
- [ ] Filename manager dengan pre_* pattern
- [ ] Path resolver
- [ ] Cleanup manager

### Phase 4: Validators
- [ ] Image validator
- [ ] Label validator
- [ ] Pair validator
- [ ] Validation reporter

### Phase 5: Integration
- [ ] UI integration
- [ ] Config handlers
- [ ] Testing with sample data
- [ ] Performance optimization

### Phase 6: Documentation
- [ ] API documentation
- [ ] Configuration guide
- [ ] Migration guide
- [ ] Best practices guide

## 💡 **Key Differences from Augmentor**

1. **File Naming**: `pre_{nominal}_{uuid}_{increment}` instead of `aug_*`
2. **Output Structure**: Separate visualizations folder
3. **Validation Focus**: Comprehensive validation pipeline
4. **Dual Output**: Both .npy (training) dan .jpg (visualization)
5. **Invalid Handling**: Move invalid files instead of skipping
6. **Metadata**: Rich visualization dengan processing metadata

## 🎁 **Added Value**

1. **Visual Monitoring**: Side-by-side comparison visualizations
2. **Quality Assurance**: Comprehensive validation pipeline
3. **Research Support**: Sampling capability untuk analysis
4. **Debug Support**: Rich metadata dan error reporting
5. **Flexibility**: Multiple normalization methods
6. **Scalability**: Parallel processing dengan progress tracking