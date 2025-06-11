# 📋 SmartCash Dataset Preprocessor API Documentation

## 🎯 Overview

SmartCash Dataset Preprocessor adalah modul terpusat untuk preprocessing dataset YOLO dengan arsitektur yang telah dikonsolidasi menggunakan prinsip **Single Responsibility Principle (SRP)** dan **DRY (Don't Repeat Yourself)**. Modul ini menyediakan normalisasi khusus YOLO, validasi komprehensif, dan integrasi progress tracking dengan UI components.

## 🏗️ Arsitektur Terkonsolidasi

### Struktur Modular
```
smartcash/dataset/preprocessor/
├── __init__.py                    # Main exports & enhanced functions
├── service.py                     # Main preprocessing service
├── core/
│   ├── engine.py                  # Simplified processing engine
│   └── validator.py               # Simplified validation wrapper
└── utils/                         # Consolidated SRP utilities
    ├── file_operations.py         # File I/O & scanning operations
    ├── validation_core.py         # Core validation logic
    ├── path_manager.py            # Path management & directory ops
    ├── progress_bridge.py         # Progress tracking integration
    ├── metadata_manager.py        # Filename & metadata management
    ├── normalization.py           # YOLO-specific normalization
    └── config_validator.py        # Configuration validation
```

### Keunggulan Konsolidasi
- **🔧 SRP Compliance**: Setiap utils file memiliki tanggung jawab tunggal
- **📦 DRY Implementation**: Eliminasi duplikasi kode ~40%
- **🌉 Progress Bridge**: Native compatibility dengan `ui/components/progress_tracker`
- **🎯 YOLO-Specific**: Optimized untuk object detection workflows
- **🔄 Backward Compatibility**: Wrapper untuk existing code

## 🚀 Main API Functions

### 1. Core Preprocessing Function

```python
def preprocess_dataset(
    config: Dict[str, Any], 
    ui_components: Dict[str, Any] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    🚀 Enhanced preprocessing pipeline dengan dual progress tracking
    
    Args:
        config: Preprocessing configuration (akan divalidasi otomatis)
        ui_components: UI components untuk progress integration (opsional)
        progress_callback: Callback function (level, current, total, message)
        
    Returns:
        {
            'success': bool,
            'message': str,
            'stats': Dict[str, Any],
            'processing_time': float,
            'configuration': Dict[str, Any]
        }
    """
```

**Contoh Penggunaan:**
```python
from smartcash.dataset.preprocessor import preprocess_dataset

# Basic usage
config = {
    'preprocessing': {
        'target_splits': ['train', 'valid'],
        'normalization': {
            'enabled': True,
            'target_size': [640, 640],
            'preserve_aspect_ratio': True
        }
    }
}

result = preprocess_dataset(config)
print(f"Success: {result['success']}")
print(f"Processing time: {result['processing_time']:.2f}s")

# With UI integration
result = preprocess_dataset(config, ui_components=my_ui_components)

# With custom progress callback
def my_progress(level, current, total, message):
    print(f"{level}: {current}/{total} - {message}")

result = preprocess_dataset(config, progress_callback=my_progress)
```

### 2. Validation Functions

```python
def validate_dataset(
    config: Dict[str, Any], 
    target_split: str = "train",
    ui_components: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    🔍 Enhanced dataset validation dengan detailed reporting
    
    Returns:
        {
            'success': bool,
            'message': str,
            'target_split': str,
            'summary': {
                'total_images': int,
                'valid_images': int,
                'validation_rate': str,
                'class_distribution': Dict[int, int],
                'common_errors': List[str]
            }
        }
    """
```

### 3. Utility Functions

```python
def get_preprocessing_samples(
    config: Dict[str, Any], 
    target_split: str = "train",
    max_samples: int = 5,
    ui_components: Dict[str, Any] = None
) -> Dict[str, Any]:
    """🎲 Get dataset samples untuk preview"""

def cleanup_preprocessed_data(
    config: Dict[str, Any], 
    target_split: str = None,
    ui_components: Dict[str, Any] = None
) -> Dict[str, Any]:
    """🧹 Cleanup preprocessed data dengan statistics"""

def get_preprocessing_status(
    config: Dict[str, Any],
    ui_components: Dict[str, Any] = None
) -> Dict[str, Any]:
    """📊 Get comprehensive system status"""
```

### 4. Specialized Functions

```python
def preprocess_for_yolo_training(
    config: Dict[str, Any],
    ui_components: Dict[str, Any] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """🎯 Preprocessing optimized untuk YOLO training"""

def preprocess_single_split(
    config: Dict[str, Any], 
    split: str,
    ui_components: Dict[str, Any] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """🎯 Process single split saja"""
```

## 🔧 Consolidated Utils API

### 1. File Operations

```python
from smartcash.dataset.preprocessor.utils import FileOperations, create_file_operations

# Create instance
file_ops = create_file_operations(config)

# Core operations
image = file_ops.read_image(image_path, target_size=(640, 640))
success = file_ops.write_image(output_path, image, quality=90)
normalized_success = file_ops.save_normalized_array(npy_path, array, metadata)

# Batch operations
results = file_ops.batch_read_images(image_paths, progress_callback=callback)
stats = file_ops.batch_copy_files(file_pairs, progress_callback=callback)

# File scanning
image_files = file_ops.scan_images(directory, recursive=False)
label_files = file_ops.scan_labels(directory)
pairs = file_ops.find_image_label_pairs(img_dir, label_dir)
orphans = file_ops.find_orphan_files(img_dir, label_dir)
```

### 2. Validation Core

```python
from smartcash.dataset.preprocessor.utils import ValidationCore, ValidationResult

validator = ValidationCore(config)

# Individual validation
img_result: ValidationResult = validator.validate_image(image_path)
label_result: ValidationResult = validator.validate_label(label_path, image_size)
pair_result: ValidationResult = validator.validate_pair(img_path, label_path)

# Batch validation
results = validator.batch_validate_images(image_paths, progress_callback)
pair_results = validator.batch_validate_pairs(image_paths, progress_callback)

# Results inspection
print(f"Valid: {img_result.is_valid}")
print(f"Errors: {img_result.errors}")
print(f"Warnings: {img_result.warnings}")
print(f"Stats: {img_result.stats}")
```

### 3. Path Management

```python
from smartcash.dataset.preprocessor.utils import PathManager, create_path_manager

path_manager = create_path_manager(config)

# Source paths
img_dir, label_dir = path_manager.get_source_paths('train')
source_dir = path_manager.get_source_split_dir('train')

# Output paths  
out_img_dir, out_label_dir = path_manager.get_output_paths('train')
output_dir = path_manager.get_output_split_dir('train')

# Structure operations
validation = path_manager.validate_source_structure(['train', 'valid'])
creation_results = path_manager.create_output_structure(['train', 'valid'])
aux_results = path_manager.create_auxiliary_dirs()

# Cleanup operations
cleanup_stats = path_manager.cleanup_output_dirs(['train'], confirm=True)
temp_cleaned = path_manager.cleanup_temp_dirs()
```

### 4. YOLO Normalization

```python
from smartcash.dataset.preprocessor.utils import YOLONormalizer, create_yolo_normalizer

normalizer = create_yolo_normalizer(config)

# YOLO preprocessing
normalized_image, metadata = normalizer.preprocess_for_yolo(image)

# Metadata contains:
# - original_shape: (h, w)
# - target_shape: (640, 640)  
# - scale_info: {'scale': float, 'pad_x': int, 'pad_y': int}
# - normalized: bool

# Coordinate transformation
transformed_bboxes = normalizer.transform_bbox_coordinates(
    bboxes, metadata['scale_info'], reverse=False
)
```

### 5. Progress Bridge

```python
from smartcash.dataset.preprocessor.utils import ProgressBridge, create_compatible_bridge

# Basic bridge
bridge = ProgressBridge()
bridge.register_callback(my_callback)

# UI-compatible bridge
bridge = create_compatible_bridge(ui_components)

# Progress updates
bridge.update('overall', 50, 100, "Processing...")
bridge.update_current(25, 50, "Current operation")
bridge.increment('overall', message="Step complete")

# State management
bridge.complete_level('current', "Operation finished")
bridge.reset_level('overall', total=200)
state = bridge.get_progress_state('overall')
```

### 6. Metadata Management

```python
from smartcash.dataset.preprocessor.utils import MetadataManager, FileMetadata

metadata_manager = MetadataManager(config)

# Filename parsing
metadata: FileMetadata = metadata_manager.parse_filename('rp_001000_uuid_001.jpg')
print(f"Type: {metadata.file_type}")
print(f"Nominal: {metadata.nominal}")
print(f"Denomination: {metadata.denomination_info}")

# Filename generation
preprocessed_name = metadata_manager.generate_preprocessed_filename(
    source_filename, variance=1
)
augmented_name = metadata_manager.generate_augmented_filename(
    source_filename, variance=2
)

# Batch operations
parsed_results = metadata_manager.batch_parse_filenames(filename_list)
generated_names = metadata_manager.batch_generate_preprocessed(source_list)

# Statistics
dataset_stats = metadata_manager.extract_dataset_statistics(all_filenames)
```

## 🔄 Progress Tracking Integration

### Compatibility dengan UI Progress Tracker

```python
# Automatic integration dengan UI components
result = preprocess_dataset(config, ui_components=ui_components)

# Manual progress callback
def progress_handler(level, current, total, message):
    """
    level: 'overall', 'step', 'current', 'primary'
    current: current progress value
    total: total progress value  
    message: status message
    """
    print(f"{level}: {current}/{total} ({current/total*100:.1f}%) - {message}")

result = preprocess_dataset(config, progress_callback=progress_handler)
```

### Progress Levels
- **overall**: Overall preprocessing progress (0-100%)
- **current**: Current operation progress 
- **step**: Step-wise progress (for multi-step operations)
- **primary**: Primary progress (single-level compatibility)

## ⚙️ Configuration Schema

### Comprehensive Config Structure

```yaml
preprocessing:
  enabled: true
  target_splits: ['train', 'valid']  # or 'all'
  output_dir: 'data/preprocessed'
  
  validation:
    enabled: true
    move_invalid: true
    check_image_quality: true
    check_labels: true
    check_coordinates: true
    
  normalization:
    enabled: true
    target_size: [640, 640]
    preserve_aspect_ratio: true
    normalize_pixel_values: true
    
  output:
    create_npy: true
    organize_by_split: true

performance:
  batch_size: 32
  use_gpu: true
  threading:
    io_workers: 8

data:
  dir: 'data'
  local:
    train: 'data/train'
    valid: 'data/valid'
    test: 'data/test'

file_naming:
  preprocessed_pattern: 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}'
  preserve_uuid: true
```

### Config Validation

```python
from smartcash.dataset.preprocessor import validate_preprocessing_config

# Automatic validation
validated_config = validate_preprocessing_config(my_config)

# Get default config
default_config = get_default_preprocessing_config()

# Config summary untuk UI
summary = get_preprocessing_config_summary(config)
```

## 🎯 YOLO-Specific Features

### 1. YOLO Normalization
- **Target Size**: Resize ke 640x640 dengan padding
- **Aspect Ratio**: Preserve aspect ratio dengan gray padding
- **Pixel Normalization**: 0-255 → 0.0-1.0 range
- **Coordinate Transform**: Automatic bbox coordinate adjustment

### 2. Output Format
- **Images**: `.npy` files (normalized float32 arrays)
- **Labels**: `.txt` files (YOLO format, coordinates adjusted)
- **Metadata**: Scale dan padding info untuk coordinate transformation

### 3. File Naming Convention
- **Raw**: `rp_{nominal}_{uuid}_{sequence}.ext`
- **Preprocessed**: `pre_rp_{nominal}_{uuid}_{sequence}_{variance}.npy`
- **Labels**: Matching filename dengan `.txt` extension

## 🔧 Convenience Functions

### One-liner Operations

```python
from smartcash.dataset.preprocessor.utils import (
    read_image_safe, write_image_safe, scan_image_files,
    validate_image_safe, validate_source_safe, 
    parse_filename_safe, preprocess_image_for_yolo
)

# Safe operations dengan error handling
image = read_image_safe(image_path, target_size=(640, 640))
success = write_image_safe(output_path, image, quality=90)
image_files = scan_image_files(directory, recursive=True)

# Quick validation
is_valid = validate_image_safe(image_path)
structure_valid = validate_source_safe(config, ['train', 'valid'])

# Quick metadata parsing
metadata = parse_filename_safe(filename)
nominal = extract_nominal_safe(filename)

# Quick YOLO preprocessing
normalized, metadata = preprocess_image_for_yolo(image, target_size=(640, 640))
```

## 📊 Error Handling & Logging

### Error Response Format
```python
{
    'success': False,
    'message': 'Error description',
    'stats': {},
    'error_type': 'validation|processing|io',
    'details': {...}  # Optional detailed error info
}
```

### Progress Error Handling
- **Silent Fail**: Progress callbacks gagal tidak akan mengganggu proses utama
- **Fallback**: Automatic fallback ke console logging jika UI tidak tersedia  
- **Error Recovery**: Continue processing meski ada error pada individual files

## 🔄 Migration dari Legacy Code

### Automatic Compatibility
```python
# Legacy imports tetap bekerja
from smartcash.dataset.preprocessor.utils import (
    ImageValidator, FileProcessor, PathResolver, FilenameManager
)

# Factory functions tersedia
validator = create_image_validator(config)
processor = create_file_processor(config)
```

### Enhanced Alternatives
```python
# Gunakan consolidated utils untuk better performance
from smartcash.dataset.preprocessor.utils import (
    ValidationCore, FileOperations, PathManager, MetadataManager
)

# Enhanced functionality
validation_core = ValidationCore(config)  # Replaces all validators
file_ops = FileOperations(config)         # Replaces FileProcessor + FileScanner
path_mgr = PathManager(config)            # Replaces PathResolver + directory ops
```

## 🧪 Testing & Development

### Configuration Testing
```python
# Test config compatibility
compatibility = check_preprocessing_compatibility()
print(f"Features available: {compatibility['enhanced_features']}")

# Test configuration
status = get_preprocessing_status(config)
print(f"Service ready: {status['service_ready']}")
```

### Sample Testing
```python
# Quick dataset preview
samples = get_preprocessing_samples(config, 'train', max_samples=3)
for sample in samples['samples']:
    print(f"File: {sample['filename']} - Size: {sample['dimensions']}")
```

## 📈 Performance Optimization

### Batch Processing
- **Threading**: I/O operations menggunakan ThreadPoolExecutor
- **Batch Size**: Configurable batch size untuk memory optimization
- **Progress Throttling**: 100ms default throttling untuk smooth UI updates

### Memory Management
- **Streaming**: Large datasets diproses secara streaming
- **Cleanup**: Automatic temporary file cleanup
- **Caching**: Metadata caching untuk repeated operations

---

## 🎯 Best Practices

1. **Selalu gunakan config validation** sebelum processing
2. **Provide progress callback** untuk long-running operations  
3. **Check preprocessing status** sebelum memulai processing
4. **Use specialized functions** untuk specific use cases (YOLO training, single split)
5. **Handle UI integration** dengan `ui_components` parameter
6. **Validate source structure** sebelum processing
7. **Use cleanup functions** untuk maintain disk space

Dokumentasi ini mencakup API lengkap untuk SmartCash Dataset Preprocessor yang telah dikonsolidasi dengan architecture yang lebih clean dan maintainable.