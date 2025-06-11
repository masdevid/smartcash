# 📚 SmartCash Dataset Preprocessor API Documentation

## 🎯 Overview

Dataset preprocessor yang fokus pada YOLO normalization dengan validasi minimal. Dirancang untuk:
- ✅ Minimal validation (filename pattern + directory structure)
- 🎯 YOLOv5-compatible normalization
- 📊 Comprehensive statistics
- 🎲 Main banknotes sampling (7 classes)
- 🧹 Configurable cleanup

## 🚀 Main Preprocessing API

### `preprocess_dataset(config, progress_callback, splits)`

Proses dataset lengkap dengan YOLO normalization.

```python
from smartcash.dataset.preprocessor import preprocess_dataset

# Basic usage
result = preprocess_dataset({
    'data': {'dir': 'data'},
    'preprocessing': {
        'target_splits': ['train', 'valid'],
        'normalization': {'target_size': [640, 640]}
    }
})

# With progress tracking
def progress_handler(level, current, total, message):
    print(f"{level}: {current}/{total} - {message}")

result = preprocess_dataset(config, progress_callback=progress_handler)
```

**Returns:**
```python
{
    'success': bool,
    'message': str,
    'stats': {
        'processing_time_seconds': float,
        'input': {'total_images': int, 'splits_processed': int},
        'output': {'total_processed': int, 'success_rate': str},
        'configuration': {...}
    }
}
```

### `get_preprocessing_status(config)`

Check readiness dan status preprocessing.

```python
status = get_preprocessing_status({'data': {'dir': 'data'}})
print(f"Service ready: {status['service_ready']}")
```

## 🎯 Normalization API (Standalone)

### `normalize_for_yolo(image, preset, **kwargs)`

Standalone normalization untuk reuse di modules lain.

```python
from smartcash.dataset.preprocessor import normalize_for_yolo, denormalize_for_visualization

# Basic normalization
normalized, metadata = normalize_for_yolo(image, 'yolov5s')

# Custom parameters
normalized, metadata = normalize_for_yolo(
    image, 
    'default',
    target_size=[832, 832],
    preserve_aspect_ratio=True
)

# Denormalize untuk visualization
original = denormalize_for_visualization(normalized, metadata)
```

### Available Presets

- `'default'` - 640x640, standard YOLO
- `'yolov5s'` - 640x640, optimized untuk YOLOv5s
- `'yolov5m'` - 640x640, optimized untuk YOLOv5m  
- `'yolov5l'` - 832x832, optimized untuk YOLOv5l
- `'yolov5x'` - 1024x1024, optimized untuk YOLOv5x
- `'inference'` - 640x640, batch processing enabled

### `transform_coordinates_for_yolo(coords, metadata, reverse)`

Transform YOLO coordinates antara original dan normalized space.

```python
# Training: original → normalized
norm_coords = transform_coordinates_for_yolo(orig_coords, metadata, reverse=False)

# Inference: normalized → original  
orig_coords = transform_coordinates_for_yolo(norm_coords, metadata, reverse=True)
```

## 🎲 Samples API

### `get_samples(data_dir, split, max_samples, class_filter)`

Get samples dari main banknotes layer (7 classes).

```python
from smartcash.dataset.preprocessor import get_samples

# Get samples dari train split
samples = get_samples('data/preprocessed', 'train', max_samples=10)

for sample in samples['samples']:
    print(f"Class: {sample['primary_class']['display']}")
    print(f"File: {sample['npy_path']}")
    print(f"Size: {sample['file_size_mb']} MB")
```

**Returns:**
```python
{
    'success': bool,
    'samples': [
        {
            'npy_path': str,           # Path ke .npy file
            'filename': str,           # Nama file
            'file_size_mb': float,     # Ukuran file
            'class_ids': [int],        # Semua class IDs
            'class_names': {int: str}, # Class ID → nama mapping
            'primary_class': {         # Main banknote class info
                'class_id': int,       # 0-6 untuk main banknotes
                'nominal': str,        # '001000', '002000', etc
                'display': str,        # 'Rp1.000', 'Rp2.000', etc
                'value': int           # 1000, 2000, etc
            },
            'uuid': str,               # File UUID
            'denormalized_path': str   # Path untuk visualization (jika ada)
        }
    ]
}
```

### `generate_sample_previews(data_dir, output_dir, splits, max_per_class)`

Generate denormalized sample images untuk preview.

```python
result = generate_sample_previews(
    'data/preprocessed', 
    'data/samples',
    max_per_class=5
)
print(f"Generated {result['total_generated']} sample images")
```

## 📊 Statistics API

### `get_dataset_stats(data_dir, splits, include_details)`

Comprehensive dataset statistics.

```python
from smartcash.dataset.preprocessor import get_dataset_stats

stats = get_dataset_stats('data')
print(f"Total files: {stats['overview']['total_files']}")
print(f"Main banknotes: {stats['main_banknotes']['total_objects']}")
```

**Returns:**
```python
{
    'overview': {
        'total_splits': int,
        'total_files': int,
        'total_size_mb': float
    },
    'file_types': {
        'raw_images': int,           # Raw rp_*.jpg files
        'preprocessed_npy': int,     # pre_*.npy files  
        'augmented_npy': int,        # aug_*.npy files
        'sample_images': int         # sample_*.jpg files
    },
    'main_banknotes': {
        'total_objects': int,
        'active_classes': int,
        'percentage_of_total': float
    },
    'layers': {
        'l1_main': {'total_objects': int, 'active_classes': int},
        'l2_security': {...},
        'l3_micro': {...}
    },
    'file_sizes': {
        'avg_image_mb': float,
        'avg_npy_mb': float
    }
}
```

### `get_file_stats(directory, file_type)`

File-specific statistics.

```python
# All files
stats = get_file_stats('data/preprocessed', 'all')

# Specific type
stats = get_file_stats('data/preprocessed/train', 'preprocessed')
```

## 🧹 Cleanup API

### `cleanup_preprocessing_files(data_dir, target, splits, confirm)`

Configurable cleanup preprocessing artifacts.

```python
from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files

# Clean only preprocessing .npy files (default)
result = cleanup_preprocessing_files('data', 'preprocessed', confirm=True)

# Clean only generated samples  
result = cleanup_preprocessing_files('data', 'samples', confirm=True)

# Clean both
result = cleanup_preprocessing_files('data', 'both', confirm=True)
```

### `get_cleanup_preview(data_dir, target, splits)`

Preview files yang akan dihapus.

```python
preview = get_cleanup_preview('data', 'preprocessed')
print(f"Will remove {preview['total_files']} files ({preview['total_size_mb']} MB)")
```

## ⚙️ Configuration

### Default Configuration Structure

```python
{
    'preprocessing': {
        'enabled': True,
        'validation': {
            'enabled': False,          # Minimal validation only
            'filename_pattern': True,  # Check research format
            'directory_structure': True,
            'auto_fix': True          # Auto-rename + create dirs
        },
        'normalization': {
            'target_size': [640, 640],
            'pixel_range': [0, 1],
            'preserve_aspect_ratio': True,
            'pad_color': 114,
            'interpolation': 'linear'
        },
        'target_splits': ['train', 'valid'],
        'sample_size': 0  # 0 = process all
    },
    'data': {
        'dir': 'data',
        'preprocessed_dir': 'data/preprocessed',
        'samples_dir': 'data/samples'
    },
    'performance': {
        'batch_size': 32,
        'use_threading': True,
        'max_workers': 4
    }
}
```

### Research Filename Patterns

- **Raw**: `rp_001000_uuid_001.jpg`
- **Preprocessed**: `pre_rp_001000_uuid_001_01.npy`
- **Augmented**: `aug_rp_001000_uuid_001_02.npy`
- **Samples**: `sample_rp_001000_uuid_001.jpg`

### Main Banknote Classes (Layer l1_main)

| Class ID | Nominal | Display | Value |
|----------|---------|---------|-------|
| 0 | 001000 | Rp1.000 | 1000 |
| 1 | 002000 | Rp2.000 | 2000 |
| 2 | 005000 | Rp5.000 | 5000 |
| 3 | 010000 | Rp10.000 | 10000 |
| 4 | 020000 | Rp20.000 | 20000 |
| 5 | 050000 | Rp50.000 | 50000 |
| 6 | 100000 | Rp100.000 | 100000 |

## 🔌 Progress Tracker Integration

Compatible dengan Progress Tracker API untuk dual-level progress reporting:

```python
# UI integration
def create_preprocessing_with_progress(ui_components):
    def progress_callback(level, current, total, message):
        tracker = ui_components.get('progress_tracker')
        if tracker:
            if level == 'overall':
                tracker.update_overall(current, message)
            elif level == 'current':
                tracker.update_current(current, message)
    
    return preprocess_dataset(config, progress_callback)
```

## 📝 Usage Examples

### Complete Preprocessing Workflow

```python
from smartcash.dataset.preprocessor import *

# 1. Check status
status = get_preprocessing_status({'data': {'dir': 'data'}})
if not status['service_ready']:
    print("⚠️ Fix structure issues first")

# 2. Preview processing
preview = get_preprocessing_preview({
    'data': {'dir': 'data'},
    'preprocessing': {'target_splits': ['train', 'valid']}
})
print(f"Will process {preview['input_summary']['total_files']} files")

# 3. Process dataset
result = preprocess_dataset({
    'data': {'dir': 'data'},
    'preprocessing': {
        'target_splits': ['train', 'valid'],
        'normalization': {'target_size': [640, 640]}
    }
})

if result['success']:
    print(f"✅ Processed {result['stats']['output']['total_processed']} files")
    
    # 4. Get samples
    samples = get_samples('data/preprocessed', 'train', max_samples=10)
    print(f"Retrieved {len(samples['samples'])} samples")
    
    # 5. Generate previews
    previews = generate_sample_previews('data/preprocessed', 'data/samples')
    print(f"Generated {previews['total_generated']} preview images")
```

### Standalone Normalization for Other Modules

```python
# Custom augmentation module
from smartcash.dataset.preprocessor import normalize_for_yolo, denormalize_for_visualization

def custom_augment_and_normalize(image):
    # Apply custom augmentations
    augmented = apply_custom_augmentations(image)
    
    # Normalize untuk YOLO
    normalized, metadata = normalize_for_yolo(augmented, 'yolov5s')
    
    return normalized, metadata

# Inference engine
def prepare_for_inference(image, model_size='yolov5s'):
    normalized, metadata = normalize_for_yolo(image, model_size)
    return normalized, metadata
```