# 📚 SmartCash Dataset Preprocessor API Documentation

## 🎯 Overview

Dataset preprocessor yang fokus pada YOLO normalization dengan validasi minimal. Dirancang untuk:
- ✅ Minimal validation (filename pattern + directory structure)
- 🎯 YOLOv5-compatible normalization
- 📊 Comprehensive statistics
- 🎲 Main banknotes sampling (7 classes)
- 🧹 Configurable cleanup dengan progress tracking

## 🚀 Main Preprocessing API

### `preprocess_dataset(config, progress_callback, ui_components, splits)`

Proses dataset lengkap dengan YOLO normalization dan progress tracking.

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
    # level: 'overall' untuk "Preprocessing train 1/2"
    # level: 'current' untuk "train step 5/100 - Processing file.jpg"
    print(f"{level}: {current}/{total} - {message}")

result = preprocess_dataset(config, progress_callback=progress_handler, ui_components=ui_components)
```

**Returns:** Processing results dengan stats dan configuration

### `get_preprocessing_status(config, ui_components)`

Check readiness dan status preprocessing.

```python
status = get_preprocessing_status({'data': {'dir': 'data'}})
print(f"Service ready: {status['service_ready']}")
```

## 🎯 Normalization API (Standalone)

### `normalize_for_yolo(image, preset, **kwargs)`

Standalone normalization untuk reuse di modules lain.

```python
# Basic normalization
normalized, metadata = normalize_for_yolo(image, 'yolov5s')

# Custom parameters
normalized, metadata = normalize_for_yolo(image, 'default', target_size=[832, 832])

# Denormalize untuk visualization
original = denormalize_for_visualization(normalized, metadata)
```

**Available Presets:**
- `'default'`, `'yolov5s'`, `'yolov5m'` - 640x640
- `'yolov5l'` - 832x832  
- `'yolov5x'` - 1024x1024
- `'inference'` - 640x640 dengan batch processing

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
samples = get_samples('data/preprocessed', 'train', max_samples=10)

for sample in samples['samples']:
    print(f"Class: {sample['primary_class']['display']}")
    print(f"File: {sample['npy_path']}")
```

**Returns:** Samples dengan primary_class info (class_id 0-6, nominal, display, value), file paths, dan metadata.

### `generate_sample_previews(data_dir, output_dir, splits, max_per_class)`

Generate denormalized sample images untuk preview.

## 📊 Statistics API

### `get_dataset_stats(data_dir, splits, include_details)`

Comprehensive dataset statistics.

```python
stats = get_dataset_stats('data')
print(f"Total files: {stats['overview']['total_files']}")
print(f"Main banknotes: {stats['main_banknotes']['total_objects']}")
```

**File Types Tracked:**
- `raw_images` - Raw rp_*.jpg files
- `preprocessed_npy` - pre_*.npy files  
- `augmented_npy` - aug_*.npy files
- `sample_images` - sample_*.jpg files

**Layer Analysis:** l1_main (7 classes), l2_security, l3_micro

## 🧹 Cleanup API

### `cleanup_preprocessing_files(data_dir, target, splits, confirm, progress_callback, ui_components)`

Configurable cleanup dengan progress tracking.

```python
from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files

# Clean preprocessing files dengan progress
result = cleanup_preprocessing_files(
    'data', 'preprocessed', confirm=True,
    progress_callback=progress_handler, ui_components=ui_components
)

# Target options:
# 'preprocessed' - pre_*.npy + pre_*.txt files (default)
# 'samples' - sample_*.jpg files
# 'both' - preprocessing + samples
```

**Progress Format:**
- Overall: "Cleaning train 1/2"  
- Current: "train step 5/50 - Removing file.npy"

### `get_cleanup_preview(data_dir, target, splits)`

Preview files yang akan dihapus tanpa execute.

## ⚙️ Configuration

### Default Structure

```python
{
    'preprocessing': {
        'validation': {
            'enabled': False,          # Minimal validation only
            'filename_pattern': True,  # Auto-rename ke research format
            'auto_fix': True          # Auto-create directories
        },
        'normalization': {
            'target_size': [640, 640],
            'preserve_aspect_ratio': True,
            'pixel_range': [0, 1]
        },
        'target_splits': ['train', 'valid']
    },
    'data': {
        'dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    }
}
```

### Research Filename Patterns

- **Raw**: `rp_001000_uuid.jpg`
- **Preprocessed**: `pre_rp_001000_uuid.npy`
- **Samples**: `sample_rp_001000_uuid.jpg`

### Main Banknote Classes

| Class ID | Nominal | Display | Value |
|----------|---------|---------|-------|
| 0-6 | 001000-100000 | Rp1K-100K | 1000-100000 |

## 🔌 Progress Integration

Compatible dengan Progress Tracker API:

```python
def progress_callback(level, current, total, message):
    tracker = ui_components.get('progress_tracker')
    if tracker:
        if level == 'overall':
            tracker.update_overall(current, message)
        elif level == 'current':
            tracker.update_current(current, message)
```

## 📝 Complete Workflow Example

```python
from smartcash.dataset.preprocessor import *

# 1. Check status
status = get_preprocessing_status({'data': {'dir': 'data'}})

# 2. Process dengan progress
result = preprocess_dataset(config, progress_callback=progress_handler)

# 3. Get samples
samples = get_samples('data/preprocessed', 'train', max_samples=10)

# 4. Cleanup jika diperlukan
cleanup_result = cleanup_preprocessing_files('data', 'preprocessed', confirm=True)