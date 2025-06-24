# 📚 SmartCash Dataset Preprocessor API Documentation

## 🎯 Overview

Dataset preprocessor yang fokus pada YOLO normalization dengan validasi minimal dan file naming manager. Dirancang untuk:
- ✅ Minimal validation (filename pattern + directory structure)
- 🎯 YOLOv5-compatible normalization dengan preset support
- 📊 Comprehensive statistics dengan layer analysis
- 🎲 Main banknotes sampling (7 classes: 0-6)
- 🧹 Configurable cleanup dengan progress tracking
- 📁 Research filename format dengan FileNamingManager

## 🚀 Main Preprocessing API

### `preprocess_dataset(config, progress_callback, ui_components, splits)`

Proses dataset lengkap dengan YOLO normalization dan progress tracking menggunakan PreprocessingService.

```python
from smartcash.dataset.preprocessor import preprocess_dataset

# Basic usage dengan default config
result = preprocess_dataset({
    'data': {'dir': 'data', 'preprocessed_dir': 'data/preprocessed'},
    'preprocessing': {
        'target_splits': ['train', 'valid'],
        'normalization': {'target_size': [640, 640]},
        'validation': {'enabled': False}  # Minimal validation
    }
})

# With progress tracking untuk UI integration
def progress_handler(level, current, total, message):
    # level: 'overall' untuk "Preprocessing train 1/2"
    # level: 'current' untuk "train step 5/100 - Processing file.jpg"
    print(f"{level}: {current}/{total} - {message}")

result = preprocess_dataset(
    config, 
    progress_callback=progress_handler, 
    ui_components=ui_components
)
```

**Returns:** Processing results dengan stats dan configuration
- `success`: Boolean status
- `message`: Status message dengan emoji
- `stats`: Comprehensive processing statistics
- `configuration`: Applied preprocessing config

### `get_preprocessing_status(config, ui_components)`

Check readiness dan status preprocessing dengan enhanced file scanning.

```python
status = get_preprocessing_status({'data': {'dir': 'data'}})
print(f"Service ready: {status['service_ready']}")
print(f"Raw images: {status['file_statistics']['train']['raw_images']}")
print(f"Preprocessed: {status['file_statistics']['train']['preprocessed_files']}")
```

**Returns:** Status dengan file counts per split menggunakan FileNamingManager patterns

## 🎯 Normalization API (Standalone)

### `normalize_for_yolo(image, preset, **kwargs)`

Standalone normalization untuk reuse di modules lain dengan preset support.

```python
from smartcash.dataset.preprocessor.api.normalization_api import normalize_for_yolo

# Basic normalization dengan preset
normalized, metadata = normalize_for_yolo(image, 'yolov5s')

# Custom parameters override
normalized, metadata = normalize_for_yolo(
    image, 
    'yolov5l', 
    target_size=[832, 832],
    preserve_aspect_ratio=True
)

# Denormalize untuk visualization
from smartcash.dataset.preprocessor.api.normalization_api import denormalize_for_visualization
original = denormalize_for_visualization(normalized, metadata)
```

**Available Presets:**
- `'default'`, `'yolov5s'`, `'yolov5m'` - 640x640 dengan standard settings
- `'yolov5l'` - 832x832 untuk larger model
- `'yolov5x'` - 1024x1024 untuk maximum accuracy
- `'inference'` - 640x640 dengan batch processing optimizations

### `transform_coordinates_for_yolo(coords, metadata, reverse)`

Transform YOLO coordinates antara original dan normalized space.

```python
# Training: original → normalized
norm_coords = transform_coordinates_for_yolo(orig_coords, metadata, reverse=False)
# Inference: normalized → original  
orig_coords = transform_coordinates_for_yolo(norm_coords, metadata, reverse=True)
```

### `normalize_image_file(image_path, output_path, preset, save_metadata)`

File-based normalization dengan FileNamingManager integration.

```python
result = normalize_image_file(
    'data/train/images/rp_001000_uuid.jpg',
    preset='yolov5s',
    save_metadata=True
)
# Output: 'data/preprocessed/train/images/pre_001000_uuid.npy'
```

## 🎲 Samples API

### `get_samples(data_dir, split, max_samples, class_filter)`

Get samples dari main banknotes layer (7 classes) dengan enhanced metadata.

```python
from smartcash.dataset.preprocessor.api.samples_api import get_samples

samples = get_samples('data/preprocessed', 'train', max_samples=10)

for sample in samples['samples']:
    print(f"Class: {sample['primary_class']['display']}")  # "Rp1K"
    print(f"File: {sample['npy_path']}")
    print(f"UUID: {sample['uuid']}")
    print(f"Nominal: {sample['nominal']}")  # "001000"
```

**Returns:** Enhanced samples dengan:
- `primary_class`: Main class info (class_id 0-6, nominal, display, value)
- `file_paths`: npy_path, label_path, denormalized_path
- `metadata`: uuid, nominal, file_size_mb
- `class_distribution`: All classes in file

### `generate_sample_previews(data_dir, output_dir, splits, max_per_class)`

Generate denormalized sample images untuk preview dengan naming convention.

```python
result = generate_sample_previews(
    'data/preprocessed',
    'data/samples',
    splits=['train', 'valid'],
    max_per_class=5
)
# Output: sample_pre_001000_uuid.jpg
```

### `get_class_samples(data_dir, class_id, split, max_samples)`

Get samples untuk specific main banknote class.

```python
# Get Rp10K samples (class_id=3)
rp10k_samples = get_class_samples('data/preprocessed', 3, 'train', 5)
```

## 📊 Statistics API

### `get_dataset_stats(data_dir, splits, include_details)`

Comprehensive dataset statistics dengan layer analysis.

```python
from smartcash.dataset.preprocessor.api.stats_api import get_dataset_stats

stats = get_dataset_stats('data', include_details=True)
print(f"Total files: {stats['overview']['total_files']}")
print(f"Main banknotes: {stats['main_banknotes']['total_objects']}")
print(f"Layer distribution: {stats['layers']}")
```

**Enhanced File Types Tracked:**
- `raw_images` - Raw rp_*.jpg files (source)
- `preprocessed_npy` - pre_*.npy files (normalized arrays)
- `augmented_npy` - aug_*.npy files (augmented data)
- `sample_images` - sample_*.jpg files (denormalized previews)

**Layer Analysis:** 
- `l1_main` - Main banknotes (classes 0-6)
- `l2_security` - Security features (classes 7-13)
- `l3_micro` - Micro text (classes 14-16)

### `get_class_distribution_stats(data_dir, splits)`

Detailed class distribution dengan balance analysis.

```python
class_stats = get_class_distribution_stats('data')
print(f"Balanced: {class_stats['class_balance']['balanced']}")
print(f"Imbalance ratio: {class_stats['class_balance']['imbalance_ratio']}")
```

### `export_stats_report(data_dir, output_path, splits)`

Export comprehensive statistics report ke JSON.

## 🧹 Cleanup API

### `cleanup_preprocessing_files(data_dir, target, splits, confirm, progress_callback, ui_components)`

Configurable cleanup dengan FileNamingManager patterns dan progress tracking.

```python
from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files

# Clean preprocessing files dengan progress
result = cleanup_preprocessing_files(
    'data', 
    target='preprocessed',  # or 'augmented', 'samples', 'both'
    splits=['train', 'valid'],
    confirm=True,
    progress_callback=progress_handler, 
    ui_components=ui_components
)

print(f"Files removed: {result['files_removed']}")
```

**Target Options:**
- `'preprocessed'` - pre_*.npy + pre_*.txt + metadata files (default)
- `'augmented'` - aug_*.npy + aug_*.txt files dengan variance pattern
- `'samples'` - sample_*.jpg files (denormalized previews)
- `'both'` - preprocessing + samples files

**Progress Format:**
- Overall: "Cleaning train 1/2"  
- Current: "train step 5/50 - Removing pre_001000_uuid.npy"

### `get_cleanup_preview(data_dir, target, splits)`

Preview files yang akan dihapus tanpa execute.

```python
preview = get_cleanup_preview('data', 'preprocessed')
print(f"Will remove {preview['total_files']} files ({preview['total_size_mb']} MB)")
```

### `cleanup_empty_directories(data_dir)`

Cleanup empty directories setelah file removal.

## ⚙️ Configuration

### Enhanced Default Structure

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
            'pixel_range': [0, 1],
            'method': 'minmax'
        },
        'target_splits': ['train', 'valid']
    },
    'data': {
        'dir': 'data',
        'preprocessed_dir': 'data/preprocessed'
    },
    'performance': {
        'batch_size': 32,
        'use_gpu': True,
        'threading': {
            'io_workers': 8,
            'cpu_workers': None  # Auto-detect
        }
    }
}
```

### FileNamingManager Patterns

Research filename patterns dengan UUID consistency dan variance support:

- **Raw**: `rp_{nominal}_{uuid}.jpg` (e.g., `rp_001000_abc123.jpg`)
- **Preprocessed**: `pre_{nominal}_{uuid}.npy` (e.g., `pre_001000_abc123.npy`)
- **Augmented**: `aug_{nominal}_{uuid}_{variance}.npy` (e.g., `aug_001000_abc123_001.npy`)
- **Samples**: `sample_pre_{nominal}_{uuid}.jpg` (e.g., `sample_pre_001000_abc123.jpg`)
- **Augmented Samples**: `sample_aug_{nominal}_{uuid}_{variance}.jpg`

### Main Banknote Classes (Enhanced)

| Class ID | Nominal | Display | Value | Layer |
|----------|---------|---------|-------|-------|
| 0 | 001000 | Rp1K | 1000 | l1_main |
| 1 | 002000 | Rp2K | 2000 | l1_main |
| 2 | 005000 | Rp5K | 5000 | l1_main |
| 3 | 010000 | Rp10K | 10000 | l1_main |
| 4 | 020000 | Rp20K | 20000 | l1_main |
| 5 | 050000 | Rp50K | 50000 | l1_main |
| 6 | 100000 | Rp100K | 100000 | l1_main |

## 🔌 Progress Integration

Compatible dengan Progress Tracker API dan UI components:

```python
def progress_callback(level, current, total, message):
    tracker = ui_components.get('progress_tracker')
    if tracker:
        if level == 'overall':
            tracker.update_overall((current/total)*100, message)
        elif level == 'current':
            tracker.update_current((current/total)*100, message)
```

## 🏗️ Service Architecture

### PreprocessingService

Core service dengan phase-based processing:

```python
# Phase 1: Validation (20%)
- Directory structure check dengan auto-fix
- Filename validation dengan auto-rename
- Minimal validation untuk performance

# Phase 2: Processing (70%)
- Split-by-split processing dengan progress tracking
- YOLO normalization menggunakan preset
- Metadata preservation dengan FileNamingManager

# Phase 3: Finalization (10%)
- Statistics compilation
- Performance metrics calculation
- Final status reporting
```

### Integration Components

- **DirectoryValidator**: Structure validation dengan auto-fix
- **FilenameValidator**: Research format validation dengan auto-rename
- **YOLONormalizer**: Multi-preset normalization engine
- **FileProcessor**: I/O operations dengan threading support
- **StatsCollector**: Comprehensive statistics collection
- **SampleGenerator**: Preview generation dengan naming patterns

## 📝 Complete Workflow Example

```python
from smartcash.dataset.preprocessor import *

# 1. Check status dengan file scanning
status = get_preprocessing_status({'data': {'dir': 'data'}})
print(f"Ready: {status['service_ready']}")

# 2. Preview processing tanpa execute
preview = get_preprocessing_preview({
    'data': {'dir': 'data'},
    'preprocessing': {'target_splits': ['train']}
})
print(f"Will process {preview['input_summary']['total_files']} files")

# 3. Process dengan progress tracking
result = preprocess_dataset(
    config, 
    progress_callback=progress_handler,
    ui_components=ui_components
)

# 4. Get enhanced statistics
stats = get_dataset_stats('data/preprocessed', include_details=True)
print(f"Main banknotes: {stats['main_banknotes']['total_objects']}")

# 5. Get samples untuk verification
samples = get_samples('data/preprocessed', 'train', max_samples=5)
for sample in samples['samples']:
    print(f"{sample['primary_class']['display']}: {sample['filename']}")

# 6. Generate previews
previews = generate_sample_previews('data/preprocessed', 'data/samples')

# 7. Cleanup jika diperlukan
cleanup_result = cleanup_preprocessing_files(
    'data', 'preprocessed', 
    confirm=True, 
    progress_callback=progress_handler
)
```

## 📜 Changelog

### 🆕 Fitur Baru

#### 1. Enhanced Progress Tracking
- Integrasi ProgressBridge untuk tracking progress yang lebih detail
- Multi-level progress reporting (overall, phase, step)
- Callback support untuk UI integration

#### 2. Advanced Normalization
- Dukungan YOLOv5-compatible normalization
- Multiple interpolation methods (linear, cubic, area)
- Aspect ratio preservation dengan padding opsional
- Denormalization untuk visualisasi

#### 3. Comprehensive Validation
- Validasi struktur direktori dengan auto-fix
- Validasi nama file dengan research format
- Support untuk multiple dataset splits

#### 4. Enhanced Statistics
- Layer analysis untuk preprocessing
- File statistics dengan metadata
- Sample generation untuk preview

#### 5. Cleanup Utilities
- Targeted cleanup (samples, normalized, all)
- Progress tracking untuk operasi cleanup
- Safe removal dengan konfirmasi

### 🔄 Perubahan Signifikan

#### 1. Arsitektur Modular
- Pemisahan komponen utama (normalizer, processor, validator)
- Factory pattern untuk komponen yang dapat dikonfigurasi
- Dependency injection untuk komponen kustom

#### 2. Performance Improvements
- Batch processing support
- Optimasi memori untuk dataset besar
- Paralelisasi untuk operasi I/O intensif

#### 3. API Enhancement
- Simplified interface untuk common tasks
- Detailed error messages dengan kode error
- Type hints untuk semua fungsi publik

### 🐛 Perbaikan
- Perbaikan handling path lintas platform
- Peningkatan error handling untuk kasus edge
- Perbaikan validasi konfigurasi

### 🔧 Breaking Changes
1. **Perubahan Struktur Config**
   ```python
   # Old
   {
       'target_size': 640,
       'normalize': True
   }
   
   # New
   {
       'preprocessing': {
           'target_splits': ['train', 'valid'],
           'normalization': {
               'target_size': [640, 640],
               'preserve_aspect_ratio': True
           }
       }
   }
   ```

2. **Perubahan Return Value**
   - Format return value yang lebih terstruktur
   - Informasi error yang lebih detail
   - Metadata tambahan untuk debugging

### 📝 Catatan Migrasi
1. Update konfigurasi sesuai format baru
2. Perbarui kode yang mengandalkan struktur return value lama
3. Manfaatkan progress callback untuk UI integration
4. Gunakan factory functions untuk komponen kustom

### 🚀 Fitur Eksperimental
1. Batch processing untuk dataset besar
2. Custom preprocessing pipeline
3. Advanced visualization tools

## 🚫 Breaking Changes dari Previous Version

1. **FileNamingManager Integration**: Semua file operations menggunakan research naming patterns
2. **Enhanced Progress Tracking**: Level-based progress dengan split tracking
3. **Preset-based Normalization**: Multiple YOLOv5 presets dengan auto-selection
4. **Layer-aware Statistics**: Class distribution berdasarkan layer analysis
5. **Variance Support**: Augmented files dengan variance numbering
6. **UI Component Integration**: Direct integration dengan progress tracker dan UI components