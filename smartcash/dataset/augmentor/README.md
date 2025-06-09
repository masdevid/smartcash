# 📚 SmartCash Augmentation API Documentation

## 🎯 Overview

SmartCash Augmentation API menyediakan pipeline lengkap untuk augmentasi dataset dengan:
- **Dual/Triple Progress Tracker** compatibility
- **Auto-resize** ke 640x640 sebelum normalisasi
- **Multiple normalization methods** (minmax, standard, imagenet)
- **Preserved business logic** untuk class balancing
- **UUID filename format** support
- **Symlink management** ke preprocessed folder

---

## 🚀 Quick Start

### Basic Usage
```python
from smartcash.dataset.augmentor import augment_and_normalize

# Minimal configuration
config = {
    'data': {'dir': 'data'},
    'augmentation': {'types': ['combined']},
    'preprocessing': {'normalization': {'method': 'minmax'}}
}

# Run complete pipeline
result = augment_and_normalize(config, target_split='train')
print(f"✅ Generated: {result['total_generated']}, Normalized: {result['total_normalized']}")
```

### Dengan Progress Tracker
```python
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker

# Create progress tracker
progress_tracker = create_dual_progress_tracker("Augmentation")

# Run dengan progress tracking
result = augment_and_normalize(
    config=config,
    target_split='train', 
    progress_tracker=progress_tracker
)
```

---

## 📋 Main API Functions

### `augment_and_normalize(config, target_split='train', progress_tracker=None, progress_callback=None)`

**🎯 Complete pipeline**: Augmentation → Normalization → Symlink creation

**Parameters:**
- `config`: Configuration dictionary
- `target_split`: Target split ('train', 'valid', 'test')
- `progress_tracker`: Progress tracker instance (dual/triple compatible)
- `progress_callback`: Custom callback function `(level, current, total, message)`

**Returns:**
```python
{
    'status': 'success',
    'total_generated': 150,      # Augmented files
    'total_normalized': 150,     # Normalized files  
    'symlinks_created': 300,     # Symlinks created
    'processing_time': 45.2,     # Total time in seconds
    'phases': {
        'augmentation': {...},   # Augmentation results
        'normalization': {...}, # Normalization results
        'symlinks': {...}       # Symlink results
    }
}
```

### `create_augmentor(config, progress_tracker=None)`

**🏭 Factory function** untuk membuat augmentation service

**Returns:** `AugmentationService` instance

### `cleanup_augmented_data(config, target_split=None, progress_tracker=None)`

**🧹 Cleanup function** untuk menghapus augmented files dan symlinks

**Parameters:**
- `target_split`: Specific split atau `None` untuk semua splits

---

## 🏗️ Core Classes

### `AugmentationService`

Main service class untuk orchestrate augmentation pipeline.

```python
service = AugmentationService(config, progress_tracker)

# Run pipeline
result = service.run_augmentation_pipeline('train', progress_callback)

# Get status
status = service.get_augmentation_status()

# Cleanup
cleanup_result = service.cleanup_augmented_data('train')
```

### `AugmentationEngine`

Core engine untuk augmentation dengan threading support.

```python
engine = AugmentationEngine(config, progress_bridge)

# Augment specific split
result = engine.augment_split('train', progress_callback)
```

### `NormalizationEngine`

Engine untuk normalization dengan resize otomatis ke 640x640.

```python
normalizer = NormalizationEngine(config, progress_bridge)

# Normalize augmented files
result = normalizer.normalize_augmented_files(aug_path, output_path, progress_callback)
```

---

## ⚙️ Configuration

### Default Config (dari `augmentation_config.yaml`)

```yaml
# Basic augmentation
augmentation:
  types: ['combined']                 # Default research type
  num_variations: 2                   # Variations per file
  target_count: 500                   # Target samples per class
  intensity: 0.7                      # [0.0-1.0]
  balance_classes: true               # Enable balancing

# Normalization (auto-resize 640x640)
preprocessing:
  normalization:
    method: 'minmax'                  # minmax|standard|imagenet|none
    denormalize: false                # Save as normalized
    target_size: [640, 640]           # Fixed for YOLO
```

### Custom Configuration

```python
config = {
    'data': {
        'dir': '/content/data'          # Base data directory
    },
    'augmentation': {
        'types': ['lighting', 'position', 'combined'],
        'num_variations': 3,
        'target_count': 1000,
        'intensity': 0.8,
        'balance_classes': True,
        'target_split': 'train'
    },
    'preprocessing': {
        'normalization': {
            'method': 'standard',       # Z-score normalization
            'denormalize': False,       # Save as float32
            'target_size': [640, 640]
        }
    }
}
```

---

## 🎨 Augmentation Types

### Default Research Types

| Type | Description | Use Case |
|------|-------------|----------|
| `lighting` | 🌟 Pencahayaan variations | Kondisi cahaya berbeda |
| `position` | 📍 Posisi variations | Sudut pengambilan berbeda |
| `combined` | 🎯 **Default research** | Gabungan lighting + position |

### Additional Types

| Type | Description | Parameters |
|------|-------------|------------|
| `geometric` | 🔄 Transformasi geometri | Rotation, scale, perspective |
| `color` | 🎨 Variasi warna | HSV, brightness, contrast |
| `noise` | 📡 Noise dan blur | Gaussian noise, motion blur |

---

## 🔧 Normalization Methods

### MinMax (Default)
```python
'method': 'minmax'
# Input: [0, 255] → Output: [0.0, 1.0]
# Best for: YOLO training
```

### Standard (Z-Score)
```python
'method': 'standard'  
# Input: [0, 255] → Output: mean=0, std=1
# Best for: Statistical analysis
```

### ImageNet
```python
'method': 'imagenet'
# Mean: [0.485, 0.456, 0.406]
# Std: [0.229, 0.224, 0.225]
# Best for: Transfer learning
```

### None (No Normalization)
```python
'method': 'none'
# Input: [0, 255] → Output: [0, 255] (float32)
# Best for: Raw processing
```

---

## 📊 Progress Tracking

### Dual Progress Tracker
```python
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker

tracker = create_dual_progress_tracker("Augmentation")

result = augment_and_normalize(
    config=config,
    progress_tracker=tracker  # Auto-detects dual API
)
```

### Custom Progress Callback
```python
def custom_callback(level, current, total, message):
    """
    level: 'overall' | 'step' | 'current'
    current: Current progress value
    total: Total progress value  
    message: Progress description
    """
    progress_pct = int((current / total) * 100)
    print(f"[{level.upper()}] {progress_pct}% - {message}")

result = augment_and_normalize(
    config=config,
    progress_callback=custom_callback
)
```

---

## 📁 File Format Support

### Input Formats
```
Raw files: rp_001000_uuid_216.jpg
           rp_{nominal}_{uuid}_{increment}.ext
```

### Output Formats
```
Augmented: aug_rp_001000_uuid_216_01.jpg
           aug_rp_{nominal}_{uuid}_{increment}_{variance}.ext

Labels:    aug_rp_001000_uuid_216_01.txt
           (corresponding YOLO format)
```

### Directory Structure
```
data/
├── train/images/          # Raw input files
├── train/labels/          # Raw label files
├── augmented/train/       # Augmented output
│   ├── images/           # aug_*.jpg files
│   └── labels/           # aug_*.txt files
└── preprocessed/train/    # Normalized + symlinks
    ├── images/           # Symlinks to augmented
    └── labels/           # Symlinks to augmented
```

---

## 🛠️ Reusable Utilities

### ProgressBridge
```python
from smartcash.dataset.augmentor.utils.progress_bridge import ProgressBridge

# Auto-detects tracker capabilities
bridge = ProgressBridge(your_progress_tracker)
bridge.update('overall', 50, 100, "Processing...")
```

### ConfigValidator
```python
from smartcash.dataset.augmentor.utils.config_validator import (
    validate_augmentation_config,
    get_default_augmentation_config
)

# Get defaults dari augmentation_config.yaml
default_config = get_default_augmentation_config()

# Validate dan merge dengan defaults
validated_config = validate_augmentation_config(user_config)
```

### FileProcessor
```python
from smartcash.dataset.augmentor.utils.file_processor import FileProcessor

processor = FileProcessor(config)
files = processor.get_split_files('train')  # Get train split files
label_path = processor.get_label_path(image_path)  # Get corresponding label
```

### FilenameManager
```python
from smartcash.dataset.augmentor.utils.filename_manager import FilenameManager

manager = FilenameManager()

# Parse existing filename
parsed = manager.parse_filename('rp_001000_uuid_216.jpg')
# Returns: {'type': 'raw', 'nominal': '001000', 'uuid': 'uuid', 'increment': '216'}

# Create augmented filename
aug_name = manager.create_augmented_filename(parsed, variance=1)
# Returns: 'aug_rp_001000_uuid_216_01'
```

### SymlinkManager
```python
from smartcash.dataset.augmentor.utils.symlink_manager import SymlinkManager

manager = SymlinkManager(config)
result = manager.create_augmented_symlinks(aug_path, prep_path)
# Creates symlinks dari augmented ke preprocessed folder
```

---

## ⚖️ Class Balancing (Preserved Logic)

### ClassBalancingStrategy
```python
from smartcash.dataset.augmentor.balancer import ClassBalancingStrategy

balancer = ClassBalancingStrategy(config)

# Calculate needs untuk split
needs = balancer.calculate_balancing_needs_split_aware(data_dir, 'train', target_count=500)
# Returns: {'0': 100, '1': 50, '2': 200, ...}

# Get priority order
priority = balancer.get_balancing_priority_order(needs)
# Returns: ['2', '0', '1', ...] (sorted by priority)
```

### FileSelectionStrategy
```python
from smartcash.dataset.augmentor.balancer import FileSelectionStrategy

selector = FileSelectionStrategy(config)

# Select files untuk augmentation
selected = selector.select_prioritized_files_split_aware(data_dir, 'train', needs)
# Returns: ['/path/to/file1.jpg', '/path/to/file2.jpg', ...]
```

---

## 🚨 Error Handling

### Common Error Patterns
```python
try:
    result = augment_and_normalize(config)
    
    if result['status'] == 'success':
        print(f"✅ Success: {result['total_generated']} files")
    else:
        print(f"❌ Error: {result['message']}")
        
except Exception as e:
    print(f"🚨 Pipeline error: {str(e)}")
```

### Validation Errors
```python
# Missing directories
result = {'status': 'error', 'message': 'No source files found for split train'}

# Invalid configuration  
result = {'status': 'error', 'message': 'Invalid augmentation type: unknown'}

# Processing errors
result = {'status': 'partial', 'message': 'Some files failed processing'}
```

---

## 🎯 Best Practices

### 1. **Configuration Management**
```python
# Always validate config
config = validate_augmentation_config(user_config)

# Use defaults sebagai base
default_config = get_default_augmentation_config()
merged_config = {**default_config, **user_config}
```

### 2. **Progress Tracking**
```python
# Prefer dual/triple tracker untuk UI
progress_tracker = create_dual_progress_tracker("Augmentation")

# Use custom callback untuk granular control
def detailed_callback(level, current, total, message):
    if level == 'overall':
        update_main_progress(current, total)
    elif level == 'step':
        update_step_progress(current, total, message)
```

### 3. **Error Recovery**
```python
# Always check result status
if result['status'] != 'success':
    # Log error dan attempt cleanup
    cleanup_augmented_data(config, target_split)
    
# Partial success handling
if result.get('total_generated', 0) > 0:
    print(f"⚠️ Partial success: {result['total_generated']} files processed")
```

### 4. **Resource Management**
```python
# Configure threading untuk your environment
config['file_processing']['max_workers'] = min(4, os.cpu_count())

# Monitor memory usage
config['performance']['max_memory_usage_gb'] = 4.0
```

---

## 📈 Performance Tips

1. **Threading**: Optimal untuk I/O-bound augmentation operations
2. **Batch Processing**: File processing dalam batches untuk memory efficiency
3. **Symlinks**: Gunakan symlinks instead of copying untuk save disk space
4. **Validation**: Disable validation untuk faster processing jika data sudah clean
5. **Progress Updates**: Reduce update frequency untuk better performance

---

## 🔗 Related Documentation

- [Configuration Guide](./configs/augmentation_config.yaml)
- [Progress Tracker API](../ui/components/progress_tracker/)
- [Class Balancing Strategy](./balancer/)
- [File Naming Convention](./utils/filename_manager.py)

---

*📝 Last updated: Compatible dengan dual progress tracker API dan preserved business logic*