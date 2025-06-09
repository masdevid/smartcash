# 📋 SmartCash Augmentation Service API

## 🎯 **Overview**
Service augmentasi untuk training YOLOv5 dengan arsitektur EfficientNet-B4 backbone.

## 📁 **Output Structure**
```
data/
├── raw/{split}/
│   ├── images/          # Raw .jpg files
│   └── labels/          # Raw .txt files
├── augmented/{split}/
│   ├── images/          # Augmented .jpg files (transformed visually)
│   └── labels/          # Transformed .txt files (bbox adjusted)
└── preprocessed/{split}/
    ├── images/          # .npy files (normalized float32 untuk training)
    └── labels/          # .txt files (copy dari augmented)
```

## 🔄 **Pipeline Flow**
1. **Raw** → **Augmented**: Transformasi visual + bbox adjustment
2. **Augmented** → **Preprocessed**: Normalisasi ke .npy untuk training

## 🚀 **Main API Functions**

### `augment_and_normalize(config, target_split, progress_tracker, progress_callback)`
**Deskripsi**: Pipeline lengkap augmentasi + normalisasi
```python
result = augment_and_normalize(
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
    'total_generated': 150,      # Files augmented
    'total_normalized': 150,     # Files normalized to .npy
    'processing_time': 45.2,
    'phases': {
        'augmentation': {...},
        'normalization': {...}
    }
}
```

### `get_sampling_data(config, target_split, max_samples, progress_tracker)`
**Deskripsi**: Ambil random samples untuk evaluasi
```python
samples = get_sampling_data(
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
            'aug_without_norm': [...],    # uint8 array (transformed)
            'aug_norm': [...],           # float32 array (normalized)
            'raw_path': 'data/raw/train/images/...',
            'aug_path': 'data/augmented/train/images/...',
            'norm_path': 'data/preprocessed/train/images/...npy'
        }
    ],
    'total_samples': 5,
    'target_split': 'train'
}
```

### `cleanup_augmented_data(config, target_split, progress_tracker)`
**Deskripsi**: Hapus semua file augmented dan preprocessed
```python
result = cleanup_augmented_data(
    config=config,
    target_split='train'  # None untuk all splits
)
```

### `get_augmentation_status(config, progress_tracker)`
**Deskripsi**: Status file augmented dan preprocessed
```python
status = get_augmentation_status(config)
```

**Returns**:
```python
{
    'service_ready': True,
    'train_augmented': 150,      # .jpg files
    'train_preprocessed': 150,   # .npy files
    'valid_augmented': 50,
    'valid_preprocessed': 50,
    'config': {...}
}
```

## ⚙️ **Configuration**

### Augmentation Parameters
```yaml
augmentation:
  num_variations: 2          # Variations per file
  target_count: 500         # Target samples per class
  intensity: 0.7            # Augmentation intensity [0.1-1.0]
  types: ['combined']       # Transform types
  target_split: 'train'     # Target split
  balance_classes: true     # Enable class balancing
```

### Normalization Parameters
```yaml
preprocessing:
  normalization:
    method: 'minmax'          # minmax|standard|imagenet|none
    target_size: [640, 640]   # Fixed size untuk YOLO
    denormalize: false        # Save as normalized (default)
```

### Supported Transform Types
- **`combined`**: Position + lighting (recommended untuk research)
- **`position`**: Rotasi, flip, translate, scale
- **`lighting`**: Brightness, contrast, HSV, shadows
- **`geometric`**: Advanced geometric transforms
- **`color`**: Color variations
- **`noise`**: Gaussian noise, motion blur

## 🔧 **Advanced Usage**

### Custom Progress Callback
```python
def progress_callback(level, current, total, message):
    if level == "overall":
        print(f"Pipeline: {current}/{total} - {message}")
    elif level == "current":
        print(f"Processing: {current}/{total} files")

result = augment_and_normalize(
    config=config,
    progress_callback=progress_callback
)
```

### Batch Processing Multiple Splits
```python
for split in ['train', 'valid']:
    result = augment_and_normalize(
        config=config,
        target_split=split
    )
    print(f"{split}: {result['total_generated']} files generated")
```

### Sampling for Model Evaluation
```python
# Get samples untuk training evaluation
train_samples = get_sampling_data(config, 'train', max_samples=10)

# Compare raw vs augmented vs normalized
for sample in train_samples['samples']:
    raw = np.array(sample['raw_image'])        # Original image
    aug = np.array(sample['aug_without_norm']) # Transformed image
    norm = np.array(sample['aug_norm'])        # Normalized for training
    
    # Visualize differences
    visualize_sample_comparison(raw, aug, norm)
```

## 📊 **File Formats**

### .npy Files (Preprocessed)
- **Format**: NumPy float32 array
- **Shape**: (640, 640, 3) - BGR channels
- **Range**: [0.0, 1.0] untuk minmax normalization
- **Usage**: Direct loading untuk YOLO training
```python
import numpy as np
normalized_image = np.load('aug_rp_001000_uuid_001_01.npy')
# Shape: (640, 640, 3), dtype: float32, range: [0.0, 1.0]
```

### Label Transformation
- **Augmented labels**: Bbox coordinates adjusted untuk transformasi
- **Format**: YOLO format (class_id x_center y_center width height)
- **Coordinate system**: Normalized [0.0, 1.0]
- **Transformations applied**:
  - Horizontal flip → x_center adjusted
  - Rotation → bbox repositioned
  - Scale/translate → coordinates scaled

## 🎯 **Best Practices**

### Recommended Configuration
```python
config = {
    'augmentation': {
        'num_variations': 3,      # Optimal untuk research
        'target_count': 500,      # Balance dataset
        'intensity': 0.7,         # Moderate augmentation
        'types': ['combined'],    # Research pipeline
        'target_split': 'train',  # Primary untuk augmentasi
        'balance_classes': True   # Layer 1 & 2 optimal
    },
    'preprocessing': {
        'normalization': {
            'method': 'minmax',     # YOLO compatible
            'target_size': [640, 640]
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
- **Validation**: Automatic bbox coordinate validation
- **Error handling**: Skip corrupted files, continue processing
- **Logging**: Detailed progress dan error reporting
- **Cleanup**: Easy removal of generated files

## 🚨 **Important Notes**

1. **No Visual Difference**: Augmented `.jpg` dan preprocessed `.jpg` identik secara visual
2. **Training Files**: Gunakan `.npy` files di preprocessed untuk training
3. **Label Sync**: Labels di preprocessed adalah copy dari augmented (sudah ter-transform)
4. **UUID Consistency**: Sampling menggunakan UUID untuk tracking consistency
5. **Memory Efficient**: .npy format lebih efisien untuk training pipeline

## 🔍 **Troubleshooting**

### Common Issues
```python
# Issue: No files generated
result = augment_and_normalize(config)
if result['total_generated'] == 0:
    # Check: Raw files exist, target_count realistic, class balance

# Issue: Sampling returns empty
samples = get_sampling_data(config)
if len(samples['samples']) == 0:
    # Check: Augmented files exist, UUID format correct

# Issue: Training files not found
# Solution: Use .npy files dari preprocessed/images/, not .jpg
```

### Debug Mode
```python
# Enable detailed logging
config['logging']['level'] = 'DEBUG'
config['logging']['log_timing'] = True

# Verify file structure
status = get_augmentation_status(config)
print(f"Augmented: {status['train_augmented']}")
print(f"Preprocessed: {status['train_preprocessed']}")
```