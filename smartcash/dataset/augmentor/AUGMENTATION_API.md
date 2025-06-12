# 📋 SmartCash Augmentation Service API (Updated)

## 🎯 **Overview**
Service augmentasi untuk training YOLOv5 dengan arsitektur EfficientNet-B4 backbone. **Updated** dengan integrasi FileNamingManager dan preprocessor API untuk consistency.

## 📁 **Output Structure**
```
data/
├── raw/{split}/
│   ├── images/          # Raw rp_*.jpg files
│   └── labels/          # Raw *.txt files
├── augmented/{split}/
│   ├── images/          # aug_{nominal}_{uuid}_{variance:03d}.jpg files
│   └── labels/          # Transformed .txt files (bbox adjusted)
└── preprocessed/{split}/
    ├── images/          # .npy files (normalized float32) + sample_aug_*.jpg 
    └── labels/          # .txt files (copy dari augmented)
```

## 🔄 **Pipeline Flow (Updated)**
1. **Raw** → **Augmented**: Visual transformation + bbox adjustment dengan **FileNamingManager variance support**
2. **Augmented** → **Preprocessed**: Normalisasi menggunakan **preprocessor API** untuk consistency
3. **Sample Generation**: Copy augmented files ke preprocessed sebagai `sample_aug_*` untuk preview

## 🚀 **Main API Functions**

### `augment_and_normalize(config, target_split, progress_tracker, progress_callback)`
**Deskripsi**: Pipeline lengkap augmentasi + normalisasi dengan FileNamingManager
```python
result = augment_and_normalize(
    config=config,
    target_split='train',
    progress_tracker=tracker,
    progress_callback=callback
)
```

**Returns (Updated)**:
```python
{
    'status': 'success',
    'total_generated': 150,      # Files augmented
    'total_normalized': 150,     # Files normalized to .npy
    'processing_time': 45.2,
    'phases': {
        'augmentation': {...},
        'normalization': {...}
    },
    'pipeline_summary': {
        'overall': {
            'file_naming': 'FileNamingManager with variance support',
            'files_flow': '150 augmented → 150 normalized'
        }
    }
}
```

### `get_sampling_data(config, target_split, max_samples, progress_tracker)` (Updated)
**Deskripsi**: Generate preview samples dengan pattern `sample_aug_*`
```python
samples = get_sampling_data(
    config=config,
    target_split='train',
    max_samples=5
)
```

**Returns (Updated)**:
```python
{
    'status': 'success',
    'samples': [
        {
            'filename': 'sample_aug_001000_uuid_001',  # NEW: Sample pattern
            'nominal': '001000',
            'variance': 1,
            'uuid': 'a1b2c3d4-e5f6-...',
            'sample_path': 'data/preprocessed/train/images/sample_aug_001000_uuid_001.jpg',
            'label_path': 'data/preprocessed/train/labels/sample_aug_001000_uuid_001.txt',
            'source_aug_path': 'data/augmented/train/images/aug_001000_uuid_001.jpg',
            'class_display': 'Rp1K',
            'pattern': 'sample_aug_{nominal}_{uuid}_{variance:03d}.jpg'
        }
    ],
    'total_samples': 5,
    'target_split': 'train'
}
```

### `cleanup_data(config, target_split, target)` (NEW: Configurable)
**Deskripsi**: Configurable cleanup dengan pilihan target
```python
# Cleanup augmented files + labels saja
result = cleanup_data(config, target_split='train', target='augmented')

# Cleanup sample files saja  
result = cleanup_data(config, target_split='train', target='samples')

# Cleanup semua: augmented + samples
result = cleanup_data(config, target_split='train', target='both')
```

**Target Options**:
- `'augmented'`: Hapus aug_*.jpg + aug_*.txt + preprocessed .npy files
- `'samples'`: Hapus sample_aug_*.jpg files dari preprocessed
- `'both'`: Hapus semua (default)

**Returns (Updated)**:
```python
{
    'status': 'success',
    'target': 'both',
    'target_split': 'train',
    'total_removed': 300,
    'augmented_removed': 250,    # aug files + preprocessed files
    'samples_removed': 50,       # sample_aug files
    'cleanup_summary': {
        'target_type': 'both',
        'files_removed': {
            'train': {
                'augmented': 150,
                'preprocessed': 100,
                'samples': 50,
                'total': 300
            }
        }
    }
}
```

### Helper Functions (NEW)
```python
# Cleanup shortcuts
cleanup_augmented_data(config, target_split)  # target='augmented'
cleanup_samples(config, target_split)         # target='samples'  
cleanup_all(config, target_split)             # target='both'
```

### `get_augmentation_status(config, progress_tracker)` (Updated)
**Deskripsi**: Status dengan FileNamingManager pattern detection
```python
status = get_augmentation_status(config)
```

**Returns (Updated)**:
```python
{
    'service_ready': True,
    'config': {
        'file_naming': 'FileNamingManager integrated'
    },
    'train_augmented': 150,      # aug_*.jpg files
    'train_preprocessed': 150,   # .npy files
    'train_sample_aug': 25,      # sample_aug_*.jpg files (NEW)
    'train_variance_count': 3,   # Unique variance count (NEW)
    'valid_augmented': 50,
    'valid_preprocessed': 50,
    'valid_sample_aug': 10
}
```

## ⚙️ **Configuration (Updated)**

### File Naming Configuration (NEW)
```yaml
# FileNamingManager patterns automatically applied
file_naming:
  raw_pattern: 'rp_{nominal}_{uuid}.jpg'
  augmented_pattern: 'aug_{nominal}_{uuid}_{variance:03d}.jpg'
  sample_pattern: 'sample_aug_{nominal}_{uuid}_{variance:03d}.jpg'
  variance_support: true
```

### Cleanup Configuration (NEW)
```yaml
cleanup:
  default_target: 'both'        # 'augmented', 'samples', 'both'
  confirm_before_cleanup: true
  backup_before_cleanup: false
  cleanup_empty_dirs: true
  
  # Target-specific settings
  targets:
    augmented:
      include_preprocessed: true  # Also remove .npy files
      patterns: ['aug_*']
    samples:
      patterns: ['sample_aug_*']
      preserve_originals: true
    both:
      sequential: true           # Cleanup augmented first, then samples
```

### Normalization Configuration (Updated)
```yaml
preprocessing:
  normalization:
    method: 'minmax'          # Uses preprocessor API
    api_source: 'preprocessor' # 'preprocessor' or 'fallback'
    target_size: [640, 640]   # Fixed size untuk YOLO
    preset_mapping:           # NEW: Method to preset mapping
      minmax: 'yolov5s'
      standard: 'yolov5m'
      imagenet: 'yolov5l'
      none: 'default'
```

## 🔧 **Advanced Usage (Updated)**

### Sample Generation dengan FileNamingManager
```python
# Generate preview samples
samples = get_sampling_data(config, 'train', max_samples=10)

for sample in samples['samples']:
    # FileNamingManager parsed info
    print(f"Nominal: {sample['nominal']}")      # '001000' 
    print(f"Variance: {sample['variance']}")    # 1, 2, 3...
    print(f"Display: {sample['class_display']}")# 'Rp1K'
    print(f"Pattern: {sample['pattern']}")      # Format info
    
    # File paths
    sample_path = sample['sample_path']         # Preview file
    source_path = sample['source_aug_path']     # Original augmented
```

### Configurable Cleanup Workflows
```python
# Development workflow: cleanup samples only
cleanup_samples(config, 'train')

# Production workflow: cleanup all
cleanup_all(config)

# Selective cleanup: augmented files only
cleanup_augmented_data(config, 'train')

# Check status after cleanup
status = get_augmentation_status(config)
print(f"Remaining samples: {status['train_sample_aug']}")
```

### Variance Analysis
```python
status = get_augmentation_status(config)

for split in ['train', 'valid', 'test']:
    aug_count = status.get(f'{split}_augmented', 0)
    variance_count = status.get(f'{split}_variance_count', 0)
    
    if aug_count > 0:
        avg_variance = aug_count / variance_count if variance_count > 0 else 0
        print(f"{split}: {aug_count} files, {variance_count} variances, {avg_variance:.1f} avg")
```

## 📊 **File Patterns (Updated)**

### FileNamingManager Integration
```python
# Automatic pattern generation
raw_file = 'rp_001000_a1b2c3d4.jpg'           # Input
aug_files = [
    'aug_001000_a1b2c3d4_001.jpg',            # Variance 1
    'aug_001000_a1b2c3d4_002.jpg',            # Variance 2  
    'aug_001000_a1b2c3d4_003.jpg'             # Variance 3
]
sample_files = [
    'sample_aug_001000_a1b2c3d4_001.jpg',     # Preview variance 1
    'sample_aug_001000_a1b2c3d4_002.jpg'      # Preview variance 2
]
```

### .npy Files (Preprocessed) - Unchanged
- **Format**: NumPy float32 array via preprocessor API
- **Shape**: (640, 640, 3) - BGR channels  
- **Range**: [0.0, 1.0] untuk minmax normalization
- **Preset**: Automatic mapping (minmax → yolov5s)

## 🎯 **Best Practices (Updated)**

### Recommended Configuration
```python
config = {
    'augmentation': {
        'num_variations': 3,      # More variance patterns
        'target_count': 500,      
        'intensity': 0.7,         
        'types': ['combined'],    
        'target_split': 'train',  
        'balance_classes': True   
    },
    'preprocessing': {
        'normalization': {
            'method': 'minmax',     # Auto-mapped to yolov5s preset
            'api_source': 'preprocessor'  # Use preprocessor API
        }
    },
    'cleanup': {
        'default_target': 'samples',  # Conservative: samples only
        'confirm_before_cleanup': True
    }
}
```

### Development Workflow
```python
# 1. Generate augmentation
result = augment_and_normalize(config, 'train')

# 2. Create preview samples  
samples = get_sampling_data(config, 'train', max_samples=10)

# 3. Verify samples
for sample in samples['samples']:
    visualize_sample(sample['sample_path'])

# 4. Cleanup samples when done
cleanup_samples(config, 'train')

# 5. Keep augmented files for training
# Files ready in data/preprocessed/train/images/*.npy
```

## 🚨 **Important Notes (Updated)**

1. **FileNamingManager**: Semua filename patterns managed centrally
2. **Variance Support**: Multiple variations per file dengan consistent naming
3. **Preprocessor API**: Normalization menggunakan same engine sebagai preprocessing
4. **Sample Generation**: Copy-based preview generation dengan proper naming
5. **Configurable Cleanup**: Granular control target cleanup
6. **UUID Consistency**: Tracking consistency across pipeline stages

## 🔍 **Troubleshooting (Updated)**

### Common Issues
```python
# Issue: Inconsistent file patterns
# Solution: FileNamingManager handles all patterns automatically

# Issue: Sample files not found
samples = get_sampling_data(config)
if len(samples['samples']) == 0:
    # Check: Augmented files exist, run sampling to generate previews

# Issue: Mixed normalization results  
# Solution: Preprocessor API ensures consistency with main preprocessing

# Issue: Cleanup too aggressive
# Solution: Use target='samples' first, then target='augmented' if needed
```

### Debug Configuration Structure
```python
# Check current config structure
from smartcash.dataset.augmentor.utils.config_validator import get_default_augmentation_config

config = get_default_augmentation_config()
print("Combined params:", config['augmentation']['combined'])

# Validate parameter extraction
preview = create_live_preview(config)
transforms = preview['augmentation_applied']
print("Applied transforms:", transforms)
```