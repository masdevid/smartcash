# 📋 SmartCash Preprocessing Service API

## 🎯 **Overview**
Service preprocessing untuk normalisasi dan validasi dataset YOLOv5 dengan arsitektur EfficientNet-B4 backbone.

## 📁 **Output Structure**
```
data/
├── raw/{split}/
│   ├── images/          # File gambar mentah (.jpg, .png)
│   └── labels/          # File label mentah (.txt)
└── preprocessed/{split}/
    ├── images/          # File .npy (normalized float32 untuk training)
    └── labels/          # File label yang sudah divalidasi (.txt)
```

## 🔄 **Pipeline Flow**
1. **Validasi**: Cek integritas file gambar dan label
2. **Preprocessing**: Normalisasi dan resize gambar ke format yang sesuai
3. **Organisasi**: Menyimpan hasil preprocessing ke direktori terstruktur

## 🚀 **Main API Functions**

### `preprocess_dataset(config, target_split, progress_tracker, progress_callback)`
**Deskripsi**: Pipeline lengkap preprocessing dengan validasi dan normalisasi
```python
from smartcash.dataset.preprocessor import preprocess_dataset

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
    'total_processed': 150,      # Total file yang diproses
    'valid_files': 145,          # File yang lolos validasi
    'invalid_files': 5,          # File yang tidak lolos validasi
    'processing_time': 32.5,     # Waktu pemrosesan (detik)
    'details': {
        'validation': {...},
        'preprocessing': {...}
    }
}
```

### `get_preprocessing_samples(config, target_split, max_samples, progress_tracker)`
**Deskripsi**: Ambil sampel acak untuk evaluasi
```python
from smartcash.dataset.preprocessor import get_preprocessing_samples

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
            'filename': 'pre_10000_uuid_001',
            'original_image': [...],     # Array uint8 (gambar asli)
            'preprocessed_image': [...], # Array float32 (setelah normalisasi)
            'original_path': 'data/raw/train/images/...',
            'preprocessed_path': 'data/preprocessed/train/images/...npy',
            'label_path': 'data/raw/train/labels/...',
            'is_valid': True,
            'validation_errors': []
        }
    ],
    'total_samples': 5,
    'target_split': 'train'
}
```

### `validate_dataset(config, target_split, progress_tracker)`
**Deskripsi**: Validasi dataset tanpa melakukan preprocessing
```python
from smartcash.dataset.preprocessor import validate_dataset

result = validate_dataset(
    config=config,
    target_split='train',
    progress_tracker=tracker
)
```

### `cleanup_preprocessed_data(config, target_split, progress_tracker)`
**Deskripsi**: Hapus semua file hasil preprocessing
```python
from smartcash.dataset.preprocessor import cleanup_preprocessed_data

result = cleanup_preprocessed_data(
    config=config,
    target_split='train'  # None untuk semua split
)
```

### `get_preprocessing_status(config, progress_tracker)`
**Deskripsi**: Dapatkan status preprocessing saat ini
```python
from smartcash.dataset.preprocessor import get_preprocessing_status

status = get_preprocessing_status(config)
```

**Returns**:
```python
{
    'service_ready': True,
    'preprocessing': {
        'enabled': True,
        'method': 'minmax',
        'target_size': [640, 640]
    },
    'validation': {
        'enabled': True,
        'move_invalid': True
    },
    'file_counts': {
        'train': {'raw': 150, 'preprocessed': 145},
        'valid': {'raw': 50, 'preprocessed': 48},
        'test': {'raw': 30, 'preprocessed': 30}
    }
}
```

## ⚙️ **Configuration**

### Preprocessing Configuration
```yaml
preprocessing:
  enabled: true
  validation:
    enabled: true             # Aktifkan validasi
    move_invalid: true        # Pindahkan file yang tidak valid
    fix_issues: false         # Coba perbaiki masalah otomatis
    
  normalization:
    method: 'minmax'          # minmax|standard|imagenet|none
    target_size: [640, 640]   # Ukuran target untuk YOLO
    preserve_aspect_ratio: false
    denormalize: false        # Simpan dalam format ternormalisasi (default)
    
  output:
    create_npy: true          # Buat file .npy
    organize_by_split: true   # Kelompokkan berdasarkan train/valid/test

# Pola penamaan file
file_naming:
  preprocessed_pattern: 'pre_{nominal}_{uuid}_{increment}'
  preserve_uuid: true        # Pertahankan UUID konsisten
```

## 📊 **Visualization Example**

```python
# Contoh visualisasi perbandingan sebelum dan sesudah preprocessing
samples = get_preprocessing_samples(config, 'train', max_samples=3)

for sample in samples['samples']:
    original = np.array(sample['original_image'])     # Gambar asli
    preprocessed = np.array(sample['preprocessed_image'])  # Setelah normalisasi
    
    # Tampilkan gambar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Denormalisasi untuk visualisasi
    if preprocessed.max() <= 1.0:  # Jika dinormalisasi
        preprocessed_vis = (preprocessed * 255).astype('uint8')
    else:
        preprocessed_vis = preprocessed.astype('uint8')
        
    ax2.imshow(preprocessed_vis)
    ax2.set_title('Preprocessed')
    ax2.axis('off')
    plt.show()
```

## 🛠️ **Troubleshooting**

### Masalah Umum
1. **File tidak terdeteksi**
   - Pastikan struktur direktori sesuai dengan yang diharapkan
   - Periksa ekstensi file yang didukung (.jpg, .png, .jpeg)

2. **Error validasi**
   - Periksa format file label YOLO
   - Pastikan koordinat bbox dalam range [0,1]
   - Verifikasi class ID sesuai dengan konfigurasi

3. **Masalah performa**
   - Kurangi jumlah worker jika terjadi memory error
   - Nonaktifkan fitur yang tidak diperlukan dalam konfigurasi

## 📝 **Catatan**
- Semua path relatif terhadap direktori `data/`
- Format label mengikuti standar YOLO (class_id x_center y_center width height)
- File yang tidak valid akan dipindahkan ke direktori `invalid/` jika `move_invalid` diaktifkan
