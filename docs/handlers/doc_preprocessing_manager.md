# Dokumentasi PreprocessingManager SmartCash

## Deskripsi

`PreprocessingManager` adalah komponen facade yang menyediakan antarmuka terpadu untuk semua operasi preprocessing dataset SmartCash. Kelas ini menggabungkan berbagai pipeline dan adapter untuk validasi, augmentasi, dan analisis dataset dengan struktur modular yang mengikuti prinsip Single Responsibility.

## Struktur Folder dan File

```
smartcash/handlers/preprocessing/
├── __init__.py                          # Export komponen utama
├── preprocessing_manager.py             # Entry point minimal (facade)
├── core/                                # Komponen inti preprocessing
│   ├── __init__.py                      # Export komponen core
│   ├── preprocessing_component.py       # Komponen dasar
│   ├── validation_component.py          # Validasi dataset
│   └── augmentation_component.py        # Augmentasi dataset
├── pipeline/                            # Pipeline dan workflow
│   ├── __init__.py                      # Export komponen pipeline
│   ├── preprocessing_pipeline.py        # Pipeline dasar
│   ├── validation_pipeline.py           # Pipeline validasi
│   └── augmentation_pipeline.py         # Pipeline augmentasi
├── integration/                         # Adapter untuk integrasi
│   ├── __init__.py                      # Export komponen integration
│   ├── validator_adapter.py             # Adapter untuk EnhancedDatasetValidator
│   ├── augmentation_adapter.py          # Adapter untuk AugmentationManager
│   ├── cache_adapter.py                 # Adapter untuk CacheManager
│   └── colab_drive_adapter.py           # Adapter untuk Google Drive
└── observers/                           # Observer pattern untuk monitoring
    ├── __init__.py                      # Export komponen observers
    ├── base_observer.py                 # Observer dasar
    └── progress_observer.py             # Monitoring progres
```

## Fitur Utama

### 1. Integrasi dengan Google Colab

- Auto-deteksi lingkungan Google Colab
- Mount/unmount Google Drive secara otomatis
- Setup symlink untuk integrasi dengan drive
- Penanganan permission dan path yang robust

### 2. Validasi Dataset

- Validasi integritas file gambar dan label
- Deteksi dan perbaikan otomatis label yang rusak
- Validasi format label YOLO dan koordinat
- Pemindahan file tidak valid ke direktori terpisah

### 3. Augmentasi Dataset

- Augmentasi dengan berbagai teknik (posisi, pencahayaan, kombinasi)
- Dukungan untuk kombinasi parameter kustom
- Validasi otomatis hasil augmentasi
- Resume proses yang terganggu

### 4. Analisis Dataset

- Analisis distribusi kelas dan layer
- Analisis ukuran gambar dan bounding box
- Deteksi ketidakseimbangan kelas
- Visualisasi hasil analisis

### 5. Pipeline Terintegrasi

- Pipeline full preprocessing (validasi, augmentasi, analisis)
- Monitoring dan pelaporan progress dengan observer pattern
- Penanganan error yang robust
- Generasi laporan hasil preprocessing

## Arsitektur dan Pola Desain

### Pola Desain yang Digunakan

1. **Facade Pattern**: 
   - `preprocessing_manager.py` sebagai facade yang menyembunyikan kompleksitas
   - Memberikan antarmuka sederhana untuk operasi kompleks

2. **Strategy Pattern**: 
   - Implementasi melalui komponen dan pipeline terpisah
   - Memungkinkan perubahan strategi preprocessing tanpa mengubah klien

3. **Adapter Pattern**: 
   - Adapter untuk komponen dari `utils` (ValidatorAdapter, AugmentationAdapter, CacheAdapter)
   - ColabDriveAdapter untuk integrasi dengan Google Drive

4. **Observer Pattern**: 
   - Monitoring progres pipeline tanpa mengubah logika utama
   - ProgressObserver untuk tqdm progress bar

5. **Pipeline Pattern**: 
   - Implementasi dengan `PreprocessingPipeline` sebagai base
   - Pipeline modular dan dapat disusun

6. **Composite Pattern**:
   - Komponen preprocessing dengan antarmuka seragam
   - Struktur hierarkis dengan komponen dasar dan turunan

## Komponen Utama

### PreprocessingManager

```python
def __init__(
    self, 
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[SmartCashLogger] = None,
    colab_mode: Optional[bool] = None,
    drive_adapter: Optional[ColabDriveAdapter] = None
)
```

Kelas utama yang bertindak sebagai facade untuk semua operasi preprocessing.

#### Parameter

- **config**: Dict konfigurasi preprocessing (opsional)
- **logger**: Logger kustom (opsional)
- **colab_mode**: Mode Google Colab (auto-detect jika None)
- **drive_adapter**: ColabDriveAdapter kustom (opsional)

### PreprocessingPipeline

```python
def __init__(
    self, 
    name: str = "PreprocessingPipeline",
    logger: Optional[SmartCashLogger] = None,
    config: Optional[Dict[str, Any]] = None
)
```

Pipeline dasar yang menggabungkan komponen preprocessing dengan observer pattern.

#### Parameter
- **name**: Nama pipeline
- **logger**: Logger kustom (opsional)
- **config**: Konfigurasi pipeline (opsional)

### ValidationPipeline

```python
def __init__(
    self, 
    config: Dict[str, Any],
    logger: Optional[SmartCashLogger] = None,
    validator_adapter: Optional[ValidatorAdapter] = None,
    add_progress_observer: bool = True
)
```

Pipeline khusus untuk validasi dataset menggunakan ValidationComponent.

#### Parameter
- **config**: Konfigurasi pipeline
- **logger**: Logger kustom (opsional)
- **validator_adapter**: Instance ValidatorAdapter (opsional)
- **add_progress_observer**: Tambahkan progress observer otomatis

### AugmentationPipeline

```python
def __init__(
    self, 
    config: Dict[str, Any],
    logger: Optional[SmartCashLogger] = None,
    augmentation_adapter: Optional[AugmentationAdapter] = None,
    add_progress_observer: bool = True
)
```

Pipeline khusus untuk augmentasi dataset menggunakan AugmentationComponent.

#### Parameter
- **config**: Konfigurasi pipeline
- **logger**: Logger kustom (opsional)
- **augmentation_adapter**: Instance AugmentationAdapter (opsional)
- **add_progress_observer**: Tambahkan progress observer otomatis

### PreprocessingComponent

```python
def __init__(
    self, 
    config: Dict[str, Any], 
    logger: Optional[SmartCashLogger] = None,
    **kwargs
)
```

Kelas abstrak dasar untuk semua komponen preprocessing.

#### Parameter
- **config**: Konfigurasi untuk komponen
- **logger**: Logger kustom (opsional)
- **kwargs**: Parameter tambahan

### ValidationComponent

```python
def __init__(
    self, 
    config: Dict[str, Any], 
    validator_adapter: Optional[ValidatorAdapter] = None,
    logger: Optional[SmartCashLogger] = None,
    **kwargs
)
```

Komponen preprocessing untuk validasi dataset.

#### Parameter
- **config**: Konfigurasi untuk komponen
- **validator_adapter**: Instance ValidatorAdapter (opsional)
- **logger**: Logger kustom (opsional)
- **kwargs**: Parameter tambahan

### AugmentationComponent

```python
def __init__(
    self, 
    config: Dict[str, Any], 
    augmentation_adapter: Optional[AugmentationAdapter] = None,
    logger: Optional[SmartCashLogger] = None,
    **kwargs
)
```

Komponen preprocessing untuk augmentasi dataset.

#### Parameter
- **config**: Konfigurasi untuk komponen
- **augmentation_adapter**: Instance AugmentationAdapter (opsional)
- **logger**: Logger kustom (opsional)
- **kwargs**: Parameter tambahan

## Metode Utama di PreprocessingManager

### 1. Pipeline Lengkap

```python
def run_full_pipeline(
    self,
    splits: Optional[List[str]] = None,
    validate_dataset: bool = True,
    fix_issues: bool = False,
    augment_data: bool = False,
    analyze_dataset: bool = True,
    report_format: str = 'json',
    **kwargs
) -> Dict[str, Any]
```

Menjalankan pipeline preprocessing lengkap dengan tahapan:
- Validasi dataset
- Analisis dataset
- Augmentasi dataset (hanya untuk split train)

#### Parameter:
- **splits**: List split yang akan diproses (default: ['train', 'valid', 'test'])
- **validate_dataset**: Lakukan validasi dataset
- **fix_issues**: Perbaiki masalah yang ditemukan
- **augment_data**: Lakukan augmentasi dataset
- **analyze_dataset**: Analisis dataset
- **report_format**: Format laporan ('json' atau 'html')
- **kwargs**: Parameter tambahan untuk pipeline

#### Return:
Dict dengan hasil pipeline dan status setiap tahap.

### 2. Validasi Dataset

```python
def validate_dataset(
    self, 
    split: str = 'train',
    fix_issues: bool = False,
    move_invalid: bool = False,
    visualize: bool = True,
    sample_size: int = 0,
    **kwargs
) -> Dict[str, Any]
```

Memvalidasi dataset untuk memastikan integritas dan kualitas.

#### Parameter:
- **split**: Split dataset ('train'/'valid'/'test')
- **fix_issues**: Perbaiki masalah otomatis
- **move_invalid**: Pindahkan file tidak valid ke direktori terpisah
- **visualize**: Buat visualisasi masalah
- **sample_size**: Jumlah sampel yang divalidasi (0 = semua)
- **kwargs**: Parameter tambahan

#### Return:
Dict dengan hasil validasi dan statistik.

### 3. Augmentasi Dataset

```python
def augment_dataset(
    self, 
    split: str = 'train',
    augmentation_types: Optional[List[str]] = None,
    num_variations: int = 3,
    output_prefix: str = 'aug',
    resume: bool = True,
    validate_results: bool = True,
    **kwargs
) -> Dict[str, Any]
```

Melakukan augmentasi dataset untuk memperkaya data pelatihan.

#### Parameter:
- **split**: Split dataset ('train'/'valid'/'test')
- **augmentation_types**: Jenis augmentasi (['combined', 'lighting', 'position'])
- **num_variations**: Jumlah variasi per gambar
- **output_prefix**: Prefix untuk file hasil
- **resume**: Lanjutkan proses yang terganggu
- **validate_results**: Validasi hasil augmentasi
- **kwargs**: Parameter tambahan

#### Return:
Dict dengan hasil augmentasi dan statistik.

### 4. Analisis Dataset

```python
def analyze_dataset(
    self,
    split: str = 'train',
    sample_size: int = 0,
    **kwargs
) -> Dict[str, Any]
```

Menganalisis karakteristik dataset secara mendalam.

#### Parameter:
- **split**: Split dataset ('train'/'valid'/'test')
- **sample_size**: Jumlah sampel yang dianalisis (0 = semua)
- **kwargs**: Parameter tambahan

#### Return:
Dict dengan hasil analisis berbagai aspek dataset.

### 5. Laporan Preprocessing

```python
def generate_report(
    self,
    results: Dict[str, Any],
    report_format: str = 'json',
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]
```

Menghasilkan laporan hasil preprocessing dalam format tertentu.

#### Parameter:
- **results**: Hasil preprocessing dari pipeline
- **report_format**: Format laporan ('json', 'html', 'md')
- **output_path**: Path output laporan (opsional)
- **kwargs**: Parameter tambahan

#### Return:
Dict dengan status dan path laporan.

### 6. Setup Colab

```python
def setup_colab(
    self,
    project_dir: str = "/content/SmartCash",
    drive_mount_point: str = "/content/drive",
    drive_project_path: str = "MyDrive/SmartCash",
    auto_mount: bool = True,
    setup_symlinks: bool = True,
    symlink_dirs: Optional[List[str]] = None
) -> Dict[str, Any]
```

Setup lingkungan Google Colab dan integrasi dengan Drive.

#### Parameter:
- **project_dir**: Path project di Colab
- **drive_mount_point**: Titik mount Google Drive
- **drive_project_path**: Path project di Drive
- **auto_mount**: Mount Drive otomatis
- **setup_symlinks**: Setup symlink otomatis
- **symlink_dirs**: Direktori yang akan di-symlink

#### Return:
Dict dengan status setup dan informasi tambahan.

## Adapters

### 1. ValidatorAdapter

```python
def __init__(
    self, 
    config: Dict[str, Any], 
    data_dir: Optional[str] = None,
    logger: Optional[SmartCashLogger] = None,
    **kwargs
)
```

Adapter untuk `EnhancedDatasetValidator` dari `utils.dataset`.

#### Parameter:
- **config**: Konfigurasi untuk validator
- **data_dir**: Direktori data (opsional)
- **logger**: Logger kustom (opsional)
- **kwargs**: Parameter tambahan

### 2. AugmentationAdapter

```python
def __init__(
    self, 
    config: Dict[str, Any], 
    output_dir: Optional[str] = None,
    logger: Optional[SmartCashLogger] = None,
    **kwargs
)
```

Adapter untuk `AugmentationManager` dari `utils.augmentation`.

#### Parameter:
- **config**: Konfigurasi untuk augmentation
- **output_dir**: Direktori output (opsional)
- **logger**: Logger kustom (opsional)
- **kwargs**: Parameter tambahan

### 3. CacheAdapter

```python
def __init__(
    self, 
    config: Dict[str, Any], 
    cache_dir: Optional[str] = None,
    logger: Optional[SmartCashLogger] = None,
    **kwargs
)
```

Adapter untuk `CacheManager` dari `utils.cache`.

#### Parameter:
- **config**: Konfigurasi untuk cache
- **cache_dir**: Direktori cache (opsional)
- **logger**: Logger kustom (opsional)
- **kwargs**: Parameter tambahan

### 4. ColabDriveAdapter

```python
def __init__(
    self, 
    project_dir: str = "/content/SmartCash",
    drive_mount_point: str = "/content/drive",
    drive_project_path: str = "MyDrive/SmartCash",
    logger: Optional[SmartCashLogger] = None,
    auto_mount: bool = False
)
```

Adapter untuk integrasi Google Drive di Google Colab.

#### Parameter:
- **project_dir**: Direktori lokal project di Colab
- **drive_mount_point**: Titik mount Google Drive
- **drive_project_path**: Jalur project di Google Drive (relatif terhadap mount_point)
- **logger**: Logger kustom (opsional)
- **auto_mount**: Mount Google Drive secara otomatis saat inisialisasi

## Observer Pattern

### BaseObserver

```python
def __init__(self, name: str = "BaseObserver")
```

Kelas dasar untuk observer pattern dalam pipeline preprocessing.

#### Parameter:
- **name**: Nama observer

### ProgressObserver

```python
def __init__(
    self, 
    name: str = "ProgressObserver",
    logger: Optional[SmartCashLogger] = None
)
```

Observer untuk monitoring progress pipeline preprocessing.

#### Parameter:
- **name**: Nama observer
- **logger**: Logger kustom (opsional)

## Konfigurasi

Konfigurasi melalui dictionary dengan struktur:

```python
config = {
    'data_dir': 'data',
    'data': {
        'preprocessing': {
            'num_workers': 4,
            'cache_dir': '.cache/smartcash',
            'cache': {
                'max_size_gb': 1.0,
                'ttl_hours': 24,
                'auto_cleanup': True,
                'cleanup_interval_mins': 30
            }
        }
    },
    'training': {
        # Parameter augmentasi
        'degrees': 30,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4
    }
}
```

## Penanganan Error

- Log error dengan level yang sesuai
- Progress bar yang konsisten dengan tqdm
- Informasi error yang detail
- Backup otomatis sebelum modifikasi data