# Ringkasan Dokumentasi Dataset Manager SmartCash

## Deskripsi

`DatasetManager` adalah komponen pusat untuk pengelolaan dataset multilayer mata uang Rupiah di SmartCash. Komponen ini menggunakan pola desain Facade Composite untuk menyediakan antarmuka terpadu bagi berbagai operasi dataset. Implementasi mengadopsi pendekatan yang modular, atomic, dan mudah diuji dengan menerapkan prinsip Single Responsibility.

## Struktur dan Komponen

```
smartcash/handlers/dataset/
├── __init__.py                          # Export komponen utama
├── dataset_manager.py                   # Entry point minimal
├── facades/                             # Facades terpisah untuk fungsi spesifik
│   ├── dataset_base_facade.py           # Kelas dasar untuk semua facade
│   ├── data_loading_facade.py           # Operasi loading dan dataloader
│   ├── data_processing_facade.py        # Validasi, augmentasi, balancing
│   ├── data_operations_facade.py        # Split, merge, cleanup
│   ├── visualization_facade.py          # Visualisasi dataset
│   ├── dataset_explorer_facade.py       # Facade untuk semua explorer
│   └── pipeline_facade.py               # Menggabungkan semua facade
├── multilayer/                          # Komponen dataset multilayer
│   ├── multilayer_dataset_base.py       # Kelas dasar
│   ├── multilayer_dataset.py            # Dataset multilayer
│   └── multilayer_label_handler.py      # Handler label
├── core/                                # Komponen inti
│   ├── dataset_loader.py                # Loader dataset spesifik
│   ├── dataset_downloader.py            # Downloader dataset
│   ├── dataset_transformer.py           # Transformasi data
│   ├── dataset_validator.py             # Validasi dataset
│   ├── dataset_augmentor.py             # Augmentasi dataset
│   ├── dataset_balancer.py              # Balancer dataset
│   └── download_manager.py              # Manager untuk download
├── operations/                          # Operasi pada dataset
│   ├── dataset_split_operation.py       # Pemecahan dataset
│   ├── dataset_merge_operation.py       # Penggabungan dataset
│   └── dataset_reporting_operation.py   # Pelaporan dataset
├── explorers/                           # Eksplorasi dataset
│   ├── base_explorer.py                 # Kelas dasar untuk semua explorer
│   ├── validation_explorer.py           # Validasi integritas
│   ├── class_explorer.py                # Distribusi kelas
│   ├── layer_explorer.py                # Distribusi layer
│   ├── image_size_explorer.py           # Ukuran gambar
│   └── bbox_explorer.py                 # Bounding box
├── integration/                         # Adapter untuk integrasi
│   ├── validator_adapter.py             # Adapter untuk EnhancedDatasetValidator
│   └── colab_drive_adapter.py           # Adapter untuk Google Drive di Colab
└── visualizations/                      # Visualisasi dataset
    ├── visualization_base.py            # Kelas dasar untuk semua visualisasi
    ├── heatmap/                         # Visualizer heatmap
    │   ├── spatial_density_heatmap.py   # Heatmap kepadatan spasial
    │   ├── class_density_heatmap.py     # Heatmap kepadatan kelas
    │   └── size_distribution_heatmap.py # Heatmap distribusi ukuran
    └── sample/                          # Visualizer sampel
        ├── sample_grid_visualizer.py    # Grid sampel gambar
        └── annotation_visualizer.py     # Visualisasi anotasi
```

## Fitur Utama

### 1. Pengelolaan Dataset Multilayer

Mendukung dataset dengan beberapa layer deteksi:
- `banknote`: Deteksi mata uang kertas (nominal penuh)
- `nominal`: Deteksi area nominal tertentu
- `security`: Deteksi fitur keamanan (tanda tangan, text, benang pengaman)

### 2. Loading dan Pemrosesan Data

- Load dataset dari sumber lokal atau Roboflow API
- Transformasi dan augmentasi data dengan berbagai teknik
- Support untuk cache dataset untuk mempercepat loading
- Penggunaan DataLoader PyTorch dengan konfigurasi optimal
- Collate functions khusus untuk dataset multilayer
- Dukungan download dataset dengan retry dan resume

### 3. Validasi dan Analisis Dataset

- Validasi integritas file gambar dan label
- Analisis distribusi kelas dan layer
- Analisis ukuran gambar dan bounding box
- Deteksi ketidakseimbangan kelas dan layer
- Perbaikan otomatis untuk masalah umum dataset
- Integrasi dengan EnhancedDatasetValidator

### 4. Augmentasi dan Penyeimbangan

- Augmentasi dataset dengan berbagai teknik (posisi, pencahayaan, kombinasi)
- Penyeimbangan distribusi kelas dengan undersampling
- Support untuk augmentasi progresif dan evaluasi hasil
- Kombinasi kustom parameter augmentasi
- Paralelisasi proses augmentasi dan balancing

### 5. Visualisasi dan Pelaporan

- Visualisasi distribusi kelas dan layer
- Visualisasi ukuran objek (width, height, area, aspect ratio)
- Heatmap kepadatan spasial, kelas, dan ukuran
- Visualisasi sampel dengan bounding box per kelas/layer
- Visualisasi perbandingan layer anotasi
- Pembuatan laporan komprehensif dataset (markdown, JSON)

### 6. Pipeline Dataset Lengkap

- Setup pipeline lengkap (download, validasi, augmentasi, balancing)
- Dukungan untuk automatic cleanup dan verifikasi hasil
- Task reporting dan progress tracking dengan tqdm
- Integrasi dengan Google Colab
- Factory pattern untuk pembuatan komponen dataset

### 7. Integrasi dengan Environment Lain

- Integrasi dengan Google Drive melalui ColabDriveAdapter
- Dukungan untuk symlink dan path mapping
- Factory pattern untuk pembuatan komponen dengan dependency injection

## Kelas Utama

### DatasetManager

Kelas utama yang menggabungkan semua facade dan menyediakan antarmuka terpadu:

```python
class DatasetManager(PipelineFacade):
    """
    Manager utama untuk dataset SmartCash.
    
    Menggunakan pola composite facade yang menggabungkan:
    - DataLoadingFacade: Operasi loading data dan pembuatan dataloader
    - DataProcessingFacade: Operasi validasi, augmentasi, dan balancing
    - DataOperationsFacade: Operasi manipulasi dataset
    - VisualizationFacade: Operasi visualisasi dataset
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi DatasetManager."""
        super().__init__(config, data_dir, cache_dir, logger)
```

### DatasetComponentFactory

Factory pattern untuk pembuatan komponen dataset:

```python
class DatasetComponentFactory:
    """
    Factory untuk membuat komponen dataset terintegrasi.
    """
    
    @staticmethod
    def create_validator(...): """Membuat validator dataset."""
    
    @staticmethod
    def create_augmentor(...): """Membuat augmentor dataset."""
    
    @staticmethod
    def create_multilayer_dataset(...): """Membuat multilayer dataset."""
    
    # ... metode factory lainnya
```

### DatasetBaseFacade

Kelas dasar untuk semua facade dataset:

```python
class DatasetBaseFacade:
    """
    Kelas dasar untuk facade dataset.
    Menyediakan fitur registrasi dan lazy initialization.
    """
    
    def __init__(self, config, data_dir=None, cache_dir=None, logger=None): """Inisialisasi."""
    
    def _get_component(self, component_id, factory_func): """Lazy initialization komponen."""
```

## Facade yang Tersedia

### 1. DataLoadingFacade

Menyediakan fungsi untuk loading dataset dan pembuatan dataloader:

- `get_dataset(split)`: Dapatkan dataset untuk split tertentu
- `get_dataloader(split)`: Dapatkan dataloader untuk split tertentu
- `get_all_dataloaders()`: Dapatkan semua dataloader (train, val, test)
- `download_dataset()`: Download dataset dari Roboflow
- `export_to_local()`: Export dataset Roboflow ke struktur folder lokal
- `convert_dataset_format()`: Konversi dataset dari satu format ke format lain

### 2. DataProcessingFacade

Menyediakan fungsi untuk pemrosesan dataset:

- `validate_dataset(split)`: Validasi dataset menggunakan validator terintegrasi
- `analyze_dataset(split)`: Analisis mendalam dataset
- `fix_dataset(split)`: Perbaiki masalah dataset
- `augment_dataset()`: Augmentasi dataset
- `balance_by_undersampling()`: Seimbangkan dataset

### 3. DataOperationsFacade

Menyediakan fungsi untuk operasi dataset:

- `get_split_statistics()`: Dapatkan statistik untuk semua split dataset
- `split_dataset()`: Pecah dataset menjadi train/val/test
- `merge_splits()`: Gabungkan semua split menjadi satu direktori flat
- `generate_dataset_report()`: Buat laporan lengkap tentang dataset

### 4. VisualizationFacade

Menyediakan fungsi untuk visualisasi dataset:

- `visualize_class_distribution()`: Visualisasi distribusi kelas
- `visualize_layer_distribution()`: Visualisasi distribusi layer
- `visualize_sample_images()`: Visualisasi sampel gambar dari dataset dengan bounding box
- `visualize_augmentation_comparison()`: Visualisasi perbandingan berbagai jenis augmentasi

### 5. DatasetExplorerFacade

Menyediakan fungsi untuk eksplorasi dataset secara mendalam:

- `analyze_dataset()`: Lakukan analisis komprehensif
- `analyze_class_distribution()`: Analisis distribusi kelas
- `analyze_layer_distribution()`: Analisis distribusi layer
- `analyze_image_sizes()`: Analisis ukuran gambar
- `analyze_bounding_boxes()`: Analisis bounding box

### 6. PipelineFacade (DatasetManager)

Menyediakan fungsi untuk pipeline dataset lengkap:

- `setup_full_pipeline()`: Setup pipeline lengkap (download, validasi, augmentasi, balancing)

## Format dan Struktur Data

### Format Label

Dataset menggunakan format label YOLO:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Struktur File

```
data/
├── train/                # Training data
│   ├── images/           # Gambar training
│   └── labels/           # Label YOLO training
├── valid/                # Validation data
│   ├── images/           # Gambar validasi
│   └── labels/           # Label YOLO validasi
└── test/                 # Test data
    ├── images/           # Gambar test
    └── labels/           # Label YOLO test

.cache/smartcash/         # Cache untuk mempercepat loading
visualizations/           # Output visualisasi
runs/train/experiments/   # Output eksperimen
```

### MultilayerDataset

Inti dari DatasetManager adalah komponen `MultilayerDataset` yang mengelola dataset dengan multiple detection layers:

```python
class MultilayerDataset(MultilayerDatasetBase):
    """Dataset untuk deteksi multilayer."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        layers: Optional[List[str]] = None,
        require_all_layers: bool = False,
        logger: Optional[SmartCashLogger] = None,
        config: Optional[Dict] = None
    ):
        # ...
```

Output item dataset dalam format berikut:

```python
{
    'image': img_tensor,  # Tensor gambar [C, H, W]
    'targets': {          # Dict target per layer
        'banknote': tensor_banknote,  # [n_classes, 5] format [x, y, w, h, conf]
        'nominal': tensor_nominal,    # [n_classes, 5]
        'security': tensor_security   # [n_classes, 5]
    },
    'metadata': {         # Metadata tambahan
        'image_path': str(path),
        'label_path': str(label_path),
        'available_layers': ['banknote', 'nominal']  # Layer yang tersedia
    }
}
```

## Konfigurasi

DatasetManager menggunakan konfigurasi dalam format dictionary yang mencakup parameter seperti direktori data, parameter preprocessing, augmentasi, dan detail layer.

## Integrasi dengan Google Colab

```python
# Deteksi dan setup untuk Google Colab
colab_adapter = ColabDriveAdapter(
    mount_path="/content/drive/MyDrive/SmartCash",
    local_path="/content/SmartCash",
    auto_mount=True
)

# Setup symlink untuk direktori penting
colab_adapter.setup_symlinks(['data', 'models', 'configs'])

# Gunakan DatasetManager dengan path lokal
dataset_manager = DatasetManager(config, data_dir="/content/SmartCash/data")
```

## Optimasi Performa

- **Multiprocessing**: Paralelisasi berbagai operasi dataset
- **Caching**: Mempercepat loading dengan cache
- **Progress Tracking**: Menggunakan tqdm untuk tracking operasi yang memakan waktu
- **Lazy Initialization**: Komponen diinisialisasi hanya saat dibutuhkan

## Panduan Migrasi

Jika menggunakan versi lama dataset_manager.py, berikut panduan migrasi ke versi baru:

### 1. Inisialisasi

```python
# Versi Baru
from smartcash.handlers.dataset import DatasetManager
from smartcash.config import get_config_manager

config_manager = get_config_manager("configs/base_config.yaml")
config = config_manager.get_config()

dataset_manager = DatasetManager(
    config=config,
    data_dir="data",
    cache_dir=".cache/smartcash"
)
```

### 2. Loading Dataset

```python
# Shortcut baru tersedia
train_loader = dataset_manager.get_train_loader(batch_size=16)
val_loader = dataset_manager.get_val_loader(batch_size=16)
```

### 3. Validasi dan Augmentasi

```python
# Validasi dengan opsi baru
dataset_manager.validate_dataset(
    split='train',
    fix_issues=True,
    visualize=True
)

# Augmentasi dengan opsi baru
dataset_manager.augment_dataset(
    split='train',
    augmentation_types=['combined', 'lighting'],
    num_variations=2,
    validate_results=True
)
```

### 4. Pipeline Lengkap

```python
dataset_manager.setup_full_pipeline(
    download_dataset=True,
    validate_dataset=True,
    fix_issues=True,
    augment_data=True,
    balance_classes=False,
    visualize_results=True
)
```

## Kesimpulan

DatasetManager SmartCash yang baru menawarkan:

1. **Struktur Modular**: Pemisahan tugas sesuai prinsip Single Responsibility
2. **Pola Desain Modern**: Facade, Adapter, Factory, Strategy, Composite
3. **Keamanan yang Lebih Baik**: Validasi input dan pengelolaan error yang lebih robust
4. **Pemeliharaan yang Lebih Mudah**: Komponen terisolasi memudahkan update
5. **Performa Lebih Baik**: Caching, parallelism, dan optimasi memory
6. **Integrasi yang Lebih Kuat**: Dengan utils/dataset dan extensions lainnya
7. **Dokumentasi yang Lebih Baik**: Docstring lengkap dan logging yang informatif