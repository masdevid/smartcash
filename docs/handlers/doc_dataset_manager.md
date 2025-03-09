# Dokumentasi Dataset Manager SmartCash (Revisi)

## Deskripsi

`DatasetManager` adalah komponen pusat untuk pengelolaan dataset multilayer mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade Composite untuk menyediakan antarmuka terpadu bagi berbagai operasi dataset. Implementasi terbaru dilakukan berdasarkan rencana restrukturisasi yang lebih modular, atomic, dan testable dengan menerapkan prinsip Single Responsibility.

## Struktur dan Komponen

Berdasarkan restrukturisasi yang dilakukan, `DatasetManager` mengadopsi struktur modular berikut:

```
smartcash/handlers/dataset/
├── __init__.py                          # Export komponen utama
├── dataset_manager.py                   # Entry point minimal
├── facades/                             # Facades terpisah untuk fungsi-fungsi spesifik
│   ├── dataset_base_facade.py           # Kelas dasar untuk semua facade
│   ├── data_loading_facade.py           # Operasi loading dan dataloader
│   ├── data_processing_facade.py        # Validasi, augmentasi, balancing
│   ├── data_operations_facade.py        # Split, merge, cleanup
│   ├── visualization_facade.py          # Visualisasi dataset
│   ├── dataset_explorer_facade.py       # Facade untuk semua explorer
│   └── pipeline_facade.py               # Menggabungkan semua facade (untuk dataset_manager)
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

`DatasetManager` menggabungkan beberapa facade terspesialisasi menjadi satu antarmuka terpadu:

- **DataLoadingFacade**: Operasi loading data dan pembuatan dataloader
- **DataProcessingFacade**: Operasi validasi, augmentasi, dan balancing dataset
- **DataOperationsFacade**: Operasi manipulasi dataset (split, merge)
- **VisualizationFacade**: Operasi visualisasi dataset
- **DatasetExplorerFacade**: Eksplorasi dataset untuk analisis mendalam
- **PipelineFacade**: Menggabungkan semua facade di atas

## Fitur Utama

### 1. Pengelolaan Dataset Multilayer

Mendukung dataset dengan beberapa layer deteksi:
- `banknote`: Deteksi mata uang kertas (nominal penuh)
- `nominal`: Deteksi area nominal tertentu
- `security`: Deteksi fitur keamanan (tanda tangan, text, benang pengaman)

### 2. Loading dan Pemrosesan Data

- Load dataset dari sumber lokal atau Roboflow API
- Transformasi dan augmentasi data dengan berbagai teknik
- Support untuk cache dataset untuk mempercepat loading (CacheManager)
- Penggunaan DataLoader PyTorch dengan konfigurasi optimal
- Collate functions khusus untuk dataset multilayer
- Dukungan download dataset dengan retry dan resume

### 3. Validasi dan Analisis Dataset

- Validasi integritas file gambar dan label
- Analisis distribusi kelas dan layer (ClassExplorer, LayerExplorer)
- Analisis ukuran gambar dan bounding box (ImageSizeExplorer, BBoxExplorer)
- Deteksi ketidakseimbangan kelas dan layer
- Perbaikan otomatis untuk masalah umum dataset
- Integrasi dengan EnhancedDatasetValidator melalui adapter pattern

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
- Visualisasi annotation density
- Pembuatan laporan komprehensif dataset (markdown, JSON)

### 6. Pipeline Dataset Lengkap

- Setup pipeline lengkap (download, validasi, augmentasi, balancing)
- Dukungan untuk automatic cleanup dan verifikasi hasil
- Task reporting dan progress tracking dengan tqdm
- Integrasi dengan Google Colab (ColabDriveAdapter)
- Factory pattern untuk pembuatan komponen dataset

### 7. Integrasi dengan Environment Lain

- Integrasi dengan Google Drive melalui ColabDriveAdapter
- Dukungan untuk symlink dan path mapping
- Factory pattern untuk pembuatan komponen dengan dependency injection

## Kelas Utama

### DatasetManager

```python
class DatasetManager(PipelineFacade):
    """
    Manager utama untuk dataset SmartCash yang menyediakan antarmuka terpadu
    untuk semua operasi dan pipeline terkait dataset.
    
    Menggunakan pola composite facade yang menggabungkan:
    - DataLoadingFacade: Operasi loading data dan pembuatan dataloader
    - DataProcessingFacade: Operasi validasi, augmentasi, dan balancing
    - DataOperationsFacade: Operasi manipulasi dataset seperti split dan merge
    - VisualizationFacade: Operasi visualisasi dataset
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetManager.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset (opsional)
            cache_dir: Direktori cache (opsional)
            logger: Logger kustom (opsional)
        """
        super().__init__(config, data_dir, cache_dir, logger)
```

### DatasetComponentFactory

Factory pattern yang digunakan untuk membuat komponen dataset dengan integrasi utils dan handlers:

```python
class DatasetComponentFactory:
    """
    Factory untuk membuat komponen dataset yang terintegrasi antara
    utils/dataset dan handlers/dataset.
    
    Implementasi Factory Pattern untuk memastikan tidak ada duplikasi
    saat membuat objek dan semua dependency dikelola dengan tepat.
    """
    
    @staticmethod
    def create_validator(
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ) -> EnhancedDatasetValidator:
        """Membuat validator dataset."""
        # ...

    @staticmethod
    def create_augmentor(
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ) -> AugmentationManager:
        """Membuat augmentor dataset."""
        # ...

    @staticmethod
    def create_multilayer_dataset(
        config: Dict,
        data_path: str,
        mode: str = 'train',
        transform=None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """Membuat multilayer dataset."""
        # ...

    # ... metode factory lainnya
```

### DatasetBaseFacade

Kelas dasar untuk semua facade dataset:

```python
class DatasetBaseFacade:
    """
    Kelas dasar untuk facade dataset yang mengelola inisialisasi komponen dan konfigurasi.
    Menyediakan fitur registrasi dan akses komponen melalui lazy initialization.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetBaseFacade.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset (opsional)
            cache_dir: Direktori cache (opsional)
            logger: Logger kustom (opsional)
        """
        # ...

    def _get_component(self, component_id: str, factory_func: Callable) -> Any:
        """
        Dapatkan komponen dengan lazy initialization.
        
        Args:
            component_id: ID unik untuk komponen
            factory_func: Fungsi factory untuk membuat komponen jika belum ada
            
        Returns:
            Komponen yang diminta
        """
        # ...
```

### Integrations

`DatasetManager` terintegrasi dengan komponen-komponen lain di SmartCash melalui pattern-pattern berikut:

1. **Adapter Pattern**: 
   - `DatasetValidatorAdapter`: Mengadaptasi EnhancedDatasetValidator dari utils/dataset
   - `ColabDriveAdapter`: Mengadaptasi Google Drive di Colab

2. **Factory Pattern**:
   - `DatasetComponentFactory`: Membuat komponen dengan dependency injection yang tepat

3. **Strategy Pattern**:
   - Diversifikasi strategi augmentasi dan transformasi
   - Collate functions berbeda untuk kebutuhan berbeda

4. **Facade Pattern**:
   - Facade terpisah untuk kategori operasi berbeda
   - Composite facade untuk menggabungkan semua fungsionalitas

5. **Komponen Terintegrasi**:
   - **EnhancedDatasetValidator**: Dari `utils.dataset` untuk validasi dataset
   - **AugmentationManager**: Dari `utils.augmentation` untuk augmentasi dataset
   - **LayerConfigManager**: Dari `utils.layer_config_manager` untuk konfigurasi layer
   - **CoordinateUtils**: Dari `utils.coordinate_utils` untuk manipulasi koordinat
   - **CacheManager**: Dari `utils.cache` untuk pengelolaan cache

## Facade yang Tersedia

### 1. DataLoadingFacade

Menyediakan fungsi untuk loading dataset dan pembuatan dataloader:

- `get_dataset(split)`: Dapatkan dataset untuk split tertentu
- `get_dataloader(split)`: Dapatkan dataloader untuk split tertentu
- `get_all_dataloaders()`: Dapatkan semua dataloader (train, val, test)
- `get_train_loader()`, `get_val_loader()`, `get_test_loader()`: Shortcut untuk dataloader spesifik
- `download_dataset()`: Download dataset dari Roboflow
- `download_dataset_format()`: Download dataset dalam format tertentu
- `download_dataset_from_custom_source()`: Download dari URL kustom
- `export_to_local()`: Export dataset Roboflow ke struktur folder lokal
- `pull_dataset()`: Download dan setup dataset dalam satu langkah
- `convert_dataset_format()`: Konversi dataset dari satu format ke format lain
- `get_dataset_info()`: Dapatkan informasi dataset dari konfigurasi dan pengecekan lokal
- `get_transform()`: Dapatkan transformasi untuk mode tertentu
- `create_custom_transform()`: Buat transformasi kustom

### 2. DataProcessingFacade

Menyediakan fungsi untuk pemrosesan dataset:

- `validate_dataset(split)`: Validasi dataset menggunakan validator terintegrasi
- `analyze_dataset(split)`: Analisis mendalam dataset
- `fix_dataset(split)`: Perbaiki masalah dataset
- `augment_dataset()`: Augmentasi dataset menggunakan AugmentationManager
- `augment_with_combinations()`: Augmentasi dataset dengan kombinasi parameter kustom
- `analyze_class_distribution()`: Analisis distribusi kelas dalam split dataset
- `balance_by_undersampling()`: Seimbangkan dataset dengan mengurangi jumlah sampel kelas dominan

### 3. DataOperationsFacade

Menyediakan fungsi untuk operasi dataset:

- `get_split_statistics()`: Dapatkan statistik untuk semua split dataset
- `get_layer_statistics()`: Dapatkan statistik layer untuk split tertentu
- `get_class_statistics()`: Dapatkan statistik kelas untuk split tertentu
- `split_dataset()`: Pecah dataset menjadi train/val/test
- `merge_splits()`: Gabungkan semua split menjadi satu direktori flat
- `merge_datasets()`: Gabungkan beberapa dataset terpisah
- `generate_dataset_report()`: Buat laporan lengkap tentang dataset

### 4. VisualizationFacade

Menyediakan fungsi untuk visualisasi dataset:

- `visualize_class_distribution()`: Visualisasi distribusi kelas dalam dataset
- `visualize_layer_distribution()`: Visualisasi distribusi layer dalam dataset
- `visualize_sample_images()`: Visualisasi sampel gambar dari dataset dengan bounding box
- `visualize_augmentation_comparison()`: Visualisasi perbandingan berbagai jenis augmentasi pada gambar

### 5. DatasetExplorerFacade

Menyediakan fungsi untuk eksplorasi dataset secara mendalam:

- `analyze_dataset()`: Lakukan analisis komprehensif pada split dataset
- `analyze_class_distribution()`: Analisis khusus untuk distribusi kelas
- `analyze_layer_distribution()`: Analisis khusus untuk distribusi layer
- `analyze_image_sizes()`: Analisis khusus untuk ukuran gambar
- `analyze_bounding_boxes()`: Analisis khusus untuk bounding box
- `get_dataset_sizes()`: Dapatkan ukuran gambar dalam dataset

### 6. PipelineFacade (DatasetManager)

Menyediakan fungsi untuk pipeline dataset lengkap:

- `setup_full_pipeline()`: Setup pipeline lengkap (download, validasi, augmentasi, balancing)

Facade ini menggabungkan semua facade di atas untuk menyediakan workflow dataset lengkap dengan parameter berikut:
- `download_dataset`: Jika True, download dataset dari Roboflow
- `validate_dataset`: Jika True, validasi dataset
- `fix_issues`: Jika True, perbaiki masalah dataset
- `augment_data`: Jika True, augmentasi dataset
- `balance_classes`: Jika True, seimbangkan distribusi kelas
- `visualize_results`: Jika True, buat visualisasi hasil

## Konfigurasi

`DatasetManager` dikonfigurasi melalui dictionary yang menyediakan parameter berikut:

```python
config = {
    # Konfigurasi umum
    'app_name': "SmartCash",
    'version': "1.0.0",
    'description': "Sistem deteksi mata uang Rupiah dengan objek deteksi",
    'author': "Alfrida Sabar",
    'data_dir': "data",
    'output_dir': "runs/train",

    # Konfigurasi data
    'data': {
        # Sumber data ('local' or 'roboflow')
        'source': "roboflow",
        
        # Direktori dataset
        'train_dir': "data/train",
        'valid_dir': "data/valid",
        'test_dir': "data/test",
        
        # Konfigurasi Roboflow
        'roboflow': {
            'api_key': "",  # Wajib diisi jika menggunakan Roboflow
            'workspace': "smartcash-wo2us",
            'project': "rupiah-emisi-2022",
            'version': "3"
        },
        
        # Pengaturan preprocessing
        'preprocessing': {
            'img_size': [640, 640],
            'cache_dir': ".cache/smartcash",
            'num_workers': 4,
            'augmentation_enabled': true,
            'normalize_enabled': true,
            'cache_enabled': true,
            
            # Ukuran cache yang ditingkatkan
            'cache': {
                'max_size_gb': 1.0,
                'ttl_hours': 24,
                'auto_cleanup': true,
                'cleanup_interval_mins': 30
            }
        }
    },

    # Konfigurasi model
    'model': {
        # Arsitektur model
        'backbone': "efficientnet_b4",
        'framework': "YOLOv5",
        
        # Jalur ke file weights
        'weights': "runs/train/weights/best.pt",
        'pretrained': true,
        
        # Parameter inferensi
        'confidence': 0.25,
        'iou_threshold': 0.45,
        'max_det': 1000,
        
        # Parameter sumber daya
        'workers': 4,
        'batch_size': 16,
        'memory_limit': 0.75,  # gunakan 75% CPU saat multiprocessing
        
        # Parameter deployment
        'half_precision': true,
        'optimized': true,
        'export_format': "torchscript"  # 'onnx', 'torchscript'
    },

    # Konfigurasi training
    'training': {
        # Parameter dasar
        'epochs': 50,
        'batch_size': 16,
        'img_size': [640, 640],
        'patience': 5,
        
        # Learning rate
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        
        # Optimizer
        'optimizer': "Adam",  # "Adam", "AdamW", "SGD"
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Scheduler
        'scheduler': "cosine",  # "linear", "cosine", "step"
        
        # Augmentasi
        'fliplr': 0.5,
        'flipud': 0.0,
        'mosaic': 1.0,
        'mixup': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'hsv_h': 0.015,  # HSV Hue
        'hsv_s': 0.7,    # HSV Saturation
        'hsv_v': 0.4,    # HSV Value
        'degrees': 45,
        
        # Early stopping
        'early_stopping_patience': 10,
        
        # Callbacks
        'save_period': 5,  # Simpan checkpoint setiap N epochs
        
        # Validasi
        'val_interval': 1  # Validasi setiap N epochs
    },

    # Konfigurasi layer deteksi
    'layers': {
        'banknote': {
            'name': "banknote",
            'description': "Deteksi uang kertas utuh",
            'classes': ["
```

## File dan Direktori

Struktur file yang diharapkan:

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

## Format Label

Dataset menggunakan format label YOLO:
```
<class_id> <x_center> <y_center> <width> <height>
```

Di mana:
- `class_id`: ID kelas global
- `x_center`, `y_center`: Koordinat pusat bounding box (0-1, dinormalisasi)
- `width`, `height`: Lebar dan tinggi bounding box (0-1, dinormalisasi)

## MultilayerDataset

Inti dari DatasetManager adalah komponen `MultilayerDataset` yang mengelola dataset mata uang dengan beberapa layer deteksi:

```python
class MultilayerDataset(MultilayerDatasetBase):
    """
    Dataset untuk deteksi multilayer dengan integrasi validator dari utils/dataset.
    
    Versi refaktor yang menghindari duplikasi dengan EnhancedDatasetValidator.
    """
    
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

Features:

1. **Multilayer Support**: Mendukung multiple detection layers (banknote, nominal, security)
2. **Validator Integration**: Menggunakan DatasetValidatorAdapter untuk validasi dataset
3. **Lazy Loading**: Loading gambar dan label hanya saat diperlukan
4. **Augmentation**: Dukungan untuk augmentasi on-the-fly
5. **Factory Creation**: Pembuatan melalui factory method `from_config()`

Output item dataset dalam format berikut:

```python
{
    'image': img_tensor,  # Tensor gambar [C, H, W]
    'targets': {          # Dict target per layer
        'banknote': tensor_banknote,  # [n_classes, 5] dengan format [x, y, w, h, conf]
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

## Panduan Migrasi

Jika Anda masih menggunakan dataset_manager.py versi lama, berikut panduan migrasi ke versi baru:

### 1. Inisialisasi

**Versi Lama**:
```python
from smartcash.dataset import DatasetManager

dataset_manager = DatasetManager(data_dir="data")
```

**Versi Baru**:
```python
from smartcash.handlers.dataset import DatasetManager
from smartcash.config import get_config_manager

# Menggunakan config manager
config_manager = get_config_manager("configs/base_config.yaml")
config = config_manager.get_config()

dataset_manager = DatasetManager(
    config=config,
    data_dir="data",
    cache_dir=".cache/smartcash"
)
```

### 2. Loading Dataset

**Versi Lama**:
```python
train_dataset = dataset_manager.get_dataset('train')
train_loader = dataset_manager.get_dataloader('train', batch_size=16)
```

**Versi Baru**:
```python
# Cara yang sama masih bekerja
train_dataset = dataset_manager.get_dataset('train')
train_loader = dataset_manager.get_dataloader('train', batch_size=16)

# Shortcut baru tersedia
train_loader = dataset_manager.get_train_loader(batch_size=16)
val_loader = dataset_manager.get_val_loader(batch_size=16)
```

### 3. Validasi dan Augmentasi

**Versi Lama**:
```python
dataset_manager.validate_dataset('train')
dataset_manager.augment_dataset(augmentation_type='combined')
```

**Versi Baru**:
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

**Versi Lama**:
```python
dataset_manager.prepare_dataset(download=True, validate=True)
```

**Versi Baru**:
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
7. **Dokumentasi yang Lebih Baik**: Docstring lengkap dan logging yang informatifCash",
    local_path="/content/SmartCash",
    auto_mount=True
)

# Setup symlink untuk direktori penting
colab_adapter.setup_symlinks(['data', 'models', 'configs'])

# Gunakan DatasetManager dengan path lokal
# DatasetManager akan mengakses file di Google Drive melalui symlink
dataset_manager = DatasetManager(config, data_dir="/content/SmartCash/data")
```

## Performa dan Optimasi

### 1. Multiprocessing dan Paralelisme

DatasetManager mendukung operasi paralel di berbagai komponen:

- **DataLoader**: Menggunakan `num_workers` untuk loading data paralel
- **Augmentation**: Paralelisasi proses augmentasi dengan `ThreadPoolExecutor`
- **Validation**: Validasi paralel dengan multiprocessing
- **Download**: Download paralel dengan `ThreadPoolExecutor`

### 2. Caching

Implementasi cache untuk mempercepat loading:

- **CacheManager**: Integrasi dengan `utils.cache.CacheManager`
- **TTL Support**: Time-to-live untuk entri cache
- **Auto Cleanup**: Pembersihan otomatis cache yang sudah tidak digunakan
- **Memory Management**: Batasan ukuran cache untuk mengontrol penggunaan memori

### 3. Progress Tracking

Tracking progress dengan tqdm di berbagai operasi yang memakan waktu:

- Download dataset
- Augmentasi
- Validasi
- Balancing
- Loading dataset

### 4. Lazy Initialization

Komponen diinisialisasi hanya saat dibutuhkan:

```python
def _get_component(self, component_id: str, factory_func: Callable) -> Any:
    """
    Dapatkan komponen dengan lazy initialization.
    """
    if component_id not in self._components:
        self._components[component_id] = factory_func()
    return self._components[component_id]
```