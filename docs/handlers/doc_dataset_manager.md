# Dokumentasi DatasetManager SmartCash

## Deskripsi

`DatasetManager` adalah komponen pusat untuk pengelolaan dataset multilayer mata uang Rupiah di SmartCash. Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu dan modular bagi berbagai operasi dataset.

## Struktur File Lengkap

```
smartcash/handlers/dataset/
├── __init__.py                     # Export komponen utama
├── dataset_manager.py              # Entry point minimal (facade utama)
│
├── core/                           # Komponen inti dataset
│   ├── dataset_loader.py           # Loader dataset
│   ├── dataset_downloader.py       # Downloader dataset
│   ├── dataset_transformer.py      # Transformasi dataset
│   ├── dataset_validator.py        # Validator dataset
│   ├── dataset_augmentor.py        # Augmentasi dataset
│   └── dataset_balancer.py         # Penyeimbangan dataset
│
├── facades/                        # Facade untuk operasi dataset
│   ├── pipeline_facade.py          # Facade untuk pipeline lengkap
│   ├── data_loading_facade.py      # Facade untuk loading data
│   ├── data_processing_facade.py   # Facade untuk preprocessing
│   ├── data_operations_facade.py   # Facade untuk operasi dataset
│   ├── visualization_facade.py     # Facade untuk visualisasi
│   └── dataset_explorer_facade.py  # Facade untuk eksplorasi dataset
│
├── explorers/                      # Komponen eksplorasi dataset
│   ├── base_explorer.py            # Kelas dasar explorer
│   ├── validation_explorer.py      # Explorer untuk validasi
│   ├── bbox_image_explorer.py      # Explorer untuk bbox dan ukuran gambar
│   └── distribution_explorer.py    # Explorer untuk distribusi kelas/layer
│
├── multilayer/                     # Komponen dataset multilayer
│   ├── multilayer_dataset_base.py  # Kelas dasar dataset multilayer
│   ├── multilayer_dataset.py       # Implementasi dataset multilayer
│   └── multilayer_label_handler.py # Handler label multilayer
│
├── operations/                     # Operasi manipulasi dataset
│   ├── dataset_split_operation.py  # Operasi split dataset
│   ├── dataset_merge_operation.py  # Operasi merge dataset
│   └── dataset_reporting_operation.py  # Operasi pelaporan dataset
│
└── dataset_utils_adapter.py        # Adapter untuk integrasi utils/dataset
```

## Fitur Utama

### 1. Arsitektur Modular
- Facade pattern untuk antarmuka terpadu
- Komponen terpisah dengan tanggung jawab spesifik
- Lazy loading untuk efisiensi memori

### 2. Manajemen Dataset Multilayer
- Dukungan untuk dataset dengan beberapa layer deteksi
- Fleksibilitas dalam konfigurasi layer
- Penanganan label multilayer yang kompleks

### 3. Preprocessing Dataset
- Transformasi gambar dengan Albumentations
- Augmentasi dengan berbagai teknik
- Validasi dan pembersihan dataset
- Penyeimbangan kelas

### 4. Loading dan Transformasi
- DataLoader PyTorch yang teroptimasi
- Konfigurasi transformasi yang fleksibel
- Dukungan untuk berbagai mode (train/val/test)

### 5. Eksplorasi dan Analisis
- Statistik distribusi kelas dan layer
- Analisis ukuran gambar dan bounding box
- Visualisasi dataset

## Kelas Utama

### 1. DatasetManager (PipelineFacade)
Facade utama yang menggabungkan semua komponen:

```python
class DatasetManager(PipelineFacade):
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

### 2. MultilayerDataset
Dataset khusus untuk mendukung multilayer:

```python
class MultilayerDataset(MultilayerDatasetBase):
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
        """Inisialisasi MultilayerDataset."""
```

### 3. DatasetUtilsAdapter
Adapter untuk integrasi komponen utils:

```python
class DatasetUtilsAdapter:
    def __init__(
        self, 
        config: Dict, 
        data_dir: Optional[str] = None, 
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi adapter."""
```

## Metode Utama

### Setup Pipeline Lengkap
```python
def setup_full_pipeline(self, **kwargs) -> Dict[str, Any]:
    """Setup pipeline lengkap untuk dataset."""
    # Download, validasi, augmentasi, penyeimbangan
```

### Loading Dataset
```python
def get_dataset(self, split: str, **kwargs):
    """Dapatkan dataset untuk split tertentu."""

def get_dataloader(self, split: str, **kwargs):
    """Dapatkan dataloader untuk split tertentu."""
```

### Preprocessing
```python
def validate_dataset(self, split: str, **kwargs):
    """Validasi dataset."""

def augment_dataset(self, **kwargs):
    """Augmentasi dataset."""

def balance_by_undersampling(self, split: str, **kwargs):
    """Seimbangkan dataset dengan undersampling."""
```

### Eksplorasi
```python
def analyze_dataset(self, split: str, **kwargs):
    """Analisis mendalam dataset."""

def visualize_class_distribution(self, split: str = 'train', **kwargs):
    """Visualisasikan distribusi kelas."""
```

## Konfigurasi

Konfigurasi dataset dalam format dictionary yang mendukung:
- Direktori data
- Konfigurasi layer
- Parameter preprocessing
- Pengaturan augmentasi

## Integrasi dengan Komponen Lain

1. **CheckpointManager**: Loading model dari checkpoint
2. **ModelManager**: Training dan evaluasi model
3. **Observer Pattern**: Monitoring event dataset
4. **Google Colab**: Dukungan lingkungan notebook

## Cara Penggunaan

### 1. Inisialisasi DatasetManager

```python
from smartcash.handlers.dataset import DatasetManager
from smartcash.config import get_config_manager

# Dapatkan konfigurasi
config_manager = get_config_manager("configs/rupiah_detection.yaml")
config = config_manager.get_config()

# Inisialisasi DatasetManager
dataset_manager = DatasetManager(
    config=config,
    data_dir="data/rupiah_banknotes",
    cache_dir=".cache/smartcash"
)
```

### 2. Download dan Setup Dataset

```python
# Download dataset dari Roboflow
dataset_path = dataset_manager.download_dataset(
    format="yolov5", 
    show_progress=True
)

# Export ke struktur lokal
train_dir, val_dir, test_dir = dataset_manager.export_to_local(dataset_path)
```

### 3. Validasi Dataset

```python
# Validasi dataset training
validation_results = dataset_manager.validate_dataset(
    split='train', 
    fix_issues=True,  # Perbaiki masalah otomatis
    visualize=True    # Buat visualisasi masalah
)
```

### 4. Augmentasi Dataset

```python
# Augmentasi dataset training
augmentation_stats = dataset_manager.augment_dataset(
    split='train',
    augmentation_types=['combined', 'lighting'],
    num_variations=2,
    validate_results=True
)
```

### 5. Penyeimbangan Kelas

```python
# Seimbangkan dataset dengan undersampling
balance_results = dataset_manager.balance_by_undersampling(
    split='train', 
    target_ratio=2.0,  # Rasio maksimal antar kelas
    backup=True
)
```

### 6. Loading DataLoader

```python
# Dapatkan DataLoader untuk training
train_loader = dataset_manager.get_train_loader(
    batch_size=16,
    num_workers=4,
    shuffle=True
)

# Dapatkan semua DataLoader sekaligus
all_loaders = dataset_manager.get_all_dataloaders(
    batch_size=16,
    require_all_layers=True
)
```

### 7. Analisis Dataset

```python
# Analisis distribusi kelas
class_analysis = dataset_manager.analyze_dataset(
    split='train', 
    sample_size=1000,  # Analisis sampel
    detailed=True
)

# Visualisasi distribusi
dataset_manager.visualize_class_distribution(
    split='train', 
    save_path='visualizations/class_distribution.png'
)
```

### 8. Pipeline Lengkap

```python
# Setup pipeline lengkap
pipeline_results = dataset_manager.setup_full_pipeline(
    download_dataset=True,
    validate_dataset=True,
    fix_issues=True,
    augment_data=True,
    balance_classes=True,
    visualize_results=True
)
```

## Kesimpulan

`DatasetManager` menyediakan:
1. Antarmuka terpadu untuk operasi dataset
2. Fleksibilitas dalam konfigurasi dan preprocessing
3. Dukungan penuh untuk dataset multilayer
4. Optimasi performa dengan lazy loading
5. Integrasi mulus dengan komponen SmartCash