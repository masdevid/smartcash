# DS01 - SmartCash Dataset Architecture Refactor Guide

## Struktur Direktori Baru

```
smartcash/dataset/
│
├── __init__.py             # Ekspor komponen publik dataset
├── manager.py              # Koordinator alur kerja dataset tingkat tinggi
│
├── services/               # Layanan khusus dataset
│   ├── __init__.py
│   │
│   ├── loader/             # Layanan loading dataset
│   │   ├── __init__.py
│   │   ├── dataset_loader.py      # Loading dataset dan dataloader
│   │   ├── multilayer_loader.py   # Loader khusus multilayer
│   │   ├── cache_manager.py       # Pengelolaan cache dataset
│   │   └── batch_generator.py     # Generator batch data
│   │
│   ├── validator/          # Layanan validasi dataset
│   │   ├── __init__.py
│   │   ├── dataset_validator.py   # Validasi dataset utama
│   │   ├── label_validator.py     # Validasi file label
│   │   ├── image_validator.py     # Validasi gambar
│   │   └── fixer.py               # Perbaikan dataset
│   │
│   ├── augmentor/          # Layanan augmentasi dataset
│   │   ├── __init__.py
│   │   ├── augmentation_service.py  # Layanan augmentasi utama
│   │   ├── image_augmentor.py     # Augmentasi gambar
│   │   ├── bbox_augmentor.py      # Augmentasi bounding box
│   │   └── pipeline_factory.py    # Pembuat pipeline augmentasi
│   │
│   ├── downloader/         # Layanan download dataset
│   │   ├── __init__.py
│   │   ├── download_service.py    # Layanan download utama
│   │   ├── roboflow_downloader.py # Download dari Roboflow
│   │   ├── local_uploader.py      # Upload dari lokal ke Colab
│   │   └── zip_processor.py       # Pemrosesan file zip
│   │
│   ├── explorer/           # Layanan eksplorasi dataset
│   │   ├── __init__.py
│   │   ├── explorer_service.py    # Layanan eksplorasi utama
│   │   ├── class_explorer.py      # Eksplorasi distribusi kelas
│   │   ├── layer_explorer.py      # Eksplorasi distribusi layer
│   │   ├── bbox_explorer.py       # Eksplorasi bounding box
│   │   └── image_explorer.py      # Eksplorasi ukuran dan format gambar
│   │
│   ├── balancer/           # Layanan balancing dataset
│   │   ├── __init__.py
│   │   ├── balance_service.py     # Layanan balancing utama
│   │   ├── undersampler.py        # Undersampling dataset
│   │   ├── oversampler.py         # Oversampling dataset
│   │   └── weight_calculator.py   # Penghitungan bobot kelas
│   │
│   └── reporter/           # Layanan pelaporan dataset
│       ├── __init__.py
│       ├── report_service.py      # Layanan pelaporan utama
│       ├── metrics_reporter.py    # Pelaporan metrik
│       ├── visualization.py       # Visualisasi dataset
│       └── export_formatter.py    # Format ekspor laporan
│
├── utils/                  # Utilitas khusus dataset
│   ├── __init__.py
│   ├── transform/                 # Transformasi dataset
│   │   ├── __init__.py
│   │   ├── albumentations_adapter.py  # Adapter untuk Albumentations
│   │   ├── bbox_transform.py      # Transformasi bounding box
│   │   ├── image_transform.py     # Transformasi gambar
│   │   ├── polygon_transform.py   # Transformasi khusus polygon (opsional)
│   │   └── format_converter.py    # Konversi format antar YOLO, COCO (opsional)
│   │
│   ├── split/                     # Utilitas split dataset
│   │   ├── __init__.py
│   │   ├── dataset_splitter.py    # Pemecah dataset
│   │   ├── merger.py              # Penggabung dataset
│   │   └── stratifier.py          # Stratified splitting
│   │
│   ├── statistics/                # Statistik dataset
│   │   ├── __init__.py
│   │   ├── class_stats.py         # Statistik kelas
│   │   ├── image_stats.py         # Statistik gambar
│   │   └── distribution_analyzer.py  # Analisis distribusi
│   │
│   ├── file/                      # Pemrosesan file
│   │   ├── __init__.py
│   │   ├── file_processor.py      # Pemroses file umum
│   │   ├── image_processor.py     # Pemroses file gambar
│   │   └── label_processor.py     # Pemroses file label
│   │
│   └── progress/                  # Tracking progres
│       ├── __init__.py
│       ├── progress_tracker.py    # Tracking progres
│       └── observer_adapter.py    # Adapter untuk observer pattern
│
└── components/             # Komponen dataset yang dapat digunakan kembali
    ├── __init__.py
    │
    ├── datasets/                  # Komponen dataset
    │   ├── __init__.py
    │   ├── base_dataset.py        # Kelas dasar dataset
    │   ├── multilayer_dataset.py  # Dataset multilayer
    │   └── yolo_dataset.py        # Dataset format YOLO
    │
    ├── geometry/                  # Komponen geometri (opsional)
    │   ├── __init__.py
    │   ├── polygon_handler.py     # Manipulasi polygon dasar
    │   ├── coord_converter.py     # Konversi koordinat antar format
    │   └── geometry_utils.py      # Utilitas geometri umum
    │
    ├── labels/                    # Komponen label
    │   ├── __init__.py
    │   ├── label_handler.py       # Penanganan label
    │   ├── multilayer_handler.py  # Penanganan label multilayer
    │   └── format_converter.py    # Konverter format label
    │
    ├── samplers/                  # Komponen sampler
    │   ├── __init__.py
    │   ├── balanced_sampler.py    # Sampler dengan balance kelas
    │   └── weighted_sampler.py    # Sampler dengan bobot
    │
    └── collate/                   # Komponen collate function
        ├── __init__.py
        ├── multilayer_collate.py  # Collate untuk multilayer
        └── yolo_collate.py        # Collate untuk YOLO
```

## Pemetaan dari Struktur Lama ke Baru

| Struktur Lama | Struktur Baru | Deskripsi Perubahan |
|---------------|---------------|---------------------|
| `handlers/dataset/dataset_manager.py` | `dataset/manager.py` | Penyederhanaan path dan fokus tanggung jawab |
| `handlers/dataset/core/dataset_loader.py` | `dataset/services/loader/dataset_loader.py` | Dipindah ke layanan loader granular |
| `handlers/dataset/core/dataset_transformer.py` | `dataset/utils/transform/image_transform.py` | Dipecah menjadi modul transformasi terpisah |
| `handlers/dataset/core/dataset_validator.py` | `dataset/services/validator/dataset_validator.py` | Dipindah ke layanan validator granular |
| `handlers/dataset/core/dataset_augmentor.py` | `dataset/services/augmentor/augmentation_service.py` | Dipindah ke layanan augmentor granular |
| `handlers/dataset/core/dataset_balancer.py` | `dataset/services/balancer/balance_service.py` | Dipindah ke layanan balancer granular |
| `handlers/dataset/core/dataset_downloader.py` | `dataset/services/downloader/download_service.py` | Dipindah ke layanan downloader granular |
| `handlers/dataset/core/file_processor.py` | `dataset/utils/file/file_processor.py` | Dipindah ke utilitas file |
| `handlers/dataset/multilayer/multilayer_dataset.py` | `dataset/components/datasets/multilayer_dataset.py` | Dipindah ke komponen datasets |
| `handlers/dataset/multilayer/multilayer_label_handler.py` | `dataset/components/labels/multilayer_handler.py` | Dipindah ke komponen labels |
| `handlers/dataset/explorers/validation_explorer.py` | `dataset/services/explorer/image_explorer.py` | Dipecah menjadi explorer terpisah |
| `handlers/dataset/explorers/distribution_explorer.py` | `dataset/services/explorer/class_explorer.py` & `.../layer_explorer.py` | Dipecah menjadi explorer terpisah |
| `handlers/dataset/explorers/bbox_image_explorer.py` | `dataset/services/explorer/bbox_explorer.py` | Dipindah ke explorer terpisah |
| `handlers/dataset/facades/*` | Terintegrasi ke `dataset/manager.py` & layanan terkait | Penyederhanaan arsitektur, facade digantikan oleh layanan |
| `utils/coordinate_utils.py` | `dataset/components/geometry/polygon_handler.py` & `.../coord_converter.py` | Dipecah menjadi komponen geometri terpisah (opsional) |
| `handlers/dataset/operations/dataset_split_operation.py` | `dataset/utils/split/dataset_splitter.py` | Dipindah ke utilitas split |
| `handlers/dataset/operations/dataset_reporting_operation.py` | `dataset/services/reporter/report_service.py` | Dipindah ke layanan reporter |
| `handlers/dataset/operations/dataset_merge_operation.py` | `dataset/utils/split/merger.py` | Dipindah ke utilitas split |
| `handlers/dataset/collate_fn.py` | `dataset/components/collate/multilayer_collate.py` & `.../yolo_collate.py` | Dipecah menjadi komponen collate terpisah |

## Konsep Arsitektur

1. **Dataset Manager**
   
   File `manager.py` bertindak sebagai koordinator utama untuk alur kerja dataset. Menyediakan antarmuka tingkat tinggi untuk layanan dataset dengan delegasi ke layanan spesifik:
   
   ```python
   class DatasetManager:
       def __init__(self, config, data_dir=None, logger=None):
           self.config = config
           self.data_dir = data_dir or config.get('data_dir', 'data')
           self.logger = logger or get_logger("dataset_manager")
           
           # Layanan akan di-inisialisasi secara lazy saat diperlukan
           self._services = {}
           
       def get_service(self, service_name):
           """Lazy initialization untuk service."""
           if service_name not in self._services:
               if service_name == 'loader':
                   from smartcash.dataset.services.loader import DatasetLoaderService
                   self._services[service_name] = DatasetLoaderService(self.config, self.data_dir, self.logger)
               # dan seterusnya untuk service lain
           return self._services[service_name]
           
       def get_dataloader(self, split, **kwargs):
           """Delegasi ke loader service."""
           return self.get_service('loader').get_dataloader(split, **kwargs)
   ```

2. **Layanan Granular**
   
   Setiap layanan dipecah menjadi subkomponen yang lebih kecil dengan tanggung jawab spesifik:
   
   ```python
   # services/loader/dataset_loader.py
   class DatasetLoader:
       """
       * old: handlers/dataset/core/dataset_loader.py
       * migrated: Fokus pada loading dataset dari disk dan membuat DataLoader
       """
       def __init__(self, config, data_dir, transformer=None, logger=None):
           self.config = config
           self.data_dir = Path(data_dir)
           self.transformer = transformer
           self.logger = logger or get_logger("dataset_loader")
   ```

3. **Utilitas Terorganisir**
   
   Utilitas diorganisir berdasarkan domain fungsional:
   
   ```python
   # utils/transform/image_transform.py
   class ImageTransformer:
       """
       * old: handlers/dataset/core/dataset_transformer.py
       * migrated: Fokus khusus pada transformasi gambar
       """
       def __init__(self, img_size=(640, 640)):
           self.img_size = img_size
   ```

4. **Komponen yang Dapat Digunakan Kembali**
   
   Komponen dataset dasar yang dapat digunakan oleh berbagai layanan:
   
   ```python
   # components/datasets/multilayer_dataset.py
   class MultilayerDataset(BaseDataset):
       """
       * old: handlers/dataset/multilayer/multilayer_dataset.py
       * migrated: Komponen dataset yang dapat digunakan oleh loader service
       """
       def __init__(self, data_path, img_size=(640, 640), mode='train', transform=None, layers=None, logger=None):
           super().__init__(data_path, img_size, mode, transform, logger)
           self.layers = layers or get_layer_config().get_layer_names()
   ```

## Dokumentasi Migrasi

Setiap metode dan kelas kritis didokumentasikan dengan catatan singkat yang menjelaskan:
- Path file baru, contoh `dataset/services/loader/dataset_loader.py`
- Lokasi implementasi sebelumnya, contoh `handlers/dataset/core/dataset_loader.py`
- Ringkasan perubahan utama

Contoh dokumentasi:
```python
class DownloadService:
    """
    * old: handlers/dataset/core/dataset_downloader.py
    * migrated: Layanan download dengan delegasi ke downloader spesifik
    * added: Dukungan untuk upload lokal dan proses file zip
    """
```

## Panduan Refaktoring

### 1. Granularitas Layanan

Setiap layanan dibagi menjadi komponen yang lebih kecil dengan tanggung jawab spesifik:

```python
# Tanpa granularitas (lama)
dataset_manager.download_dataset(workspace, project)

# Dengan granularitas (baru)
dataset_manager.download_from_roboflow(workspace, project)
dataset_manager.upload_from_local(zip_path)
```

### 2. Komposisi atas Inheritance

Gunakan pola komposisi daripada inheritance yang kompleks:

```python
# DatasetLoader yang menggunakan komponen-komponen kecil
class DatasetLoader:
    def __init__(self, config, data_dir):
        self.config = config
        self.data_dir = data_dir
        self.transformer = ImageTransformer(config.get('img_size', (640, 640)))
        self.cache_manager = CacheManager(config.get('cache_dir'))
        
    def get_dataloader(self, split):
        dataset = MultilayerDataset(
            data_path=self._get_split_path(split),
            transform=self.transformer.get_transform(split)
        )
        return DataLoader(dataset, batch_size=self.config.get('batch_size', 16))
```

### 3. Service Provider Interface

Layanan utama bertindak sebagai provider interface untuk sub-layanan terkait:

```python
# Contoh Service Provider Interface
class AugmentorService:
    def __init__(self, config, data_dir, logger=None):
        self.config = config
        self.data_dir = data_dir
        self.logger = logger
        
        # Provider untuk sub-layanan
        self.image_augmentor = ImageAugmentor(config, logger)
        self.bbox_augmentor = BBoxAugmentor(config, logger)
        self.pipeline_factory = PipelineFactory(config, logger)
    
    def augment_dataset(self, split, **kwargs):
        """Augmentasi dataset menggunakan komponen yang sesuai."""
        pipeline = self.pipeline_factory.create_pipeline(**kwargs)
        return pipeline.process_dataset(self.data_dir / split)
```

### 4. Observer Service Integration

Setiap layanan dapat menggunakan komponen progress tracking:

```python
# Integrasi observer ke layanan
from smartcash.dataset.utils.progress.observer_adapter import ProgressObserver

class ValidatorService:
    def __init__(self, config, data_dir, logger=None):
        self.config = config
        self.data_dir = data_dir
        self.logger = logger
        self.progress_observer = ProgressObserver(logger)
    
    def validate_dataset(self, split, **kwargs):
        """Validasi dataset dengan progress tracking."""
        files = self._get_files_to_validate(split)
        self.progress_observer.start_task(len(files), f"Validasi {split}")
        
        for i, file in enumerate(files):
            # Validasi file
            # ...
            
            # Update progress
            self.progress_observer.update(i + 1)
```

### 5. Adapter Pattern untuk Komponen Eksternal

Gunakan adapter pattern untuk mengintegrasikan library eksternal:

```python
# Adapter untuk Albumentations
class AlbumentationsAdapter:
    """Adapter untuk library Albumentations."""
    
    def __init__(self, img_size=(640, 640)):
        self.img_size = img_size
        
    def create_train_transform(self):
        """Buat transformasi untuk training."""
        import albumentations as A
        return A.Compose([
            A.RandomResizedCrop(height=self.img_size[1], width=self.img_size[0]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

## Contoh Penggunaan API Baru

### Inisialisasi Dataset Manager

```python
from smartcash.dataset.manager import DatasetManager

# Inisialisasi manager dengan konfigurasi
dataset_manager = DatasetManager(config)
```

### Download dan Persiapan Dataset

```python
# Download dari Roboflow
dataset_manager.download_from_roboflow(
    workspace='smartcash',
    project='rupiah-detection',
    version='3'
)

# Atau upload dari lokal
dataset_manager.upload_local_dataset('path/to/dataset.zip')

# Validasi dataset
validation_results = dataset_manager.validate_dataset('train')

# Perbaiki masalah yang ditemukan
if validation_results['invalid_labels'] > 0:
    dataset_manager.fix_dataset('train', fix_labels=True)
```

### Augmentasi dan Balancing Dataset

```python
# Augmentasi dataset
dataset_manager.augment_dataset(
    split='train',
    augmentation_types=['position', 'lighting'],
    num_variations=3
)

# Balance dataset
dataset_manager.balance_dataset(
    split='train',
    strategy='undersampling',
    target_ratio=1.5
)
```

### Mendapatkan DataLoader

```python
# Dapatkan dataloader untuk training
train_loader = dataset_manager.get_dataloader(
    split='train',
    batch_size=16,
    num_workers=4,
    shuffle=True
)

# Dapatkan dataloader untuk validasi
val_loader = dataset_manager.get_dataloader(
    split='valid',
    batch_size=16,
    num_workers=4
)
```

### Eksplorasi dan Analisis Dataset

```python
# Eksplorasi distribusi kelas
class_distribution = dataset_manager.explore_class_distribution('train')

# Eksplorasi ukuran bounding box
bbox_stats = dataset_manager.explore_bbox_statistics('train')

# Dapatkan laporan lengkap
report = dataset_manager.generate_dataset_report(
    splits=['train', 'valid', 'test'],
    include_visualizations=True
)
```

## Kesimpulan

Refaktor arsitektur dataset SmartCash dengan layanan yang lebih granular membawa beberapa keuntungan utama:

1. **Domain Boundary yang Jelas**: Setiap file dan direktori memiliki tanggung jawab yang jelas dan terfokus
2. **Mental Mapping yang Cepat**: Struktur dan penamaan yang intuitif memudahkan pemahaman codebase
3. **Kolokasi Fungsionalitas Terkait**: Kode yang terkait ditempatkan bersama dalam direktori yang sama
4. **Struktur yang Skalabel**: Mudah menambahkan layanan, utilitas, atau komponen baru
5. **Navigasi Intuitif**: Struktur direktori yang terorganisir memudahkan navigasi

Dengan pendekatan ini, kode dataset SmartCash menjadi lebih modular, dapat diuji, dan mudah dikembangkan dengan tim yang lebih besar.
