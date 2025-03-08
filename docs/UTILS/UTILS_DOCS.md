# Dokumentasi Perubahan Modul Utils SmartCash

## Ringkasan Perubahan

Modul `utils` pada SmartCash telah mengalami restrukturisasi signifikan untuk meningkatkan modularitas, pemeliharaan, dan mempermudah pengembangan di masa depan. Perubahan utama meliputi:

1. **Restrukturisasi Sistem Logging** - Penggabungan fungsionalitas `simple_logger.py` ke dalam implementasi `SmartCashLogger` yang lebih komprehensif
2. **Paket Visualisasi Terpadu** - Migrasi dari `visualization.py` tunggal menjadi subpaket terstruktur dengan komponen khusus
3. **Reorganisasi Utilitas Koordinat** - Penggabungan `coordinate_normalizer.py` dan `polygon_metrics.py` menjadi modul terpadu `coordinate_utils.py`
4. **Restrukturisasi Training Pipeline** - Transformasi `training_pipeline.py` menjadi subpaket `training/` dengan komponen spesifik untuk berbagai aspek training
5. **Pembaruan Sistem Metrik** - Migrasi dari fungsi metrik sederhana menjadi kelas `MetricsCalculator` yang lebih powerful
6. **Penghapusan Komponen Usang** - Penghapusan `debug_helper.py` dengan pengalihan fungsionalitasnya ke modul logger yang baru
7. **Optimalisasi Performa** - Penerapan thread-safety, caching, dan teknik optimasi memory
8. **Penambahan Augmentation Utils** - Pengembangan subpaket `augmentation/` yang komprehensif untuk memperkaya dataset dengan berbagai teknik augmentasi
9. **Pembaruan Dataset Utils** - Pengembangan subpaket `dataset/` untuk validasi, pembersihan, dan analisis dataset dengan kemampuan perbaikan otomatis

## Perubahan Detail dan Panduan Migrasi

### 1. Sistem Logging

#### Perubahan dari `simple_logger.py` ke `logger.py`

File `simple_logger.py` telah digantikan oleh implementasi `SmartCashLogger` yang lebih kaya fitur dalam `logger.py`.

**Fitur Baru:**
- Thread safety menggunakan `threading.RLock`
- Output ke berbagai target (file, konsol, dan Google Colab)
- Emoji kontekstual untuk pesan log
- Dukungan teks berwarna untuk highlight pesan penting
- Metode logging khusus dan level yang lebih lengkap

#### Panduan Migrasi:

```python
# Kode lama (simple_logger.py)
from smartcash.utils.simple_logger import SimpleLogger

logger = SimpleLogger("module_name")
logger.log("Inisialisasi aplikasi")
logger.log_info("Proses berjalan...")
logger.log_error("Terjadi kesalahan")
logger.log_success("Berhasil menyimpan data")

# Kode baru (logger.py)
from smartcash.utils.logger import get_logger

logger = get_logger("module_name")
logger.info("Inisialisasi aplikasi")
logger.info("Proses berjalan...")
logger.error("Terjadi kesalahan")
logger.success("Berhasil menyimpan data")

# Tambahan metode baru yang tersedia
logger.warning("Peringatan: file tidak lengkap")
logger.start("Memulai proses training")
logger.metric("Accuracy: 92.5%, Precision: 88.7%")
logger.time("Waktu inferensi: 45.2 ms (22.1 FPS)")
logger.model("Loading model EfficientNet-B4")
logger.config("Menggunakan konfigurasi: batch_size=32")
```

#### Pemetaan Metode:

| SimpleLogger (Lama) | SmartCashLogger (Baru) | Deskripsi |
|---------------------|------------------------|-----------|
| log() | info() | Log informasi umum |
| log_info() | info() | Log informasi umum |
| log_warning() | warning() | Log peringatan |
| log_error() | error() | Log error |
| log_success() | success() | Log keberhasilan |
| N/A | start() | Log awal proses |
| N/A | metric() | Log metrik evaluasi |
| N/A | time() | Log informasi waktu |
| N/A | model() | Log informasi model |
| N/A | config() | Log informasi konfigurasi |
| N/A | progress() | Progress bar |

### 2. Visualisasi

#### Perubahan dari `visualization.py` ke Paket `visualization/`

File `visualization.py` telah dipecah menjadi subpaket terstruktur dengan komponen khusus untuk meningkatkan modularitas dan pemeliharaan:

```
utils/visualization/
├── __init__.py        # Ekspor komponen utama
├── base.py            # Kelas dasar visualisasi
├── detection.py       # Visualisasi deteksi objek
├── metrics.py         # Visualisasi metrik evaluasi
├── research.py        # Visualisasi hasil penelitian
└── research_utils.py  # Utilitas untuk research visualizer
```

#### Panduan Migrasi:

```python
# Kode lama (visualization.py)
from smartcash.utils.visualization import visualize_detections, plot_metrics, plot_confusion_matrix

# Visualisasi deteksi
result_img = visualize_detections(image, detections, output_path="hasil.jpg")

# Plot confusion matrix
plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix")

# Plot metrik training
plot_metrics(training_history, validation_history, title="Training Metrics")

# Kode baru (visualization package)
from smartcash.utils.visualization import (
    DetectionVisualizer, visualize_detection,
    MetricsVisualizer, plot_confusion_matrix
)

# Visualisasi deteksi (cara 1 - instance lengkap)
detector_vis = DetectionVisualizer(output_dir="results/deteksi")
result_img = detector_vis.visualize_detection(image, detections, filename="hasil.jpg")

# Visualisasi deteksi (cara 2 - fungsi helper)
result_img = visualize_detection(image, detections, output_path="hasil.jpg")

# Visualisasi metrics
metrics_vis = MetricsVisualizer(output_dir="results/metrics")
fig = metrics_vis.plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix")
fig = metrics_vis.plot_training_metrics(training_history, title="Training Metrics")
```

### 3. Utilitas Koordinat

#### Perubahan dari `coordinate_normalizer.py` dan `polygon_metrics.py` ke `coordinate_utils.py`

File `coordinate_normalizer.py` dan `polygon_metrics.py` telah digabungkan menjadi modul terpadu `coordinate_utils.py` untuk menyediakan API koordinat yang konsisten.

#### Panduan Migrasi:

```python
# Kode lama (coordinate_normalizer.py)
from smartcash.utils.coordinate_normalizer import normalize_coordinates, denormalize_coordinates, convert_format

# Normalisasi koordinat
norm_coords = normalize_coordinates(coords, image_size, format='pascal_voc')

# Kode lama (polygon_metrics.py)
from smartcash.utils.polygon_metrics import calculate_iou, calculate_area, calculate_perimeter

# Hitung IoU
iou = calculate_iou(bbox1, bbox2)

# Kode baru (coordinate_utils.py)
from smartcash.utils.coordinate_utils import CoordinateUtils

# Normalisasi koordinat
norm_coords = CoordinateUtils.normalize_coordinates(coords, image_size, format='pascal_voc')

# Hitung IoU
iou = CoordinateUtils.calculate_iou(bbox1, bbox2)

# Konversi format
coco_format = CoordinateUtils.yolo_to_coco(yolo_bbox, image_size)
```

### 4. Restrukturisasi Training Pipeline

#### Perubahan dari `training_pipeline.py` ke Paket `training/`

File `training_pipeline.py` telah ditransformasikan menjadi subpaket terstruktur dengan komponen khusus untuk berbagai aspek training:

```
utils/training/
├── __init__.py           # Ekspor komponen utama
├── training_pipeline.py  # Kelas utama pipeline training
├── training_callbacks.py # Sistem callback untuk event handling
├── training_metrics.py   # Pengelolaan metrik training
├── training_epoch.py     # Handler untuk satu epoch training
└── validation_epoch.py   # Handler untuk satu epoch validasi
```

#### Fitur Baru:

- **Sistem Callback Terintegrasi** - Event-driven callback system untuk hooks pada berbagai titik training
- **Pengelolaan Metrik yang Lebih Baik** - Pencatatan history metrik dengan persistensi ke CSV dan JSON
- **Manajemen Epoch Terpisah** - Pemisahan logika epoch training dan validasi untuk memudahkan kustomisasi
- **Early Stopping Fleksibel** - Konfigurasi early stopping yang lebih canggih dengan validasi multi-metrik
- **Thread-safety** - Dukungan untuk menjalankan pipeline secara paralel
- **Resume Training** - Kemampuan untuk melanjutkan training dari checkpoint

#### Panduan Migrasi:

```python
# Kode lama (training_pipeline.py)
from smartcash.utils.training_pipeline import train_model

# Training model
results = train_model(
    model, train_loader, val_loader, 
    num_epochs=30, 
    lr=0.001, 
    device='cuda'
)

# Kode baru (training package)
from smartcash.utils.training import TrainingPipeline

# Inisialisasi pipeline dengan konfigurasi
pipeline = TrainingPipeline(
    config={
        'training': {
            'epochs': 30,
            'batch_size': 16,
            'early_stopping_patience': 10
        }
    },
    model_handler=model_handler,
    data_manager=data_manager,
    logger=logger
)

# Register custom callback
def on_epoch_end(epoch, metrics, **kwargs):
    print(f"Epoch {epoch} selesai: val_loss={metrics['val_loss']:.4f}")

pipeline.register_callback('epoch_end', on_epoch_end)

# Jalankan training
results = pipeline.train(
    dataloaders={
        'train': train_loader,
        'val': val_loader
    },
    resume_from_checkpoint='path/to/checkpoint.pt',
    save_every=5
)
```

#### Pemetaan Komponen:

| Fungsi/Kelas Lama | Kelas/Metode Baru | Deskripsi |
|-------------------|-------------------|-----------|
| train_model() | TrainingPipeline.train() | Fungsi utama training |
| calculate_metrics() | ValidationEpoch._calculate_metrics() | Perhitungan metrik |
| save_checkpoint() | TrainingPipeline._handle_checkpoints() | Penyimpanan checkpoint |
| EarlyStopping | TrainingPipeline (internal) | Early stopping |

### 5. Pembaruan Sistem Metrik

#### Perubahan dari Fungsi Metrik Sederhana ke `MetricsCalculator`

File-file kecil yang berisi fungsi perhitungan metrik telah digabungkan dan digantikan oleh kelas `MetricsCalculator` yang lebih lengkap dan terstruktur.

#### Fitur Baru:

- **Tracking Metrik per Kelas** - Perhitungan metrik per kelas secara otomatis
- **Pengukuran Waktu Inferensi** - Integrasi dengan pengukuran performa
- **Konfusion Matrik** - Perhitungan dan visualisasi matriks konfusi
- **Batch Processing** - Pemrosesan batch untuk efisiensi yang lebih baik
- **Support Format Berbeda** - Mendukung berbagai format input (YOLO, COCO, Pascal VOC)
- **Persistensi** - Kemampuan menyimpan dan memuat metrik dari disk

#### Panduan Migrasi:

```python
# Kode lama (berbagai fungsi metrik)
from smartcash.utils.metrics import calculate_precision_recall, calculate_map, calculate_confusion_matrix

# Hitung metrik
precision, recall = calculate_precision_recall(predictions, targets)
mAP = calculate_map(predictions, targets)
conf_matrix = calculate_confusion_matrix(predictions, targets, num_classes)

# Kode baru (MetricsCalculator)
from smartcash.utils.metrics import MetricsCalculator

# Inisialisasi calculator
metrics_calc = MetricsCalculator()

# Update metrics dengan batch baru (dapat dilakukan berulang kali)
metrics_calc.update(predictions, targets)

# Compute final metrics
final_metrics = metrics_calc.compute()

# Akses metrik
precision = final_metrics['precision']
recall = final_metrics['recall']
mAP = final_metrics['mAP']
f1 = final_metrics['f1']
inference_time = final_metrics['inference_time']  # ms
```

### 6. Penghapusan `debug_helper.py`

File `debug_helper.py` telah dihapus dan fungsinya telah diintegrasikan ke dalam modul logging baru (`logger.py`) yang lebih kaya fitur.

#### Pemetaan Fungsi:

| Fungsi debug_helper.py (Lama) | Metode logger.py (Baru) |
|-------------------------------|-------------------------|
| debug_print() | logger.debug() |
| print_memory_usage() | Diintegrasikan ke dalam SmartCashLogger.get_system_info() |
| format_elapsed_time() | Diimplementasikan secara internal dalam logger.time() |
| setup_debugging() | Tidak diperlukan, dihandle secara otomatis oleh SmartCashLogger |

#### Panduan Migrasi:

```python
# Kode lama (debug_helper.py)
from smartcash.utils.debug_helper import debug_print, print_memory_usage, format_elapsed_time

# Debugging
debug_print("Nilai variabel:", var)
print_memory_usage()
print(f"Waktu eksekusi: {format_elapsed_time(start_time, end_time)}")

# Kode baru (logger.py)
from smartcash.utils.logger import get_logger

logger = get_logger("module_name")

# Debugging
logger.debug(f"Nilai variabel: {var}")

# Memory info
system_info = logger.get_system_info()
logger.info(f"Penggunaan GPU: {system_info['gpu_memory_used_mb']:.2f} MB")

# Timing (hitung otomatis)
import time
start_time = time.time()
# ... kode yang diukur ...
elapsed = time.time() - start_time
logger.time(f"Waktu eksekusi: {elapsed:.3f} detik")
```

### 8. Penambahan Augmentation Utils

#### Pengembangan Baru: Paket `augmentation/`

Pengembangan terbaru meliputi penambahan subpaket `augmentation/` yang komprehensif untuk memperkaya dataset dengan berbagai teknik augmentasi gambar dan label:

```
utils/augmentation/
├── __init__.py                  # Ekspor komponen utama
├── augmentation_base.py         # Kelas dasar untuk augmentasi
├── augmentation_pipeline.py     # Pipeline transformasi gambar
├── augmentation_processor.py    # Prosesor gambar dan label
├── augmentation_validator.py    # Validasi hasil augmentasi
├── augmentation_checkpoint.py   # Pengelolaan checkpoint
└── augmentation_manager.py      # Manager utama proses augmentasi
```

#### Fitur Baru:

- **Pipeline Augmentasi Lengkap** - Berbagai teknik augmentasi (posisi, pencahayaan, kombinasi, rotasi ekstrim)
- **Paralelisasi dengan ThreadPool** - Mempercepat proses augmentasi dengan multithreading
- **Validasi Hasil Otomatis** - Memvalidasi kualitas dan konsistensi hasil augmentasi
- **Checkpoint untuk Resume** - Kemampuan menyimpan progres augmentasi untuk dilanjutkan kemudian
- **Dukungan Format Multilayer** - Dukungan untuk format label multilayer dengan `LayerConfigManager`
- **Statistik Komprehensif** - Pelaporan statistik lengkap tentang prosess dan hasil augmentasi

#### Perbandingan Sebelum dan Sesudah:

**Sebelum:**
```python
import cv2
import numpy as np
import random

def augment_data(image, label, prob=0.5):
    # Basic augmentation like horizontal flip
    if random.random() < prob:
        image = cv2.flip(image, 1)
        # Manual update label coordinates for horizontal flip
        # ...
    return image, label

def process_dataset(images_dir, labels_dir):
    # Manual loop through images
    for img_file in os.listdir(images_dir):
        # Load image and label
        # Apply augmentation
        # Save results with manual naming
        # ...
```

**Sesudah:**
```python
from smartcash.utils.augmentation import AugmentationManager
from smartcash.utils.logger import get_logger

# Inisialisasi komponen
logger = get_logger("augmentation")
augmentor = AugmentationManager(
    config=config,
    output_dir="data",
    logger=logger,
    num_workers=4
)

# Jalankan augmentasi dengan berbagai teknik
stats = augmentor.augment_dataset(
    split='train',
    augmentation_types=['combined', 'lighting'],
    num_variations=3,
    output_prefix='aug',
    resume=True,
    validate_results=True
)
```

#### Panduan Migrasi:

```python
# Konfigurasi augmentasi
config = {
    'data_dir': 'data',
    'layers': ['banknote', 'nominal', 'security'],
    'training': {
        # Parameter augmentasi
        'degrees': 30,        # Rotasi maksimum
        'translate': 0.1,     # Translasi maksimum
        'scale': 0.5,         # Skala maksimum
        'fliplr': 0.5,        # Probabilitas flip horizontal
        'hsv_h': 0.015,       # Shift hue
        'hsv_s': 0.7,         # Shift saturation
        'hsv_v': 0.4          # Shift value
    }
}

# Inisialisasi manager augmentasi
augmentor = AugmentationManager(
    config=config,
    output_dir="data",
    logger=logger,
    num_workers=4
)

# Jalankan augmentasi dataset
stats = augmentor.augment_dataset(
    split='train',
    augmentation_types=['combined', 'lighting'],
    num_variations=3,
    output_prefix='aug',
    resume=True,
    validate_results=True
)
```.utils.augmentation import AugmentationManager

# Inisialisasi manager augmentasi
augmentor = AugmentationManager(
    config=config,
    output_dir="data",
    logger=logger,
    num_workers=4,
    checkpoint_interval=50
)

# Jalankan augmentasi dataset
stats = augmentor.augment_dataset(
    split='train',
    augmentation_types=['combined', 'lighting'],
    num_variations=3,
    output_prefix='aug',
    resume=True,
    validate_results=True
)

# Akses statistik hasil
print(f"Total gambar yang dihasilkan: {stats['augmented']}")
print(f"Waktu eksekusi: {stats['duration']:.2f} detik")
```

#### Komponen Utama:

1. **AugmentationManager** - Kelas utama untuk mengelola seluruh proses augmentasi
2. **AugmentationPipeline** - Definisi pipeline transformasi dengan Albumentations
3. **AugmentationProcessor** - Prosesor untuk mengaplikasikan transformasi ke gambar dan label
4. **AugmentationValidator** - Validator untuk memastikan kualitas dan konsistensi hasil
5. **AugmentationCheckpoint** - Pengelola checkpoint untuk melanjutkan proses yang terganggu

#### Konfigurasi Augmentasi:

Konfigurasi augmentasi tersedia dalam bagian `training` dari konfigurasi utama:

```python
config = {
    'training': {
        # Konfigurasi augmentasi
        'degrees': 30,        # Rotasi maksimum dalam derajat
        'translate': 0.1,     # Translasi maksimum (fraksi dari ukuran)
        'scale': 0.5,         # Penskalaan maksimum
        'fliplr': 0.5,        # Probabilitas flip horizontal
        'hsv_h': 0.015,       # Pergeseran hue maksimum
        'hsv_s': 0.7,         # Pergeseran saturation maksimum
        'hsv_v': 0.4,         # Pergeseran value maksimum
        
        # Konfigurasi training lainnya
        'epochs': 30,
        'batch_size': 16
    }
}
```

#### Alur Kerja Augmentasi:

1. **Persiapan** - Inisialisasi manager dan penyiapan direktori
2. **Penentuan Strategi** - Pemilihan jenis augmentasi dan jumlah variasi
3. **Augmentasi** - Pemrosesan gambar dan transformasi label secara paralel
4. **Validasi** - Validasi konsistensi dan kualitas hasil augmentasi
5. **Laporan** - Generasi laporan statistik lengkap

#### Checkpoint dan Resume:

Untuk proses augmentasi yang panjang, sistem checkpoint memungkinkan penghentian dan melanjutkan proses:

```python
# Jalankan dengan resume=True untuk melanjutkan proses yang terganggu
stats = augmentor.augment_dataset(
    split='train',
    resume=True
)
```

### 9. Pembaruan Dataset Utils

#### Pengembangan Baru: Paket `dataset/`

Modul baru `dataset/` telah ditambahkan untuk mempermudah validasi, pembersihan, dan analisis dataset dengan kemampuan perbaikan otomatis:

```
utils/dataset/
├── __init__.py                   # Ekspor komponen utama
├── dataset_analyzer.py           # Analisis mendalam dataset
├── dataset_cleaner.py            # Pembersihan dataset
├── dataset_fixer.py              # Perbaikan otomatis masalah dataset
├── dataset_utils.py              # Utilitas umum dataset
├── dataset_validator_core.py     # Inti validasi dataset
└── enhanced_dataset_validator.py # Validator dataset yang ditingkatkan
```

#### Fitur Baru:

- **Validasi Multilayer** - Validasi dataset multi-layer dengan format YOLO
- **Analisis Distribusi** - Analisis distribusi kelas, ukuran gambar, dan bounding box
- **Perbaikan Otomatis** - Deteksi dan perbaikan koordinat yang tidak valid
- **Pembersihan Dataset** - Pemindahan dan pembersihan file augmentasi atau tidak valid
- **Proses Paralel** - Dukungan multithreading untuk proses validasi dan perbaikan
- **Visualisasi Masalah** - Pembuatan visualisasi otomatis untuk file yang bermasalah

#### Perbandingan Sebelum dan Sesudah:

**Sebelum:**
```python
def validate_dataset(data_dir):
    issues = []
    for img_file in os.listdir(f"{data_dir}/images"):
        # Baca gambar secara manual
        img = cv2.imread(f"{data_dir}/images/{img_file}")
        if img is None:
            issues.append(f"Gambar {img_file} tidak dapat dibaca")
            continue
            
        # Periksa label
        label_file = f"{data_dir}/labels/{os.path.splitext(img_file)[0]}.txt"
        if not os.path.exists(label_file):
            issues.append(f"Label untuk {img_file} tidak ditemukan")
            continue
            
        # Baca dan validasi label secara manual
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Validasi format dan koordinat secara manual
                # ...
    
    return issues

# Penggunaan:
issues = validate_dataset("data/train")
print(f"Ditemukan {len(issues)} masalah")
```

**Sesudah:**
```python
from smartcash.utils.dataset import EnhancedDatasetValidator

validator = EnhancedDatasetValidator(
    config=config,
    data_dir="data",
    logger=logger,
    num_workers=4
)

# Validasi dengan fitur perbaikan otomatis dan paralel processing
validation_stats = validator.validate_dataset(
    split='train',
    fix_issues=True,  # Perbaiki masalah otomatis
    move_invalid=True,  # Pindahkan file tidak valid
    visualize=True,  # Buat visualisasi masalah
    sample_size=0  # 0 = proses semua file
)

# Analisis statistik dataset
analysis = validator.analyze_dataset(split='train')

# Informasi terstruktur tentang validasi dan statistik
print(f"Gambar valid: {validation_stats['valid_images']}/{validation_stats['total_images']}")
print(f"Label valid: {validation_stats['valid_labels']}/{validation_stats['total_labels']}")
print(f"Ukuran gambar dominan: {analysis['image_size_distribution']['dominant_size']}")
print(f"Ketidakseimbangan kelas: {analysis['class_balance']['imbalance_score']:.2f}/10")
```

#### Panduan Migrasi:

```python
from smartcash.utils.dataset import EnhancedDatasetValidator, DatasetAnalyzer, DatasetFixer, DatasetCleaner
from smartcash.utils.logger import get_logger

# Inisialisasi logger
logger = get_logger("dataset_validator")

# Inisialisasi validator
validator = EnhancedDatasetValidator(
    config=config,
    data_dir="data",
    logger=logger,
    num_workers=4
)

# Validasi dataset
validation_stats = validator.validate_dataset(
    split='train',
    fix_issues=True,  # Perbaiki masalah otomatis
    move_invalid=True,  # Pindahkan file tidak valid
    visualize=True,  # Buat visualisasi masalah
    sample_size=0  # 0 = proses semua file
)

# Pembersihan dataset (opsional)
cleaner = DatasetCleaner(config, "data", logger)
cleanup_stats = cleaner.cleanup(
    augmented_only=True,  # Hanya hapus file hasil augmentasi
    create_backup=True    # Buat backup sebelum menghapus
)
```

#### Penggunaan Baru:

```python
from smartcash.utils.dataset import EnhancedDatasetValidator, DatasetAnalyzer, DatasetFixer, DatasetCleaner

# Inisialisasi validator dataset
validator = EnhancedDatasetValidator(
    config=config,
    data_dir="data",
    logger=logger,
    num_workers=4
)

# Validasi dataset
validation_stats = validator.validate_dataset(
    split='train',
    fix_issues=True,  # Perbaiki masalah otomatis
    move_invalid=True,  # Pindahkan file tidak valid ke direktori terpisah
    visualize=True,  # Buat visualisasi masalah
    sample_size=0  # 0 = proses semua file
)

# Analisis lebih mendalam
analysis = validator.analyze_dataset(split='train')

# Bersihkan file augmentasi yang tidak diinginkan
cleaner = DatasetCleaner(config, "data", logger)
cleanup_stats = cleaner.cleanup(
    augmented_only=True,  # Hanya hapus file hasil augmentasi
    create_backup=True,  # Buat backup sebelum menghapus
)

# Perbaikan masalah tertentu
fixer = DatasetFixer(config, "data", logger)
fix_stats = fixer.fix_dataset(
    split='train',
    fix_coordinates=True,
    fix_labels=True,
    fix_images=False,
    backup=True
)
```

#### Komponen Utama:

1. **EnhancedDatasetValidator** - Kelas utama untuk validasi dataset
2. **DatasetAnalyzer** - Analisis distribusi dan statistik dataset
3. **DatasetFixer** - Perbaikan otomatis berbagai masalah dataset
4. **DatasetCleaner** - Pembersihan dan pemindahan file tidak valid
5. **DatasetUtils** - Utilitas umum untuk operasi pada dataset

#### Alur Kerja Validasi Dataset:

1. **Validasi** - Periksa integritas gambar dan label
2. **Analisis** - Analisis distribusi kelas dan ukuran objek
3. **Perbaikan** - Perbaiki masalah yang terdeteksi (opsional)
4. **Visualisasi** - Buat visualisasi masalah (opsional)
5. **Reporting** - Laporan statistik komprehensif

## Panduan Migrasi Dataset Utils

Untuk mengimplementasikan validasi dan analisis dataset menggunakan komponen baru, ikuti langkah-langkah berikut:

### 1. Konfigurasi Dataset Utils

```python
config = {
    'data_dir': 'data',
    'layers': ['banknote', 'nominal', 'security'],
    'cleanup': {
        'augmentation_patterns': [
            r'aug_.*',
            r'.*_augmented.*',
            r'.*_modified.*'
        ],
        'ignored_patterns': [
            r'.*\.gitkeep',
            r'.*\.DS_Store'
        ]
    }
}
```

### 2. Validasi Dataset

```python
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.logger import get_logger

# Inisialisasi logger
logger = get_logger("dataset_validator")

# Inisialisasi validator
validator = EnhancedDatasetValidator(
    config=config,
    data_dir="data",
    logger=logger,
    num_workers=4
)

# Validasi dataset
validation_stats = validator.validate_dataset(
    split='train',
    fix_issues=True,  # Perbaiki masalah otomatis
    move_invalid=True,  # Pindahkan file tidak valid
    visualize=True,  # Buat visualisasi masalah
    sample_size=0  # 0 = proses semua file
)

# Cetak ringkasan hasil
print(f"✨ Validasi selesai: {validation_stats['valid_images']}/{validation_stats['total_images']} gambar valid")
print(f"📊 Label valid: {validation_stats['valid_labels']}/{validation_stats['total_labels']} label")
```

### 3. Analisis Dataset

```python
# Lakukan analisis mendalam
analysis = validator.analyze_dataset(split='train', sample_size=100)

# Periksa hasil
print(f"📏 Ukuran gambar dominan: {analysis['image_size_distribution']['dominant_size']}")
print(f"📊 Ketidakseimbangan kelas: {analysis['class_balance']['imbalance_score']:.2f}/10")
print(f"📊 Ketidakseimbangan layer: {analysis['layer_balance']['imbalance_score']:.2f}/10")
```

### 4. Pembersihan Dataset

```python
from smartcash.utils.dataset import DatasetCleaner

# Inisialisasi cleaner
cleaner = DatasetCleaner(
    config=config,
    data_dir="data",
    logger=logger,
    num_workers=4
)

# Jalankan pembersihan
cleanup_stats = cleaner.cleanup(
    augmented_only=True,  # Hanya hapus file hasil augmentasi
    create_backup=True,  # Buat backup sebelum menghapus
)

# Periksa hasil
print(f"🧹 Pembersihan selesai: {cleanup_stats['removed']['images']} gambar dihapus")
print(f"🧹 Label dihapus: {cleanup_stats['removed']['labels']} file")
```

### 5. Perbaikan Dataset

```python
from smartcash.utils.dataset import DatasetFixer

# Inisialisasi fixer
fixer = DatasetFixer(
    config=config,
    data_dir="data",
    logger=logger
)

# Jalankan perbaikan
fix_stats = fixer.fix_dataset(
    split='train',
    fix_coordinates=True,  # Perbaiki koordinat tidak valid
    fix_labels=True,       # Perbaiki format label
    fix_images=False,      # Jangan perbaiki gambar
    backup=True            # Buat backup
)

# Periksa hasil
print(f"🔧 Perbaikan selesai: {fix_stats['processed']} gambar diproses")
print(f"🔧 Label diperbaiki: {fix_stats['fixed_labels']} file")
print(f"🔧 Koordinat diperbaiki: {fix_stats['fixed_coordinates']} koordinat")
```

## Kesimpulan

Perubahan besar dalam modul `utils` SmartCash telah secara signifikan meningkatkan modularitas, kemudahan pemeliharaan, dan performa kode dengan memperkenalkan struktur yang lebih terorganisir dan komponen-komponen khusus. 

Perubahan utama yang telah diimplementasikan meliputi:

1. **Sistem Logging**: Dari `simple_logger.py` menjadi `SmartCashLogger` yang komprehensif dengan dukungan emoji, warna, dan berbagai target output

2. **Visualisasi**: Dari file tunggal menjadi paket terstruktur yang mencakup visualisasi deteksi objek, metrik, dan hasil penelitian

3. **Utilitas Koordinat**: Penggabungan `coordinate_normalizer.py` dan `polygon_metrics.py` ke dalam API koordinat terpadu

4. **Training Pipeline**: Restrukturisasi menjadi kelas modular dengan sistem callback dan pemisahan komponen-komponen

5. **Sistem Metrik**: Pengembangan kelas `MetricsCalculator` yang komprehensif untuk menggantikan fungsi-fungsi metrik sederhana

6. **Augmentation Utils**: Penambahan subpaket untuk augmentasi data dengan berbagai metode dan dukungan paralelisme

7. **Dataset Utils**: Pengembangan alat untuk validasi, analisis, dan perbaikan dataset yang komprehensif

Perubahan ini mendorong pengembangan model deteksi mata uang Rupiah yang lebih akurat dan robust dengan performa yang lebih baik, keandalan yang lebih tinggi, dan kemudahan pengembangan di masa depan.