"""
# Dokumentasi Modul Dataset SmartCash

## Pendahuluan

Modul Dataset SmartCash menyediakan serangkaian komponen untuk melakukan validasi, analisis, perbaikan, dan pembersihan dataset untuk keperluan pelatihan model. Modul ini dirancang dengan pendekatan modular, memisahkan tanggung jawab ke dalam kelas-kelas khusus yang dapat digunakan secara independen atau bersama-sama.

## Struktur Modul

Modul dataset terorganisir dalam paket `smartcash.utils.dataset` dengan struktur sebagai berikut:

```
smartcash/utils/dataset/
├── __init__.py                     # Inisialisasi paket
├── dataset_analyzer.py             # Analisis statistik dataset
├── dataset_cleaner.py              # Pembersihan dataset
├── dataset_fixer.py                # Perbaikan dataset
├── dataset_utils.py                # Utilitas dataset umum
├── dataset_validator_core.py       # Validasi inti dataset
└── enhanced_dataset_validator.py   # Validator utama dataset
```

## Komponen Utama

### EnhancedDatasetValidator

Validator utama dataset yang mengkoordinasikan komponen lainnya untuk melakukan validasi menyeluruh.

```python
from smartcash.utils.dataset import EnhancedDatasetValidator

validator = EnhancedDatasetValidator(config, data_dir="data", logger=logger)

# Validasi dataset
results = validator.validate_dataset(
    split='train',
    fix_issues=True,
    move_invalid=True,
    visualize=True
)

# Analisis dataset
analysis = validator.analyze_dataset(split='train')

# Perbaiki dataset
fix_results = validator.fix_dataset(
    split='train',
    fix_coordinates=True,
    fix_labels=True,
    backup=True
)
```

### DatasetAnalyzer

Kelas untuk analisis mendalam dataset dan menghasilkan statistik.

```python
from smartcash.utils.dataset import DatasetAnalyzer

analyzer = DatasetAnalyzer(config, data_dir="data", logger=logger)

# Analisis ukuran gambar
size_stats = analyzer.analyze_image_sizes(split='train')

# Analisis keseimbangan kelas
class_stats = analyzer.analyze_class_balance(validation_results)

# Analisis keseimbangan layer
layer_stats = analyzer.analyze_layer_balance(validation_results)

# Analisis statistik bounding box
bbox_stats = analyzer.analyze_bbox_statistics(split='train')
```

### DatasetFixer

Kelas untuk memperbaiki masalah umum dalam dataset.

```python
from smartcash.utils.dataset import DatasetFixer

fixer = DatasetFixer(config, data_dir="data", logger=logger)

# Perbaiki dataset
fix_results = fixer.fix_dataset(
    split='train',
    fix_coordinates=True,  # Perbaiki koordinat yang tidak valid
    fix_labels=True,       # Perbaiki format label
    fix_images=False,      # Perbaiki gambar yang rusak
    backup=True            # Buat backup sebelum perbaikan
)
```

### DatasetCleaner

Kelas untuk membersihkan dataset dari file-file augmentasi atau yang tidak valid.

```python
from smartcash.utils.dataset import DatasetCleaner

cleaner = DatasetCleaner(config, data_dir="data", logger=logger)

# Bersihkan dataset
stats = cleaner.cleanup(
    augmented_only=True,    # Hanya hapus file hasil augmentasi
    create_backup=True,     # Buat backup sebelum menghapus
    backup_dir="backup/datasets"  # Direktori backup
)
```

### DatasetValidatorCore

Kelas inti untuk validasi dataset, fokus pada validasi gambar dan label.

```python
from smartcash.utils.dataset import DatasetValidatorCore

validator_core = DatasetValidatorCore(config, data_dir="data", logger=logger)

# Validasi satu pasang gambar-label
result = validator_core.validate_image_label_pair(
    img_path=Path("data/train/images/image1.jpg"),
    labels_dir=Path("data/train/labels")
)

# Visualisasi masalah
validator_core.visualize_issues(
    img_path=Path("data/train/images/image1.jpg"),
    result=result,
    vis_dir=Path("data/visualizations/train")
)
```

### DatasetUtils

Kelas utilitas untuk operasi umum pada dataset.

```python
from smartcash.utils.dataset import DatasetUtils

utils = DatasetUtils(logger=logger)

# Temukan file gambar
images = utils.find_image_files(Path("data/train/images"))

# Ambil sampel acak
sample = utils.get_random_sample(images, sample_size=100)

# Buat backup direktori
backup_dir = utils.backup_directory(
    source_dir=Path("data/train"),
    suffix="before_cleanup"
)

# Pindahkan file tidak valid
move_stats = utils.move_invalid_files(
    source_dir=Path("data/train/images"),
    target_dir=Path("data/invalid/train/images"),
    file_list=invalid_images
)
```

## Contoh Penggunaan Lengkap

Berikut adalah contoh penggunaan lengkap untuk workflow tipikal validasi dan analisis dataset:

```python
import yaml
from pathlib import Path
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import (
    EnhancedDatasetValidator,
    DatasetCleaner
)

# Inisialisasi logger
logger = SmartCashLogger("dataset_workflow")

# Load konfigurasi
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup path data
data_dir = Path("data")

# 1. Validasi dataset
logger.info("🔍 Memulai validasi dataset...")
validator = EnhancedDatasetValidator(config, data_dir=data_dir, logger=logger)

validation_results = validator.validate_dataset(
    split='train',
    fix_issues=True,   # Perbaiki masalah yang ditemukan
    move_invalid=True, # Pindahkan file tidak valid
    visualize=True,    # Visualisasikan masalah
    sample_size=0      # 0 = semua file
)

# 2. Analisis dataset
logger.info("📊 Memulai analisis dataset...")
analysis_results = validator.analyze_dataset(
    split='train',
    sample_size=500    # Sampel 500 gambar untuk analisis
)

# 3. Pembersihan dataset (hapus file augmentasi)
logger.info("🧹 Memulai pembersihan dataset...")
cleaner = DatasetCleaner(config, data_dir=data_dir, logger=logger)

cleanup_stats = cleaner.cleanup(
    augmented_only=True,
    create_backup=True,
    backup_dir="backup/datasets"
)

# 4. Log ringkasan
logger.success(
    f"✨ Workflow dataset selesai:\n"
    f"   • Validasi: {validation_results['valid_images']} gambar valid, "
    f"{validation_results['invalid_images']} tidak valid\n"
    f"   • Perbaikan: {validation_results['fixed_labels']} label diperbaiki\n"
    f"   • Analisis: skor ketidakseimbangan kelas = "
    f"{analysis_results['class_balance']['imbalance_score']:.1f}/10\n"
    f"   • Pembersihan: {cleanup_stats['removed']['images']} file augmentasi dihapus"
)
```

## Workflow Validasi Dataset

Modul ini mendukung workflow validasi dataset berikut:

1. **Validasi**: Periksa integritas gambar dan label, identifikasi masalah
2. **Visualisasi**: Buat visualisasi dari masalah yang ditemukan
3. **Perbaikan**: Perbaiki masalah umum seperti koordinat yang tidak valid
4. **Analisis**: Hasilkan statistik komprehensif tentang dataset
5. **Pembersihan**: Hapus file yang tidak diperlukan seperti hasil augmentasi

## Statistik dan Metrik

Modul ini menyediakan berbagai statistik dan metrik dataset:

- **Ukuran Gambar**: Distribusi ukuran dan rasio aspek gambar
- **Distribusi Kelas**: Jumlah objek per kelas
- **Distribusi Layer**: Jumlah objek per layer (banknote, nominal, security)
- **Keseimbangan Kelas**: Skor ketidakseimbangan (0-10) dan identifikasi kelas yang kurang terwakili
- **Statistik Bounding Box**: Ukuran rata-rata, rasio aspek, dan distribusi per kelas

## Mengintegrasikan dengan Training Pipeline

Modul dataset dapat diintegrasikan dengan pipeline training untuk memastikan dataset yang digunakan berkualitas tinggi:

```python
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.training import TrainingPipeline

# Validasi dataset
validator = EnhancedDatasetValidator(config, data_dir=data_dir)
validation_results = validator.validate_dataset(split='train')

# Hanya lanjutkan training jika dataset valid
if validation_results['valid_images'] / validation_results['total_images'] >= 0.95:
    # Minimal 95% gambar valid
    pipeline = TrainingPipeline(config)
    training_results = pipeline.train()
else:
    logger.error(f"❌ Dataset tidak memenuhi standar kualitas minimum")
```

## Troubleshooting

### Masalah Umum dan Solusi

1. **Gambar Tidak Dapat Dibaca**:
   - Periksa format gambar dan pastikan format didukung (JPG, PNG)
   - Perbaiki dengan `fix_dataset(fix_images=True)`

2. **Label Tidak Valid**:
   - Periksa format label YOLO (class_id, x, y, width, height)
   - Perbaiki dengan `fix_dataset(fix_labels=True)`

3. **Koordinat Di Luar Range [0,1]**:
   - Koordinat YOLO harus dalam range [0,1]
   - Perbaiki dengan `fix_dataset(fix_coordinates=True)`

4. **Ketidakseimbangan Kelas**:
   - Gunakan analisis keseimbangan kelas untuk mengidentifikasi kelas yang kurang terwakili
   - Tambahkan data atau gunakan augmentasi untuk kelas yang kurang terwakili

5. **Error Out of Memory**:
   - Kurangi `num_workers` untuk mengurangi penggunaan memori
   - Gunakan `sample_size` untuk analisis pada subset data

## Menggunakan Adapter untuk Kode Legacy

Untuk kode yang sudah menggunakan handler lama, adapter tersedia untuk mempertahankan kompatibilitas:

```python
from smartcash.handlers.dataset_cleanup import DatasetCleanupHandler

# Inisialisasi handler (cara lama)
cleanup_handler = DatasetCleanupHandler(
    config_path="configs/config.yaml",
    data_dir="data",
    backup_dir="backup"
)

# Jalankan pembersihan
stats = cleanup_handler.cleanup(
    augmented_only=True,
    create_backup=True
)
```

## Kesimpulan

Modul dataset SmartCash menyediakan set alat komprehensif untuk mengelola dataset dengan pendekatan modular yang memudahkan pemeliharaan dan pengembangan. Dengan menggunakan modul ini, Anda dapat memastikan dataset Anda berkualitas tinggi dan siap untuk digunakan dalam pelatihan model.
"""