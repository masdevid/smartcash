# Panduan Pemeliharaan dan Pengembangan Modul Utils

Dokumen ini memberikan panduan untuk pemeliharaan dan pengembangan lebih lanjut pada modul utils SmartCash.

## Struktur Direktori Saat Ini

```
smartcash/utils/
├── __init__.py
├── config_manager.py
├── coordinate_utils.py
├── debug_helper.py
├── early_stopping.py
├── enhanced_cache.py
├── enhanced_dataset_validator.py
├── environment_manager.py
├── experiment_tracker.py
├── layer_config_manager.py
├── logger.py
├── memory_optimizer.py
├── metrics.py
├── model_exporter.py
├── optimized_augmentation.py
├── preprocessing.py
├── roboflow_downloader.py
├── training_pipeline.py
├── ui_utils.py
└── visualization/
    ├── __init__.py
    ├── base.py
    ├── detection.py
    ├── metrics.py
    └── research.py
```

## Praktik Terbaik untuk Pengembangan

### 1. Konvensi Penamaan

- **Nama file**: Gunakan `snake_case` dan pastikan deskriptif
- **Nama kelas**: Gunakan `PascalCase` (misalnya `EnhancedCache`, `ConfigManager`)
- **Nama metode/fungsi**: Gunakan `snake_case` (misalnya `get_cache_key`, `load_config`)
- **Konstanta**: Gunakan `UPPER_SNAKE_CASE` (misalnya `DEFAULT_CONFIG`)

### 2. Struktur File

Setiap file Python harus memiliki struktur:

```python
"""
File: smartcash/utils/nama_file.py
Author: Alfrida Sabar
Deskripsi: Deskripsi singkat tentang tujuan file
"""

# Imports standar
import os
import sys

# Imports pihak ketiga
import numpy as np
import torch

# Imports lokal
from smartcash.utils.logger import get_logger

# Definisi kelas/fungsi
class ClassName:
    """Deskripsi kelas."""
    
    def __init__(self, ...):
        """Deskripsi metode."""
        ...
```

### 3. Dokumentasi Kode

- Gunakan docstrings gaya Google untuk kelas dan metode
- Sertakan tipe parameter dan return value
- Berikan contoh penggunaan untuk fungsi-fungsi kompleks

Contoh:

```python
def process_image(image_path: str, params: Dict[str, Any]) -> np.ndarray:
    """
    Proses gambar dengan parameter yang ditentukan.
    
    Args:
        image_path: Path ke file gambar
        params: Parameter untuk pemrosesan
            - resize: Ukuran resize (tuple)
            - normalize: Flag normalisasi
            
    Returns:
        Array numpy berisi gambar yang diproses
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
    
    Example:
        >>> img = process_image("path/to/image.jpg", {"resize": (224, 224)})
    """
```

### 4. Error Handling

- Gunakan blok try-except untuk menangani kemungkinan exception
- Log error dengan detail yang cukup untuk membantu debugging
- Gunakan custom exception untuk kasus spesifik

```python
try:
    result = complex_operation()
except FileNotFoundError as e:
    logger.error(f"❌ File tidak ditemukan: {str(e)}")
    raise FileNotFoundException(f"File tidak dapat diakses: {str(e)}")
except Exception as e:
    logger.error(f"❌ Error tidak terduga: {str(e)}")
    raise
```

### 5. Logging

- Gunakan logger dari `smartcash.utils.logger`
- Gunakan level log yang sesuai (debug, info, warning, error)
- Gunakan emoji kontekstual untuk meningkatkan keterbacaan

```python
logger = get_logger(__name__)

logger.debug("🔍 Detail debugging untuk developer")
logger.info("ℹ️ Informasi umum tentang proses")
logger.warning("⚠️ Peringatan yang mungkin perlu diperhatikan")
logger.error("❌ Error yang memengaruhi operasi")
logger.success("✅ Operasi berhasil diselesaikan")
```

### 6. Threading & Concurrency

- Gunakan lock untuk operasi yang tidak thread-safe
- Gunakan ThreadPoolExecutor untuk paralelisasi
- Lindungi variabel shared dengan semaphore atau lock

```python
self._lock = threading.RLock()

with self._lock:
    # Operasi kritis yang perlu dilindungi
    self.shared_resource = new_value
```

## Panduan Pengembangan Komponen

### 1. Menambahkan Utilitas Baru

Ketika membuat utilitas baru:

1. Cek apakah bisa masuk ke dalam salah satu kategori yang ada
2. Buat file Python baru dalam `utils/` atau subpaket yang sesuai
3. Ikuti template struktur file di atas
4. Update `__init__.py` untuk mengekspos fungsi/kelas utama
5. Tambahkan dokumentasi penggunaan

### 2. Memodifikasi Utils yang Ada

Ketika memodifikasi utilitas yang ada:

1. Pertahankan kompatibilitas ke belakang jika memungkinkan
2. Tambahkan parameter dengan default values
3. Untuk perubahan besar, gunakan versioning (mis. `v1_method`, `v2_method`)
4. Update docstring untuk mencerminkan perubahan

### 3. Menambahkan Visualizer Baru

Untuk menambahkan visualisasi baru:

1. Buat file baru dalam `utils/visualization/`
2. Turunkan dari `base.VisualizationHelper` untuk konsistensi
3. Implementasikan metode utama untuk visualisasi
4. Tambahkan fungsi helper untuk penggunaan cepat
5. Update `visualization/__init__.py`

```python
"""
File: smartcash/utils/visualization/new_visualizer.py
Author: Alfrida Sabar
Deskripsi: Visualisasi untuk <tujuan spesifik>
"""

from smartcash.utils.visualization.base import VisualizationHelper
from smartcash.utils.logger import get_logger

class NewVisualizer(VisualizationHelper):
    """Visualisasi untuk <tujuan spesifik>."""
    
    def __init__(self, output_dir="results/new", logger=None):
        self.output_dir = self.create_output_directory(output_dir)
        self.logger = logger or get_logger("new_visualizer")

    def visualize_data(self, data, **kwargs):
        """Visualisasikan data."""
        # Implementasi
        ...

# Fungsi helper
def visualize_data_quick(data, output_path=None):
    """Helper untuk visualisasi cepat."""
    visualizer = NewVisualizer()
    return visualizer.visualize_data(data, filename=Path(output_path).name if output_path else None)
```

## Pengujian Komponen

### 1. Unit Tests

- Buat test untuk setiap kelas/metode utama
- Gunakan pytest untuk framework testing
- Struktur file test mirip dengan struktur modul

```
tests/utils/
├── test_config_manager.py
├── test_enhanced_cache.py
└── test_visualization/
    ├── test_detection.py
    └── test_metrics.py
```

### 2. Integrasi & Benchmark

- Buat test integrasi antar komponen utils
- Buat benchmark untuk fungsi-fungsi kritis performa
- Dokumentasikan hasil benchmark setiap perubahan signifikan

## Kontribusi

Ketika berkontribusi pada modul utils:

1. Buat branch fitur terpisah
2. Ikuti konvensi kode yang ada
3. Tambahkan/update unit test
4. Pastikan semua test lulus
5. Update dokumentasi
6. Buat pull request dengan deskripsi jelas

## Optimasi Performa

Beberapa area untuk optimasi:

1. **Caching**: Gunakan `EnhancedCache` untuk hasil komputasi yang mahal
2. **Paralelisasi**: Gunakan threading atau multiprocessing untuk operasi IO-bound atau CPU-bound
3. **Lazy Loading**: Hanya load resource berat saat dibutuhkan
4. **Batching**: Proses data dalam batch untuk mengoptimalkan memory
5. **Vectorization**: Gunakan operasi vectorized numpy alih-alih looping manual

## Tips Debugging

### 1. Menggunakan DebugHelper

Kelas `DebugHelper` dirancang khusus untuk membantu debugging issues:

```python
from smartcash.utils.debug_helper import DebugHelper

debug = DebugHelper(logger=logger)

# Cek file konfigurasi
config_check = debug.check_config_file("configs/problem_config.yaml")
print(f"Issues: {config_check['errors']}")

# Buat laporan debug
report = debug.generate_debug_report()
print(report)

# Simpan laporan ke file
debug_path = debug.save_debug_report("debug_report.txt")
```

### 2. Debugging Visualisasi

Untuk debugging visualisasi:

1. Cek parameter input (nilai NaN, ukuran array tidak konsisten)
2. Cek akses file (permission, path yang benar)
3. Cek formatting data (tipe kolom yang benar pada DataFrame)

### 3. Mencari Memory Leaks

Jika ada masalah dengan penggunaan memory:

```python
from smartcash.utils.memory_optimizer import MemoryOptimizer

optimizer = MemoryOptimizer(logger=logger)

# Cek penggunaan memory sebelum operasi
optimizer.check_gpu_status()

# Lakukan operasi yang dicurigai
suspect_function()

# Cek lagi penggunaan memory
optimizer.check_gpu_status()

# Bersihkan memory
optimizer.clear_gpu_memory()
```

## Pengelolaan Konfigurasi

### 1. Struktur Konfigurasi Standar

Berikut struktur konfigurasi yang direkomendasikan:

```yaml
# Konfigurasi umum
app:
  name: "SmartCash"
  version: "1.0.0"
  mode: "development"  # development, production

# Konfigurasi data
data:
  source: "local"  # local, roboflow
  dir: "data"
  splits:
    train: 0.8
    valid: 0.1
    test: 0.1
  roboflow:
    api_key: "YOUR_API_KEY"
    workspace: "smartcash-wo2us"
    project: "rupiah-emisi-2022"
    version: "3"

# Konfigurasi model
model:
  backbone: "efficientnet-b4"  # efficientnet-b4, cspdarknet
  input_size: [640, 640]
  weights: "pretrained"  # pretrained, null
  workers: 4

# Konfigurasi training
training:
  batch_size: 16
  epochs: 100
  optimizer: "Adam"  # Adam, SGD
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  lr_scheduler: "cosine"  # cosine, step, reduce_on_plateau
  early_stopping_patience: 10

# Konfigurasi layer
layers:
  - name: "banknote"
    description: "Deteksi uang kertas utuh"
    classes: ["001", "002", "005", "010", "020", "050", "100"]
    class_ids: [0, 1, 2, 3, 4, 5, 6]
    threshold: 0.25
    enabled: true
  - name: "nominal"
    description: "Deteksi area nominal"
    classes: ["l2_001", "l2_002", "l2_005", "l2_010", "l2_020", "l2_050", "l2_100"]
    class_ids: [7, 8, 9, 10, 11, 12, 13]
    threshold: 0.3
    enabled: true
  - name: "security"
    description: "Deteksi fitur keamanan"
    classes: ["l3_sign", "l3_text", "l3_thread"]
    class_ids: [14, 15, 16]
    threshold: 0.35
    enabled: true
```

### 2. Penggunaan ConfigManager

```python
from smartcash.utils.config_manager import ConfigManager

# Inisialisasi ConfigManager
config_manager = ConfigManager(base_dir="project_dir", logger=logger)

# Load konfigurasi dari berbagai sumber
config = config_manager.load_config("configs/experiment_config.yaml")

# Akses konfigurasi
batch_size = config.get('training', {}).get('batch_size', 16)
model_backbone = config.get('model', {}).get('backbone', 'efficientnet-b4')

# Update konfigurasi (dot notation)
config_manager.update("training.batch_size", 32)
config_manager.update("model.backbone", "cspdarknet")

# Simpan konfigurasi
config_manager.save("configs/updated_config.yaml")

# Verifikasi konfigurasi
validation_results = config_manager.validate_config()
if not validation_results['valid']:
    print(f"Konfigurasi tidak valid: {validation_results['errors']}")
```

## Rencana Pengembangan ke Depan

Berikut ini beberapa ide pengembangan untuk modul utils:

1. **Distributed Training Support**: Tambahkan utilitas untuk distributed training dengan multi-GPU
2. **Experiment Tracking Integration**: Integrasi dengan MLflow atau Weights & Biases
3. **API Wrapper**: Tambahkan utilitas untuk menyediakan model sebagai REST API
4. **Dashboard Interaktif**: Utilitas untuk membuat dashboard visualisasi interaktif
5. **Database Integration**: Tambahkan utilitas untuk menyimpan/load dari database
6. **Cloud Storage**: Tambahkan dukungan untuk storage di AWS/GCP/Azure

## FAQ dan Troubleshooting

### Cache Issues

**Q: EnhancedCache tidak menyimpan hasil?**
A: Pastikan direktori cache memiliki izin tulis. Cek juga ukuran cache dengan `cache.get_stats()`.

**Q: Data dalam cache tidak valid?**
A: Gunakan `cache.verify_integrity(fix=True)` untuk memeriksa dan memperbaiki masalah.

### Visualisasi Issues

**Q: Plot tidak muncul?**
A: Pastikan matplotlib backend yang benar digunakan. Dalam Jupyter, gunakan `%matplotlib inline`.

**Q: Label plot terpotong?**
A: Gunakan `plt.tight_layout()` atau atur parameter figsize yang lebih besar.

### Lingkungan Issues

**Q: Kode error di Google Colab tetapi berjalan di lokal?**
A: Gunakan `EnvironmentManager` untuk mendeteksi dan menangani perbedaan lingkungan.

**Q: Google Drive tidak mount di Colab?**
A: Gunakan `EnvironmentManager.mount_drive()` dengan error handling yang tepat.

## Kesimpulan

Modul utils SmartCash telah mengalami restrukturisasi signifikan untuk meningkatkan modularitas, pemeliharaan, dan kemudahan pengembangan. Panduan ini memberikan informasi untuk membantu Anda memahami, memelihara, dan mengembangkan modul ini lebih lanjut.

Ingatlah untuk selalu:
1. Mengikuti konvensi penamaan dan struktur yang konsisten
2. Mendokumentasikan kode dengan jelas
3. Menambahkan error handling yang tepat
4. Menggunakan logging untuk visibilitas
5. Menulis unit test untuk validasi

Dengan mengikuti praktik-praktik terbaik ini, modul utils akan tetap terorganisir, mudah dipelihara, dan siap untuk pengembangan fitur baru di masa depan.
