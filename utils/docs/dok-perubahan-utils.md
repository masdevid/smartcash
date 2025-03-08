# Dokumentasi Perubahan Modul Utils SmartCash

## Ringkasan Perubahan

Modul `utils` pada SmartCash telah mengalami restrukturisasi signifikan untuk meningkatkan modularitas, pemeliharaan, dan mempermudah pengembangan di masa depan. Perubahan utama meliputi:

1. **Restrukturisasi Sistem Logging** - Implementasi `SmartCashLogger` yang lebih komprehensif
2. **Reorganisasi Visualisasi** - Pemecahan fungsi visualisasi ke dalam paket terstruktur
3. **Optimalisasi Utilitas** - Penggabungan dan penyempurnaan kelas-kelas utilitas umum
4. **Integrasi Koordinat** - Penggabungan fungsi-fungsi koordinat ke dalam modul terpadu

## Perubahan Spesifik

### 1. Sistem Logging

File `logger.py` telah ditingkatkan dengan fitur-fitur:

- Dukungan thread safety menggunakan `threading.RLock`
- Output ke berbagai target (file, konsol, dan Google Colab)
- Penambahan emoji kontekstual untuk pesan log
- Dukungan teks berwarna untuk highlight pesan penting
- Metode logging khusus seperti `success`, `start`, `metric`, dll

Contoh penggunaan:

```python
from smartcash.utils.logger import get_logger

# Inisialisasi logger
logger = get_logger(
    name="training", 
    log_to_file=True,
    log_to_console=True,
    log_dir="logs"
)

# Penggunaan berbagai tipe log
logger.info("Memulai proses loading data")
logger.success("Model berhasil disimpan")
logger.error("Gagal membuka file konfigurasi")
logger.metric(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")
```

### 2. Paket Visualisasi

Modul visualisasi telah direorganisasi menjadi paket terstruktur:

```
utils/visualization/
├── __init__.py       # Ekspor komponen utama
├── base.py           # Kelas dasar visualisasi
├── detection.py      # Visualisasi deteksi objek
├── metrics.py        # Visualisasi metrik evaluasi
└── research.py       # Visualisasi hasil penelitian
```

**Kelas-kelas utama:**
- `VisualizationHelper` - Kelas dasar dengan fungsi umum
- `DetectionVisualizer` - Visualisasi hasil deteksi dengan bounding box dan label
- `MetricsVisualizer` - Visualisasi metrik seperti confusion matrix dan training curves
- `ResearchVisualizer` - Visualisasi dan analisis hasil penelitian

Contoh penggunaan:

```python
from smartcash.utils.visualization import DetectionVisualizer, MetricsVisualizer

# Visualisasi deteksi
visualizer = DetectionVisualizer(output_dir="results/deteksi")
vis_img = visualizer.visualize_detection(
    image=img,
    detections=detections,
    filename="hasil_deteksi.jpg",
    show_value=True
)

# Visualisasi metrik
metrics_vis = MetricsVisualizer(output_dir="results/metrik")
fig = metrics_vis.plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=class_names,
    title="Confusion Matrix",
    filename="confusion_matrix.png"
)
```

### 3. Utilitas Koordinat

Utilitas koordinat telah disatukan dalam kelas `CoordinateUtils`:

- Konversi antara format koordinat (YOLO, Pascal VOC, COCO)
- Operasi pada polygon (IoU, area, perimeter)
- Normalisasi koordinat untuk berbagai ukuran gambar
- Validasi koordinat bounding box dan polygon

### 4. Optimalisasi Performa

Beberapa kelas utilitas telah dioptimalkan untuk performa:

- `EnhancedCache` - Sistem caching dengan dukungan TTL dan garbage collection
- `OptimizedAugmentation` - Sistem augmentasi dengan dukungan multi-layer dan paralelisasi
- `MemoryOptimizer` - Utilitas untuk mengoptimalkan penggunaan memori GPU
- `TrainingPipeline` - Pipeline training dengan mekanisme callback dan checkpointing

### 5. Utilitas Pembantu

Penambahan berbagai utilitas pembantu:

- `ConfigManager` - Pengelolaan konfigurasi terpusat
- `EarlyStopping` - Handler early stopping dengan dukungan multiple metrics
- `DebugHelper` - Utilitas debugging untuk konfigurasi dan interface
- `EnvironmentManager` - Pendeteksi dan pengelola environment (Colab/local)

## Panduan Migrasi

### Dari Logger Lama ke SmartCashLogger

```python
# Sebelum
from smartcash.utils.simple_logger import SimpleLogger
logger = SimpleLogger("module_name")
logger.log("Memulai proses")
logger.log_error("Terjadi kesalahan")

# Sesudah
from smartcash.utils.logger import get_logger
logger = get_logger("module_name")
logger.info("Memulai proses")
logger.error("Terjadi kesalahan")
```

### Dari VisualizationUtils ke Paket Visualization

```python
# Sebelum
from smartcash.utils.visualization_utils import visualize_detections
vis_img = visualize_detections(img, detections, save_path="hasil.jpg")

# Sesudah
from smartcash.utils.visualization import visualize_detection
vis_img = visualize_detection(img, detections, output_path="hasil.jpg")
```

## Manfaat Perubahan

1. **Pemeliharaan lebih mudah** - Kode lebih terorganisir dan modular
2. **Performa lebih baik** - Optimasi untuk multi-threading dan management memori
3. **Lebih extensible** - Lebih mudah menambahkan fitur baru tanpa mengubah kode yang ada
4. **Debugging lebih baik** - Logger yang lebih kaya dan utilitas debugging
5. **Dokumentasi lebih baik** - Docstrings yang lebih lengkap dan contoh penggunaan

## Rekomendasi Penggunaan

1. Gunakan `get_logger()` untuk semua kebutuhan logging
2. Manfaatkan visualisasi terpisah sesuai kebutuhan (`DetectionVisualizer`, `MetricsVisualizer`, dll)
3. Gunakan `EnhancedCache` untuk operasi yang membutuhkan caching hasil
4. Manfaatkan paralelisasi dengan `num_workers` pada kelas yang mendukungnya
5. Gunakan `ConfigManager` untuk mengelola konfigurasi secara terpusat
