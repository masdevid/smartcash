# Contoh Penggunaan Modul Utils SmartCash

Berikut ini contoh penggunaan modul utils SmartCash yang telah diperbarui untuk membantu Anda mengintegrasikan komponen-komponen ini dengan mudah ke dalam kode Anda.

## 1. Penggunaan Logger

### Setup Logger dengan Kustomisasi

```python
from smartcash.utils.logger import get_logger, SmartCashLogger

# Cara 1: Menggunakan fungsi factory
logger = get_logger(
    name="trainer",
    level=logging.INFO,
    log_to_file=True,
    log_to_console=True,
    log_to_colab=None,  # None untuk deteksi otomatis
    log_dir="logs/training"
)

# Cara 2: Inisialisasi langsung
custom_logger = SmartCashLogger(
    name="detector",
    level=logging.INFO,
    log_to_file=True,
    log_to_console=True,
    use_colors=True,
    use_emojis=True
)
```

### Contoh Penggunaan Logger

```python
# Logging berbagai tipe pesan
logger.info("Memulai training epoch 10")
logger.success("Model berhasil disimpan ke runs/weights/best.pt")
logger.warning("Accuracy menurun, mungkin perlu menyesuaikan learning rate")
logger.error("File konfigurasi tidak ditemukan")

# Logging informasi khusus
logger.start("Memulai proses augmentasi dataset")
logger.metric(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
logger.model("Loading model EfficientNet-B4 dengan backbone YOLO")
logger.time(f"Waktu inferensi: {inference_time:.3f} ms ({fps:.2f} FPS)")
logger.config("Konfigurasi training: batch_size=32, epochs=50, lr=0.001")

# Menggunakan progress bar
with logger.progress(total=100, desc="Preprocessing") as pbar:
    for i in range(100):
        # Proses
        pbar.update(1)
```

## 2. Penggunaan Visualizer

### Visualisasi Deteksi Mata Uang

```python
from smartcash.utils.visualization import DetectionVisualizer, visualize_detection

# Cara 1: Menggunakan kelas lengkap
visualizer = DetectionVisualizer(output_dir="results/deteksi")

# Visualisasi satu gambar
hasil_img = visualizer.visualize_detection(
    image=img,
    detections=detections,
    filename="hasil_deteksi_100rb.jpg",
    conf_threshold=0.35,
    show_labels=True,
    show_conf=True,
    show_total=True,
    show_value=True
)

# Visualisasi grid gambar
grid_img = visualizer.visualize_detections_grid(
    images=[img1, img2, img3, img4],
    detections_list=[detections1, detections2, detections3, detections4],
    title="Hasil Deteksi Berbagai Denominasi",
    filename="grid_deteksi.jpg",
    grid_size=(2, 2)
)

# Cara 2: Menggunakan fungsi helper sederhana
hasil_img = visualize_detection(
    image=img,
    detections=detections,
    output_path="results/deteksi_cepat.jpg",
    conf_threshold=0.25
)
```

### Visualisasi Metrik Evaluasi

```python
from smartcash.utils.visualization import MetricsVisualizer, plot_confusion_matrix

# Inisialisasi visualizer metrik
metrics_vis = MetricsVisualizer(output_dir="results/metrik")

# Visualisasi confusion matrix
fig = metrics_vis.plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=["1rb", "2rb", "5rb", "10rb", "20rb", "50rb", "100rb"],
    title="Confusion Matrix Deteksi Denominasi",
    filename="confusion_matrix.png",
    normalize=True
)

# Visualisasi metrik training
fig = metrics_vis.plot_training_metrics(
    metrics=training_history,
    title="Training Metrics EfficientNet-B4",
    filename="training_metrics.png",
    include_lr=True
)

# Visualisasi perbandingan model
fig = metrics_vis.plot_model_comparison(
    comparison_data=model_comparison_df,
    metric_cols=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP'],
    title="Perbandingan Model Deteksi Mata Uang",
    filename="model_comparison.png"
)

# Visualisasi dengan fungsi helper
fig = plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=class_names,
    output_path="confusion_matrix_simple.png"
)
```

### Visualisasi Hasil Penelitian

```python
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi visualizer penelitian
research_vis = ResearchVisualizer(output_dir="results/penelitian")

# Visualisasi perbandingan eksperimen
result = research_vis.visualize_experiment_comparison(
    results_df=experiment_df,
    title="Perbandingan Model Backbone",
    filename="backbone_comparison.png",
    highlight_best=True
)

# Akses hasil analisis
analysis = result['analysis']
best_model = analysis['best_model']['name']
recommendation = analysis['recommendation']
styled_df = result['styled_df']

print(f"Model terbaik: {best_model}")
print(f"Rekomendasi: {recommendation}")
display(styled_df)  # Untuk notebook

# Visualisasi skenario penelitian
result = research_vis.visualize_scenario_comparison(
    results_df=scenario_df,
    title="Perbandingan Skenario Pencahayaan",
    filename="scenario_comparison.png"
)
```

## 3. Penggunaan Cache dan Optimasi

### EnhancedCache untuk Menyimpan Hasil Preprocessing

```python
from smartcash.utils.enhanced_cache import EnhancedCache

# Inisialisasi cache
cache = EnhancedCache(
    cache_dir=".cache/preprocessing",
    max_size_gb=1.0,
    ttl_hours=24,
    auto_cleanup=True,
    logger=logger
)

# Dapatkan cache key berdasarkan file dan parameter
cache_key = cache.get_cache_key(file_path, preprocessing_params)

# Coba ambil dari cache dulu
cached_result = cache.get(cache_key)
if cached_result is not None:
    # Gunakan hasil dari cache
    processed_data = cached_result
else:
    # Proses data jika tidak ada di cache
    processed_data = preprocess_function(file_path, **preprocessing_params)
    
    # Simpan hasil ke cache
    cache.put(cache_key, processed_data)

# Verifikasi dan bersihkan cache
cache.verify_integrity(fix=True)
stats = cache.get_stats()
print(f"Cache hit ratio: {stats['hit_ratio']:.2f}%")
```

### Optimasi Memori untuk GPU

```python
from smartcash.utils.memory_optimizer import MemoryOptimizer

# Inisialisasi optimizer
memory_optimizer = MemoryOptimizer(logger=logger)

# Cek status GPU
memory_optimizer.check_gpu_status()

# Bersihkan memori GPU
freed_memory = memory_optimizer.clear_gpu_memory()
print(f"Memori yang dibebaskan: {freed_memory:.2f} MB")

# Optimasi model untuk inferensi
optimized_model = memory_optimizer.optimize_for_inference(model)

# Temukan batch size optimal
optimal_batch_size = memory_optimizer.optimize_batch_size(
    model, 
    target_memory_usage=0.7
)
print(f"Batch size optimal: {optimal_batch_size}")
```

## 4. Penggunaan Utilitas Koordinat

```python
from smartcash.utils.coordinate_utils import CoordinateUtils

# Konversi format koordinat
yolo_coords = [0.5, 0.5, 0.3, 0.2]  # x_center, y_center, width, height
image_size = (640, 480)  # width, height

# Konversi YOLO ke pixel coordinates (x1, y1, x2, y2)
pixel_coords = CoordinateUtils.yolo_to_pixel(yolo_coords, image_size)

# Konversi YOLO ke COCO format
coco_bbox = CoordinateUtils.yolo_to_coco(yolo_coords, image_size)

# Hitung IoU antara dua bounding box
iou = CoordinateUtils.calculate_iou(bbox1, bbox2)

# Validasi koordinat
valid = CoordinateUtils.validate_bbox(yolo_coords, format='yolo')
if not valid:
    corrected_coords = CoordinateUtils.correct_bbox(yolo_coords, format='yolo')
```

## 5. Penggunaan Augmentasi Teroptimasi

```python
from smartcash.utils.optimized_augmentation import OptimizedAugmentation

# Inisialisasi augmentasi
augmentor = OptimizedAugmentation(
    config=config,
    output_dir="data/augmented",
    logger=logger,
    num_workers=4
)

# Augmentasi dataset
stats = augmentor.augment_dataset(
    split='train',
    augmentation_types=['position', 'lighting', 'combined'],
    num_variations=3,
    output_prefix='aug',
    resume=True,
    validate_results=True
)

# Lihat statistik hasil
print(f"Total hasil augmentasi: {stats['augmented']}")
print(f"Gagal: {stats['failed']}")
print(f"Statistik per tipe:")
for aug_type, count in stats['per_type'].items():
    print(f"  - {aug_type}: {count}")
```

## 6. Penggunaan Konfigurasi Terpusat

```python
from smartcash.utils.config_manager import ConfigManager

# Inisialisasi config manager
config_manager = ConfigManager(
    base_dir="project_root",
    logger=logger
)

# Load konfigurasi
config = config_manager.load_config("configs/experiment_config.yaml")

# Akses dan update konfigurasi
batch_size = config.get('training', {}).get('batch_size', 32)
config_manager.update("training.batch_size", 64)
config_manager.update("model.backbone", "efficientnet-b4")

# Simpan konfigurasi
config_manager.save()

# Dukungan deep update dan merge
updated_config = config_manager.deep_update(
    current_config=config,
    new_values={"training": {"optimizer": {"lr": 0.001}}}
)
```

## 7. Penggunaan Training Pipeline

```python
from smartcash.utils.training_pipeline import TrainingPipeline

# Inisialisasi pipeline
pipeline = TrainingPipeline(
    config=config,
    model_handler=model_handler,
    data_manager=data_manager,
    logger=logger
)

# Register callback
def on_epoch_end(epoch, metrics):
    print(f"Epoch {epoch}: val_loss={metrics['val_loss']:.4f}")

pipeline.register_callback('epoch_end', on_epoch_end)

# Jalankan training
result = pipeline.train(
    dataloaders=dataloaders,
    resume_from_checkpoint="runs/last.pt",
    save_every=5,
    epochs=50
)

# Akses hasil
best_val_loss = result['best_val_loss']
training_history = result['training_history']
```

## 8. Penggunaan Debugging Helper

```python
from smartcash.utils.debug_helper import DebugHelper

# Inisialisasi debug helper
debug_helper = DebugHelper(logger=logger)

# Periksa file konfigurasi
result = debug_helper.check_config_file("configs/base_config.yaml")
if not result['valid_yaml']:
    print(f"Error dalam YAML: {result['errors']}")

# Debug masalah penyimpanan konfigurasi
test_result = debug_helper.test_config_save(
    config_manager, 
    test_config={"test_key": "test_value"}
)

# Generate dan simpan laporan debug
report = debug_helper.generate_debug_report()
report_path = debug_helper.save_debug_report("debug_report.txt")
```

## 9. Penggunaan Environment Manager

```python
from smartcash.utils.environment_manager import EnvironmentManager

# Inisialisasi environment manager
env_manager = EnvironmentManager(logger=logger)

# Cek apakah berjalan di Colab
if env_manager.is_colab:
    # Mount Google Drive
    success, msg = env_manager.mount_drive()
    print(msg)
    
    # Setup symlinks
    env_manager.create_symlinks()

# Setup direktori project
stats = env_manager.setup_directories(use_drive=env_manager.is_drive_mounted)

# Dapatkan path yang disesuaikan dengan environment
data_path = env_manager.get_path("data/train")

# Dapatkan informasi sistem
system_info = env_manager.get_system_info()
print(f"Python version: {system_info['python_version']}")
print(f"CUDA available: {system_info['cuda_available']}")
```

Gunakan contoh-contoh ini sebagai referensi untuk mengimplementasikan modul utils yang diperbarui dalam proyek SmartCash Anda. Setiap komponen telah dirancang untuk saling bekerja sama dengan baik sambil tetap menjaga fleksibilitas.
