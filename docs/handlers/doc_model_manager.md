# Dokumentasi Model Manager SmartCash

## Deskripsi

`ModelManager` adalah komponen pusat untuk pengelolaan model deteksi mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi model. Implementasi telah dioptimasi dengan pendekatan yang modular dan dapat dipelihara dengan mudah.

## Struktur dan Komponen

`ModelManager` mengadopsi struktur modular berikut:

```
smartcash/handlers/model/
├── __init__.py                     # Export komponen utama
├── model_manager.py                # Entry point minimal (facade)

├── core/                           # Komponen inti model
│   ├── model_component.py          # Kelas dasar komponen model
│   ├── model_factory.py            # Factory pembuatan model dan arsitektur
│   ├── backbone_factory.py         # Factory pembuatan backbone
│   ├── optimizer_factory.py        # Factory untuk optimizer dan scheduler
│   ├── model_trainer.py            # Komponen training model
│   ├── model_evaluator.py          # Komponen evaluasi model
│   └── model_predictor.py          # Komponen prediksi dengan model

├── experiments/                    # Eksperimen dan riset
│   ├── experiment_manager.py       # Manajer eksperimen
│   └── backbone_comparator.py      # Komponen khusus untuk perbandingan backbone

├── observers/                      # Observer untuk monitoring
│   ├── base_observer.py            # Observer dasar
│   ├── metrics_observer.py         # Monitoring metrik training
│   └── colab_observer.py           # Observer khusus Colab

├── integration/                    # Adapter untuk integrasi
│   ├── checkpoint_adapter.py       # Adapter untuk CheckpointManager
│   ├── metrics_adapter.py          # Adapter untuk MetricsCalculator
│   ├── environment_adapter.py      # Adapter untuk environment
│   ├── experiment_adapter.py       # Adapter untuk experiment tracking
│   └── exporter_adapter.py         # Adapter untuk export model

└── visualizations/                 # Visualisasi training dan evaluasi
    ├── metrics_visualizer.py       # Visualisasi metrik training
    └── comparison_visualizer.py    # Visualisasi perbandingan model
```

`ModelManager` menggabungkan beberapa komponen terspesialisasi menjadi satu antarmuka terpadu:

- **ModelFactory**: Pembuatan model dengan berbagai backbone
- **BackboneFactory**: Pembuatan backbone (EfficientNet, CSPDarknet)
- **OptimizerFactory**: Pembuatan optimizer dan scheduler
- **ModelTrainer**: Training model
- **ModelEvaluator**: Evaluasi model pada dataset
- **ModelPredictor**: Prediksi menggunakan model
- **ExperimentManager**: Eksperimen dan perbandingan model

## Fitur Utama

### 1. Pembuatan Model

- Dukungan untuk multiple backbone architectures (EfficientNet, CSPDarknet)
- Factory pattern untuk pembuatan model yang fleksibel
- Integrasi backbone ke dalam arsitektur YOLO secara transparan
- Loading model dari checkpoint
- Pembekuan dan unfreeze backbone untuk transfer learning
- Parameter injection melalui konfigurasi

### 2. Training Model

- Integrasi dengan TrainingPipeline dari utils
- Optimizer dan scheduler dengan konfigurasi yang fleksibel
- Early stopping dengan kriteria yang dapat dikonfigurasi
- Checkpoint otomatis dan save best model
- Observer pattern untuk monitoring training
- Dukungan khusus untuk Google Colab dengan visualisasi real-time
- Integrasi dengan experiment tracking untuk analisis eksperimen

### 3. Evaluasi Model

- Evaluasi model pada dataset test
- Perhitungan metrik evaluasi (mAP, precision, recall, F1)
- Dukungan untuk evaluasi pada layer spesifik
- Pengukuran waktu inferensi dan performa
- Visualisasi hasil evaluasi

### 4. Prediksi

- Prediksi pada gambar atau batch gambar
- Pre-processing dan post-processing otomatis
- Visualisasi hasil deteksi
- Prediksi pada video dengan visualisasi frame-by-frame
- Rescaling koordinat ke ukuran gambar asli

### 5. Eksperimen dan Perbandingan

- Perbandingan berbagai backbone dengan kondisi yang sama
- Perbandingan model dengan ukuran gambar berbeda
- Perbandingan strategi augmentasi
- Eksekusi eksperimen secara paralel untuk efisiensi
- Visualisasi perbandingan performa
- Tracking eksperimen dengan metrik-metrik penting

### 6. Export Model

- Export model ke format deployment (TorchScript, ONNX)
- Optimasi model untuk inferensi
- Dukungan untuk half precision
- Pengukuran performa model yang diexport
- Integrasi dengan Google Drive di Colab

## Kelas Utama

### ModelManager

Kelas utama yang berfungsi sebagai facade, menyembunyikan kompleksitas implementasi dan meningkatkan usability. Mengelola lazy-loading komponen dan menyediakan metode-metode utama untuk:
- Membuat dan memuat model
- Training model
- Evaluasi model
- Prediksi dengan model
- Perbandingan backbone
- Setup environment
- Export model
- Tracking eksperimen

### ModelFactory

Factory untuk pembuatan model dengan berbagai backbone. Bertanggung jawab untuk:
- Membuat model dengan backbone yang ditentukan
- Memuat model dari checkpoint
- Membekukan dan melepas pembekuan backbone untuk fine-tuning

### BackboneFactory

Factory untuk pembuatan backbone dengan berbagai arsitektur. Menyediakan fungsi untuk:
- Membuat backbone dengan tipe yang ditentukan (EfficientNet, CSPDarknet)
- Mendapatkan dimensi fitur output backbone

### ModelTrainer

Komponen untuk training model. Mengimplementasikan:
- Training model dengan dataset yang diberikan
- Integrasi dengan TrainingPipeline
- Setup callback dan observer untuk monitoring
- Tracking eksperimen

### ModelEvaluator

Komponen untuk evaluasi model. Menyediakan fungsi untuk:
- Evaluasi model pada dataset test
- Evaluasi model pada layer spesifik

### ModelPredictor

Komponen untuk prediksi menggunakan model. Bertanggung jawab untuk:
- Prediksi pada gambar
- Prediksi pada video
- Pre-processing dan post-processing gambar
- Visualisasi hasil deteksi

### ExperimentManager

Manager untuk eksperimen model, fokus pada perbandingan backbone. Menyediakan fungsi untuk:
- Perbandingan beberapa backbone dengan kondisi yang sama
- Running eksperimen secara paralel atau serial
- Visualisasi dan analisis hasil perbandingan

### BackboneComparator

Komponen khusus untuk perbandingan backbone dengan parameter yang berbeda. Memperluas fungsionalitas ExperimentManager dengan opsi perbandingan untuk:
- Ukuran gambar yang berbeda
- Strategi augmentasi yang berbeda

## Observers dan Monitoring

### BaseObserver

Kelas dasar untuk monitoring model dan training. Mendefinisikan interface untuk:
- Update berdasarkan events (training_start, training_end, epoch_start, epoch_end)
- Callback untuk events tersebut

### MetricsObserver

Observer untuk monitoring dan tracking metrik training. Bertanggung jawab untuk:
- Menyimpan history metrik training
- Visualisasi metrik training
- Penyimpanan metrik ke disk

### ColabObserver

Observer khusus untuk Google Colab. Menyediakan:
- Visualisasi real-time menggunakan ipywidgets
- Progress bar yang kompatibel dengan Colab
- Update grafik metrik secara dinamis

## Adapters untuk Integrasi

### CheckpointAdapter

Adapter untuk integrasi dengan CheckpointManager. Menyediakan fungsi untuk:
- Menyimpan checkpoint model
- Memuat checkpoint model
- Menemukan checkpoint terbaik

### MetricsAdapter

Adapter untuk integrasi dengan MetricsCalculator. Mengelola:
- Reset dan update metrik
- Perhitungan metrik final
- Pengukuran waktu inferensi

### EnvironmentAdapter

Adapter untuk integrasi dengan EnvironmentManager. Bertanggung jawab untuk:
- Deteksi environment (Colab)
- Mount Google Drive
- Setup project environment
- Penyesuaian path berdasarkan environment

### ExperimentAdapter

Adapter untuk integrasi dengan ExperimentTracker. Menyediakan fungsi untuk:
- Setting nama eksperimen
- Start dan end eksperimen
- Logging metrik
- Perbandingan eksperimen

### ExporterAdapter

Adapter untuk integrasi dengan ModelExporter. Bertanggung jawab untuk:
- Export model ke TorchScript
- Export model ke ONNX
- Salin model ke Google Drive

## Format Hasil

### Hasil Training

```python
{
    'epoch': 30,                      # Epoch terakhir
    'best_epoch': 25,                 # Epoch terbaik
    'best_val_loss': 0.125,           # Validation loss terbaik
    'early_stopped': True,            # Flag early stopping
    'metrics_history': { ... },       # History metrik
    'best_checkpoint_path': '...',    # Path checkpoint terbaik
    'last_checkpoint_path': '...',    # Path checkpoint terakhir
    'execution_time': 3600.5          # Waktu eksekusi (detik)
}
```

### Hasil Evaluasi

```python
{
    'mAP': 0.92,                      # Mean Average Precision
    'precision': 0.88,                # Precision
    'recall': 0.89,                   # Recall
    'f1': 0.885,                      # F1 Score
    'inference_time': 0.023,          # Waktu inferensi per gambar (detik)
    'class_metrics': { ... },         # Metrik per kelas
    'execution_time': 120.5,          # Waktu eksekusi evaluasi (detik)
    'num_test_batches': 100,          # Jumlah batch test
    'conf_threshold': 0.25,           # Threshold konfidiensi
    'iou_threshold': 0.45             # Threshold IoU
}
```

### Hasil Prediksi

```python
{
    'num_images': 5,                  # Jumlah gambar
    'detections': [ ... ],            # Hasil deteksi per gambar
    'visualization_paths': [ ... ],   # Path hasil visualisasi
    'execution_time': 0.85,           # Waktu eksekusi (detik)
    'fps': 5.88                       # Frame per detik
}
```

### Hasil Perbandingan Backbone

```python
{
    'experiment_name': '...',         # Nama eksperimen
    'experiment_dir': '...',          # Direktori output
    'num_backbones': 2,               # Jumlah backbone yang dibandingkan
    'execution_time': 7200.5,         # Waktu eksekusi total (detik)
    'backbones': ['efficientnet', 'cspdarknet'],  # Backbone yang dibandingkan
    'results': { ... },               # Hasil per backbone
    'summary': { ... },               # Ringkasan perbandingan
    'visualization_paths': { ... }    # Path visualisasi
}
```

## Konfigurasi

ModelManager menggunakan beberapa bagian dari konfigurasi utama:

1. **model**: Konfigurasi arsitektur model, parameter inferensi, dan deployment
2. **training**: Parameter training, optimizer, scheduler, dan augmentasi
3. **experiment**: Konfigurasi experiment tracking dan visualisasi
4. **inference**: Parameter deteksi dan visualisasi hasil

## Integrasi dengan Google Colab

ModelManager mendukung integrasi dengan Google Colab melalui:
- Deteksi otomatis environment Colab
- ColabObserver untuk visualisasi real-time
- Integrasi dengan Google Drive via EnvironmentAdapter
- Progress bar yang kompatibel dengan notebook

## Pola Desain yang Digunakan

1. **Facade Pattern**: ModelManager sebagai entry point
2. **Factory Pattern**: Pembuatan komponen model, backbone, dan optimizer
3. **Observer Pattern**: Monitoring training dan metrik
4. **Adapter Pattern**: Integrasi dengan komponen SmartCash lainnya
5. **Component Pattern**: ModelComponent sebagai kelas dasar
6. **Lazy-loading Pattern**: Loading komponen saat dibutuhkan

## Optimasi Performa

1. **GPU Acceleration**: Otomatisasi penggunaan GPU dan half precision
2. **Paralelisasi**: Eksperimen paralel dengan ThreadPoolExecutor
3. **Memory Management**: Lazy-loading dan torch.no_grad()
4. **Caching**: Checkpoint otomatis dan statistik performa
5. **Optimasi Model**: Export dengan optimasi untuk inferensi

## Kesimpulan

ModelManager adalah komponen pusat untuk pengelolaan model di SmartCash yang menyediakan:

1. **Antarmuka Terpadu**: Entry point untuk semua operasi model
2. **Modularitas**: Pemisahan komponen dengan tanggung jawab yang jelas
3. **Fleksibilitas**: Dukungan untuk berbagai backbone dan konfigurasi
4. **Eksperimen**: Kemampuan untuk membandingkan model dan parameter
5. **Optimasi**: Fokus pada performa dan efisiensi
6. **Integrasi**: Integrasi mulus dengan komponen lain di SmartCash
7. **Colab Support**: Dukungan khusus untuk Google Colab

ModelManager memfasilitasi pembuatan, pelatihan, evaluasi, dan deployment model deteksi mata uang Rupiah dengan antarmuka yang konsisten dan mudah digunakan, mendukung seluruh siklus hidup pengembangan model.