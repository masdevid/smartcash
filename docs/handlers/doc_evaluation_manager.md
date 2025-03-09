# Ringkasan EvaluationManager SmartCash

## Deskripsi

`EvaluationManager` adalah komponen pusat untuk evaluasi model deteksi mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi evaluasi model. Implementasi telah dioptimasi dengan pendekatan pipeline yang modular dan mendukung berbagai skenario pengujian.

## Struktur dan Komponen

```
smartcash/handlers/evaluation/
├── __init__.py                          # Export komponen utama
├── evaluation_manager.py                # Entry point sebagai facade
├── core/                                # Komponen inti evaluasi
│   ├── evaluation_component.py          # Komponen dasar
│   ├── model_evaluator.py               # Evaluasi model
│   └── report_generator.py              # Generator laporan
├── pipeline/                            # Pipeline dan workflow
│   ├── base_pipeline.py                 # Pipeline dasar
│   ├── evaluation_pipeline.py           # Pipeline evaluasi standar
│   ├── batch_evaluation_pipeline.py     # Pipeline evaluasi batch
│   └── research_pipeline.py             # Pipeline penelitian
├── integration/                         # Adapter untuk integrasi
│   ├── metrics_adapter.py               # Adapter untuk MetricsCalculator
│   ├── model_manager_adapter.py         # Adapter untuk ModelManager
│   ├── dataset_adapter.py               # Adapter untuk DatasetManager
│   ├── checkpoint_manager_adapter.py    # Adapter untuk CheckpointManager
│   ├── visualization_adapter.py         # Adapter untuk visualisasi
│   └── adapters_factory.py              # Factory untuk adapter
└── observers/                           # Observer pattern
    ├── base_observer.py                 # Observer dasar
    ├── progress_observer.py             # Monitoring progres
    └── metrics_observer.py              # Monitoring metrik
```

`EvaluationManager` menggabungkan beberapa pipeline terspesialisasi menjadi satu antarmuka terpadu:

- **EvaluationPipeline**: Evaluasi model tunggal
- **BatchEvaluationPipeline**: Evaluasi batch model secara paralel
- **ResearchPipeline**: Evaluasi skenario penelitian dan perbandingan model

## Fitur Utama

### 1. Evaluasi Model Tunggal

- Evaluasi model dengan berbagai metrik standar (mAP, F1, precision, recall)
- Pengukuran waktu inferensi dan performa
- Perhitungan metrik per kelas dan layer
- Validasi model dan dataset otomatis
- Dukungan untuk model multilayer dengan evaluasi detil per layer

### 2. Evaluasi Batch

- Evaluasi beberapa model secara paralel dengan dataset yang sama
- Perbandingan performa berbagai model
- Analisis model terbaik berdasarkan metrik yang dipilih
- Visualisasi perbandingan dengan berbagai plot
- Thread-safety untuk eksekusi paralel

### 3. Evaluasi Skenario Penelitian

- Evaluasi model dalam skenario penelitian yang berbeda
- Analisis performa model dengan backbone berbeda (EfficientNet vs CSPDarknet)
- Perbandingan ketahanan model pada berbagai kondisi (posisi, pencahayaan)
- Analisis statistik dengan perhitungan rata-rata dan standar deviasi
- Multiple runs untuk mengukur stabilitas hasil

### 4. Pembuatan Laporan

- Generasi laporan dalam berbagai format (JSON, CSV, Markdown, HTML)
- Visualisasi hasil dengan berbagai jenis plot
- Analisis komprehensif termasuk insight dan rekomendasi
- Dukungan untuk eksport dan sharing hasil
- Penyimpanan metrik untuk analisis jangka panjang

### 5. Integrasi dengan Komponen Lain

- Adapter pattern untuk integrasi dengan komponan lain
- Factory pattern untuk inisialisasi komponen
- Observer pattern untuk progress monitoring dan metrics tracking
- Integrasi dengan logger untuk informatif logs
- Pemanfaatan berbagai utilitas SmartCash

## Kelas Utama

### EvaluationManager

Manager utama evaluasi sebagai facade yang menyederhanakan antarmuka untuk evaluasi model dengan menggunakan berbagai adapter dan pipeline.

### Pipeline-Pipeline Evaluasi

#### EvaluationPipeline

Pipeline evaluasi dengan berbagai komponen yang dapat dikonfigurasi, menggabungkan beberapa komponen evaluasi menjadi satu alur kerja.

#### BatchEvaluationPipeline

Pipeline untuk evaluasi batch model yang dapat mengevaluasi beberapa model dengan dataset yang sama secara paralel.

#### ResearchPipeline

Pipeline untuk evaluasi skenario penelitian dan perbandingan model dalam konteks perbandingan skenario penelitian dengan visualisasi hasil.

### Core Components

#### ModelEvaluator

Komponen untuk evaluasi model dengan berbagai strategi yang melakukan proses evaluasi pada model dengan dataset yang diberikan.

#### ReportGenerator

Generator laporan hasil evaluasi model dalam berbagai format (JSON, CSV, Markdown, HTML).

## Metode Utama di EvaluationManager

### evaluate_model

Mengevaluasi satu model tertentu menggunakan dataset tertentu. Jika tidak diberikan model_path, akan menggunakan checkpoint terbaik yang tersedia. Jika tidak diberikan dataset_path, akan menggunakan test_dir dari konfigurasi.

### evaluate_batch

Mengevaluasi beberapa model secara paralel dengan dataset yang sama. Memberikan perbandingan komprehensif antar model.

### evaluate_research_scenarios

Mengevaluasi berbagai skenario penelitian, seperti model dengan backbone berbeda (EfficientNet vs CSPDarknet) pada berbagai kondisi pengujian (variasi posisi, pencahayaan dll).

### generate_report

Membuat laporan hasil evaluasi dalam berbagai format. Format yang didukung: JSON, CSV, Markdown, dan HTML.

### visualize_results

Membuat visualisasi dari hasil evaluasi, seperti perbandingan mAP, F1-score, waktu inferensi, dan metrik lainnya.

## Adapters

### MetricsAdapter

Adapter untuk MetricsCalculator dari utils.metrics yang menyediakan antarmuka yang konsisten untuk menghitung dan mengelola metrik evaluasi.

### ModelManagerAdapter

Adapter untuk ModelManager yang menyediakan antarmuka untuk loading dan persiapan model untuk evaluasi.

### DatasetAdapter

Adapter untuk DatasetManager yang menyediakan antarmuka untuk akses dataset dan pembuatan dataloader untuk evaluasi.

### CheckpointManagerAdapter

Adapter untuk CheckpointManager yang menyediakan antarmuka untuk pencarian dan validasi checkpoint model.

### VisualizationAdapter

Adapter untuk integrasi visualisasi evaluasi yang menghubungkan pipeline evaluasi dengan komponen visualisasi.

## Observer Pattern

### ProgressObserver

Observer untuk monitoring progres evaluasi yang menampilkan progress bar dan informasi runtime.

### MetricsObserver

Observer untuk monitoring dan pencatatan metrik evaluasi yang berguna untuk tracking eksperimen dan visualisasi hasil.

## Format Hasil

### Hasil Evaluasi Model Tunggal

Hasil evaluasi berisi informasi seperti metrik (mAP, precision, recall, F1), waktu inferensi, informasi model, dan dataset.

### Hasil Evaluasi Batch

Hasil evaluasi batch berisi informasi tentang evaluasi beberapa model, metrik perbandingan, dan visualisasi perbandingan performa.

### Hasil Evaluasi Skenario Penelitian

Hasil evaluasi skenario penelitian berisi informasi tentang berbagai skenario, perbandingan backbone, dan kondisi pengujian yang berbeda.

## Konfigurasi

EvaluationManager menggunakan bagian `evaluation` dari konfigurasi utama yang mencakup berbagai parameter seperti threshold deteksi, metrik evaluasi, dan konfigurasi visualisasi.

## Integrasi dengan Google Colab

EvaluationManager mendukung integrasi dengan Google Colab melalui parameter `colab_mode` untuk menyesuaikan tampilan di notebook dan integrasi dengan Google Drive.

## Kesimpulan

EvaluationManager SmartCash menawarkan:

1. **Fleksibilitas Evaluasi**: Mendukung evaluasi model tunggal, batch, dan skenario penelitian
2. **Modularitas**: Berbagai komponen dapat digunakan secara independen atau bersama-sama
3. **Optimasi Performa**: Paralelisasi, caching, dan optimasi memory
4. **Visualisasi Komprehensif**: Berbagai jenis plot dan format laporan
5. **Analisis Mendalam**: Perbandingan backbone, kondisi pengujian, dan performa model
6. **Integrasi Mulus**: Dengan komponen lain di SmartCash melalui adapter pattern
7. **Kemudahan Penggunaan**: Antarmuka facade yang sederhana namun powerful