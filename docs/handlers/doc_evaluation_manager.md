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
│   ├── dataset_adapter.py               # Adapter untuk DatasetManager
│   ├── checkpoint_manager_adapter.py    # Adapter untuk CheckpointManager
│   ├── visualization_adapter.py         # Adapter untuk visualisasi
│   └── adapters_factory.py              # Factory untuk adapter
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

### 5. Integrasi dengan Observer Pattern

- Implementasi langsung dengan `utils/observer` untuk memantau proses
- Notifikasi event untuk setiap tahap evaluasi
- Observer untuk progress tracking dan metrics monitoring
- Integrasi dengan logger untuk informatif logs

## Metode Utama di EvaluationManager

### evaluate_model

Mengevaluasi satu model tertentu menggunakan dataset tertentu. Jika tidak diberikan model_path, akan menggunakan checkpoint terbaik yang tersedia. Jika tidak diberikan dataset_path, akan menggunakan test_dir dari konfigurasi.

### evaluate_batch

Mengevaluasi beberapa model secara paralel dengan dataset yang sama. Memberikan perbandingan komprehensif antar model.

### evaluate_research_scenarios

Mengevaluasi berbagai skenario penelitian, seperti model dengan backbone berbeda (EfficientNet vs CSPDarknet) pada berbagai kondisi pengujian (variasi posisi, pencahayaan dll).

### generate_report

Membuat laporan hasil evaluasi dalam berbagai format. Format yang didukung: JSON, CSV, Markdown, dan HTML.

## Format Hasil

### Hasil Evaluasi Model Tunggal

```python
{
    'metrics': {
        'mAP': 0.92,                      # Mean Average Precision
        'precision': 0.88,                # Precision
        'recall': 0.89,                   # Recall
        'f1': 0.885,                      # F1 Score
        'inference_time': 0.023,          # Waktu inferensi rata-rata
        'fps': 43.5,                      # Frame per detik
        'class_metrics': {...}            # Metrik per kelas
    },
    'total_execution_time': 120.5,        # Waktu eksekusi total (detik)
    'visualization_paths': {
        'confusion_matrix': '/path/to/confusion_matrix.png',
        'pr_curve': '/path/to/pr_curve.png'
    },
    'report_path': '/path/to/report.json' # Path laporan hasil
}
```

### Hasil Evaluasi Batch

```python
{
    'model_results': {                    # Hasil untuk setiap model
        'model1': {...},                  # Hasil model 1
        'model2': {...}                   # Hasil model 2
    },
    'summary': {                          # Ringkasan perbandingan
        'best_model': 'model1',           # Model terbaik
        'best_map': 0.92,                 # mAP terbaik
        'average_map': 0.89,              # Rata-rata mAP
        'metrics_table': {...}            # Tabel perbandingan
    },
    'plots': {                            # Visualisasi perbandingan
        'map_comparison': '/path/to/map_comparison.png',
        'inference_time': '/path/to/inference_time.png'
    },
    'total_execution_time': 300.8         # Waktu eksekusi total (detik)
}
```

### Hasil Evaluasi Skenario Penelitian

```python
{
    'scenario_results': {                 # Hasil untuk setiap skenario
        'scenario1': {...},               # Hasil skenario 1
        'scenario2': {...}                # Hasil skenario 2
    },
    'summary': {                          # Ringkasan perbandingan
        'best_scenario': 'scenario1',     # Skenario terbaik
        'best_map': 0.94,                 # mAP terbaik
        'backbone_comparison': {          # Perbandingan backbone
            'efficientnet': {'mAP': 0.93, 'F1': 0.91},
            'cspdarknet': {'mAP': 0.89, 'F1': 0.88}
        },
        'condition_comparison': {         # Perbandingan kondisi
            'Posisi Bervariasi': {'mAP': 0.88, 'F1': 0.86},
            'Pencahayaan Bervariasi': {'mAP': 0.85, 'F1': 0.84}
        }
    },
    'plots': {                            # Visualisasi perbandingan
        'backbone_comparison': '/path/to/backbone_comparison.png',
        'condition_comparison': '/path/to/condition_comparison.png'
    },
    'total_execution_time': 600.5         # Waktu eksekusi total (detik)
}
```

## Event Utama

- `evaluation.manager.start`: Evaluasi model dimulai
- `evaluation.manager.complete`: Evaluasi model selesai
- `evaluation.batch.start`: Evaluasi batch dimulai
- `evaluation.batch.complete`: Evaluasi batch selesai
- `evaluation.research.start`: Evaluasi skenario penelitian dimulai
- `evaluation.research.complete`: Evaluasi skenario penelitian selesai
- `evaluation.pipeline.start`: Pipeline evaluasi dimulai
- `evaluation.pipeline.complete`: Pipeline evaluasi selesai
- `evaluation.batch_start`: Batch evaluasi dimulai
- `evaluation.batch_complete`: Batch evaluasi selesai
- `evaluation.report.start`: Pembuatan laporan dimulai
- `evaluation.report.complete`: Pembuatan laporan selesai

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