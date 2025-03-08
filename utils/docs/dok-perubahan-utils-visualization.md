# Dokumentasi Perubahan `utils/visualization` SmartCash

## Ringkasan Perubahan

Modul `utils/visualization` telah direstrukturisasi untuk meningkatkan modularitas, pemeliharaan, dan mempermudah pengembangan di masa depan. Perubahan utama meliputi pemecahan file besar menjadi modul yang lebih kecil dengan fokus tertentu, penambahan paket `analysis`, dan perbaikan struktur kode secara keseluruhan.

## Struktur Baru

```
smartcash/utils/visualization/
├── __init__.py                  # Ekspor komponen-komponen utama
├── base.py                      # Kelas dasar untuk visualisasi
├── detection.py                 # Visualisasi deteksi objek
├── metrics.py                   # Visualisasi metrik evaluasi
├── research.py                  # Modul utama untuk visualisasi hasil penelitian
├── experiment_visualizer.py     # Visualisasi eksperimen
├── scenario_visualizer.py       # Visualisasi skenario penelitian
├── research_base.py             # Fungsionalitas dasar untuk visualisasi penelitian
├── research_utils.py            # Fungsi utilitas untuk visualisasi penelitian
├── analysis/                    # Paket baru untuk analisis data penelitian
│   ├── __init__.py              # Ekspor kelas analisis
│   ├── experiment_analyzer.py   # Analisis hasil eksperimen
│   └── scenario_analyzer.py     # Analisis hasil skenario penelitian
```

## Komponen Utama

### 1. Base Components

- **VisualizationHelper** (`base.py`): Kelas utilitas dasar dengan fungsi-fungsi umum untuk styling plot, penyimpanan gambar, dan lainnya.

### 2. Visualisasi Deteksi

- **DetectionVisualizer** (`detection.py`): Visualisasi hasil deteksi objek dengan bounding box, label, dan informasi tambahan.

### 3. Visualisasi Metrik

- **MetricsVisualizer** (`metrics.py`): Visualisasi berbagai metrik evaluasi seperti confusion matrix, training curves, dll.

### 4. Visualisasi Penelitian

- **ResearchVisualizer** (`research.py`): Kelas utama yang mengintegrasikan komponen visualisasi eksperimen dan skenario penelitian.
- **ExperimentVisualizer** (`experiment_visualizer.py`): Visualisasi khusus untuk hasil eksperimen model.
- **ScenarioVisualizer** (`scenario_visualizer.py`): Visualisasi khusus untuk skenario penelitian.

### 5. Analisis

- **ExperimentAnalyzer** (`analysis/experiment_analyzer.py`): Analisis mendalam hasil eksperimen dan rekomendasi model terbaik.
- **ScenarioAnalyzer** (`analysis/scenario_analyzer.py`): Analisis dan rekomendasi berdasarkan skenario penelitian.

## Petunjuk Penggunaan

### Visualisasi Deteksi Objek

```python
from smartcash.utils.visualization import DetectionVisualizer, visualize_detection

# Metode 1: Menggunakan kelas DetectionVisualizer
visualizer = DetectionVisualizer(output_dir="results/detections")
visualized_img = visualizer.visualize_detection(
    image=img,
    detections=detections,
    filename="hasil_deteksi.jpg",
    conf_threshold=0.5,
    show_labels=True,
    show_total=True
)

# Metode 2: Menggunakan fungsi helper sederhana
visualized_img = visualize_detection(
    image=img,
    detections=detections,
    output_path="results/detections/hasil_deteksi.jpg"
)
```

### Visualisasi Metrik Evaluasi

```python
from smartcash.utils.visualization import MetricsVisualizer, plot_confusion_matrix

# Metode 1: Menggunakan kelas MetricsVisualizer
visualizer = MetricsVisualizer(output_dir="results/metrics")

# Visualisasi confusion matrix
fig = visualizer.plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=class_names,
    title="Confusion Matrix Model A",
    filename="confusion_matrix.png"
)

# Visualisasi metrik training
fig = visualizer.plot_training_metrics(
    metrics=training_history,
    title="Training Metrics",
    filename="training_metrics.png"
)

# Metode 2: Menggunakan fungsi helper
fig = plot_confusion_matrix(
    cm=confusion_matrix,
    class_names=class_names,
    output_path="results/confusion_matrix.png"
)
```

### Visualisasi Hasil Penelitian

```python
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi visualizer
visualizer = ResearchVisualizer(output_dir="results/research")

# Visualisasi perbandingan eksperimen
result = visualizer.visualize_experiment_comparison(
    results_df=experiment_data,
    title="Perbandingan Model",
    filename="perbandingan_model.png",
    highlight_best=True
)

# Visualisasi perbandingan skenario
result = visualizer.visualize_scenario_comparison(
    results_df=scenario_data,
    title="Perbandingan Skenario Penelitian",
    filename="perbandingan_skenario.png"
)

# Akses hasil analisis
analysis = result['analysis']
recommendation = analysis['recommendation']
print(f"Rekomendasi: {recommendation}")

# Akses styled DataFrame untuk ditampilkan
styled_df = result['styled_df']
display(styled_df)  # Untuk notebook
```

### Menggunakan Komponen Analisis Secara Terpisah

```python
from smartcash.utils.visualization.analysis import ExperimentAnalyzer, ScenarioAnalyzer

# Analisis eksperimen
experiment_analyzer = ExperimentAnalyzer()
analysis = experiment_analyzer.analyze_experiment_results(
    df=experiment_data,
    metric_cols=['Akurasi', 'Precision', 'Recall', 'F1-Score'],
    time_col='Waktu Inferensi (ms)'
)
print(f"Model terbaik: {analysis['best_model']['name']}")
print(f"Rekomendasi: {analysis['recommendation']}")

# Analisis skenario
scenario_analyzer = ScenarioAnalyzer()
analysis = scenario_analyzer.analyze_scenario_results(
    df=scenario_data,
    backbone_col='Backbone',
    condition_col='Kondisi'
)
print(f"Skenario terbaik: {analysis['best_scenario']['name']}")
print(f"Rekomendasi: {analysis['recommendation']}")
```

### Utilitas Penelitian

```python
from smartcash.utils.visualization.research_utils import (
    clean_dataframe, format_metric_name, create_benchmark_table
)

# Bersihkan DataFrame
clean_df = clean_dataframe(raw_df)

# Format nama metrik untuk tampilan yang lebih baik
formatted_name = format_metric_name('f1_score')  # Hasil: 'F1-Score'

# Buat tabel benchmark
benchmark_table = create_benchmark_table(
    metrics=metrics_dict,
    models=model_names,
    metric_names=['Precision', 'Recall', 'F1-Score']
)
```

## Keuntungan Restrukturisasi

1. **Modularitas**: Setiap komponen memiliki tanggung jawab yang jelas dan terfokus
2. **Pemeliharaan**: Lebih mudah menemukan dan memperbaiki bug dalam file yang lebih kecil
3. **Perluasan**: Lebih mudah menambahkan fitur atau visualisasi baru tanpa mengubah kode yang ada
4. **Testabilitas**: Komponen yang lebih kecil dan terfokus lebih mudah diuji
5. **Reusabilitas**: Komponen-komponen dapat digunakan kembali di berbagai bagian aplikasi

## Tips Penggunaan

1. **Gunakan helper functions** untuk visualisasi cepat dalam notebook atau skrip singkat
2. **Gunakan kelas lengkap** untuk kontrol lebih lanjut dan akses ke fitur-fitur tambahan
3. **Simpan hasil visualisasi** dengan parameter `filename` untuk referensi
4. **Akses hasil analisis** dari return value fungsi visualisasi
5. **Gunakan paket `analysis`** secara langsung untuk analisis tanpa visualisasi

## Konfigurasi Output

Secara default, semua visualizer akan menyimpan hasil ke direktori berikut:

- **DetectionVisualizer**: `results/detections/`
- **MetricsVisualizer**: `results/metrics/`
- **ResearchVisualizer**: `results/research/`
  - **ExperimentVisualizer**: `results/research/experiments/`
  - **ScenarioVisualizer**: `results/research/scenarios/`

Anda dapat mengubah direktori output dengan parameter `output_dir` saat inisialisasi visualizer.

## Pengaturan Style Plot

Untuk mengatur style plot secara global:

```python
from smartcash.utils.visualization import setup_visualization

# Setup style
setup_visualization()
```

Perintah ini akan mengatur style plot yang konsisten untuk semua visualisasi.