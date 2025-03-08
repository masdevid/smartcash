# Dokumentasi Visualisasi Hasil Penelitian

Modul `research.py` dalam paket visualisasi telah diperbarui untuk memberikan visualisasi yang lebih komprehensif untuk hasil penelitian dengan fokus pada perbandingan model dan skenario eksperimen.

## Ringkasan Fitur

Kelas `ResearchVisualizer` memiliki fitur-fitur berikut:

1. **Visualisasi Perbandingan Model**
   - Perbandingan metrik (accuracy, precision, recall, F1, mAP)
   - Analisis trade-off kecepatan vs akurasi
   - Rekomendasi model terbaik dengan analisis kontekstual

2. **Visualisasi Skenario Penelitian**
   - Perbandingan berbagai skenario penelitian
   - Analisis berdasarkan backbone dan kondisi testing
   - Visualisasi dengan grouping untuk analisis lebih dalam

3. **Visualisasi Inferensi**
   - Perbandingan waktu inferensi dan FPS
   - Trade-off kecepatan vs ukuran model

4. **Styling Otomatis dan Penekanan**
   - Highlighting nilai terbaik dalam tabel
   - Visualisasi color-coded untuk identifikasi cepat
   - Alert dan annotations untuk informasi penting

## Contoh Penggunaan

### 1. Visualisasi Perbandingan Model

```python
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi visualizer
visualizer = ResearchVisualizer(output_dir="results/research")

# Visualisasi perbandingan model
result = visualizer.visualize_experiment_comparison(
    results_df=model_comparison_df,
    title="Perbandingan Backbone Deteksi Mata Uang",
    filename="model_comparison.png",
    metric_cols=['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP'],
    time_col='Waktu Inferensi (ms)',
    highlight_best=True
)

# Akses hasil
analysis = result['analysis']
best_model = analysis['best_model']['name']
recommendation = analysis['recommendation']
styled_df = result['styled_df']

print(f"Model terbaik: {best_model}")
print(f"Rekomendasi: {recommendation}")
display(styled_df)  # Untuk Jupyter/Colab
```

### 2. Visualisasi Skenario Penelitian

```python
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi visualizer
visualizer = ResearchVisualizer(output_dir="results/research")

# Visualisasi skenario penelitian
result = visualizer.visualize_scenario_comparison(
    results_df=scenario_df,
    title="Perbandingan Skenario Penelitian",
    filename="scenario_comparison.png",
    backbone_col='Backbone',
    condition_col='Kondisi',
    metric_cols=['Akurasi', 'Precision', 'F1-Score'],
    time_col='Waktu Inferensi (ms)'
)

# Akses hasil
analysis = result['analysis']
best_scenario = analysis['best_scenario']['name']
backbone_analysis = analysis['backbone_analysis']
condition_analysis = analysis['condition_analysis']

print(f"Skenario terbaik: {best_scenario}")
print(f"Analisis backbone: {backbone_analysis}")
print(f"Analisis kondisi: {condition_analysis}")
```

## Format Input Data

### DataFrame untuk Perbandingan Model

```python
model_comparison_df = pd.DataFrame({
    'Model': ['EfficientNet-B4', 'CSPDarknet', 'MobileNet'],
    'Akurasi': [95.2, 93.8, 91.5],
    'Precision': [94.8, 92.7, 90.2],
    'Recall': [95.5, 93.9, 92.1],
    'F1-Score': [95.1, 93.3, 91.1],
    'mAP': [92.8, 91.2, 89.5],
    'Waktu Inferensi (ms)': [45.2, 38.7, 22.3],
    'Ukuran Model (MB)': [25.6, 32.1, 12.8]
})
```

### DataFrame untuk Skenario Penelitian

```python
scenario_df = pd.DataFrame({
    'Skenario': ['S1', 'S2', 'S3', 'S4'],
    'Backbone': ['EfficientNet-B4', 'EfficientNet-B4', 'CSPDarknet', 'CSPDarknet'],
    'Kondisi': ['Pencahayaan Normal', 'Pencahayaan Rendah', 'Pencahayaan Normal', 'Pencahayaan Rendah'],
    'Akurasi': [95.2, 91.8, 93.8, 90.5],
    'Precision': [94.8, 90.7, 92.7, 89.2],
    'Recall': [95.5, 92.9, 93.9, 91.1],
    'F1-Score': [95.1, 91.8, 93.3, 90.1],
    'Waktu Inferensi (ms)': [45.2, 46.1, 38.7, 39.2]
})
```

## Hasil Keluaran

Output dari fungsi visualisasi adalah dictionary dengan struktur:

```python
{
    'fig': matplotlib_figure,       # Figure matplotlib
    'analysis': {                   # Hasil analisis
        'best_model': {             # Informasi model terbaik
            'name': 'EfficientNet-B4',
            'metrics': {...}
        },
        'fastest_model': {          # Informasi model tercepat
            'name': 'MobileNet',
            'inference_time': 22.3
        },
        'recommendation': 'Berdasarkan analisis...',  # Teks rekomendasi
        'metrics': {                # Rincian metrik
            'accuracy_range': [91.5, 95.2],
            'time_range': [22.3, 45.2]
        }
    },
    'styled_df': styled_dataframe,  # DataFrame dengan styling
    'plots': {                      # Dictionary grafik tambahan
        'metrics_plot': fig1,
        'tradeoff_plot': fig2
    }
}
```

## Detail Implementasi Utama

### Algoritma Rekomendasi Model

Rekomendasi model menggunakan pendekatan scoring terbobot:

1. Score akurasi (70% bobot):
   - Hitung persentase dari maksimum untuk setiap metrik akurasi
   - Rata-ratakan semua metrik akurasi untuk mendapatkan skor akurasi

2. Score kecepatan (30% bobot):
   - Normalize waktu inferensi terbaik / waktu model ini

3. Final score = (0.7 * accuracy_score) + (0.3 * speed_score)

### Analisis Trade-off

Analisis trade-off memplot metrik akurasi (y-axis) vs waktu inferensi (x-axis) untuk memvisualisasikan:

1. Kuadran optimal (akurasi tinggi, waktu rendah)
2. Kuadran kecepatan (akurasi rendah, waktu rendah)
3. Kuadran akurasi (akurasi tinggi, waktu tinggi)
4. Kuadran sub-optimal (akurasi rendah, waktu tinggi)

### Styling Tabel

Styling tabel menggunakan pandas Styler untuk:

1. Highlight nilai terbaik pada setiap kolom metrik dengan warna hijau
2. Highlight nilai tercepat pada kolom waktu inferensi
3. Format numerik dengan 2 desimal dan unit yang sesuai
4. Striping baris untuk memudahkan pembacaan

## Kustomisasi Visualisasi

### Parameter Kustom

| Parameter | Deskripsi | Default |
|-----------|-----------|---------|
| `metric_cols` | Kolom metrik yang akan divisualisasikan | `['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP']` |
| `time_col` | Kolom waktu inferensi | `'Waktu Inferensi (ms)'` |
| `highlight_best` | Highlight nilai terbaik | `True` |
| `figsize` | Ukuran figure | `(12, 10)` |
| `cmap` | Colormap untuk plot | `'Blues'` |

### Menambahkan Plot Kustom

```python
def visualize_with_custom_plot(results_df, custom_plot_func):
    """
    Visualisasi dengan plot kustom tambahan.
    
    Args:
        results_df: DataFrame hasil
        custom_plot_func: Fungsi plot kustom, menerima (results_df, ax)
    """
    # Buat visualisasi standar
    result = visualizer.visualize_experiment_comparison(results_df, ...)
    
    # Tambahkan plot kustom
    fig, ax = plt.subplots(figsize=(10, 6))
    custom_plot_func(results_df, ax)
    plt.tight_layout()
    
    # Tambahkan ke hasil
    result['plots']['custom_plot'] = fig
    
    return result
```

## Integrasi dengan Komponen Lain

### Integrasi dengan ExperimentTracker

```python
from smartcash.utils.experiment_tracker import ExperimentTracker
from smartcash.utils.visualization import ResearchVisualizer

# Inisialisasi tracker dan visualizer
tracker = ExperimentTracker(experiment_name="backbone_comparison")
visualizer = ResearchVisualizer(output_dir="results/research")

# Load data eksperimen
experiments = tracker.list_experiments()
experiment_data = {}

for exp_name in experiments:
    exp_tracker = ExperimentTracker(experiment_name=exp_name)
    exp_tracker.load_metrics()
    experiment_data[exp_name] = exp_tracker.get_metrics()

# Konversi ke DataFrame
exp_df = pd.DataFrame([
    {
        'Model': exp_name,
        'Akurasi': data.get('accuracy', 0) * 100,
        'Precision': data.get('precision', 0) * 100,
        'Recall': data.get('recall', 0) * 100,
        'F1-Score': data.get('f1', 0) * 100,
        'mAP': data.get('mAP', 0) * 100,
        'Waktu Inferensi (ms)': data.get('inference_time', 0)
    }
    for exp_name, data in experiment_data.items()
])

# Visualisasi
result = visualizer.visualize_experiment_comparison(
    results_df=exp_df,
    title="Perbandingan Eksperimen",
    filename="experiment_comparison.png"
)
```

### Integrasi dengan DetectionVisualizer

```python
from smartcash.utils.visualization import ResearchVisualizer, DetectionVisualizer

# Visualisasi hasil penelitian
research_vis = ResearchVisualizer(output_dir="results/research")
detection_vis = DetectionVisualizer(output_dir="results/detections")

# Visualisasi skenario
scenario_result = research_vis.visualize_scenario_comparison(
    results_df=scenario_df,
    title="Perbandingan Skenario",
    filename="scenario_comparison.png"
)

# Visualisasi deteksi dari model terbaik
best_scenario = scenario_result['analysis']['best_scenario']['name']
best_detections = scenario_detections[best_scenario]

# Visualisasi deteksi dari model terbaik
detection_grid = detection_vis.visualize_detections_grid(
    images=test_images,
    detections_list=best_detections,
    title=f"Deteksi Model Terbaik: {best_scenario}",
    filename="best_model_detections.jpg"
)
```

## Praktik Terbaik Visualisasi Penelitian

1. **Konsistensi**: Gunakan format data yang konsisten untuk perbandingan
2. **Konteks**: Sertakan konteks penelitian dalam visualisasi (tujuan, metode)
3. **Kejelasan**: Pastikan anotasi dan label cukup jelas untuk pemahaman
4. **Comparability**: Gunakan skala dan format yang sama saat membandingkan
5. **Objektivitas**: Tampilkan data secara objektif tanpa bias

## Troubleshooting

### Issue: Metrik tidak muncul dalam visualisasi

Solusi:
- Periksa nama kolom dalam DataFrame, pastikan sesuai dengan `metric_cols`
- Periksa tipe data kolom, pastikan numerik
- Periksa nilai NaN dalam DataFrame

### Issue: Plot trade-off tidak informatif

Solusi:
- Atur batas axis dengan parameter `tradeoff_xlim` dan `tradeoff_ylim`
- Pastikan ada variasi yang cukup dalam data (beberapa model dengan kecepatan/akurasi berbeda)

### Issue: Visualisasi gagal pada dataset besar

Solusi:
- Gunakan subset data dengan parameter `max_models`
- Kurangi ukuran figsize untuk mengurangi memory usage

## Kesimpulan

Modul visualisasi penelitian ini dirancang untuk membuat analisis model dan skenario penelitian lebih mudah dan informatif. Gunakan panduan ini untuk memanfaatkan fitur-fitur visualisasi yang kaya dan melakukan kustomisasi sesuai kebutuhan Anda.
