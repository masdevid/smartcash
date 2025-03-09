# Rencana Restrukturisasi Dataset Manager SmartCash

## Tujuan

Restrukturisasi `dataset_manager.py` untuk mengikuti prinsip Single Responsibility dengan membuat komponen lebih atomic, modular, dan dapat diuji. Refaktorisasi akan menerapkan pola desain modern seperti Factory, Adapter, dan Facade.

## ⚠️ Peringatan Duplikasi

**HINDARI menduplikasi implementasi yang sudah ada di folder `utils`!** Berdasarkan `UTILS_DOCS.md`, beberapa komponen yang sudah direfaktor di utils:

- `EnhancedDatasetValidator` - Gunakan adapter pattern untuk integrasi
- `AugmentationManager` - Gunakan eksisting daripada reimplementasi
- `MetricsCalculator` - Gunakan untuk perhitungan metrik
- `LayerConfigManager` - Gunakan untuk konfigurasi layer

## Struktur Folder dan File

```
smartcash/handlers/dataset/
├── __init__.py                          # Export komponen utama
├── dataset_manager.py                   # Entry point minimal

├── facades/                             # Facades terpisah untuk fungsi-fungsi spesifik
│   ├── dataset_base_facade.py           # Kelas dasar untuk semua facade
│   ├── data_loading_facade.py           # Operasi loading dan dataloader
│   ├── data_processing_facade.py        # Validasi, augmentasi, balancing
│   ├── data_operations_facade.py        # Split, merge, cleanup
│   ├── dataset_explorer_facade.py       # Facade untuk semua explorer
│   ├── visualization_facade.py          # Semua visualisasi dataset
│   └── pipeline_facade.py               # Menggabungkan semua facade (untuk dataset_manager)

├── multilayer/                           # Komponen dataset multilayer
│   ├── multilayer_dataset_base.py        # Kelas dasar
│   ├── multilayer_dataset.py             # Dataset multilayer
│   └── multilayer_label_handler.py       # Handler label

├── core/                                 # Komponen inti
│   ├── dataset_loader.py                 # Loader dataset spesifik
│   ├── dataset_downloader.py             # Downloader dataset
│   ├── dataset_transformer.py            # Transformasi data
│   ├── dataset_validator.py              # Validasi dataset
│   ├── dataset_augmentor.py              # Augmentasi dataset
│   └── dataset_balancer.py               # Balancer dataset

├── operations/                           # Operasi pada dataset
│   ├── dataset_split_operation.py        # Pemecahan dataset
│   ├── dataset_merge_operation.py        # Penggabungan dataset
│   └── dataset_reporting_operation.py    # Pelaporan dataset

├── explorers/                           
│   ├── base_explorer.py                  # Kelas dasar untuk semua explorer
│   ├── validation_explorer.py            # Validasi integritas
│   ├── class_explorer.py                 # Distribusi kelas
│   ├── layer_explorer.py                 # Distribusi layer
│   ├── image_size_explorer.py            # Ukuran gambar
│   └── bbox_explorer.py                  # Bounding box

├── integration/                          # Adapter untuk integrasi
│   ├── validator_adapter.py              # Adapter untuk EnhancedDatasetValidator
│   └── colab_drive_adapter.py            # Adapter untuk Google Drive di Colab

└── visualizations/                       # Visualisasi dataset
    ├── visualization_base.py             # Kelas dasar untuk semua visualisasi
    ├── heatmap/
    │   ├── spatial_density_heatmap.py     # Heatmap kepadatan spasial objek
    │   ├── class_density_heatmap.py       # Heatmap kepadatan per kelas
    │   ├── size_distribution_heatmap.py   # Heatmap distribusi ukuran objek
    │   ├── confidence_heatmap.py          # Heatmap tingkat confidence deteksi
    │   └── layer_overlap_heatmap.py       # Heatmap tumpang tindih antar layer
    ├── charts/
    │   ├── distribution_chart.py          # Chart distribusi (kelas, layer)
    │   ├── metrics_chart.py               # Chart metrik dataset
    │   ├── comparison_chart.py            # Chart perbandingan antar split
    │   └── evolution_chart.py             # Chart evolusi dataset (augmentasi)
    └── sample/
        ├── sample_grid_visualizer.py      # Grid sampel gambar
        ├── annotation_visualizer.py       # Visualisasi anotasi gambar
        ├── augmentation_visualizer.py     # Visualisasi hasil augmentasi
        └── error_samples_visualizer.py    # Visualisasi sampel dengan error
```

## Pola Desain yang Digunakan

1. **Facade Pattern**: Menyediakan antarmuka sederhana untuk subsistem yang kompleks
   - Pemecahan facade berdasarkan kategori operasi
   - Komposisi facade untuk membentuk `dataset_manager.py`

2. **Adapter Pattern**: Untuk integrasi dengan komponen dari `utils`
   - `validator_adapter.py` untuk integrasi dengan `EnhancedDatasetValidator`
   - `colab_drive_adapter.py` untuk integrasi dengan Google Drive di Colab

3. **Factory Pattern**: Untuk pembuatan objek dengan konfigurasi yang tepat
   - Pembuatan komponen dengan lazy initialization 

4. **Strategy Pattern**: Untuk operasi dengan algoritma yang dapat dipertukarkan
   - Diversifikasi strategi augmentasi dan transformasi

5. **Builder Pattern**: Untuk pembuatan objek kompleks secara bertahap
   - Pembuatan visualisasi kompleks

## Langkah Implementasi

1. Implementasi facade dasar dan kelas dasar
2. Implementasi komponen inti (core)
3. Implementasi facade yang mengakses komponen inti
4. Implementasi komponen spesifik (explorers, visualizations)
5. Implementasi pipeline facade dan dataset_manager

## Kompatibilitas dengan Google Colab

- **Integration Adapter**: Komponen `colab_drive_adapter.py` sebagai adapter untuk Google Drive
- **Deteksi Environment**: Deteksi otomatis penggunaan di Google Colab
- **Mount Drive**: Helper untuk mounting Google Drive
- **Path Mapping**: Konversi otomatis path lokal ke path Google Drive
- **Symlink Support**: Dukungan untuk symlink di Google Drive
- **Persistensi**: Strategi untuk menyimpan hasil operasi di Google Drive

# Implementasikan file berikut:

1. smartcash/handlers/dataset/visualizations/heatmap/confidence_heatmap.py
   - Visualisasi tingkat confidence deteksi
   - Fitur penting: heatmap tingkat confidence antar kelas/layer
   - Highlight pada prediksi dengan false positive/negative tinggi

2. smartcash/handlers/dataset/visualizations/heatmap/layer_overlap_heatmap.py
   - Visualisasi tumpang tindih antar layer
   - Metrics IoU antar layer
   - Analisis area yang sering overlap

3. smartcash/handlers/dataset/visualizations/charts/distribution_chart.py
   - Chart yang lebih detail untuk distribusi kelas dan layer
   - Opsi tambahan: pie chart, bar chart, histogram

4. smartcash/handlers/dataset/visualizations/charts/metrics_chart.py
   - Visualisasi metrik evaluasi dataset
   - Support untuk berbagai metrik: distribution, confusion matrix

5. smartcash/handlers/dataset/visualizations/charts/comparison_chart.py
   - Chart untuk membandingkan distribusi antar split
   - Opsi tambahan: line chart untuk progress training

6. smartcash/handlers/dataset/visualizations/charts/evolution_chart.py
   - Chart yang menunjukkan evolusi dataset setelah augmentasi
   - Analisis perubahan distribusi kelas

7. smartcash/handlers/dataset/visualizations/sample/augmentation_visualizer.py
   - Visualisasi hasil augmentasi untuk satu gambar
   - Support berbagai mode augmentasi: lighting, geometric, combined

8. smartcash/handlers/dataset/visualizations/sample/error_samples_visualizer.py
   - Visualisasi sampel yang mengalami error deteksi
   - Highlight false positive/negative
   - Interface untuk menandai sampel problematik
