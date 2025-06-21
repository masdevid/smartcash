# 📊 Modul Visualisasi Dataset SmartCash

Modul ini menyediakan antarmuka visual untuk menganalisis dataset, menampilkan statistik, dan memvisualisasikan hasil augmentasi data.

## 🚀 Fitur Utama

- **Statistik Dataset**: Tampilkan ringkasan statistik dataset termasuk distribusi kelas, ukuran gambar, dan metrik lainnya
- **Visualisasi Augmentasi**: Lihat contoh hasil augmentasi data dengan berbagai transformasi
- **Antarmuka Interaktif**: UI berbasis Jupyter widgets yang mudah digunakan
- **Integrasi dengan Preprocessing**: Terintegrasi dengan modul preprocessing SmartCash

## 🛠️ Instalasi

Pastikan dependensi berikut terinstall:

```bash
pip install ipywidgets plotly matplotlib opencv-python
```

Aktifkan ekstensi Jupyter widgets:

```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## 🚀 Cara Menggunakan

### 1. Tampilkan Visualisasi

```python
from smartcash.ui.dataset.visualization import show_visualization

# Tampilkan antarmuka visualisasi
controller = show_visualization()
```

### 2. Gunakan Controller Langsung

```python
from smartcash.ui.dataset.visualization import VisualizationController

# Buat instance controller
controller = VisualizationController()

# Muat dataset
success = controller.load_dataset('nama_dataset_anda')

# Tampilkan UI
controller.display()
```

## 📊 Komponen Tersedia

### 1. DatasetStatsComponent

Komponen untuk menampilkan statistik dataset.

```python
from smartcash.ui.dataset.visualization.components import DatasetStatsComponent

# Buat instance
stats_component = DatasetStatsComponent()

# Perbarui data statistik
stats_component.update_stats(stats_data)

# Tampilkan komponen
stats_component.display()
```

### 2. AugmentationVisualizer

Komponen untuk memvisualisasikan hasil augmentasi data.

```python
from smartcash.ui.dataset.visualization.components import AugmentationVisualizer

# Buat instance
visualizer = AugmentationVisualizer(dataset_path='path/ke/dataset')

# Muat gambar acak
visualizer.load_random_image(split='train')

# Terapkan augmentasi
visualizer.apply_augmentations()

# Tampilkan hasil
visualizer.display()
```

## 🧪 Testing

Jalankan test dengan perintah:

```bash
pytest smartcash/ui/dataset/visualization/tests/test_visualization.py -v
```

## 📝 Catatan

- Modul ini dirancang untuk digunakan di lingkungan Jupyter Notebook/Lab
- Pastikan dataset sudah melalui proses preprocessing sebelum divisualisasikan
- Untuk dataset besar, beberapa visualisasi mungkin membutuhkan waktu untuk dirender

## 🤝 Berkontribusi

Kontribusi dipersilakan! Silakan buat pull request atau issue untuk melaporkan bug dan permintaan fitur.

## 📜 Lisensi

MIT
