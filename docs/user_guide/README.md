# 📱 Panduan Pengguna SmartCash

## 📋 Daftar Isi
- [Instalasi](#instalasi)
- [Penggunaan CLI](#cli)
- [Persiapan Dataset](#dataset)
- [Pelatihan Model](#training)
- [Evaluasi Model](#evaluasi)
- [Troubleshooting](#troubleshooting)

## 🔧 Instalasi <a id="instalasi"></a>

### Persyaratan Sistem
- Python 3.9+
- CUDA-capable GPU (opsional)
- 8GB RAM minimum
- 20GB ruang disk

### Setup Environment

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/smartcash.git
   cd smartcash
   ```

2. Buat environment:
   ```bash
   # Dengan conda
   conda create -n smartcash python=3.9
   conda activate smartcash

   # Atau dengan venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Setup environment variables:
   ```bash
   cp .env.example .env
   # Edit .env sesuai kebutuhan
   ```

## 🖥️ Penggunaan CLI <a id="cli"></a>

### Memulai Aplikasi
```bash
python run.py
```

### Menu Utama
```
=== SmartCash - Sistem Deteksi Uang Kertas ===

1. Pelatihan Model
2. Evaluasi Model
3. Keluar

Pilih menu (1-3):
```

### Navigasi Menu
- Gunakan angka untuk memilih menu
- Enter untuk konfirmasi
- Ctrl+C untuk keluar
- Backspace untuk kembali

## 📊 Persiapan Dataset <a id="dataset"></a>

### Struktur Data
```
data/
├── raw/                 # Data mentah
│   ├── images/         # Gambar uang
│   └── labels/         # Label YOLO
├── processed/          # Data terproses
│   ├── train/
│   ├── valid/
│   └── test/
└── augmented/         # Data augmentasi
```

### Menggunakan Dataset Lokal
1. Siapkan gambar di `data/raw/images/`
2. Buat label di `data/raw/labels/`
3. Jalankan preprocessing:
   ```python
   python -m smartcash.utils.preprocess
   ```

### Menggunakan Roboflow
1. Set API key di `.env`:
   ```
   ROBOFLOW_API_KEY=your_key_here
   ```
2. Update konfigurasi di `configs/base_config.yaml`
3. Download dataset:
   ```python
   python -m smartcash.utils.download_dataset
   ```

## 🚀 Pelatihan Model <a id="training"></a>

### Konfigurasi Training
1. Edit `configs/base_config.yaml`:
   ```yaml
   training:
     batch_size: 16
     epochs: 100
     learning_rate: 0.001
     optimizer: adamw
   ```

2. Pilih mode training:
   - Full training
   - Fine-tuning
   - Transfer learning

### Monitoring
- Progress bar realtime
- Metrik per epoch
- TensorBoard logging
- Model checkpoints

## 📈 Evaluasi Model <a id="evaluasi"></a>

### Mode Evaluasi
1. Regular Evaluation
   - Test dataset
   - Metrik standar

2. Research Mode
   - Custom dataset
   - Analisis mendalam
   - Visualisasi hasil

### Metrik
- mAP (mean Average Precision)
- Precision & Recall
- F1-Score
- Inference time

## ❗ Troubleshooting <a id="troubleshooting"></a>

### Masalah Umum

1. **Import Error**
   ```
   ModuleNotFoundError: No module named 'smartcash'
   ```
   ➡️ Pastikan working directory benar
   
2. **CUDA Error**
   ```
   RuntimeError: CUDA out of memory
   ```
   ➡️ Kurangi batch size

3. **Dataset Error**
   ```
   FileNotFoundError: data/raw not found
   ```
   ➡️ Periksa struktur folder

### Tips & Tricks

1. **Memory Usage**
   - Gunakan batch size yang sesuai
   - Clear cache saat perlu
   - Monitor GPU memory

2. **Performance**
   - Aktifkan CUDA jika ada
   - Optimalkan preprocessing
   - Cache hasil preprocessing

3. **Data Quality**
   - Validasi format gambar
   - Periksa label
   - Monitor class balance

## 🔄 Updates & Maintenance

### Version Control
- Gunakan Git untuk tracking
- Follow branching guidelines
- Regular commits

### Backup
- Model checkpoints
- Dataset versions
- Configuration files

### Logging
- Training logs
- Error logs
- Performance metrics
